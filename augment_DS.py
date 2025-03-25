import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Directories
train_images_dir = "split/train_images"
train_masks_dir = "split/train_masks"
output_dir = "split/dark_3/darkened_images"
mask_output_dir = "split/dark_3/darkened_images_masks"  # Folder for saving masks
os.makedirs(output_dir, exist_ok=True)
os.makedirs(mask_output_dir, exist_ok=True)

# Global HSL adjustments for outside (example values)
brightness_adjustment = -40       # Global Lightness adjustment (additive) for outside
temp_adjustment = 0                # Global Hue adjustment (additive) for outside
saturation_adjustment = 10       # Global Saturation adjustment (-100..+100, used multiplicatively) for outside

# HSL Adjustments (Lightroom-style) for specific hue ranges (applied only outside)
# Format: (Hue shift, Saturation shift, Luminance shift)
# Here, saturation shifts will be applied multiplicatively.
hsl_adjustments = {
    "red":      (10,  -50, 20),
    "orange":   (0,   -40, 20),
    "yellow":   (-30,  -60, 0),
    "green":    (-70,  -50, 20),
    "cyan":     (-130,  -65, 0),
    "blue":     (-160,  -65, 0),
    "purple":   (0,    -35,   20),
    "magenta":  (0,    -35,   20)
}

# Additional filter parameters for the inside region and final processing:
inside_brightness_adjustment = -40 #-120 #-40   # Additional brightness boost inside the lesion
inside_temp_adjustment = 3            # Raise temperature inside (additive to H channel)
inside_sat_factor = 0.4                # Lower saturation inside (multiplicative factor)
blur_kernel = (15, 15)                 # Gaussian blur kernel size
noise_sigma = 10                       # Gaussian noise standard deviation

# Tone-curve points for Red & Green channels
red_curve_points = [(0, 0), (25, 70), (255, 255)]
green_curve_points = [(0, 0), (80, 40), (255, 255)]

def apply_curves(channel, curve_points):
    """Applies a tone curve (lookup table) to a single channel."""
    curve = np.interp(np.arange(256),
                      [p[0] for p in curve_points],
                      [p[1] for p in curve_points]).astype(np.uint8)
    return cv2.LUT(channel, curve)

def saturation_factor(shift_value):
    """
    Converts a Lightroom-style shift in [-100..100] to a multiplicative factor in [0..2].
      -100 -> 0.0  (fully desaturated)
       0   -> 1.0  (no change)
      100  -> 2.0  (double saturation)
    """
    return 1.0 + (shift_value / 100.0)

# Process all images in the directory
image_files = sorted([f for f in os.listdir(train_images_dir) if f.startswith("ISIC_")])

for img_name in image_files:
    img_path = os.path.join(train_images_dir, img_name)
    mask_path = os.path.join(train_masks_dir, os.path.splitext(img_name)[0] + "_segmentation.png")
    
    # Load image and mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        print(f"Skipping {img_name} due to missing file.")
        continue
    
    # Normalize mask to [0,1]: 1 = inside lesion, 0 = outside
    mask = mask.astype(np.float32) / 255.0
    
    # ----- STEP 1: Apply Tone Curves (Red & Green) -----
    b, g, r = cv2.split(image)
    
    r_curve = apply_curves(r, red_curve_points).astype(np.float32)
    g_curve = apply_curves(g, green_curve_points).astype(np.float32)
    
    # For outside: use curved values; for inside: keep original
    r_new = r_curve * (1 - mask) + r.astype(np.float32) * mask
    g_new = g_curve * (1 - mask) + g.astype(np.float32) * mask
    b_new = b.astype(np.float32)  # Blue remains unchanged
    
    image_curved = cv2.merge([b_new.astype(np.uint8),
                              g_new.astype(np.uint8),
                              r_new.astype(np.uint8)])
    
    # ----- STEP 2: Process Outside Region via HSL Adjustments -----
    hls_image = cv2.cvtColor(image_curved, cv2.COLOR_BGR2HLS).astype(np.float32)
    processed_hls = hls_image.copy()
    
    processed_hls[..., 0] += temp_adjustment * (1 - mask)  # Hue adjustment
    processed_hls[..., 1] += brightness_adjustment * (1 - mask)  # Lightness adjustment
    processed_hls[..., 1] = np.clip(processed_hls[..., 1], 0, 255)
    
    global_sat_factor = saturation_factor(saturation_adjustment)
    outside_mask = (1 - mask)
    old_sat = processed_hls[..., 2]
    new_sat = old_sat * (1 - outside_mask) + (old_sat * global_sat_factor) * outside_mask
    processed_hls[..., 2] = np.clip(new_sat, 0, 255)
    
    for color, (hue_shift, sat_shift, lum_shift) in hsl_adjustments.items():
        if color == "red":
            color_range = (0, 15)
        elif color == "orange":
            color_range = (15, 30)
        elif color == "yellow":
            color_range = (30, 60)
        elif color == "green":
            color_range = (60, 110)
        elif color == "cyan":
            color_range = (110, 140)
        elif color == "blue":
            color_range = (140, 180)
        elif color == "purple":
            color_range = (180, 220)
        elif color == "magenta":
            color_range = (220, 255)
        
        hue_mask = (((processed_hls[..., 0] >= color_range[0]) & 
                     (processed_hls[..., 0] <= color_range[1])).astype(np.float32)
                    * (1 - mask))
        
        processed_hls[..., 0] += hue_shift * hue_mask
        processed_hls[..., 1] += lum_shift * hue_mask
        
        factor = saturation_factor(sat_shift)
        old_sat = processed_hls[..., 2]
        new_sat = old_sat * (1 - hue_mask) + (old_sat * factor) * hue_mask
        processed_hls[..., 2] = new_sat
    
    processed_hls[..., 0] = np.clip(processed_hls[..., 0], 0, 255)
    processed_hls[..., 1] = np.clip(processed_hls[..., 1], 0, 255)
    processed_hls[..., 2] = np.clip(processed_hls[..., 2], 0, 255)

    # Modify brightness after HSL adjustments only outside the lesion
    brightness_factor = 0.55  
    processed_hls[..., 1] = processed_hls[..., 1] *  mask + (processed_hls[..., 1] * brightness_factor) * (1 - mask)
    processed_hls[..., 1] = np.clip(processed_hls[..., 1], 0, 255)
    
    # ----- STEP 3: Process Inside Region Separately -----
    inside_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)
    inside_hls[..., 1] += inside_brightness_adjustment
    inside_hls[..., 1] = np.clip(inside_hls[..., 1], 0, 255)
    inside_hls[..., 2] *= inside_sat_factor
    inside_hls[..., 2] = np.clip(inside_hls[..., 2], 0, 255)
    inside_hls[..., 0] += inside_temp_adjustment
    inside_hls[..., 0] = np.clip(inside_hls[..., 0], 0, 255)
    
    # ----- STEP 4: Blend Outside and Inside Regions -----
    final_hls = processed_hls * (1 - mask[..., None]) + inside_hls * mask[..., None]
    adjusted_image = cv2.cvtColor(final_hls.astype(np.uint8), cv2.COLOR_HLS2BGR)
    
    # ----- STEP 5: Apply Gaussian Blur Over the Entire Image -----
    blurred_image = cv2.GaussianBlur(adjusted_image, blur_kernel, 0)
    
    # ----- STEP 6: Add Gaussian Noise -----
    noise = np.random.normal(0, noise_sigma, blurred_image.shape).astype(np.float32)
    noisy_image = blurred_image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    # Save the final output image and its mask with the new naming scheme
    # Naming: {index}_6.jpeg for image and {index}_6_mask.png for mask (index starting at 1)
    output_index = image_files.index(img_name) + 1
    output_filename = f"{output_index}_7.jpeg"
    output_mask_filename = f"{output_index}_7_mask.png"
    
    cv2.imwrite(os.path.join(output_dir, output_filename), noisy_image)
    cv2.imwrite(os.path.join(mask_output_dir, output_mask_filename), (mask * 255).astype(np.uint8))