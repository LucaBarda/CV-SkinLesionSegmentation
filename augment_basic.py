import os
import numpy as np
import tensorflow as tf
from PIL import Image  # To save images & masks

# Define dataset directories
train_images_dir = "train_images"
train_masks_dir = "train_masks"

# Define output directories
sampled_images_dir = "data_augmented_images_4"
sampled_masks_dir = "data_augmented_masks_4"  # Existing folder where masks should be saved

# Ensure output directories exist
os.makedirs(sampled_images_dir, exist_ok=True)
os.makedirs(sampled_masks_dir, exist_ok=True)

def get_filenames(image_folder, mask_folder):
    """Retrieve filenames from both image and mask folders ensuring alignment"""
    image_filenames = sorted([f for f in os.listdir(image_folder) if f.endswith(".png") or f.endswith(".jpg")])
    mask_filenames = [f.replace(".jpg", ".png").replace(".png", "_segmentation.png") for f in image_filenames]
    return image_filenames, mask_filenames

def augment_image(image, mask):
    """Apply multiple augmentations to an image and return a list of augmented images."""
    augmented_images = []
    augmented_masks = []

    def clip(img):
        return tf.clip_by_value(img, 0.0, 1.0)  # Ensure values stay in [0,1]

    # Original
    augmented_images.append(clip(image))
    augmented_masks.append(clip(mask))

    # Flip Horizontally
    img_flip_h = tf.image.flip_left_right(image)
    augmented_images.append(clip(img_flip_h))
    mask_flip_h = tf.image.flip_left_right(mask)
    augmented_masks.append(clip(mask_flip_h))

    # Rotate 90 degrees
    img_rot90 = tf.image.rot90(image)
    augmented_images.append(clip(img_rot90))
    mask_rot90 = tf.image.rot90(mask)
    augmented_masks.append(clip(mask_rot90))

    # Hue Shift (toward brown)
    img_hue = tf.image.adjust_hue(image, delta=-0.2)
    augmented_images.append(clip(img_hue))
    augmented_masks.append(clip(mask))  # Mask remains unchanged

    # Gaussian Noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
    img_noise = image + noise
    augmented_images.append(clip(img_noise))
    augmented_masks.append(clip(mask))  # Mask remains unchanged

    # **Stretching Augmentation (Random Scaling)**
    def stretch(image, mask):
        """Stretch image, crop center, and resize to 256x256 to remove black bands."""
        scale_x = np.random.uniform(0.7, 1.5)  # Random horizontal stretch factor
        scale_y = np.random.uniform(0.7, 1.5)  # Random vertical stretch factor

        original_height = tf.shape(image)[0]
        original_width = tf.shape(image)[1]

        new_width = tf.maximum(tf.cast(tf.round(tf.cast(original_width, tf.float32) * scale_x), tf.int32), 256)
        new_height = tf.maximum(tf.cast(tf.round(tf.cast(original_height, tf.float32) * scale_y), tf.int32), 256)

        # Stretch image & mask
        stretched_img = tf.image.resize(image, (new_height, new_width), method='bilinear')
        stretched_mask = tf.image.resize(mask, (new_height, new_width), method='nearest')

        # Compute valid crop coordinates
        crop_x = tf.maximum((new_width - 256) // 2, 0)
        crop_y = tf.maximum((new_height - 256) // 2, 0)

        # Ensure crop remains within image boundaries
        crop_x = tf.minimum(crop_x, new_width - 256)
        crop_y = tf.minimum(crop_y, new_height - 256)

        # Crop to 256x256
        cropped_img = tf.image.crop_to_bounding_box(stretched_img, crop_y, crop_x, 256, 256)
        cropped_mask = tf.image.crop_to_bounding_box(stretched_mask, crop_y, crop_x, 256, 256)

        return cropped_img, cropped_mask


    img_stretched, mask_stretched = stretch(image, mask)
    augmented_images.append(clip(img_stretched))
    augmented_masks.append(clip(mask_stretched))

    return augmented_images, augmented_masks

def process_and_save_data(image_folder, mask_folder):
    """Select images, apply augmentation, and save correctly paired results."""
    
    # Get sorted filenames
    image_filenames, mask_filenames = get_filenames(image_folder, mask_folder)

    # Use all images
    selected_pairs = list(zip(image_filenames, mask_filenames))

    for idx, (img_filename, mask_filename) in enumerate(selected_pairs, start=1):
        # Load original image & mask
        img_path = os.path.join(image_folder, img_filename)
        mask_path = os.path.join(mask_folder, mask_filename)

        image = np.array(Image.open(img_path).convert("RGB")) / 255.0  # Normalize to [0,1]
        mask = np.array(Image.open(mask_path).convert("L")) / 255.0  # Normalize to [0,1]

        # Convert to TensorFlow tensors
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

        # Ensure the mask has shape (H, W, 1)
        mask_tensor = tf.expand_dims(mask_tensor, axis=-1)

        # Apply augmentations
        augmented_images, augmented_masks = augment_image(image_tensor, mask_tensor)

        for aug_idx, aug_img in enumerate(augmented_images, start=1):
            # Convert back to NumPy and rescale to [0,255]
            aug_img_np = (aug_img.numpy() * 255).astype(np.uint8)
            mask_np = (augmented_masks[aug_idx - 1].numpy().squeeze() * 255).astype(np.uint8)  # Remove extra dim

            # Define new filenames
            img_filename_new = f"{idx}_{aug_idx}.png"
            mask_filename_new = f"{idx}_{aug_idx}_mask.png"

            # Save augmented images & masks in the correct directory
            Image.fromarray(aug_img_np).save(os.path.join(sampled_images_dir, img_filename_new))
            Image.fromarray(mask_np).save(os.path.join(sampled_masks_dir, mask_filename_new))

    print(f"✅ Images and masks augmented successfully. Saved in '{sampled_images_dir}' & '{sampled_masks_dir}'.")

    
# Run the function
process_and_save_data(train_images_dir, train_masks_dir)
