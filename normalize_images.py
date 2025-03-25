import os
import cv2
import numpy as np
from glob import glob

# Define directories
input_img_dir = "ISIC-2017_Test_Data"  # Folder containing original images
input_mask_dir = "ISIC-2017_Test_GroundTruth" #  Folder containing original segmentation masks
output_hair_removed_dir = "hair_removed_test_images"  # Folder for hair-removed images (original resolution)
output_img_dir = "Normalized_ISIC-2017_Test_Data"  # Folder for resized 256x256 images (hair-free)
output_mask_dir = "Normalized_ISIC-2017_Test_GroundTruth"  # Folder for resized segmentation masks (256x256)

# Create output directories
os.makedirs(output_hair_removed_dir, exist_ok=True)
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

# Target image size
target_size = 256

def remove_hair(image):
    """
    Applies hair removal using Black Top-Hat filtering and inpainting.
    """
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define structuring element (kernel) for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))  # Adjust for hair thickness

    # Apply morphological closing (X · Y)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Compute Black Top-Hat transformation: B_TH = (X · Y) - X
    black_tophat = closed - gray

    # Threshold to create a binary mask of hair pixels
    _, mask = cv2.threshold(black_tophat, 10, 255, cv2.THRESH_BINARY)

    # Inpaint the detected hair pixels using neighboring pixels
    inpainted = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

    return inpainted

def resize_image(image, target_size=256, interpolation=cv2.INTER_LINEAR):
    """
    Resizes the image directly to target_size x target_size without keeping aspect ratio.
    No padding or cropping is applied.
    """
    return cv2.resize(image, (target_size, target_size), interpolation=interpolation)

# Get all image files (excluding segmentation masks)
image_files = sorted(glob(os.path.join(input_img_dir, "ISIC_*[0-9].jpg")))  # Ensures order

for img_path in image_files:
    base_name = os.path.basename(img_path).split('.')[0]  # Extract file name without extension
    mask_path = os.path.join(input_mask_dir, f"{base_name}_segmentation.png")

    # Read image and mask
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None

    if image is None or mask is None:
        print(f"Skipping {img_path}, missing image or mask.")
        continue

    # Step 1: Apply hair removal on the original image
    hair_removed_image = remove_hair(image)

    # Save the hair-removed image in original resolution
    cv2.imwrite(os.path.join(output_hair_removed_dir, os.path.basename(img_path)), hair_removed_image)

    # Step 2: Resize the hair-free image and mask to 256×256
    resized_hair_removed = resize_image(hair_removed_image, target_size, interpolation=cv2.INTER_LINEAR)
    resized_mask = resize_image(mask, target_size, interpolation=cv2.INTER_NEAREST)

    # Save the resized images and masks
    cv2.imwrite(os.path.join(output_img_dir, os.path.basename(img_path)), resized_hair_removed)  # 256x256 resized
    cv2.imwrite(os.path.join(output_mask_dir, os.path.basename(mask_path)), resized_mask)  # 256x256 resized mask

    print(f"Processed {img_path} → Hair Removed & Normalized (256×256).")

print("✅ All images processed: Hair removed first, then resized to 256×256.")
