import tensorflow as tf
import os

def load_image(image_path):
    """Load an image, decode it, and resize to 256x256"""
    img = tf.io.read_file(image_path)               # Read image
    img = tf.image.decode_jpeg(img, channels=3)       # Decode as JPG (assuming JPG images)
    img = tf.image.resize(img, (256, 256))           # Resize to target size
    img = img / 255.0                               # Normalize to range [0, 1]
    return img

def load_mask(mask_path):
    """Load a mask, decode it, and resize to 256x256"""
    mask = tf.io.read_file(mask_path)               # Read mask image
    mask = tf.image.decode_png(mask, channels=1)     # Decode as PNG (assuming masks are single-channel)
    mask = tf.image.resize(mask, (256, 256))         # Resize to target size
    mask = mask / 255.0                             # Normalize to range [0, 1]
    return mask

def load_data(image_folder, mask_folder):
    # Get image and mask file paths
    image_paths = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder)])
    mask_paths = sorted([os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder)])
    
    # Shuffle the paths (optional)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

    # Load images and masks
    dataset = dataset.map(lambda x, y: (load_image(x), load_mask(y)))
    
    # Shuffle and batch the dataset
    dataset = dataset.shuffle(buffer_size=1000)  # Shuffle with buffer_size (adjust as needed)
    dataset = dataset.batch(batch_size=16)  # Batch size for training (adjust as needed)
    
    return dataset
