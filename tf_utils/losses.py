import tensorflow as tf
from tensorflow.keras import backend as K

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)  # Flatten tensors
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)  # Compute intersection
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)  # Dice loss

def bce_dice_loss(y_true, y_pred, bce_weight=0.1):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce_weight * bce + (1 - bce_weight) * dice  # Hybrid loss (adjust weighting if needed)