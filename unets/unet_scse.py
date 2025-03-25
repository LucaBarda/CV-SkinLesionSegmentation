from scse import SCSEBlock

import tensorflow as tf
from tensorflow.keras import layers, models

class UNet4SCSE(tf.keras.Model):
    def __init__(self, input_size=(256, 256, 3), num_classes=1):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Contracting Path (Encoder)
        self.conv1 = self.conv_block(64)
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.scse1 = SCSEBlock(64)  # SCSE Block after conv1

        self.conv2 = self.conv_block(128)
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.scse2 = SCSEBlock(128)  # SCSE Block after conv2

        self.conv3 = self.conv_block(256)
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.scse3 = SCSEBlock(256)  # SCSE Block after conv3

        self.conv4 = self.conv_block(512)
        self.pool4 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        self.scse4 = SCSEBlock(512)  # SCSE Block after conv4
        
        # Bottleneck
        self.bottleneck = self.conv_block(1024)
        self.scse_bottleneck = SCSEBlock(1024)  # SCSE Block at the bottleneck
        
        # Expansive Path (Decoder)
        self.upsample4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')
        self.upconv4 = self.conv_block(512)
        self.scse_up4 = SCSEBlock(512)  # SCSE Block after upconv4

        self.upsample3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.upconv3 = self.conv_block(256)
        self.scse_up3 = SCSEBlock(256)  # SCSE Block after upconv3

        self.upsample2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.upconv2 = self.conv_block(128)
        self.scse_up2 = SCSEBlock(128)  # SCSE Block after upconv2

        self.upsample1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.upconv1 = self.conv_block(64)
        self.scse_up1 = SCSEBlock(64)  # SCSE Block after upconv1
        
        # Output layer with a 1x1 convolution (sigmoid for binary classification)
        self.output_layer = layers.Conv2D(self.num_classes, (1, 1), activation='sigmoid', padding='same')

    def conv_block(self, filters, kernel_size=(3, 3), padding='same', strides=(1, 1)):
        return models.Sequential([
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(filters, kernel_size, strides=strides, padding=padding),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
    
    def call(self, inputs):

        # Contracting path (encoder)
        enc1 = self.conv1(inputs)  # 256x256x3 -> 256x256x64
        enc1 = self.scse1(enc1)  # Apply SCSE Block after conv1
        pool1 = self.pool1(enc1)   # 256x256x64 -> 128x128x64

        enc2 = self.conv2(pool1)   # 128x128x64 -> 128x128x128
        enc2 = self.scse2(enc2)  # Apply SCSE Block after conv2
        pool2 = self.pool2(enc2)   # 128x128x128 -> 64x64x128

        enc3 = self.conv3(pool2)   # 64x64x128 -> 64x64x256
        enc3 = self.scse3(enc3)  # Apply SCSE Block after conv3
        pool3 = self.pool3(enc3)   # 64x64x256 -> 32x32x256

        enc4 = self.conv4(pool3)   # 32x32x256 -> 32x32x512
        enc4 = self.scse4(enc4)  # Apply SCSE Block after conv4
        pool4 = self.pool4(enc4)   # 32x32x512 -> 16x16x512
        
        # Bottleneck
        bottleneck = self.bottleneck(pool4)  # 16x16x512 -> 16x16x1024
        bottleneck = self.scse_bottleneck(bottleneck)  # Apply SCSE Block at bottleneck
        
        # Expansive path (decoder)
        up4 = self.upsample4(bottleneck)   # 16x16x1024 -> 32x32x512
        up4 = layers.Concatenate(axis=-1)([up4, enc4])  # 32x32x512 + 32x32x512 -> 32x32x1024
        up4 = self.upconv4(up4) # 32x32x1024 -> 32x32x512
        up4 = self.scse_up4(up4)  # Apply SCSE Block after upconv4

        up3 = self.upsample3(up4) # 32x32x512 -> 64x64x256
        up3 = layers.Concatenate(axis=-1)([up3, enc3]) # 64x64x256 + 64x64x256 -> 64x64x512
        up3 = self.upconv3(up3) # 64x64x512 -> 64x64x256
        up3 = self.scse_up3(up3)  # Apply SCSE Block after upconv3

        up2 = self.upsample2(up3) # 64x64x256 -> 128x128x128
        up2 = layers.Concatenate(axis=-1)([up2, enc2]) # 128x128x128 + 128x128x128 -> 128x128x256
        up2 = self.upconv2(up2) # 128x128x256 -> 128x128x128
        up2 = self.scse_up2(up2)  # Apply SCSE Block after upconv2

        up1 = self.upsample1(up2) # 128x128x128 -> 256x256x64
        up1 = layers.Concatenate(axis=-1)([up1, enc1]) # 256x256x64 + 256x256x64 -> 256x256x128
        up1 = self.upconv1(up1) # 256x256x128 -> 256x256x64
        up1 = self.scse_up1(up1)  # Apply SCSE Block after upconv1
        
        # Output layer
        output = self.output_layer(up1)  # 256x256x64 -> 256x256x1
        
        return output