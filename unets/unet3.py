import tensorflow as tf
from tensorflow.keras import layers, models

class UNet3(tf.keras.Model):
    def __init__(self, input_size=(256, 256, 3), num_classes=1):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Contracting Path (Encoder)
        self.conv1 = self.conv_block(64)
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv2 = self.conv_block(128)
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2))

        self.conv3 = self.conv_block(256)
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        
        # Bottleneck
        self.bottleneck = self.conv_block(512)
        
        # Expansive Path (Decoder)
        self.upsample3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')
        self.upconv3 = self.conv_block(256)

        self.upsample2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')
        self.upconv2 = self.conv_block(128)

        self.upsample1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')
        self.upconv1 = self.conv_block(64)
        
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
        pool1 = self.pool1(enc1)   # 256x256x64 -> 128x128x64

        enc2 = self.conv2(pool1)   # 128x128x64 -> 128x128x128
        pool2 = self.pool2(enc2)   # 128x128x128 -> 64x64x128

        enc3 = self.conv3(pool2)   # 64x64x128 -> 64x64x256
        pool3 = self.pool3(enc3)   # 64x64x256 -> 32x32x256
        
        # Bottleneck
        bottleneck = self.bottleneck(pool3)  # 32x32x256 -> 32x32x512
        
        # Expansive path (decoder)
        up3 = self.upsample3(bottleneck) # 32x32x512 -> 64x64x256
        up3 = layers.Concatenate(axis=-1)([up3, enc3]) # 64x64x256 + 64x64x256 -> 64x64x512
        up3 = self.upconv3(up3) # 64x64x512 -> 64x64x256

        up2 = self.upsample2(up3) # 64x64x256 -> 128x128x128
        up2 = layers.Concatenate(axis=-1)([up2, enc2]) # 128x128x128 + 128x128x128 -> 128x128x256
        up2 = self.upconv2(up2) # 128x128x256 -> 128x128x128

        up1 = self.upsample1(up2) # 128x128x128 -> 256x256x64
        up1 = layers.Concatenate(axis=-1)([up1, enc1]) # 256x256x64 + 256x256x64 -> 256x256x128
        up1 = self.upconv1(up1) # 256x256x128 -> 256x256x64
        
        # Output layer
        output = self.output_layer(up1)  # 256x256x64 -> 256x256x1
        
        return output
