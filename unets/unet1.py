import tensorflow as tf
from tensorflow.keras import layers, models

class UNet1(tf.keras.Model):
    def __init__(self, input_size=(256, 256, 3), num_classes=1):
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Contracting Path (Encoder)
        self.conv1 = self.conv_block(64)
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2))
        
        # Bottleneck
        self.bottleneck = self.conv_block(128)
        
        # Expansive Path (Decoder)
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
        
        # Bottleneck
        bottleneck = self.bottleneck(pool1)  # 128x128x64 -> 128x128x128
        
        # Expansive path (decoder)
        up1 = self.upsample1(bottleneck) # 128x128x128 -> 256x256x64
        up1 = layers.Concatenate(axis=-1)([up1, enc1]) # 256x256x64 + 256x256x64 -> 256x256x128
        up1 = self.upconv1(up1) # 256x256x128 -> 256x256x64
        
        # Output layer
        output = self.output_layer(up1)  # 256x256x64 -> 256x256x1
        
        return output
