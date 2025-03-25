import tensorflow as tf
from tensorflow.keras import layers, models

# Define the SCSEBlock as a custom layer
class SCSEBlock(layers.Layer):
    def __init__(self, filters):
        super(SCSEBlock, self).__init__()
        self.filters = filters
        
        # Channel Attention
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // 2, activation="relu")
        self.dense2 = layers.Dense(filters, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, filters))
        
        # Spatial Attention
        self.spatial_conv = layers.Conv2D(1, kernel_size=1, activation="sigmoid")
    
    def call(self, inputs):
        # Channel Attention
        ch_attn = self.global_avg_pool(inputs)
        ch_attn = self.dense1(ch_attn)
        ch_attn = self.dense2(ch_attn)
        ch_attn = self.reshape(ch_attn)

        # Spatial Attention
        sp_attn = self.spatial_conv(inputs)

        # Apply Channel and Spatial Attention
        x = layers.Multiply()([inputs, ch_attn])
        x = layers.Multiply()([x, sp_attn])

        # Residual Connection
        # return layers.Add()([x, inputs])
        return x