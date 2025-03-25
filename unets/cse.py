import tensorflow as tf
from tensorflow.keras import layers, models

class cSEBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // 2, activation="relu")
        self.dense2 = layers.Dense(filters, activation="sigmoid")
        self.reshape = layers.Reshape((1, 1, filters))

    def call(self, inputs):
        ch_attn = self.global_avg_pool(inputs)
        ch_attn = self.dense1(ch_attn)
        ch_attn = self.dense2(ch_attn)
        ch_attn = self.reshape(ch_attn)
        return layers.Multiply()([inputs, ch_attn])