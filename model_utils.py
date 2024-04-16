import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import KFold
from keras.optimizers.legacy import Adam
import tensorflow as tf 
from keras.models import Model
from keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class CrossModalAttentionLayer(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multi_head_attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
    
    def call(self, query, key_value):
        attn_output = self.multi_head_attn(query, key_value, key_value)
        return attn_output
