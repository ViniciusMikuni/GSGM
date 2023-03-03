from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers




def DeepSetsAtt(
        #num_part,
        num_feat,
        time_embedding,
        num_heads=4,
        num_transformer = 4,
        projection_dim=32,    
):
    
    inputs = Input((None,num_feat))
    #Include the time information as an additional feature fixed for all particles
    time = tf.reshape(time_embedding,(-1,1,tf.shape(time_embedding)[-1]))
    time = tf.tile(time,(1,tf.shape(inputs)[1],1))
    inputs_time = tf.concat([inputs,time],-1)

    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    tdd = TimeDistributed(Dense(projection_dim,activation='swish',use_bias=False))(inputs_time)
    encoded_patches = layers.LayerNormalization(epsilon=1e-6)(tdd)
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        #x1 =encoded_patches

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim,
            dropout=0.0,use_bias=False,
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        #x3=tf.concat([x2,reshaped_time],-1)
        #x3=x2
        # MLP.
        #tf.nn.gelu
        x3 = layers.Dense(2*projection_dim,activation="gelu",use_bias=False)(x3)
        x3 = layers.Dense(projection_dim,activation="gelu",use_bias=False)(x3)
        #time_vit = layers.Dense(projection_dim,activation=self.activation)(reshaped_time)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    outputs = Dense(num_feat,activation=None)(representation)    
    return  inputs, outputs
