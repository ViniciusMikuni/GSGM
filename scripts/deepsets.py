from tensorflow.keras.layers import BatchNormalization, Layer, TimeDistributed
from tensorflow.keras.layers import Dense, Input, ReLU, Masking,Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def DeepSetsAtt(
        #num_part,
        num_feat,
        time_embedding,
        num_heads=4,
        num_transformer = 4,
        projection_dim=32,
        mask = None,
):


    inputs = Input((None,num_feat))
    masked_inputs = layers.Masking(mask_value=0.0,name='Mask')(inputs)
    masked_features = Dense(projection_dim,activation=None)(masked_inputs)
    masked_features = layers.LeakyReLU(alpha=0.01)(masked_features)
    
    #Include the time information as an additional feature fixed for all particles
    time = layers.Dense(2*projection_dim,activation=None)(time_embedding)
    time = layers.LeakyReLU(alpha=0.01)(time)
    time = layers.Dense(projection_dim)(time)

    time = layers.Reshape((1,-1))(time)
    time = tf.tile(time,(1,tf.shape(inputs)[1],1))

    
    #Use the deepsets implementation with attention, so the model learns the relationship between particles in the event
    concat = layers.Concatenate(-1)([masked_features,time])
    tdd = Dense(projection_dim,activation=None)(concat)
    tdd = layers.LeakyReLU(alpha=0.01)(tdd)
    encoded_patches = Dense(projection_dim)(tdd)

    mask_matrix = tf.matmul(mask,tf.transpose(mask,perm=[0,2,1]))
    
    for _ in range(num_transformer):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim//num_heads,
            dropout=0.1)(x1, x1, attention_mask=tf.cast(mask_matrix,tf.bool))
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
            
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(2*projection_dim,activation="gelu")(x3)
        x3 = layers.Dense(projection_dim,activation="gelu")(x3)

        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])
        

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    representation_mean = layers.GlobalAvgPool1D()(representation)
    representation_mean = layers.Concatenate(-1)([representation_mean,time_embedding])
    representation_mean = layers.Reshape((1,-1))(representation_mean)
    representation_mean = tf.tile(representation_mean,(1,tf.shape(inputs)[1],1))

    
    add = layers.Concatenate(-1)([tdd,representation,representation_mean])
    representation =  Dense(2*projection_dim,activation=None)(add)    
    representation =  layers.LeakyReLU(alpha=0.01)(representation)
    outputs = Dense(num_feat,activation=None,kernel_initializer="zeros")(representation)
    
    return  inputs, outputs



def Resnet(
        inputs,
        end_dim,
        time_embedding,
        num_embed,
        num_layer = 3,
        mlp_dim=128,
):
    
    act = layers.LeakyReLU(alpha=0.01)
    #act = swish

    def resnet_dense(input_layer,hidden_size,nlayers=2):
        layer = input_layer
        residual = layers.Dense(hidden_size)(layer)
        for _ in range(nlayers):
            layer=act(layers.Dense(hidden_size,activation=None)(layer))
            layer = layers.Dropout(0.1)(layer)
        return residual + layer
    
    embed = act(layers.Dense(mlp_dim)(time_embedding))
    inputs_dense = act(layers.Dense(mlp_dim)(inputs))
    residual = act(layers.Dense(2*mlp_dim)(tf.concat([inputs_dense,embed],-1)))    
    residual = layers.Dense(mlp_dim)(residual)
    layer = residual
    for _ in range(num_layer-1):
        # time = act(layers.Dense(mlp_dim)(embed))
        # layer =  resnet_dense(tf.concat([layer,time],-1),mlp_dim)
        layer =  resnet_dense(layer,mlp_dim)

    layer = act(layers.Dense(mlp_dim)(residual+layer))
    outputs = layers.Dense(end_dim,kernel_initializer="zeros")(layer)
    
    return outputs
