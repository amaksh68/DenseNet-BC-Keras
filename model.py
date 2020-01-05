import keras
import keras.backend as K

from keras.models import Model
from keras.layers import Input, concatenate, Activation, Dropout, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import os
import yaml
import numpy as np

def add_layer(x, nb_channels, kernel_size=3, bottleneck=True, dropout=0., l2_reg=1e-4):
    out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    out = Activation('relu')(out)
    
    if bottleneck:
        inter_channel = nb_channels * 4  

        out = Convolution2D(inter_channel, (1, 1), init='he_normal', border_mode='same', bias=False,
                   W_regularizer=l2(l2_reg))(out)
        out = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(out)
        out = Activation('relu')(out)    

    out = Convolution2D(nb_channels, kernel_size, kernel_size,
                        border_mode='same', init='he_normal',
                        W_regularizer=l2(l2_reg), bias=False)(out)
    if dropout > 0:
        out = Dropout(dropout)(out)
    return out

def dense_block(x, nb_layers, growth_rate, dropout=0., l2_reg=1e-4):
    for i in range(nb_layers):
        # Get layer output
        out = add_layer(x, growth_rate, bottleneck=True, dropout=dropout, l2_reg=l2_reg)
        if K.common.image_dim_ordering() == 'tf':
            merge_axis = -1
        elif K.common.image_dim_ordering() == 'th':
            merge_axis = 1
        else:
            raise Exception('Invalid dim_ordering: ' + K.common.image_dim_ordering())
        # Concatenate input with layer ouput
        d = int(i*2/nb_layers)*growth_rate
        x = Lambda(lambda x: x[:,:,:,d:])(x)
        x = concatenate([x, out], -1)
    return x

def transition_block(x, nb_channels, dropout=0., l2_reg=1e-4):
    x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    
    x = Convolution2D(int(nb_channels*0.5), 1, 1, border_mode='same',
                       init='he_normal', W_regularizer=l2(l2_reg))(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x

def densenet_model(nb_blocks, nb_layers, growth_rate, dropout=0., l2_reg=1e-4):
    n_channels = 2*growth_rate
    inputs = Input(shape=(32, 32, 3))
    x = Convolution2D(n_channels, 3, 3, border_mode='same',
                      init='he_normal', W_regularizer=l2(l2_reg),
                      bias=False)(inputs)
    for i in range(nb_blocks - 1):
        # Create a dense block
        x = dense_block(x, nb_layers, growth_rate,
                        dropout=dropout, l2_reg=l2_reg)
        # Update the number of channels
        n_channels += nb_layers*growth_rate
        # Transition layer
        x = transition_block(x, n_channels, dropout=dropout, l2_reg=l2_reg)
        
        n_channels = int(n_channels * 0.5)


    # Add last dense_block
    x = dense_block(x, nb_layers, growth_rate, dropout=dropout, l2_reg=l2_reg)
    # Add final BN-Relu
    x = BatchNormalization(gamma_regularizer=l2(l2_reg),
                             beta_regularizer=l2(l2_reg))(x)
    x = Activation('relu')(x)
    # Global average pooling
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, W_regularizer=l2(l2_reg))(x)
    x = Activation('softmax')(x)

    model = Model(input=inputs, output=x)
    return model
