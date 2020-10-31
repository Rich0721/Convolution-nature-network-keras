'''
Author: Rich, wu
'''

import os

import tensorflow as tf
from keras.models import Model
from keras import layers
from keras.layers import Input, BatchNormalization, Reshape
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Activation
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, LeakyReLU
from keras.engine import get_source_inputs
from keras.utils import get_file, layer_utils
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings

def resnet_identity_block(input_tensor, kernel_size, filters, stage, block, strides=(1, 1)):

    filter1, filter2, filter3 = filters

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    conv_block = "conv" + str(stage) + "_" + block

    
    x = Conv2D(filter1, (1, 1), strides=(1, 1), padding='same', use_bias=False,name=conv_block + "/conv1")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv_block + "/bn1")(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filter2, kernel_size, strides=(1, 1), padding='same', use_bias=False, name=conv_block + "/conv2")(x)
    x = BatchNormalization(axis=bn_axis, name=conv_block + "/bn2")(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filter3, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation='linear', name=conv_block + "/conv3")(x)
    x = BatchNormalization(axis=bn_axis, name=conv_block + "/bn3")(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, use_bias=False, name="shortcut" + str(stage) + "_" + block)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name="shortcut" + str(stage) + "_" + block+ "/bn")(shortcut)

    x = layers.add([x, shortcut])
    x = LeakyReLU()(x)

    return x


def cross_stage(input_tensor, temp_tensor, filters, stage):

    filter1, filter2, filter3, filter4, filter5 = filters

    cross = "cross_stage_conv_" + str(stage) + "_"
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    
    x = Conv2D(filter1, (1, 1), strides=(1, 1), padding='same',use_bias=False, name=cross + "a")(input_tensor)
    x = BatchNormalization(axis=bn_axis, name= cross + 'a/bn')(x)
    x = LeakyReLU()(x)

    x = layers.add([x, temp_tensor])
    x = Conv2D(filter2, (1, 1), strides=(1, 1), padding='same', use_bias=False,name=cross + "b")(x)
    x = BatchNormalization(axis=bn_axis, name=cross + 'b/bn')(x)
    x = LeakyReLU()(x)
    
    x = Conv2D(filter3, (3, 3), strides=(2, 2), padding='same', use_bias=False,name=cross + "c")(x)
    x = BatchNormalization(axis=bn_axis, name=cross + 'c/bn')(x)
    x = LeakyReLU()(x)

    
    temp = Conv2D(filter4, (1, 1), strides=(1, 1), padding='same', use_bias=False,activation='linear', name=cross + "temp")(x)
    temp = BatchNormalization(axis=bn_axis, name=cross + 'temp/bn')(temp)

    
    x = Conv2D(filter5, (1, 1), strides=(1, 1), padding='same', use_bias=False, activation='linear', name=cross + "d")(x)
    x = BatchNormalization(axis=bn_axis, name=cross + 'd/bn')(x)

    return x, temp


def cspresnet50(include_top=False, input_shape=None, input_tensor=None, weights_file=None, classes=1000):

    input_shape = _obtain_input_shape(input_shape,
                                    default_size=224,
                                    min_size=32,
                                    data_format=K.image_data_format(),
                                    weights=None,
                                    require_flatten=True)
    

    if input_tensor is None:
        img_input = Input(input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    
    
    x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=False, padding='same', name="conv1")(img_input)
    x = BatchNormalization(axis=bn_axis, name="conv1/bn")(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)
    
    
    temp = Conv2D(128, (1, 1), strides=(1, 1), padding='same')(x)
    temp = BatchNormalization(axis=bn_axis, name='conv2/bn')(temp)
    temp = LeakyReLU()(temp)

    
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(name='conv3/bn')(x)
    x = LeakyReLU()(x)

    # Resnet stage 1
    x = resnet_identity_block(x, (3, 3), [64, 64, 128], stage=1, block="a")
    x = resnet_identity_block(x, (3, 3), [64, 64, 128], stage=1, block="b")
    x = resnet_identity_block(x, (3, 3), [64, 64, 128], stage=1, block="c")

    # Cross Stage 1 
    x, temp = cross_stage(x, temp, [128, 128, 128, 256, 256], stage=1)

    # Resnet stage 2
    x = resnet_identity_block(x, (3, 3), [128, 128, 256], stage=2, block='a')
    x = resnet_identity_block(x, (3, 3), [128, 128, 256], stage=2, block='b')
    x = resnet_identity_block(x, (3, 3), [128, 128, 256], stage=2, block='c')

    # Cross Stage 2
    x, temp = cross_stage(x, temp, [256, 256, 256, 512, 512], stage=2)
  

    # Resnet stage 3
    x = resnet_identity_block(x, (3, 3), [256, 256, 512], stage=3, block='a')
    x = resnet_identity_block(x, (3, 3), [256, 256, 512], stage=3, block='b')
    x = resnet_identity_block(x, (3, 3), [256, 256, 512], stage=3, block='c')
    x = resnet_identity_block(x, (3, 3), [256, 256, 512], stage=3, block='d')
    x = resnet_identity_block(x, (3, 3), [256, 256, 512], stage=3, block='e')


    # Cross Stage 3
    x, temp = cross_stage(x, temp, [512, 512, 512, 1024, 1024], stage=3)

    # Resnet stage 4
    x = resnet_identity_block(x, (3, 3), [512, 512, 1024], stage=4, block='a')
    x = resnet_identity_block(x, (3, 3), [512, 512, 1024], stage=4, block='b')
    
    
    x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='conv4')(x)
    x = BatchNormalization(axis=bn_axis, name='conv4/bn')(x)
    x = LeakyReLU()(x)

    x = layers.add([x, temp])
    x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='conv5')(x)
    x = LeakyReLU()(x)

    x = AveragePooling2D((7, 7), name='pool5')(x)
    
    if include_top:
        x = Flatten()(x)
        x = Dense(classes, name='fc')
        x = Activation('softmax')(x)
    else:
        x = GlobalMaxPooling2D()(x)
    
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    model = Model(inputs, x, name="CSPResnet")
    model.summary()

    if weights_file is not None:
        model.load_weights(weights_file)
    
    return model