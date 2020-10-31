'''
Author: Rich, wu
'''

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, SeparableConv2D, Activation
from tensorflow.python.keras.layers import AveragePooling2D, Flatten, Dense, add
from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPooling2D
from tensorflow.python.keras.engine import get_source_inputs
from tensorflow.python.keras.utils import layer_utils, get_file
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings

def residual_separable(inputs, filters, block, use_bias=False, bn_axis=-1):

    residual_name = "conv" + str(block) +  "_residual"
    sparable_name = "conv" + str(block) + "_separable"

    residual = Conv2D(filters[0], (1, 1), strides=(2, 2), padding='same', use_bias=use_bias, name=residual_name)(inputs)
    residual = BatchNormalization(axis=bn_axis, name=residual_name + "_bn")(residual)
    
    if block != 2:
        inputs = Activation('relu')(inputs)
    x = SeparableConv2D(filters[1], (3, 3), padding='same', use_bias=use_bias, name=sparable_name + "1")(inputs)
    x = BatchNormalization(axis=bn_axis, name=sparable_name + "1_bn")(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters[2], (3, 3), padding='same', use_bias=use_bias, name=sparable_name + "2")(x)
    x = BatchNormalization(axis=bn_axis, name=sparable_name + "2_bn")(x)

    # Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = add([x, residual])

    return x

def middle_flow(inputs, filters, block, use_bias=False, bn_axis=-1):

    sparable_name = "conv" + str(block) + "_separable"
    residual = inputs

    x = Activation('relu')(inputs)
    x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=use_bias, name=sparable_name + "1")(x)
    x = BatchNormalization(axis=bn_axis, name=sparable_name + "1_bn")(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=use_bias, name=sparable_name + "2")(x)
    x = BatchNormalization(axis=bn_axis, name=sparable_name + "2_bn")(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(filters, (3, 3), padding='same', use_bias=use_bias, name=sparable_name + "3")(x)
    x = BatchNormalization(axis=bn_axis, name=sparable_name + "2_bn")(x)

    x = add([x, residual])

    return x

def xception(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000):
    
    input_shape = _obtain_input_shape(input_shape=input_shape,
                                    default_size=299,
                                    min_size=71,
                                    data_format=K.image_data_format(),
                                    require_flatten=include_top,
                                    weights=weights)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor():
            img_input = Input(shape=input_shape, tensor=input_tensor)
        else:
            img_input = input_tensor

    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1
    

    # Block 1
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name="conv1_1")(img_input)
    x = BatchNormalization(axis=bn_axis, name="conv1_1_bn")(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name="conv1_2")(x)
    x = BatchNormalization(axis=bn_axis, name="conv1_2_bn")(x)
    x = Activation('relu')(x)

    # Block 2
    x = residual_separable(x, [128, 128, 128], block=2, bn_axis=bn_axis)
    
    # Block 3
    x = residual_separable(x, [256, 256, 256], block=3, bn_axis=bn_axis)

    # Block 4
    x = residual_separable(x, [728, 728, 728], block=4, bn_axis=bn_axis)

    # Block 5->12
    for i in range(5, 13, 1):
        x = middle_flow(x, 728, block=i, bn_axis=bn_axis)
    
    # Block 13
    x = residual_separable(x, [1024, 728, 1024], block=13, bn_axis=bn_axis)

    # Block14
    
    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name= "conv14_1")(x)
    x = BatchNormalization(axis=bn_axis, name="conv14_1_bn")(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name="conv14_2")(x)
    x = BatchNormalization(axis=bn_axis, name="conv14_2_bn")(x)
    x = Activation('relu', name="pool5")(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    if input_tensor is None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    model = Model(inputs, x, name='xception')
    model.summary()

    return model