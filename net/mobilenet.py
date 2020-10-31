'''
Author: Rich, wu
'''

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, BatchNormalization
from keras.layers import Conv2D, DepthwiseConv2D, ReLU
from keras.layers import AveragePooling2D, Flatten, Dense, add
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.engine import get_source_inputs
from keras.utils import layer_utils, get_file
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings

# mobilenet version 1
def _depthwise_separable_Conv(inputs, filters, block, bn_axis, alpha=1.0, use_bias=False, strides=(1, 1)):

    depth = "conv" + str(block) + "_" + "depth"
    width = "conv" + str(block) + "_" + "width"

    num_filters = round(filters * alpha)

    # Depth
    x = DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=use_bias, name=depth)(inputs)
    x = BatchNormalization(axis=bn_axis, name=depth + "_bn")(x)
    x = ReLU(6, name=depth + "_relu")(x)

    # width
    x = Conv2D(num_filters, (1, 1), strides=(1, 1), name=width)(x)
    x = BatchNormalization(axis=bn_axis, name=width + '_bn')(x)
    x = ReLU(6, name=width + "_relu")(x)
    return x

def mobilenet(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000):

    input_shape = _obtain_input_shape(input_shape=input_shape,
                                    default_size=224,
                                    min_size=32,
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

    # Input(224, 224, 3)
    # Output(112, 112, 64)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name="conv1_bn")(x)
    x = ReLU(6, name="conv1_relu")(x)
    x = _depthwise_separable_Conv(x, 64, block=2, bn_axis=bn_axis, strides=(1, 1))

    # Input(112, 112, 64)
    # Output(56, 56, 128)
    x = _depthwise_separable_Conv(x, 128, block=3, bn_axis=bn_axis, strides=(2, 2))
    x = _depthwise_separable_Conv(x, 128, block=4, bn_axis=bn_axis, strides=(1, 1))


    # Input(56, 56, 128)
    # Output(28, 28, 256)
    x = _depthwise_separable_Conv(x, 256, block=5, bn_axis=bn_axis, strides=(2, 2))
    x = _depthwise_separable_Conv(x, 256, block=6, bn_axis=bn_axis, strides=(1, 1))
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='conv3_3')(x)

    # Input(28, 28, 256)
    # Output(14, 14, 512)
    x = _depthwise_separable_Conv(x, 512, block=7, bn_axis=bn_axis, strides=(2, 2))
    for i in range(8, 13, 2):
        x = _depthwise_separable_Conv(x, 512, block=i, bn_axis=bn_axis, strides=(1, 1))

    # Input(14, 14, 612)
    # Output(7, 7, 1024)
    x = _depthwise_separable_Conv(x, 1024, block=13, bn_axis=bn_axis, strides=(2, 2))
    x = _depthwise_separable_Conv(x, 1024, block=14, bn_axis=bn_axis, strides=(1, 1))

    x = AveragePooling2D((7, 7), name='pool')(x)

    if include_top:
        x = Flatten()(x)
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
    
    model = Model(inputs, x, name='moblienet')
    model.summary()

    return model


# mobilenet version 2
def _bottleneck(inputs, filters, kernel_size,  block,t, alpha, strides=(1, 1), r=False):

    expansion = "conv" + str(block) + "_Expansion"
    depth_conv = "conv" + str(block) + "_Depthwise"
    point_conv = "conv" + str(block) + "_point"

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] # Depth
    cchannel = int(filters * alpha) # Width

    shortcut = inputs

    x = Conv2D(tchannel , (1, 1), strides=(1, 1), padding='same', use_bias=False, name=expansion)(inputs)
    x = BatchNormalization(axis=channel_axis, name=expansion+"_bn")(x)
    x = ReLU(6, name=expansion + "_relu")(x)

    x =  DepthwiseConv2D((3, 3), strides=strides, padding='same', use_bias=False, name=depth_conv)(x)
    x = BatchNormalization(axis=channel_axis, name=depth_conv + "_bn")(x)
    x = ReLU(6, name=depth_conv + "relu")(x)

    x = Conv2D(cchannel, (1, 1), strides=(1, 1), use_bias=False, name=point_conv)(x)
    x = BatchNormalization(axis=channel_axis, name=point_conv + "_bn")(x)

    if r:
        x = add([x, inputs])
    return x

def _residual_block(inputs, filters, kernel_size, block, t, alpha, strides, n):

    x = _bottleneck(inputs, filters, kernel_size, block=block, t=t, alpha=alpha, strides=strides)

    for i in range(block+1, block+n):
        x = _bottleneck(x, filters, kernel_size, block=i, t=t, alpha=alpha, strides=(1, 1), r=True)
    
    return x

def mobilenetv2(include_top=True, weights="imagenet", input_tensor=None, input_shape=None, pooling=None, classes=1000, t=[1, 6, 6, 6, 6, 6, 6], alpha=1.0):

    input_shape = _obtain_input_shape(input_shape=input_shape,
                                    default_size=224,
                                    min_size=32,
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

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name="conv1_bn")(x)
    x = ReLU(6, name="conv1_relu")(x)

    x = _residual_block(x, 16, (3, 3), block=2, t=t[0], alpha=alpha, strides=(1, 1), n=1)
    x = _residual_block(x, 24, (3, 3), block=3, t=t[1], alpha=alpha, strides=(2, 2), n=2)
    x = _residual_block(x, 32, (3, 3), block=5, t=t[2], alpha=alpha, strides=(2, 2), n=3)
    x = _residual_block(x, 64, (3, 3), block=8, t=t[3], alpha=alpha, strides=(2, 2), n=4)
    x = _residual_block(x, 96, (3, 3), block=12, t=t[4], alpha=alpha, strides=(1, 1), n=3)
    x = _residual_block(x, 160, (3, 3), block=15, t=t[5], alpha=alpha, strides=(2, 2), n=3)
    x = _residual_block(x, 1280, (3, 3), block=18, t=t[6], alpha=alpha, strides=(1, 1), n=1)

    x = Conv2D(1280, (1, 1), strides=(1, 1), name="conv2")(x)
    x = BatchNormalization(axis=bn_axis, name="conv2_bn")(x)
    x = ReLU(6, name="conv2_relu")(x)

    x = AveragePooling2D((7, 7), name="avgpool")(x)

    if include_top:
        x = Flatten()(x)
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
    
    model = Model(inputs, x, name='moblienetv2')
    model.summary()

    return model