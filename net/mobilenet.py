'''
Author: Rich, wu
'''

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, BatchNormalization
from tensorflow.python.keras.layers import Conv2D, DepthwiseConv2D, ReLU
from tensorflow.python.keras.layers import AveragePooling2D, Flatten, Dense
from tensorflow.python.keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.python.keras.engine import get_source_inputs
from tensorflow.python.keras.utils import layer_utils, get_file
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings

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
        bn_axis = 3
    else:
        bn_axis = 1

    # Input(224, 224, 3)
    # Output(112, 112, 64)
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu', name='conv1_1')(img_input)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='conv1_2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1_2/bn')(x)
    x = ReLU()(x)
    x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv1_3')(x)
    x = BatchNormalization(axis=bn_axis, name='conv1_3/bn')(x)
    x = ReLU()(x)

    # Input(112, 112, 64)
    # Output(56, 56, 128)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='conv2_1')(x)
    x = BatchNormalization(axis=bn_axis, name='conv2_1/bn')(x)
    x = ReLU()(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv2_2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv2_2/bn')(x)
    x = ReLU()(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='conv2_3')(x)
    x = BatchNormalization(axis=bn_axis, name='conv2_3/bn')(x)
    x = ReLU()(x)
    x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv2_4')(x)
    x = BatchNormalization(axis=bn_axis, name='conv2_4/bn')(x)
    x = ReLU()(x)

    # Input(56, 56, 128)
    # Output(28, 28, 256)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='conv3_1')(x)
    x = BatchNormalization(axis=bn_axis, name='conv3_1/bn')(x)
    x = ReLU()(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv3_2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv3_2/bn')(x)
    x = ReLU()(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='conv3_3')(x)
    x = BatchNormalization(axis=bn_axis, name='conv3_3/bn')(x)
    x = ReLU()(x)
    x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv3_4')(x)
    x = BatchNormalization(axis=bn_axis, name='conv3_4/bn')(x)
    x = ReLU()(x)

    # Input(28, 28, 256)
    # Output(14, 14, 512)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='conv4_1')(x)
    x = BatchNormalization(axis=bn_axis, name='conv4_1/bn')(x)
    x = ReLU()(x)
    x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv4_2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv4_2/bn')(x)
    x = ReLU()(x)
    for i in range(3, 13, 2):
        x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='conv4_'+str(i))(x)
        x = BatchNormalization(axis=bn_axis, name='conv4_'+str(i)+'/bn')(x)
        x = ReLU()(x)
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv4_'+str(i+1))(x)
        x = BatchNormalization(axis=bn_axis, name='conv4_'+str(i+1)+'/bn')(x)
        x = ReLU()(x)

    # Input(14, 14, 612)
    # Output(7, 7, 1024)
    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', name='conv5_1')(x)
    x = BatchNormalization(axis=bn_axis, name='conv5_1/bn')(x)
    x = ReLU()(x)
    x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='conv5_2')(x)
    x = BatchNormalization(axis=bn_axis, name='conv5_2/bn')(x)
    x = ReLU()(x)
    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', name='conv5_3')(x)
    x = BatchNormalization(axis=bn_axis, name='conv5_3/bn')(x)
    x = ReLU()(x)
    x = Conv2D(1024, (1, 1), strides=(1, 1), padding='same', name='conv5_4')(x)
    x = BatchNormalization(axis=bn_axis, name='conv5_4/bn')(x)
    x = ReLU()(x)

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