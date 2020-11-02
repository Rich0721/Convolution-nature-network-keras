'''
Author: Rich, wu
'''

import os

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, BatchNormalization, Reshape, ZeroPadding2D
from keras import layers
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from keras.layers import GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D
from keras.regularizers import l2
from keras.engine import get_source_inputs
from keras.utils import get_file, layer_utils
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings

def resnet_identity_block(input_tensor, kernel_size, filters, stage, block, bias=False, l2_norm=5e-4):

    filter1, filter2, filter3 = filters

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    
    conv1_reduce_name = "conv" + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_increase"
    conv3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filter1, (1, 1), use_bias=bias, name=conv1_reduce_name,
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name+"/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, use_bias=bias, padding='same', name=conv3_name,
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name+"/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), use_bias=bias, name=conv1_increase_name,
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def resnet_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), bias=False, l2_norm=5e-4):

    filter1, filter2, filter3 = filters

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = "conv" + str(stage) + "_" + str(block) + "_1x1_increase"
    conv1_proj_name = "conv" + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filter1, (1, 1), strides=strides, use_bias=bias, name=conv1_reduce_name,
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filter2, kernel_size, padding='same', use_bias=bias, name=conv3_name,
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn")(x)
    x = Activation('relu')(x)

    x = Conv2D(filter3, (1, 1), name=conv1_increase_name, use_bias=bias,
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm))(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name+"/bn")(x)

    shortcut = Conv2D(filter3, (1, 1), strides=strides, use_bias=bias, 
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm), name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv1_proj_name + "/bn")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50(include_top=True, weights='vggface', input_tensor=None, input_shape=None,
            pooling=None, classes=1000, l2_norm=5e-4):
    

    RESNET50_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
    RESNET50_WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_resnet50.h5'

    input_shape = _obtain_input_shape(input_shape, 
                                    default_size=224,
                                    min_size=32,
                                    data_format=K.image_data_format(),
                                    require_flatten=include_top,
                                    weights=weights)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    
    #x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2),use_bias=False, padding='same',
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm), name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2_bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)
    
    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name='pool5')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
    
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    model = Model(inputs, x, name='vggface_resnet50')
    #model.summary()
    '''
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                    RESNET50_WEIGHTS_PATH,
                                    cache_subdir='./models')
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_resnet50.h5',
                                     RESNET50_WEIGHTS_PATH_NO_TOP,
                                    cache_dir="./models")
    
    model.load_weights(weights_path)
    '''
    if K.backend() == "theano":
        layer_utils.convert_all_kernels_in_model(model)
        if include_top:
            maxpool = model.get_layer(name='avg_pool')
            shape = maxpool.output_shape[1:]
            dense = model.get_layer(name='classifier')
            layer_utils.convert_dense_weights_data_format(dense, shape, 'channels_first')
        
    if K.image_data_format() == "channels_first" and K.backend() == 'tensorflow':
        warnings.warn('You are using the TensorFlow backend, yet you '
                        'are using the Theano '
                        'image data format convention '
                        '(`image_data_format="channels_first"`). '
                        'For best performance, set '
                        '`image_data_format="channels_last"` in '
                        'your Keras config '
                        'at ~/.keras/keras.json.')

    
    
    return model



def resnet101(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000):


    input_shape = _obtain_input_shape(input_shape, 
                                    default_size=224,
                                    min_size=32,
                                    data_format=K.image_data_format(),
                                    require_flatten=include_top,
                                    weights=weights)
    
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(shape=input_shape, tensor=input_tensor)
        else:
            img_input = input_tensor
    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1


    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                kernel_initializer='he_normal', kernel_regularizer=l2(l2_norm), name='conv1/7*7')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7*7/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = resnet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = resnet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = resnet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = resnet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    for i in range(2, 25):
        x = resnet_identity_block(x, 3, [256, 256, 1024], stage=4, block=i)
    
    x = resnet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = resnet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        else:
            x = GlobalMaxPooling2D()(x)

            