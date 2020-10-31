'''
Authors: Rich, Wu
'''
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras import layers
from keras.layers import Input, BatchNormalization, Lambda, concatenate
from keras.layers import Conv2D, Dense, Flatten, Activation
from keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.regularizers import l2
from keras.engine import get_source_inputs
from keras.utils import get_file, layer_utils
from keras_applications.imagenet_utils import _obtain_input_shape
import warnings

def identity_block(input_tensor, kernel_size, filters, stage, block, bias=False, group_channels=4, cardinality=32, weight_decay=5e-4):
    
    filter1, filter2, filter3 = filters

    if K.image_data_format() == "channels_last":
        bn_axis = 1
    else:
        bn_axis = 3
    
    conv_1x1_first = "conv" + str(stage) + "_" + str(block) + "_first_1x1"
    conv_3x3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"
    conv_1x1_last = "conv" + str(stage) + "_" + str(block) + "_last_1x1"

    x = Conv2D(filter1, (1, 1), use_bias=bias, kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_1x1_first)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv_1x1_first+"/bn")(x)
    x = Activation("relu")(x)


    group_convolution = []
    for c in range(cardinality):
        temp = Lambda(lambda z: z[:, :, :, c *  group_channels : (c+1) * group_channels])(x)
        temp = Conv2D(group_channels, (3, 3), padding='same', use_bias=bias, kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_3x3_name + "/" + str(c))(temp)
        group_convolution.append(temp)
    
    group_merge = concatenate(group_convolution, axis=-1)
    x = BatchNormalization(axis=bn_axis, name=conv_3x3_name+"/bn")(group_merge)
    x = Activation("relu")(x)


    x = Conv2D(filter3, (1, 1), use_bias=bias, kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_1x1_last)(x)
    x = BatchNormalization(axis=bn_axis, name=conv_1x1_last+"/bn")(x)

    x = layers.add([x, input_tensor])
    x = Activation("relu")(x)
    
    return x

def resnet_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), bias=False, group_channels=4, cardinality=32, weight_decay=5e-4):

    filter1, filter2, filter3 = filters

    if K.image_data_format() == "channels_last":
        bn_axis = 1
    else:
        bn_axis = 3
    
    conv_1x1_first = "conv" + str(stage) + "_" + str(block) + "_first_1x1"
    conv_3x3_name = "conv" + str(stage) + "_" + str(block) + "_3x3"
    conv_1x1_last = "conv" + str(stage) + "_" + str(block) + "_last_1x1"
    conv_shortcut = "conv" + str(stage) + "_" + str(block) + "_shortcut"

    x = Conv2D(filter1, (1, 1), use_bias=bias, strides=strides, kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_1x1_first)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv_1x1_first+"/bn")(x)
    x = Activation("relu")(x)

    group_convolution = []
    for c in range(cardinality):
        temp = Lambda(lambda z: z[:, :, :, c * group_channels: (c + 1) * group_channels])(x)
        temp = Conv2D(group_channels, (3, 3), padding='same', use_bias=bias, kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_3x3_name + "/" + str(c))(temp)
        group_convolution.append(temp)
    
    group_merge = concatenate(group_convolution, axis=-1)
    x = BatchNormalization(axis=bn_axis, name=conv_3x3_name+"/bn")(group_merge)
    x = Activation("relu")(x)

    x = Conv2D(filter3, (1, 1), use_bias=bias,kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_1x1_last)(x)
    x = BatchNormalization(axis=bn_axis, name=conv_1x1_last+"/bn")(x)

    shortcut = Conv2D(filter3, (1, 1), use_bias=bias, strides=strides, kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name=conv_shortcut)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=conv_shortcut+"/bn")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("relu")(x)
    return x

def resnext50(include_top=True, weights='vggface', input_tensor=None, input_shape=None,
            pooling=None, classes=8631, weight_decay=5e-4):

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

    x = Conv2D(64, (7, 7), use_bias=False, strides=(2, 2), padding='same',kernel_initializer='he_normal', 
                    kernel_regularizer=l2(weight_decay), name='conv1/7x7_s2')(img_input)
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = resnet_block(x, (3, 3), [128, 128, 256], stage=2, block=1, strides=(1, 1))
    x = identity_block(x, (3, 3), [128, 128, 256], stage=2, block=2)
    x = identity_block(x, (3, 3), [128, 128, 256], stage=2, block=3)

    x = resnet_block(x, (3, 3), [256, 256, 512], stage=3, block=1)
    x = identity_block(x, (3, 3), [256, 256, 512], stage=3, block=2)
    x = identity_block(x, (3, 3), [256, 256, 512], stage=3, block=3)
    x = identity_block(x, (3, 3), [256, 256, 512], stage=3, block=4)

    x = resnet_block(x, (3, 3), [512, 512, 1024], stage=4, block=1)
    x = identity_block(x, (3, 3), [512, 512, 1024], stage=4, block=2)
    x = identity_block(x, (3, 3), [512, 512, 1024], stage=4, block=3)
    x = identity_block(x, (3, 3), [512, 512, 1024], stage=4, block=4)
    x = identity_block(x, (3, 3), [512, 512, 1024], stage=4, block=5)
    x = identity_block(x, (3, 3), [512, 512, 1024], stage=4, block=6)

    x = resnet_block(x, (3, 3), [1024, 1024, 2048], stage=5, block=1)
    x = identity_block(x, (3, 3), [1024, 1024, 2048], stage=5, block=2)
    x = identity_block(x, (3, 3), [1024, 1024, 2048], stage=5, block=3)

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
    
    model = Model(inputs, x, name='resnext50')
    model.summary()

    return model