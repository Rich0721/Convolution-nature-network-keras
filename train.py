'''
Author: Rich, Wu
Datetime: 2020/06/24
'''
import os

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.python.keras.engine import get_source_inputs
from tensorflow.python.keras.utils import get_file, layer_utils
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD 

from net.vgg import VGG16, VGG19
from net.cspresnet import cspresnet50
from net.resnet import resnet50
from net.resnet_sp import resnetsp50

class Train(object):

    def __init__(self, train_folder, vali_folder, weights_file, storage_file, width=224, height=224, batch_size=8, min_delta=1e-3, patience=3):
        
        
        self._batch_size = batch_size
        self._width = width
        self._height = height
        self._nb_classes, self._train_genrator, self._vali_generator = self._dataGenerator(train_folder, vali_folder)
        
        self._weights_file = weights_file
        self._storage_file = storage_file
        
        self._checkpoint = ModelCheckpoint(storage_file, monitor='val_acc', verbose=1, save_best_only=True)
        self._monitor = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='auto', restore_best_weights=True)

    def _dataGenerator(self, train_folder, vali_folder):

        nb_class = len(os.listdir(train_folder))

        train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
        )

        vali_datagen = ImageDataGenerator(
        rescale=1./255,
        )

        train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=(self._height, self._width),
        batch_size=8,
        class_mode='categorical'
        )

        vail_generator = vali_datagen.flow_from_directory(
        vali_folder,
        target_size=(self._height, self._width),
        batch_size=8,
        class_mode='categorical'
        )

        return nb_class, train_generator, vail_generator

    def vgg16(self, epochs=100):


        model = VGG16(include_top=False, input_shape=(self._height, self._width, 3), weights_file=self._weights_file)
        last_layer = model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dense(4096, activation='relu', name='fc7')(x)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size
        callbacks =[self._checkpoint, self._monitor]
        
        train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps,
            callbacks=callbacks)

    def vgg19(self, epochs=100):

        model = VGG19(include_top=False, input_shape=(self._height, self._width, 3), weights_file=self._weights_file)
        last_layer = model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        x = Dense(4096, activation='relu', name='fc6')(x)
        x = Dense(4096, activation='relu', name='fc7')(x)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size
        callbacks =[self._checkpoint, self._monitor]
        
        train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps,
            callbacks=callbacks)

    def resnet(self, epochs=100):

        model = resnet50(include_top=False, input_shape=(self._height, self._width, 3))
        last_layer = model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size

        callbacks =[self._checkpoint, self._monitor]
        
        train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps,
            callbacks=callbacks)
    def cspresnet(self, epochs=100):

        model = cspresnet50(include_top=False, input_shape=(self._height, self._width, 3))
        last_layer = model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size

        callbacks =[self._checkpoint, self._monitor]
        
        train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps,
            callbacks=callbacks)

    def resnetsp(self, epochs=100):

        model = resnetsp50(include_top=False, input_shape=(self._height, self._width, 3))
        last_layer = model.get_layer("pool5").output
        x = Flatten(name='flatten')(last_layer)
        out = Dense(self._nb_classes, activation='softmax', name='fc8')(x)
        train_model = Model(model.input, out)
        train_model.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=1e-4, momentum=0.9),
                            metrics=['accuracy'])

        steps_per_epoch =  self._train_genrator.n // self._batch_size
        validation_steps =  self._vali_generator.n // self._batch_size

        callbacks =[self._checkpoint, self._monitor]
        
        train_model.fit_generator(
            self._train_genrator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=self._vali_generator,
            validation_steps=validation_steps,
            callbacks=callbacks)


if __name__ == "__main__":
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    train = Train(train_folder="./pattern/train", vali_folder="./pattern/veri", weights_file=None, storage_file="./models/resnetsp_2.hdf5")
    train.resnetsp()
    '''
    K.clear_session()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(sess)
    train = Train(train_folder="./pattern/train", vali_folder="./pattern/veri", weights_file=None, storage_file="./models/resnet.hdf5")
    train.resnet()
    '''