
import cv2
import os
import warnings
import numpy as np
from glob import glob

import tensorflow as tf
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras import backend as K


import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
class Test(object):

    def __init__(self, test_folder, model_file):
        self._test_folder = test_folder
        self._nb_classes = os.listdir(test_folder)
        print(self._nb_classes)
        self._model = load_model(model_file)
        self._class_acc = {}
        self._confusion_matrix = np.zeros((len(self._nb_classes), len(self._nb_classes)), dtype='int32')
    
    @property
    def class_acc(self):
        return self._class_acc
    
    def verification(self):
        
        all_acc = 0
        all_total = 0
        for p in self._nb_classes:

            images = glob(os.path.join(self._test_folder, p, "*.jpg"))
            acc = 0
            total = 0

            for image in images:

                img = cv2.imread(image)
                img = img[:, :, [2, 1, 0]]
                img = img / 255
                img = np.expand_dims(img, axis=0)

                index = self._model.predict(img)
                list_predict = np.ndarray.tolist(index[0])
                
                max_index = list_predict.index(max(list_predict))
                self._confusion_matrix[self._nb_classes.index(p)][max_index] += 1

                #print("{}:{}".format(p, self._nb_classes[max_index]))
                if self._nb_classes[max_index] == p:
                    acc += 1
                    all_acc += 1
                all_total += 1
                total += 1
            print("{} verified!".format(p))
            self._class_acc[p] = round(acc/ total, 2)

    def printResult(self):
        acc = 0
        
        for p in self._nb_classes:
            print("{}: {}".format(p, self._class_acc[p]))
            acc += self._class_acc[p]

        return (acc/len(self._nb_classes))
    
    def confusionMatrix(self, title):

        labels = self._nb_classes
        df_cm = pd.DataFrame(self._confusion_matrix, index=labels, columns=labels)
        sb.set(font_scale=1.3)
        fig = plt.figure(figsize=(10, 10))
        heat_map = sb.heatmap(df_cm, fmt='d',
                            cmap='BuPu', annot=True, cbar=True, center=True,
                            linewidths=0.5, linecolor='w', square=True,
                            cbar_kws={'label': 'Number of Prediction', 'orientation': 'vertical'})
        
        heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=0, fontsize=10)
        heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=10)
        heat_map.xaxis.set_ticks_position("bottom")
        heat_map.set_ylim(len(self._nb_classes), 0)
        plt.title("Confusion matrix : {}".format(title))
        plt.xlabel("Predicted Tree", labelpad=10)
        plt.ylabel("True Tree", labelpad=10)
        plt.savefig(os.path.join("./confusion", title + ".jpg"))


if __name__ == "__main__":
    K.clear_session()
    
    test = Test(test_folder="./Original/test", model_file='./models/vgg19_ori.hdf5')
    print("VGG19_ori")
    test.verification()
    test.confusionMatrix(title="vgg19_ori")
    accuracy['VGG19_ori']  = test.printResult()