## Frame processing

import blazepose_extract as bp
from blazepose_extract import PoseExt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from LBP import LocalBinaryPattern
import os
from skimage import color
import cv2
from matplotlib import pyplot as plt

## Gabor Filtering
def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# Prepare filter bank kernel - exact
# prepare filter bank kernels
# kernels = []
# for theta in range(4):
#     theta = theta / 4. * np.pi
#     for sigma in (1, 3):
#         for frequency in (0.05, 0.25):
#             kernel = np.real(gabor_kernel(frequency, theta=theta,
#                                           sigma_x=sigma, sigma_y=sigma))
#             kernels.append(kernel)


# CNN Classifier Implementation
class CNN_Model(object):

    def __init__(self, train_images, train_labels):
        self.train_images = np.array(train_images).astype(float)
        self.input_shape = self.train_images.shape
        print(self.input_shape)
        self.train_labels = np.array(train_labels).astype(int)
        self.model = self.build_model()


    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape[1:]))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (3,3), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2, activation="relu"))
        return model

    def summary(self):
        return self.model.summary()


    def fit_model(self):
        self.model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        history = self.model.fit(self.train_images, self.train_labels, epochs=30, validation_split=0.2)
        self.model.save("CNN_Model")
        return history




def train_cnn():
    training_images, training_labels = merge_keypts()
    cnn_model = CNN_Model(train_images=training_images, train_labels=training_labels)
    cnn_model.summary()
    fitted_model = cnn_model.fit_model()



# def merge_keypts():
#     data = []
#     labels = []
#     for each_exercise in ["PushUp", "Squats"]:
#         loc = "C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/KeyPts/{}".format(each_exercise)
#         for files in os.listdir(loc):
#             file_loc = loc + "/" +files
#             img = cv2.imread(file_loc)
#             img = color.rgb2gray(img)
#             img = np.atleast_3d(img)
#             data.append(img)

#             if each_exercise == "PushUp":
#                 labels.append(0)
#             else:
#                 labels.append(1)

#     return data, labels
        
# Train the CNN neural network - please ensure that you have sufficient memory allocation. Best done on GPU. 
#train_cnn()

class CNN_Model2(object):

    def __init__(self, train_images, train_labels):
        self.train_images = np.array(train_images).astype(float)
        self.input_shape = self.train_images.shape
        print(self.input_shape)
        self.train_labels = np.array(train_labels).astype(int)
        self.model = self.model()


    def model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (1,1), activation='relu', input_shape=(5,12,1)))
        model.add(layers.MaxPooling2D(pool_size=(1,1)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(64, (1,1), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(1,1)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, (1,1), activation="relu"))
        model.add(layers.MaxPooling2D(pool_size=(1,1)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2, activation="relu"))
        return model


    def summary(self):
        return self.model.summary()


    def fit_model(self):
        self.model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        history = self.model.fit(self.train_images, self.train_labels, epochs=500, validation_split=0.2)
        self.model.save("CNN_Model")
        return history




def get_training_data():
    data = []
    labels = []
    
    # Loop over each exercise in the training dataset
    for each_exercise in ["pushups", "squats"]:

        each_exercise = each_exercise + "/"

        ## Initialise the training_dir 
        training_dir = "C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train/"

        # Loop over all the training images from the training datasets
        for imagePath in os.listdir(os.path.join(training_dir, each_exercise)):
            with open("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train/"+each_exercise+imagePath, 'rb') as f:
                imageArray = np.load(f)                
                imageArray = np.resize(imageArray, (5,12))
                imageArray = np.atleast_3d(imageArray)

            data.append(imageArray)
            # 0 for push up 1 for squats
            if each_exercise.split(sep="/")[0] == "pushups":
                labels.append(0)
            elif each_exercise.split(sep="/")[0]=="squats":
                labels.append(1)
                
    return data, labels

data, labels = get_training_data()
cnn = CNN_Model2(train_images=data, train_labels=labels)
history = cnn.fit_model()


    
