import matplotlib.pyplot as plt
import numpy as np
from LBP import LocalBinaryPattern
from sklearn import svm
import cv2
import os
import pickle

class SVM_Classifier(object):

    def __init__(self, svm_c, feature_extractors):
        self.C = svm_c
        self.feature_extractors = feature_extractors
        # Hardcoded Feature Set for the SVM Model
        self.file_name = "finalized_model_rgb.pkl"
        self.exercise = ['pushups', 'squats']
        if self.feature_extractors == "LBP":
            self.feature_extractors = LocalBinaryPattern(numpoints=24, radius=8, epsilon=1e-7)
        
    def normalisation(self):
        pass

    def fitTrainingData(self):
        data = []
        labels = []
        
        # Loop over each exercise in the training dataset
        for each_exercise in self.exercise:
            print("Training for " + str(each_exercise))

            each_exercise = each_exercise + "/"

            # Loop over all the training images from the training datasets
            for imagePath in os.listdir(os.path.join("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train/", each_exercise)):

                with open("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train/"+each_exercise+imagePath, 'rb') as f:
                    imageArray = np.load(f)

                # load the distance based image vector from concatenation of 4 key points and describe it as a HOG
                #image = cv2.imread(imageArray)
                hist = self.feature_extractors.describe(imageArray, each_exercise)
            
                # extract the label from the image path & then update the labels and data list
                # Eg. labels = ['push up', 'pull up', 'squats']
                #      data = [[array, array, array, array], [array, array], [array, array, array]]

                data.append(hist)
                labels.append(each_exercise.split(sep="/")[0])

        # Train the SVM classifier
        model = svm.LinearSVC(C=self.C, penalty='l2', loss='hinge', random_state=42, verbose=0, max_iter=1000)
        model.fit(data, labels)

        # Save the model
        with open(self.file_name, 'wb') as file:
            pickle.dump(model, file)

        return model   

    def predictTestData(self, image, testing_data, debug=False):
        # Train the model based on training data
        #model = self.fitTrainingData()

        # Load the saved model
        with open(self.file_name, 'rb') as file:
            loaded_model = pickle.load(file)
        
        # Load the distance based metrics
        #DBM = cv2.imread(testing_data)
        hist = self.feature_extractors.describe(testing_data, exercise=None)
        hist = hist.reshape(1,-1)

        # Use the trained model to predict the hist
        pred = loaded_model.predict(hist)
        #print("SVM Model predicted: " + str(pred[0]))

        # Display on the raw image file
        #cv2.putText(image, pred[0], (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        if debug==True:
            cv2.imshow("Image", Image)
            cv2.waitKey(0)

        return pred, image
        







    

