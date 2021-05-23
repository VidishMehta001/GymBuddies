from SVM import SVM_Classifier
from LBP import LocalBinaryPattern
import numpy as np
from skimage import color
import cv2
import tensorflow as tf

class ExerciseClassifier(object):

    # Initialise the class
    def __init__(self, model, image, feature_maps, keypts, predict, pred, Training=False):
        self.model = model
        self.predict = predict
        self.pred = pred
        self.Training = Training
        if self.model == "SVM":
            self.feature_extractors = "LBP"
        elif self.model == "CNN":
            self.feature_extractors = "KeyPts"
        else:
            raise ValueError("Model selected does not have feature selector! Please use another model")
        if self.Training == False:
            if self.model == "SVM":
                self.model_file = "finalized_model_rgb.pkl"
            elif self.model == "CNN":
                self.model_file = "CNN_Model"
            else:
                raise ValueError("Model selected does not have feature selector! Please use another model")
        self.image = image
        self.feature_maps = feature_maps
        self.keypts = keypts

    # Main function for calling the classifier
    def exercise_classifier(self):
        if self.model=="SVM":
            return self.svm_classifier()
        elif self.model == "CNN":
            return self.image_cnn_prep()
        else:
            raise ValueError("Model selected does not have feature selector! Please use another model")


    # Used for SVM based classification
    def svm_classifier(self):
        if (self.predict==False) and (len(self.feature_maps) >=2):
            svm = SVM_Classifier(svm_c=100,feature_extractors=self.feature_extractors)
            self.pred, self.image = svm.predictTestData(self.image, np.matrix(self.feature_maps))
            self.predict = True
            return self.pred, self.predict
        return self.pred, self.predict
        

    # Used for CNN Based classification
    def image_cnn_prep(self):
        if (self.keypts == True) and (self.predict==False):
            self.image = cv2.resize(self.image, (1280, 720), interpolation=cv2.INTER_AREA)
            self.image = color.rgb2gray(self.image)
            self.image = np.atleast_3d(self.image)
            cnn_model = tf.keras.models.load_model('CNN_Model')
            self.image = np.array([self.image])
            pred = cnn_model.predict(self.image).tolist()
            cnn_pred = pred[0][1]
            if cnn_pred == 0:
                self.pred = "PushUp"
            else:
                self.pred = "Squats"
            self.predict = True
        
        return self.pred, self.predict



    



    

        
        