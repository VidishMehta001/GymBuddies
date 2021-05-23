from sklearn import svm
from sklearn import metrics
import pickle
import cv2
from skimage import exposure

class HogSvmClassifier:
    def __init__(self):
        pass

    def hog_svm_predict(self, image_c):
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        IMG_WIDTH = 640
        IMG_HEIGHT = 416
        hog = cv2.HOGDescriptor((IMG_WIDTH, IMG_HEIGHT), blockSize, blockStride, cellSize, nbins)
        img = cv2.resize(image_c, (IMG_WIDTH, IMG_HEIGHT))
        img_hog = hog.compute(img)
        img_hog = exposure.rescale_intensity(img_hog, out_range=(0, 255)).astype("uint8")

        with open('hog_svm_model.pkl', 'rb') as file:
            clf = pickle.load(file)
        img_hog = img_hog.reshape(1, -1)
        pred2 = clf.predict(img_hog)
        pred = pred2[0]
        return pred