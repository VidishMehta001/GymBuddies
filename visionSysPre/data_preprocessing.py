import numpy as np
import pandas as pd
import os

class DataPreprocessing(object):

    def __init__(self, n_samp):
        self.n_samp = n_samp

    def frame_sampling(self, frame_no):
        sampled_frame = False
        if frame_no % self.n_samp == 0:
            sampled_frame = True
            frame_no = frame_no + 1
            return sampled_frame, frame_no
        frame_no = frame_no + 1
        return sampled_frame,frame_no

    # Counter function - checks for gradient transitions between 1 to 0
    @staticmethod
    def BinTransitions(grds):
        # Initialise Count
        Count = 0
        # for i in range(0, len(grds)):
        if len(grds) > 2:
            if ((grds[-2:][0] == 1) and (grds[-1:][0] == 0)):
                Count = Count + 1
            else:
                pass
        return Count

    #Check if key points need to be saved
    @staticmethod
    def keyPointCheck(grds):
        kp_check = 0 #No key points detected
        for i in range(0, len(grds)):
            if len(grds) > 2:
                if ((grds[i-1] == 1) and (grds[i] == 0)):
                    kp_check = 1 # Minima
                elif ((grds[i-1]==0) and (grds[i] == 1)):
                    kp_check = 2 # Maxima
                else:
                    pass
        return kp_check

    #Length of data pts check
    @staticmethod
    def checkLen(refx, refy, x, y):
        if (len(x)==len(refx) and len(y)==len(refy)):
            return True
        return False


    # Get the nested pair list of each of the x,y cds - RGB Frame Videos
    @staticmethod
    def getNestedPair(refx, refy, x, y):
        ## 4 list - refx = [0.2,0.3...]
        #Initialise the ref pair x,y and current pair x,y
        refxy_pair, curxy_pair = [], []
        if DataPreprocessing.checkLen(refx, refy, x, y) == True:

            # Zip & enumerate if all lengths are equal
            for index, items in enumerate(zip(refx, refy, x, y)):
                refxy_pair.append([refx[index], refy[index]])
                curxy_pair.append([x[index], y[index]])

            return refxy_pair, curxy_pair

        return refxy_pair, curxy_pair


    # Save to temporary file
    def saveArray(self, matrix):
        outfile = TemporaryFile()
        np.save(outfile, matrix)
        return

    # Save to a .npy file extension
    def saveFeatureMap(self, matrix, exercise, version, video_type):
        print(video_type)
        file_name = exercise+"_"+version
        base_path = os.getcwd()
        if video_type == "RGB-D":
            with open("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train/{}/{}.npy".format(exercise,file_name), 'wb') as f:
                np.save(f, matrix)
        else:
            with open("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train_rgb/{}/{}.npy".format(exercise,file_name), 'wb') as f:
                np.save(f, matrix)
        return

    @staticmethod
    def saveFM(matrix, exercise, version):
        file_name = exercise+"_"+version
        with open("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/exercise_train/{}/{}.npy".format(exercise,file_name), 'wb') as f:
                np.save(f, matrix)
        return

        