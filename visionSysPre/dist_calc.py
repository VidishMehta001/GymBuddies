from scipy.spatial import distance
import numpy as np
import pandas as pd

class PairDistCalc(object):

    def __init__(self, refx, refy, x, y, refxy_pair, curxy_pair):
        self.refx = refx
        self.refy = refy
        self.x = x
        self.y = y
        self.refxy_pair = refxy_pair #(x,y) cds for each of the measured points - Nested list of the ref x,y pairs
        self.curxy_pair = curxy_pair #(x,y) cds for each of the new points - Nested list of the xy pair


    #Length of data pts check
    def lenCheck(self):

        if (len(self.x)==len(self.refx) and len(self.y)==len(self.refy)):
            return True
        return False

    def pairLenCheck(self):
        if (len(self.refxy_pair) == len(self.curxy_pair)):
            return True
        return False

    # Euclidean distance implementation - for rep counting
    def eucDist(self):
        if self.lenCheck()==True:
            # Calculate Euclidean dist between x & refx
            xdist = distance.euclidean(self.x,self.refx)            
            # Calculate Euclidean dist between y & refy
            ydist = distance.euclidean(self.y, self.refy)

            return xdist, ydist

    def isNoneCheck(self):
        if (len(self.refxy_pair)==0 and len(self.curxy_pair) == 0):
            return True
        return False

    # Euclidean distance implementation - for feature map (LBP)
    def eucDist_FM(self):
        # Initialise the array
        FM_Array = []

        if ((self.pairLenCheck() == True) and (self.isNoneCheck() == False)):
            # Calculate the Euclidean Distance between each pair
            # sqrt((x1-x2)^2 +(y1-y2)^2) # xy are lists

            for index, (refxy, curxy) in enumerate(zip(self.refxy_pair, self.curxy_pair)):
                # Calculate the euclidean distance between refxy and curxy
                dist = distance.euclidean(refxy, curxy)
                # Append the distance to the Feature Map
                FM_Array.append(dist)

            return FM_Array
        
        return FM_Array




