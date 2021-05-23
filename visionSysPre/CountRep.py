import numpy as np
import pandas as pd

class CountRepitition(object):

    def __init__(self, Dist):
        self.Dist = Dist
        self.dist_x = [i[0] for i in Dist]
        self.dist_y = [i[1] for i in Dist]
        self.amplitude_x = max(self.dist_x)
        self.amplitude_y = max(self.dist_y)
        self.gradient_x = 1 # Initialise to a positive gradient slope in the x direction
        self.gradient_y = 1    # Initialise to a negative gradient slope in the y direction
        self.threshold = 0.8
        self.peak_check = False

    def gradientCalc_4pts(self, prev4_cds):
        Grds = []
        for i in range(0, len(prev4_cds)):
            grd = (prev4_cds[i+1]-prev4_cds[i])/prev4_cds[i]
            Grds.append(grd)
        return Grds

    def gradientCalc_2pts(self, prev2_cds):
        if prev2_cds[0]==0:
            prev2_cds[0]=0.000001
        return (prev2_cds[1]-prev2_cds[0])/prev2_cds[0]

    def checkPosToNeg(self, Grds):
        return (numpy.diff(numpy.sign(Grds)) != 0)*1

    # Get based on the past 4 cds - old method
    def gradientRepCounter_4pts(self):
        # Check for gradient change only in positive to negative direction for count increase
        prev4_xcds = self.dist_x[-4]
        prev4_ycds = self.dist_y[-4]
        self.checkPosToNeg(self.gradientCalc(prev4_xcds))

    def gradientRepCounter_2pts(self):
        # Check for gradient change for the last 2 pts only
        if len(self.Dist) > 2: 
            prev2_xcds = self.dist_x[-2:]
            prev2_ycds = self.dist_y[-2:]

            # Compute the gradient value in both x & y direction
            grd_x = self.gradientCalc_2pts(prev2_xcds)
            grd_y = self.gradientCalc_2pts(prev2_ycds)

            # Return the gradient in both x & y direction - if grd_x & grd_y are positive return 0, else return 1
            if grd_x < 0:
                self.gradient_x = 0
            if grd_y < 0:
                self.gradient_y = 0

        return self.gradient_x, self.gradient_y


    def getPeak(self):
        # Check only if there are already more than 2 pts in the Dist Array
        if len(self.Dist) > 2:
            # Get Max Peak in the data distrubution
            maxPeak_x = max(self.dist_x)
            maxPeak_y = max(self.dist_y)

            # Check if current 2 cds are within threshold of the max peak
            currentPeak_x = max(self.dist_x[-2:])
            currentPeak_y = max(self.dist_y[-2:])

            # Check if threshold breahced
            if currentPeak_x > (self.threshold*maxPeak_x):
                self.peak_check = True

            if currentPeak_y > (self.threshold * maxPeak_y):
                self.peak_check = True
            else:
                self.peak_check = False

        return self.peak_check
        




