## Distance calculator && plotter
from scipy.spatial import distance

class PairDistCalc(object):

    def __init__(self, refx, refy, x, y):
        self.refx = refx
        self.refy = refy
        self.x = x
        self.y = y

    #Length of data pts check
    def lenCheck(self):
        if (len(self.x)==len(self.refx) and len(self.y)==len(self.refy)):
            return True
        return False

    # Euclidean distance implementation
    def eucDist(self):
        if self.lenCheck()==True:
            # Calculate Euclidean dist between x & refx
            xdist = distance.euclidean(self.x,self.refx)
            
            # Calculate Euclidean dist between y & refy
            ydist = distance.euclidean(self.y, self.refy)

            return xdist, ydist

    # # Cdist implementation - euclidean, cityblock, cosine, hamming, jaccard, 
    # def cdist(self):
    #     if self.lenCheck()==True:
    #         # Calculate the cdist between x & ref x






