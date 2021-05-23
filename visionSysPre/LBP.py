from skimage import feature
import numpy as np
import matplotlib.pyplot as plt
import random

class LocalBinaryPattern(object):

    def __init__(self, numpoints, radius, epsilon, plot="Yes"):
        # store the number of points and radius
        self.numpoints = numpoints
        self.radius = radius
        self.epsilon = epsilon
        self.plot=plot
        

    def describe(self, image, exercise):
        # Compute the local binary representation of the distance metrics
        # Create the histogram of the patterns

        lbp = feature.local_binary_pattern(image, self.numpoints, self.radius, method="uniform")
        #print("---------------------")
        #Plt the lbp features if specifically requested

        (hist,_) = np.histogram(lbp.ravel(), bins=np.arange(0, self.numpoints + 3), range=(0, self.numpoints + 2))

        #Normalise the histogram
        hist = hist.astype("float")
        #hist = hist/(hist.sum()+self.epsilon)

        if self.plot != "No":
            self.pltHist(image, hist, exercise)

        #Return the histogram of the local binary patterns
        #print(hist)
        return hist

    
    def pltHist(self, image, hist1, exercise):
        #print(hist1)
        fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2)
        #ax1.axis("off")
        #ax2.axis("off")
        ax1.set_title("Distance Image Vector")
        ax2.set_title("LBP Feature Vector")
        ax2.hist(hist1, density=False, bins=26)
        ax2.set_ylabel("LBP Frequency Histogram")
        ax2.set_xlabel("Bins")
        ax1.set_ylabel("Frequency of Distance Image Vector")
        ax1.set_xlabel("Bins")
        ax1.hist(image, density=False, bins=26)
        fig.tight_layout()
        fig.savefig("C:/Users/vidis/OneDrive/Desktop/project/Gym_Buddies/visionSysPre/LBP/{}_{}.png".format(exercise, str(random.randint(1,10))))
        return


