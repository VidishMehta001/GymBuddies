# Wrapper function for the 3-d/2-d video pose extract

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import train_test_split
import time

class ModelEval(object):

    def __init__(self, video_type):
        self.video_type = video_type
        if self.video_type == "3D":
            self.model_type = "3D-Video-Extract"
        else:
            self.model_type = "2D-Video-Extract"
        
    # Train test split for the 
    def train_test_split(self):
        pass

    
