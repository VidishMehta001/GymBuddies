from tensorflow.keras import datasets, layers, models
import os
import numpy as np

class ClassifyExercise:
    def __init__(self, stride = 30):
        self.window_size = 30
        self.stride = stride
        self.stack = []
        self.output_stack = []
        self.frame = 0
        self.model = models.load_model("exercise_classifier_model_e10.h5")
        self.filtered_output = None

    def add_to_stack(self, x,y,z):
        new_frame = np.stack([x,y,z])
        new_frame = new_frame.reshape(3,12,1)
        self.stack.append(np.nan_to_num(new_frame))

        if len(self.stack) > self.window_size:
            self.stack.pop(0)

    def add_to_output_stack(self, output):
        self.output_stack.append(output)
        if len(self.output_stack) > 5:
            self.output_stack.pop(0)

        vals, counts = np.unique(self.output_stack, return_counts=True)
        index = np.argmax(counts)
        return vals[index]

    def predict(self,x,y,z):
        self.frame +=1
        self.add_to_stack(x,y,z)

        if len(self.stack)==self.window_size and self.frame%self.stride==0:
            output = np.argmax(self.model.predict(np.array([self.stack])))
            self.filtered_output = self.add_to_output_stack(output)

        return self.filtered_output
