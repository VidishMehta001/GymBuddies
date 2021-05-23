import sklearn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

# Implementation of the MLP classifier based on the saved feature maps
# Networks such as MLP classifier are trained separately
class MLPClassifier(object):

    def __init__(self, training_data, training_labels):
        self.training_data = training_data
        self.training_labels = training_labels


    def train(self):
        pass

    def save_model(self):
        pass
