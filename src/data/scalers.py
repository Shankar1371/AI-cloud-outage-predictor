import numpy as np

class PerMachineStandardizer:
    def __init__(self, base):
        self.means= None
        self.stds = None
    #this is a constructer for the class
    #initializes an instance attribute self. means to None.

    def fit(self, X_list):

        # X_list: list of (Ti, F) arrays per machine for train split
        stacked = np.concatenate(X_list, axis=0)
        self.means = stacked.mean(axis=0)
        self.stds = stacked.std(axis=0) + 1e-6
        #calculates the standard deviation and mean
        return self

    def transform(self, X):
        return (X - self.means) / self.stds
