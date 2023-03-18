import pandas as pd
import numpy as np


class SVM:
    def __init__(self, dataframe: pd.DataFrame, learn_rate=0.001, num_epochs=10000, c=0.1, train_ratio=0.8):
        self.dataframe = dataframe
        self.lear_rate = learn_rate
        self.num_epochs = num_epochs
        self.c = c
        self.num_samples, self.dimensions = dataframe.shape
        self.train_ratio = train_ratio
        self.w = np.zeros(self.dimensions)
        self.train_set = dataframe[:int(self.num_samples * train_ratio)]
        self.test_set = dataframe[int(self.num_samples - self.num_samples * train_ratio):]

