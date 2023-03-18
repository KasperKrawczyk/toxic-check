import pandas as pd
import numpy as np


class SVM:
    def __init__(self, dataframe: pd.DataFrame, class_col_name: str, learn_rate=0.001, num_epochs=10000, c=0.1, train_ratio=0.8):
        self.dataframe = dataframe
        self.class_col_name = class_col_name
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.c = c
        self.num_samples, self.dimensions = dataframe.shape
        self.train_ratio = train_ratio
        self.w = np.zeros(self.dimensions)
        self.bias = 0
        self.train_set = dataframe[:int(self.num_samples * train_ratio)]
        self.test_set = dataframe[int(self.num_samples - self.num_samples * train_ratio):]

    def predict(self, x_i: pd.Series):
        return np.sign(np.dot(x_i, self.w) - self.bias)

    def train_predict(self, x_i_class: int, x_i: pd.Series):
        return x_i_class * (np.dot(x_i, self.w) - self.bias) >= 1

    def fit(self):
        samples_classes = np.where(self.train_set[self.class_col_name] == 0, -1, 1)

        for epoch in range(self.num_epochs):
            for index, x_i in enumerate(self.train_set):
                is_correctly_predicted = self.train_predict(samples_classes[index], x_i)

                if is_correctly_predicted:
                    self.w -= self.learn_rate * (0.5 * self.c * self.w)
                else:
                    self.w -= self.learn_rate * (0.5 * self.c * self.w - np.dot(x_i, samples_classes[index]))
                    self.bias -= self.learn_rate * samples_classes[index]


