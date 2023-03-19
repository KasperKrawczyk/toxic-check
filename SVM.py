import pandas as pd
import numpy as np


class SVM:
    def __init__(self, input_data: np.ndarray,
                 class_col_ind: int,
                 learn_rate=0.001,
                 num_epochs=1000,
                 c=0.01,
                 train_ratio=0.8):
        self.train_data = None
        self.test_data = None
        self.num_train_samples = None
        self.num_test_samples = None

        self.data = input_data[:, class_col_ind + 1:]
        self.num_samples, self.dimensions = self.data.shape
        self.class_col_ind = class_col_ind
        self.class_col = input_data[:, class_col_ind]
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.c = c

        self.train_ratio = train_ratio
        self.w = np.zeros(self.dimensions)
        self.bias = 0
        self._split_data()

    def _split_data(self):
        self.num_train_samples = int(self.num_samples * self.train_ratio)
        self.num_test_samples = int(self.num_train_samples * (1 - self.train_ratio))
        self.train_data = self.data[:self.num_train_samples]
        self.test_data = self.data[self.num_train_samples:]

    def _predict(self, x_i: np.ndarray):
        return np.sign(np.dot(x_i, self.w) - self.bias)

    def _train_predict(self, x_i_class: int, x_i: np.ndarray):
        return x_i_class * (np.dot(x_i, self.w) - self.bias) >= 1

    def fit(self):
        samples_classes = np.where(self.class_col == 0, -1, 1)

        for epoch in range(1, self.num_epochs):
            correct_predictions_per_epoch = 0
            incorrect_predictions_per_epoch = 0
            learning_rate = 1 / epoch
            for index, x_i in enumerate(self.train_data):
                is_correctly_predicted = self._train_predict(samples_classes[index], x_i)

                if is_correctly_predicted:
                    correct_predictions_per_epoch += 1
                    self.w -= self.learn_rate * (self.c * self.w)
                else:
                    incorrect_predictions_per_epoch += 1
                    self.w -= self.learn_rate * (2 * self.c * self.w - np.dot(x_i, samples_classes[index]))
                    self.bias -= self.learn_rate * samples_classes[index]

            print('Epoch {}, incorrect predictions {}, correct predictions {}'
                  .format(epoch, incorrect_predictions_per_epoch, correct_predictions_per_epoch))


if __name__ == '__main__':
    data = np.load('C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\vectorised_matrix.npy')
    svm = SVM(data, 0)
    svm.fit()
