import string

import numpy as np
import pandas as pd
import sklearn.utils

from TfIdfVectoriser import TfIdfVectoriser


class SVM:
    c_params = [10 ** p for p in range(-5, 1)]
    c_param_default = 10 ** -5

    def __init__(self,
                 train_data_class_col: np.ndarray,
                 input_train_data: np.ndarray,
                 test_data_class_col: np.ndarray,
                 input_test_data: np.ndarray,
                 learn_rate=0.1,
                 num_epochs=100,
                 train_ratio=0.8):
        np.random.shuffle(input_train_data)
        np.random.shuffle(input_test_data)
        self.num_train_samples = input_train_data.shape[0]
        self.num_test_samples = input_test_data.shape[0]

        # add the intercept term (bias) as last column, filled with 1s
        self.train_data = np.c_[input_train_data, np.ones(self.num_train_samples)]
        self.test_data = np.c_[input_test_data, np.ones(self.num_test_samples)]
        # shape should return a 2-tuple
        _, self.num_features = self.train_data.shape
        self.train_class_col = train_data_class_col
        self.test_class_col = test_data_class_col
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.c = SVM.c_param_default

        self.train_ratio = train_ratio
        self.w = np.zeros(self.num_features)

    def _predict(self, x_i: np.ndarray):
        return np.sign(np.dot(x_i, self.w))

    def _train_predict(self, x_i_class: int, x_i: np.ndarray):
        return x_i_class * (np.dot(x_i, self.w)) >= 1

    def fit_gd(self, c_param: float = c_param_default, print_epoch_result: bool = True, wipe: bool = True):
        if wipe:
            self._wipe()

        print("c param={}".format(c_param))
        samples_classes = np.where(self.train_class_col == 0, -1, 1)

        for epoch in range(1, self.num_epochs):
            self.train_data, samples_classes = sklearn.utils.shuffle(self.train_data, samples_classes,
                                                                     random_state=0)
            correct_predictions_per_epoch = 0
            incorrect_predictions_per_epoch = 0
            self.learn_rate = 1 / epoch
            for index, x_i in enumerate(self.train_data):
                is_correctly_predicted = self._train_predict(samples_classes[index], x_i)

                if is_correctly_predicted:
                    correct_predictions_per_epoch += 1
                    self.w -= (1 - self.learn_rate) * (c_param * self.w)
                else:
                    incorrect_predictions_per_epoch += 1
                    self.w -= (1 - self.learn_rate) * (c_param * self.w - np.dot(x_i, samples_classes[index]))

            if print_epoch_result and epoch % 10 == 0:
                print('Epoch={}, incorrect predictions={}, correct predictions={}'
                      .format(epoch, incorrect_predictions_per_epoch, correct_predictions_per_epoch))

    def fit_sgd(self, c_param: float = c_param_default, print_epoch_result: bool = True, wipe: bool = True):
        if wipe:
            self._wipe()

        samples_classes = np.where(self.train_class_col == 0, -1, 1)

        for epoch in range(1, self.num_epochs):
            self.train_data, samples_classes = sklearn \
                .utils \
                .shuffle(self.train_data, samples_classes, random_state=np.random.randint(0, 42))
            correct_predictions_per_epoch = 0
            incorrect_predictions_per_epoch = 0
            self.learn_rate = 1 / epoch
            for index, x_i in enumerate(self.train_data):

                cost, predicted = self.calculate_cost_sgd(samples_classes[index], x_i, c_param)
                self.w -= (self.learn_rate * cost)

                if predicted:
                    correct_predictions_per_epoch += 1
                else:
                    incorrect_predictions_per_epoch += 1

            if print_epoch_result and epoch % 10 == 0:
                print('Epoch={}, incorrect predictions={}, correct predictions={}'
                      .format(epoch, incorrect_predictions_per_epoch, correct_predictions_per_epoch))

    def calculate_cost_sgd(self, x_i_class: int, x_i: np.ndarray, c_param: float):
        margin = 1 - (x_i_class * (np.dot(x_i, self.w)))
        weights_d = np.zeros(len(self.w))

        # hinge loss function
        if max(0, margin) == 0:
            weights_d += self.w
            predicted = True
        else:
            weights_d += self.w - (c_param * x_i_class)
            predicted = False

        return weights_d, predicted

    def test(self):
        samples_classes = np.where(self.test_class_col == 0, -1, 1)
        correct_predictions = 0
        incorrect_predictions = 0

        for index, x_i in enumerate(self.test_data):
            predicted_class = self._predict(x_i)
            if predicted_class == samples_classes[index]:
                correct_predictions += 1
            else:
                incorrect_predictions += 1

        print('Test data set. incorrect predictions={}, correct predictions={}, {}% correct'
              .format(incorrect_predictions, correct_predictions, (correct_predictions / len(self.test_data)) * 100))

    def _wipe(self):
        self.w = np.zeros(self.num_features)

    def iterate_c_params(self):
        for c_param in SVM.c_params:
            self.fit_gd(c_param=c_param, print_epoch_result=False)
            self.test()

    def test_new(self, vectoriser: TfIdfVectoriser, text: string):
        vectorised_text = vectoriser.process_new(text)
        vectorised_text = np.append(vectorised_text, [1])
        return self._predict(vectorised_text)


# def svm_train_and_test(train_data: np.ndarray, test_data: np.ndarray, reg_params):
#
#     train_err = np.zeros(np.shape(reg_params))
#     test_err = np.zeros(np.shape(reg_params))
#
#     for i in range(len(reg_params)):
#         Ypredt, w_train = svmPredict(Xtr, Ytr, Xtr, reg_params[i])
#         train_err[i] = calcErrorSVM(Ypredt, Ytr)
#
#         Ypredtr, w_pred = svmPredict(Xtr, Ytr, Xte, reg_params[i])
#         test_err[i] = calcErrorSVM(Ypredtr, Yte)
#
#     return train_err, test_err

def split_dataframe(dataframe: pd.DataFrame, train_ratio: float):
    repr('Returns 1. train set, 2. test set')
    num_samples = dataframe.shape[0]
    num_train_samples = int(num_samples * train_ratio)
    return dataframe.iloc[:num_train_samples, :].copy(), dataframe.iloc[num_train_samples:, :].copy()


if __name__ == '__main__':
    root_path = 'C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\'
    df = pd.read_csv('data/train.csv', nrows=10000)
    train_df, test_df = split_dataframe(df, 0.8)
    tf_idf_vectoriser = TfIdfVectoriser()
    y_train, train_tf_idf_mat = tf_idf_vectoriser.fit_transform(train_df)
    y_test, test_tf_idf_mat = tf_idf_vectoriser.transform(test_df)


    svm = SVM(y_train, train_tf_idf_mat, y_test, test_tf_idf_mat)
    # svm.iterate_c_params()
    svm.fit_gd(c_param=0.1, print_epoch_result=False)
    # svm.iterate_c_params()
    print(svm.test_new(tf_idf_vectoriser, 'i will murder you! this is what happens when a bunch of fucking idiots get down to editing a highly specialised articleyou fucking cunt. this is what happens when a bunch of fucking idiots get down to editing a highly specialised article.'))
    print(svm.test_new(tf_idf_vectoriser, 'Perfect intro in tf-idf, thank you very much! Very interesting, I’ve wanted to study this field for a long time and you posts it is a real gift. It would be very interesting to read more about use-cases of the technique. And may be you’ll be interested, please, to shed some light on other methods of text corpus representation, if they exists?'))
