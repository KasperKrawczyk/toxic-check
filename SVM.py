import random
import string
import numpy as np
import pandas as pd
import sklearn.utils
import ETL
from TfIdfVectoriser import TfIdfVectoriser


def print_stats(epoch, false_neg, false_pos, true_neg, true_pos):
    correct_predictions = true_pos + true_neg
    incorrect_predictions = false_pos + false_neg
    per_cent_correct = (correct_predictions / (correct_predictions + incorrect_predictions)) * 100
    acc = ((true_pos + true_neg) / (correct_predictions + incorrect_predictions)) * 100
    print('Epoch={}, TP={}, TN={}, FP={}, FN={}, Acc={}'
          .format(epoch, true_pos, true_neg, false_pos, false_neg, acc))
    print('incorrect predictions={}, correct predictions={}, {}% correct'
          .format(incorrect_predictions, correct_predictions,
                  per_cent_correct))


class SVM:
    c_params = [10 ** p for p in range(-6, 1)]
    c_param_default = 10 ** -5

    def __init__(self,
                 train_data_class_col: np.ndarray,
                 input_train_data: np.ndarray,
                 test_data_class_col: np.ndarray,
                 input_test_data: np.ndarray,
                 learn_rate=0.1,
                 num_epochs=200,
                 train_ratio=0.8):

        self.num_train_samples = input_train_data.shape[0]
        self.num_test_samples = input_test_data.shape[0]
        self.train_data = input_train_data
        self.test_data = input_test_data
        # shape should return a 2-tuple
        _, self.num_features = self.train_data.shape
        self.train_class_col = train_data_class_col
        self.test_class_col = test_data_class_col
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.c = SVM.c_param_default
        self.b = 0

        self.train_ratio = train_ratio
        self.w = np.zeros(self.num_features)

    def _predict(self, x_i: np.ndarray):
        cosine = np.dot(x_i, self.w) - self.b
        return np.sign(cosine)

    def _train_predict(self, x_i_class: int, x_i: np.ndarray):
        cosine = np.dot(x_i, self.w) - self.b
        return cosine, (x_i_class * cosine >= 1)

    def fit_gd(self, c_param: float = c_param_default, print_epoch_result: bool = True, wipe: bool = True):
        if wipe:
            self._wipe()

        print("c param={}".format(c_param))
        samples_classes = np.where(self.train_class_col == 0, -1, 1)

        for epoch in range(1, self.num_epochs):
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            self.learn_rate = 1 / epoch
            for index, x_i in enumerate(self.train_data):
                x_i = np.array(x_i).ravel()
                cosine, is_correctly_predicted = self._train_predict(samples_classes[index], x_i)

                if is_correctly_predicted:
                    if samples_classes[index] == -1:
                        true_neg += 1
                    else:
                        true_pos += 1
                    self.w -= self.learn_rate * (c_param * self.w)
                else:
                    if samples_classes[index] == -1:
                        false_neg += 1
                    else:
                        false_pos += 1
                    self.w -= self.learn_rate * (c_param * self.w - np.dot(x_i, samples_classes[index]))
                    self.b -= self.learn_rate * samples_classes[index]

            if print_epoch_result and epoch % 10 == 0:
                print_stats(epoch, false_neg, false_pos, true_neg, true_pos)

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
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0

        for i, x_i in enumerate(self.test_data):
            predicted_class = self._predict(x_i)
            if predicted_class == samples_classes[i]:
                if samples_classes[i] == -1:
                    true_neg += 1
                else:
                    true_pos += 1
            else:
                if samples_classes[i] == -1:
                    false_neg += 1
                else:
                    false_pos += 1

        print_stats("TEST", false_neg, false_pos, true_neg, true_pos)

    def _wipe(self):
        self.w = np.zeros(self.num_features)
        self.b = 0

    def iterate_c_params(self):
        for c_param in SVM.c_params:
            self.fit_gd(c_param=c_param, print_epoch_result=True)
            self.test()

    def test_new(self, v: TfIdfVectoriser, raw_text: string):
        vectorised_text = v.process_new(raw_text)
        return self._predict(vectorised_text)



def split_dataframe(dataframe: pd.DataFrame, train_ratio: float):
    repr('Returns 1. train set, 2. test set')
    num_samples = dataframe.shape[0]
    num_train_samples = int(num_samples * train_ratio)
    return dataframe.iloc[:num_train_samples, :].copy(), dataframe.iloc[num_train_samples:, :].copy()


if __name__ == '__main__':
    root_path = 'C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\'
    df = pd.read_csv('data/train.csv', header=0, skiprows=lambda i: i > 0 and random.random() > 0.075)
    train_df, test_df = split_dataframe(df, 0.8)
    train_df = ETL.clean_dataset(train_df)
    test_df = ETL.clean_dataset(test_df)

    tf_idf_vectorizer = TfIdfVectoriser(min_stem_occurrence=1)
    y_train, train_tf_idf = tf_idf_vectorizer.fit_transform(train_df)
    y_test, test_tf_idf = tf_idf_vectorizer.transform(test_df)

    np.savetxt('train_tf_idf_lib.csv', train_tf_idf, delimiter=',')

    svm = SVM(y_train, train_tf_idf, y_test, test_tf_idf)
    # svm.iterate_c_params()
    svm.fit_gd(c_param=0.00001, print_epoch_result=True)
    svm.test()

    while True:
        text = input(">>> ")
        print(svm.test_new(tf_idf_vectorizer, text))
