import numpy as np
import pandas as pd
import sklearn.utils

from TfIdfVectoriser import TfIdfVectoriser


class SVM:
    c_params = [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
    c_param_default = 100000

    def __init__(self,
                 input_train_data: np.ndarray,
                 input_test_data: np.ndarray,
                 class_col_ind: int,
                 learn_rate=0.1,
                 num_epochs=100,
                 train_ratio=0.8):
        np.random.shuffle(input_train_data)
        np.random.shuffle(input_test_data)
        self.train_data = input_train_data
        self.test_data = input_test_data
        self.num_train_samples = self.train_data.shape[0]
        self.num_test_samples = self.test_data.shape[0]

        # slice input so that data doesn't include the classification column
        # self.train_data = train_data[:, class_col_ind + 1:]
        # add the intercept term (bias) as last column, filled with 1s
        self.train_data = np.c_[self.train_data, np.ones(self.train_data.shape[0])]
        self.test_data = np.c_[self.test_data, np.ones(self.test_data.shape[0])]
        # shape should return a 2-tuple
        self.num_train_samples, self.num_features = self.train_data.shape
        self.num_test_samples, _ = self.test_data.shape
        self.class_col_ind = class_col_ind
        self.train_class_col = self.train_data[:, class_col_ind]
        self.test_class_col = self.test_data[:, class_col_ind]
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.c = SVM.c_param_default

        self.train_ratio = train_ratio
        self.w = np.zeros(self.num_features)
        # self._split_data()

    # def _split_data(self):
    #     self.num_train_samples = int(self.num_train_samples * self.train_ratio)
    #     self.num_test_samples = int(self.num_train_samples * (1 - self.train_ratio))
    #     self.train_data = self.data[:self.num_train_samples]
    #     self.test_data = self.data[self.num_train_samples:]
    #     self.class_col_train = self.train_class_col[:self.num_train_samples]
    #     self.class_col_test = self.train_class_col[self.num_train_samples:]

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
            self.train_data, samples_classes = sklearn.utils.shuffle(self.train_data, samples_classes, random_state=0)
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

            if print_epoch_result:
                print('Epoch={}, incorrect predictions={}, correct predictions={}'
                      .format(epoch, incorrect_predictions_per_epoch, correct_predictions_per_epoch))

    def fit_sgd(self, c_param: float = c_param_default, print_epoch_result: bool = True, wipe: bool = True):
        if wipe:
            self._wipe()

        samples_classes = np.where(self.train_class_col == 0, -1, 1)

        for epoch in range(1, self.num_epochs):
            self.train_data, samples_classes = sklearn\
                .utils\
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

            if print_epoch_result:
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
            print('c param = {}'.format(c_param))
            self.fit_sgd(c_param=c_param, print_epoch_result=True)
            self.test()


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
    return dataframe.iloc[:num_train_samples, :], dataframe.iloc[num_train_samples:, :]


if __name__ == '__main__':
    root_path = 'C:\\Users\\kaspe\\OneDrive\\Pulpit\\test\\'
    df = pd.read_csv('data/train.csv', nrows=10000)
    train_df, test_df = split_dataframe(df, 0.8)
    tf_idf_vectoriser = TfIdfVectoriser()
    train_tf_idf_mat, _, _ = tf_idf_vectoriser.fit_transform(train_df)
    test_tf_idf_mat, _, _ = tf_idf_vectoriser.transform(test_df)


    # ETL.process_dataset(
    #     'data/train.csv',
    #     root_path + 'clean_full.csv',
    #     root_path + 'vectorised_matrix.npy',
    #     root_path + 'vocabulary_full.csv',
    #     3,
    #     10000,
    #     limit_nrows=True)
    # data = np.load('C:\\Users\\kaspe\\OneDrive\\Pulpit\\full\\vectorised_matrix.npy')
    svm = SVM(train_tf_idf_mat, test_tf_idf_mat, 0)
    svm.iterate_c_params()
