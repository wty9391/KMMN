import numpy as np
import sys
import math
import random
from sklearn.preprocessing import normalize
from scipy.stats import norm


class SoftmaxClassifier:
    def __init__(self, feature_dimension, label_dimension=2):
        self.feature_dimension = feature_dimension
        self.label_dimension = label_dimension
        # for each type of label, we have a parameter vector
        self.w = np.zeros(shape=(label_dimension, feature_dimension))
        # self.w = np.random.normal(loc=0, scale=1, size=(label_dimension, feature_dimension))

        print("Initialize SoftmaxClassifier(label_dimension={0:d}, feature_dimension={1:d})"
              .format(self.label_dimension, self.feature_dimension))

    def _validate_label_dimension(self, datum):
        # datum is a m x label_dimension matrix, each row represents a record
        (_, label_dimension) = np.shape(datum)
        assert label_dimension == self.label_dimension, \
            "training sample label's dimension must be equal to {0:d}".format(self.label_dimension)

    def _validate_feature_dimension(self, datum):
        # datum is a m x feature_dimension matrix, each row represents a record
        (_, feature_dimension) = np.shape(datum)
        assert feature_dimension == self.feature_dimension, \
            "training sample feature's dimension must be equal to {0:d}".format(self.feature_dimension)

    def _unnormorlized_probability(self, datum):
        # datum is a m x feature_dimension matrix, each row represents a record
        # returns m x label_dimension matrix, each row represents a record
        # self._validate_feature_dimension(datum)
        return datum.dot(self.w.transpose())
        # return np.matmul(datum, self.w.transpose())

    def _softmax(self, datum):
        # datum is a m x label_dimension matrix, each row represents a record
        # for sake of numerical stability
        # e ^ (x - max(x)) / sum(e^(x - max(x))
        # datum = np.array(datum)
        # self._validate_label_dimension(datum)
        max_x = np.max(datum, axis=1)
        # only if the trailing axes have the same dimension, we can use subtraction/division operator
        # i.e., we can subtract/divide every row of matrix by vector
        datum = (datum.transpose() - max_x).transpose()
        e_x = np.exp(datum)
        return (e_x.transpose() / e_x.sum(axis=1)).transpose()

    def predict(self, datum):
        # datum is a m x feature_dimension matrix, each row represents a record
        # datum = np.array(datum)
        # self._validate_feature_dimension(datum)
        return self._softmax(self._unnormorlized_probability(datum))

    def fit(self, label, feature, batch_size=256, epoch=100, eta=1e-3, labda=0.0, verbose=1):
        # label is a m x label_dimension matrix, each row represents a record
        # feature is a m x feature_dimension matrix, each row represents a record
        # label = np.array(label)
        # feature = np.array(feature)
        self._validate_label_dimension(label)
        self._validate_feature_dimension(feature)

        (m, _) = np.shape(label)

        for i in range(epoch):
            starts = [i * batch_size for i in range(int(math.ceil(m / batch_size)))]
            ends = [i * batch_size for i in range(1, int(math.ceil(m / batch_size)))]
            ends.append(m)

            # mini-batch update
            for start, end in zip(starts, ends):
                prediction_error = label[start:end, :] - self.predict(feature[start:end, :])
                w_derivative = prediction_error.transpose() @ feature[start:end, :]
                self.w = self.w + eta / (end - start) * w_derivative - labda * self.w

            if verbose == 1:
                print("softmax epoch: {}".format(str(i)))
                print("i-th label \t log likelihood")
                label_predicted = self.predict(feature)
                log_likelihood = []
                for i in range(self.label_dimension):
                    log_likelihood.append(np.dot(label[:, i], np.log(label_predicted[:, i])) + \
                                      np.dot((1 - label[:, i]), np.log(1 - label_predicted[:, i])))
                    print("{0:d}\t{1:.5f}".format(i, log_likelihood[i]/m))
                print("The overall log likelihood: {0:.5f}".format(sum(log_likelihood)/m))


if __name__ == '__main__':
    sample_size = 100
    feature_size = 10000
    lable_size = 3
    test = SoftmaxClassifier(feature_dimension=feature_size, label_dimension=lable_size)
    # [ 0.8360188   0.11314284  0.05083836]
    # print(test._softmax([[3.0, 1.0, 0.2], [3.0, 1.0, 0.2]]))

    feature = np.random.rand(sample_size, feature_size)
    label = np.random.rand(sample_size, lable_size)

    test.fit(label, feature, epoch=100, eta=1e-3, labda=0.0)

