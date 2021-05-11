import numpy as np
import sklearn.metrics
from sklearn import linear_model

class Truthful_bidder():
    def __init__(self, loss='log', penalty='l2', alpha=1e-6, verbose=1, n_jobs=4, max_iter=100):
        self.model = linear_model.SGDClassifier(loss, penalty, alpha, verbose, n_jobs, max_iter)
        self.alpha = 0

    def fit(self, x, y, z):
        # X is a m x feature_dimension matrix, each row represents a bid request
        # Y is a m x 1 matrix
        # Z is a m x 1 matrix

        self.model.fit(x, y.ravel())
        self.alpha = np.sum(z, axis=0) / np.sum(y, axis=0)

        return self.model, self.alpha

    def bid(self, x):
        return np.ceil(self.model.predict_proba(x)[:, 1] * self.alpha)

    def evaluate(self, x, y):
        auc = sklearn.metrics.roc_auc_score(y, self.model.predict_proba(x)[:, 1])
        print('Evaluation: classify auc is: {}%'.format(auc * 100))
        return auc


