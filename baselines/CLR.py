import numpy as np
import sys
import pickle
import math
import random
from sklearn.preprocessing import normalize
from scipy.stats import norm

import sklearn.metrics
from sklearn import linear_model

epsilon = sys.float_info.epsilon

class MixtureModel:
    def __init__(self, feature_dimension, b_train, variance=1):
        self.feature_dimension = feature_dimension
        self.b_train = b_train
        self.winner = WinningPredictor()
        self.w_lm = np.zeros(shape=(1, feature_dimension))
        self.w_clm = np.zeros(shape=(1, feature_dimension))
        self.variance = variance

    def _censor_data(self, z, x):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns z_win, b_lose, x_win, x_lose

        (m, _) = np.shape(z)
        bids = self.b_train
        win = bids > z
        lose = bids <= z

        m_win = len(z[win])
        m_lose = len(bids[lose])

        return z[win].reshape((m_win, 1)), \
               bids[lose].reshape((m_lose, 1)), \
               x[win.reshape((m,)), :], \
               x[lose.reshape((m,)), :]

    def fit(self, z, x, epoch=10, batch_size=512, eta_w=1e-1, verbose=1):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        (record_size, feature_size) = np.shape(x)
        z_win, b_lose, x_win, x_lose = self._censor_data(z, x)

        if verbose == 1:
            print("now begin to fit winning rate predictor")

        bids = self.b_train
        y = np.zeros((record_size, 1))
        y[bids > z] = 1
        self.winner = WinningPredictor()
        self.winner.fit(x, y)
        self.winner.evaluate(x, y)

        if verbose == 1:
            print("now begin to fit mixture model, hyper-parameters: epoch:{0:d}, batch_size:{1:d}, eta:{2:.3f}"
                  .format(epoch, batch_size, eta_w))
            print("z_win's shape: ", z_win.shape,
                  "b_lose's shape: ", b_lose.shape,
                  "x_win's shape: ", x_win.shape,
                  "x_lose's shape: ", x_lose.shape,)

        for i in range(epoch):
            if verbose == 1:
                print("=== E-M epoch: {} ===".format(str(i)))
                error_lm = z - self.lm_predict(x)
                error_clm = z - self.clm_predict(x)
                error_mm = z - self.predict(x)

                print("MSE for lm:{0:.3f} clm:{1:.3f} mm:{2:.3f}".format(
                    (error_lm.transpose() @ error_lm)[0, 0] / record_size,
                    (error_clm.transpose() @ error_clm)[0, 0] / record_size,
                    (error_mm.transpose() @ error_mm)[0, 0] / record_size
                ))

            eta = eta_w / np.sqrt(i+1)
            (m_win, _) = np.shape(z_win)
            (m_lose, _) = np.shape(b_lose)

            # mini-batch gradient ascend
            starts_win = [i * batch_size for i in range(int(math.ceil(m_win / batch_size)))]
            ends_win = [i * batch_size for i in range(1, int(math.ceil(m_win / batch_size)))]
            ends_win.append(m_win)
            wins = [1 for i in starts_win]

            starts_lose = [i * batch_size for i in range(int(math.ceil(m_lose / batch_size)))]
            ends_lose = [i * batch_size for i in range(1, int(math.ceil(m_lose / batch_size)))]
            ends_lose.append(m_lose)
            loses = [2 for i in starts_lose]

            index = []
            index.extend(list(zip(starts_win, ends_win, wins)))
            index.extend(list(zip(starts_lose, ends_lose, loses)))
            random.shuffle(index)

            for start, end, win_lose in index:
                if win_lose == 1:
                    # mini-batch update for winning cases
                    error_lm = z_win[start:end, :] - self.lm_predict(x_win[start:end, :])
                    self.w_lm = self.w_lm + eta / (end - start) * error_lm.transpose() @ x_win[start:end, :]
                    error_clm = z_win[start:end, :] - self.clm_predict(x_win[start:end, :])
                    self.w_clm = self.w_clm + eta / (end - start) * error_clm.transpose() @ x_win[start:end, :]
                elif win_lose == 2:
                    # mini-batch update for losing cases
                    error = (self.clm_predict(x_lose[start:end, :]) - b_lose[start:end, :]) / self.variance
                    p = (norm.pdf(error, loc=0, scale=np.sqrt(self.variance))+epsilon) / (norm.cdf(error, loc=0, scale=np.sqrt(self.variance))+epsilon)
                    self.w_clm = self.w_clm + eta*1000 / (end - start) / np.sqrt(self.variance)\
                        * (p.transpose() @ x_lose[start:end, :])
                else:
                    print("unknown type: {0:d}".format(win_lose))
                    continue

    def lm_predict(self, x):
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns zs: m x 1 matrix
        return x @ self.w_lm.transpose()

    def clm_predict(self, x):
        return x @ self.w_clm.transpose()

    def predict(self, x):
        (m, _) = np.shape(x)
        win = self.winner.win(x)
        lose = 1 - win
        return (win * self.lm_predict(x)[:, 0] + lose * self.clm_predict(x)[:, 0]).reshape((m, 1))

    def anlp(self, x, z):
        m, _ = x.shape
        win = self.winner.win(x)
        lose = 1 - win

        return -np.sum(win*np.log(epsilon+norm.pdf(z-self.predict(x), loc=0, scale=np.sqrt(self.variance)))[:, 0] +
                       lose*np.log(epsilon+norm.cdf(self.predict(x)-z, loc=0, scale=np.sqrt(self.variance)))[:, 0]) / m


class WinningPredictor:
    def __init__(self, loss='log', penalty='l2', alpha=1e-6, verbose=1, n_jobs=4, max_iter=150):
        self.model = linear_model.SGDClassifier(loss, penalty, alpha, verbose, n_jobs, max_iter)

    def fit(self, x, y):
        # X is a m x feature_dimension matrix, each row represents a bid request
        # Y is a m x 1 matrix

        self.model.fit(x, y.ravel())

        return self.model

    def win(self, x):
        return self.model.predict_proba(x)[:, 1]

    def evaluate(self, x, y):
        auc = sklearn.metrics.roc_auc_score(y, self.model.predict_proba(x)[:, 1])
        print('Evaluation: classify auc is: {}%'.format(auc * 100))
        return auc

    def save(self, path):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path):
        self.model = pickle.load(path, 'rb')
