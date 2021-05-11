import numpy as np
import sys
import math
import random
from sklearn.preprocessing import normalize
from scipy.stats import norm

from baselines.CGMM.gaussian_mixture import Gaussian
from baselines.CGMM.softmax_gaussian_mixture import SoftmaxGaussianMixture

from baselines.CGMM.dataset_processor import Processor

epsilon = sys.float_info.epsilon
# epsilon = 1e-15

class SoftmaxCensoredGaussianMixture(SoftmaxGaussianMixture):
    def __init__(self, b_train_origin, feature_dimension, label_dimension, min_z, max_z, mean=[1, 5], variance=[5, 10]):
        self.b_train_origin = b_train_origin
        # self.multinoulli is duplicated, since we use softmax to handle the posterior
        SoftmaxGaussianMixture.__init__(self, feature_dimension, label_dimension, min_z, max_z, mean, variance)
        # self.softmax = SoftmaxClassifier(feature_dimension=feature_dimension, label_dimension=label_dimension)

    def _censor_data(self, z, x, mask):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns z_win, b_lose, x_win, x_lose

        # (m, ) = np.shape(mask)
        m = mask.sum()

        bids = self.b_train_origin[mask].reshape((m, 1))
        win = bids > z
        lose = bids <= z

        m_win = len(z[win])
        m_lose = len(bids[lose])

        return z[win].reshape((m_win, 1)),\
               bids[lose].reshape((m_lose, 1)),\
               x[win.reshape((m,)), :],\
               x[lose.reshape((m,)), :]

    def e_step(self, z_win, b_lose, x_win, x_lose):
        # z_win is a m_win x 1 matrix, each row represents a market price observed in winning case
        # b_lose is a m_lose x 1 matrix, each row represents a bid price observed in losing case
        # x_win is a m_win x feature_dimension matrix, each row represents a winning bid request
        # x_lose is a m_lose x feature_dimension matrix, each row represents a losing bid request
        # returns responsibilities, i.e., posterior probability, for each bid request
        # for winning case, the responsibilities are p(h|z,x)
        # for losing case, the responsibilities are pr(h|z>b,x)

        (m_win, _) = np.shape(z_win)
        (m_lose, _) = np.shape(b_lose)

        likelihood_win = np.zeros(shape=(m_win, self.num))
        likelihood_lose = np.zeros(shape=(m_lose, self.num))
        likelihood_lose_softmax = np.zeros(shape=(m_lose, self.num))

        # likelihood for each record
        for i in range(self.num):
            gaussian = Gaussian(self.mean[i], np.sqrt(self.variance[i]))
            likelihood_win[:, i] = gaussian.pdf(z_win[:, 0]) + 1e-4
            likelihood_lose[:, i] = 1 - gaussian.cdf(b_lose[:, 0])
            likelihood_lose_softmax[:, i] = gaussian.pdf(b_lose[:, 0])

        # element-wise multiplication and normalize probability
        return normalize(np.multiply(self.softmax.predict(x_win), likelihood_win), norm='l1', axis=1), \
            normalize(np.multiply(self.softmax.predict(x_lose), likelihood_lose), norm='l1', axis=1), \
            normalize(np.multiply(self.softmax.predict(x_lose), likelihood_lose_softmax), norm='l1', axis=1),

    def _show_likelihood(self, z_win, b_lose, x_win, x_lose):
        # calculating likelihood of observing winning cases and losing cases
        (m_win, _) = np.shape(z_win)
        (m_lose, _) = np.shape(b_lose)
        likelihood_win = []
        likelihood_lose = []
        h_win = self.softmax.predict(x_win)  # m_win * self.num
        h_lose = self.softmax.predict(x_lose)  # m_lose * self.num

        for i in range(self.num):
            gaussian = Gaussian(self.mean[i], np.sqrt(self.variance[i]))
            likelihood_win.append(np.dot(h_win[:, i], gaussian.pdf(z_win[:, 0])) / m_win)
            likelihood_lose.append(np.dot(h_lose[:, i], (1 - gaussian.cdf(b_lose[:, 0]))) / m_lose)
        print("mix proportions for winning cases:", np.sum(h_win, axis=0) / m_win)
        print("mix proportions for losing cases:", np.sum(h_lose, axis=0) / m_lose)
        print("total likelihood: ", sum(likelihood_win) + sum(likelihood_lose))
        print("likelihood win: ", likelihood_win, "sum: ", sum(likelihood_win))
        print("likelihood lose: ", likelihood_lose, "sum: ", sum(likelihood_lose))

        return

    def m_step(self, z_win, b_lose, x_win, x_lose, rs_win, rs_lose, lose_pi, batch_size=512, eta_w=2e-2, eta_mean=1e0, eta_variance=5e0, labda=0.0, verbose=1):
        # z_win is a m_win x 1 matrix, each row represents a market price observed in winning case
        # b_lose is a m_lose x 1 matrix, each row represents a bid price observed in losing case
        # x_win is a m_win x feature_dimension matrix, each row represents a winning bid request
        # x_lose is a m_lose x feature_dimension matrix, each row represents a losing bid request
        # rs_win is a m_win x label_dimension matrix, each row represents the posterior for this winning bid request
        # rs_lose is a m_lose x label_dimension matrix, each row represents the posterior for this losing bid request
        # returns nothing, but update model's parameters in m-step

        # print("m-step start:")
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
        # if not make copies, the losing cases are much more than winning cases,
        # the softmax tend to classify training samples to the gaussian distribution with largest mean
        copies = round(len(loses)/len(wins))
        copies = 1 if copies < 1 else copies
        # print("winning cases are copied {0:d} times to balance the winning cases and losing cases".format(copies))
        index.extend(list(zip(starts_win, ends_win, wins))*copies)
        index.extend(list(zip(starts_lose, ends_lose, loses)))
        random.shuffle(index)

        if verbose == 1:
            self._show_likelihood(z_win, b_lose, x_win, x_lose)
            print(self)

        for start, end, win_lose in index:
            if win_lose == 1:
                # mini-batch update for winning cases

                # update softmax's parameters
                softmax_error = rs_win[start:end, :] - self.softmax.predict(x_win[start:end, :])
                w_derivative = softmax_error.transpose() @ x_win[start:end, :]
                self.softmax.w = self.softmax.w + eta_w / (end - start) * (w_derivative - labda * self.softmax.w)

                # update gaussian mixture's parameters
                for i in range(self.num):
                    mean_derivative = (rs_win[start:end, i] * (z_win[start:end, 0] - self.mean[i]) / self.variance[
                        i]).sum() / (end - start)
                    self.mean[i] = self.mean[i] + eta_mean * mean_derivative
                    variance_derivative = (rs_win[start:end, i] * (
                            (z_win[start:end, 0] - self.mean[i]) ** 2 / (2 * self.variance[i] ** 2) - 1 / (
                                2 * self.variance[i]))
                                           ).sum() / (end - start)
                    self.variance[i] = self.variance[i] + eta_variance * variance_derivative
                    self.variance[i] = 100 if self.variance[i] <= 0 else self.variance[i]

                    if verbose == 1:
                        # print("win for {0:d}-th gaussian distribution, mean_derivative:{1:.5f}, variance_derivative:{2:.5f}"
                        #       .format(i, mean_derivative, variance_derivative))
                        pass

            elif win_lose == 2:
                # mini-batch update for losing cases
                # update softmax's parameters
                softmax_error = rs_lose[start:end, :] - self.softmax.predict(x_lose[start:end, :])
                w_derivative = softmax_error.transpose() @ x_lose[start:end, :]
                self.softmax.w = self.softmax.w + eta_w / (end - start) * (w_derivative - labda * self.softmax.w)

                # update gaussian mixture's parameters
                for i in range(self.num):
                    sigma = math.sqrt(self.variance[i])
                    p = (norm.pdf(b_lose[start:end, 0], loc=self.mean[i], scale=sigma) + epsilon*10) \
                        / (1 - norm.cdf(b_lose[start:end, 0], loc=self.mean[i], scale=sigma) + epsilon)
                    # print(p)
                    # print(np.max(b_lose))
                    # TODO: WHY mean_derivative is always greater than 0?
                    # TODO: Because losing cases attempt to increase the mean so that the likelihood can be improved
                    mean_derivative = (rs_lose[start:end, i] / sigma * p).sum() / (end - start)
                    self.mean[i] = self.mean[i] + eta_mean * mean_derivative
                    variance_derivative = (rs_lose[start:end, i] * (b_lose[start:end, 0] - self.mean[i])
                                           / 2 / (sigma**3) * p).sum() / (end - start)
                    self.variance[i] = self.variance[i] + eta_variance * variance_derivative
                    self.variance[i] = 100 if self.variance[i] <= 0 else self.variance[i]

                    if verbose == 1:
                        # print("lose for {0:d}-th gaussian distribution, mean_derivative:{1:.5f}, variance_derivative:{2:.5f}"
                        #       .format(i, mean_derivative, variance_derivative))
                        pass
            else:
                print("unknown type: {0:d}".format(win_lose))
                continue

    def fit(self, z, x, test_z, test_x, sample_rate=1.0, epoch=10, batch_size=512, eta_w=2e-2, eta_mean=1e0, eta_variance=5e0, labda=0.0, verbose=1):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request

        (m, _) = np.shape(z)
        (m_test, _) = np.shape(test_z)
        mask = np.random.choice([False, True], m, p=[1 - sample_rate, sample_rate])
        mask_test = np.random.choice([False, True], m_test, p=[0.9, 0.1])

        # self.test_pay = Processor(min_price=self.min_z, max_price=self.max_z)
        self.test_pay.load_by_array(test_z)

        z_win, b_lose, x_win, x_lose = self._censor_data(z[mask, :], x[mask, :], mask)

        if verbose == 1:
            print("now begin to fit, hyper-parameters: epoch:{0:d}, batch_size:{1:d}, eta_w:{2:.3f}, eta_mean:{3:.3f}, eta_variance:{4:.3f}, lambda:{5:.3f}"
                  .format(epoch, batch_size, eta_w, eta_mean, eta_variance, labda))
            print("{0:d} records have been sampled".format(z[mask, :].shape[0]))
            print("z_win's shape: ", z_win.shape,
                  "b_lose's shape: ", b_lose.shape,
                  "x_win's shape: ", x_win.shape,
                  "x_lose's shape: ", x_lose.shape,)

        for i in range(epoch):
            rs_win, rs_lose, lose_pi = self.e_step(z_win, b_lose, x_win, x_lose)
            if verbose == 1:
                print("============== E-M epoch: {} ==============".format(str(i)))
                # print(z_win[0], rs_win[0, :])
                # print(b_lose[0], rs_lose[0, :])
                # print(b_lose[0], lose_pi[0, :])
                # print("rs_win's shape: ", rs_win.shape,
                #       "rs_lose's shape: ", rs_lose.shape)

            self.m_step(z_win, b_lose, x_win, x_lose, rs_win, rs_lose, lose_pi, batch_size=batch_size,
                        eta_w=eta_w/math.sqrt(i+1), eta_mean=eta_mean/math.sqrt(i+1), eta_variance=eta_variance/math.sqrt(i+1),
                        labda=labda, verbose=verbose)

            self.evaluate(test_z[mask_test, :], test_x[mask_test, :])

        if verbose == 1:
            self._show_likelihood(z_win, b_lose, x_win, x_lose)
            print(self)
