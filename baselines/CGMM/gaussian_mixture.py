import numpy as np
import sys
import math
import random
from sklearn.preprocessing import normalize
from scipy.stats import norm

np.set_printoptions(precision=2)
epsilon = sys.float_info.epsilon

class Gaussian:
    "Model univariate Gaussian"

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def pdf(self, datum):
        "Probability of a data point given the current parameters"
        z_score = (datum - self.mu) / self.sigma
        y = (2 * np.pi * self.sigma ** 2) ** (-1 / 2) * np.exp(-1 / 2 * z_score ** 2)
        return y+epsilon

    def cdf(self, datum):
        # z_score = (datum - self.mu) / self.sigma
        # return 1/2 * (1 + math.erf(z_score/math.sqrt(2)))
        return norm.cdf(datum, loc=self.mu, scale=self.sigma)

    def __repr__(self):
        return 'Gaussian({}, {})'.format(self.mu, self.sigma)


class GaussianMixture:
    def __init__(self, num=2, multinoulli=[0.8, 0.2], mean=[1, 5], variance=[5, 10]):
        self.num = num
        self.multinoulli = [m/sum(multinoulli) for m in multinoulli]
        self.mean = mean
        self.variance = variance
        print("Initialize GaussianMixture(mix={}, mean={}, variance={})".format(self.multinoulli, self.mean,
                                                                                self.variance))

    def __repr__(self):
        return "GaussianMixture(mix=[" + ','.join(["{0:.3f}".format(i) for i in self.multinoulli]) + \
               "], mean=[" + ', '.join(["{0:.3f}".format(i) for i in self.mean]) + \
               "], variance=[" + ', '.join(["{0:.3f}".format(i) for i in self.variance]) + \
               "])"

    def e_step(self, datum):
        data = np.array(datum).reshape((len(datum), 1))
        rs = np.zeros(shape=(len(datum), self.num))
        for i in range(self.num):
            gaussian = Gaussian(self.mean[i], np.sqrt(self.variance[i]))
            priori = self.multinoulli[i]
            likelihood = gaussian.pdf(data)
            posterior = priori * likelihood
            rs[:, i] = posterior[:, 0]
        # normalize probability
        rs = normalize(rs, norm='l1', axis=1)
        return rs

    def m_step(self, datum, rs, verbose=1):
        data = np.array(datum).reshape((len(datum), 1))
        rs_sum = rs.sum(axis=0)
        for i in range(self.num):
            # update parameter
            self.multinoulli[i] = rs_sum[i] / len(datum)
            self.variance[i] = (rs[:, i] * (data[:, 0] - self.mean[i]) ** 2).sum() / rs_sum[i]
            self.mean[i] = (rs[:, i] * data[:, 0]).sum() / rs_sum[i]

            if self.variance[i] < epsilon:
                # To fix singularity problem, i.e. variance = 0
                print("Singularity problem encountered: mix index:{0:d}, mean:{1:.5f}, variance:{2:.5f}".format(i, self.mean[i], self.variance[i]))
                self.variance[i] = random.randint(10, 100)
                self.mean[i] = random.randint(10, 100)

        if verbose == 1:
            print(self)

    def fit(self, datum, epoch=10, verbose=1):
        for i in range(epoch):
            if verbose == 1:
                print("epoch: {}".format(str(i)))
            rs = self.e_step(datum)
            self.m_step(datum, rs, verbose)

    def pdf(self, datum):
        data = np.array(datum).reshape((len(datum), 1))
        gaussian_pdf = np.zeros(shape=(len(datum), self.num))
        for i in range(self.num):
            gaussian = Gaussian(self.mean[i], np.sqrt(self.variance[i]))
            gaussian_pdf[:, i] = gaussian.pdf(data)[:, 0] * self.multinoulli[i]
        return gaussian_pdf.sum(axis=1)

    def cdf(self, datum):
        data = np.array(datum).reshape((len(datum), 1))
        gaussian_cdf = np.zeros(shape=(len(datum), self.num))
        for i in range(self.num):
            gaussian = Gaussian(self.mean[i], np.sqrt(self.variance[i]))
            gaussian_cdf[:, i] = gaussian.cdf(data)[:, 0] * self.multinoulli[i]
        return gaussian_cdf.sum(axis=1)

    def predict_z(self, min_z=1, max_z=300):
        # x is a m x feature_dimension matrix, each row represents a bid request
        # min_z and max_z are the lower bound and upper bound to integrate
        # returns m x 1 matrix, each row is the predicted market price

        zs = np.asmatrix(np.arange(min_z, max_z+1, 1)).transpose()

        return self.pdf(zs) @ zs

    def ANLP(self, z):
        # z is a m x 1 matrix, each row represents a market price
        # returns a scala, which is average negative log probability (ANLP)

        (m, _) = np.shape(z)
        nlp = np.zeros(shape=(m, 1))

        for i in range(m):
            nlp[i, 0] = -np.log(1e-8 + self.pdf([z[i, 0]]))

        return np.sum(nlp) / m
