import numpy as np
import sys
import random
from sklearn.preprocessing import normalize
from scipy.stats import entropy, wasserstein_distance
from sklearn.metrics import mean_squared_error

from baselines.CGMM.gaussian_mixture import Gaussian, GaussianMixture
from baselines.CGMM.softmax import SoftmaxClassifier

from baselines.CGMM.dataset_processor import Processor
import baselines.CGMM.init_util as init_util

epsilon = sys.float_info.epsilon

class SoftmaxGaussianMixture(GaussianMixture):
    def __init__(self, feature_dimension, label_dimension, min_z, max_z, mean=[1, 5], variance=[5, 10]):
        # self.multinoulli is duplicated, since we use softmax to handle the posterior
        GaussianMixture.__init__(self, label_dimension, [], mean, variance)
        self.softmax = SoftmaxClassifier(feature_dimension=feature_dimension, label_dimension=label_dimension)
        self.min_z = min_z
        self.max_z = max_z
        self.test_pay = Processor(min_price=self.min_z, max_price=self.max_z)

    def e_step(self, z, x):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns responsibilities, i.e., posterior probability, for each bid request
        (m, _) = np.shape(z)
        likelihood = np.zeros(shape=(m, self.num))

        # likelihood for each market price
        for i in range(self.num):
            gaussian = Gaussian(self.mean[i], np.sqrt(self.variance[i]))
            likelihood[:, i] = gaussian.pdf(z[:, 0])

        # element-wise multiplication and normalize probability
        return normalize(np.multiply(self.softmax.predict(x), likelihood), norm='l1', axis=1)

    def m_step(self, z, x, rs, batch_size=512, epoch=10, eta=2e-2, labda=0.0, verbose=1):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # rs is a m x label_dimension matrix, each row represents the posterior for this bid request
        # returns nothing, but update model's parameters in m-step

        # update softmax
        self.softmax.fit(rs, x, batch_size=batch_size, epoch=epoch, eta=eta, labda=labda, verbose=verbose)

        rs_sum = rs.sum(axis=0)

        for i in range(self.num):
            # update variance and mean
            self.variance[i] = (rs[:, i] * (z[:, 0] - self.mean[i]) ** 2).sum() / rs_sum[i]
            self.mean[i] = (rs[:, i] * z[:, 0]).sum() / rs_sum[i]

            if self.variance[i] < epsilon:
                # To fix singularity problem, i.e. variance = 0
                print("Singularity problem encountered: mix index:{0:d}, mean:{1:.5f}, variance:{2:.5f}".format(i, self.mean[i], self.variance[i]))
                self.variance[i] = random.randint(10, 100)
                self.mean[i] = random.randint(10, 100)

        if verbose == 1:
            print(self)

    def fit(self, z, x, test_z, test_x, sample_rate=1.0, epoch=10, softmax_epoch=10, batch_size=512, eta=2e-2, labda=0.0, evaluate=0, verbose=1):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        (m, _) = np.shape(z)
        (m_test, _) = np.shape(test_z)
        mask = np.random.choice([False, True], m, p=[1-sample_rate, sample_rate])
        mask_test = np.random.choice([False, True], m_test, p=[0.9, 0.1])

        print("Start to fit, sample_rate:{0:.2f}, epoch:{1:d}, softmax_epoch:{2:d}, "
              "batch_size:{3:d}, eta:{4:.4f}, labda:{5:.4f}".format(
            sample_rate, epoch, softmax_epoch, batch_size, eta, labda
        ))
        self.test_pay.load_by_array(test_z)

        for i in range(epoch):
            if verbose == 1:
                print("============== E-M epoch: {} ==============".format(str(i)))
                print("{0:d} records have been sampled".format(z[mask, :].shape[0]))

            self.m_step(z[mask, :], x[mask, :], self.e_step(z[mask, :], x[mask, :]),
                        batch_size=batch_size, epoch=softmax_epoch, eta=eta/np.sqrt(i+1),
                        labda=labda, verbose=verbose)

            self.evaluate(test_z[mask_test, :], test_x[mask_test, :])

            # trick
            if i < epoch-1:
                self.variance = [v/5 for v in self.variance]

    def pdf_overall(self, z, x):
        # z is n x 1 matrix, each row represents a market price
        # x is m x feature_dimension matrix, each row represents a bid request to be margin out
        # returns n x 1 matrix, each row is the probability to be taken by the corresponding z
        (m, _) = np.shape(x)
        self.multinoulli = (self.softmax.predict(x).sum(axis=0) / m).tolist()
        # print(self)

        return super().pdf(z)

    def pdf(self, z, x):
        # z is a n x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns m x n matrix, each row is the probability to be taken by the corresponding x and z

        (n, _) = np.shape(z)
        (m, _) = np.shape(x)
        gaussian_pdf = np.zeros(shape=(m, n))
        multinoulli = self.softmax.predict(x)  # m x n

        for i in range(m):
            self.multinoulli = multinoulli[i, :].tolist()
            gaussian_pdf[i, :] = super().pdf(z[:, 0])

        return gaussian_pdf

    def cdf_overall(self, z, x):
        # z is n x 1 matrix, each row represents a market price
        # x is m x feature_dimension matrix, each row represents a bid request to be margin out
        # returns n x 1 matrix, each row is the probability to be taken by the corresponding z
        (m, _) = np.shape(x)
        self.multinoulli = (self.softmax.predict(x).sum(axis=0) / m).tolist()
        # print(self)

        return super().cdf(z)

    def cdf(self, z, x):
        # z is a n x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns n x m matrix, each row is the probability to be taken by the corresponding z and x

        (n, _) = np.shape(z)
        (m, _) = np.shape(x)
        gaussian_cdf = np.zeros(shape=(m, n))
        multinoulli = self.softmax.predict(x)

        for i in range(m):
            self.multinoulli = multinoulli[i, :].tolist()
            gaussian_cdf[i, :] = super().cdf(z[:, 0])

        return gaussian_cdf

    def predict_z(self, x):
        # x is a m x feature_dimension matrix, each row represents a bid request
        # min_z and max_z are the lower bound and upper bound to integrate
        # returns m x 1 matrix, each row is the predicted market price

        zs = np.asmatrix(np.arange(self.min_z, self.max_z+1, 1)).transpose()

        return self.pdf(zs, x) @ zs

    def ANLP(self, z, x):
        # z is a m x 1 matrix, each row represents a market price
        # x is a m x feature_dimension matrix, each row represents a bid request
        # returns a scala, which is average negative log probability (ANLP)

        (m, _) = np.shape(x)
        nlp = np.zeros(shape=(m, 1))
        multinoulli = self.softmax.predict(x)  # m x n

        for i in range(m):
            self.multinoulli = multinoulli[i, :].tolist()
            nlp[i, 0] = -np.log(super().pdf([z[i, 0]]))

        return np.sum(nlp) / m


    def evaluate(self, z, x):
        # please input the test z and test x to evaluate KLD, WD, ANLP and MSE
        zs = list(range(self.min_z, self.max_z))
        pdf = self.pdf_overall(zs, x)
        cdf = self.cdf_overall(zs, x)

        truth_cdf = [self.test_pay.cdf[z] if self.test_pay.cdf[z] > 0 else 1e-6 for z in zs]
        truth_pdf = init_util.cdf_to_pdf(truth_cdf)

        KLD_pdf = entropy(truth_pdf, pdf)
        KLD_cdf = entropy(truth_cdf, cdf)

        WD_pdf = wasserstein_distance(truth_pdf, pdf)
        WD_cdf = wasserstein_distance(truth_cdf, cdf)

        ANLP = self.ANLP(z, x)
        MSE = mean_squared_error(z, self.predict_z(x))

        print("pdf_KLD\tpdf_WD\tcdf_KLD\tcdf_WD\tANLP\tMSE")
        print("{0:.5f}\t{1:.5f}\t{2:.5f}\t{3:.5f}\t{4:.3f}\t{5:.1f}\t".format(
            KLD_pdf, WD_pdf, KLD_cdf, WD_cdf, ANLP, MSE
        ))

        return KLD_pdf, WD_pdf, KLD_cdf, WD_cdf, ANLP, MSE
