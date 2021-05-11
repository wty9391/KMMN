import numpy as np
from util import util

class KM():
    def __init__(self, max_market_price=301):
        self.max_market_price = max_market_price
        self.d = []
        self.n = []
        self.h = []
        self.s = []
        self.w = []
        self.pdf = []
        self.z = 0

    def fit(self, z, b):
        """

        :param z: n x 1
        :param b: n x 1
        :return:
        """
        self.__init__(max_market_price=self.max_market_price)

        win = (b > z)[:, 0]
        lose = np.logical_not(win)

        # in_case_of_0 = np.asarray(list(range(self.max_market_price))).reshape((-1, ))

        (unique_z, counts_z) = np.unique(z[win], return_counts=True)  # the unique has been sorted
        (unique_b, counts_b) = np.unique(b[lose], return_counts=True)  # the unique has been sorted

        unique_z = unique_z.tolist()
        unique_b = unique_b.tolist()

        win_z_count = []  # win cases with z=i
        lose_b_count = []  # lose cases with b=i
        lose_b_sum_count = []  # lose cases with b>=i
        win_z_sum_count = []  # win cases with z>=i

        for i in range(self.max_market_price):
            this_z_count = counts_z[unique_z.index(i)] if i in unique_z else 0
            this_b_count = counts_b[unique_b.index(i)] if i in unique_b else 0
            win_z_count.append(this_z_count)
            lose_b_count.append(this_b_count)

        lose_b_sum_count = [sum(lose_b_count[i:]) for i in range(self.max_market_price)]
        win_z_sum_count = [sum(win_z_count[i:]) for i in range(self.max_market_price)]

        for i in range(self.max_market_price):
            d = win_z_count[i]
            n = lose_b_sum_count[i] + win_z_sum_count[i]
            h = 1 - d / n if n > 0 else 1
            self.d.append(d)
            self.n.append(n)
            self.h.append(h)

        for i in range(self.max_market_price):
            s = np.prod(self.h[:i+1])
            w = 1 - s
            self.s.append(s)
            self.w.append(w)

        self.pdf = util.Util.cdf_to_pdf(self.w)
        self.w = util.Util.pdf_to_cdf(self.pdf)  # normalize

        for i in range(self.max_market_price):
            self.z += i * self.pdf[i]

        return

    def predict_z(self, x):
        (record_size, _) = np.shape(x)

        z = np.zeros((record_size, 1)) + self.z

        return z

    def anlp(self, x, z):
        (record_size, _) = np.shape(x)
        z = z.astype(np.int16).reshape((record_size,))

        nlp = [-np.log(self.pdf[i]) for i in z.tolist()]

        return sum(nlp) / record_size

