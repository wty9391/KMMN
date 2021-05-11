import sys
import math
import numpy as np
from scipy.stats import norm
import collections

epsilon = sys.float_info.epsilon


class Processor:
    def __init__(self, price_name='payprice', min_price=1, max_price=300, price_upper_bound=500):
        self.name_col = {}  # feature_name:origin_index
        self.pdf = {}  # price:probability
        self.cdf = {}  # price:probability
        self.number_record = 0
        self.price_name = price_name
        self.min_price = min_price
        self.max_price = max_price
        self.price_upper_bound = price_upper_bound
        self.data = []

        return

    def load(self, path):
        f = open(path, 'r', encoding="utf-8")
        first = True  # first line is header
        for l in f:
            s = l.split('\t')
            if first:
                for i in range(0, len(s)):
                    self.name_col[s[i].strip()] = i
                price_index = self.name_col[self.price_name]
                first = False
                continue
            price = float(s[price_index])+epsilon
            price_int = math.floor(price)
            price_int = price_int if price_int < self.price_upper_bound else self.price_upper_bound
            self.max_price = price_int if self.max_price < price_int else self.max_price

            if price_int in self.pdf:
                self.pdf[price_int] += 1
            else:
                self.pdf[price_int] = 1

            self.data.append(price_int)
            self.number_record += 1

        for price in range(self.min_price, self.max_price+1):
            if price not in self.pdf:
                self.pdf[price] = 0

        for price in self.pdf:
            self.pdf[price] = self.pdf[price]/self.number_record

        for price in self.pdf:
            p = 0
            for j in self.pdf:
                p += self.pdf[j] if j <= price else 0
            self.cdf[price] = p

        return self.cdf, self.pdf

    def load_by_array(self, z):
        # z is m x 1 matrix

        for s in z.tolist():
            price = float(s[0]) + epsilon
            price_int = math.floor(price)
            price_int = price_int if price_int < self.price_upper_bound else self.price_upper_bound
            self.max_price = price_int if self.max_price < price_int else self.max_price

            if price_int in self.pdf:
                self.pdf[price_int] += 1
            else:
                self.pdf[price_int] = 1

            self.data.append(price_int)
            self.number_record += 1

        for price in range(self.min_price, self.max_price+1):
            if price not in self.pdf:
                self.pdf[price] = 0

        for price in self.pdf:
            self.pdf[price] = self.pdf[price]/self.number_record

        for price in self.pdf:
            p = 0
            for j in self.pdf:
                p += self.pdf[j] if j <= price else 0
            self.cdf[price] = p

        return self.cdf, self.pdf

    def validate(self):
        if self.number_record == 0:
            print("This dataset processor has not been loaded")
            return False

        for i in self.pdf:
            p = 0
            for j in self.pdf:
                p += self.pdf[j] if j <= i else 0
            if self.cdf[i] != p:
                print("pdf: 0:{} is {}, cdf:{} is {}".format(str(i), str(p), str(i), str(self.cdf[i])))
                return False
        print("This dataset processor has validated successfully")
        return True

class Censored_processor():
    def __init__(self, min_price=1, max_price=300):
        self.min_price = min_price
        self.max_price = max_price
        self.truth = {"pdf": {}, "cdf": {}, "data": []}
        self.win = {"pdf": {}, "cdf": {}, "data": []}
        self.lose = {"pdf": {}, "cdf": {}, "data": []}
        self.bid = {"pdf": {}, "cdf": {}, "data": []}
        self.number_record = 0
        self.survive = {"pdf": {}, "cdf": {}, "data": {"b": [], "d": [], "n": []}}
        self.price_upper_bound = 500

        return

    def load(self, x, z, bidder):
        assert x.shape[0] == z.shape[0], "features' number must be equal prices' number"

        bids = bidder.bid(x).reshape((x.shape[0], 1))

        truncate = z > self.price_upper_bound
        z[truncate] = self.price_upper_bound

        win = bids > z
        lose = bids <= z

        self.max_price = max(z[:, 0])

        self.truth = self._count(z[:, 0].tolist())
        self.win = self._count(z[win].tolist())
        self.lose = self._count(z[lose].tolist())
        self.bid = self._count(bids[:, 0].tolist())
        self.number_record = x.shape[0]

        # fit survival model
        zs = list(range(self.min_price, self.max_price+1))

        counter_win_z = collections.Counter(z[win].tolist())
        counter_lose_bid = collections.Counter(bids[lose].tolist())

        counter_win_z_sum = {}
        counter_lose_bid_sum = {}

        for i in range(len(zs)):
            count_win = 0
            count_lose_bid = 0
            for b in zs[i:]:
                count_win = count_win + counter_win_z[b]
                count_lose_bid = count_lose_bid + counter_lose_bid[b]
            counter_win_z_sum[zs[i]] = count_win
            counter_lose_bid_sum[zs[i]] = count_lose_bid

        for b in zs:
            self.survive["data"]["b"].append(b)
            self.survive["data"]["d"].append(counter_win_z[b-1])
            self.survive["data"]["n"].append(counter_win_z_sum[b] + counter_lose_bid_sum[b])

        # calculate cdf
        for b in zs:
            pr_lose = 1.0
            for j in range(zs[0], b):
                index = self.survive["data"]["b"].index(j)
                if self.survive["data"]["n"][index] == 0:
                    pr_lose = 0
                else:
                    pr_lose = pr_lose * (self.survive["data"]["n"][index] - self.survive["data"]["d"][index])\
                        / self.survive["data"]["n"][index]
            self.survive["cdf"][b] = 1 - pr_lose if 1 - pr_lose <= 1.0 else self.survive["cdf"][b-1]
        self.survive["cdf"][0] = 1e-6  # in case of zero
        # calculate pdf
        for b in zs[:-1]:
            self.survive["pdf"][b] = self.survive["cdf"][b+1] - self.survive["cdf"][b] + 1e-6  # in case of zero
        self.survive["pdf"][zs[-1]] = 1e-6

        return

    def _count(self, data):
        pdf = {}
        cdf = {}

        for d in data:
            price = math.floor(float(d) + epsilon)
            if price in pdf:
                pdf[price] += 1
            else:
                pdf[price] = 1

        for price in range(self.min_price, self.max_price+1):
            if price not in pdf:
                pdf[price] = 0

        for price in pdf:
            pdf[price] = pdf[price]/len(data)

        for price in pdf:
            p = 0
            for j in pdf:
                p += pdf[j] if j <= price else 0
            cdf[price] = p

        return {"pdf": pdf, "cdf": cdf, "data": data}


