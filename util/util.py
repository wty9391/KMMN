import sys
import math
import numpy as np
import time

eps = sys.float_info.epsilon


class Util:
    def __init__(self):
        pass

    @staticmethod
    def generate_batch_index(size, batch_size=4096):
        starts = [int(i * batch_size) for i in range(int(math.ceil(size / batch_size)))]
        ends = [int(i * batch_size) for i in range(1, int(math.ceil(size / batch_size)))]
        ends.append(int(size))

        return starts, ends

    @staticmethod
    def cdf_to_pdf(cdf):
        cdf = cdf.copy()
        cdf.insert(0, 0.0 + eps)
        pdf = [cdf[i + 1] - cdf[i] if cdf[i + 1] - cdf[i] > 0 else eps for i in range(len(cdf) - 1)]
        # normalize
        pdf_sum = sum(pdf)

        return [p / pdf_sum for p in pdf]

    @staticmethod
    def pdf_to_cdf(pdf):
        # normalize
        pdf_sum = sum(pdf)
        pdf = [pdf[i] / pdf_sum if pdf[i] > 0 else eps for i in range(len(pdf))]

        cdf = [sum(pdf[0:i+1]) for i in range(len(pdf))]
        return cdf


class Timeit:
    def __init__(self):
        self.previous = time.time()
        self.time = [self.previous]

    def tick(self):
        current = time.time()
        print("--- elapse {.2f} seconds ---".format(current - self.previous))
        self.previous = current
        self.time.append(current)

