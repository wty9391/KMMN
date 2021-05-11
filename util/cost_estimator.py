import numpy as np




class Cost_estimator():
    def __init__(self, pdf, min_z, max_z):
        # min_z must equal to 1, i.e., pdf[0] is the probability of market price being equal to 1
        self.pdf = pdf
        self.min_z = min_z
        self.max_z = max_z
        self.cost = [0]  # cost[index] means the estimated ad cost if b=index

        for i in range(len(pdf)):
            self.cost.append(self.cost[-1] + (i+1)*pdf[i])

        return

    def predict(self, b):
        # b is m x 1 matrix
        b = np.rint(b.ravel()).astype(np.int).tolist()
        cost = [self.cost[i] if i < self.max_z else self.cost[self.max_z] for i in b]

        return sum(cost)


