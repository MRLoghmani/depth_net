import numpy as np
# Based on code from http://www.johndcook.com/blog/standard_deviation/
# Original algorithm in 1962 paper by B. P. Welford and is presented
# in Donald Knuth’s Art of Computer Programming, Vol 2, page 232, 3rd edition.


class Estimator:

    def __init__(self):
        self.reset()

    def reset(self):
        self.mean = None
        self.k = 0

    def push(self, val):
        self.k += 1
        if self.mean is None:
            self.mean = np.array(val, dtype='float32')
            self.var = np.zeros(val.shape, dtype='float32')
            return
        new_mean = self.mean + (val - self.mean) / self.k
        self.var = self.var + (val - self.mean) * (val - new_mean)
        self.mean = new_mean

    def get_mean(self):
        return self.mean

    def get_std(self):
        return np.sqrt(self.var / self.k)
