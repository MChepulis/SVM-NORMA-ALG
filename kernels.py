import numpy as np


def dot(x, y):
    summa = 0
    for i in range(len(x)):
        summa += x[i] * y[i]
    return summa


class kernel_NORMA2:
    def __init__(self):
        pass

    def kernel_func(self, x):
        summa = 0
        for i in range(len(x)):
            summa += x[i] ** 2
        return np.sqrt(summa)


class kernel_LINEAR:
    def __init__(self):
        pass

    def kernel_func(self, x, y):
        summa = 0
        for i in range(len(x)):
            summa += x[i] * y[i]
        return summa


class kernel_GAUSS:
    def __init__(self, sigma):
        self.sigma = sigma

    def kernel_func(self, x, y):
        result = np.exp(-(dot(x, y) / (2 * (self.sigma ** 2))))
        return result

