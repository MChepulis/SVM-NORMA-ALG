import numpy as np


def dot(x, y):
    summa = 0
    for i in range(len(x)):
        summa += x[i] * y[i]
    return summa


def norma2(x):
    summa = 0
    for i in range(len(x)):
        summa += x[i] ** 2
    return np.sqrt(summa)


class kernel_LINEAR:
    def __init__(self, kernel_param={}):
        pass

    def __str__(self):
        result = self.get_name() + "()"
        return result

    @staticmethod
    def get_name():
        return "Liner"

    def kernel_func(self, x, y):
        summa = 0
        for i in range(len(x)):
            summa += x[i] * y[i]
        return summa


class kernel_GAUSS:
    def __init__(self, kernel_param):
        sigma = kernel_param["sigma"]
        self.sigma = sigma

    def __str__(self):
        result = self.get_name() + "(%s)" % format(self.sigma, "")
        return result

    @staticmethod
    def get_name():
        return "Gauss"

    def kernel_func(self, x, y):
        sub_x_y = [x[i] - y[i] for i in range(len(x))]
        result = np.exp(-(norma2(sub_x_y) / (2 * (self.sigma ** 2))))
        return result



