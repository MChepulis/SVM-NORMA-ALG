import numpy as np


def norma2(x):
    summa = 0
    for i in range(len(x)):
        summa += x[i] ** 2
    return np.sqrt(summa)


def dot(x, y):
    summa = 0
    for i in range(len(x)):
        summa += x[i] * y[i]
    return summa

sigma = None
def gauss_kernel(x, y):
    global sigma
    if sigma is None:
        sigma = 1
    result = np.exp(-(dot(x, y) / (2 * (sigma ** 2))))
    return result


def linear_kernel(x, y):
    result = 0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result
