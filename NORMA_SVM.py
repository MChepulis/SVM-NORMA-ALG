import numpy as np


class NORMA:

    def linear_kernel(self, x, y):
        result = 0
        for i in range(len(x)):
            result += x[i] * y[i]
        return result

    def __init__(self, sample):
        self.sample = sample
        length = sample.length()
        self.length = length

        self.defaul = {
            "lambda_var" : 40,
            "ro" : 1,
            "kernel" : self.linear_kernel,
            "ny_coef" : 0.5
        }

    def init_lear(self):
        self.C = 1 / (2 * self.lambda_var * self.length)
        self.ny = 1 / self.lambda_var * self.ny_coef

        self.tay = self.length
        self.alpha = []
        self.beta = []
        self.coef = []
        self.b = 0


    def los_func(self, f_x_t, y_t):
        if y_t * f_x_t <= self.ro:
            return -y_t
        else:
            return 0


    def classify(self, x):
        result = 0
        for i in range(self.length):
            result += self.coef[i] * self.kernel(self.sample.points[i].value, x)

        return np.sign(result + self.b)

    def f_t(self, point, t):
        ind_start = max(1, t - self.tay)
        ind_end = t - 1
        result = 0
        for i in range(ind_start, ind_end):
            x_i = self.sample.points[i].value
            result += self.coef[i] * self.kernel(x_i, point)
        return result

    def learn(self, lambda_var = None, ro=None, kernel=None, ny_coef=None):

        if lambda_var is None:
            self.lambda_var = self.defaul["lambda_var"]
        else:
            self.lambda_var = lambda_var

        if ro is None:
            self.ro = self.defaul["ro"]
        else:
            self.ro = ro

        if kernel is None:
            self.kernel = self.defaul["kernel"]
        else:
            self.kernel = kernel

        if ny_coef is None:
            self.ny_coef = self.defaul["ny_coef"]
        else:
            self.ny_coef = ny_coef

        self.init_lear()
        t = 1
        self.tay = self.length
        self.alpha.append(0)
        b_t = 0
        for point in self.sample.points:
            x_t = point.value
            y_t = point.mark
            f_x_t = self.f_t(x_t, t)
            ny_t = self.ny * (t ** -0.5)

            if y_t * f_x_t > self.ro:
                sigma_t = 0
            else:
                sigma_t = 1

            alpha_t = ny_t * sigma_t * y_t
            for i in range(len(self.coef)):
                self.coef[i] = self.coef[i] * (1 - ny_t * self.lambda_var)
            self.coef.append(alpha_t)
            b_t = b_t + alpha_t

            t += 1
        self.b = b_t