import numpy as np
from enum import Enum

class NORMA:
    def __init__(self, sample):
        self.sample = sample
        self.length = sample.length()

    def init_learn(self):
        self.C = 1 / (2 * self.lambda_var * self.length)
        self.ny = 1 / self.lambda_var * self.ny_coef

        self.tay = self.length
        self.alpha = []
        self.beta = []
        self.coef = []
        self.b = 0

    def los_func_deriv(self, f_x_t, y_t):
        if y_t * f_x_t <= self.ro:
            return -y_t
        else:
            return 0

    def classify(self, x):
        result = 0
        for i in range(self.length):
            result += self.coef[i] * self.kernel.kernel_func(self.sample.points[i].value, x)

        return np.sign(result + self.b)

    def classify(self, x):
        result = 0
        for i in range(self.length):
            result += self.coef[i] * self.kernel.kernel_func(self.sample.points[i].value, x)

        return np.sign(result + self.b)

    class AnswerType(Enum):
        LEFT = 1
        RIGHT = 2
        LEFT_MARGIN = 3
        RIGHT_MARGIN = 4

    def get_classify_answer(self, x):
        f_x = 0
        for i in range(self.length):
            f_x += self.coef[i] * self.kernel.kernel_func(self.sample.points[i].value, x)
        g_x = f_x + self.b
        if self.ro < g_x:
            return self.AnswerType.RIGHT
        elif 0 < g_x < self.ro:
            return self.AnswerType.RIGHT_MARGIN
        elif -self.ro < g_x < 0:
            return self.AnswerType.LEFT_MARGIN
        elif g_x < -self.ro:
            return self.AnswerType.LEFT

    def f_t(self, point, t):
        ind_start = max(1, t - self.tay)
        ind_end = t - 1
        result = 0
        for i in range(ind_start, ind_end):
            x_i = self.sample.points[i].value
            result += self.coef[i] * self.kernel.kernel_func(x_i, point)
        return result

    def learn(self, lambda_var, ro, kernel, ny_coef):
        self.lambda_var = lambda_var
        self.ro = ro
        self.kernel = kernel
        self.ny_coef = 0.5

        self.init_learn()
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