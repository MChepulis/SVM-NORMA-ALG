import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from celluloid import Camera


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
        self.coef_on_step = []
        self.b_on_step = []

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

    def classify_func(self, x):
        result = 0
        for i in range(self.length):
            result += self.coef[i] * self.kernel.kernel_func(self.sample.points[i].value, x)

        return result + self.b

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
        if self.ro <= g_x:
            return self.AnswerType.RIGHT
        elif 0 <= g_x < self.ro:
            return self.AnswerType.RIGHT_MARGIN
        elif -self.ro <= g_x < 0:
            return self.AnswerType.LEFT_MARGIN
        elif g_x <= -self.ro:
            return self.AnswerType.LEFT

    def f_t(self, point, t):
        ind_start = max(1, t - self.tay)
        ind_end = t - 1
        result = 0
        for i in range(ind_start, ind_end):
            x_i = self.sample.points[i].value
            result += self.coef[i] * self.kernel.kernel_func(x_i, point)
        return result


    def save_coef(self):
        line = np.zeros(self.length)
        for i in range(len(self.coef)):
            line[i] = self.coef[i]
        self.coef_on_step.append(line)

    def save_b(self):
        self.b_on_step.append(self.b)

    def learn(self, lambda_var, ro, kernel, ny_coef):
        self.lambda_var = lambda_var
        self.ro = ro
        self.kernel = kernel
        self.ny_coef = ny_coef

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
            self.save_coef()
            self.save_b()

            t += 1
        self.b = b_t

    def save_coef_on_step_as_gif(self):
        name = "coef_on_step"
        fig, ax = plt.subplots(1, 1)
        camera = Camera(fig)
        # plt.ylim((-0.02, 0.015))
        for line in self.coef_on_step:
            x_plot = range(len(line))
            ax.bar(x_plot, line, color='dodgerblue')
            camera.snap()

        anim = camera.animate()
        anim.save("%s.gif" % name, writer="imagemagick")
        plt.close(fig)

    def show_coef_on_step(self, step = -1):

        fig, ax = plt.subplots(1, 1)
        y_plot = self.coef_on_step[step]
        x_plot = range(len(y_plot))
        ax.bar(x_plot, y_plot, color='dodgerblue')
        fig.show()
        # plt.close(fig)

