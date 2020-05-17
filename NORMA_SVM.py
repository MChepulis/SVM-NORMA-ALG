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
        return np.sign(self.classify_func(x))

    def reduced_classify(self, x, barrier_percent=0.001):
        return np.sign(self.reduced_classify_func(x, barrier_percent=barrier_percent))

    def classify_func(self, x):
        result = 0
        for i in range(self.length):
            result += self.coef[i] * self.kernel.kernel_func(self.sample.points[i].value, x)
        return result + self.b

    def reduced_classify_func(self, x, barrier_percent=0.1):
        result = 0
        barrier = barrier_percent * np.max(np.abs(self.coef))
        tmp_coef = []
        for i in range(self.length):
            if np.abs(self.coef[i]) <= barrier:
                tmp_coef.append(0)
            else:
                tmp_coef.append(self.coef[i])
        for i in range(self.length):
            result += tmp_coef[i] * self.kernel.kernel_func(self.sample.points[i].value, x)
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
        return result + self.b


    def save_coef(self):
        line = np.zeros(self.length)
        for i in range(len(self.coef)):
            line[i] = self.coef[i]
        self.coef_on_step.append(line)

    def save_b(self, b_t):
        self.b_on_step.append(b_t)

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
            self.b = b_t
            self.save_coef()
            self.save_b(b_t)

            t += 1
        self.b = b_t

    def separate_by_mark(self, x_plot, y_plot):
        first_x = []
        first_y = []
        second_x = []
        second_y = []
        for i in range(len(self.sample.points)):
            point = self.sample.points[i]
            if point.mark == +1:
                first_x.append(x_plot[i])
                first_y.append(y_plot[i])
            else:
                second_x.append(x_plot[i])
                second_y.append(y_plot[i])
        return first_x, first_y, second_x, second_y

    def save_coef_on_step_as_gif(self, name="coef_on_step"):
        fig, ax = plt.subplots(1, 1)
        camera = Camera(fig)
        # plt.ylim((-0.02, 0.015))
        for line in self.coef_on_step:
            x_plot = range(len(line))
            y_plot = line
            first_x, first_y, second_x, second_y = self.separate_by_mark(x_plot, y_plot)
            ax.bar(first_x, first_y, color='dodgerblue')
            ax.bar(second_x, second_y, color='darkorange')
            camera.snap()

        anim = camera.animate()
        anim.save("%s.gif" % name, writer="imagemagick")
        plt.close(fig)

    def show_b_on_step(self):

        fig, ax = plt.subplots(1, 1)
        y_plot = self.b_on_step
        x_plot = range(len(y_plot))
        first_x, first_y, second_x, second_y = self.separate_by_mark(x_plot, y_plot)
        ax.plot(x_plot, y_plot)
        ax.plot(first_x, first_y, ".", color='dodgerblue')
        ax.plot(second_x, second_y, ".", color='darkorange')
        fig.show()
        # plt.close(fig)

    def show_coef_on_step(self, step=-1):

        fig, ax = plt.subplots(1, 1)
        y_plot = self.coef_on_step[step]
        x_plot = range(len(y_plot))
        first_x, first_y, second_x, second_y = self.separate_by_mark(x_plot, y_plot)
        ax.bar(first_x, first_y, color='dodgerblue')
        ax.bar(second_x, second_y, color='darkorange')
        fig.show()
        # plt.close(fig)

    def classify_func_on_step(self, step, x):
        kernel = self.kernel
        coef = self.coef_on_step[step]
        b = self.b_on_step[step]
        result = 0
        for i in range(self.length):
            result += coef[i] * kernel.kernel_func(self.sample.points[i].value, x)

        return result + b

    def reduced_classify_func_on_step(self, step, x, barrier_percent=0.1):
        kernel = self.kernel
        coef = self.coef_on_step[step]
        b = self.b_on_step[step]
        result = 0
        tmp_coef = []
        barrier = barrier_percent * np.max(np.abs(coef))
        for i in range(self.length):
            if np.abs(coef[i]) <= barrier:
                tmp_coef.append(0)
            else:
                tmp_coef.append(coef[i])
        for i in range(self.length):
            result += tmp_coef[i] * kernel.kernel_func(self.sample.points[i].value, x)

        return result + b

    def count_non_zero_coef(self, step=-1, barrier_percent=0):
        coef = self.coef_on_step[step]
        count = 0
        for value in coef:
            if value != 0:
                count += 1

        barrier = barrier_percent * np.max(np.abs(coef))
        count_red = 0
        for i in range(self.length):
            if np.abs(coef[i]) > barrier:
                count_red += 1

        return count, count_red, self.length








