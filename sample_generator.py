import numpy as np
import random as rand


def generate_shift_normal(capacity, shift=[3, 3], alpha=1/2, p=0, mean=[0, 0]):
    cov1 = [[1, p], [p, 1]]
    mean1 = mean
    mean2 = [mean1[i] + shift[i] for i in range(len(mean1))]

    data = []
    for i in range(0, capacity):
        flag = rand.random()
        if flag < alpha:
            value = np.random.multivariate_normal(mean1, cov1, 1)
            tmp = Point(value[0], 1)
        else:
            value = np.random.multivariate_normal(mean2, cov1, 1)
            tmp = Point(value[0], -1)
        data.append(tmp)
    return data


def generate_uniform_sin(capacity):
    tmp_data = np.random.uniform(-1, 1, capacity)
    data = []
    for value in tmp_data:
        x = np.pi * value
        mark = np.sign(np.sin(x))
        tmp = Point(value, mark)
        data.append(tmp)

    return data


class Point:
    def __init__(self, value, mark):
        self.value = value
        self.mark = mark

    def get_value(self):
        return self.value

    def get_mark(self):
        return self.mark


class Sample:
    def __init__(self, points):
        self.points = points
        self.size = len(points)

    def length(self):
        return self.size

    def get_arrays_for_plot(self):
        res_x = []
        res_y = []
        for point in self.points:
            res_x.append(point.value)
            res_y.append(point.mark)

        return res_x, res_y

    def draw(self, fig, ax, axis_num=None, marker=None):
        if axis_num is None:
            axis_num = [0, 1]
        if len(axis_num) > 2:
            print("draw only 2D plots")
            axis_num = [0, 1]
        x, y = self.get_arrays_for_plot()
        flags = [mark == +1 for mark in y]

        x1 = []
        x2 = []
        y1 = []
        y2 = []

        for i in range(len(y)):
            if flags[i]:
                y1.append(x[i][axis_num[1]])
                x1.append(x[i][axis_num[0]])
            else:
                y2.append(x[i][axis_num[1]])
                x2.append(x[i][axis_num[0]])
        if marker:
            ax.plot(x1, y1, ".", color="k", marker=".")
            ax.plot(x2, y2, ".", color="k", marker="x")
        else:
            ax.plot(x1, y1, ".", color="r")
            ax.plot(x2, y2, ".", color="b")


class SampleGenerator:
    def __init__(self, generate_func_):
        self.generate_func = generate_func_

    def get_func(self):
        return self.generate_func

    def generate(self, num):
        return Sample(self.generate_func(num))


generate_dict = {
    'shift_normal': SampleGenerator(generate_shift_normal),
    'uniform_sin': SampleGenerator(generate_uniform_sin),
}