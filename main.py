import matplotlib.pyplot as plt
import numpy as np

from sample_generator import generate_dict, Sample


class NORMA:

    def kernel(self, x, y):
        result = 0
        for i in range(len(x)):
            result += x[i] * y[i]
        return result

    def __init__(self, sample):
        self.sample = sample
        self.lambda_var = 1
        l = sample.length()
        self.l = l
        self.C = 1 / (2 * self.lambda_var * l)
        self.ny = 1 / self.lambda_var * 0.5
        self.k = self.kernel
        self.ro = 0
        self.tay = l
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
        for i in range(self.l):
            result += self.coef[i] * self.k(self.sample.points[i].value, x)

        print(result, result + self.b)
        return np.sign(result)

    def f_t(self, point, t):
        ind_start = max(1, t - self.tay)
        ind_end = t - 1
        result = 0
        for i in range(ind_start, ind_end):
            x_i = self.sample.points[i].value
            result += self.coef[i] * self.k(x_i, point)
            #result += self.alpha[i] * self.beta[i] * self.k(x_i, point)
        return result

    def learn(self):
        t = 1
        self.tay = self.l
        #self.beta = [(1 - self.lambda_var * self.ny) ** i for i in range(0, self.tay + 1)]
        self.alpha.append(0)
        b_t = 0
        for point in self.sample.points:
            x_t = point.value
            y_t = point.mark
            f_x_t = self.f_t(x_t, t)
            ny_t = self.ny * (t ** -0.5)
            b_t = b_t - ny_t * self.los_func(f_x_t, y_t)
            alpha_t = - ny_t * self.los_func(f_x_t, y_t)
            # self.alpha.append(alpha_t)
            for i in range(len(self.coef)):
                self.coef[i] = self.coef[i] * (1 - ny_t * self.lambda_var)
            self.coef.append(alpha_t)

            t += 1
        print(b_t)
        self.b = b_t
        #self.coef = [self.alpha[i] * self.beta[i] for i in range(len(self.alpha))]

def main():
    generator = generate_dict["shift_normal"]
    gen_f = generator.get_func()
    shift = [4, 0]
    p = -1
    train_sample = Sample(gen_f(1000, shift=shift, p=p))

    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    fig.show()
    plt.close(fig)

    classificator = NORMA(train_sample)
    classificator.learn()

    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    sample = Sample(gen_f(100, shift=shift, p=p))
    for point in sample.points:
        mark = classificator.classify(point.value)
        if mark == 1:
            ax.plot(point.value[0], point.value[1], ".r")
        elif mark == -1:
            ax.plot(point.value[0], point.value[1], "xb")
        else:
            ax.plot(point.value[0], point.value[1], ".g")

    fig.show()
    plt.close(fig)
    print("\n\n\n")
    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    num_of_elem = 50
    start_x = -3
    start_y = -3
    end_x = 15
    end_y = 15
    for i in range(num_of_elem):
        row = start_y + (end_y - start_y) * i / num_of_elem
        for j in range(num_of_elem):
            col = start_x + (end_x - start_x) * j / num_of_elem
            point = [col, row]

            mark = classificator.classify(point)
            if mark == 1:
                ax.plot(point[0], point[1], ".r")
            elif mark == -1:
                ax.plot(point[0], point[1], "xb")
            else:
                ax.plot(point[0], point[1], ".g")

    fig.show()
    plt.close(fig)



if __name__ == "__main__":
    main()













