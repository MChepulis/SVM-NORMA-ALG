import matplotlib.pyplot as plt
import numpy as np

from sample_generator import generate_dict, Sample
from NORMA_SVM import NORMA


def draw_divide_line(train_sample, classificator):

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


def test(train_sample, test_sample, classificator, ):
    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)

    for point in test_sample.points:
        mark = classificator.classify(point.value)
        if mark == 1:
            ax.plot(point.value[0], point.value[1], ".r")
        elif mark == -1:
            ax.plot(point.value[0], point.value[1], "xb")
        else:
            ax.plot(point.value[0], point.value[1], ".g")
    fig.show()
    plt.close(fig)


def main():
    generator = generate_dict["shift_normal"]
    gen_f = generator.get_func()
    shift = [10, 10]
    p = 0  # коэф корреляции x_1 и x_2
    sample_capacity = 1000

    read_from_file_flag = False
    data_filename = "data.csv"

    if read_from_file_flag:
        train_sample = Sample([])
        train_sample.read_sample_from_csv_file(data_filename)
    else:
        train_sample = Sample(gen_f(sample_capacity, shift=shift, p=p))
        train_sample.write_sample_to_csv_file(data_filename)

    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    fig.show()
    plt.close(fig)

    classificator = NORMA(train_sample)
    classificator.learn(lambda_var=25, ro=1, kernel=None)

    test_sample_capacity = 100
    test_sample = Sample(gen_f(test_sample_capacity, shift=shift, p=p))

    test(train_sample, test_sample, classificator)

    print("\n\n\n")

    draw_divide_line(train_sample, classificator)


if __name__ == "__main__":
    main()













