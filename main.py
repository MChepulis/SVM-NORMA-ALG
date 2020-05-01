import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import random

from sample_generator import generate_dict, Sample
from NORMA_SVM import NORMA
from kernels import kernel_LINEAR, kernel_GAUSS


def draw_divide_line(train_sample, classificator):
    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    num_of_elem = 20

    start_x = train_sample.points[0].value[0]
    start_y = train_sample.points[0].value[1]
    end_x = train_sample.points[0].value[0]
    end_y = train_sample.points[0].value[1]

    offset = 2

    for pt in train_sample.points:
        if pt.value[0] < start_x:
            start_x = pt.value[0]
        if pt.value[1] < start_y:
            start_y = pt.value[1]
        if pt.value[0] > end_x:
            end_x = pt.value[0]
        if pt.value[1] > end_y:
            end_y = pt.value[1]

    start_x = start_x - offset
    start_y = start_y - offset
    end_x = end_x + offset
    end_y = end_y + offset

    for i in range(num_of_elem):
        row = start_y + (end_y - start_y) * i / (num_of_elem - 1)
        for j in range(num_of_elem):
            col = start_x + (end_x - start_x) * j / (num_of_elem - 1)
            point = [col, row]
            mark = classificator.get_classify_answer(point)
            if mark == classificator.AnswerType.RIGHT:
                ax.plot(point[0], point[1], ".r")
            elif mark == classificator.AnswerType.RIGHT_MARGIN:
                ax.plot(point[0], point[1], ".g")
            elif mark == classificator.AnswerType.LEFT:
                ax.plot(point[0], point[1], "xb")
            elif mark == classificator.AnswerType.LEFT_MARGIN:
                ax.plot(point[0], point[1], "xg")
            else:
                ax.plot(point[0], point[1], "hm")

    fig.show()
    # plt.close(fig)


def test(train_sample, test_sample, classificator):
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


def run_NORMA(sample_capacity, correlation, shift, lamda, ro, ny, kernel, gauss_sigma):
    gen_flag = 1
    if gen_flag == 1:
        key = "circle"
        generator = generate_dict["circle"]
        generator_args = {
            "p": correlation,
            "alpha": 1 / 2,
            "mean": [0, 0],
            "r_mean": 20,
            "r_scale": 0.5,
        }
    else:
        key = "shift_normal"
        generator = generate_dict["shift_normal"]
        generator_args = {
            "shift": [shift, shift],
            "p": correlation,
            "alpha": 1 / 2,
            "mean": [0, 0]
        }

    read_from_file_flag = False
    data_filename = "data.csv"
    train_sample_capacity = sample_capacity

    if read_from_file_flag:
        train_sample = Sample([])
        train_sample.read_sample_from_csv_file(data_filename, keyword=key)
    else:
        train_sample = generator.generate(train_sample_capacity, generator_args)
        train_sample.write_sample_to_csv_file(data_filename, keyword=key)

    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    fig.show()
    plt.close(fig)

    k = None
    if kernel == 'Linear':
        k = kernel_LINEAR()
    elif kernel == 'Gauss':
        k = kernel_GAUSS(gauss_sigma)

    classificator = NORMA(train_sample)
    classificator.learn(lamda, ro, k, ny)

    test_sample_capacity = 100
    test_sample = generator.generate(test_sample_capacity, generator_args)

    test(train_sample, test_sample, classificator)

    print("\n\n\n")

    draw_divide_line(train_sample, classificator)

    print("\n\n\n")

    accuracy = accuracy_test(test_sample, classificator)
    print("accuracy = %s" % format(accuracy, ""))

def accuracy_test(test_sample, classificator):
    good_answer = 0
    for point in test_sample.points:
        answer = classificator.classify(point.value)
        if answer == point.mark:
            good_answer += 1
    accuracy = good_answer / test_sample.length()
    return accuracy


def cross_validation(train_sample, test_sample,  norma_params):
    lambda_var = norma_params["lambda_var"]
    ro = norma_params["ro"]
    kernel = norma_params["kernel"]
    ny = norma_params["ny"]

    classificator = NORMA(train_sample)
    classificator.learn(lambda_var, ro, kernel, ny)
    accuracy = accuracy_test(test_sample, classificator)
    return accuracy


def cv_test(sample, norma_param, order=1,  train_percent=0.8):
    divide_ind = int(np.floor(sample.length() * train_percent))

    accuracy_arr = []
    for i in range(order):
        points = sample.points.copy()
        random.shuffle(points)

        train_sample = Sample(points[:divide_ind])
        test_sample = Sample(points[divide_ind:])

        curr_accuracy = cross_validation(train_sample, test_sample, norma_param)
        accuracy_arr.append(curr_accuracy)

    accuracy_mean = np.mean(accuracy_arr)
    accuracy_var = np.var(accuracy_arr)
    return accuracy_mean, accuracy_var


def greed_args_brute_force():
    lambda_min = 0.1
    lambda_max = 20
    lambda_num_of_steps = 10
    lambda_arr = np.linspace(lambda_min, lambda_max, lambda_num_of_steps)

    ro_min = 0.001
    ro_max = 2
    ro_num_of_steps = 10
    ro_arr = np.linspace(ro_min, ro_max, ro_num_of_steps)

    sigma_min = 0
    sigma_max = 5
    sigma_num_of_steps = 10
    sigma_arr = np.linspace(sigma_min, sigma_max, sigma_num_of_steps)

    kernel_name_arr = ["kernel_GAUSS", "kernel_LINEAR"]
    sample_capacity = 100

    gen_flag = 1
    if gen_flag == 1:
        key = "circle"
        generator = generate_dict["circle"]
        generator_args = {
            "p": 0,
            "alpha": 1 / 2,
            "mean": [0, 0],
            "r_mean": 20,
            "r_scale": 0.5,
        }
    else:
        key = "shift_normal"
        generator = generate_dict["shift_normal"]
        generator_args = {
            "shift": [10, 10],
            "p": 0,
            "alpha": 1 / 2,
            "mean": [0, 0]
        }

    for kernel_name in kernel_name_arr:
        if kernel_name == "kernel_GAUSS":
            tmp_sigma_arr = sigma_arr
        elif kernel_name == "kernel_LINEAR":
            tmp_sigma_arr = [1]
            kernel = kernel_LINEAR()
        else:
            tmp_sigma_arr = [1]

        for sigma in tmp_sigma_arr:
            if kernel_name == "kernel_GAUSS":
                kernel = kernel_GAUSS(sigma)

            for lambda_var in lambda_arr:
                for ro in ro_arr:
                    norma_param = {
                        "lambda_var": lambda_var,
                        "ro": ro,
                        "kernel": kernel,
                        "ny": 1/2
                    }

                    sample = generator.generate(sample_capacity, generator_args)
                    accuracy_mean, accuracy_var = cv_test(sample, norma_param, 1, 0.8)
                    print("kernel = %s" % kernel.get_name(), end="\t")
                    print("lambda_var = %s" % format(lambda_var, ""), end="\t")
                    print("ro = %s" % format(ro, ""), end="\t")
                    print("accuracy_var = %s" % format(accuracy_var, ""), end="\t")
                    print("accuracy_mean = %s" % format(accuracy_mean, ""), end="\t")
                    print()

        # TODO обработаьть


def main():
    # default params
    sample_capacity = 100
    shift = 10
    correlation = 0
    lamda = 10
    ro = 1
    ny = 0.5
    gauss_sigma = 1

    sg.theme('DarkAmber')

    layout = [[sg.Text('Sample settings', font=("Helvetica", 15), text_color='blue')],
              [sg.Text('Sample capacity:'), sg.InputText(key='sample_capacity', default_text=format(sample_capacity, ""))],
              [sg.Text('Сorrelation:'), sg.InputText(key='correlation', default_text=format(correlation, ""))],
              [sg.Text('Shift:'), sg.InputText(key='shift', default_text=format(shift, ""))],

              [sg.Text('NORMA settings', font=("Helvetica", 15), text_color='blue')],
              [sg.Text('lambda:'), sg.InputText(key='lambda', default_text=format(lamda, ""))],
              [sg.Text('ro:'), sg.InputText(key='ro', default_text=format(ro, ""))],
              [sg.Text('ny:'), sg.InputText(key='ny', default_text=format(ny, ""))],

              [sg.Text('Kernel:'),
               sg.Listbox(values=["Linear", "Gauss"], size=(30, 2), key='kernel', default_values=["Linear"])],

              [sg.Text('Gauss kernel settings', font=("Helvetica", 15), text_color='blue')],
              [sg.Text('sigma:'), sg.InputText(key='gauss_sigma', default_text=format(gauss_sigma, ""))],

              [sg.Button('Ok'), sg.Button('Cancel')]
              ]

    window = sg.Window('Window Title', layout)
    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):
            break
        if event in (None, 'Ok'):
            print(values['correlation'], values['sample_capacity'], values['shift'], values['lambda'], values['ro'])
            run_NORMA(
                sample_capacity=int(values['sample_capacity']),
                correlation=float(values['correlation']),
                shift=float(values['shift']),
                lamda=float(values['lambda']),
                ro=float(values['ro']),
                ny=float(values['ny']),
                kernel=values['kernel'][0],
                gauss_sigma=float(values['gauss_sigma'])
            )


if __name__ == "__main__":
    # greed_args_brute_force()
    main()

















