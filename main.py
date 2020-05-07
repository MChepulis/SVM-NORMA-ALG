import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import random

from sample_generator import generate_dict, Sample
from NORMA_SVM import NORMA
from kernels import kernel_LINEAR, kernel_GAUSS
from auto_preprocessing import accuracy_test, greed_args_brute_force


def draw_divide_line_with_contour(train_sample, classificator, num_of_elem=10, offset=2):
    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)

    start_x = train_sample.points[0].value[0]
    start_y = train_sample.points[0].value[1]
    end_x = train_sample.points[0].value[0]
    end_y = train_sample.points[0].value[1]

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
    x = np.linspace(start_x, end_x, num_of_elem)
    y = np.linspace(start_y, end_y, num_of_elem)
    marks = []

    for i in range(num_of_elem):
        row = y[i]
        marks_line = []
        for j in range(num_of_elem):
            col = x[j]
            point = [col, row]
            mark = classificator.classify_func(point)
            marks_line.append(mark)
        marks.append(marks_line)

    lev = [0]
    ax.contour(x, y, marks, levels=lev, colors='r')
    fig.show()
    # plt.close(fig)


def draw_divide_line(train_sample, classificator, num_of_elem=10, offset=2):
    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    num_of_elem = 50

    start_x = train_sample.points[0].value[0]
    start_y = train_sample.points[0].value[1]
    end_x = train_sample.points[0].value[0]
    end_y = train_sample.points[0].value[1]

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
                ax.plot(point[0], point[1], "xb")
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

    read_from_file_flag = True
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
        k = kernel_LINEAR({})
    elif kernel == 'Gauss':
        k = kernel_GAUSS({"sigma": gauss_sigma})

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


def greed_auto_tuning():
    gen_flag = 0
    is_need_save_coef_on_step = False
    if gen_flag == 1:
        key = "circle"
        seed = 1
        generator = generate_dict["circle"]
        generator_args = {
            "p": 0,
            "alpha": 1 / 2,
            "mean": [0, 0],
            "r_mean": 20,
            "r_scale": 0.5,
        }
        train_sample_capacity = 100
        test_sample_capacity = 1000

        ro = 1
        ny_coef = 1/2

        c_min_pow = -25
        c_max_pow = 10
        c_num_of_steps = 25
        c_pow_arr = np.linspace(c_min_pow, c_max_pow, c_num_of_steps)
        c_arr = [2 ** alpha for alpha in c_pow_arr]

        sigma_min = 0.1
        sigma_max = 3
        sigma_num_of_steps = 10
        sigma_arr = np.linspace(sigma_min, sigma_max, sigma_num_of_steps)

        kernel_name_arr = ["kernel_GAUSS"]

        first_cv_order = 3
        second_cv_order = 6

    else:
        key = "shift_normal"
        seed = 1
        generator = generate_dict["shift_normal"]
        generator_args = {
            "shift": [10, 10],
            "p": 0,
            "alpha": 1 / 2,
            "mean": [0, 0]
        }
        train_sample_capacity = 100
        test_sample_capacity = 1000

        ro = 1
        ny_coef = 1/2

        c_min_pow = -10
        c_max_pow = 10
        c_num_of_steps = 25
        c_pow_arr = np.linspace(c_min_pow, c_max_pow, c_num_of_steps)
        c_arr = [2 ** alpha for alpha in c_pow_arr]

        sigma_arr = [1]

        kernel_name_arr = ["kernel_LINEAR"]

        # first_cv_order = 4
        # second_cv_order = 8

        first_cv_order = 4
        second_cv_order = 8

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    train_sample = generator.generate(train_sample_capacity, generator_args)

    best_accuracy, best_params = greed_args_brute_force(train_sample, c_arr, sigma_arr, kernel_name_arr, ro=ro,
                                                        first_cv_order=first_cv_order, second_cv_order=second_cv_order,
                                                        ny_coef=ny_coef)

    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    fig.show()
    plt.close(fig)

    kernel = best_params["kernel"]
    lamda = best_params["lambda_var"]
    ro = best_params["ro"]
    ny = best_params["ny"]

    classificator = NORMA(train_sample)
    classificator.learn(lamda, ro, kernel, ny)

    test_sample = generator.generate(test_sample_capacity, generator_args)

    test(train_sample, test_sample, classificator)

    print("\n\n\n")

    draw_divide_line_with_contour(train_sample, classificator, num_of_elem=50, offset=2)

    print("\n\n\n")

    accuracy = accuracy_test(test_sample, classificator)
    print("accuracy = %s" % format(accuracy, ""))

    accuracy = accuracy_test(train_sample, classificator)
    print("train accuracy = %s" % format(accuracy, ""))

    classificator.show_coef_on_step()
    if is_need_save_coef_on_step:
        classificator.save_coef_on_step_as_gif()

if __name__ == "__main__":
    greed_auto_tuning()

    # main()

















