import matplotlib.pyplot as plt
import numpy as np

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
    #plt.close(fig)


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


def run_NORMA(sample_capasity, correlation, shift, lamda, ro, ny, kernel, gauss_sigma):
    gen_flag = 2
    if gen_flag == 1:
        generator = generate_dict["shift_normal"]
        generator_args = {
            "shift": [shift, shift],
            "p": correlation,
            "alpha": 1 / 2,
            "mean": [0, 0]
        }
    elif gen_flag == 2:
        generator = generate_dict["circle"]
        generator_args = {
            "p": correlation,
            "alpha": 1 / 2,
            "mean": [0, 0],
            "r_mean": 20,
            "r_scale": 0.5,
        }

    sample_capacity = sample_capasity

    read_from_file_flag = False
    data_filename = "data.csv"
    key = "circle"

    if read_from_file_flag:
        train_sample = Sample([])
        train_sample.read_sample_from_csv_file(data_filename, keyword=key)
    else:
        train_sample = generator.generate(sample_capacity, generator_args)
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


if __name__ == "__main__":
    import PySimpleGUI as sg

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
              [sg.Text('Sample capasity:'), sg.InputText(key='sample_capasity', default_text=sample_capacity)],
              [sg.Text('Сorrelation:'), sg.InputText(key='correlation', default_text=correlation)],
              [sg.Text('Shift:'), sg.InputText(key='shift', default_text=shift)],

              [sg.Text('NORMA settings', font=("Helvetica", 15), text_color='blue')],
              [sg.Text('lambda:'), sg.InputText(key='lambda', default_text=lamda)],
              [sg.Text('ro:'), sg.InputText(key='ro', default_text=ro)],
              [sg.Text('ny:'), sg.InputText(key='ny', default_text=ny)],

              [sg.Text('Kernel:'), sg.Listbox(values=["Linear", "Gauss"], size=(30, 2), key='kernel', default_values=["Linear"])],

              [sg.Text('Gauss kernel settings', font=("Helvetica", 15), text_color='blue')],
              [sg.Text('sigma:'), sg.InputText(key='gauss_sigma', default_text=gauss_sigma)],

              [sg.Button('Ok'), sg.Button('Cancel')]
    ]

    window = sg.Window('Window Title', layout)
    while True:
        event, values = window.read()
        if event in (None, 'Cancel'):
            break
        if event in (None, 'Ok'):
            print(values['correlation'], values['sample_capasity'], values['shift'], values['lambda'], values['ro'])
            run_NORMA(
                sample_capasity=int(values['sample_capasity']),
                correlation=float(values['correlation']),
                shift=float(values['shift']),
                lamda=float(values['lambda']),
                ro=float(values['ro']),
                ny=float(values['ny']),
                kernel=values['kernel'][0],
                gauss_sigma=float(values['gauss_sigma'])
            )














