import matplotlib.pyplot as plt
import numpy as np
import PySimpleGUI as sg
import random

from sample_generator import generate_dict, Sample
from NORMA_SVM import NORMA
from kernels import kernel_LINEAR, kernel_GAUSS
from auto_preprocessing import accuracy_test, greed_args_brute_force


def draw_divide_line_with_contour(train_sample, classificator, num_of_elem=10, offset=2,
                                  is_need_plot_reduced_line=False, barrier=0.001):
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
    marks_reduce = []
    for i in range(num_of_elem):
        row = y[i]
        marks_line = []
        mark_reduce_line = []
        for j in range(num_of_elem):
            col = x[j]
            point = [col, row]
            mark = classificator.classify_func(point)
            marks_line.append(mark)

            if is_need_plot_reduced_line:
                mark_reduce = classificator.reduced_classify_func(point, barrier_percent=barrier)
                mark_reduce_line.append(mark_reduce)

        marks.append(marks_line)
        if is_need_plot_reduced_line:
            marks_reduce.append(mark_reduce_line)

    lev = [0]
    ax.contour(x, y, marks, levels=lev, colors='r')
    ax.plot([start_x], [start_y], color="r", label="separate line")
    if is_need_plot_reduced_line:
        ax.contour(x, y, marks_reduce, levels=lev, colors='g')
        ax.plot([start_x], [start_y], color="g", label="reduced separate line({})".format(barrier))
        ax.legend()
    fig.show()
    # plt.close(fig)


from celluloid import Camera


def draw_line_on_step(classificator, num_of_elem=10, offset=2, name="line_on_step",
                      is_need_plot_reduced_line=False, barrier=0.001):
    fig, ax = plt.subplots(1, 1)
    camera = Camera(fig)
    train_sample = classificator.sample

    ax.plot([0], [0], color="r", label="separate line")
    ax.plot([0], [0], color="g", label="reduced separate line({})".format(barrier))
    fig.legend()

    for i in range(len(classificator.sample.points)):
        print("{}\t". format(i, ""), end="")
        tmp_sample = Sample(classificator.sample.points[:i])
        # train_sample.draw(fig, ax, marker=True)
        tmp_sample.draw(fig, ax, marker=False)
        draw_divide_line_with_contour_on_step(fig, ax, train_sample, classificator, num_of_elem, offset, i,
                                              is_need_plot_reduced_line=is_need_plot_reduced_line, barrier=barrier)

        camera.snap()

    print()
    print("save init")
    anim = camera.animate()
    anim.save("%s.gif" % name, writer="imagemagick")
    print("save done")
    plt.close(fig)


def draw_divide_line_with_contour_on_step(fig, ax, train_sample, classificator, num_of_elem=10, offset=2, step=-1,
                                          is_need_plot_reduced_line=False, barrier=0.001):

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
    marks_reduce = []
    for i in range(num_of_elem):
        row = y[i]
        marks_line = []
        mark_reduce_line = []
        for j in range(num_of_elem):
            col = x[j]
            point = [col, row]
            mark = classificator.classify_func_on_step(x=point, step=step)
            marks_line.append(mark)

            if is_need_plot_reduced_line:
                mark_reduce = classificator.reduced_classify_func_on_step(x=point, step=step, barrier_percent=barrier)
                mark_reduce_line.append(mark_reduce)

        marks.append(marks_line)
        if is_need_plot_reduced_line:
            marks_reduce.append(mark_reduce_line)

    lev = [0]
    ax.contour(x, y, marks, levels=lev, colors='r')
    if is_need_plot_reduced_line:
        ax.contour(x, y, marks_reduce, levels=lev, colors='g')



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
    gen_flag = 0
    if gen_flag == 1:
        key = "circle"
        seed = 1
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
        seed = 1
        generator = generate_dict["shift_normal"]
        generator_args = {
            "shift": [shift, shift],
            "p": correlation,
            "alpha": 1 / 2,
            "mean": [0, 0]
        }

    train_sample_capacity = sample_capacity

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    train_sample = generator.generate(train_sample_capacity, generator_args)
    train_sample = Sample(sorted(train_sample.points, key=lambda value: value.mark))

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

    draw_divide_line_with_contour(train_sample, classificator)

    print("\n\n\n")

    accuracy = accuracy_test(test_sample, classificator)
    print("accuracy = %s" % format(accuracy, ""))
    draw_line_on_step(classificator, name="lllllll", num_of_elem=50, offset=2)

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
    is_need_save_coef_on_step = True
    if gen_flag == 1:
        key = "circle"
        line_on_step_gif_name = "gauss_line_on_step"
        coef_on_step_gif_name = "gauss_coef_on_step"
        seed = 1
        generator = generate_dict["circle"]
        generator_args = {
            "p": 0,
            "alpha": 1 / 2,
            "mean": [0, 0],
            "r_mean": 3,
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
        sigma_max = 5
        sigma_num_of_steps = 10
        sigma_arr = np.linspace(sigma_min, sigma_max, sigma_num_of_steps)

        kernel_name_arr = ["kernel_GAUSS"]

        first_cv_order = 6
        second_cv_order = 1
        train_percent = 0.8

        barrier_percent = 0.1
        is_need_plot_reduced_line = True
    else:
        key = "shift_normal"
        line_on_step_gif_name = "linear_line_on_step"
        coef_on_step_gif_name = "linear_coef_on_step"
        seed = 1
        generator = generate_dict["shift_normal"]
        generator_args = {
            "shift": [3, 3],
            "p": -1,
            "alpha": 1 / 2,
            "mean": [0, 0]
        }
        train_sample_capacity = 300
        test_sample_capacity = 1000

        ro = 1
        ny_coef = 1/2

        c_min_pow = -25
        c_max_pow = 10
        c_num_of_steps = 50
        c_pow_arr = np.linspace(c_min_pow, c_max_pow, c_num_of_steps)
        c_arr = [2 ** alpha for alpha in c_pow_arr]

        sigma_arr = [1]

        kernel_name_arr = ["kernel_LINEAR"]

        first_cv_order = 8
        second_cv_order = 1
        train_percent = 0.8

        barrier_percent = 0.1
        is_need_plot_reduced_line = True


    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    train_sample = generator.generate(train_sample_capacity, generator_args)

    best_accuracy, best_params, best_machine = greed_args_brute_force(train_sample, c_arr, sigma_arr, kernel_name_arr,
                                                                      ro=ro, first_cv_order=first_cv_order,
                                                                      second_cv_order=second_cv_order,
                                                                      ny_coef=ny_coef, train_percent=train_percent)

    fig, ax = plt.subplots(1, 1)
    train_sample.draw(fig, ax, marker=True)
    fig.show()
    plt.close(fig)
    if gen_flag == 1:
        print("best machine params: sigma = {}, c = {}".format(best_machine.kernel.sigma, best_machine.C))

    kernel = best_params["kernel"]
    lamda = best_params["lambda_var"]
    ro = best_params["ro"]
    ny = best_params["ny"]

    #
    #divide_ind = int(np.floor(train_sample.length() * train_percent))
    #random.shuffle(train_sample.points)
    #train_sample = Sample(train_sample.points[:divide_ind])


    # classificator = NORMA(train_sample)
    # classificator.learn(lamda, ro, kernel, ny)

    classificator = best_machine
    print("-------------------------------------------")
    print("Best Params")
    print("lambda = {}, C = {}, kernel={}, ro = {}, ny = {}".format(classificator.lambda_var, classificator.C,
                                                                    classificator.kernel, classificator.ro,
                                                                    classificator.ny_coef))
    print("-------------------------------------------")
    test_sample = generator.generate(test_sample_capacity, generator_args)

    test(train_sample, test_sample, classificator)

    print("\n\n\n")

    draw_divide_line_with_contour(train_sample, classificator, num_of_elem=50, offset=2,
                                  is_need_plot_reduced_line=is_need_plot_reduced_line, barrier=barrier_percent)

    print("\n\n\n")

    test_accuracy, test_accuracy_red = accuracy_test(test_sample, classificator)
    train_accuracy, train_accuracy_red = accuracy_test(train_sample, classificator, barrier=barrier_percent)
    count, count_red, amount = classificator.count_non_zero_coef(barrier_percent=barrier_percent)
    print("Classify func")
    print("Support vectors mun = {} (amount = {})".format(count, amount))
    print("test accuracy  = %s" % format(test_accuracy, ""))
    print("train accuracy = %s" % format(train_accuracy, ""))

    print()
    print("Reduced classify func")
    print("Support vectors mun = {} (amount = {})".format(count_red, amount))
    print("test accuracy  = %s" % format(test_accuracy_red, ""))
    print("train accuracy = %s" % format(train_accuracy_red, ""))

    classificator.show_b_on_step()
    classificator.show_coef_on_step()
    if is_need_save_coef_on_step:
        classificator.save_coef_on_step_as_gif(name=coef_on_step_gif_name)
    draw_line_on_step(classificator, name=line_on_step_gif_name, num_of_elem=50, offset=2,
                      is_need_plot_reduced_line=is_need_plot_reduced_line, barrier=barrier_percent)


if __name__ == "__main__":
    greed_auto_tuning()
    # main()




















