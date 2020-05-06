import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from sample_generator import Sample
from NORMA_SVM import NORMA
from kernels import kernel_LINEAR, kernel_GAUSS


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


def count_mark(sample):
    mark = +1
    first_amount = 0
    second_amount = 0
    for point in sample.points:
        if point.mark == mark:
            first_amount += 1
        else:
            second_amount += 1

    return first_amount, second_amount


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


def get_potential_best_params(potential_best_params, best_accuracy, percent_for_potential):
    result = []
    for curr_params in potential_best_params:
        curr_accuracy = curr_params["accuracy_mean"]
        if curr_accuracy >= percent_for_potential * best_accuracy:
            result.append(curr_params)
    return result


def greed_args_brute_force(sample):
    lambda_min = 0.1
    lambda_max = 50
    lambda_num_of_steps = 25
    lambda_arr = np.linspace(lambda_min, lambda_max, lambda_num_of_steps)

    ro_min = 1
    ro_max = 1
    ro_num_of_steps = 1
    ro_arr = np.linspace(ro_min, ro_max, ro_num_of_steps)

    sigma_min = 0.1
    sigma_max = 5
    sigma_num_of_steps = 5
    sigma_arr = np.linspace(sigma_min, sigma_max, sigma_num_of_steps)

    kernel_name_arr = ["kernel_GAUSS", "kernel_LINEAR"]

    train_percent = 0.6
    order_of_first_selection = 8
    order_of_second_selection = 8

    best_accuracy = -1
    tmp_potential_best_params = []
    percent_for_potential = 0.9

    for kernel_name in kernel_name_arr:
        if kernel_name == "kernel_GAUSS":
            tmp_sigma_arr = sigma_arr
        elif kernel_name == "kernel_LINEAR":
            tmp_sigma_arr = [1]
        else:
            tmp_sigma_arr = [1]
        accuracy_matrix = []
        for sigma in tmp_sigma_arr:
            accuracy_matrix_line = []
            if kernel_name == "kernel_GAUSS":
                kernel_param = {
                    "sigma": sigma,
                }
                kernel = kernel_GAUSS(kernel_param)
            else:
                kernel_param = {}
                kernel = kernel_LINEAR(kernel_param)

            for lambda_var in lambda_arr:
                for ro in ro_arr:
                    norma_param = {
                        "lambda_var": lambda_var,
                        "ro": ro,
                        "kernel": kernel,
                        "ny": 1/2
                    }

                    accuracy_mean, accuracy_var = cv_test(sample, norma_param, order_of_first_selection, train_percent)
                    accuracy_matrix_line.append(accuracy_mean)
                    print("accuracy_mean = %s" % format(accuracy_mean, ""), end="\t")
                    print("accuracy_var = %s" % format(accuracy_var, ""), end="\t")
                    print("kernel = %s" % format(kernel, ""), end="\t")
                    print("lambda_var = %s" % format(lambda_var, ""), end="\t")
                    print("ro = %s" % format(ro, ""), end="\t")
                    print()

                    if accuracy_mean >= percent_for_potential * best_accuracy:
                        tmp_node = {
                            "norma_param": norma_param,
                            "kernel_param":  kernel_param,
                            "accuracy_mean": accuracy_mean,
                            "accuracy_var": accuracy_var,
                        }
                        tmp_potential_best_params.append(tmp_node)
                        if accuracy_mean > best_accuracy:
                            best_accuracy = accuracy_mean

            accuracy_matrix.append(accuracy_matrix_line)

        ax = sns.heatmap(accuracy_matrix, vmin=0, vmax=1)
        plt.show()

    potential_best_params = get_potential_best_params(tmp_potential_best_params, best_accuracy, percent_for_potential)
    best_accuracy = -1
    best_norma_param = {}

    for node in potential_best_params:
        norma_param = node["norma_param"]
        old_accuracy = node["accuracy_mean"]
        accuracy_mean, accuracy_var = cv_test(sample, norma_param, order_of_second_selection, train_percent)
        print("curr_accuracy_mean = %s (%s)" % (format(accuracy_mean, ""), format(old_accuracy, "")))
        print("curr_norma_param = ", end="")
        for k, v in norma_param.items():
            print("\'%s\': %s" % (format(k, ""), format(v, "")), end=",   ")
        print()

        if accuracy_mean > best_accuracy:
            best_norma_param = norma_param
            best_accuracy = accuracy_mean
            best_accuracy_var = accuracy_var
    print("best_accuracy = %s" % format(best_accuracy, ""))
    print("best_accuracy_var = %s" % format(best_accuracy_var, ""))
    print("best_norma_param = ", end="")
    for k, v in best_norma_param.items():
        print("\'%s\': %s" % (format(k, ""), format(v, "")), end=",   ")
    print()
    return best_accuracy, best_norma_param
