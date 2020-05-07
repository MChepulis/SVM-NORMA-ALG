import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

from functions import print_dict
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
    return accuracy, classificator


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
    local_best_accur = 0
    best_classificator = None
    for i in range(order):
        points = sample.points.copy()
        random.shuffle(points)

        train_sample = Sample(points[:divide_ind])
        test_sample = Sample(points[divide_ind:])

        curr_accuracy, classificator = cross_validation(train_sample, test_sample, norma_param)
        if curr_accuracy > local_best_accur:
            local_best_accur = curr_accuracy
            best_classificator = classificator

        accuracy_arr.append(curr_accuracy)

    accuracy_mean = np.mean(accuracy_arr)
    accuracy_var = np.var(accuracy_arr)
    return accuracy_mean, accuracy_var, best_classificator


def get_potential_best_params(potential_best_params, best_accuracy, percent_for_potential):
    result = []
    for curr_params in potential_best_params:
        curr_accuracy = curr_params["accuracy_mean"]
        if curr_accuracy >= percent_for_potential * best_accuracy:
            result.append(curr_params)
    return result


def greed_args_brute_force(sample, c_arr, sigma_arr, kernel_name_arr,  ro=1, ny_coef=1/2, train_percent=0.6,
                           first_cv_order=8, second_cv_order=8, show_heat_map=True):
    best_accuracy = -1
    best_machine = None
    tmp_potential_best_params = []
    percent_for_potential = 0.9
    length = int(np.floor(sample.length() * train_percent))
    lambda_arr = [1 / (2 * c * length) for c in c_arr]
    for kernel_name in kernel_name_arr:
        accuracy_matrix = []
        for sigma in sigma_arr:
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
                norma_param = {
                    "lambda_var": lambda_var,
                    "ro": ro,
                    "kernel": kernel,
                    "ny": ny_coef
                }

                accuracy_mean, accuracy_var, machine = cv_test(sample, norma_param, first_cv_order, train_percent)
                accuracy_matrix_line.append(accuracy_mean)
                print("best_accur = %s" % format(best_accuracy, ""), end="\t")
                print("accuracy_mean = %s" % format(accuracy_mean, ""), end="\t")
                print("accuracy_var = %s" % format(accuracy_var, ""), end="\t")
                print("kernel = %s" % format(kernel, ""), end="\t")
                print("lambda_var = %s" % format(lambda_var, ""), end="\t")
                print("ro = %s" % format(ro, ""), end="\t")
                print()

                if accuracy_mean >= percent_for_potential * best_accuracy:
                    print("-----------------------------------------------")
                    tmp_node = {
                        "norma_param": norma_param,
                        "kernel_param": kernel_param,
                        "accuracy_mean": accuracy_mean,
                        "accuracy_var": accuracy_var,
                    }
                    tmp_potential_best_params.append(tmp_node)
                    if accuracy_mean > best_accuracy:
                        print("**********************************************")
                        best_accuracy = accuracy_mean
                        best_machine = machine

            accuracy_matrix.append(accuracy_matrix_line)
        if show_heat_map:
            xticklabels = ['{:.5f}'.format(value) for value in lambda_arr]
            yticklabels = ['{:.5f}'.format(value) for value in sigma_arr]
            sns.heatmap(accuracy_matrix, xticklabels=xticklabels, yticklabels=yticklabels, vmin=0, vmax=1)
            plt.show()

    potential_best_params = get_potential_best_params(tmp_potential_best_params, best_accuracy, percent_for_potential)
    best_accuracy = -1
    best_norma_param = {}

    for node in potential_best_params:
        norma_param = node["norma_param"]
        old_accuracy = node["accuracy_mean"]
        old_accuracy_var = node["accuracy_var"]
        accuracy_mean, accuracy_var, machine = cv_test(sample, norma_param, second_cv_order, train_percent)
        print("curr_accuracy_mean = %s (%s)" % (format(accuracy_mean, ""), format(old_accuracy, "")))
        print("curr_accuracy_var  = %s (%s)" % (format(accuracy_var, ""), format(old_accuracy_var, "")))
        print("curr_norma_param = ", end="")
        print_dict(norma_param)
        print()

        if accuracy_mean > best_accuracy:
            best_norma_param = norma_param
            best_accuracy = accuracy_mean
            best_accuracy_var = accuracy_var
    print("best_accuracy = %s" % format(best_accuracy, ""))
    print("best_accuracy_var = %s" % format(best_accuracy_var, ""))
    print("best_norma_param = ", end="")
    print_dict(best_norma_param)
    print()
    return best_accuracy, best_norma_param, best_machine
