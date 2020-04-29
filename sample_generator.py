import numpy as np
import random as rand
import csv

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

    @staticmethod
    def is_table_empty(table):
        for row in table:
            for cell in row:
                if cell !=  "":
                    return False

        return True

    def write_sample_to_csv_file(self, file_csv_name="data.csv", keyword="keyword"):
        try:
            file_d = open(file_csv_name, mode='x')
            file_d.close()
            new_table = []
            table_len = 0
            row_len = 0
            start_col = 0
            old_size = 0
            prev_delim = ""
        except IOError:
            # already exist
            file_d = open(file_csv_name, mode='r')
            table = []
            new_table = []
            prev_delim = ""
            for row in csv.reader(file_d, delimiter=';'):
                table.append(row)
            file_d.close()
            # если таблица пустая, то можно ничего не искать
            if self.is_table_empty(table):
                new_table = []
                table_len = 0
                row_len = 0
                start_col = 0
                old_size = 0
            else:
                # найдём столбец с ключевым словом
                keyword_row = table[0]
                size_row = table[1]
                start_col = -1
                old_size = -1
                for i in range(len(keyword_row)):
                    value = keyword_row[i]
                    if value == keyword:
                        start_col = i
                        old_size = int(size_row[i])
                        break

                if start_col == -1:
                    start_col = len(keyword_row)
                    prev_delim = ";"
                    old_size = 0

                new_size = len(self.points[0].value) + 1

                row_len = len(table[0])
                table_len = len(table)
                for k in range(table_len):
                    row = table[k]
                    prev_data = ""
                    for i in range(start_col):
                        prev_data = prev_data + row[i] + ";"

                    next_data = ""
                    for i in range(start_col + old_size, row_len):
                        next_data = next_data + row[i] + ";"

                    if k == 0:
                        cur_data = keyword + ";" * new_size
                    elif k == 1:
                        cur_data = format(new_size, "") + ";" * new_size
                    elif k < self.size:
                        cur_data = ""
                        cur_point = self.points[k]
                        for i in range(len(cur_point.value)):
                            cur_data = cur_data + format(cur_point.value[i], "") + ";"
                        cur_data = cur_data + format(cur_point.mark, "") + ";"
                    else:
                        cur_data = ";" * new_size

                    new_row = prev_data + prev_delim + cur_data + next_data
                    new_table.append(new_row)

        new_size = len(self.points[0].value) + 1
        if table_len < self.size + 2:
            for k in range(table_len, self.size + 2):
                prev_data = ";" * start_col
                next_data = ";" * (row_len - (start_col + old_size) - 1)
                if k == 0:
                    cur_data = keyword + ";" * (new_size - 1)
                elif k == 1:
                    cur_data = format(new_size, "") + ";" * (new_size - 1)
                else:
                    cur_data = ""
                    cur_point = self.points[k-2]
                    for i in range(len(cur_point.value)):
                        cur_data = cur_data + format(cur_point.value[i], "") + ";"
                    cur_data = cur_data + format(cur_point.mark, "") + ";"

                new_row = prev_data + prev_delim + cur_data + next_data
                new_table.append(new_row)

        outfile = open(file_csv_name, 'w')
        for row in new_table:
            outfile.write(row)
            outfile.write("\n")
        outfile.close()

    def read_sample_from_csv_file(self, file_csv_name="data.csv", keyword="keyword"):
        file_d = open(file_csv_name, mode='r')
        table = []
        for row in csv.reader(file_d, delimiter=';'):
            table.append(row)
        file_d.close()

        keyword_row = table[0]
        size_row = table[1]
        start_col = -1
        new_size = -1
        for i in range(len(keyword_row)):
            value = keyword_row[i]
            if value == keyword:
                start_col = i
                new_size = int(size_row[i])
                break

        if start_col == -1:
            new_points = []
        else:
            new_points = []
            table_len = len(table)
            for k in range(table_len):
                row = table[k]
                if k == 0:
                    continue
                    # new_keyword = row[start_col]
                elif k == 1:
                    continue
                    # new_size = row[start_col]
                else:
                    if row[start_col] == "":
                        break
                    new_x = []
                    for x_ind in range(start_col, start_col + new_size - 1):
                        new_x.append(float(row[x_ind]))

                    y_ind = start_col + new_size - 1
                    new_y = float(row[y_ind])
                    new_point = Point(new_x, new_y)
                    new_points.append(new_point)
        self.points = new_points
        self.size = len(new_points)

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