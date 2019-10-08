#!/usr/bin/env python

# Homework 1 for Learning Based controls
# Lisa Dischinger
# 10/1/19

########################################################################################
# This script is to first learn from a single set of data and will apply
# the found weights to the next three sets of data. Inputs are x1, x2, known goals are
# t1, t2, while the output from the learning is known as y1, y2
# T1 is success and occurs when y1 = 0, and y2 = 1
# t2 is failure and occurs when y1 = 1 and y2 = 0
#########################################################################################

import csv
from numpy import array, identity, empty, random, add, linalg, asarray
import numpy as np
from math import e, exp
import matplotlib.pyplot as plt
from statistics import mean

# network constants
x_n = 2                             # number of inputs
y_n = 2                             # number of outputs
s_n = 2                             # number of hidden elements

e = np.empty([2, 1])                         # error; t - y
e_list = []                             # will be used to plot the progression of our error as the learning goes on
i_list = []
e1_list = []
e2_list = []
e3_list = []
# d = empty([2, 1])                         # back propogated deltas
y = empty([1, y_n])                         # found output form the summation
yp = empty([y_n, 1])                        # 1-y matrix
S = empty([1, s_n])                         # the returns form the seigmond activation functions
Sp = empty([s_n, 1])                        # 1-S matrix
x = []                                      # data points
t = []                                      # training answers
x_t = []                                    # test data, and answers
t_t = []

v = np.random.rand(y_n, s_n)                   # create an initial guess of the V weights [[v1, v3], [v2, v4]]
w = np.random.rand(s_n, x_n)                   # weights used between inputs and hidden layer
dw_sum = np.zeros([s_n, x_n])
dv_sum = np.zeros([y_n, s_n])                         # used to store deltas for updating the weights
b1 = random.rand(1, s_n)                   # biases used with the first sigmoid activation functions
b2 = random.rand(1, y_n)
b1_sum = 0
b2_sum = 0

step = 0.1                 # size of weight step
check = 0.25

update_rate = 10.0


def readTrainData():
    global x, t
    with open('test1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            x.append([float(row[0]), float(row[1])])              # inputs are the first two values of the row
            t.append([float(row[2]), float(row[3])])
        x = asarray(x)                   # transfer over to numpy arrays rather than lists
        t = asarray(t)


def readTestData(title):
    global x_t, t_t
    with open(title) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            x_t.append([float(row[0]), float(row[1])])              # inputs are the first two values of the row
            t_t.append([float(row[2]), float(row[3])])
        x_t = asarray(x_t)                   # transfer over to numpy arrays rather than lists
        t_t = asarray(t_t)


def sig_func(x, w, b):
    """ pass through a sigmoid function; this will be our activation function
    :param x: Input array, both X1 and X2
    :param w: the weight that is being multiplied with X1
    :param b: the bias added for this activation function
    :return: s
    """
    z = np.dot(x, np.transpose(w)) + b
    try:
        s = 1 / (1 + exp(-z))
    except OverflowError:
        s = float('inf')
    return s


def weight_average(list):
    # this assumes a list of weights of [2xn]x5 where n is the number of hidden elements
    lil_sum = 0
    average = []
    n = np.size(list[0], 1)                             # number of columns in one of the sub elements
    for i in range(5):                                  # go through each 2xn matrix
        list[i].flatten()                               # turn from 2xn matricies into 1x2n matricies
    for j in range(2*n):                                # through each sub element
        for i in range(5):                              # through each of the sets
            lil_sum += list[i][j]
        average.append(lil_sum / 5.0)
    np.reshape(average, [2, n])                         # pop it back to its original shape


def find_means():
    # clump and find the averages and reset all fo these lists
    global dw_sum, dv_sum , b1_sum, b2_sum, update_rate
    dw = np.true_divide(dw_sum, update_rate)  # take the average of the list
    dv = np.true_divide(dv_sum, update_rate)
    db1 = np.true_divide(b1_sum, update_rate)
    db2 = np.true_divide(b2_sum, update_rate)
    dw_sum = np.zeros([2, 2])  # clear ou the sums
    dv_sum = np.zeros([2, 2])
    b1_sum = 0
    b2_sum = 0
    return dw, dv, db1, db2


def zerofy(m):
    # makes everything not on the diagonals zero; assumes a square matrix
    for i in range(np.size(m, 0)):            # for each row
        for j in range(np.size(m, 1)):         # for each column
            if not i == j:              # diagonals will have the same indices
                m[i, j] = 0
    return m


if __name__ == "__main__":
    readTrainData()
    z = 0                                               # counts how many times we have looped through the epoch
    f_flag = False
    # print("number of rows: ", np.size(x, 0))
    while not f_flag:
        for i in range(np.size(x, 0)):                          # go through an epoch; through all data points
            # feed forward calculation
            for j in range(2):                                  # to run through teh full set of weights
                S[0, j] = sig_func(x[i], w[j], b1[0, j])        # i refers to data line, j refers to what goes with S1
            Sp = np.reshape(1 - S, (2, 1))
            for k in range(2):                                  # to run through the second activation function
                y[0, k] = sig_func(S, v[k], b2[0, k])
            yp = np.reshape(1 - y, (2, 1))                  # y is a 1x2 matrix; yp is a 2x1 matrix

            # find e
            e = np.reshape(t[i] - y, (2, 1))                     # e is a 2x1 matrix
            error = np.linalg.norm(e)                            # magnitude off the error, used to break the learning
            # i_list.append(z*100 + i)
            # e_list.append(error)                   # store this data to be plotted later on
            if error < check:
                f_flag = True

            # add the sudo-updates to be averaged later; update the weights between the S1 and the output activation
            sub = zerofy(e*y)                                   # let only the diagonals keep their data
            delta_2 = np.dot(sub, yp)                           # delta between S1 and the output
            dv_sum = np.add(dv_sum, step*(delta_2*S))           # dv = step*e*y*(1-y)*S
            b2_sum = np.add(b2_sum, step*delta_2)

            # Find delta
            # a = np.dot(w, delta_2) * S                                # sum(w*delta_2) * S
            set1 = np.dot(w, delta_2)                                   # gets half of the sums
            set2 = np.dot(w, np.flip(delta_2))
            a = add(set1, set2) * S
            delta_1 = np.dot(a, Sp)                                   # delta between input and S1
            dw_sum = np.add(dw_sum, step*(delta_1*x[i]))
            b1_sum = np.add(b1_sum, step*delta_1)

            # update the weights and biases every certain amount of time
            if i and (i % update_rate == 0 or i == 99):
                dw, dv, db1, db2 = find_means()
                v = np.add(v, dv)
                b2 = np.add(b2, np.transpose(db2))
                w = np.add(w, dw)
                b1 = np.add(b1, np.transpose(db1))
                i_list.append(z*100 + i)
                e_list.append(error)                   # store this data to be plotted later on    # print("i: ", i)
                print("error: ", round(error, 3))

        z += 1                                                    # increase the count of epochs gone through
        print(z)

    # plot the error over the time
    fig, ax = plt.subplots()
    ax.plot(i_list, e_list)
    plt.title("Training data set")

    ###############################################
    # Run on Training Set 1
    ###############################################
    readTestData('test1.csv')
    for i in range(np.size(x_t, 0)):  # go through an epoch; through all data points
        # feed forward calculation
        for j in range(2):  # to run through teh full set of weights
            S[0, j] = sig_func(x_t[i], w[j], b1[0, j])  # i refers to data line, j refers to what goes with S1
        for k in range(2):  # to run through the second activation function
            y[0, k] = sig_func(S, v[k], b2[0, k])

        # find e
        e = t_t[i] - y  # e is a 1x2 matrix
        e1_list.append(linalg.norm(e))  # store this data to be plotted later on
    print("The average error from T1 is: {}% ".format(round(mean(e1_list)*100), 3))
    x_t = []
    t_t = []

    ###############################################
    # Run on Training Set 2
    ###############################################
    readTestData('test2.csv')
    for i in range(np.size(x_t, 0)):  # go through an epoch; through all data points
        # feed forward calculation
        for j in range(2):  # to run through teh full set of weights
            S[0, j] = sig_func(x_t[i], w[j], b1[0, j])  # i refers to data line, j refers to what goes with S1
        for k in range(2):  # to run through the second activation function
            y[0, k] = sig_func(S, v[k], b2[0, k])

        # find e
        e = t_t[i] - y  # e is a 1x2 matrix
        e2_list.append(linalg.norm(e))  # store this data to be plotted later on
    print("The average error from T2 is: {}% ".format(round(mean(e2_list)*100), 3))
    x_t = []
    t_t = []

    ###############################################
    # Run on Training Set 3
    ###############################################
    readTestData('test3.csv')
    for i in range(np.size(x_t, 0)):  # go through an epoch; through all data points
        # feed forward calculation
        for j in range(2):  # to run through teh full set of weights
            S[0, j] = sig_func(x_t[i], w[j], b1[0, j])  # i refers to data line, j refers to what goes with S1
        for k in range(2):  # to run through the second activation function
            y[0, k] = sig_func(S, v[k], b2[0, k])

        # find e
        e = t_t[i] - y                          # e is a 1x2 matrix
        e3_list.append(linalg.norm(e))                   # store this data to be plotted later on
    print("The average error from T3 is: {}% ".format(round(mean(e3_list)*100), 3))
    x_t = []
    t_t = []

    plt.show()
    ref = 47
