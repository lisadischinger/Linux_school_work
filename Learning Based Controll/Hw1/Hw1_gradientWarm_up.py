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
y = empty([1, y_n])                         # found output form the summation
yp = empty([y_n, 1])                        # 1-y matrix
S = empty([1, s_n])                         # the returns form the seigmond activation functions
Sp = empty([s_n, 1])                        # 1-S matrix
x = []                                      # data points
t = []                                      # training answers

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
    x_t = []                                            # test data, and answers
    t_t = []
    with open(title) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            x_t.append([float(row[0]), float(row[1])])              # inputs are the first two values of the row
            t_t.append([float(row[2]), float(row[3])])
        x_t = asarray(x_t)                   # transfer over to numpy arrays rather than lists
        t_t = asarray(t_t)

        return x_t, t_t


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


def testFeedForward(x, t):
    # given data pointspredict what the answer will be
    global e_list
    z1 = np.dot(x, w1) + b1
    h1 = 1.0 / (1 + np.exp(-z1))  # output of hidden layer
    z2 = np.dot(h1, w2) + b2  # sum of (hw2 + b_2)
    y = 1.0 / (1 + np.exp(-z2))  # predicted output

    return y


def clean():
    x_t = []
    t_t = []
    return x_t, t_t


def runItAll(hidden_units, lr, training_steps):
    global e_list
    w1 = np.random.rand(x.shape[1], hidden_units)
    w2 = np.random.rand(hidden_units, t.shape[1])

    b1 = np.random.rand(1, hidden_units)
    b2 = np.random.rand(1, t.shape[1])

    for epoch in range(training_steps):
        for i in range(100):
            # Feedforward
            # sum(xw + b)
            z1 = np.dot(x[i], w1) + b1

            # output of hidden layer
            h1 = 1.0 / (1 + np.exp(-z1))

            # sum of (hw2 + b_2)
            z2 = np.dot(h1, w2) + b2

            # predicted output
            y = 1.0 / (1 + np.exp(-z2))

            # Backpropagation
            # back it up!!!! backprop?
            e = t[i] - y
            error = np.linalg.norm(e)  # magnitude off the error, used to break the learning
            i_list.append(epoch * 100 + i)
            e_list.append(error)

            # delta_2 = C'(t, y) * sigma'(t)
            delta_2 = (y * (1 - y)) * e

            # delta_1 = w2 * delta_2 * sigma'(h)
            delta_1 = np.dot(delta_2, w2) * (h1 * (1 - h1))

            # updates weights
            w2 += lr * np.dot(h1.T, delta_2)
            w1 += lr * np.dot(x[i].reshape(1, 2).T, delta_1)

            # updates biases
            b2 += delta_2
            b1 += delta_1

    print('done')
    # plot the error over the time
    plt.plot(i_list, e_list)
    plt.title("Training data set")

    ##############################################
    # Run on Training Set 1
    ###############################################
    x_t, t_t = readTestData('test1.csv')
    for i in range(100):
        e_list = testFeedForward(x_t[i], t_t[i], w1, w2)
    print("The average error from T1 is: {}% ".format(round(mean(e_list) * 100), 3))
    x_t, t_t, e_list = clean()

    ###############################################
    # Run on Training Set 2
    ###############################################
    readTestData('test2.csv')
    for i in range(100):
        e_list = testFeedForward(x_t[i], t_t[i], w1, w2)
    print("The average error from T2 is: {}% ".format(round(mean(e_list) * 100), 3))
    x_t, t_t, e_list = clean()

    ###############################################
    # Run on Training Set 3
    ###############################################
    readTestData('test3.csv')
    for i in range(np.size(x_t, 0)):  # go through an epoch; through all data points
        e_list = testFeedForward(x_t[i], t_t[i], w1, w2)
    print("The average error from T3 is: {}% ".format(round(mean(e_list) * 100), 3))
    x_t, t_t, e_list = clean()

    plt.show()


def lmd_overlay3(x_data, plot_a, plot_a2, plot_a3, x_label, ya_label, labels, title):
    a = labels[0]
    b = labels[1]
    c = labels[2]
    # share x only
    ax1 = plt.subplot(311)
    plt.plot(x_data, plot_a, 'r', label=a)
    plt.plot(x_data, plot_a2, 'b', label=b)
    plt.plot(x_data, plot_a3, 'g', label=c)
    plt.ylabel(ya_label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    readTrainData()
    z = 0                                               # counts how many times we have looped through the epoch
    f_flag = False

    hidden_units = 4
    lr = 0.1
    training_steps = 1000

    w1 = np.random.rand(x.shape[1], hidden_units)
    w2 = np.random.rand(hidden_units, t.shape[1])

    b1 = np.random.rand(1, hidden_units)
    b2 = np.random.rand(1, t.shape[1])
    for epoch in range(1000):
        for i in range(100):
            # Feedforward
            # sum(xw + b)
            z1 = np.dot(x[i], w1) + b1

            # output of hidden layer
            h1 = 1.0 / (1 + np.exp(-z1))

            # sum of (hw2 + b_2)
            z2 = np.dot(h1, w2) + b2

            # predicted output
            y = 1.0 / (1 + np.exp(-z2))

            # Backpropagation
            # back it up!!!! backprop?
            e = t[i] - y
            error = np.linalg.norm(e)  # magnitude off the error, used to break the learning
            i_list.append(epoch*100 + i)
            e_list.append(error)

            # delta_2 = C'(t, y) * sigma'(t)
            delta_2 = (y * (1 - y)) * e

            # delta_1 = w2 * delta_2 * sigma'(h)
            delta_1 = np.dot(delta_2, w2.T) * (h1 * (1 - h1))         # used to not have w2 transposed  LMD

            # updates weights
            w2 += lr * np.dot(h1.T,   delta_2)
            w1 += lr * np.dot(x[i].reshape(1, 2).T, delta_1)

            # updates biases
            b2 += delta_2
            b1 += delta_1

        i_list.append(epoch * 100 + i)
        e_list.append(error)


    ##############################################
    # Run on Training Set 1
    ###############################################
    x_t, t_t = readTestData('test1.csv')
    count1 = 0
    count1_ls = []
    for i in range(100):
        y = testFeedForward(x_t[i], t_t[i])
        count1 += np.all(y.round() == t_t[i])
        count1_ls.append(count1/100.0)

    plt.plot(range(len(count1_ls)), count1_ls, label = "Test1")
    x_t, t_t = clean()

    ###############################################
    # Run on Training Set 2
    ###############################################
    x_t, t_t = readTestData('test2.csv')
    count2 = 0
    count2_ls = []
    for i in range(100):
        y = testFeedForward(x_t[i], t_t[i])
        count2 += np.all(y.round() == t_t[i])
        count2_ls.append(count2 / 100.0)

    plt.plot(range(len(count2_ls)), count2_ls, label = "Test2")
    x_t, t_t = clean()

    ###############################################
    # Run on Training Set 3
    ###############################################
    x_t, t_t = readTestData('test3.csv')
    count3 = 0
    count3_ls = []
    for i in range(100):
        y = testFeedForward(x_t[i], t_t[i])
        count3 += np.all(y.round() == t_t[i])
        count3_ls.append(count3 / 100.0)

    plt.plot(range(len(count3_ls)), count3_ls, label = "Test3")
    plt.title('Accuracy Comparison')
    plt.xlabel('Data point')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
