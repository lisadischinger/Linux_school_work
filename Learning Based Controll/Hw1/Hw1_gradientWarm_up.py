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
import numpy as np
from math import e, exp
import matplotlib.pyplot as plt

x = []                      # inputs
y = []                    # found answer based off of weights and inputs
t = []                      # known answer

wm1 = np.ones(4)            # inital guesses of the weights, that are being directly multiplied by the inputs
ws = np.ones(4)             # inital guesses of the weights, that are being added into the act. functions
wm2 = np.ones(4)            # weights that are being multiplied with the Sfunction results
w0 = 1                      # weight to be added to the summation function

sum_list = []               # used to store values that will need to be summed up to create the guessed answer

def readTestData():
    global x, t
    with open('test1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            x.append(row[:2])              # inputs are the first two values of the row
            t.append(row[2:])               # inputs are the first two values of the row
        x = np.asarray(x)                   # transfer over to numpy arrays rather than lists
        t = np.asarray(t)


def sig_func(x, w, w_add):
    """ pass through a sigmoid function; this will be our activation function
    :param x: Input
    :param w: the weight that is being multiplied with the input
    :param w_add: the weight that is added to the product of xw to create z
    :return: s
    """
    z = (x * w) + w_add
    s = 1 / (1 + exp(-z))
    return s


def sum_up():
    """ sum up everything for this sample to create Y1 or Y2
    :return: sum fo the sum_list form this specific sample """
    global sum_list, w0
    sum = 0
    for item in sum_list:
        sum += item
    sum += w0
    sum_list = []               # clean out the sum_list
    return sum


if __name__ == "__main__":
    readTestData()
    for i in range(x.np.size):                 # to run through each data sample
        x_sub = [x[i][0], x[i][0], x[i][1], x[i][1]]        # translating to a array of 4 for eas of the next for loop
        for j in range(4):                  # to run through teh full set of weights
            r = sig_func(x_sub[j], wm1[j], ws[j])
            sum_list.append(r*wm2[j])
        sum_up()

