#!/usr/bin/env python

# Homework 1 for Learning Based controls
# Lisa Dischinger
# 10/10/19

########################################################################################
# Testing out different search and optimization algorithms on the traveling salesman problem
# will implement Simulater annealing, evolution, and population based search algorithms
# on various sized TSP problems
# first create the city distance matrix for each city config
# experiments will track the run time, solution quality and repeatability for each approach
#  along with how many solutions the algorithms generated
#########################################################################################

import csv
from numpy import random
import numpy as np
import math
import random
from collections import Counter
import time
import matplotlib.pyplot as plt
from statistics import mean, stdev

class TSM:
    def __init__(self, file_name):
        self.p = []                    # array that hold all of the city coordinates
        self.init = True                # used to mark the initial run of something

        self.n = 0  # number of cities
        self.get_data(file_name)        # gather data from CSV
        self.dist_matrix()              # create the distance matrix

        self.S0 = []                    # current best solution list; specifies which cities to go to
        self.d0 = 0                     # objective answer with this solution

        # variables for simulated annealing
        self.P1_SA = []                     # will gather data over multiple runs, problem 1 with simulated annealing
        self.P1_SA_dt = []
        self.Tmax = 2000
        self.Tmin = 1.25
        self.Tfactor = -math.log(self.Tmax / self.Tmin)         # exponential cooling
        self.max_cycles = 1000                                # max number of times we will run te annealling algorithm

        # variables for evolution algorithm
        self.P1_EV = []  # will gather data over multiple runs, problem 1 with simulated annealing
        self.P1_EV_dt = []
        self.n_k = 15                                           # number of solutions looking at
        self.dtype = [('Solution', list), ('cost', float)]
        self.k = np.empty([1, self.n_k], dtype=self.dtype)         # list of solutions, [solution, value]
        self.k_temp = np.empty([1, self.n_k], dtype=self.dtype)    # temporary during generation of mutants, will append onto
        self.k_reject = []                                  # store rejected past solutions
        self.n_swap = 1                                         # number of times to swap cities in a mutation

        # variables for the beam method
        self.P1_BM = []                     # will gather data over multiple runs, problem 1 with simulated annealing
        self.P1_BM_dt = []
        self.n_bm = self.n                                               # number  of initial trees to create

    def get_data(self, file_name):
        # read file and save as a numpy array
        with open(file_name) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                self.p.append(row)
            self.p = np.asarray(self.p, dtype=np.float32)                      # translate to a numpy array
            self.n = np.size(self.p, 0)                                         # number of cities

    def dist_matrix(self):

        self.d_mat = np.empty([self.n, self.n])
        for i in range(self.n):
            for j in range(self.n):
                self.d_mat[i, j] = distance(self.p[i], self.p[j])

    def sol_value(self, S):
        # find the sum of the distance traveled specified by the potential solution, this is the value that will
        # describe how good of a solution it is, the higher the number the worse and more ineffiecent the solution is
        sum = 0
        for i in range(len(S) - 1):             # go by index
            d = self.d_mat[S[i], S[i+1]]        # look distance up in the distance_matrix
            sum += d
        d = self.d_mat[S[0], S[-1]]             # The salesman returns back to the first city, add this to the distance
        sum += d
        return sum

    def sim_annealing(self, runs):
        # prime hub for the simulated annealing algorithm

        for i in range(runs):
            start_time = time.time()
            # create two initial random solutions
            self.S0 = random.sample(range(0, self.n), self.n)
            self.d0 = self.sol_value(self.S0)

            for cycle in range(self.max_cycles):
                S1 = random.sample(range(0, self.n), self.n)
                d1 = self.sol_value(S1)
                dE = abs(d1 - self.d0)

                # compare and decide the victor
                T = self.Tmax * math.exp(self.Tfactor*cycle / self.max_cycles)
                if d1 < self.d0 and math.exp(-dE / T) > random.random():       # Transfer the thrown
                    self.S0 = S1
                    self.d0 = d1

            dt = time.time() - start_time
            self.P1_SA_dt.append(dt)
            self.P1_SA.append(self.d0)

        # Find the mean and standard deviation
        self.P1_SA_dt = [mean(self.P1_SA_dt), stdev(self.P1_SA_dt)]
        self.P1_SA = [mean(self.P1_SA), stdev(self.P1_SA)]

        print(" Simulated Annealing with {} cities:".format(self.n))
        print("the run time mean and stdev is {} and {}".format(round(self.P1_SA_dt[0], 4), round(self.P1_SA_dt[1], 4)))
        print("the cost mean and stdev is {} and {}".format(round(self.P1_SA[0], 4), round(self.P1_SA[1], 4)))

        return self.S0, self.d0

    def sim_evolution(self, runs):
        # apply an evolution algorithm; start with a random guess, create random mutated solutions based off of that
            # guess, then keep half of them to go on to be further mutated

        for i in range(runs):
            start_time = time.time()
            # create n_k initial random solutions
            for i in range(self.n_k):                            # create the k numbered initial random solutions
                key_list = random.sample(range(0, self.n), self.n)
                value = self.sol_value(key_list)
                self.k[0, i] = key_list, value

            for cycle in range(self.max_cycles):                        # run for a certain number of cycles
                for i in range(self.n_k):                       # mutate each current solution in set
                    j = 0
                    while j < self.n_swap:       # loop for how many times we want to swap cities for a single mutation
                        a = random.randint(0, self.n-1)
                        b = random.randint(0, self.n-1)
                        while b == a:                           # make sure we are actually swapping two spots
                            b = random.randint(0, self.n - 1)
                        p_sol = self.k[0, i][0]
                        val = np.copy(p_sol)                       # copy to ensure that we dont mess with the original
                        val[a], val[b] = p_sol[b], p_sol[a]        # swap the terms
                        # check to see if we already have this as a solution
                        test_k = []
                        for k in range(self.n_k):
                            test_k.append(str(self.k[0, k][0]))
                        yup = str(list(val))
                        if np.isin(yup, test_k) or np.isin(yup, self.k_reject):
                            j -= 1                                 # give us another chance to calculate a new solution
                        else:
                            temp_val = self.sol_value(val)
                            self.k_temp[0, i] = list(val), temp_val
                            j += 1

            self.k = np.append(self.k, self.k_temp, axis=1)                    # add to the parent dict.
            self.k_temp = np.empty([1, self.n_k], dtype=self.dtype)                # clear out the temp. one

            sol_sorted = np.sort(self.k, order='cost')                         # sort based off of the cost, lowest will be up front
            self.k = sol_sorted[0, 0:self.n_k].reshape((1, self.n_k))
            for i in range(np.size(sol_sorted, 1)):
                self.k_reject.append(str(sol_sorted[0, i][0]))

            dt = time.time() - start_time
            self.P1_EV_dt.append(dt)
            self.P1_EV.append(self.k[0, 0][1])

        # Find the mean and standard deviation
        self.P1_EV_dt = [mean(self.P1_EV_dt), stdev(self.P1_EV_dt)]
        self.P1_EV = [mean(self.P1_EV), stdev(self.P1_EV)]

        print("")
        print("Evolution with {} cities:".format(self.n))
        print("the mean and standard deviation for run time is {} and {}".format(round(self.P1_EV_dt[0], 4),
                                                                                 round(self.P1_EV_dt[1], 4)))
        print("the mean and standard deviation for solution cost is {} and {}".format(round(self.P1_EV[0], 4),
                                                                                          round(self.P1_EV[1], 4)))

        return self.k[0, 0][0], self.k[0, 0][1]     # ev_sol, ev_val

    def beam_method(self, runs):
        # will use the beam branch method
        num_list = np.arange(self.n)                              # create a list for all the cities
        for i in range(runs):
            start_time = time.time()
            parent_list = np.random.choice(num_list, self.n_bm, replace=False)      # list of the initial trucks of the trees
            parent_champs = np.empty([1, self.n_bm], dtype=self.dtype)
            # follow each tree to its completion
            j = 0
            for parent in parent_list:
                index = np.where(num_list == parent)
                self.sub_array = np.delete(num_list, np.where(num_list == parent))        # take the parent out of the list of possible next steps
                self.parent = [parent]
                for i in range(1, self.n):                                      # this is how many branch offshoots we have to deal with
                    child = np.empty([1, self.n-i], dtype=object)
                    child, self.sub_array = self.next_gen(self.parent, child, self.sub_array)   # create all the possible offspring
                    self.parent, cost = self.find_best(child, i)                         # now force them to compete for your affection
                parent_champs[0, j] = list(self.parent), cost
                j += 1

            sol_sorted = np.sort(parent_champs, order='cost')
            dt = time.time() - start_time
            self.P1_BM_dt.append(dt)
            self.P1_BM.append(sol_sorted[0, 0][1])

        # Find the mean and standard deviation
        self.P1_BM_dt = [mean(self.P1_BM_dt), stdev(self.P1_BM_dt)]
        self.P1_BM = [mean(self.P1_BM), stdev(self.P1_BM)]
        print("")
        print("Beam Method with {} cities:".format(self.n))
        print("the mean and standard deviation for run time is {} and {}".format(round(self.P1_BM_dt[0], 4),
                                                                                 round(self.P1_BM_dt[1], 4)))
        print("the mean and standard deviation for solution cost is {} and {}".format(round(self.P1_BM[0], 4),
                                                                                          round(self.P1_BM[1], 4)))


    def find_best(self, array, i):
        # used with the beam method to find the cheapest of the children branches
        costs = []
        for j in range(np.size(array, axis=1)):
            child = array[0, j]
            if i == self.n-1:                         # for the last round we need to include the cost of returning
                child = np.append(child, [child[0]])
            cost = self.sol_value(child)                # find the cost for that specific branch
            costs.append(cost)
        minPos = costs.index(min(costs))                # the index of the lowest cost will also be the index of the branch we want
        return array[0, minPos], min(costs)

    def next_gen(self, parent, child, sub_array):
        # used within the beam method for finding the next branch offspring
        if len(parent) != 1:
            reject_letter = parent[-1]                                  # figure out what the most recent addon has been
            reject_index = np.where(sub_array == reject_letter)         # find its index so that we can take it out of the subarray
            sub_array = np.delete(sub_array, reject_index)
        j = 0
        for addon in sub_array:                                     # go through the remaining possibilities
            child[0, j] = parent + [addon]                    # add one of the remaining possibilities to create a branch
            j += 1

        return child, sub_array


def distance(p1, p2):
    # finds the distance between two points
    d = math.sqrt(pow((p1[0]-p2[0]), 2) + pow((p1[1]-p2[1]), 2))
    return d


if __name__ == "__main__":
    TSM_15 = TSM('15cities.csv')
    TSM_25 = TSM('25cities.csv')
    TSM_100 = TSM('100cities.csv')
    TSM_25a = TSM('25cities_a.csv')

    # For Problem 1 with 15 cities
    anneal_solution, anneal_val = TSM_15.sim_annealing(12)
    ev_sol, ev_val = TSM_15.sim_evolution(12)
    TSM_15.beam_method(12)

    # For Problem 2 with 25 cities
    print(" ")
    anneal_solution, anneal_val = TSM_25.sim_annealing(12)
    ev_sol, ev_val = TSM_25.sim_evolution(12)
    TSM_25.beam_method(12)
    #
    # # For Problem 2 with 100 cities
    print(" ")
    anneal_solution, anneal_val = TSM_100.sim_annealing(12)
    ev_sol, ev_val = TSM_100.sim_evolution(12)
    TSM_100.beam_method(12)

