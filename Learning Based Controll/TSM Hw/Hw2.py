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
        self.dtype = [('Solution', list), ('cost', float)]

        self.current_sol = []                    # current best solution list; specifies which cities to go to
        self.current_cost= 0                     # objective answer with this solution

        # variables for simulated annealing
        self.P1_SA = []                     # will gather data over multiple runs, problem 1 with simulated annealing
        self.P1_SA_dt = []
        self.T = 100
        self.T_stop = 0.01
        self.T_step = 0.999
        self.SA_best = np.empty([1, 1], dtype=self.dtype)                       # keep track of the best solution found
        self.SA_best[0, 0] = random.sample(range(0, 5), 5), 1000          # just initialization

        # variables for evolution algorithm
        self.P1_EV = []  # will gather data over multiple runs, problem 1 with simulated annealing
        self.P1_EV_dt = []
        self.n_k = int(0.5 * self.n)                                # number of solutions looking at
        self.k = np.empty([1, self.n_k], dtype=self.dtype)          # list of solutions, [solution, value]
        self.k_temp = np.empty([1, self.n_k], dtype=self.dtype)     # temporary during generation of mutants, will append onto
        self.k_reject = []                                          # store rejected past solutions
        self.n_swap = 1                                             # number of times to swap cities in a mutation
        self.EV_best = np.empty([1, 1], dtype=self.dtype)           # keep track of the best solution found
        self.EV_best[0, 0] = random.sample(range(0, 5), 5), 1000    # just initialization

        # variables for the beam method
        self.P1_BM = []                     # will gather data over multiple runs, problem 1 with simulated annealing
        self.P1_BM_dt = []
        self.n_bm = self.n                                               # number  of initial trees to create
        self.BM_sols_created = 0
        self.BM_best = np.empty([1, 1], dtype=self.dtype)                   # keep track of the best solution found
        self.BM_best[0, 0] = random.sample(range(0, 5), 5), 1000            # just initialization

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
        num_sols = 1
        for i in range(runs):
            start_time = time.time()
            # create two initial random solutions
            self.current_sol = random.sample(range(0, self.n), self.n)
            self.current_cost = self.sol_value(self.current_sol)

            cost = []

            while self.T > self.T_stop:
                new_sol = random.sample(range(0, self.n), self.n)   # generate random solution
                new_cost = self.sol_value(new_sol)

                cost_delta = self.current_cost - new_cost           # calculate loss
                probability = math.exp(cost_delta / self.T)         # determine acceptance probability

                # accept new solution if its better or by probability
                if cost_delta > 0 or probability > random.random():
                    self.current_sol = new_sol
                    self.current_cost = new_cost

                cost.append(self.current_cost)                      # append new solution to cost list

                self.T *= self.T_step                               # update temperature
                num_sols += 1

            dt = time.time() - start_time
            self.P1_SA_dt.append(dt)
            self.P1_SA.append(self.current_cost)

            if self.current_cost < self.SA_best[0, 0][1]:             # update what has been the best solution over all the runs
                self.SA_best[0, 0] = self.current_sol, self.current_cost

        # Find the mean and standard deviation
        self.P1_SA_dt = [mean(self.P1_SA_dt), stdev(self.P1_SA_dt)]
        self.P1_SA = [mean(self.P1_SA), stdev(self.P1_SA)]

        print(" Simulated Annealing with {} cities:".format(self.n))
        print("the run time mean and stdev is {} and {}".format(round(self.P1_SA_dt[0], 4), round(self.P1_SA_dt[1], 4)))
        print("the cost mean and stdev is {} and {}".format(round(self.P1_SA[0], 4), round(self.P1_SA[1], 4)))
        print("Number of solutions created: ", num_sols)

        return self.SA_best

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

            num_sols = self.n_k
            for cycle in range(1000):                        # run for a certain number of cycles
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

                        num_sols += 1                                # creates child == created solution

            self.k = np.append(self.k, self.k_temp, axis=1)                    # add to the parent dict.
            self.k_temp = np.empty([1, self.n_k], dtype=self.dtype)                # clear out the temp. one

            sol_sorted = np.sort(self.k, order='cost')                         # sort based off of the cost, lowest will be up front
            self.k = sol_sorted[0, 0:self.n_k].reshape((1, self.n_k))
            for i in range(np.size(sol_sorted, 1)):
                self.k_reject.append(str(sol_sorted[0, i][0]))

            dt = time.time() - start_time
            self.P1_EV_dt.append(dt)
            self.P1_EV.append(self.k[0, 0][1])

            if self.k[0, 0][1] < self.EV_best[0, 0][1]:      # update what has been the best solution over all the runs
                self.EV_best[0, 0] = self.k[0, 0]

        # Find the mean and standard deviation
        self.P1_EV_dt = [mean(self.P1_EV_dt), stdev(self.P1_EV_dt)]
        self.P1_EV = [mean(self.P1_EV), stdev(self.P1_EV)]

        print("")
        print("Evolution with {} cities:".format(self.n))
        print("the mean and standard deviation for run time is {} and {}".format(round(self.P1_EV_dt[0], 4),
                                                                                 round(self.P1_EV_dt[1], 4)))
        print("the mean and standard deviation for solution cost is {} and {}".format(round(self.P1_EV[0], 4),
                                                                                          round(self.P1_EV[1], 4)))
        print("Number of solutions created: ", num_sols)
        return self.EV_best     # ev_sol, ev_val

    def beam_method(self, runs):
        # will use the beam branch method
        num_list = np.arange(self.n)                              # create a list for all the cities
        for i in range(runs):
            start_time = time.time()
            parent_list = np.random.choice(num_list, self.n_bm, replace=False)  # list of the initial trucks of the trees
            parent_champs = np.empty([1, self.n_bm], dtype=self.dtype)
            # follow each tree to its completion
            j = 0
            num_sols = self.n_bm                                        # start off with number of parents
            for parent in parent_list:
                index = np.where(num_list == parent)
                self.sub_array = np.delete(num_list, np.where(num_list == parent))        # take the parent out of the list of possible next steps
                self.parent = [parent]
                for i in range(1, self.n):                                      # this is how many branch offshoots we have to deal with
                    child = np.empty([1, self.n-i], dtype=object)
                    child, self.sub_array = self.next_gen(self.parent, child, self.sub_array)   # create all the possible offspring
                    num_sols += np.size(child, 1)                              # each mini branch counts as another solution
                    self.parent, cost = self.find_best(child, i)                         # now force them to compete for your affection
                parent_champs[0, j] = list(self.parent), cost
                j += 1

            sol_sorted = np.sort(parent_champs, order='cost')
            dt = time.time() - start_time
            self.P1_BM_dt.append(dt)
            self.P1_BM.append(sol_sorted[0, 0][1])

            if sol_sorted[0, 0][1] < self.BM_best[0, 0][1]:   # update what has been the best solution over all the runs
                self.BM_best[0, 0] = sol_sorted[0, 0]

        # Find the mean and standard deviation
        self.P1_BM_dt = [mean(self.P1_BM_dt), stdev(self.P1_BM_dt)]
        self.P1_BM = [mean(self.P1_BM), stdev(self.P1_BM)]
        print("")
        print("Beam Method with {} cities:".format(self.n))
        print("the mean and standard deviation for run time is {} and {}".format(round(self.P1_BM_dt[0], 4),
                                                                                 round(self.P1_BM_dt[1], 4)))
        print("the mean and standard deviation for solution cost is {} and {}".format(round(self.P1_BM[0], 4),
                                                                                          round(self.P1_BM[1], 4)))
        print("Number of solutions created: ", num_sols)
        return self.BM_best

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
            self.BM_sols_created += 1                           # I am defining that child = solution

        return child, sub_array


def distance(p1, p2):
    d = math.sqrt(pow((p1[0]-p2[0]), 2) + pow((p1[1]-p2[1]), 2))        # finds the distance between two points
    return d


def plot_paths(cities, SA_best, EV_best, BM_best):
    # plot the path taken by the salesman for each method
    SA_list = []
    EV_list = []
    BM_list = []
    init = True
    for i in range(np.size(cities, 0)):             # go through all city spots
        if init:
            SA_list = cities[SA_best[0, 0][0][i]]
            EV_list = cities[EV_best[0, 0][0][i]]
            BM_list = cities[BM_best[0, 0][0][i]]
            init = False
        else:
            SA_list = np.row_stack((SA_list, cities[SA_best[0, 0][0][i]]))
            EV_list = np.row_stack((EV_list, cities[EV_best[0, 0][0][i]]))
            BM_list = np.row_stack((BM_list, cities[BM_best[0, 0][0][i]]))

    # three subplot
    plt.suptitle("Best Path found for {} cities".format(np.size(cities, 0)))
    ax1 = plt.subplot(311)
    plt.plot(SA_list[:, 0], SA_list[:, 1], '-o')
    ax1.title.set_text("Simulated Annealing")
    plt.setp(ax1.get_xticklabels(), fontsize=2)

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(EV_list[:, 0], EV_list[:, 1], '-o')
    ax2.title.set_text("Evolution Algorithm")
    plt.setp(ax2.get_xticklabels(), visible=False)  # make these tick labels invisible

    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(BM_list[:, 0], BM_list[:, 1], '-o')
    ax3.title.set_text("Beam Search")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    TSM_15 = TSM('15cities.csv')
    TSM_25 = TSM('25cities.csv')
    TSM_100 = TSM('100cities.csv')
    TSM_25a = TSM('25cities_a.csv')

    n_runs = 12                      # for repeatability testing; how many times do you want to run each method

    # # For Problem 1 with 15 cities
    SA_best_15 = TSM_15.sim_annealing(n_runs)
    EV_best_15 = TSM_15.sim_evolution(n_runs)
    BM_best_15 = TSM_15.beam_method(n_runs)

    # For Problem 2 with 25 cities
    print(" ")
    SA_best_25 = TSM_25.sim_annealing(n_runs)
    EV_best_25 = TSM_25.sim_evolution(n_runs)
    BM_best_25 = TSM_25.beam_method(n_runs)

    # # For Problem 2 with 100 cities
    print(" ")
    SA_best_100 = TSM_100.sim_annealing(n_runs)
    EV_best_100 = TSM_100.sim_evolution(n_runs)
    BM_best_100 = TSM_100.beam_method(n_runs)

    # for problem 3; comparing the two sets of 25 cities
    print("For 25 A")
    SA_best_25a = TSM_25a.sim_annealing(n_runs)
    EV_best_25a = TSM_25a.sim_evolution(n_runs)
    BM_best_25a = TSM_25a.beam_method(n_runs)

    # plot stuff
    plot_paths(TSM_15.p, SA_best_15, EV_best_15, BM_best_15)
    plot_paths(TSM_25.p, SA_best_25, EV_best_25, BM_best_25)
    plot_paths(TSM_25a.p, SA_best_25a, EV_best_25a, BM_best_25a)
    plot_paths(TSM_100.p, SA_best_100, EV_best_100, BM_best_100)

