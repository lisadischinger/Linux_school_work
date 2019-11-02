#!/usr/bin/env python

# Homework 3 for Learning Based controls
# Lisa Dischinger
# 10/24/19

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


class R_learning:
    def __init__(self):
        self.n_rows = 5
        self.n_columns = 10
        self.n_steps = 20                   # the maximum number of steps allowed

        self.state = [0, 0]                 # the current location of the agent

        self.goal = [3, 9]  # location of the end goal to begin with
        self.wall = []      # where the wall is
        for i in range(2, 5):
            self.wall.append([i, 7])

        # initialize the grid world optimisticly
        self.v_grid = np.full((self.n_rows, self.n_columns), 20)

    def reward_scheme(self, state):
        # ths is the reward structure for this world

        if state == self.goal:
            return 20.0
        elif state in self.wall:
            return -100.0
        else:
            return -1

    def state_for_action(self, action, state):
        # provide the state for the action desired; directions are swapped as 0,0 is in the upper left corner
        if action == "stay":
            n_state = state
        elif action == "up":
            n_state = [state[0]-1, state[1]]
        elif action == "down":
            n_state = [state[0]+1, state[1]]
        elif action == "right":
            n_state = [state[0], state[1]+1]
        elif action == "left":
            n_state = [state[0], state[1]-1]
        else:
            print("Action provided does not fit within our posibilities")

        # make sure that the state is still on the grid world
        if self.check_if_legal(n_state):
            return n_state
        return state        # the move desired takes us off the grid so just stay in place

    def check_if_legal(self, state):
        # check if we are able to be in the stated pose, if not return false
        if 0 < n_state[0] < self.rows and 0 < n_state[1] < self.columns:    # check that we are still in the grid world
            if not state in self.wall:                                      # make sure we are no in a wall
                return True
        return False

    def learn(self):
        # this is the main script for learning. Will be applying some weird algorithm...who knows!!!!!!!!!

        init_pose = False
        while not init_pose:
            # create a random inital state
            self.state = [random.randint(0, self.n_rows), random.randint(0, self.n_columns)]
            init_pose = self.check_if_legal(self.state)
        print("initial state = {}".format(self.state))
        action = "up"
        self.state = self.state_for_action(action, self.state)
        print(" the new location of the agent is {}".format(self.state))


if __name__ == "__main__":
    Agent0 = R_learning()
    Agent0.learn()


