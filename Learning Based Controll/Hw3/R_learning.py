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

from numpy import random
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


class R_learning:
    def __init__(self):
        self.n_rows = 5
        self.n_columns = 10
        self.n_steps = 20                   # the maximum number of steps allowed
        self.n_runs = 1000                  # number of times we want it try to get to the goal

        self.state = [0, 0]                 # the current location of the agent

        self.goal = [3, 9]  # location of the end goal to begin with
        self.wall = []      # where the wall is
        for i in range(2, 5):
            self.wall.append([i, 7])

        # initialize the grid world optimisticly
        # 3D dimension goes as stay, up, down, right, left actions
        self.Q_table = np.full((self.n_rows, self.n_columns, 5), 20.0)
        # set the area where there is a wall with -50 reward
        self.Q_table[2:5, 7, :] = -50.0
        self.v = 0                                  # the sum of the values from this run
        self.v_est = 0                              # the estimate of the next step ahead??
        self.a = 0.75                                # learning rate? randomly selected
        self.g = 0.90                               # percentage of exploration
        self.z = 0.75                               # docking rate of looking to the future

    def reward_scheme(self, state):
        # ths is the reward structure for this world
        if state[:2] == self.goal:
            return 20.0
        elif state[:2] in self.wall:
            return -50.0
        else:
            return -1

    def state_for_action(self, state):
        # provide the state for the action desired; directions are swapped as 0,0 is in the upper left corner
        if state[2] == 0:                           # up
            n_state = [state[0]-1, state[1]]
        elif state[2] == 1:                         # down
            n_state = [state[0]+1, state[1]]
        elif state[2] == 2:                         # right
            n_state = [state[0], state[1]+1]
        elif state[2] == 3:                         # left
            n_state = [state[0], state[1]-1]
        elif state[2] == 4:                         # stay
            n_state = state[:2]
        else:
            print("Action provided does not fit within our posibilities")

        # make sure that the state is still on the grid world
        if self.check_if_legal(n_state):
            return n_state
        return state        # the move desired takes us off the grid so just stay in place

    def check_if_legal(self, state):
        # check if we are able to be in the stated pose, if not return false
        if 0 <= state[0] < self.n_rows and 0 <= state[1] < self.n_columns:    # check that we are still in the grid world
            if not (state in self.wall):                                      # make sure we are not in a wall
                return True
        return False

    def policy(self, S):
        # stuff

        if not random.random() > self.g:                                # get to explore
            return np.argmax(self.Q_table[S[0], S[1]])
        else:                                                           # go be greedy
            return random.randint(0, 4)


    def update_Q(self, Q_i, Q_i2, r):
        """
        Update the current state action pair
        :param Q_i: The index of the Q value we will be updating; [row, column, action]
        :param Q_i2: The index of the state that the action takes you to from the initial state
        :param r: reward from the current step taken
        :return: new state action pair
        """
        # pick the best next action to go with the next state and then use that one in the correction
        possibilities = []
        for i in range(5):              # go through each action of the next state
            possibilities.append(self.Q_table[Q_i2[0], Q_i2[1], i])

        if not random.random() > self.g:                        # get to explore
            j = possibilities.index(max(possibilities))
        else:                                               # go be greedy
            m = possibilities.index(max(possibilities))
            num_list = list(range(0, 5))                         # create list of indecies
            num_list.remove(m)                              # remove the one associated with the max value
            j = random.choice(num_list)                     # select from the remaining
        current = self.Q_table[Q_i[0], Q_i[1], Q_i[2]]
        next = self.Q_table[Q_i2[0], Q_i2[1], j]
        self.Q_table[Q_i[0], Q_i[1], Q_i[2]] = current + self.a*((r + self.z*next) - current)
        Q_i2.append(j)

        return Q_i2[:]                   # new state action pair

    def create_color_map(self, data, directions):
        # make a map of the Q_table
        # data = np.flip(data, 0)

        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(-100, 100, 256))
        cms = ListedColormap(newcolors)

        fig, ax = plt.subplots()
        im = ax.imshow(data)
        # fig, axs = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        # psm = axs.pcolormesh(data, s = "a")
        # # Loop over data dimensions and create text annotations.
        # for i in range(self.n_rows):
        #     for j in range(self.n_columns):
        #         text = ax.text(j, i, directions[i, j], ha="center", va="center", color="w")
        # fig.colorbar(psm, ax=axs)
        plt.show()

    def learn(self):
        # this is the main script for learning. Will be applying some weird algorithm...who knows!!!!!!!!!
        ep_rewards = []
        for i in range(self.n_runs):
            r_sum = 0
            init_pose = False
            for steps in range(self.n_steps):                       # keep track fo steps taken for this run
                while not init_pose:                                # create a random initial state
                    self.state = [random.randint(0, self.n_rows), random.randint(0, self.n_columns)]
                    init_pose = self.check_if_legal(self.state)
                    # 0:up 1:down 2:right 3:left 4:stay
                    action = self.policy(self.state)
                    # self.state.append(random.randint(0, 4))           # add a random action to this state

                new_location = self.state_for_action(self.state, action)          # get the next valid state based off of this action
                r = self.reward_scheme(self.state)                        # now get the direct value for this step taken
                r_sum += r
                self.state = self.update_Q(self.state, new_location[:], r)
            ep_rewards.append(r_sum)

        # plot the Q_table
        value_map = np.zeros((self.n_rows, self.n_columns))                      # used to store values for the color map to show
        direction_map = np.chararray((self.n_rows, self.n_columns))                  # text for showing best action for that location
        # find the best action value for each location state
        for j in range(self.n_rows):
            for k in range(self.n_columns):
                value_list = self.Q_table[j, k]
                max_val = max(value_list)
                value_map[j, k] = max_val
                action = np.where(value_list == max_val)
                if action == 0:
                    direction_map[j, k] = '^'
                elif action == 1:
                    direction_map[j, k] = 'v'
                elif action == 2:
                    direction_map[j, k] = '>'
                elif action == 3:
                    direction_map[j, k] = '<'
                elif action == 4:
                    direction_map[j, k] = '.'

        self.create_color_map(value_map, direction_map)

        # plot reward curve
        print(ep_rewards)
        plt.plot(list(range(self.n_runs)), ep_rewards)
        plt.title("Reward Curve")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        plt.show()


if __name__ == "__main__":
    Agent0 = R_learning()
    Agent0.learn()


