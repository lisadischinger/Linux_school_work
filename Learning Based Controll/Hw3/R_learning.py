#!/usr/bin/env python

# Homework 3 for Learning Based controls
# Lisa Dischinger
# 10/24/19

########################################################################################
# Testing out different reinforcment learning algorithms. we are testing on an agent within a 5x10 grid world
# the agent needs to learn the best parht to a set goal. methods tested out are epsilon-greedy SARSA, Q_learning
# with a static goal, and Q_learning with a moving goal
#########################################################################################

from numpy import random
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from statistics import mean
from matplotlib.colors import ListedColormap


class R_learning:
    def __init__(self):
        self.n_rows = 5
        self.n_columns = 10
        self.n_steps = 20                   # the maximum number of steps allowed
        self.n_runs = 10000                  # number of times we want it try to get to the goal

        self.state = [0, 0]                 # the current location of the agent
        self.action = 0

        self.goal = [3, 9]  # location of the end goal to begin with
        self.wall = []      # where the wall is
        for i in range(2, 5):
            self.wall.append([i, 7])

        self.Q_table = np.full((self.n_rows, self.n_columns, 5), 0.0)
        self.a = 0.1                                # learning rate? randomly selected
        self.g = 0.1                                # percentage of exploration
        self.z = 0.99                               # docking rate of looking to the future

    def reward_scheme(self, state):                 # ths is the reward structure for this world
        if state == self.goal:
            return 20.0
        else:
            return -1

    def state_for_action(self, state, action):
        # provide the state for the action desired; directions are swapped as 0,0 is in the upper left corner
        if action == 0:                           # up
            n_state = [state[0]-1, state[1]]
        elif action == 1:                         # down
            n_state = [state[0]+1, state[1]]
        elif action == 2:                         # right
            n_state = [state[0], state[1]+1]
        elif action == 3:                         # left
            n_state = [state[0], state[1]-1]
        elif action == 4:                         # stay
            n_state = state
        else:
            print("Action provided does not fit within our posibilities")
        if self.check_if_legal(n_state):             # make sure that the state is still on the grid world
            return n_state
        return state                                # the move desired takes us off the grid so just stay in place

    def check_if_legal(self, state):
        if 0 <= state[0] < self.n_rows and 0 <= state[1] < self.n_columns:  # check that we are still in the grid world
            if not (state in self.wall):                                    # make sure we are not in a wall
                return True
        return False

    def epsilon_policy(self, state):
        if random.random() > self.g:                                    # be greedy
            return np.argmax(self.Q_table[state[0], state[1]])
        else:                                                           # explore
            return random.randint(0, 4)

    def update_Q(self, state, action, reward, next_state, next_action):

        td_target = reward + self.z * self.Q_table[next_state[0], next_state[1], next_action]
        td_error = td_target - self.Q_table[state[0], state[1], action]

        self.Q_table[state[0], state[1], action] += self.a * td_error

    def create_color_map(self, data, title):            # make a map of the Q_table
        data = np.flip(data, 0)

        viridis = cm.get_cmap('viridis', 256)
        newcolors = viridis(np.linspace(-100, 100, 256))
        cms = ListedColormap(newcolors)

        fig, axs = plt.subplots(1, 1, figsize=(6, 3), constrained_layout=True)
        psm = axs.pcolormesh(data)
        fig.colorbar(psm, ax=axs)
        axs.set_title(title)

        # save picture
        path = "/home/disch/PycharmProjects/Linux_school_work/Learning Based Controll/Hw3"
        plt.savefig(path + " " + title + ".png")

        plt.close()

    def check_for_sucess(self, state):
        if state == self.goal:
            return True
        return False

    def plot_total_rewards(self, runs, rewards, title):
        # do the overtime how do our total rewards look map
        plt.plot(runs, rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
        plt.title(title)

        # save picture
        path = "/home/disch/PycharmProjects/Linux_school_work/Learning Based Controll/Hw3"
        plt.savefig(path + " " + title + ".png")

        plt.close()

    def learn(self):
        # this is the main script for learning. Will be applying some weird algorithm...who knows!!!!!!!!!
        ep_rewards = []
        all_r_sum = []
        list_o_runs = []
        for run in range(self.n_runs):
            r_sum = 0
            a = random.randint(0, self.n_rows)
            self.state = [random.randint(0, self.n_rows), random.randint(0, self.n_columns)]
            while not self.check_if_legal(self.state):
                self.state = [random.randint(0, self.n_rows), random.randint(0, self.n_columns)]

            self.action = self.epsilon_policy(self.state)                            # 0:up 1:down 2:right 3:left 4:stay
            for steps in range(self.n_steps):                       # keep track fo steps taken for this run
                next_state = self.state_for_action(self.state, self.action)          # get the next valid state based off of this action
                next_action = self.epsilon_policy(next_state)                           # get the action to go with the new state
                r = self.reward_scheme(self.state)                              # now get the direct value for this step taken
                r_sum += r
                self.update_Q(self.state, self.action, r, next_state, next_action)
                if self.check_for_sucess(self.state):
                    break
                self.state = next_state
                self.action = next_action

            all_r_sum.append(r_sum)
            if run%50.0 == 0 and not run == 0:                              # every 50th of a run
                loops = run / 50                        # get integer number of times we have looped
                avg = mean(all_r_sum[50*(loops-1):])
                ep_rewards.append(avg)

                list_o_runs.append(run)

        # plot the Q_table
        value_map = np.zeros((self.n_rows, self.n_columns))             # used to store values for the color map to show
        for j in range(self.n_rows):
            for k in range(self.n_columns):
                value_list = self.Q_table[j, k]
                max_val = max(value_list)
                value_map[j, k] = max_val

        self.create_color_map(value_map, "SARSA Map")
        self.plot_total_rewards(list_o_runs, ep_rewards, "SARSA Rewards Curve")

    def Q_learn(self, f_moving_target):
        # f_moving_target is true when the target is moving
        ep_rewards = []
        all_r_sum = []
        list_o_runs = []
        for run in range(self.n_runs):
            r_sum = 0
            self.state = [random.randint(0, self.n_rows), random.randint(0, self.n_columns)]
            while not self.check_if_legal(self.state):
                self.state = [random.randint(0, self.n_rows), random.randint(0, self.n_columns)]

            for steps in range(self.n_steps):                                   # keep track fo steps taken for this run
                action = self.epsilon_policy(self.state)                        # 0:up 1:down 2:right 3:left 4:stay
                next_state = self.state_for_action(self.state, action)          # get the next valid state based off of this action
                next_action = np.argmax(self.Q_table[next_state[0], next_state[1]])                      # get the action to go with the new state
                r = self.reward_scheme(self.state)                              # now get the direct value for this step taken
                r_sum += r
                self.update_Q(self.state, action, r, next_state, next_action)
                if self.check_for_sucess(self.state):
                    break
                self.state = next_state

                if f_moving_target:                             # for problem 3, the target will be moving once per step
                    goal_action = random.randint(0, 4)          # randomly select an action
                    self.goal = self.state_for_action(self.goal, goal_action)

            all_r_sum.append(r_sum)
            if run%50.0 == 0 and not run == 0:                              # every 50th of a run
                loops = run / 50                        # get integer number of times we have looped
                avg = mean(all_r_sum[50*(loops-1):])
                ep_rewards.append(avg)

                list_o_runs.append(run)

        # plot the Q_table
        value_map = np.zeros((self.n_rows, self.n_columns))             # used to store values for the color map to show
        for j in range(self.n_rows):
            for k in range(self.n_columns):
                value_list = self.Q_table[j, k]
                max_val = max(value_list)
                value_map[j, k] = max_val

        # plot stuff
        if not f_moving_target:
            self.create_color_map(value_map, "Q_learning with stationary goal")
            self.plot_total_rewards(list_o_runs, ep_rewards, "Static Q Rewards Curve")
        else:
            self.create_color_map(value_map, "Q_learning with moving goal")
            self.plot_total_rewards(list_o_runs, ep_rewards, "Dynamic Q Rewards Curve")


if __name__ == "__main__":
    Agent0 = R_learning()
    Agent0.learn()
    Agent0.Q_learn(False)                   # Q-learning algorithm with a stationary goal
    Agent0.Q_learn(True)                    # Q-learning algorithm with a moving goal


