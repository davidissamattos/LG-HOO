from __future__ import division

import numpy as np

# Definitions of bandit algorithms
from hoo.lghoo import *


# numerical libraries
# Multi-armed bandits library


def bern_arm_uniform_50(x):
    return np.random.choice(2, 1, p=[0.5, 0.5])[0]

def discrete_function(x):
    if x < 0.2:
        p = 0.2
    if x >= 0.2 and x < 0.3:
        p = 0.8
    if x >= 0.3:
        p = 0.1
    return p

def bern_arm_distr_reward_discrete(x):
    p = discrete_function(x)
    return np.random.choice(2, 1, p=[1-p, p])[0]


def trig_function(x):
    p = 1 / (12 * (np.sin(13 * x) * np.sin(27 * x) + 1))
    return p

def bern_arm_distr_reward_trig(x):
    p = trig_function(x)
    reward = np.random.choice(2, 1, p=[1-p, p])[0]
    return reward


def test_algorithm(arm_range, horizon):
    """
    """

    algo = LGHOO(arm_range, height_limit=10, rho=0.5, minimum_grow=20)
    # initial vectors representing the variables that will be returned
    chosen_arms = [0.0 for i in range(horizon)]
    rewards = [0.0 for i in range( horizon)]
    cumulative_rewards = [0.0 for i in range(horizon)]
    sim_nums = [0.0 for i in range(horizon)]
    times = [0.0 for i in range(horizon)]


    #starting with 1 simulation
    #algo.initialize(len(arms))

    #soma = 0
    for t in range(horizon):
        # each arm in the simulation
        index = t
        times[index] = t
        arm = algo.select_arm()
        chosen_arms[index] = arm

        #choice of the underlying distribution
        reward = bern_arm_distr_reward_discrete(arm)
        # reward = bern_arm_distr_reward_trig(arm)
        rewards[index] = reward
        #soma = reward + soma
        #
        # if t == 1:
        #     cumulative_rewards[index] = reward
        # else:
        #     cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

        algo.update(arm, reward)
    #print algo.get_full_arm_list()
    #print soma/horizon
    #algo.save_list_arms_to_file()
    print "debug: arm arm_bound arm_count", algo.debug_arms_and_bounds()
    x_axis = np.linspace(algo.arm_range_min,algo.arm_range_max, num=500)
    y_axis = np.apply_along_axis(np.vectorize(discrete_function),0,x_axis)
    algo.plot_graph_with_function(x_axis,y_axis)
    return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]



if __name__ == "__main__":

    test_algorithm([0, 1], 10000)