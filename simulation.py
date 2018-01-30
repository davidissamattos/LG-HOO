from __future__ import division
import numpy as np

# Definitions of bandit algorithms
from algo.lghoo import *

from underlying_functions.functions import *
from arms.arms import *

def test_algorithm(arm_range, horizon, func, plot=True, save=False):
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
        arm = np.random.uniform(0,1,1)[0] #algo.select_arm()
        chosen_arms[index] = arm
        #choice of the underlying distribution
        reward = BernoulliArm(func(arm))
        rewards[index] = reward
        if t == 1:
            cumulative_rewards[index] = reward
        else:
            cumulative_rewards[index] = cumulative_rewards[index - 1] + reward

        algo.update(arm, reward)

    print "debug: arm arm_bound arm_count", algo.debug_arms_and_bounds()

    if plot==True:
        x_axis, y_axis = generate_xy(func,[algo.arm_range_min, algo.arm_range_max])
        filename = func.__name__+"-"+str(horizon)+".png"
        algo.plot_graph_with_function(x_axis,y_axis,rescale_y=3, save=save, filename=filename)
        return
    else:
        return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]



if __name__ == "__main__":
    test_algorithm([0, 1], 100, step1, plot=True, save=False)

    # horizon = 1000
    # test_algorithm([0, 1], horizon, step1, plot=True, save=True)
    # test_algorithm([0, 1], horizon, normal80, plot=True, save=True)
    # test_algorithm([0, 1], horizon, complex_trig, plot=True, save=True)
    # test_algorithm([0, 1], horizon, uniform_50, plot=True, save=True)
    # test_algorithm([0, 1], horizon, triangle30, plot=True, save=True)
    # test_algorithm([0, 1], horizon, linear, plot=True, save=True)
    # test_algorithm([0, 1], horizon, binormal4080, plot=True, save=True)
    #
    # horizon = 10000
    # test_algorithm([0, 1], horizon, step1, plot=True, save=True)
    # test_algorithm([0, 1], horizon, normal80, plot=True, save=True)
    # test_algorithm([0, 1], horizon, complex_trig, plot=True, save=True)
    # test_algorithm([0, 1], horizon, uniform_50, plot=True, save=True)
    # test_algorithm([0, 1], horizon, triangle30, plot=True, save=True)
    # test_algorithm([0, 1], horizon, linear, plot=True, save=True)
    # test_algorithm([0, 1], horizon, binormal4080, plot=True, save=True)