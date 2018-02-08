from __future__ import division
import numpy as np

# Definitions of bandit algorithms
from algo.lghoo import *

from underlying_functions.functions import *
from arms.arms import *

import pandas as pd
import sys
import pickle

#simple progress bar
import pyprind

import time

## Right now the simulation works only for underlying functions between 0 and 1
# and with rewards between 0 and 1



def test_algorithm(arm_range, horizon, underfunc, plot=True, save=False,num_sim=1, height_limit=10, minimum_grow=20):
    """
    This function runs the LGHOO algorithm
    :param arm_range: range of the algorithm, with the current underlying functions it is only between 0 and 1
    :param horizon: horizon of the algorithm
    :param underfunc: one of the underlying functions
    :param plot: True if we want to plot the graph only the LAST run. If plot is false it will return the metric for the metrics
    :param save: if we want to save the graph the plotted graph of the LAST run
    :param num_sim: number of times we will run the whole algorithm (for monte carlo)
    :param height_limit:
    :param minimum_grow:
    :return:
    """
    print "Starting the algorithm"
    bar = pyprind.ProgBar(num_sim, stream=sys.stdout)
    #creating the LGHOO object
    algo = []

    # initial vectors representing the variables that will be returned
    regret = np.zeros(num_sim)
    cumulative_rewards = np.zeros(num_sim)
    euclidian_distance = np.zeros(num_sim)
    time_spent = np.zeros(num_sim)


    #starting with 1 simulation
    #algo.initialize(len(arms))

    for i in range(0,num_sim):
        algo = []
        algo = LGHOO(arm_range, height_limit=height_limit, rho=0.5, minimum_grow=minimum_grow)
        bar.update()
        # Underlying function
        #reevaluating it so we get updates
        func = underfunc().eval
        x_axis, y_axis = generate_xy(func, [algo.arm_range_min, algo.arm_range_max])
        xmax, ymax = getMaxFunc(x_axis, y_axis)
        max_reward = 1.0

        t0 = time.time()
        for t in range(horizon):
            # each arm in the simulation
            index = t
            arm = algo.select_arm()

            #choice of the underlying distribution
            reward = BernoulliArm(func(arm))

            #Get the cumulative reward
            cumulative_rewards[i] = cumulative_rewards[i] + reward

            #update the algorithm
            algo.update(arm, reward)
        t1 = time.time()
        time_spent[i] = t1-t0

        ## Collection of the metrics Metrics
        #Get the the best arm - Modification
        best_arm = algo.get_best_arm_value()
        #Get the best arm - Original selection of the higher node of the tree
        #best_arm = algo.get_original_best_arm()

        #calculate the distance (not perfect specially if there is more than one maximum)
        euclidian_distance[i] = np.absolute(xmax-best_arm)

        #calculate the regret
        max_exp_value = ymax*max_reward*horizon
        regret[i] = max_exp_value - cumulative_rewards[i]

    if plot==True:
        # Underlying function
        func = underfunc().__init__().eval
        x_axis, y_axis = generate_xy(func, [algo.arm_range_min, algo.arm_range_max])
        xmax, ymax = getMaxFunc(x_axis, y_axis)
        filename = func.__name__+"-"+str(horizon)+".png"
        algo.plot_graph_with_function(x_axis,y_axis,rescale_y=3, save=save, filename=filename)
        return
    else:
        return [cumulative_rewards, euclidian_distance, regret, time_spent]


def MonteCarloSim(n, func, horizon,height_limit, minimum_grow):
    t = range(1, n + 1)
    cumulative_rewards, euclidian_distance, regret, time_spent = test_algorithm([0, 1], horizon, func, plot=False, save=False,
                                                                    num_sim=n)
    data = pd.DataFrame({'cumulative_rewards': cumulative_rewards,
                         'euclidian_distance': euclidian_distance,
                         'regret': regret,
                         'time_spent':time_spent})

    #print data
    mainfile = sys.argv[0]
    pathname = os.path.join(os.path.dirname(mainfile), "data")
    filename = "montecarlo_" + func.__name__ + "-" + "numsim" + str(n) + "mingrow" + str(minimum_grow) + ".csv"
    output = os.path.join(pathname, filename)
    data.to_csv(output)

if __name__ == "__main__":
    MonteCarloSim(n=1000, func=randomPoly, horizon=1000, height_limit=20, minimum_grow=0)




    # For testing the algorithm only once
    # horizon = 1000
    # test_algorithm([0, 1], horizon, step().eval, plot=True, save=True)
    # test_algorithm([0, 1], horizon, normal().eval, plot=True, save=True)
    # test_algorithm([0, 1], horizon, complex_trig().eval, plot=True, save=True)
    # test_algorithm([0, 1], horizon, uniform().eval, plot=True, save=True)
    # test_algorithm([0, 1], horizon, triangle().eval, plot=True, save=True)
    # test_algorithm([0, 1], horizon, linear().eval, plot=True, save=True)
    # test_algorithm([0, 1], horizon, binormal().eval, plot=True, save=True)
    #
