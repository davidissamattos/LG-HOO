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


#Threading
from threading import Thread


## Right now the simulation works only for underlying functions between 0 and 1
# and with rewards between 0 and 1



def test_algorithm(arm_range, horizon, underfunc, plot=True, save=False, num_sim=1, height_limit=10, minimum_grow=20, best_arm_policy='new'):
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
    :param best_arm_policy: type of policy for selecting the best arm
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
    object_size = np.zeros(num_sim)

    #starting with 1 simulation
    #algo.initialize(len(arms))
    func = []

    for i in range(0,num_sim):
        algo = []
        #reset the algorithm every round
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

        best_arm = []
        #Get the best arm - Original selection of the higher node of the tree
        if best_arm_policy == 'original':
            best_arm = algo.get_original_best_arm()
        if best_arm_policy == 'new':
            best_arm = algo.get_best_arm_value()
        #calculate the distance (not perfect specially if there is more than one maximum)
        euclidian_distance[i] = np.absolute(xmax-best_arm)

        #calculate the regret
        max_exp_value = ymax*max_reward*horizon
        regret[i] = max_exp_value - cumulative_rewards[i]

        #log the object size
        object_size[i] = sys.getsizeof(algo)

    if plot==True:
        # Underlying function
        x_axis, y_axis = generate_xy(func, [algo.arm_range_min, algo.arm_range_max])
        # xmax, ymax = getMaxFunc(x_axis, y_axis)
        filename = underfunc.__name__+"-"+str(horizon)+".png"
        algo.plot_graph_with_function(x_axis,y_axis,rescale_y=3, save=save, filename=filename)
        return
    else:
        return [cumulative_rewards, euclidian_distance, regret, time_spent, object_size]

def MonteCarloSim(n, func, horizon, height_limit, minimum_grow, best_arm_policy='new',plot=False,save=True):
    if plot == True:
        test_algorithm([0, 1],
                       horizon=horizon,
                       underfunc=func,
                       minimum_grow=minimum_grow,
                       height_limit=height_limit,
                       plot=True,
                       save=save,
                       num_sim=n,
                       best_arm_policy=best_arm_policy)
    else:
        #in this case we never save
        cumulative_rewards, euclidian_distance, regret, time_spent, object_size = test_algorithm([0, 1],
                                                                                                 horizon=horizon,
                                                                                                 underfunc=func,
                                                                                                 minimum_grow=minimum_grow,
                                                                                                 height_limit=height_limit,
                                                                                                 plot=False,
                                                                                                 save=False,
                                                                                                 num_sim=n,
                                                                                                 best_arm_policy=best_arm_policy)
        data = pd.DataFrame({'cumulative_rewards': cumulative_rewards,
                             'euclidian_distance': euclidian_distance,
                             'regret': regret,
                             'time_spent': time_spent,
                             'object_size': object_size})
        # print data
        mainfile = sys.argv[0]
        pathname = os.path.join(os.path.dirname(mainfile), "data")
        filename = "montecarlo-" + func.__name__ + "-" + "-numsim-" + str(n) + "mingrow-" + str(
            minimum_grow) + "-arm_policy-" + str(best_arm_policy) + "-horizon-" + str(horizon) + ".csv"
        output = os.path.join(pathname, filename)
        data.to_csv(output, index=False, header=True)

def Case1():
    MonteCarloSim(n=1000, func=randomPoly, horizon=1000, height_limit=20, minimum_grow=20, best_arm_policy='new')

def Case2():
    MonteCarloSim(n=1000, func=randomPoly, horizon=1000, height_limit=20, minimum_grow=0, best_arm_policy='original')


if __name__ == "__main__":
    #Simulation of one function
    MonteCarloSim(n=1, func=normal, horizon=1000, height_limit=20, minimum_grow=20, best_arm_policy='new', plot=True, save=False)

    #Comparison cases
    #Case1()
    #Case2()

    #If threading
    # t1 = Thread(target=Case1, args=[])
    # t2 = Thread(target=Case2, args=[])
    # t1.start()
    # t2.start()


    # For testing the algorithm only once
    # horizon = 10000
    # MonteCarloSim(n=1,  horizon=horizon, func=step, height_limit=10, minimum_grow=20, best_arm_policy='new', plot=True,
    #               save=True)
    # MonteCarloSim(n=1, horizon=horizon, func=normal, height_limit=10, minimum_grow=20, best_arm_policy='new',
    #               plot=True,
    #               save=True)
    # MonteCarloSim(n=1,  horizon=horizon, func=complex_trig, height_limit=10, minimum_grow=20, best_arm_policy='new',
    #               plot=True,
    #               save=True)
    # MonteCarloSim(n=1,  horizon=horizon, func=uniform, height_limit=10, minimum_grow=20, best_arm_policy='new',
    #               plot=True,
    #               save=True)
    # MonteCarloSim(n=1, horizon=horizon, func=triangle, height_limit=10, minimum_grow=20, best_arm_policy='new',
    #               plot=True,
    #               save=True)
    # MonteCarloSim(n=1, horizon=horizon, func=linear, height_limit=10, minimum_grow=20, best_arm_policy='new',
    #               plot=True,
    #               save=True)
    # MonteCarloSim(n=1,horizon=horizon, func=binormal, height_limit=10, minimum_grow=20, best_arm_policy='new',
    #               plot=True,
    #               save=True)