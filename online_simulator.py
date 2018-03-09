# This is the online simulator for the online ACE framework
# This framework implements the LGHOO algorithm
# Need a configuration file to work
#
#
#
#
#



from __future__ import division
import numpy as np

# Definitions of bandit algorithms
from algo.lghoo import *

from underlying_functions.functions import *
from arms.arms import *
import requests
import json
import numpy as np
import uuid
import time
import pickle


local_server = False

server = ""
if local_server:
    server = "http://127.0.0.1:5000/api"
    apikey = ""
    experiment_name = "lghoo.test.local"
else:
    server = "https://ace-framework.appspot.com/api"
    apikey = "?key=AIzaSyDJ4TTikXIErcWL5nE45z3nIhhv2LPwlBg"
    experiment_name = "lghoo.test.server"

#Server configurations and URL
configURL = "/config"
requestURL = "/request"
logURL = "/log"
loggedDataURL = "/raw_exp_data"
getlghooURL = "/get/lghoo"
clearlghooURL = "/clear/lghoo"

#Experiment parameters



#THIS IS AN ONLINE SIMULATOR FOR x between 0 and 1 only.
# Needs some rework to plot in another area

class Connection:
    def __init__(self):
        self.RequestURL = server + requestURL + apikey
        self.LogURL = server + logURL + apikey
        self.getlghooURL = server + getlghooURL + apikey
        self.clearlghooURL = server + clearlghooURL + apikey

    def log(self,data):
        r = requests.post(self.LogURL, json=data)
        if r.status_code == 200:
            pass
            #print "Log succeeded"
        else:
            print "Log failed"

    def request(self,data):
        r = requests.post(self.RequestURL, json=data)
        if r.status_code == 200:
            #print "json ", r.json()
            return r.json()
        else:
            return "error"

    def getlghoo(self,data):
        r = requests.post(self.getlghooURL, json=data)
        #print "getting lghoo"
        if r.status_code == 200:
            lghoo_dic = r.json()
            return pickle.loads(lghoo_dic['lghoo'])
        else:
            return "error"

    def clearlghoo(self,name):
        data = {"experiment_name": name}
        r = requests.post(self.clearlghooURL, json=data)
        if r.status_code == 200:
            print "Clear succeeded"
        else:
            print "clear failed"

class User:
    def __init__(self,func):
        #creates a user id
        self.userid = str(uuid.uuid4())
        #underlying function
        self.func = func
        self.conn = Connection()


    def playArm(self):
        print self.func(self.arm)
        return BernoulliArm(self.func(self.arm))

    def requestArm(self):
        data = {"experiment_name": experiment_name,
                "unit_diversion": self.userid}
        self.arm = self.conn.request(data)
        return self.arm

    def logReward(self,reward):
        data = {
            "experiment_name": experiment_name,
            'variations':
                {'x': self.arm},
            'signals':
                {'clicked': reward}
        }
        self.conn.log(data)
        return

    def getLghooList(self):
        data = {"experiment_name": experiment_name}
        lghoo = self.conn.getlghoo(data)
        return lghoo


def test_algorithm(arm_range, horizon, func, plot=True, save=False):
    """
    Testing the algorithm with the server
    """
    for t in range(horizon):

        #1 choice of the underlying distribution
        user = User(func)
        #2 Find an arm to play
        arm = user.requestArm()

        #3 play the arm and get the travel time
        #rescaling the arm as well
        reward = user.playArm()
        #4 log the traveltime
        user.logReward(reward)
        print "Arm: ", arm, "Reward: ", reward
        #5 sleeping for 1 second
        #time.sleep(1)
        #going for a new cycle

    user = User(func)
    #To plot the graphs we load an object as in the algorithm and replace only the arm_list
    lghoo = LGHOO(arm_range, height_limit=10, rho=0.5, minimum_grow=20)
    lghoo.arm_list = user.getLghooList()


    if plot==True:
        x_axis, y_axis = generate_xy(func,[lghoo.arm_range_min, lghoo.arm_range_max])
        filename = func.__name__+"-"+str(horizon)+".png"
        lghoo.plot_graph_with_function(x_axis, y_axis,rescale_y=3, save=save, filename=filename)
        return
    else:
        print lghoo._get_full_arm_list()
        print lghoo.get_best_arm_value()
        return

if __name__ == "__main__":
    print server
    print experiment_name

    conn = Connection()
    conn.clearlghoo(experiment_name)
    test_algorithm([0.0, 1.0], 1000, normal80, plot=True, save=False)

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