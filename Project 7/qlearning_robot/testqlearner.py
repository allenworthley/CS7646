""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a Q Learner in a navigation problem.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
2016-10-20 Added "quicksand" and uncertain actions.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Tucker Balch (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import random as rand  		  	   		   	 		  		  		    	 		 		   		 		  
import time  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import QLearner as ql  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def printmap(data):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Prints out the map  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param data: 2D array that stores the map  		  	   		   	 		  		  		    	 		 		   		 		  
    :type data: array  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    print("--------------------")  		  	   		   	 		  		  		    	 		 		   		 		  
    for row in range(0, data.shape[0]):  		  	   		   	 		  		  		    	 		 		   		 		  
        for col in range(0, data.shape[1]):  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 0:  # Empty space  		  	   		   	 		  		  		    	 		 		   		 		  
                print(" ", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 1:  # Obstacle  		  	   		   	 		  		  		    	 		 		   		 		  
                print("O", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 2:  # El roboto  		  	   		   	 		  		  		    	 		 		   		 		  
                print("*", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 3:  # Goal  		  	   		   	 		  		  		    	 		 		   		 		  
                print("X", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 4:  # Trail  		  	   		   	 		  		  		    	 		 		   		 		  
                print(".", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 5:  # Quick sand  		  	   		   	 		  		  		    	 		 		   		 		  
                print("~", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 6:  # Stepped in quicksand  		  	   		   	 		  		  		    	 		 		   		 		  
                print("@", end=" ")  		  	   		   	 		  		  		    	 		 		   		 		  
        print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print("--------------------")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def getrobotpos(data):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Finds where the robot is in the map  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param data: 2D array that stores the map  		  	   		   	 		  		  		    	 		 		   		 		  
    :type data: array  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the position of the robot  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: int, int  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    R = -999  		  	   		   	 		  		  		    	 		 		   		 		  
    C = -999  		  	   		   	 		  		  		    	 		 		   		 		  
    for row in range(0, data.shape[0]):  		  	   		   	 		  		  		    	 		 		   		 		  
        for col in range(0, data.shape[1]):  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 2:  		  	   		   	 		  		  		    	 		 		   		 		  
                C = col  		  	   		   	 		  		  		    	 		 		   		 		  
                R = row  		  	   		   	 		  		  		    	 		 		   		 		  
    if (R + C) < 0:  		  	   		   	 		  		  		    	 		 		   		 		  
        print("warning: start location not defined")  		  	   		   	 		  		  		    	 		 		   		 		  
    return R, C  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def getgoalpos(data):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    find where the goal is in the map  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param data: 2D array that stores the map  		  	   		   	 		  		  		    	 		 		   		 		  
    :type data: array  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the position of the goal  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: tuple(int, int)  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    R = -999  		  	   		   	 		  		  		    	 		 		   		 		  
    C = -999  		  	   		   	 		  		  		    	 		 		   		 		  
    for row in range(0, data.shape[0]):  		  	   		   	 		  		  		    	 		 		   		 		  
        for col in range(0, data.shape[1]):  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[row, col] == 3:  		  	   		   	 		  		  		    	 		 		   		 		  
                C = col  		  	   		   	 		  		  		    	 		 		   		 		  
                R = row  		  	   		   	 		  		  		    	 		 		   		 		  
    if (R + C) < 0:  		  	   		   	 		  		  		    	 		 		   		 		  
        print("warning: goal location not defined")  		  	   		   	 		  		  		    	 		 		   		 		  
    return (R, C)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def movebot(data, oldpos, a):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    move the robot and report reward  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param data: 2D array that stores the map  		  	   		   	 		  		  		    	 		 		   		 		  
    :type data: array  		  	   		   	 		  		  		    	 		 		   		 		  
    :param oldpos: old position of the robot  		  	   		   	 		  		  		    	 		 		   		 		  
    :type oldpos: int, int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param a: the action to take  		  	   		   	 		  		  		    	 		 		   		 		  
    :type a: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the new position of the robot and the reward  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: tuple(int, int), int  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    testr, testc = oldpos  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    randomrate = 0.20  # how often do we move randomly  		  	   		   	 		  		  		    	 		 		   		 		  
    quicksandreward = -100  # penalty for stepping on quicksand  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # decide if we're going to ignore the action and  		  	   		   	 		  		  		    	 		 		   		 		  
    # choose a random one instead  		  	   		   	 		  		  		    	 		 		   		 		  
    if rand.uniform(0.0, 1.0) <= randomrate:  # going rogue  		  	   		   	 		  		  		    	 		 		   		 		  
        a = rand.randint(0, 3)  # choose the random direction  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # update the test location  		  	   		   	 		  		  		    	 		 		   		 		  
    if a == 0:  # north  		  	   		   	 		  		  		    	 		 		   		 		  
        testr = testr - 1  		  	   		   	 		  		  		    	 		 		   		 		  
    elif a == 1:  # east  		  	   		   	 		  		  		    	 		 		   		 		  
        testc = testc + 1  		  	   		   	 		  		  		    	 		 		   		 		  
    elif a == 2:  # south  		  	   		   	 		  		  		    	 		 		   		 		  
        testr = testr + 1  		  	   		   	 		  		  		    	 		 		   		 		  
    elif a == 3:  # west  		  	   		   	 		  		  		    	 		 		   		 		  
        testc = testc - 1  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    reward = -1  # default reward is negative one  		  	   		   	 		  		  		    	 		 		   		 		  
    # see if it is legal. if not, revert  		  	   		   	 		  		  		    	 		 		   		 		  
    if testr < 0:  # off the map  		  	   		   	 		  		  		    	 		 		   		 		  
        testr, testc = oldpos  		  	   		   	 		  		  		    	 		 		   		 		  
    elif testr >= data.shape[0]:  # off the map  		  	   		   	 		  		  		    	 		 		   		 		  
        testr, testc = oldpos  		  	   		   	 		  		  		    	 		 		   		 		  
    elif testc < 0:  # off the map  		  	   		   	 		  		  		    	 		 		   		 		  
        testr, testc = oldpos  		  	   		   	 		  		  		    	 		 		   		 		  
    elif testc >= data.shape[1]:  # off the map  		  	   		   	 		  		  		    	 		 		   		 		  
        testr, testc = oldpos  		  	   		   	 		  		  		    	 		 		   		 		  
    elif data[testr, testc] == 1:  # it is an obstacle  		  	   		   	 		  		  		    	 		 		   		 		  
        testr, testc = oldpos  		  	   		   	 		  		  		    	 		 		   		 		  
    elif data[testr, testc] == 5:  # it is quicksand  		  	   		   	 		  		  		    	 		 		   		 		  
        reward = quicksandreward  		  	   		   	 		  		  		    	 		 		   		 		  
        data[testr, testc] = 6  # mark the event  		  	   		   	 		  		  		    	 		 		   		 		  
    elif data[testr, testc] == 6:  # it is still quicksand  		  	   		   	 		  		  		    	 		 		   		 		  
        reward = quicksandreward  		  	   		   	 		  		  		    	 		 		   		 		  
        data[testr, testc] = 6  # mark the event  		  	   		   	 		  		  		    	 		 		   		 		  
    elif data[testr, testc] == 3:  # it is the goal  		  	   		   	 		  		  		    	 		 		   		 		  
        reward = 1  # for reaching the goal  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    return (testr, testc), reward  # return the new, legal location  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def discretize(pos):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    convert the location to a single integer  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param pos: the position to discretize  		  	   		   	 		  		  		    	 		 		   		 		  
    :type pos: int, int  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the discretized position  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    return pos[0] * 10 + pos[1]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def test(map, epochs, learner, verbose):  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    function to test the code  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param map: 2D array that stores the map  		  	   		   	 		  		  		    	 		 		   		 		  
    :type map: array  		  	   		   	 		  		  		    	 		 		   		 		  
    :param epochs: each epoch involves one trip to the goal  		  	   		   	 		  		  		    	 		 		   		 		  
    :type epochs: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :param learner: the qlearner object  		  	   		   	 		  		  		    	 		 		   		 		  
    :type learner: QLearner  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: the total reward  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: np.float64  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    # each epoch involves one trip to the goal  		  	   		   	 		  		  		    	 		 		   		 		  
    startpos = getrobotpos(map)  # find where the robot starts  		  	   		   	 		  		  		    	 		 		   		 		  
    goalpos = getgoalpos(map)  # find where the goal is  		  	   		   	 		  		  		    	 		 		   		 		  
    scores = np.zeros((epochs, 1))  		  	   		   	 		  		  		    	 		 		   		 		  
    for epoch in range(1, epochs + 1):  		  	   		   	 		  		  		    	 		 		   		 		  
        total_reward = 0  		  	   		   	 		  		  		    	 		 		   		 		  
        data = map.copy()  		  	   		   	 		  		  		    	 		 		   		 		  
        robopos = startpos  		  	   		   	 		  		  		    	 		 		   		 		  
        state = discretize(robopos)  # convert the location to a state  		  	   		   	 		  		  		    	 		 		   		 		  
        action = learner.querysetstate(  		  	   		   	 		  		  		    	 		 		   		 		  
            state  		  	   		   	 		  		  		    	 		 		   		 		  
        )  # set the state and get first action  		  	   		   	 		  		  		    	 		 		   		 		  
        count = 0  		  	   		   	 		  		  		    	 		 		   		 		  
        while (robopos != goalpos) & (count < 10000):  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
            # move to new location according to action and then get a new action  		  	   		   	 		  		  		    	 		 		   		 		  
            newpos, stepreward = movebot(data, robopos, action)  		  	   		   	 		  		  		    	 		 		   		 		  
            if newpos == goalpos:  		  	   		   	 		  		  		    	 		 		   		 		  
                r = 1  # reward for reaching the goal  		  	   		   	 		  		  		    	 		 		   		 		  
            else:  		  	   		   	 		  		  		    	 		 		   		 		  
                r = stepreward  # negative reward for not being at the goal  		  	   		   	 		  		  		    	 		 		   		 		  
            state = discretize(newpos)  		  	   		   	 		  		  		    	 		 		   		 		  
            action = learner.query(state, r)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[robopos] != 6:  		  	   		   	 		  		  		    	 		 		   		 		  
                data[robopos] = 4  # mark where we've been for map printing  		  	   		   	 		  		  		    	 		 		   		 		  
            if data[newpos] != 6:  		  	   		   	 		  		  		    	 		 		   		 		  
                data[newpos] = 2  # move to new location  		  	   		   	 		  		  		    	 		 		   		 		  
            robopos = newpos  # update the location  		  	   		   	 		  		  		    	 		 		   		 		  
            # if verbose: time.sleep(1)  		  	   		   	 		  		  		    	 		 		   		 		  
            total_reward += stepreward  		  	   		   	 		  		  		    	 		 		   		 		  
            count = count + 1  		  	   		   	 		  		  		    	 		 		   		 		  
        if count == 100000:  		  	   		   	 		  		  		    	 		 		   		 		  
            print("timeout")  		  	   		   	 		  		  		    	 		 		   		 		  
        if verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            printmap(data)  		  	   		   	 		  		  		    	 		 		   		 		  
        if verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
            print(f"{epoch}, {total_reward}")  		  	   		   	 		  		  		    	 		 		   		 		  
        scores[epoch - 1, 0] = total_reward  		  	   		   	 		  		  		    	 		 		   		 		  
    return np.median(scores)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# run the code to test a learner  		  	   		   	 		  		  		    	 		 		   		 		  
def test_code():  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    verbose = True  # print lots of debug stuff if True  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    # read in the map  		  	   		   	 		  		  		    	 		 		   		 		  
    filename = "testworlds/world01.csv"  		  	   		   	 		  		  		    	 		 		   		 		  
    inf = open(filename)  		  	   		   	 		  		  		    	 		 		   		 		  
    data = np.array(  		  	   		   	 		  		  		    	 		 		   		 		  
        [list(map(float, s.strip().split(","))) for s in inf.readlines()]  		  	   		   	 		  		  		    	 		 		   		 		  
    )  		  	   		   	 		  		  		    	 		 		   		 		  
    originalmap = (  		  	   		   	 		  		  		    	 		 		   		 		  
        data.copy()  		  	   		   	 		  		  		    	 		 		   		 		  
    )  # make a copy so we can revert to the original map later  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    if verbose:  		  	   		   	 		  		  		    	 		 		   		 		  
        printmap(data)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    rand.seed(5)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    ######## run non-dyna test ########  		  	   		   	 		  		  		    	 		 		   		 		  
    learner = ql.QLearner(  		  	   		   	 		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		   	 		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		   	 		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		   	 		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		   	 		  		  		    	 		 		   		 		  
        rar=0.98,  		  	   		   	 		  		  		    	 		 		   		 		  
        radr=0.999,  		  	   		   	 		  		  		    	 		 		   		 		  
        dyna=0,  		  	   		   	 		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		   	 		  		  		    	 		 		   		 		  
    )  # initialize the learner  		  	   		   	 		  		  		    	 		 		   		 		  
    epochs = 500  		  	   		   	 		  		  		    	 		 		   		 		  
    total_reward = test(data, epochs, learner, verbose)  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{epochs}, median total_reward {total_reward}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    non_dyna_score = total_reward  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    ######## run dyna test ########  		  	   		   	 		  		  		    	 		 		   		 		  
    learner = ql.QLearner(  		  	   		   	 		  		  		    	 		 		   		 		  
        num_states=100,  		  	   		   	 		  		  		    	 		 		   		 		  
        num_actions=4,  		  	   		   	 		  		  		    	 		 		   		 		  
        alpha=0.2,  		  	   		   	 		  		  		    	 		 		   		 		  
        gamma=0.9,  		  	   		   	 		  		  		    	 		 		   		 		  
        rar=0.5,  		  	   		   	 		  		  		    	 		 		   		 		  
        radr=0.99,  		  	   		   	 		  		  		    	 		 		   		 		  
        dyna=200,  		  	   		   	 		  		  		    	 		 		   		 		  
        verbose=False,  		  	   		   	 		  		  		    	 		 		   		 		  
    )  # initialize the learner  		  	   		   	 		  		  		    	 		 		   		 		  
    epochs = 50  		  	   		   	 		  		  		    	 		 		   		 		  
    data = originalmap.copy()  		  	   		   	 		  		  		    	 		 		   		 		  
    total_reward = test(data, epochs, learner, verbose)  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"{epochs}, median total_reward {total_reward}")  		  	   		   	 		  		  		    	 		 		   		 		  
    dyna_score = total_reward  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print()  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"results for {filename}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"non_dyna_score: {non_dyna_score}")  		  	   		   	 		  		  		    	 		 		   		 		  
    print(f"dyna_score    : {dyna_score}")  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    test_code()  		  	   		   	 		  		  		    	 		 		   		 		  
