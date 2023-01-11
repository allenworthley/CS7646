""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
"""

import math
import sys
import numpy as np
import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl
import BagLearner as bl
import InsaneLearner as il
import matplotlib.pyplot as plt
from datetime import datetime


def get_sample(x, y):
    if len(y.shape) == 1:
        y = np.reshape(y, (-1, 1))
    data = np.append(x, y, axis=1)
    obs = data.shape[0]
    rand_ind = np.random.randint(obs, size=obs)
    new_data = np.zeros(data.shape)
    for row in range(0, obs):
        new_data[row] = data[rand_ind[row]]
    new_x = new_data[:, 0:data.shape[1]-1]
    new_y = new_data[:, -1]
    return new_x, new_y


def get_data(file, trim_row_headers=True):
    """
    :param file: testing dataset
    :return: X and Y in same np array
    """
    with open(file) as f:
        alldata = np.genfromtxt(f, delimiter=",")
        # Cleaning
        if trim_row_headers:
            alldata = alldata[1:, 1:]  # drops row/date column and headers
        # Spliting datasets to match add_evidence requirement
        num_cols = alldata.shape[1]
        X = alldata[:, 0:num_cols - 1]
        Y = alldata[:, -1]
    return X, Y


if __name__ == "__main__":

    x, y = get_data(sys.argv[1])

    # compute how much of the data is training and testing
    train_rows = int(0.6 * x.shape[0])
    test_rows = x.shape[0] - train_rows

    np.random.seed(903646612)
    # separate out training and testing data
    train_x, train_y = x[:train_rows, :], y[:train_rows]
    test_x = x[train_rows:, :]
    test_y = y[train_rows:]


    # --------------------------------- #
    # ---------- Experiment 1 --------- #
    # --------------------------------- #
    # Note: Over fitting occurs when in-sample error decreases but out of sample error increases
    # Calculate in and out of sample error for leaf sizes [1:50]
    # initialize objects
    num_leafs = 50
    learners = []
    for i in range(0, num_leafs):
        # initialize
        learners.append(dt.DTLearner(leaf_size=i))
        # build model
        learners[i].add_evidence(train_x, train_y)

    in_sample = []
    out_sample = []
    for i in range(0, num_leafs):
        # In sample calcs
        pred_y = learners[i].query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample.append(rmse)

        # Out of sample calcs
        pred_y = learners[i].query(test_x)  # get the predictions
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample.append(rmse)

    in_sample = np.array(in_sample).reshape((-1, 1))
    out_sample = np.array(out_sample).reshape((-1, 1))

    plot_data = np.append(in_sample, out_sample, axis=1)

    # Plot
    plt.figure(1)
    plt.axis([num_leafs, 0, 0, 0.010])
    plt.plot(plot_data)
    # note legend goes after plot
    plt.legend(["in_sample_error", "out_sample_error"])
    plt.xlabel("Number of Leafs")
    plt.ylabel("RMSE")
    plt.title("Experiment 1")
    plt.savefig("experiment_1.png")

    # --------------------------------- #
    # ---------- Experiment 2 --------- #
    # --------------------------------- #
    # Note: Over fitting occurs when in-sample error decreases but out of sample error increases
    # Calculate in and out of sample error for leaf sizes [1:50]
    # initialize objects
    num_leafs = 50
    learners = []

    #  def __init__(self, learner=object, kwargs={}, bags=1, boost=False, verbose=False):
    for i in range(0, num_leafs):
        # initialize
        learners.append(bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": i}, bags=20, verbose=False))
        # build model
        learners[i].add_evidence(train_x, train_y)

    in_sample = []
    out_sample = []
    for i in range(0, num_leafs):
        # In sample calcs
        pred_y = learners[i].query(train_x)  # get the predictions
        rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
        in_sample.append(rmse)

        # Out of sample calcs
        pred_y = learners[i].query(test_x)  # get the predictions
        rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
        out_sample.append(rmse)

    in_sample = np.array(in_sample).reshape((-1, 1))
    out_sample = np.array(out_sample).reshape((-1, 1))

    plot_data = np.append(in_sample, out_sample, axis=1)

    # Plot
    plt.figure(2)
    plt.axis([num_leafs, 0, 0, 0.010])
    plt.plot(plot_data)
    # note legend goes after plot
    plt.legend(["in_sample_error", "out_sample_error"])
    plt.xlabel("Number of Leafs")
    plt.ylabel("RMSE")
    plt.title("Experiment 2")
    plt.savefig("experiment_2.png")


    # --------------------------------- #
    # ---------- Experiment 3 --------- #
    # --------------------------------- #

    num_trials = 50
    num_leafs = 30

    dt_time_to_build = []
    rt_time_to_build = []
    dt_mae = []
    rt_mae = []

    learners = [[], []]  # [0] = dt objs, [1] = rt objs
    for i in range(num_trials):
        learners[0].append(dt.DTLearner(leaf_size=num_leafs))
        learners[1].append(rt.RTLearner(leaf_size=num_leafs))

    for i in range(num_trials):
        trial_train_x, trial_train_y = get_sample(train_x, train_y)
        # DT
        dt_now = datetime.now()
        learners[0][i].add_evidence(trial_train_x, trial_train_y)
        dt_later = datetime.now()
        dt_pred_y = learners[0][i].query(test_x)
        # time
        dt_diff = (dt_later - dt_now).total_seconds()
        dt_time_to_build.append(dt_diff)
        # mae
        dt_mae_temp = float(abs(test_y - dt_pred_y).sum() / test_y.shape[0])
        dt_mae.append(dt_mae_temp)

        # RT
        rt_now = datetime.now()
        learners[1][i].add_evidence(trial_train_x, trial_train_y)
        rt_later = datetime.now()
        rt_pred_y = learners[1][i].query(test_x)
        # time
        rt_diff = (rt_later - rt_now).total_seconds()
        rt_time_to_build.append(rt_diff)
        # mae
        rt_mae_temp = float(abs(test_y - rt_pred_y).sum() / test_y.shape[0])
        rt_mae.append(rt_mae_temp)

    dt_time_to_build = np.array(dt_time_to_build).reshape((-1, 1))
    rt_time_to_build = np.array(rt_time_to_build).reshape((-1, 1))

    dt_mae = np.array(dt_mae).reshape((-1, 1))
    rt_mae = np.array(rt_mae).reshape((-1, 1))

    time_plot_data = np.append(dt_time_to_build, rt_time_to_build, axis=1)

    # Plot for time
    plt.figure(3)
    plt.axis([0, num_trials, np.amin(time_plot_data)-0.001, np.amax(time_plot_data) + 0.001])
    plt.plot(time_plot_data)
    # note legend goes after plot
    plt.legend(["dt_time_to_build", "rt_time_to_build"])
    plt.xlabel("Trials")
    plt.ylabel("Time to build")
    plt.title("Experiment 3 - Time to Build DT vs RT")
    plt.savefig("experiment_3_time.png")

    mae_plot_data = np.append(dt_mae, rt_mae, axis=1)

    # Plot for mae
    plt.figure(4)
    plt.axis([0, num_trials, np.amin(mae_plot_data)-0.001, np.amax(mae_plot_data) + 0.001])
    plt.plot(mae_plot_data)
    # note legend goes after plot
    plt.legend(["dt_mae", "rt_mae"])
    plt.xlabel("Trials")
    plt.ylabel("Mean Absolute Error")
    plt.title("Experiment 3 - MAE DT vs RT")
    plt.savefig("experiment_3_mae.png")

    # stats for report
    dt_average_build_time = np.mean(dt_time_to_build)  # 0.03663862
    rt_average_build_time = np.mean(rt_time_to_build)  # 0.00613308
    dt_average_mae = np.mean(dt_mae)  # 0.00502806
    rt_average_mae = np.mean(rt_mae)  # 0.0053026714

    diff_total_err = dt_average_mae - rt_average_mae # -0.0002

    pass