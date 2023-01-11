""""""  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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

import numpy as np
import numpy.ma as ma


def remove_constant_factors(a):
    keep_index = np.array([i for i in range(a.shape[1])])
    delete_index = keep_index
    i = a.shape[1] - 1
    while i >= 0:
        flag = np.all(a[:, i] == a[0, i])
        if flag:
            keep_index = np.delete(keep_index, i)
        i -= 1

    delete_index = np.delete(delete_index, keep_index)

    new_a = np.delete(a, delete_index, axis=1)

    return new_a, keep_index


def choose_factor(x, y):
    # Edge case, nan's or zeros in the data. np.coef_arr doesn't handle nan's
    ma_x = ma.masked_invalid(x)
    ma_y = ma.masked_invalid(y)

    # Edge case 2, same values in factor column are the same. remove these factors from corrcoef
    # update dictionary map of movements of removing factors with same rows
    cleaned_x, map_to_original_index = remove_constant_factors(ma_x)

    # Find highest correlation for series of X's on Y
    #coef_arr = abs(np.corrcoef(x, y, rowvar=False)[-1])[0:x.shape[1]]
    coef_arr = abs(np.corrcoef(cleaned_x, ma_y, rowvar=False)[-1])
    # remove last row as this is y and always 1
    coef_arr = np.delete(coef_arr, -1)

    # Find factor to split on, max correlation (+/-), choose first one
    max_coef_index = np.where(coef_arr == max(coef_arr))[0][0]

    x_factor = map_to_original_index[max_coef_index]
    return x_factor


def build_tree(np_data, leaf_size):
    # Base Case
    if np_data.shape[0] <= leaf_size:
        leaf_requirement_val = float(np.mean(np_data[:, -1]))

        return np.asarray([["leaf", leaf_requirement_val, "NA", "NA"]])
    elif np.all(np_data[:, -1] == np_data[0, -1]):
        same_y_val = float(np_data[0, -1])
        return np.asarray([["leaf", same_y_val, "NA", "NA"]])

    # Recursive Operations
    factor = choose_factor(np_data[:, 0:np_data.shape[1] - 1], np_data[:, -1])
    split_val = np.median(np_data[:, factor])

    # --- Edge case --- #
    # Stuck in recursion when split_val doesn't returned spliced np_data, return mean to split left/right
    # if resulting split returns an empty right side branch, change split value to mean calc
    # occurs because repeated sample rows have same value and median,
    if np_data[np_data[:, factor] > split_val].shape[0] == 0:
        split_val = np.mean(np_data[:, factor])
        # sad how long this took to figure out, i hope its straightforward grading this

    ## Recursive Call
    left_tree = build_tree(np_data[np_data[:, factor] <= split_val], leaf_size)
    right_tree = build_tree(np_data[np_data[:, factor] > split_val], leaf_size)

    # Organize the data
    root = np.array(["x" + str(factor), split_val, 1, left_tree.shape[0] + 1])  # once tree has been constructed

    return np.vstack((np.vstack((root, left_tree)), right_tree))


class DTLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    This is a Linear Regression Learner. It is implemented correctly.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size, verbose=False, tree=None):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.tree = tree
        self.leaf_size = leaf_size
        self.verbose = verbose

        pass  # move along, these aren't the drones you're looking for

    def author(self):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        return "mworthley3"  # replace tb34 with your Georgia Tech username

    def add_evidence(self, data_x, data_y):  		  	   		   	 		  		  		    	 		 		   		 		  
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Add training data to learner, construct self.tree 		  	   		   	 		  		  		    	 		 		   		 		   	   		   	 		  		  		    	 		 		   		 		  
        :param data_x: A set of feature values used to train the learner  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_x: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        :param data_y: The value we are attempting to predict given the X data  		  	   		   	 		  		  		    	 		 		   		 		  
        :type data_y: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        # Convert data_y into correct size for concatenation
        if len(data_y.shape) == 1:
            data_y = np.reshape(data_y, (data_x.shape[0], 1))

        np_arr = np.append(data_x, data_y, axis=1)
        self.tree = build_tree(np_arr, leaf_size=self.leaf_size)

    def query(self, points):
        # Expecting x values 1-3 for example
        # Tree is a series of 1x4 arrays: [Factor, Split Value, Left node, Right Node
        y_hat = np.zeros((points.shape[0], ), dtype=float)
        for j in range(points.shape[0]):
            i = 0
            while i < self.tree.shape[0]:
                # pull relevant factor from points
                var = self.tree[i][0]
                if var != "leaf":
                    var_index = int(var.replace("x", ""))
                    if points[j, var_index] <= float(self.tree[i][1]):
                        # left side
                        jump_index = int(self.tree[i][2])
                    else:
                        # right side
                        jump_index = int(self.tree[i][3])
                else:
                    # once the leaf is found return y value
                    y_hat[j] = float(self.tree[i][1])
                    break
                i += jump_index

        return ma.masked_invalid(y_hat)
