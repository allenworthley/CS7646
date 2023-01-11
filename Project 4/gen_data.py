""""""
"""  		  	   		   	 		  		  		    	 		 		   		 		  
template for generating data to fool learners (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
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
GT User ID: mworthley (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import math
import numpy as np



def sub_divide(orig_data, data=np.array([])):

    # Random split
    var = np.random.randint(low=0, high=orig_data.shape[1])

    # Base Case
    if not data.shape[0] and data.shape[0] < 14:
        return orig_data

    # Recursive operations
    filter_arr = np.array([data[i, var] >= np.median(data[:, var]) for i in range(data.shape[0])])
    data[filter_arr, var] = np.power(data[filter_arr, var], 2)


    return sub_divide(orig_data, data=data[filter_arr, :])


# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees  		  	   		   	 		  		  		    	 		 		   		 		  
def best_4_lin_reg(seed=1):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		   	 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with LinRegLearner than DTLearner.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
    """

    # Linear models do great when Y is a linear function of the variables
    # and all variables are statistically important

    np.random.seed(seed)
    x = np.random.random(size=(100, 3))
    y = x[:, 0] + x[:, 1] + x[:, 2]

    return x, y


def best_4_dt(seed=1):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 		  		  		    	 		 		   		 		  
    The data set should include from 2 to 10 columns in X, and one column in Y.  		  	   		   	 		  		  		    	 		 		   		 		  
    The data should contain from 10 (minimum) to 1000 (maximum) rows.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param seed: The random seed for your data generation.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type seed: int  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: Returns data that performs significantly better with DTLearner than LinRegLearner.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: numpy.ndarray  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    # linear models don't perform with non-linear models like parabola
    np.random.seed(seed)
    x = np.random.random(size=(100, 5))
    y = x[:, 0] + np.power(x[:, 0], 2) + np.power(x[:, 0], 3)
    return x, y


def author():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return "mworthley3"  # Change this to your user ID


if __name__ == "__main__":
    print("they call me Tim.")
