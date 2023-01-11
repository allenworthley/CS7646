"""MC3-H1: Best4{LR,DT} - grading script.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Usage:  		  	   		   	 		  		  		    	 		 		   		 		  
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).  		  	   		   	 		  		  		    	 		 		   		 		  
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:  		  	   		   	 		  		  		    	 		 		   		 		  
    PYTHONPATH=ml4t:MC3-P1/jdoe7 python ml4t/mc3_p1_grading/grade_learners.py  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
  		  	   		   	 		  		  		    	 		 		   		 		  
import functools  		  	   		   	 		  		  		    	 		 		   		 		  
import math  		  	   		   	 		  		  		    	 		 		   		 		  
import os  		  	   		   	 		  		  		    	 		 		   		 		  
import sys  		  	   		   	 		  		  		    	 		 		   		 		  
import time  		  	   		   	 		  		  		    	 		 		   		 		  
import traceback as tb  		  	   		   	 		  		  		    	 		 		   		 		  
from collections import namedtuple  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
import pytest  		  	   		   	 		  		  		    	 		 		   		 		  
from DTLearner import DTLearner  		  	   		   	 		  		  		    	 		 		   		 		  
from grading.grading import (  		  	   		   	 		  		  		    	 		 		   		 		  
    GradeResult,  		  	   		   	 		  		  		    	 		 		   		 		  
    IncorrectOutput,  		  	   		   	 		  		  		    	 		 		   		 		  
    grader,  		  	   		   	 		  		  		    	 		 		   		 		  
    run_with_timeout,  		  	   		   	 		  		  		    	 		 		   		 		  
    time_limit,  		  	   		   	 		  		  		    	 		 		   		 		  
)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# These two lines will be commented out in the final grading script.  		  	   		   	 		  		  		    	 		 		   		 		  
from LinRegLearner import LinRegLearner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
seconds_per_test_case = 5  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
max_points = 100.0  		  	   		   	 		  		  		    	 		 		   		 		  
html_pre_block = (  		  	   		   	 		  		  		    	 		 		   		 		  
    True  # surround comments with HTML <pre> tag (for T-Square comments field)  		  	   		   	 		  		  		    	 		 		   		 		  
)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# Test cases  		  	   		   	 		  		  		    	 		 		   		 		  
Best4TestCase = namedtuple(  		  	   		   	 		  		  		    	 		 		   		 		  
    "Best4TestCase",  		  	   		   	 		  		  		    	 		 		   		 		  
    [  		  	   		   	 		  		  		    	 		 		   		 		  
        "description",  		  	   		   	 		  		  		    	 		 		   		 		  
        "group",  		  	   		   	 		  		  		    	 		 		   		 		  
        "max_tests",  		  	   		   	 		  		  		    	 		 		   		 		  
        "needed_wins",  		  	   		   	 		  		  		    	 		 		   		 		  
        "row_limits",  		  	   		   	 		  		  		    	 		 		   		 		  
        "col_limits",  		  	   		   	 		  		  		    	 		 		   		 		  
        "seed",  		  	   		   	 		  		  		    	 		 		   		 		  
    ],  		  	   		   	 		  		  		    	 		 		   		 		  
)  		  	   		   	 		  		  		    	 		 		   		 		  
best4_test_cases = [  		  	   		   	 		  		  		    	 		 		   		 		  
    Best4TestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Test Case 1: Best4LinReg",  		  	   		   	 		  		  		    	 		 		   		 		  
        group="best4lr",  		  	   		   	 		  		  		    	 		 		   		 		  
        max_tests=15,  		  	   		   	 		  		  		    	 		 		   		 		  
        needed_wins=10,  		  	   		   	 		  		  		    	 		 		   		 		  
        row_limits=(10, 1000),  		  	   		   	 		  		  		    	 		 		   		 		  
        col_limits=(2, 10),  		  	   		   	 		  		  		    	 		 		   		 		  
        seed=1489683274,  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    Best4TestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Test Case 2: Best4DT",  		  	   		   	 		  		  		    	 		 		   		 		  
        group="best4dt",  		  	   		   	 		  		  		    	 		 		   		 		  
        max_tests=15,  		  	   		   	 		  		  		    	 		 		   		 		  
        needed_wins=10,  		  	   		   	 		  		  		    	 		 		   		 		  
        row_limits=(10, 1000),  		  	   		   	 		  		  		    	 		 		   		 		  
        col_limits=(2, 10),  		  	   		   	 		  		  		    	 		 		   		 		  
        seed=1489683274,  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    Best4TestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Test for author() method",  		  	   		   	 		  		  		    	 		 		   		 		  
        group="author",  		  	   		   	 		  		  		    	 		 		   		 		  
        max_tests=None,  		  	   		   	 		  		  		    	 		 		   		 		  
        needed_wins=None,  		  	   		   	 		  		  		    	 		 		   		 		  
        row_limits=None,  		  	   		   	 		  		  		    	 		 		   		 		  
        col_limits=None,  		  	   		   	 		  		  		    	 		 		   		 		  
        seed=None,  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# Test functon(s)  		  	   		   	 		  		  		    	 		 		   		 		  
@pytest.mark.parametrize(  		  	   		   	 		  		  		    	 		 		   		 		  
    "description,group,max_tests,needed_wins,row_limits,col_limits,seed",  		  	   		   	 		  		  		    	 		 		   		 		  
    best4_test_cases,  		  	   		   	 		  		  		    	 		 		   		 		  
)  		  	   		   	 		  		  		    	 		 		   		 		  
def test_learners(  		  	   		   	 		  		  		    	 		 		   		 		  
    description,  		  	   		   	 		  		  		    	 		 		   		 		  
    group,  		  	   		   	 		  		  		    	 		 		   		 		  
    max_tests,  		  	   		   	 		  		  		    	 		 		   		 		  
    needed_wins,  		  	   		   	 		  		  		    	 		 		   		 		  
    row_limits,  		  	   		   	 		  		  		    	 		 		   		 		  
    col_limits,  		  	   		   	 		  		  		    	 		 		   		 		  
    seed,  		  	   		   	 		  		  		    	 		 		   		 		  
    grader,  		  	   		   	 		  		  		    	 		 		   		 		  
):  		  	   		   	 		  		  		    	 		 		   		 		  
    """Test data generation methods beat given learner.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    Requires test description, test case group, and a grader fixture.  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    points_earned = 0.0  # initialize points for this test case  		  	   		   	 		  		  		    	 		 		   		 		  
    incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
    msgs = []  		  	   		   	 		  		  		    	 		 		   		 		  
    try:  		  	   		   	 		  		  		    	 		 		   		 		  
        data_x, data_y = None, None  		  	   		   	 		  		  		    	 		 		   		 		  
        same_data_x, same_data_y = None, None  		  	   		   	 		  		  		    	 		 		   		 		  
        diff_data_x, diff_data_y = None, None  		  	   		   	 		  		  		    	 		 		   		 		  
        better_learner, worse_learner = None, None  		  	   		   	 		  		  		    	 		 		   		 		  
        if group == "author":  		  	   		   	 		  		  		    	 		 		   		 		  
            try:  		  	   		   	 		  		  		    	 		 		   		 		  
                from gen_data import author  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
                auth_string = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    author, seconds_per_test_case, (), {}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                if auth_string == "tb34":  		  	   		   	 		  		  		    	 		 		   		 		  
                    incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                    msgs.append("   Incorrect author name (tb34)")  		  	   		   	 		  		  		    	 		 		   		 		  
                    points_earned = -10  		  	   		   	 		  		  		    	 		 		   		 		  
                elif auth_string == "":  		  	   		   	 		  		  		    	 		 		   		 		  
                    incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                    msgs.append("   Empty author name")  		  	   		   	 		  		  		    	 		 		   		 		  
                    points_earned = -10  		  	   		   	 		  		  		    	 		 		   		 		  
                else:  		  	   		   	 		  		  		    	 		 		   		 		  
                    incorrect = False  		  	   		   	 		  		  		    	 		 		   		 		  
            except Exception as e:  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "   Exception occured when calling author() method: {}"  		  	   		   	 		  		  		    	 		 		   		 		  
                    .format(e)  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                points_earned = -10  		  	   		   	 		  		  		    	 		 		   		 		  
        else:  		  	   		   	 		  		  		    	 		 		   		 		  
            if group == "best4dt":  		  	   		   	 		  		  		    	 		 		   		 		  
                from gen_data import best_4_dt  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
                data_x, data_y = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    best_4_dt, seconds_per_test_case, (), {"seed": seed}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                same_data_x, same_data_y = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    best_4_dt, seconds_per_test_case, (), {"seed": seed}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                diff_data_x, diff_data_y = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    best_4_dt, seconds_per_test_case, (), {"seed": seed + 1}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                better_learner = DTLearner  		  	   		   	 		  		  		    	 		 		   		 		  
                worse_learner = LinRegLearner  		  	   		   	 		  		  		    	 		 		   		 		  
            elif group == "best4lr":  		  	   		   	 		  		  		    	 		 		   		 		  
                from gen_data import best_4_lin_reg  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
                data_x, data_y = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    best_4_lin_reg, seconds_per_test_case, (), {"seed": seed}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                same_data_x, same_data_y = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    best_4_lin_reg, seconds_per_test_case, (), {"seed": seed}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                diff_data_x, diff_data_y = run_with_timeout(  		  	   		   	 		  		  		    	 		 		   		 		  
                    best_4_lin_reg, seconds_per_test_case, (), {"seed": seed + 1}  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                better_learner = LinRegLearner  		  	   		   	 		  		  		    	 		 		   		 		  
                worse_learner = DTLearner  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
            num_samples = data_x.shape[0]  		  	   		   	 		  		  		    	 		 		   		 		  
            cutoff = int(num_samples * 0.6)  		  	   		   	 		  		  		    	 		 		   		 		  
            worse_better_err = []  		  	   		   	 		  		  		    	 		 		   		 		  
            for run in range(max_tests):  		  	   		   	 		  		  		    	 		 		   		 		  
                permutation = np.random.permutation(num_samples)  		  	   		   	 		  		  		    	 		 		   		 		  
                train_x, train_y = (  		  	   		   	 		  		  		    	 		 		   		 		  
                    data_x[permutation[:cutoff]],  		  	   		   	 		  		  		    	 		 		   		 		  
                    data_y[permutation[:cutoff]],  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                test_x, test_y = (  		  	   		   	 		  		  		    	 		 		   		 		  
                    data_x[permutation[cutoff:]],  		  	   		   	 		  		  		    	 		 		   		 		  
                    data_y[permutation[cutoff:]],  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                better = better_learner()  		  	   		   	 		  		  		    	 		 		   		 		  
                worse = worse_learner()  		  	   		   	 		  		  		    	 		 		   		 		  
                better.add_evidence(train_x, train_y)  		  	   		   	 		  		  		    	 		 		   		 		  
                worse.add_evidence(train_x, train_y)  		  	   		   	 		  		  		    	 		 		   		 		  
                better_pred = better.query(test_x)  		  	   		   	 		  		  		    	 		 		   		 		  
                worse_pred = worse.query(test_x)  		  	   		   	 		  		  		    	 		 		   		 		  
                better_err = np.linalg.norm(test_y - better_pred)  		  	   		   	 		  		  		    	 		 		   		 		  
                worse_err = np.linalg.norm(test_y - worse_pred)  		  	   		   	 		  		  		    	 		 		   		 		  
                worse_better_err.append((worse_err, better_err))  		  	   		   	 		  		  		    	 		 		   		 		  
            worse_better_err.sort(  		  	   		   	 		  		  		    	 		 		   		 		  
                key=functools.cmp_to_key(  		  	   		   	 		  		  		    	 		 		   		 		  
                    lambda a, b: int((b[0] - b[1]) - (a[0] - a[1]))  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
            )  		  	   		   	 		  		  		    	 		 		   		 		  
            better_wins_count = 0  		  	   		   	 		  		  		    	 		 		   		 		  
            for worse_err, better_err in worse_better_err:  		  	   		   	 		  		  		    	 		 		   		 		  
                if better_err < 0.9 * worse_err:  		  	   		   	 		  		  		    	 		 		   		 		  
                    better_wins_count = better_wins_count + 1  		  	   		   	 		  		  		    	 		 		   		 		  
                    points_earned += 5.0  		  	   		   	 		  		  		    	 		 		   		 		  
                if better_wins_count >= needed_wins:  		  	   		   	 		  		  		    	 		 		   		 		  
                    break  		  	   		   	 		  		  		    	 		 		   		 		  
            incorrect = False  		  	   		   	 		  		  		    	 		 		   		 		  
            if (data_x.shape[0] < row_limits[0]) or (  		  	   		   	 		  		  		    	 		 		   		 		  
                data_x.shape[0] > row_limits[1]  		  	   		   	 		  		  		    	 		 		   		 		  
            ):  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "    Invalid number of rows. Should be between {},"  		  	   		   	 		  		  		    	 		 		   		 		  
                    " found {}".format(row_limits, data_x.shape[0])  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                points_earned = max(0, points_earned - 20)  		  	   		   	 		  		  		    	 		 		   		 		  
            if (data_x.shape[1] < col_limits[0]) or (  		  	   		   	 		  		  		    	 		 		   		 		  
                data_x.shape[1] > col_limits[1]  		  	   		   	 		  		  		    	 		 		   		 		  
            ):  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "    Invalid number of columns. Should be between {},"  		  	   		   	 		  		  		    	 		 		   		 		  
                    " found {}".format(col_limits, data_x.shape[1])  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                points_earned = max(0, points_earned - 20)  		  	   		   	 		  		  		    	 		 		   		 		  
            if better_wins_count < needed_wins:  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "    Better learner did not exceed worse learner. Expected"  		  	   		   	 		  		  		    	 		 		   		 		  
                    " {}, found {}".format(needed_wins, better_wins_count)  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
            if not (np.array_equal(same_data_y, data_y)) or not (  		  	   		   	 		  		  		    	 		 		   		 		  
                np.array_equal(same_data_x, data_x)  		  	   		   	 		  		  		    	 		 		   		 		  
            ):  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "    Did not produce the same data with the same seed.\n"  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      First data_x:\n{}\n".format(data_x)  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      Second data_x:\n{}\n".format(same_data_x)  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      First data_y:\n{}\n".format(data_y)  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      Second data_y:\n{}\n".format(same_data_y)  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                points_earned = max(0, points_earned - 20)  		  	   		   	 		  		  		    	 		 		   		 		  
            if np.array_equal(diff_data_y, data_y) and np.array_equal(  		  	   		   	 		  		  		    	 		 		   		 		  
                diff_data_x, data_x  		  	   		   	 		  		  		    	 		 		   		 		  
            ):  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "    Did not produce different data with different"  		  	   		   	 		  		  		    	 		 		   		 		  
                    " seeds.\n"  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      First data_x:\n{}\n".format(data_x)  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      Second data_x:\n{}\n".format(diff_data_x)  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      First data_y:\n{}\n".format(data_y)  		  	   		   	 		  		  		    	 		 		   		 		  
                    + "      Second data_y:\n{}\n".format(diff_data_y)  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                points_earned = max(0, points_earned - 20)  		  	   		   	 		  		  		    	 		 		   		 		  
        if incorrect:  		  	   		   	 		  		  		    	 		 		   		 		  
            if group == "author":  		  	   		   	 		  		  		    	 		 		   		 		  
                raise IncorrectOutput(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "Test failed on one or more criteria.\n  {}".format(  		  	   		   	 		  		  		    	 		 		   		 		  
                        "\n".join(msgs)  		  	   		   	 		  		  		    	 		 		   		 		  
                    )  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
            else:  		  	   		   	 		  		  		    	 		 		   		 		  
                inputs_str = "    Residuals: {}".format(worse_better_err)  		  	   		   	 		  		  		    	 		 		   		 		  
                raise IncorrectOutput(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "Test failed on one or more output criteria.\n "  		  	   		   	 		  		  		    	 		 		   		 		  
                    " Inputs:\n{}\n  Failures:\n{}".format(  		  	   		   	 		  		  		    	 		 		   		 		  
                        inputs_str, "\n".join(msgs)  		  	   		   	 		  		  		    	 		 		   		 		  
                    )  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
        else:  		  	   		   	 		  		  		    	 		 		   		 		  
            if group != "author":  		  	   		   	 		  		  		    	 		 		   		 		  
                avg_ratio = 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
                worse_better_err.sort(  		  	   		   	 		  		  		    	 		 		   		 		  
                    key=functools.cmp_to_key(  		  	   		   	 		  		  		    	 		 		   		 		  
                        lambda a, b: int(  		  	   		   	 		  		  		    	 		 		   		 		  
                            np.sign((b[0] - b[1]) - (a[0] - a[1]))  		  	   		   	 		  		  		    	 		 		   		 		  
                        )  		  	   		   	 		  		  		    	 		 		   		 		  
                    )  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
                for we, be in worse_better_err[:10]:  		  	   		   	 		  		  		    	 		 		   		 		  
                    avg_ratio += float(we) - float(be)  		  	   		   	 		  		  		    	 		 		   		 		  
                avg_ratio = avg_ratio / 10.0  		  	   		   	 		  		  		    	 		 		   		 		  
                if group == "best4dt":  		  	   		   	 		  		  		    	 		 		   		 		  
                    grader.add_performance(np.array([avg_ratio, 0]))  		  	   		   	 		  		  		    	 		 		   		 		  
                else:  		  	   		   	 		  		  		    	 		 		   		 		  
                    grader.add_performance(np.array([0, avg_ratio]))  		  	   		   	 		  		  		    	 		 		   		 		  
    except Exception as e:  		  	   		   	 		  		  		    	 		 		   		 		  
        # Test result: failed  		  	   		   	 		  		  		    	 		 		   		 		  
        msg = "Description: {} (group: {})\n".format(description, group)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)  		  	   		   	 		  		  		    	 		 		   		 		  
        tb_list = tb.extract_tb(sys.exc_info()[2])  		  	   		   	 		  		  		    	 		 		   		 		  
        for i in range(len(tb_list)):  		  	   		   	 		  		  		    	 		 		   		 		  
            row = tb_list[i]  		  	   		   	 		  		  		    	 		 		   		 		  
            tb_list[i] = (  		  	   		   	 		  		  		    	 		 		   		 		  
                os.path.basename(row[0]),  		  	   		   	 		  		  		    	 		 		   		 		  
                row[1],  		  	   		   	 		  		  		    	 		 		   		 		  
                row[2],  		  	   		   	 		  		  		    	 		 		   		 		  
                row[3],  		  	   		   	 		  		  		    	 		 		   		 		  
            )  # show only filename instead of long absolute path  		  	   		   	 		  		  		    	 		 		   		 		  
        tb_list = [row for row in tb_list if (row[0] == "gen_data.py")]  		  	   		   	 		  		  		    	 		 		   		 		  
        if tb_list:  		  	   		   	 		  		  		    	 		 		   		 		  
            msg += "Traceback:\n"  		  	   		   	 		  		  		    	 		 		   		 		  
            msg += "".join(tb.format_list(tb_list))  # contains newlines  		  	   		   	 		  		  		    	 		 		   		 		  
        elif "grading_traceback" in dir(e):  		  	   		   	 		  		  		    	 		 		   		 		  
            msg += "Traceback:\n"  		  	   		   	 		  		  		    	 		 		   		 		  
            msg += "".join(tb.format_list(e.grading_traceback))  		  	   		   	 		  		  		    	 		 		   		 		  
        msg += "{}: {}".format(e.__class__.__name__, str(e))  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # Report failure result to grader, with stacktrace  		  	   		   	 		  		  		    	 		 		   		 		  
        grader.add_result(  		  	   		   	 		  		  		    	 		 		   		 		  
            GradeResult(outcome="failed", points=points_earned, msg=msg)  		  	   		   	 		  		  		    	 		 		   		 		  
        )  		  	   		   	 		  		  		    	 		 		   		 		  
        raise  		  	   		   	 		  		  		    	 		 		   		 		  
    else:  		  	   		   	 		  		  		    	 		 		   		 		  
        # Test result: passed (no exceptions)  		  	   		   	 		  		  		    	 		 		   		 		  
        grader.add_result(  		  	   		   	 		  		  		    	 		 		   		 		  
            GradeResult(outcome="passed", points=points_earned, msg=None)  		  	   		   	 		  		  		    	 		 		   		 		  
        )  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		   	 		  		  		    	 		 		   		 		  
    pytest.main(["-s", __file__])  		  	   		   	 		  		  		    	 		 		   		 		  
