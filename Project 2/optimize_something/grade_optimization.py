"""MC1-P2: Optimize a portfolio - grading script.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Usage:  		  	   		   	 		  		  		    	 		 		   		 		  
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).  		  	   		   	 		  		  		    	 		 		   		 		  
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:  		  	   		   	 		  		  		    	 		 		   		 		  
    PYTHONPATH=ml4t:MC1-P2/jdoe7 python ml4t/mc1_p2_grading/grade_optimization.py  		  	   		   	 		  		  		    	 		 		   		 		  
"""  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import datetime  		  	   		   	 		  		  		    	 		 		   		 		  
import os  		  	   		   	 		  		  		    	 		 		   		 		  
import sys  		  	   		   	 		  		  		    	 		 		   		 		  
import traceback as tb  		  	   		   	 		  		  		    	 		 		   		 		  
from collections import namedtuple  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		   	 		  		  		    	 		 		   		 		  
import pytest  		  	   		   	 		  		  		    	 		 		   		 		  
from grading.grading import GradeResult, IncorrectOutput, grader, time_limit  		  	   		   	 		  		  		    	 		 		   		 		  
from util import get_data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# from portfolio.analysis import get_portfolio_value, get_portfolio_stats  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# Student code  		  	   		   	 		  		  		    	 		 		   		 		  
# main_code = "portfolio.optimization"  # module name to import  		  	   		   	 		  		  		    	 		 		   		 		  
main_code = "optimization"  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
def str2dt(strng):  		  	   		   	 		  		  		    	 		 		   		 		  
    year, month, day = map(int, strng.split("-"))  		  	   		   	 		  		  		    	 		 		   		 		  
    return datetime.datetime(year, month, day)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# Test cases  		  	   		   	 		  		  		    	 		 		   		 		  
OptimizationTestCase = namedtuple(  		  	   		   	 		  		  		    	 		 		   		 		  
    "OptimizationTestCase", ["inputs", "outputs", "description"]  		  	   		   	 		  		  		    	 		 		   		 		  
)  		  	   		   	 		  		  		    	 		 		   		 		  
optimization_test_cases = [  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2010-01-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2010-12-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.0, 0.4, 0.6, 0.0]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Wiki example 1",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2004-01-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2006-01-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["AXP", "HPQ", "IBM", "HNZ"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.78, 0.22, 0.0, 0.0]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Wiki example 2",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2004-12-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2006-05-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["YHOO", "XOM", "GLD", "HNZ"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.0, 0.07, 0.59, 0.34]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Wiki example 3",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2005-12-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2006-05-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["YHOO", "HPQ", "GLD", "HNZ"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.0, 0.1, 0.25, 0.65]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Wiki example 4",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2005-12-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2007-05-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["MSFT", "HPQ", "GLD", "HNZ"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.0, 0.27, 0.11, 0.62]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="MSFT vs HPQ",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2006-05-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2007-05-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["MSFT", "AAPL", "GLD", "HNZ"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.42, 0.32, 0.0, 0.26]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="MSFT vs AAPL",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2011-01-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2011-12-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["AAPL", "GLD", "GOOG", "XOM"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.46, 0.37, 0.0, 0.17]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Wiki example 1 in 2011",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
    OptimizationTestCase(  		  	   		   	 		  		  		    	 		 		   		 		  
        inputs=dict(  		  	   		   	 		  		  		    	 		 		   		 		  
            start_date=str2dt("2010-01-01"),  		  	   		   	 		  		  		    	 		 		   		 		  
            end_date=str2dt("2010-12-31"),  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols=["AXP", "HPQ", "IBM", "HNZ"],  		  	   		   	 		  		  		    	 		 		   		 		  
        ),  		  	   		   	 		  		  		    	 		 		   		 		  
        outputs=dict(allocs=[0.0, 0.0, 0.0, 1.0]),  		  	   		   	 		  		  		    	 		 		   		 		  
        description="Year of the HNZ",  		  	   		   	 		  		  		    	 		 		   		 		  
    ),  		  	   		   	 		  		  		    	 		 		   		 		  
]  		  	   		   	 		  		  		    	 		 		   		 		  
abs_margins = dict(  		  	   		   	 		  		  		    	 		 		   		 		  
    sum_to_one=0.02, alloc_range=0.02, alloc_match=0.1  		  	   		   	 		  		  		    	 		 		   		 		  
)  # absolute margin of error for each component  		  	   		   	 		  		  		    	 		 		   		 		  
points_per_component = dict(  		  	   		   	 		  		  		    	 		 		   		 		  
    sum_to_one=2.0, alloc_range=2.0, alloc_match=4.0  		  	   		   	 		  		  		    	 		 		   		 		  
)  # points for each component, for partial credit  		  	   		   	 		  		  		    	 		 		   		 		  
points_per_test_case = sum(points_per_component.values())  		  	   		   	 		  		  		    	 		 		   		 		  
seconds_per_test_case = 10  # execution time limit  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# Grading parameters (picked up by module-level grading fixtures)  		  	   		   	 		  		  		    	 		 		   		 		  
max_points = float(len(optimization_test_cases) * points_per_test_case)  		  	   		   	 		  		  		    	 		 		   		 		  
html_pre_block = (  		  	   		   	 		  		  		    	 		 		   		 		  
    True  # surround comments with HTML <pre> tag (for T-Square comments field)  		  	   		   	 		  		  		    	 		 		   		 		  
)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
# Test functon(s)  		  	   		   	 		  		  		    	 		 		   		 		  
@pytest.mark.parametrize("inputs,outputs,description", optimization_test_cases)  		  	   		   	 		  		  		    	 		 		   		 		  
def test_optimization(inputs, outputs, description, grader):  		  	   		   	 		  		  		    	 		 		   		 		  
    """Test find_optimal_allocations() returns correct allocations.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    Requires test inputs, expected outputs, description, and a grader fixture.  		  	   		   	 		  		  		    	 		 		   		 		  
    """  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    points_earned = 0.0  # initialize points for this test case  		  	   		   	 		  		  		    	 		 		   		 		  
    try:  		  	   		   	 		  		  		    	 		 		   		 		  
        # Try to import student code (only once)  		  	   		   	 		  		  		    	 		 		   		 		  
        if not main_code in globals():  		  	   		   	 		  		  		    	 		 		   		 		  
            import importlib  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
            # * Import module  		  	   		   	 		  		  		    	 		 		   		 		  
            mod = importlib.import_module(main_code)  		  	   		   	 		  		  		    	 		 		   		 		  
            globals()[main_code] = mod  		  	   		   	 		  		  		    	 		 		   		 		  
            # * Import methods to test (refactored out, spring 2016, --BPH)  		  	   		   	 		  		  		    	 		 		   		 		  
            # for m in ['find_optimal_allocations']:  		  	   		   	 		  		  		    	 		 		   		 		  
            #     globals()[m] = getattr(mod, m)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # Unpack test case  		  	   		   	 		  		  		    	 		 		   		 		  
        start_date = inputs["start_date"]  		  	   		   	 		  		  		    	 		 		   		 		  
        end_date = inputs["end_date"]  		  	   		   	 		  		  		    	 		 		   		 		  
        symbols = inputs["symbols"]  # e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # Read in adjusted closing prices for given symbols, date range  		  	   		   	 		  		  		    	 		 		   		 		  
        # dates = pd.date_range(start_date, end_date)  		  	   		   	 		  		  		    	 		 		   		 		  
        # prices_all = get_data(symbols, dates)  # automatically adds SPY  		  	   		   	 		  		  		    	 		 		   		 		  
        # prices = prices_all[symbols]  # only portfolio symbols  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # Run student code with time limit (in seconds, per test case)  		  	   		   	 		  		  		    	 		 		   		 		  
        port_stats = {}  		  	   		   	 		  		  		    	 		 		   		 		  
        with time_limit(seconds_per_test_case):  		  	   		   	 		  		  		    	 		 		   		 		  
            # * Find optimal allocations  		  	   		   	 		  		  		    	 		 		   		 		  
            (  		  	   		   	 		  		  		    	 		 		   		 		  
                student_allocs,  		  	   		   	 		  		  		    	 		 		   		 		  
                student_cr,  		  	   		   	 		  		  		    	 		 		   		 		  
                student_adr,  		  	   		   	 		  		  		    	 		 		   		 		  
                student_sddr,  		  	   		   	 		  		  		    	 		 		   		 		  
                student_sr,  		  	   		   	 		  		  		    	 		 		   		 		  
            ) = optimization.optimize_portfolio(  		  	   		   	 		  		  		    	 		 		   		 		  
                sd=start_date, ed=end_date, syms=symbols, gen_plot=False  		  	   		   	 		  		  		    	 		 		   		 		  
            )  		  	   		   	 		  		  		    	 		 		   		 		  
            student_allocs = np.float32(  		  	   		   	 		  		  		    	 		 		   		 		  
                student_allocs  		  	   		   	 		  		  		    	 		 		   		 		  
            )  # make sure it's a NumPy array, for easier computation  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # Verify against expected outputs and assign points  		  	   		   	 		  		  		    	 		 		   		 		  
        incorrect = False  		  	   		   	 		  		  		    	 		 		   		 		  
        msgs = []  		  	   		   	 		  		  		    	 		 		   		 		  
        correct_allocs = outputs["allocs"]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # * Check sum_to_one: Allocations sum to 1.0 +/- margin  		  	   		   	 		  		  		    	 		 		   		 		  
        sum_allocs = np.sum(student_allocs)  		  	   		   	 		  		  		    	 		 		   		 		  
        if abs(sum_allocs - 1.0) > abs_margins["sum_to_one"]:  		  	   		   	 		  		  		    	 		 		   		 		  
            incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
            msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                "    sum of allocations: {} (expected: 1.0)".format(sum_allocs)  		  	   		   	 		  		  		    	 		 		   		 		  
            )  		  	   		   	 		  		  		    	 		 		   		 		  
            student_allocs = (  		  	   		   	 		  		  		    	 		 		   		 		  
                student_allocs / sum_allocs  		  	   		   	 		  		  		    	 		 		   		 		  
            )  # normalize allocations, if they don't sum to 1.0  		  	   		   	 		  		  		    	 		 		   		 		  
        else:  		  	   		   	 		  		  		    	 		 		   		 		  
            points_earned += points_per_component["sum_to_one"]  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # * Get daily portfolio value and statistics, for comparison  		  	   		   	 		  		  		    	 		 		   		 		  
        # port_val = get_portfolio_value(prices, allocs, start_val)  		  	   		   	 		  		  		    	 		 		   		 		  
        # cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(port_val)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        # * Check alloc_range: Each allocation is within [0.0, 1.0] +/- margin  		  	   		   	 		  		  		    	 		 		   		 		  
        # * Check alloc_match: Each allocation matches expected value +/- margin  		  	   		   	 		  		  		    	 		 		   		 		  
        points_per_alloc_range = points_per_component["alloc_range"] / len(  		  	   		   	 		  		  		    	 		 		   		 		  
            correct_allocs  		  	   		   	 		  		  		    	 		 		   		 		  
        )  		  	   		   	 		  		  		    	 		 		   		 		  
        points_per_alloc_match = points_per_component["alloc_match"] / len(  		  	   		   	 		  		  		    	 		 		   		 		  
            correct_allocs  		  	   		   	 		  		  		    	 		 		   		 		  
        )  		  	   		   	 		  		  		    	 		 		   		 		  
        for symbol, alloc, correct_alloc in zip(  		  	   		   	 		  		  		    	 		 		   		 		  
            symbols, student_allocs, correct_allocs  		  	   		   	 		  		  		    	 		 		   		 		  
        ):  		  	   		   	 		  		  		    	 		 		   		 		  
            if alloc < -abs_margins["alloc_range"] or alloc > (  		  	   		   	 		  		  		    	 		 		   		 		  
                1.0 + abs_margins["alloc_range"]  		  	   		   	 		  		  		    	 		 		   		 		  
            ):  		  	   		   	 		  		  		    	 		 		   		 		  
                incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                    "    {} - allocation out of range: {} (expected: [0.0,"  		  	   		   	 		  		  		    	 		 		   		 		  
                    " 1.0])".format(symbol, alloc)  		  	   		   	 		  		  		    	 		 		   		 		  
                )  		  	   		   	 		  		  		    	 		 		   		 		  
            else:  		  	   		   	 		  		  		    	 		 		   		 		  
                points_earned += points_per_alloc_range  		  	   		   	 		  		  		    	 		 		   		 		  
                if abs(alloc - correct_alloc) > abs_margins["alloc_match"]:  		  	   		   	 		  		  		    	 		 		   		 		  
                    incorrect = True  		  	   		   	 		  		  		    	 		 		   		 		  
                    msgs.append(  		  	   		   	 		  		  		    	 		 		   		 		  
                        "    {} - incorrect allocation: {} (expected: {})"  		  	   		   	 		  		  		    	 		 		   		 		  
                        .format(symbol, alloc, correct_alloc)  		  	   		   	 		  		  		    	 		 		   		 		  
                    )  		  	   		   	 		  		  		    	 		 		   		 		  
                else:  		  	   		   	 		  		  		    	 		 		   		 		  
                    points_earned += points_per_alloc_match  		  	   		   	 		  		  		    	 		 		   		 		  
        # points_earned = round(points_earned)  # round off points earned to nearest integer (?)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        if incorrect:  		  	   		   	 		  		  		    	 		 		   		 		  
            inputs_str = (  		  	   		   	 		  		  		    	 		 		   		 		  
                "    start_date: {}\n"  		  	   		   	 		  		  		    	 		 		   		 		  
                "    end_date: {}\n"  		  	   		   	 		  		  		    	 		 		   		 		  
                "    symbols: {}\n".format(start_date, end_date, symbols)  		  	   		   	 		  		  		    	 		 		   		 		  
            )  		  	   		   	 		  		  		    	 		 		   		 		  
            # If there are problems with the stats and all of the values returned match the template code, exactly, then award 0 points  		  	   		   	 		  		  		    	 		 		   		 		  
            # if check_template(student_allocs, student_cr, student_adr, student_sddr, student_sr):  		  	   		   	 		  		  		    	 		 		   		 		  
            points_earned = 0  		  	   		   	 		  		  		    	 		 		   		 		  
            raise IncorrectOutput(  		  	   		   	 		  		  		    	 		 		   		 		  
                "Test failed on one or more output criteria.\n  Inputs:\n{}\n "  		  	   		   	 		  		  		    	 		 		   		 		  
                " Failures:\n{}".format(inputs_str, "\n".join(msgs))  		  	   		   	 		  		  		    	 		 		   		 		  
            )  		  	   		   	 		  		  		    	 		 		   		 		  
    except Exception as e:  		  	   		   	 		  		  		    	 		 		   		 		  
        # Test result: failed  		  	   		   	 		  		  		    	 		 		   		 		  
        msg = "Test case description: {}\n".format(description)  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
        tb_list = [row for row in tb_list if row[0] == "optimization.py"]  		  	   		   	 		  		  		    	 		 		   		 		  
        if tb_list:  		  	   		   	 		  		  		    	 		 		   		 		  
            msg += "Traceback:\n"  		  	   		   	 		  		  		    	 		 		   		 		  
            msg += "".join(tb.format_list(tb_list))  # contains newlines  		  	   		   	 		  		  		    	 		 		   		 		  
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
