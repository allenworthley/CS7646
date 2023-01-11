
Description of Files:

QLearner.py - contains the class QLearner which constructs a Q table, and has functionality to query the Q table. StrategyLearner.py
passes a state and reward in the Query funtion. This method updates the q table. Strategy learner also receives an action 
when it uses the test_query function based on the current value of State. 

ManualStrategy.py - constructs a simple strategy by using the indicator functions in indicators.py. Only has the TestPolicy function
within the class. This file also uses functions outside of the main class to clean the raw long or short signal into trade 
orders (create_trade_orders) or create a order file (create_orders_file) that can be consumed by marketsimcode.py's 
compute_portvals function. The create_orders_file function ensures that holdings never exceed +/- 1000.

StrategyLearner.py - this file essentially wraps QLearner.py's QLearner method and adapts it to the trading problem. The file
first pulls pricing data given a data range and calls three functions from indicators.py to calc the corresponding indicators.
It then discretizes each indicator using the function create_bins. Once discretized, the file trains the QLearner through
several trials or epochs. The file also creates a "reward" function by passing the QLearner the percent returns as the 
reward for each day. The reward is either positive or negative depending on the action of hold, long or short. Finally, this
file also uses the create_trade_orders and create_orders_file to clean up the raw long/short signal into trade orders that 
are consistent with the holding requirements.

indicators.py - houses all indicator functions. Each function takes a dataframe/series of pricing data to construct the 
respective index and a trade_signal flag. This flag tells each function to either return a long/hold/sell signal or
a continuous series of data. The ManualStrategy class uses the signal while the StrategyLearner uses the continous data.
Note: considerable changes have been made to improve the performance of this file relative to the version in project 6.
Given these changes, I wasn't able to exactly replicate the values of MACD in project 6 due to how pandas calculates ewm.
However, the MACD in this file qualitatively matches the data in project 6.

experiment1.py - code to organize chart creation and calculations for experiment1. This has the main function run_experiment1
which is called in testproject.py. This function takes no arguments as each variable is set within the function. Also, this
file uses compute_portvals to get the portfolio value for the experiment and uses the ManualStrategy and StrategyLearner classes.
 
experiment2.py - code to organize chart creation and calculations for experiment1. This has the main function run_experiment1
which is called in testproject.py. This function takes no arguments as each variable is set within the function. Also, this
file uses compute_portvals to get the portfolio value for the experiment and uses the StrategyLearner class.
 
marketsimcode.py - main use for this file is calculating the portfolio value given a series of prices. It's leveraged in 
experiment1.py, experiment2.py, and testproject.py

testproject.py - calculates all charts and calculated statistics for initial evaluation of ManualStrategy class and runs
experiment functions to create dependent charts. 


Directions on how to run code:

Step1: Run testproject.py - this creates all charts, tables for evaluations and experiments

Note: testproject.py sets a seed for reproducible charts. 





