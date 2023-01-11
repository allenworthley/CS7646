Instructions on how to run code

indicators.py - This file takes adj close, high, low, and volume data to produce 5 indicators using the UTIL get_data
    functions. Each indicator function takes one required argument, adj_close which is the adjusted close price. The
    money_flow function takes all of the other pricing data (low, high, volume). This file also contains some useful
    plotting functions I made to clean up the code; each plotting function takes the data and a title.

TheoreticallyOptimalStrategy.py - This file calculates the TOS trades dataframe for processing in the testproject file.
    It only has one function, testPolicy() which takes in trade data to calculate returns and determine trade signal.

marketsimcode.py - Used in conjunction with the TOS file. This file takes in an orders dataframe, start date, and end date
    to calculate portfolio value. Holding values were not explicitly limited but were mathematically constrained. Also,
    the order df needs to be in the same format as the orders-01.csv file that was given.

testproject.py - This file contains all necessary code to run the project. It's split up into two sections corresponding
    to the project sections. Note, for get_data you must use a list element for the symbol, symbol=["Insert_Ticker"].
    For what ever reason, couldn't get it to work any other way. Could be a version problem.