""""""

"""  		  	   		   	 		  		  		    	 		 		   		 		  
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
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
import datetime
import datetime as dt
import pandas as pd
import random as rand
import numpy as np
import util as ut
import matplotlib.pyplot as plt

# files
import indicators as ind
import marketsimcode as mksim
import QLearner as qlt


def author():
    return "mworthley3"


def create_trade_orders(signals):
    trades = pd.DataFrame(data=0, columns=["shares"], index=signals.index.values)

    # Clean up trade signals into trade orders
    # restricts trades to only holding at max -1000, 0, 1000 shares at a time

    if signals.iloc[0] != 0:
        trades.iloc[0] = signals.iloc[0]

    # for remaining
    for i in range(1, trades.shape[0]):
        # Previously zero and has position
        if signals.iloc[i] != 0 and signals.iloc[i - 1] == 0:
            trades.iloc[i] = signals.iloc[i] * 1

        # Zero after trade - close out signal
        elif signals.iloc[i] == 0 and signals.iloc[i - 1] != 0:
            trades.iloc[i] = signals.iloc[i - 1] * -1

        # Switching positions on the next day
        elif signals.iloc[i] != signals.iloc[i - 1] and signals.iloc[i - 1] != 0:
            trades.iloc[i] = signals.iloc[i - 1] * -2

        # Taking same position on the next day
        elif signals.iloc[i] == signals.iloc[i - 1] and signals.iloc[i - 1] != 0:
            trades.iloc[i] = 0

    return trades * 1000


def create_orders_file(clean_trades, stock="JPM"):
    # Create psuedo orders file with trade signals
    orders = pd.DataFrame(index=clean_trades.index.values, columns=["Symbol", "Order", "Shares"])

    # Clean
    orders["Symbol"] = stock
    orders["Order"] = clean_trades.where(clean_trades < 1, "BUY").where(clean_trades >= 0, "SELL")
    orders["Shares"] = clean_trades
    orders = orders.drop(orders[orders.Order == 0].index)

    return orders


class StrategyLearner(object):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param verbose: If “verbose” is True, your code can print out information for debugging.  		  	   		   	 		  		  		    	 		 		   		 		  
        If verbose = False your code should not generate ANY output.  		  	   		   	 		  		  		    	 		 		   		 		  
    :type verbose: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    :param impact: The market impact of each transaction, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type impact: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :param commission: The commission amount charged, defaults to 0.0  		  	   		   	 		  		  		    	 		 		   		 		  
    :type commission: float  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    bin1 = []
    bin2 = []
    bin3 = []

    # constructor  		  	   		   	 		  		  		    	 		 		   		 		  
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Constructor method  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.num_bins = 10

        pass

    def author(self):
        return "mworthley3"

    def create_bins(self, var1, var2, var3):
        bins = 10
        step_size = int(len(var1)/bins)

        # sort data
        s1 = np.sort(var1)
        s2 = np.sort(var2)
        s3 = np.sort(var3)

        # stack
        v1 = np.vstack((s1, s2))
        v2 = np.vstack((v1, s3))

        # Create bins
        for i in range(0, len(var1), step_size):
            if i > 0:
                # Lower and upper bound
                self.bin1.append([s1[i - step_size], s1[i]])
                self.bin2.append([s2[i - step_size], s2[i]])
                self.bin3.append([s3[i - step_size], s3[i]])
        pass

    def get_state_from_bins(self, var1, var2, var3):
        ind1 = 0
        ind2 = 0
        ind3 = 0
        for i in range(len(self.bin1), 0, -1):
            if var1 < self.bin1[i-1][1]:
                ind1 = i - 1
            if var2 < self.bin2[i - 1][1]:
                ind2 = i - 1
            if var3 < self.bin3[i - 1][1]:
                ind3 = i - 1
        state = int(str(ind1) + str(ind2) + str(ind3))
        return state

    def add_evidence(
            self,
            symbol="JPM",
            sd=dt.datetime(2008, 1, 1),
            ed=dt.datetime(2009, 1, 1),
            sv=100000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Trains your strategy learner over a given time frame.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol to train on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        """

        # ------- Get training and indicators data ------- #
        # note: Golden indicator is null for first 50 days of trading. This code grabs a quarter's worth of trading
        # data prior to the start date as this allows for continuous data from the intended start date.
        data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True,
                           colname="Adj Close").drop(columns=["SPY"])  # column 0

        # RSI Strategy
        data["rsi"] = ind.rsi_calc(data[symbol], time_horizon=14, trade_signal=False)  # column 1

        # MACD Strategy
        data["MACD"] = ind.MACD(data[symbol], short_term=12, long_term=26, trade_signal=False) # column 2

        # Golden Cross
        data["Golden"] = ind.golden_death_cross(data[symbol], trade_signal=False)  #column 3

        # Setup columns for iteration:
        data["trade_signal"] = 0  # column 4
        data["state"] = 0  # column 5

        # ------- Calculate returns for reward function ------- #
        data["lvl_returns"] = data[symbol] - data[symbol].shift(1)  # column 6d, aily rets, for terminal condition
        data["pct_returns"] = data[symbol].pct_change()  # column 7, daily percentage returns

        # Drop data not in the specified date range
        data = data.iloc[50:, :].to_numpy()

        # ------- Discretize indicators ------- #
        self.create_bins(data[:, 1], data[:, 2], data[:, 3])
        for i in range(len(data)):
            data[i, 5] = int(self.get_state_from_bins(data[i, 1], data[i, 2], data[i, 3]))

        # ------- Train QLearner ------- #
        # Initialize Learner Class
        self.learner = qlt.QLearner(
            num_states=1000,
            num_actions=3,
            alpha=0.2,  # learning rate, 0.2
            gamma=0.9,  # discount rate
            rar=0.9,  # random action rate, probs of selecting action
            radr=0.99,  # random action decay rate rar = rar * radr
            dyna=0,
            verbose=False
        )

        epochs = 500
        counter = 0
        scores = np.zeros((epochs, 1))

        while counter < epochs:
            # Iterate over number of epochs and keep score
            total_reward = 0
            state = int(data[0, 5])  # first state of time series
            action = self.learner.querysetstate(state)  # set the state and get first action

            data[0, 4] = action  # initialize action at zero

            # Learner starts at sd and ends at ed. Iterates over rows or days
            for i in range(1, len(data) - 1):
                # -- Reward function -- #
                # Argmax prefers index 0 if information or reward is unknown, hold if unknown
                if action == 0:
                    # hold
                    factor = 0
                elif action == 1:
                    # long
                    factor = 1
                elif action == 2:
                    # short
                    factor = -1

                reward = data[i, 7] * factor - self.impact  # pct returns, impact assumed as % of prior "price * shares"
                total_reward += data[i, 6] * factor

                # Record action history for reward function
                data[i, 4] = factor

                # Update for new state based on time series
                new_state = int(data[i, 5])

                # Provide Q table with new_state and reward for prior action
                # Update action for iteration
                action = self.learner.query(int(new_state), reward)

            # Update scores for epoch
            scores[counter, 0] = total_reward

            # Terminal Conditions
            # end epochs
            if counter > 1:
                if scores[counter, 0] - scores[counter - 1, 0] < 0:
                    break

            # Increment
            counter += 1
        pass

    # this method should use the existing policy and test it against new data
    def testPolicy(
            self,
            symbol="JPM",
            sd=dt.datetime(2010, 1, 1),
            ed=dt.datetime(2011, 12, 31),
            sv=100000,
    ):
        """  		  	   		   	 		  		  		    	 		 		   		 		  
        Tests your learner using data outside of the training data  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
        :param symbol: The stock symbol that you trained on on  		  	   		   	 		  		  		    	 		 		   		 		  
        :type symbol: str  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sd: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 		  		  		    	 		 		   		 		  
        :type ed: datetime  		  	   		   	 		  		  		    	 		 		   		 		  
        :param sv: The starting value of the portfolio  		  	   		   	 		  		  		    	 		 		   		 		  
        :type sv: int  		  	   		   	 		  		  		    	 		 		   		 		  
        :return: A DataFrame with values representing trades for each day. Legal values are +1000.0 indicating  		  	   		   	 		  		  		    	 		 		   		 		  
            a BUY of 1000 shares, -1000.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.  		  	   		   	 		  		  		    	 		 		   		 		  
            Values of +2000 and -2000 for trades are also legal when switching from long to short or short to  		  	   		   	 		  		  		    	 		 		   		 		  
            long so long as net holdings are constrained to -1000, 0, and 1000.  		  	   		   	 		  		  		    	 		 		   		 		  
        :rtype: pandas.DataFrame  		  	   		   	 		  		  		    	 		 		   		 		  
        """
        # ------- Get testing and indicators data ------- #
        # note: Golden indicator is null for first 50 days of trading. This code grabs a quarter's worth of trading
        # data prior to the start date as this allows for continuous data from the intended start date.
        test_data = ut.get_data([symbol], pd.date_range(sd, ed), addSPY=True,
                           colname="Adj Close").drop(columns=["SPY"])

        # RSI Strategy
        test_data["rsi"] = ind.rsi_calc(test_data[symbol], time_horizon=14, trade_signal=False)

        # MACD Strategy
        test_data["MACD"] = ind.MACD(test_data[symbol], short_term=12, long_term=26, trade_signal=False)

        # Golden Cross
        test_data["Golden"] = ind.golden_death_cross(test_data[symbol], trade_signal=False)
        test_data["signal"] = 0
        test_data["state"] = 0
        # Drop data with nans/zeros - limited by 50 day moving average
        # convert to numpy for speed
        df = test_data.iloc[50:, :]
        test_data = df.to_numpy()

        # ------- Create Discretized Spaces and Query Q table ------- #
        for i in range(len(test_data)):
            # Code state column
            # Get current state
            test_data[i, 5] = self.get_state_from_bins(test_data[i, 1], test_data[i, 2], test_data[i, 3])

            action = self.learner.test_query(int(test_data[i, 5]))

            if action == 0:
                # hold
                factor = 0
            elif action == 1:
                # Long
                factor = 1
            elif action == 2:
                # short
                factor = -1

            test_data[i, 4] = factor

        trade_signals = pd.DataFrame(data=test_data[:, 4], index=df.index.values)
        trades = create_trade_orders(trade_signals.squeeze())

        return trades
