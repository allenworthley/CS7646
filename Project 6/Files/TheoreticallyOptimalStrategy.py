import numpy as np
import datetime as dt
import pandas as pd
from util import get_data, plot_data


def author(self):
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "mworthley3"  # replace tb34 with your Georgia Tech username


def testPolicy(symbol="AAPL", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
    # Get Data
    data = get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop(columns=["SPY"])

    # Get Returns
    rets = data.diff()

    # Create buy / sell signal, close position next day
    ## if rets are positive buy day prior, if negative short sell day prior
    trade_signal = rets.where(rets > 0, "BUY").where(rets < 0, "SELL")
    trade_signal = trade_signal.shift(-1)

    trades = pd.DataFrame(data=0.000, columns=["shares"], index=trade_signal.index.values)

    # Close out next day
    for i in range(trade_signal.shape[0]):
        # Factor flips position impact calculation
        # Buy add 1000 today, sells tomorrow
        # Sell subtracts today, buys tomorrow
        if trade_signal.iloc[i, 0] == "BUY":
            factor = 1
        else:
            factor = -1

        trades.shares.iloc[i] += 1000 * factor
        if i + 1 < trade_signal.shape[0]:
            trades.shares.iloc[i + 1] -= 1000 * factor

    return trades

