import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals
from util import get_data

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np


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


def run_experiment2():

    # In-sample
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000
    impt = 0
    comm = 0
    ticker = "JPM"

    # initialize datastructure
    avg_returns = []
    number_of_trades = []

    # ------------------------------------------------------------ #
    # -------------- Leaner Strategy, impact at 0 --------------- #
    # ------------------------------------------------------------ #
    impt1 = 0
    learner = sl.StrategyLearner(impact=impt)
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date)
    trades = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date)

    # Create orders file
    orders_file = create_orders_file(clean_trades=trades, stock=ticker)

    # -------------- Get Port Stats --------------- #
    port_value = compute_portvals(orders_file, start_date, end_date, commission=comm, impact=impt)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update data
    avg_returns.append(np.mean(port_value.pct_change()))
    number_of_trades.append(len(orders_file))

    # --------------------------------------------------------------- #
    # -------------- Leaner Strategy, impact at 0.100 --------------- #
    # --------------------------------------------------------------- #
    impt2 = 0.05
    learner = sl.StrategyLearner(impact=impt)
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date)
    trades = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date)

    # Create orders file
    orders_file = create_orders_file(clean_trades=trades, stock=ticker)

    # -------------- Get Port Stats --------------- #
    port_value = compute_portvals(orders_file, start_date, end_date, commission=comm, impact=impt)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update data
    avg_returns.append(np.mean(port_value.pct_change()))
    number_of_trades.append(len(orders_file))

    # --------------------------------------------------------------- #
    # -------------- Leaner Strategy, impact at 0.500 --------------- #
    # --------------------------------------------------------------- #
    impt3 = 0.5
    learner = sl.StrategyLearner(impact=impt)
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date)
    trades = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date)

    # Create orders file
    orders_file = create_orders_file(clean_trades=trades, stock=ticker)

    # -------------- Get Port Stats --------------- #
    port_value = compute_portvals(orders_file, start_date, end_date, commission=comm, impact=impt)
    port_value = port_value / port_value.iloc[0]
    port_value = port_value.to_frame()

    # Update data
    avg_returns.append(np.mean(port_value.pct_change()))
    number_of_trades.append(len(orders_file))

    # -------------------------------------------------------------- #
    # -------------- Aggregate data and create chart --------------- #
    # -------------------------------------------------------------- #
    avg = pd.DataFrame(data=np.array(avg_returns),
                       index=[impt1, impt2, impt3])

    num_t = pd.DataFrame(data=np.array(number_of_trades),
                       index=[impt1, impt2, impt3])


    #combined = np.hstack((avg, num_t))
    #data = pd.DataFrame(data=combined, index=[0.00, 0.025, 0.5], columns=["Avg_rets", "Num_trades"])

    # ------------- Comparative Charts ---------- #
    # figure 1
    fig = plt.figure()
    plt.plot(avg, 'g')
    # note legend goes after plot
    plt.legend(["Avg_return"])
    plt.xlabel("Impact")
    plt.ylabel("Metric Value")
    plt.title("Experiment2, Avg_returns and Impact")
    plt.savefig("experiment2_avg_rets.png")
    plt.close()

    # Figure 2
    fig = plt.figure()
    plt.plot(num_t, 'r')
    # note legend goes after plot
    plt.legend(["Number of Trades"])
    plt.xlabel("Impact")
    plt.ylabel("Metric Value")
    plt.title("Experiment2, Number of Trades and Impact")
    plt.savefig("experiment2_num_trades.png")
    plt.close()

    pass