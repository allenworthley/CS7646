import ManualStrategy as ms
import StrategyLearner as sl
from marketsimcode import compute_portvals
from util import get_data

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd


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


def run_experiment1():

    # In-sample
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 12, 31)
    starting_value = 100000
    impt = 0
    comm = 0
    ticker = "JPM"

    # ---------------------------------------- #
    # -------------- Benchmark --------------- #
    # ---------------------------------------- #

    data = 1000 * get_data([ticker], pd.date_range(start_date, end_date), addSPY=True, colname="Adj Close").drop(columns="SPY")
    data["benchmark"] = data / data.iloc[0, 0]

    # ---------------------------------------------- #
    # -------------- Manual Strategy --------------- #
    # ---------------------------------------------- #
    # Initialize Obj
    manual = ms.ManualStrategy()

    # Get trades from strategy
    trades = manual.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=starting_value)

    # Create orders file
    orders_file = create_orders_file(clean_trades=trades, stock=ticker)

    # -------------- Get Port Stats --------------- #
    port_value = compute_portvals(orders_file, start_date, end_date, commission=comm, impact=impt)
    port_value = port_value / port_value.iloc[0]
    data["ms_port"] = port_value

    # ---------------------------------------------- #
    # -------------- Leaner Strategy --------------- #
    # ---------------------------------------------- #
    learner = sl.StrategyLearner()
    learner.add_evidence(symbol=ticker, sd=start_date, ed=end_date)
    trades = learner.testPolicy(symbol=ticker, sd=start_date, ed=end_date)

    # Create orders file
    orders_file = create_orders_file(clean_trades=trades, stock=ticker)

    # -------------- Get Port Stats --------------- #
    port_value = compute_portvals(orders_file, start_date, end_date, commission=comm, impact=impt)
    port_value = port_value / port_value.iloc[0]
    data["sl_port"] = port_value

    # ------------- Comparative Chart ---------- #
    fig = plt.figure()
    plt.plot(data.benchmark, 'g')
    # note legend goes after plot
    plt.plot(data.ms_port, 'r')
    plt.plot(data.sl_port, 'b')
    plt.legend(["benchmark", "ManualStrategy", "StrategyLearner"])
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("Experiment1, Strategy Comparison")
    fig.autofmt_xdate()
    plt.savefig("experiment1.png")
    plt.close()

    pass