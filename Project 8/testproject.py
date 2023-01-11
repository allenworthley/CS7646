import random

import ManualStrategy as ms
import StrategyLearner as sl
from experiment1 import run_experiment1
from experiment2 import run_experiment2
from marketsimcode import compute_portvals
from util import get_data

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np


def author():
    return "mworthley3"


def get_stats(df):
    df = df.to_frame()

    lvl_rets = df - df.shift(1)
    pct_rets = df.pct_change()

    cr = np.sum(lvl_rets)
    stdev = pct_rets.std()
    ar = np.mean(pct_rets)

    return cr, stdev, ar


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

random.seed(1000)

# ------------------------------------------------ #
# ---------- Manual Strategy Charts -------------- #
# ------------------------------------------------ #
# ---------- in-sample ---------- #
# Parameters
start_date = dt.datetime(2008, 1, 1)
end_date = dt.datetime(2009, 12, 31)
starting_value = 100000
ticker = "JPM"

# Initialize Obj
manual = ms.ManualStrategy()

# Get trades from strategy
trades = manual.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=starting_value)

# Create orders file
orders_file = create_orders_file(clean_trades=trades, stock=ticker)

# -------------- Port Stats --------------- #
insample_port_val = compute_portvals(orders_file, start_date, end_date, commission=9.95, impact=0.005)
port_value = insample_port_val / insample_port_val.iloc[0]

# --------------- Bench Mark --------------- #
data = 1000 * get_data([ticker], pd.date_range(start_date, end_date), addSPY=True, colname="Adj Close").drop(
    columns="SPY")
data["benchmark"] = data / data.iloc[0, 0]
data["portfolio"] = port_value

# --------------- get raw trade signal from manual --------------- #
trade_signal = manual.raw_signal.to_numpy()
data["buy"] = np.where(trade_signal < 0, 0, trade_signal)
data["sell"] = np.where(trade_signal > 0, 0, trade_signal)

# ------------- Comparative Chart ---------- #
fig = plt.figure()
plt.plot(data.benchmark, 'g')
# note legend goes after plot
plt.plot(data.portfolio, 'r')
plt.bar(data.index, data.buy*15, color='blue')
plt.bar(data.index, data.sell*15, color='black')
plt.legend(["benchmark", "portfolio"])
plt.xlabel("Date")
plt.ylabel("Portfolio Value, Long/Short Indicators")
plt.title("Manual Strategy vs Benchmark insample")
fig.autofmt_xdate()
plt.savefig("Manual Strategy vs Benchmark insample.png")
plt.close()

# ------------- summary statistics ---------- #
in_sample_stats = np.zeros((2, 3))

# benchmark
cum_rets, std_rets, avg_rets = get_stats(data.JPM)

in_sample_stats[0, 0] = cum_rets
in_sample_stats[0, 1] = std_rets
in_sample_stats[0, 2] = avg_rets

# portfolio
cum_rets, std_rets, avg_rets = get_stats(insample_port_val)

in_sample_stats[1, 0] = cum_rets
in_sample_stats[1, 1] = std_rets
in_sample_stats[1, 2] = avg_rets

# save
np.savetxt("manual_insample_stats.csv", in_sample_stats, delimiter=',')

# ---------- out of sample ---------- #
# Parameters
start_date = dt.datetime(2010, 1, 1)
end_date = dt.datetime(2011, 12, 31)
starting_value = 100000
ticker = "JPM"

# Initialize Obj
manual = ms.ManualStrategy()

# Get trades from strategy
trades = manual.testPolicy(symbol=ticker, sd=start_date, ed=end_date, sv=starting_value)

# Create orders file
orders_file = create_orders_file(clean_trades=trades, stock=ticker)

# -------------- Port Stats --------------- #
out_sample_port_val = compute_portvals(orders_file, start_date, end_date, commission=9.95, impact=0.005)
port_value = out_sample_port_val / out_sample_port_val.iloc[0]

# --------------- Bench Mark --------------- #
data = 1000 * get_data([ticker], pd.date_range(start_date, end_date), addSPY=True, colname="Adj Close").drop(
    columns="SPY")
data["benchmark"] = data / data.iloc[0, 0]
data["portfolio"] = port_value

# --------------- get raw trade signal from manual --------------- #
trade_signal = manual.raw_signal.to_numpy()
data["buy"] = np.where(trade_signal < 0, 0, trade_signal)
data["sell"] = np.where(trade_signal > 0, 0, trade_signal)

# ------------- Comparative Chart ---------- #
fig = plt.figure()
plt.plot(data.benchmark, 'g')
# note legend goes after plot
plt.plot(data.portfolio, 'r')
plt.bar(data.index, data.buy*15, color='blue')
plt.bar(data.index, data.sell*15, color='black')
plt.legend(["benchmark", "portfolio"])
plt.xlabel("Date")
plt.ylabel("Portfolio Value, Long/Short Indicators")
plt.title("Manual Strategy vs Benchmark outsample")
fig.autofmt_xdate()
plt.savefig("Manual Strategy vs Benchmark outsample.png")
plt.close()

# ------------- summary statistics ---------- #
out_sample_stats = np.zeros((2, 3))

# benchmark
cum_rets, std_rets, avg_rets = get_stats(data.JPM)

out_sample_stats[0, 0] = cum_rets
out_sample_stats[0, 1] = std_rets
out_sample_stats[0, 2] = avg_rets

# portfolio
cum_rets, std_rets, avg_rets = get_stats(out_sample_port_val)

out_sample_stats[1, 0] = cum_rets
out_sample_stats[1, 1] = std_rets
out_sample_stats[1, 2] = avg_rets

# save
np.savetxt("manual_outsample_stats.csv", out_sample_stats, delimiter=',')

# ------------------------------------------------ #
# ---------- Run Experiments --------------------- #
# ------------------------------------------------ #

run_experiment1()
run_experiment2()
