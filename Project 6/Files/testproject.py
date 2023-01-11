import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from util import get_data
from indicators import golden_death_cross, exp_mov_avg, rsi_calc, money_flow, MACD
import TheoreticallyOptimalStrategy as tos
from marketsimcode import compute_portvals


def author(self):
    return "mworthley3"  # replace tb34 with your Georgia Tech username


def plot_ind(df, plot_title):
    # Plot
    fig = plt.figure()
    plt.plot(df)
    # note legend goes after plot
    plt.legend(df.columns)
    plt.xlabel("Date")

    plt.ylabel("Price, Index Value ")

    plt.title(plot_title)
    fig.autofmt_xdate()
    plt.savefig(plot_title + ".png")
    plt.close()
    pass


def split_plot_ind(df, plot_title):
    fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False)

    # Overall Title
    plt.title("JPM Stock Price (Above) vs " + plot_title + " (Below)")

    # Subplot 1 - JPM Price
    axes[0].plot(df["JPM"])
    plt.legend(df.columns)

    # Subplot 2 - Indexes
    axes[1].plot(df.iloc[:, 1:])
    plt.legend(df.columns[1:])
    plt.ylabel("Index Value, Price")

    plt.xlabel("Date")
    fig.autofmt_xdate()
    plt.savefig(plot_title + ".png")
    plt.close()
    pass


if __name__ == "__main__":

    # parameters for stock data pull
    sd = dt.date(2008, 1, 1)
    ed = dt.date(2009, 12, 31)
    date_ranges = pd.date_range(sd, ed)
    symbols = ["JPM"]  # must be a singular list element

    # ---------------------------------------------- #
    # ----------------- Indicators ----------------- #
    # ---------------------------------------------- #

    # Get data to process indicators
    # include SPY to ensure only pulls trading days, drop spy after
    high = get_data(symbols, pd.date_range(sd, ed), addSPY=True, colname="High").drop(columns=["SPY"])
    low = get_data(symbols, date_ranges, addSPY=True, colname="Low").drop(columns=["SPY"])
    volume = get_data(symbols, date_ranges, addSPY=True, colname="Volume").drop(columns=["SPY"])
    adj_close = get_data(symbols, date_ranges, addSPY=True, colname="Adj Close").drop(columns=["SPY"])

    # ----------------- Golden Death Cross ----------------- #
    gdc = golden_death_cross(adj_close)
    plot_ind(gdc, "Golden Death Cross")

    # ----------------- EMA ----------------- #
    ema_ = exp_mov_avg(adj_close, 30)
    plot_ind(ema_, "Exponential Moving Average")

    # ----------------- Relative Strength Indicator ----------------- #
    rsi = rsi_calc(adj_close)
    split_plot_ind(rsi, "RSI")

    # ----------------- Money Flow ----------------- #
    mf = money_flow(adj_close, low, high, volume)
    split_plot_ind(mf, "Money Flow")

    # ----------------- MACD ----------------- #
    macd = MACD(adj_close)
    split_plot_ind(macd, "MACD")

    # --------------------------------------- #
    # ----------------- TOS ----------------- #
    # --------------------------------------- #

    # Get TOS trades as +/- shares
    ### have to use symbol=["JPM"] due to error by UTIl function, expecting an iterable list item
    df_trades = tos.testPolicy(symbol="JPM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=100000)

    # Creat psuedo orders file
    orders = pd.DataFrame(index=df_trades.index.values, columns=["Symbol", "Order", "Shares"])
    orders["Symbol"] = "JPM"
    orders["Order"] = df_trades.where(df_trades > 0, "BUY").where(df_trades < 0, "SELL")
    orders["Shares"] = abs(df_trades)

    # -------------- Port Stats --------------- #
    port_value = compute_portvals(orders, sd, ed)
    port_value = port_value / port_value.iloc[0]

    # --------------- Bench Mark --------------- #
    data = 1000 * get_data(["JPM"], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop(columns="SPY")
    data["benchmark"] = data / data.iloc[0, 0]
    data["portfolio"] = port_value

    # ------------- Comparative Chart ---------- #
    fig = plt.figure()
    plt.plot(data.benchmark, 'g')
    # note legend goes after plot
    plt.plot(data.portfolio, 'r')
    plt.legend(["benchmark", "portfolio"])
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.title("TOS Portfolio vs Benchmark")
    fig.autofmt_xdate()
    plt.savefig("TOS Portfolio vs Benchmark.png")
    plt.close()

    # ------------- Tables ------------- #
    # Stats
    daily_rets = data.diff().dropna()
    # Bench
    bench_std = round(daily_rets.benchmark.std(), 4)
    bench_cum_rets = round(daily_rets.benchmark.sum(), 4)
    bench_avg_rets = round(daily_rets.benchmark.mean(), 4)
    # port
    port_std = round(daily_rets.portfolio.std(), 4)
    port_cum_rets = round(daily_rets.portfolio.sum(), 4)
    port_avg_rets = round(daily_rets.portfolio.mean(), 4)

    # Output
    headers = ['Portfolio', 'STD', 'Cummulative Rets', 'Average Rets']
    rows = [['benchmark', bench_std, bench_cum_rets, bench_avg_rets],
            ['TOS portfolio', port_std, port_cum_rets, port_avg_rets]]

    df = pd.DataFrame(data=rows, columns=headers)
    df.to_csv(r'p6_results.txt', header=True, index=None, sep='\t', mode='a')
    pass