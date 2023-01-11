import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from util import get_data

def author(self):
    """
    :return: The GT username of the student
    :rtype: str
    """
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


def golden_death_cross(df):
    pass_df = df.copy()
    pass_df["sma_20"] = df.rolling(window=20).mean()
    pass_df["sma_50"] = df.rolling(window=50).mean()

    return pass_df.dropna()


def exp_mov_avg(df, lag=30):
    # Copy df
    pass_df = df.copy()
    pass_df["ema"] = df.ewm(span=lag).mean()
    return pass_df.dropna()


def rsi_calc(df, time_horizon=14):
    """
    Relative strength index
    # assume df has only adjusted close price
    # uses simple moving average, has sensitivity to extreme movements
    """

    pass_df = df.copy()

    # Get daily percentage returns
    dly_pct_rets = pass_df.pct_change(periods=1)

    # Clip for rolling calculation
    gains = dly_pct_rets.clip(lower=0)
    losses = abs(dly_pct_rets.clip(upper=0))

    # Get average returns for gains and lowers
    avg_gain = gains.rolling(window=time_horizon).mean()
    avg_loss = losses.rolling(window=time_horizon).mean()

    # RSI Calculation
    pass_df["rsi"] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # Indicator Levels
    pass_df["Upper"] = 70
    pass_df["Lower"] = 30

    return pass_df.dropna()


def money_flow(df_adj_close, df_low, df_high, df_volume):

    # Get copy of df
    pass_df = df_adj_close.copy()

    # Typical Price
    typ_price = (df_low + df_high + df_adj_close)/3

    # determine if return is +/- to apply to raw_money_flow
    rets = typ_price.diff()
    flag = rets.where(rets > 0, -1).where(rets < 0, 1)
    temp = df_volume * typ_price
    raw_money_flow = temp * flag

    # Separate raw_money into +/-
    gains = raw_money_flow.clip(lower=0)
    losses = abs(raw_money_flow.clip(upper=0))

    # Money Flow Ratio
    tot_gain = gains.rolling(window=14).sum()
    tot_loss = losses.rolling(window=14).sum()
    mfr = tot_gain / tot_loss

    mfr = mfr.dropna()

    # Money Flow Calc
    pass_df["Money Flow"] = 100 - (100 / (1 + mfr))
    pass_df["Upper"] = 80
    pass_df["Lower"] = 20

    return pass_df.dropna()


def MACD(df, short_term=12, long_term=26):

    # create return df
    pass_df = df.copy()

    # Calculate moving averages
    st = df.ewm(span=short_term).mean()
    lt = df.ewm(span=long_term).mean()

    # store macd
    pass_df["macd"] = st - lt

    # Make signal line
    pass_df["signal"] = pass_df["macd"].ewm(span=9).mean()

    return pass_df.dropna()
