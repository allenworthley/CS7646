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


def money_flow(df_adj_close, df_low, df_high, df_volume, trade_signal=False):

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

    # Trade signal = True is for manual
    if trade_signal:
        pass_df["mf_signal"] = 0
        for i in range(1, pass_df.shape[0]):
            # Buy
            if pass_df["Money Flow"].iloc[i - 1] < 30 and pass_df["Money Flow"].iloc[i] >= 30:
                pass_df["mf_signal"].iloc[i] = 1
            # Sell
            elif pass_df["Money Flow"].iloc[i - 1] > 70 and pass_df["Money Flow"].iloc[i] <= 70:
                pass_df["mf_signal"].iloc[i] = -1
        return pass_df["mf_signal"]
    else:
        return pass_df.dropna()


def golden_death_cross(df, trade_signal=False):

    # convert to numpy for better performance
    data = df.to_numpy()

    # calc moving averages
    short = moving_average(data, 20)
    long = moving_average(data, 50)

    # adjust size due to lagging calculations
    short_size_adj = df.shape[0] - len(short)
    short = np.append(np.zeros(short_size_adj), short)

    long_size_adj = df.shape[0] - len(long)
    long = np.append(np.zeros(long_size_adj), long)

    # Return buy signal or df depending on need

    if trade_signal:
        gdc_signal = np.zeros(df.shape[0])
        for i in range(long_size_adj, df.shape[0]):
            # Buy
            if short[i - 1] < long[i] < short[i]:
                gdc_signal[i] = 1
            # Sell
            elif short[i - 1] > long[i] > short[i]:
                gdc_signal[i] = -1
        return gdc_signal
    else:
        # Strategy Learner
        # make continuous for learner
        return long - short


def numpy_ewma_vectorized_v2(data, window):

    alpha = 2 / (window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out


def moving_average(a, n):
    N=len(a)
    return np.array([np.mean(a[i:i+n]) for i in np.arange(0, N-n+1)])


def rsi_calc(df, time_horizon=14, trade_signal=False):
    """
    Relative strength index
    # assume df has only adjusted close price
    # uses simple moving average, has sensitivity to extreme movements
    """

    # convert to numpy for better performance
    data = df.to_numpy()
    # numpy doesn't keep track of dates

    # Get daily percentage returns, numpy doesn't keep track of dates, drops
    returns = np.diff(data) / data[:-1]

    # Filter on series of positive or negative returns
    gains = np.where(returns < 0, 0, returns)
    losses = abs(np.where(returns > 0, 0, returns))

    # Get average returns for gains and lowers
    avg_gain = moving_average(gains, time_horizon)
    avg_loss = moving_average(losses, time_horizon)

    # RSI Calculation
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # adjustment due to lagging calculations, avg loss/gain same size
    size_adj = df.shape[0] - len(avg_loss)

    rsi = np.append(np.zeros(size_adj), rsi)
    # Return buy signal or df depending on need

    if trade_signal:
        # size should match original data submission
        rsi_signal = np.zeros(df.shape[0])
        for i in range(size_adj, df.shape[0]):
            # Buy
            if rsi[i-1] < 30 and rsi[i] >= 30:
                rsi_signal[i] = 1
            # Sell
            elif rsi[i - 1] > 70 and rsi[i] <= 70:
                rsi_signal[i] = -1
        return rsi_signal
    else:
        return rsi


def MACD(df, short_term=12, long_term=26, trade_signal=False):

    # Re did logic for MACD to take advantage of vectorization and optimization

    # Calculate moving averages
    st = numpy_ewma_vectorized_v2(df.to_numpy(), short_term)
    lt = numpy_ewma_vectorized_v2(df.to_numpy(), long_term)

    # store macd
    macd = st - lt

    # Make signal line
    signal = numpy_ewma_vectorized_v2(macd, 9)

    macd_signal = np.zeros(df.shape[0])

    # trade_signal = True return data for manual strategy
    if trade_signal:
        for i in range(1, df.shape[0]):
            # Buy
            if macd[i-1] > signal[i] >= macd[i]:
                macd_signal[i] = 1
            # Sell
            elif macd[i-1] < signal[i] <= macd[i]:
                macd_signal[i] = -1

        return macd_signal
    else:
        # Return for strategy learner
        return macd - signal


if __name__ == "__main__":
    sd = dt.datetime(2010, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    test_data = get_data(["JPM"], pd.date_range(sd, ed), addSPY=True,
                            colname="Adj Close").drop(columns=["SPY"])

    returned_data = MACD(test_data["JPM"])
    np.savetxt("macd_new.csv", returned_data, delimiter=",")
    print("new data size is: ", len(returned_data))

    returned_data = rsi_calc(test_data["JPM"])
    np.savetxt("rsi_new.csv", returned_data, delimiter=",")
    print("new data size is: ", len(returned_data))

    returned_data = golden_death_cross(test_data["JPM"])
    np.savetxt("gdc_new.csv", returned_data, delimiter=",")
    print("old size is ", returned_data.shape[0])








