# files
import indicators as ind
import marketsimcode as msc
from util import get_data
from marketsimcode import compute_portvals

# packages
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def author():
    return "mworthley3"


def create_trade_orders(signals):
    trades = pd.DataFrame(data=0, columns=["shares"], index=signals.index.values)

    # Clean up trade signals into trade orders
    # restricts trades to only holding at max -1000, 0, 1000 shares at a time
    for i in range(1, trades.shape[0]):
        # Previously zero and has position
        if signals.iloc[i] != 0 and signals.iloc[i-1] == 0:
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


class ManualStrategy:
    raw_signal = []

    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=100000):
        # ------- Get testing data ------- #
        self.data = get_data([symbol], pd.date_range(sd, ed), addSPY=True, colname="Adj Close").drop(columns=["SPY"])

        # RSI Strategy
        self.data["rsi"] = ind.rsi_calc(self.data[symbol], time_horizon=14, trade_signal=True)

        # MACD Strategy
        self.data["MACD"] = ind.MACD(self.data[symbol], short_term=12, long_term=26, trade_signal=True)

        # Golden Cross
        self.data["Golden"] = ind.golden_death_cross(self.data[symbol], trade_signal=True)

        # Simple aggregation of all signals
        self.raw_signal = self.data.Golden + self.data.MACD + self.data.rsi

        # Normalize signals to 1, 0, -1
        self.raw_signal.loc[self.raw_signal > 1] = 1
        self.raw_signal.loc[self.raw_signal < -1] = -1

        # Create trades Orders from signal
        trade_orders = create_trade_orders(self.raw_signal)

        return trade_orders

    def author(self):
        return "mworthley3"