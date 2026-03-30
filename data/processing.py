import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt


def test_train(tickers, interval, start, end):

    data = yf.download(tickers, interval, start, end)['Close']

    data = data.ffill().dropna()

    returns = np.log(data / data.shift(1)).dropna()

    split = int(len(returns)*.8)

    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]

    return train_returns, test_returns


if __name__ == "__main__":

    assets = [
        'SPY',       # 1. The Benchmark (Broad US Market)
        'TLT',       # 2. The Safe Haven (20+ Year Bonds for risk-off days)
        'GLD',       # 3. The Hedge (Gold, non-correlated to tech)
        'BTC-USD',   # 4. The Modern Crypto (Massive volatility and growth)
        'NVDA',      # 5. The AI Tech Giant (High momentum, sector-specific)
        # 6. The Cult Stock (Retail-driven, highly volatile movements)
        'TSLA',
        # 7. Energy Sector (Captures oil/gas movements without single-company risk like XOM)
        'XLE'
    ]

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2025, 1, 1)
    train_set, test_set = test_train(assets, "1d", start, end)

    train_set.to_csv("data/train_set.csv")
    test_set.to_csv("data/test_set.csv")
