import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt 

def training_data(tickers, interval):
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2025, 1, 1)

    data = yf.download(tickers, interval, start, end)['Close']

    data = data.ffill().dropna()

    returns = data.pct_change().dropna()

    split = int(len(returns)*.8)

    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]
    
    return train_returns, test_returns

if __name__ == "__main__":

    tickers = []

