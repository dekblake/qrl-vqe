import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt 

def test_train(tickers, interval):
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

    assets = [
    'SPY',   # US Large Cap Equities 
    'TLT',   # 20+ Year Treasury Bonds 
    'GLD',   # Gold 
    'USO',   # Crude Oil
    'VNQ',   # Real Estate / REITs
    'EEM',   # Emerging Markets 
    'UUP',   # US Dollar Index 
    'XOM',   # Exxon Mobil 
    'JNJ',   # Johnson & Johnson 
    'TSLA'   # Tesla 
]

train_set, test_set = test_train(assets)

train_set.to_csv("data/raw/train_set.csv")
test_set.to_csv("data/raw/test_set.csv")