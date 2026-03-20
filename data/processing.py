import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt 

def training_data(start=, end, tickers, interval):

    data = yf.Tickers(tickers).download(interval, start, end)
    data = pd.dataframe()
    
