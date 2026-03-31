import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import multiprocessing
import tensorflow as tf

cores = multiprocessing.cpu_count()
tf.config.threading.set_inter_op_parallelism_threads(cores)
tf.config.threading.set_intra_op_parallelism_threads(cores)
print(f"Forcing TensorFlow to use all {cores} CPU cores!")


import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from statsmodels.tsa.arima.model import ARIMA
import cirq

from arima_garch import mf2_garch_estimate
from vqe_portfolio import portfolio_optimisation
from variational_eigensolver import eigen_circuit


def test_train(tickers, start, end):
    data = yf.download(tickers, start, end)['Close']
    data = data.ffill().dropna()

    # Log returns
    returns = np.log(data / data.shift(1)).dropna()

    split = int(len(returns)*.8)
    train_returns = returns.iloc[:split]
    test_returns = returns.iloc[split:]

    return train_returns, test_returns


if __name__ == "__main__":

    assets = [
        'SPY', 'TLT', 'GLD', 'BTC-USD', 'NVDA', 'TSLA', 'XLE'
    ]

    start = dt.datetime(2020, 1, 1)
    end = dt.datetime(2025, 1, 1)

    # test/train split
    print("Downloading Data...")
    train_set, test_set = test_train(assets, start, end)

    full_data = pd.concat([train_set, test_set])
    m_window = 252

    # -----------------------------------------------------------------
    # NEW: Create empty columns for our honest predictions
    # -----------------------------------------------------------------
    for asset in assets:
        full_data[f"{asset}_mu"] = np.nan
        full_data[f"{asset}_var"] = np.nan

    # running expanding window ARIMA-GARCH to prevent look-ahead bias
    for asset in assets:
        print(f"Estimating rolling ARIMA and MF2-GARCH for {asset} (This will take a while)...")
        
        # Start from day 252 (we need 1 year of history to make the first prediction)
        for i in range(m_window, len(full_data)):
            
            # ONLY grab past prices up to today (index 'i'). No future peeking!
            history_window = full_data[asset].iloc[:i].to_numpy()
            
            # --- ARIMA ---
            try:
                # Fit on history, forecast exactly 1 step into the future
                arima_model = ARIMA(history_window, order=(1, 0, 0)).fit()
                mu_pred = arima_model.forecast(steps=1)[0]
            except:
                mu_pred = 0.0 # Safety net if math fails to converge

            # --- MF2-GARCH ---
            try:
                coeff, e, h, tau, V_m = mf2_garch_estimate(history_window, m=m_window)
                var_pred = h[-1] * tau[-1]
            except:
                var_pred = 0.0001 # Safety net if math fails to converge

            # Assign predictions to today's date
            date_index = full_data.index[i]
            full_data.loc[date_index, f"{asset}_mu"] = mu_pred
            full_data.loc[date_index, f"{asset}_var"] = var_pred

            # Print an update every 100 days so we know it's working
            if i % 100 == 0:
                print(f"   ...processed {i}/{len(full_data)} days for {asset}")

    # Drop the 252 warmup days where we couldn't make predictions
    master_df = full_data.dropna()
    # -----------------------------------------------------------------

    # vqe circuit
    print("Preparing VQE Circuit...")
    vqe_qubits = cirq.GridQubit.rect(1, 14)  # 14 qubits for 7 assets
    vqe_circuit, vqe_params = eigen_circuit(vqe_qubits, layer_count=3, seed=42)
    vqe_param_strings = [str(p) for p in vqe_params]

    # solver
    print("Running Daily VQE Solver...")
    for asset in assets:
        master_df[f"{asset}_vqe"] = 0

    for date, row in master_df.iterrows():
        mu_today = [row[f"{asset}_mu"] for asset in assets]
        var_today = [row[f"{asset}_var"] for asset in assets]

        # Pass the 'assets' list so we only grab the original 7 columns!
        recent_returns_df = full_data[assets].loc[:date].tail(60)

        # Solve for today's optimal portfolio
        optimal_tiers = portfolio_optimisation(
            mu_today, var_today, recent_returns_df, vqe_circuit, vqe_param_strings)

        for i, asset in enumerate(assets):
            master_df.at[date, f"{asset}_vqe"] = optimal_tiers[i]

    # test/train split pt.2
    print("Splitting and Saving...")
    test_start_date = test_set.index[0]

    master_train = master_df[master_df.index < test_start_date]
    master_test = master_df[master_df.index >= test_start_date]

    # Save to CSVs
    master_train.to_csv("data/master_train_env.csv")
    master_test.to_csv("data/master_test_env.csv")

    print(f"Done! Saved {len(master_train)} Train days and {len(master_test)} Test days.")