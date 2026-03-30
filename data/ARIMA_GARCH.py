import yfinance as yf
import pandas as pd
import datetime as dt
import numpy as np
from math import pi
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# import csv and create dataframe


def mf2_garch_core(params, y, m):

    mu, alpha, gamma, beta, lambda_0, lambda_1, lambda_2 = params

    h = np.ones(len(y))
    tau = np.ones(len(y))*np.mean(y**2)
    V = np.zeros(len(y))
    V_m = np.zeros(len(y))

    for t in range(2, m):

        if y[t-1] - mu < 0:
            h[t] = (1-alpha-gamma/2-beta) + (alpha + gamma) * \
                (y[t-1] - mu)**2 / tau[t-1] + beta * h[t-1]
        else:
            h[t] = (1-alpha-gamma/2-beta) + alpha * \
                (y[t-1] - mu)**2 / tau[t-1] + beta * h[t-1]

    for t in range(m+1, len(y)):

        if y[t-1] - mu < 0:
            h[t] = (1-alpha-gamma/2-beta) + (alpha + gamma) * \
                (y[t-1] - mu)**2 / tau[t-1] + beta * h[t-1]
        else:
            h[t] = (1-alpha-gamma/2-beta) + alpha * \
                (y[t-1] - mu)**2 / tau[t-1] + beta * h[t-1]

        V[t] = (y[t] - mu)**2 / h[t]
        V_m[t] = np.sum(V[t-(m-1):t])/m

        tau[t] = lambda_0 + lambda_1 * V_m[t-1] + lambda_2 * tau[t-1]

    var = np.maximum(h * tau, 1e-12)
    e = (y-mu) / np.sqrt(var)

    h = h[2*252+1:len(y)]
    tau = tau[2*252+1:len(y)]
    e = e[2*252+1:len(y)]
    V_m = V_m[2*252+1:len(y)]

    return e, h, tau, V_m


def likelihood_mf2_garch(params, y, m):

    # mf2_garch_core

    e, h, tau, V_m = mf2_garch_core(params, y, m)
    y_eff = y[2*252+1:]
    mu = params[0]
    var = np.maximum(h * tau, 1e-12)
    log_likelihood = -0.5 * \
        (np.log(2*pi) + np.log(var) + ((y_eff - mu)**2 / var))
    log_like_sum = -np.sum(log_likelihood)

    return log_like_sum


def mf2_garch_estimate(y, m):

    t = len(y)

    param_init = [0.02,                             # mu
                  0.007,                            # alpha
                  0.14,                             # gamma
                  0.85,                             # beta
                  np.mean(y**2)*(1-0.07-0.91),      # lambda_0
                  0.07,                             # lambda_1
                  0.91                              # lambda_2
                  ]

    LB = [-1, 0.0, -0.5, 0.0, 0.000001, 0.0, 0.0]
    UB = [1, 1.0, 0.5, 1.0, 10.0, 1.0, 1.0]
    bounds = list(zip(LB, UB))

    constraints = [{'type': 'ineq', 'fun': lambda p: p[1]},                            # alpha >= 0
                   # alpha + gamma/2 + beta <=1
                   {'type': 'ineq', 'fun': lambda p: 1 -
                    (p[1] + 0.5*p[2] + p[3])},
                   # lambda_1 >= 0
                   {'type': 'ineq', 'fun': lambda p: p[5]},
                   # lambda_1 + lambda_2 <=1
                   {'type': 'ineq', 'fun': lambda p: 1 - (p[5] + p[6])}
                   ]

    result = minimize(
        fun=likelihood_mf2_garch,
        x0=param_init,
        args=(y, m),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    coeff = result.x
    e, h, tau, V_m = mf2_garch_core(coeff, y, m)

    return coeff, e, h, tau, V_m


def sum_predetermined(r2, h, t, m):

    start_idx = max(0, t-m+1)

    return np.sum(r2[start_idx: t+1] / h[start_idx: t+1])


def predicted(y, h, tau, coeff, m):

    mu, alpha, gamma, beta, lambda_0, lambda_1, lambda_2 = coeff

    y_today = y[-1]
    h_today = h[-1]
    tau_today = tau[-1]

    # The squared shock - can add ARIMA estimate
    r2_today = (y_today - mu)**2

    if (y_today - mu) >= 0:
        h_tomorrow = (1 - (alpha + gamma/2) - beta) + alpha * \
            (r2_today/tau_today) + beta * h_today

    else:
        h_tomorrow = (1 - (alpha + gamma/2) - beta) + \
            (alpha + gamma) * (r2_today/tau_today) + beta * h_today

    r2_array = (y - mu)**2
    t = len(h) - 1
    sum_predetermined_m = sum_predetermined(r2_array, h, t, m)
    tau_tomorrow = lambda_0 + lambda_1 * \
        (1/m) * sum_predetermined_m + lambda_2 * tau_today

    var_tomorrow = h_tomorrow * tau_tomorrow

    return var_tomorrow
