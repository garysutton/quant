import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# define ticker symbol
ticker_symbol = 'AAPL'
# function
def fetch_stock_data(ticker_symbol, start_date, end_date,
                     interval = '1d'):
    ticker_data = yf.Ticker(ticker_symbol)
    stock_data = ticker_data.history(start = start_date,
                                     end = end_date,
                                     interval = interval)
    return stock_data
stock_data = fetch_stock_data('AAPL', '2023-10-01', '2024-04-30')

# subset to include only the 'Close' column
close_prices = stock_data[['Close']]
# reset the index to have 0, 1, 2, 3, etc.
close_prices = close_prices.reset_index(drop = True)
# display the data
print(close_prices.info())
print(close_prices.head())
print(close_prices.tail())

# plot the data
plt.subplots()
plt.plot(close_prices)
plt.xlabel('Trading Days\n'
              'October 1, 2023 through April 30, 2024')
plt.ylabel('Closing Price (USD)')
plt.title('Daily Closing Stock Price - AAPL')
plt.grid()
plt.show()

# max and min - code not necessarily in book
max_close = close_prices['Close'].max()
min_close = close_prices['Close'].min()
print(max_close)
print(min_close)

# non-stationary and stationary quadrant of plots - only plots in book
# generate non-stationary data
np.random.seed(0)
# plot 1: Non-stationary data with increasing trend
plt.figure(figsize = (12, 10))
plt.subplot(221)
t = np.arange(0, 100)
data1 = np.cumsum(np.random.randn(100)) + np.linspace(0, 20, 100)
plt.plot(t, data1, label = 'Non-stationary with trend')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
# plot 2: Non-stationary data with seasonal pattern
plt.subplot(222)
data2 = np.sin(t / 10) + np.random.randn(100) * 0.5
plt.plot(t, data2, label = 'Non-stationary with seasonality')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
# plot 3: Non-stationary data with increasing variance
plt.subplot(223)
data3 = np.cumsum(np.random.randn(100)) + np.linspace(0, 50, 100) + np.random.randn(100) * 10
plt.plot(t, data3, label = 'Non-stationary with increasing variance')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
# plot 4: Stationary data oscillating around a constant mean
plt.subplot(224)
data4 = np.sin(t / 5)
plt.plot(t, data4, label = 'Stationary')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.tight_layout()
plt.show()

# create train and test
bln_msk = (close_prices.index < len(close_prices) - 22)
train = close_prices[bln_msk].copy()
test = close_prices[~bln_msk].copy()

# check if data is stationary - acf and pacf plots
acf = plot_acf(train)
pacf = plot_pacf(train)

# check if data is stationary - perform the Augmented Dickey-Fuller test on the closing prices
adf_test = adfuller(train)
print('p-value:', adf_test[1]) # data is non-stationary

# difference the data
train_diff = train.diff().dropna()
# plot the time series
train_diff.plot()
plt.xlabel('Trading Days:\n'
              'October 1, 2023 through March 31, 2024')
plt.ylabel('Differences in Closing Prices')
plt.title('1st Order Differencing')
plt.legend().set_visible(False)
plt.grid()
plt.show()

# check again if data is stationary - acf and pacf plots
fig, (ax1, ax2) = plt.subplots(2, 1)
plot_acf(train_diff, ax = ax1)
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(train_diff, ax = ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# 4x4 plot - old and new
fig, axes = plt.subplots(2, 2)
plot_acf(train, ax = axes[0, 0])
axes[0, 0].set_title('ACF - Original')
plot_pacf(train, ax = axes[0, 1])
axes[0, 1].set_title('PACF - Original')
plot_acf(train_diff, ax = axes[1, 0])
axes[1, 0].set_title('ACF - Differenced')
plot_pacf(train_diff, ax = axes[1, 1])
axes[1, 1].set_title('PACF - Differenced')
plt.tight_layout()
plt.show()

# check again if data is stationary - perform the Augmented Dickey-Fuller test on the closing prices
adf_test_diff = adfuller(train_diff)
print('p-value:', adf_test_diff[1]) # data is non-stationary

# if we needed a second order of differencing - need to difference the difference - so, from p, d, q
# we now have p, 1, q - d represents the order of distancing to get stationarity, which is 1 (typically otherwise 0 or 2)

# use the most recent acf and pcf plots to subjectively determine p and q (keep one of these 0?)

# model fitting
model = ARIMA(train, order = (1, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# check residuals - should look like white noise; no obvious pattern
residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1, 2)
residuals.plot(title = 'Residuals', ax = ax[0])
residuals.plot(title = 'Density', kind = 'kde', ax = ax[1])
plt.tight_layout()
plt.show()
# ACF / PACF plots of the residuals; code not in book
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (10, 8))
plot_acf(residuals, ax = ax1)
ax1.set_title('Autocorrelation of Residuals')
plot_pacf(residuals, ax=ax2)
ax2.set_title('Partial Autocorrelation of Residuals')
plt.tight_layout()
plt.show()
acf_res = plot_acf(residuals) # standalone; not in book
pacf_res = plot_pacf(residuals) # standalone; not in book

# forecast
forecast_test = model_fit.forecast(len(test)) # 173 and change
close_prices['Forecast'] = [None] * len(train) + list(forecast_test)
close_prices.plot()
plt.xlabel('Trading Days:\n'
           'October 1, 2023 through April 30, 2024')
plt.ylabel('Differences in Closing Prices')
plt.title('Actual vs. Forecasted Closing Price - AAPL')
plt.grid()
plt.show()

# subset data on last 22 trading days
last_22_days = close_prices.tail(22)
# calculate the mean closing price for the last 22 rows; compare to #173+
mean_closing_price = last_22_days['Close'].mean()
print(mean_closing_price)

# exponential smoothing
# ses
ses_model = SimpleExpSmoothing(train).fit()
print(ses_model.summary()) # not in book
# des
des_model = ExponentialSmoothing(train, trend = 'add').fit()
print(des_model.summary()) # not in book

# h-w
hw_model = ExponentialSmoothing(train,
                                trend = 'add',
                                seasonal = 'add',
                                seasonal_periods = 5).fit()
print(f"AIC: {hw_model.aic}")
print(f"BIC: {hw_model.bic}")
forecast_test = hw_model.forecast(len(test)) # 173 and change
close_prices['Forecast'] = [None] * len(train) + list(forecast_test)
close_prices.plot()
plt.xlabel('Trading Days:\n'
           'October 1, 2023 through April 30, 2024')
plt.ylabel('Differences in Closing Prices')
plt.title('Actual vs. Forecasted Closing Price - AAPL')
plt.grid()
plt.show()

# basic stats - code not in book
minimum_value = np.min(forecast_test)
print(minimum_value)
maximum_value = np.max(forecast_test)
print(maximum_value)
mean_value = np.mean(forecast_test)
print(mean_value)

