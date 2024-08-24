#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kimhyunji
"""

import pandas as pd
from datetime import datetime
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from statsmodels.tsa.stattools import acf, q_stat
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.dates as mdates

data = pd.read_csv(file_path, thousands=',')
data

data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')
data= data.drop(columns=['Vol.'])

##########
#EMA12, EMA26
##########

data['EMA12'] = data['Price'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Price'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['MACD'], label='MACD', color='red')
plt.plot(data['Date'], data['Signal_Line'], label='Signal Line', color='blue')

plt.gcf().autofmt_xdate()  
plt.title('MACD and Signal Line')
plt.xlabel('Date')  
plt.ylabel('Value')  
plt.legend()
plt.show()

##########
#RSI
##########
delta = data['Price'].diff()

gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)

avg_gain = gain.rolling(window=14, min_periods=14).mean().dropna()
avg_loss = loss.rolling(window=14, min_periods=14).mean().dropna()


for i in range(14, len(avg_gain)):
    avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * 13 + gain.iloc[i]) / 14
    avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * 13 + loss.iloc[i]) / 14


rs = avg_gain / avg_loss


rsi = 100 - (100 / (1 + rs))
data = data.assign(RSI=rsi)
data.head(20)  # 첫 20개 행을 출력하여 확인


plt.figure(figsize=(10, 7))
plt.plot(data['Date'], data['RSI'], label='RSI', color='purple')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gcf().autofmt_xdate()
plt.title('RSI over time')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.axhline(70, color='red', linestyle='--', linewidth=0.5)
plt.axhline(30, color='green', linestyle='--', linewidth=0.5)
plt.legend()
plt.show()

##########
#SMA5, SMA20
##########
window_length_5 = 5  
window_length_20 = 20 
data['SMA5'] = data['Price'].rolling(window=window_length_5).mean()
data['SMA20'] = data['Price'].rolling(window=window_length_20).mean()
print(data[['Date', 'Price', 'SMA5','SMA20']].head(25))

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Price'], label='Price', color='gray', alpha=0.5)
plt.plot(data['Date'], data['SMA5'], label='5-Week SMA', color='red')
plt.plot(data['Date'], data['SMA20'], label='20-Week SMA', color='blue')
plt.title('Price and SMA Crosses')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

def calculate_wma(data, period):
    weights = np.arange(1, period + 1)
    return data.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)


##########
#WMA5, WMA20
##########
data['WMA5'] = calculate_wma(data['Price'], 5)
data['WMA20'] = calculate_wma(data['Price'], 20)

print(data[['Date', 'Price', 'WMA5', 'WMA20']].tail(25))

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Price'], label='Price', color='gray', alpha=0.5)
plt.plot(data['Date'], data['WMA5'], label='5-Week WMA', color='red')
plt.plot(data['Date'], data['WMA20'], label='20-Week WMA', color='blue')
plt.title('Price and WMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()


##########
#CCI
##########
data['TP'] = (data['High'] + data['Low'] + data['Price']) / 3
data['SMA_TP'] = data['TP'].rolling(window=20).mean()
data['MD'] = data['TP'].rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
data['CCI'] = (data['TP'] - data['SMA_TP']) / (0.015 * data['MD'])
print(data[['Date', 'TP', 'SMA_TP', 'MD', 'CCI']].tail(25))

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['CCI'], label='CCI', color='blue')
plt.axhline(100, color='red', linestyle='--', linewidth=0.5)  
plt.axhline(-100, color='green', linestyle='--', linewidth=0.5)  
plt.title('CCI over time')
plt.xlabel('Date')
plt.ylabel('CCI')
plt.legend()
plt.show()


##########
# Stochastic %K
##########
n = 14
data['L14'] = data['Low'].rolling(window=n).min()
data['H14'] = data['High'].rolling(window=n).max()
data['%K'] = 100 * ((data['Price'] - data['L14']) / (data['H14'] - data['L14']))
print(data[['Date', '%K']].tail(25))

plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['%K'], label='%K', color='blue')
plt.axhline(80, color='red', linestyle='--', linewidth=0.5) 
plt.axhline(20, color='green', linestyle='--', linewidth=0.5) 
plt.title('Stochastic %K over time')
plt.xlabel('Date')
plt.ylabel('%K')
plt.legend()
plt.show()

