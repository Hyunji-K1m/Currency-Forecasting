#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kimhyunji
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
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
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from arch import arch_model
##############################################################################
#function
##############################################################################
def month_to_date (date_string):
    year, month, day = date_string.split('-')
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

data_path='file_path'
data = pd.read_excel(data_path)
data = data.iloc[19:].set_index(data.columns[0])
data.index = data.index.astype(str)  
data.index = data.index.map(month_to_date)
data

df=data.drop(['Price','Open','High','Low','Change %'], axis=1)
df

train_x = df.loc[data.index<'2023-12-01']
valid_x = df.loc[(data.index>='2023-12-01')&(data.index<'2024-02-01')]
test_x = df.loc[(data.index >='2024-02-01')]

train_y = data.loc[data.index<'2023-12-01', 'Price']
valid_y = data.loc[(data.index>='2023-12-01')&(data.index<'2024-02-01'), 'Price']
test_y = data.loc[(data.index >='2024-02-01'),'Price']



param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]
}

grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(train_x, train_y)
best_params = grid_search.best_params_
print("Best parameters:", best_params)

svr_best = SVR(kernel='rbf', C=best_params['C'], epsilon=best_params['epsilon'], gamma=best_params['gamma'])
svr_best.fit(train_x, train_y)


predictions = svr_best.predict(test_x)

print(f"MSE: {MSE(test_y, predictions)}")
print(f"MAE: {MAE(test_y, predictions)}")

test_predictions = svr_best.predict(test_x)
print(f"Test MSE: {MSE(test_y, test_predictions)}")
print(f"Test MAE: {MAE(test_y, test_predictions)}")


residuals = test_y - predictions

garch_model = arch_model(residuals, p=1, q=1)
garch_result = garch_model.fit(update_freq=5)

print(garch_result.summary())


rmse = np.sqrt(MSE(test_y, predictions))
print(f"Test RMSE: {rmse}")

p_values = garch_result.pvalues
print(f"Alpha p-value: {p_values['alpha[1]']}")
print(f"Beta p-value: {p_values['beta[1]']}")


plt.figure(figsize=(10,7))
plt.plot(train_x.index, train_y, color='blue', label='Train Data')
plt.plot(valid_x.index, valid_y, color='green', label='Validation Data')
plt.plot(test_x.index, test_y, color='red', label='Test Data')
plt.plot(test_x.index, predictions, color='orange', linestyle='--', label='SVR Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Train/Validation/Test Data and SVR Predictions for GBP/USD')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#########
#prediction
#########
start_date = '2024-04-01'
end_date = '2024-06-01'
prediction_dates = pd.date_range(start=start_date, end=end_date)
predicted_prices = svr_best.predict(prediction_dates)
predicted_prices_df = pd.DataFrame(predicted_prices, index=prediction_dates.index, columns=['Predicted_Price'])
print(predicted_prices_df)
plt.figure(figsize=(10,6))
plt.plot(data.index, data['Price'], color='blue', label='Actual Prices')
plt.plot(prediction_dates.index, predicted_prices_df['Predicted_Price'], color='orange', linestyle='--', label='Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual and Predicted GBP/USD Prices')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



"""
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5, 1],
    'gamma': ['scale', 'auto', 0.01, 0.1, 1, 10]
}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(df.drop(['TP'], axis=1), df['TP'])

best_params = grid_search.best_params_
print("Best parameters:", best_params)


optimal_svr = SVR(**best_params)
optimal_svr.fit(df.drop(['TP'], axis=1), df['TP'])

best_params = grid_search.best_params_
print("Best parameters:", best_params)


optimal_svr = SVR(**best_params)
optimal_svr.fit(df.drop(['TP'], axis=1), df['TP'])

best_params = grid_search.best_params_
print("Best parameters:", best_params)


optimal_svr = SVR(**best_params)
optimal_svr.fit(df.drop(['TP'], axis=1), df['TP'])


predictions = optimal_svr.predict(df.drop(['TP'], axis=1))

garch_model = arch_model(residuals, p=1, q=1)
garch_fit = garch_model.fit(update_freq=5)
print(garch_fit.summary())
"""
