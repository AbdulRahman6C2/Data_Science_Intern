import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

## LOAD DATASET
df = pd.read_csv("time_series_data.csv")
df['observation_date'] = pd.to_datetime(df['observation_date'])
df.set_index('observation_date', inplace=True)
df.rename(columns={'MRTSSM448USN': 'value'}, inplace=True)

print(df.head())
print(df.info())

## PLOT TIME SERIES
plt.figure()
plt.plot(df, label='Original Data')
plt.title("Time Series Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()

## DECOMPOSITION
decomposition = seasonal_decompose(df, model='additive', period=12)
decomposition.plot()
plt.show()

## MOVING AVERAGE
df['Moving_Avg'] = df.rolling(window=7).mean()

plt.figure()
plt.plot(df['sales'], label='Original')
plt.plot(df['Moving_Avg'], label='Moving Average')
plt.legend()
plt.show()

## EXPONENTIAL SMOOTHING
exp_model = ExponentialSmoothing(df['sales'], trend='add', seasonal=None)
exp_fit = exp_model.fit()
df['Exp_Smoothing'] = exp_fit.fittedvalues

plt.figure()
plt.plot(df['sales'], label='Original')
plt.plot(df['Exp_Smoothing'], label='Exponential Smoothing')
plt.legend()
plt.show()

## TRAIN-TEST-SPLIT
train = df.iloc[:-10]
test = df.iloc[-10:]

## ARIMA MODEL
arima_model = ARIMA(train['sales'], order=(1, 1, 1))
arima_fit = arima_model.fit()

forecast = arima_fit.forecast(steps=len(test))

## EVALUATION(RMSE)
rmse = np.sqrt(mean_squared_error(test['sales'], forecast))
print("RMSE:", rmse)

## FORECAST PLOT
plt.figure()
plt.plot(train['sales'], label='Train')
plt.plot(test['sales'], label='Test')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.show()
