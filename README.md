# Predictive-Rainfall-forecasting
A seasonality based predictive model for Rainfall prediction in Kerala using data from www.gov.in

A combination of two models:
- SARIMAX (Seasonal Autoregressive Integrated Moving Average)
- Prophet by Facebook

SARIMAX helps model time series data by adding the following hyperparameters:
- P: Seasonal autoregressive order.
- D: Seasonal difference order.
- Q: Seasonal moving average order.
- m: The number of time steps for a single seasonal period.

Time series analysis on rainfall data over the past 50 years were considered to accurately predict the rainfall pattern for any given year.


## SARIMAX Model performance
**RMSE: 90.15**

![sarimax](https://user-images.githubusercontent.com/29833297/56853418-1ce5e880-6945-11e9-9e9f-fdcd657226b3.png)


## Prophet Model performance
**RMSE: 81.32**
![alt text]https://raw.githubusercontent.com/parasnaren/predictive-rainfall-forecasting/edit/master/prophet.png
