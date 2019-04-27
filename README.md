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



## Prophet Model performance
**RMSE: 81.32**
