import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
from pylab import rcParams
from fbprophet import Prophet
from statsmodels.tsa.api import ExponentialSmoothing

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# All functions
def get_monthly(monthly, i, j):
    j += 1
    years = [x for x in range(i,j)]
    monthly = monthly.where(monthly['Date'].dt.year.isin (years)).dropna()
    monthly = monthly.set_index('Date')
    monthly = monthly['Rainfall'].resample('MS').mean()
    return monthly

def rainfall_plot(monthly):
    #monthly = get_monthly(monthly, y, y)
    monthly.plot(figsize=(15,6))
    plt.show()
    
def decompose_graph(monthly):
    #monthly = get_monthly(monthly, i, j)
    decomposition = seasonal_decompose(monthly, model='additive')
    fig = decomposition.plot()
    plt.show()
    
def get_train_param(monthly):
    #monthly = get_monthly(monthly, i, j)
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    print('Examples of parameter combinations for Seasonal ARIMA...')
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
    print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
    print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
    
    min_aic = 9999999999
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(monthly,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = mod.fit()
                #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                if results.aic < min_aic:
                    min_aic = results.aic
                    min_param = param
                    min_seasonal = param_seasonal
            except:
                continue
            
    print('ARIMA{}x{}12 - AIC:{}'.format(min_param, min_seasonal, min_aic))
    return min_param, min_seasonal
    
            
def train_sarimax(monthly, order, seasonal):
    #monthly = get_monthly(monthly, i, j)
    mod = sm.tsa.statespace.SARIMAX(monthly,
                                    order=order,
                                    seasonal_order=seasonal,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    results = mod.fit(maxiter=200, method='nm')
    print(results.summary().tables[1])
    
    results.plot_diagnostics(figsize=(16, 8))
    plt.show()
    
    return results

def sarimax_predictions(X, y, results, train_start, pred_start):
    #y = get_monthly(y, train_start, pred_start)
    start = pd.to_datetime(str(pred_start) + '-01-01')
    end = pd.to_datetime(str(pred_start) + '-12-01')
    pred = results.get_prediction(start=start, end=end, dynamic=False)
    pred_ci = pred.conf_int()
    x = pd.concat([X,y], axis=0)
    ax = x.plot(label='observed')
    pred.predicted_mean.plot(ax=ax, label='SARIMAX Forecast', alpha=.7, figsize=(14, 7))
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rainfall')
    plt.legend()
    plt.show()
    return pred

def train_prophet(x):
    prophet = Prophet()
    x.columns=['ds', 'y']
    p = prophet.fit(x)
    return p

def prophet_predictions(prophet, X, y):
    y.columns=['ds', 'y']
    forecast = prophet.predict(y)
    #forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    X = X.set_index('ds')
    y = y.set_index('ds')
    pred = forecast.copy()[['ds','yhat']].set_index('ds')
    
    x = pd.concat([X,y], axis=0)
    #ax = x.plot(label='observed')
    plt.plot(x, label='Actual')
    plt.plot(pred, label='Prophet Forecast', alpha=.7)
    plt.xlabel('Date')
    plt.ylabel('Rainfall')
    plt.legend()
    plt.show()
    #prophet.plot_components(forecast)
    return forecast['yhat']
    
# Start here
def get_in_2d_array(path, state):
    df0 = pd.read_excel(path)
    monthly = df0.where(df0['SUBDIVISION'] == state).dropna().reset_index(drop=True).drop('SUBDIVISION', axis=1)
    monthly['YEAR'] = monthly['YEAR'].astype('int64').astype('str')
    
    d = {}
    months =  monthly.columns[1:13]
    for i in range(len(monthly)):
        for j in range(12):
            if j <= 8:
                year = monthly.at[i, 'YEAR'] + '-0' + str(j+1) + '-01'
            else:
                year = monthly.at[i, 'YEAR'] + '-' + str(j+1) + '-01'
            d[year] = monthly.at[i, months[j]]
            
    monthly = pd.DataFrame.from_dict(d, orient='index').reset_index()
    monthly.columns = ['Date','Rainfall']
    monthly['Date'] = pd.to_datetime(monthly['Date'])
    return monthly

def train_predict_holtwinter(X, y):
    X = X.set_index('ds')
    fit = ExponentialSmoothing(np.asarray(X) ,seasonal_periods=12 ,trend='add', seasonal='add',).fit()
    pred = fit.forecast(len(y))
    pred = pd.DataFrame(pred).set_index(y['ds'])
    y = y.set_index('ds')
    plt.figure(figsize=(16,8))
    x = pd.concat([X,y], axis=0)
    plt.plot(x, label='Actual')
    plt.plot(pred, label='Holt Winter forecast', alpha=.7)
    plt.legend(loc='best')
    plt.show()
    return pred[0]
    
def get_rmse(true, pred):
    return (((true - pred) ** 2).mean() ** .5)
    

######################################
if __name__ == "__main__":
    
    """
    run this file with own values of i and j
    """
    
    i = 1980
    j = 2017
    
    path = 'datafile.xls'
    states = ['Kerala','Coastal Karnataka','Konkan & Goa']
    monthly = get_in_2d_array(path, states[0])
    monthly = get_monthly(monthly, i, j)
    
    # Visualisation
    rainfall_plot(monthly)
    decompose_graph(monthly)
    
    # For SARIMAX
    X = monthly[str(i): str(j-1)]
    y = monthly[str(j):]
    order, seasonal = get_train_param(X)
    results = train_sarimax(X, order, seasonal)
    sarimax_pred = sarimax_predictions(X, y, results, i, j)
    
    # For Prophet
    X = monthly[str(i): str(j-1)].reset_index()
    y = monthly[str(j):].reset_index()
    prophet = train_prophet(X)
    prophet_pred = prophet_predictions(prophet, X, y)
    
    # For Holt Winter
    holt_pred = train_predict_holtwinter(X, y)
    
    print('SARIMAX RMSE: ', get_rmse(y['y'], sarimax_pred.predicted_mean.reset_index()[0]))
    print('Prophet RMSE: ', get_rmse(y['y'], prophet_pred))
    print('Holt Winter RMSE: ', get_rmse(y['y'].values, holt_pred))

################################################

import collections
import itertools
t = input()
t = int(t)
while t > 0:
    n = int(input())
    for i in range(n):
        x, y
    n, a, b, c = list(input().split(' '))
    c = int(c)
    a = collections.Counter([int(d) for d in a])
    b = [int(d) for d in b]
    perms = set(itertools.permutations(b))
    flag = 0
    for perm in perms:
        num = ''
        for i in perm:
            num += str(i)
        num = int(num)
        num = str(c-num)
        tmp = collections.Counter([int(d) for d in num])
        if tmp == a:
            flag = 1
            print('YES')
            break
    if not flag:
        print('NO')
    t-=1

"""
"""
#df1 = pd.read_excel('datafile.xls')
#df1['SUBDIVISION'].unique()
#monthly.to_csv('kerala.csv', index=False)
df2 = pd.read_excel('datafile2.xls')
df2 = df2.where(df2['SUBDIVISION'] == 'KERALA').dropna(how='all').reset_index(drop=True)

df3 = pd.read_excel('datafile3.xls')

df4 = pd.read_excel('datafile4.xls')
"""