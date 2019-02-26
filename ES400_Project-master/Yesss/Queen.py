# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:43:03 2019

@author: Kevin
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pylab as pyl
import os
import datetime
import matplotlib

import warnings
import itertools

warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm

matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'


# make a list of names of csv files in the Current Working Directory (CWD)
csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv')]
dataframes = []
Whresamples = []
Whperday = []
meanyear = []
pd.set_option('display.max_rows',10000 )



# iterate over each of the filenames stored in the csv_files list
# store the df in the dataframes list
# store the resample dataframe in the resamples list
for f in csv_files:
    df = pd.read_csv(f, header=2, usecols = [0,1,2,3,4,5])
    df.index = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
    dataframes.append(df)
    df = df.resample('H').sum()
    #print df.head(24)
    Whresamples.append(df)
    df = df.resample('D').mean()
    #print df.head(35)
    Whperday.append(df)
    df = df.resample('A').mean()
    meanyear.append(df)
    

# merge the dataframes of years 1998-2017 into one continuous dataframe

solution = pd.concat(Whperday)
kWhperday = solution.div(1000)
y = kWhperday['GHI']
y.interpolate(inplace=True)
print y.head(365)


#sol1 = pd.concat(Whperday[0:5])
#sol1 = sol1.div(1000)
#sol1.plot(figsize=(15,6))
#plt.show()
#meanyears = pd.concat(meanyear)
#meanyears.plot(figsize=(15,6))
#plt.show()
# find the GHI means grouped by days of each month

duo = y.plot(figsize=(15,6))
plt.show()

fig = duo.get_figure()
fig.savefig("ARIMA1.png",bbox_inches='tight')

# trend, seasonality, noise
pyl.rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()



# ARIMA
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
#print('Examples of parameter combinations for Seasonal ARIMA...')
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
#print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
#print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#for param in pdq:
#    for param_seasonal in seasonal_pdq:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(y,order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False)
#            results = mod.fit()
#
#            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#        except:
#            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 0, 0, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(12, 16))
plt.show()

#mod = sm.tsa.statespace.SARIMAX(y,
#                                order=(1, 1, 0),
#                                seasonal_order=(0, 1, 0, 12),
#                                enforce_stationarity=False,
#                                enforce_invertibility=False)
#results = mod.fit()
#print(results.summary().tables[1])
#results.plot_diagnostics(figsize=(12, 16))
#plt.show()

fig = results.get_figure()
fig.savefig("1.png",bbox_inches='tight')


pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(30, 8))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('GHI(kiloWatts per meter squared per day)')
plt.legend()
plt.show()

fig = ax.get_figure()
fig.savefig("2.png",bbox_inches='tight')


y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))

pred_uc = results.get_forecast(steps=500)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(16, 8))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('GHI(kiloWatts per meter squared per day)')
plt.legend()
plt.show()

fig = ax.get_figure()
fig.savefig("3.png",bbox_inches='tight')

        
        
#fig = plot.get_figure()
#fig.savefig('GHI.png', bbox_inches='tight')



