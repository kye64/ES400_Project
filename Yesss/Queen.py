# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:09:34 2019

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab 

df1998 = pd.read_csv('1998.csv', header=2, usecols = [0,1,2,3,4,5])
time_seq = pd.to_datetime(df1998[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df1998.index = time_seq
df1998 = df1998.drop(columns=['Year', 'Month', 'Day','Hour', 'Minute'])
resample1998 = df1998.resample('D').max()


df1999 = pd.read_csv('1999.csv', header=2, usecols = [0,1,2,3,4,5])
time_seq = pd.to_datetime(df1999[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df1999.index = time_seq
df1999 = df1999.drop(columns=['Year', 'Month', 'Day','Hour', 'Minute'])
resample1999 = df1999.resample('D').max()

df = pd.read_csv('2000.csv', header=2, usecols = [0,1,2,3,4,5])
time_seq = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
df.index = time_seq
df = df.drop(columns=['Year', 'Month', 'Day','Hour', 'Minute'])
resample2000 = df.resample('D').max()




























print(resample1998.head())
print(resample1999.head())
print(resample2000.head())

#group = df.groupby(pd.Grouper(freq='D'))
#df['GHI'].max()

#print(group.head())


#print grouped

#df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]]



#grouped = df.groupby(pd.TimeGrouper('D'))
#grouped['power'].max()

