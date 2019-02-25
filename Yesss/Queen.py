# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:09:34 2019

@author: Kevin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab 
import os

# make a list of names of csv files in the Current Working Directory (CWD)
csv_files = [f for f in os.listdir(os.getcwd()) if f.endswith('.csv')]
dataframes = []
resamples = []

# iterate over each of the filenames stored in the csv_files list
# store the df in the dataframes list
# store the resample dataframe in the resamples list
for f in csv_files:
	df = pd.read_csv(f, header=2, usecols = [0,1,2,3,4,5])
	df.index = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
	df = df.drop(columns=['Year', 'Month', 'Day', 'Hour', 'Minute'])
	dataframes.append(df)
	resamples.append(df.resample('D').max())

# you can access items in the lists by indexing into them like: dataframes[0]

#group = df.groupby(pd.Grouper(freq='D'))
#df['GHI'].max()

#print(group.head())


#print grouped

#df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]]



#grouped = df.groupby(pd.TimeGrouper('D'))
#grouped['power'].max()

