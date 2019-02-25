# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:43:03 2019

@author: Kevin
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pylab 
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
  
# print "resamples[0]: ", resamples[0]


np.asarray(resamples[0])
np.asarray(resamples[1])
np.asarray(resamples[2])
np.asarray(resamples[3])
np.asarray(resamples[4])
np.asarray(resamples[5])
np.asarray(resamples[6])
np.asarray(resamples[7])
np.asarray(resamples[8])
np.asarray(resamples[9])
np.asarray(resamples[10])
np.asarray(resamples[11])
np.asarray(resamples[12])
np.asarray(resamples[13])
np.asarray(resamples[14])
np.asarray(resamples[15])
np.asarray(resamples[16])
np.asarray(resamples[17])
np.asarray(resamples[18])
np.asarray(resamples[19])

df0 = pd.DataFrame(resamples[0])
df1 = pd.DataFrame(resamples[1])

# merge the dataframes of years 1998-2017 into one continuous dataframe
solution = pd.concat(resamples)
print (solution)

# find the GHI means grouped by days of each month
solute = solution.groupby([solution.index.month, solution.index.day]).mean()
print (solute)   

df0.plot(figsize=(15,6))
plt.show()
solution.plot(figsize=(15,6))
plt.show()


#print (resamples[0])
#print (resamples[19])

#a = resamples.groupby([resamples.index.month, resamples.index.day]).mean()

#dataf1 = pd.DataFrame({'col':dataframes[0]})
#print (dataf1)









#mean = np.mean( np.array([ resamples[0:19]]), axis=0 )
#a = np.array([resamples[0],resamples[1]])
#np.mean(zip(resamples[0],resamples[1]),axis=1)
    

# you can access items in the lists by indexing into them like: dataframes[0]






#group = df.groupby(pd.Grouper(freq='D'))
#df['GHI'].max()

#print(group.head())


#print grouped

#df.loc[df.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]]



#grouped = df.groupby(pd.TimeGrouper('D'))
#grouped['power'].max()