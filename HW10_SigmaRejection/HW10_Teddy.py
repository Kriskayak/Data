#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:30:09 2020

@author: teddyrosenbaum
"""

import pandas as pd
import matplotlib.pyplot as plt
# js - This is a better way to import the functions you want...
from scipy.special import erf,erfinv
df = pd.read_csv('test_distribution.txt',names=['data'])

plt.hist(df['data'],bins=25)

robustMean = sp.median(df['data'])
print('robust mean: ' + str(robustMean))

# js - you can use the mean absolute deviation (MAD) or numpy's std function...
tot = 0
for i in range(len(df)):
    tot += (robustMean - df['data'][i]) * (robustMean - df['data'][i])
std = sp.sqrt(tot/(len(df)-1))
print('std from robust mean: ' + str(std))

distance = erfinv(1.0-(1.0/len(df)))*std*sp.sqrt(2)
print('distance: ' + str(distance))

# js - I appreciate your CS approach to these problems. But in a high level language
# like python, this is totally unnecessary. You've done all the work in the lines above
# to exclude all datapoints. You can do it in one line...

newdf = df[np.abs(df['data']-robustMean) < distance]


def rmProb(data,prob=0.05):
    index = []
    for i in range(len(df)):
        if(1.0-erf(sp.absolute(robustMean-df['data'][i])/(std*sp.sqrt(2))) <= 1-prob):
            index.append(i)
    data = []
    for i in range(len(index)):
        if i in index:
            data.append(df['data'][i])
    newdf = pd.DataFrame(data,columns=['data'])
    return newdf
newdf = rmProb(df)
print('mean / std of data: ' + str(sp.mean(df['data'])) + ' / ' + str(sp.std(df['data'])))
print('mean / std of filtered data: ' + str(sp.mean(newdf['data'])) + ' / ' + str(sp.std(newdf['data'])))


'''
js comments
-----------
 - Your CS background has given you lots of tools. It would be helpful for you to take
   some time to fully appreciate a high level language like python. See my one liner 
   that bypasses 12 of your lines.

 - Also, your logic is convoluted and too complex. You've got the right idea, but your 
   approach needs to be refined and simplified.

17/20

'''
