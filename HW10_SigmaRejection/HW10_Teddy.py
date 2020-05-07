#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:30:09 2020

@author: teddyrosenbaum
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
df = pd.read_csv('test_distribution.txt',names=['data'])

plt.hist(df['data'],bins=25)

robustMean = sp.median(df['data'])
print('robust mean: ' + str(robustMean))

tot = 0
for i in range(len(df)):
    tot += (robustMean - df['data'][i]) * (robustMean - df['data'][i])
std = sp.sqrt(tot/(len(df)-1))
print('std from robust mean: ' + str(std))

distance = sp.special.erfinv(1.0-(1.0/len(df)))*std*sp.sqrt(2)
print('distance: ' + str(distance))

def rmProb(data,prob=0.05):
    index = []
    for i in range(len(df)):
        if(1.0-sp.special.erf(sp.absolute(robustMean-df['data'][i])/(std*sp.sqrt(2))) <= 1-prob):
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