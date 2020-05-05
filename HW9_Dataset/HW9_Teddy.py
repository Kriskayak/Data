#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:46:07 2020

@author: teddyrosenbaum
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sp
pd.options.display.max_columns =None
pd.options.display.max_rows =40
filename = 'Teddy_data.csv'
df = pd.read_csv(filename,header=0,index_col=0) 
df['pless'] = df['pless'].fillna(0.0)
df['ordna'] = df['ordna'].fillna(0.0)
#set nan values to 0
#column names are pless, pfault ordna ofault and bin
#I don't know what they mean

pless = df['pless'].values.tolist()
ordna = df['ordna'].values.tolist()
#there are 555 rows of data with some bad values as 0 as well as outlieing data
#remove 0s from  nans
for i in range(pless.count(0.0)):
    pless.remove(0.0)
for i in range(ordna.count(0.0)):
    ordna.remove(0.0)
#need to remove outlier values
pless1 = []
for i  in range(len(pless)):
    if(pless[i]<100.0 and pless[i]>0.0):
        pless1.append(pless[i])
pless = pless1

ordna1 = []
for i  in range(len(ordna)):
    if(ordna[i]<100.0 and ordna[i]>0.0):
        ordna1.append(ordna[i])
ordna = ordna1

print('Good data in pless: ' + str(len(pless)))
print('Good data in ordna: ' + str(len(ordna)))


hist = plt.hist(pless,alpha=0.5,bins=25)
plt.hist(ordna,alpha=0.5,bins=25)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()
plt.savefig('Teddy_Hist.png',dpi=300)

print('Pless mean: ' + str(np.mean(pless)))
print('Pless median: ' + str(np.median(pless)))
print('Pless std: ' + str(np.std(pless)))

print('Ordna mean: ' + str(np.mean(ordna)))
print('Ordna median: ' + str(np.median(ordna)))
print('Ordna std: ' + str(np.std(ordna)))

print('Probability pless is normaly distributed: ' + str(sp.normaltest(pless)))
print('Probability ordna is normaly distributed: ' + str(sp.normaltest(ordna)))

print('Probability pless and ordna are consistant: ' + str(sp.ks_2samp(pless,ordna)))

#pless could be normaly distributed while ordna is likely not
#there is a high probablity that pless and ordna are consistant
#both data sets have similar means, medians, and stds

