#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:07:01 2020

@author: teddyrosenbaum
"""

import pandas as pd
df = pd.read_csv('life-expectancy-years-vs-real-gdp-per-capita-2011us.csv')
x = len(df)

def nContains(data,item):
    for i in range(len(data)):
        if(data[i]  == item):
            return False
    return True

def firstIndex(entity):
    for i in range(x):
        if(df['Entity'][i] == entity):
            return i

def lastIndex(entity):
    e = 0;
    for i in range(x):
        if(df['Entity'][i] == entity):
            e = i
    return e

def countryData(entity = 'Afghanistan'):
    s = firstIndex(entity)
    e = lastIndex(entity) + 1
    return df[s:e]

entityList = []
for i in range(x):
    if(nContains(entityList,df['Entity'][i])):
        entityList.append(df['Entity'][i])



data = {}
for i in range(len(entityList)):
    data.update({entityList[i] : countryData(entityList[i])})



'''
js comments
-----------
 - You've got a pretty clean dataset, which is good. But you have a bit of 
   cleaning up to do with this dataset. You have the tools to do this, it's
   just a matter of doing it. 

 - 9/10

'''
