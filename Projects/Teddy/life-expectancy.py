#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:07:01 2020

@author: teddyrosenbaum
"""

import matplotlib.pyplot as  plt
import pandas as pd
df = pd.read_csv('life-expectancy-years-vs-real-gdp-per-capita-2011us.csv')
df['Real GDP per capita in 2011US$ ($)'] = df['Real GDP per capita in 2011US$ ($)'].fillna(0.0)
df['Life expectancy at birth'] = df['Life expectancy at birth'].fillna(0.0)
df['Code'] = df['Code'].fillna('Na')
x = len(df)


def nContains(dataList,item):
    for i in range(len(dataList)):
        if(dataList[i]  == item):
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

def makeCountryDic():
    data = {}
    for i in range(len(entityList)):
        data.update({entityList[i] : countryData(entityList[i])})
    return data

def plot2yearsGDP(first,second):
    firstYear = []
    for i in range(x):
        if(df['Year'][i] ==  first and df['Real GDP per capita in 2011US$ ($)'][i] != 0.0 and df['Code'][i] != 'Na'):
            firstYear.append(df['Real GDP per capita in 2011US$ ($)'][i])
    secondYear = []
    for i in range(x):
        if(df['Year'][i] ==  second and df['Real GDP per capita in 2011US$ ($)'][i] != 0.0 and df['Code'][i] != 'Na'):
            secondYear.append(df['Real GDP per capita in 2011US$ ($)'][i])
    plt.hist(firstYear,bins=50,alpha=0.5)
    plt.hist(secondYear,bins=50,alpha=0.5)

def plot2yearsLife(first,second):
    firstYear = []
    for i in range(x):
        if(df['Year'][i] == first and df['Life expectancy at birth'][i] != 0.0 and df['Code'][i] != 'Na'):
            firstYear.append(df['Life expectancy at birth'][i])
    secondYear = []
    for i in range(x):
        if(df['Year'][i] ==  second and df['Life expectancy at birth'][i] != 0.0 and df['Code'][i] != 'Na'):
            secondYear.append(df['Life expectancy at birth'][i])
    plt.hist(firstYear,bins=50,alpha=0.5)
    plt.hist(secondYear,bins=50,alpha=0.5)


'''
js comments
-----------
 - You've got a pretty clean dataset, which is good. But you have a bit of 
   cleaning up to do with this dataset. You have the tools to do this, it's
   just a matter of doing it. 

 - 9/10

'''
