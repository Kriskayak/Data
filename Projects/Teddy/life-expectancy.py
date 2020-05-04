#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:07:01 2020

@author: teddyrosenbaum
"""

import pandas as pd
df = pd.read_csv('life-expectancy-years-vs-real-gdp-per-capita-2011us.csv')
x = len(df)
def countryData(countryCode = 'AFG'):
    country = pd.DataFrame(columns = ['Entity','Code','Year','Life expectancy at birth','Real GDP per capita in 2011US$ ($)','Population by country'])
    for i in range(x):
        if(df['Code'][i].equals(countryCode)):
            country = pd.concat(country,df[i])
    return country
codeList = []
for i in range(x):
    if(df['Code'][i] in codeList == False):
        codeList.append(df['Code'][i])
dataList = []
for i in range(len(codeList)):
    dataList.append(countryData(countryCode = codeList[i]))


'''
js comments
-----------
 - You've got a pretty clean dataset, which is good. But you have a bit of 
   cleaning up to do with this dataset. You have the tools to do this, it's
   just a matter of doing it. 

 - 9/10

'''
