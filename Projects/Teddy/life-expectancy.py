#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:07:01 2020

@author: teddyrosenbaum
"""

import pandas as pd
pd.options.display.max_columns =None
pd.options.display.max_rows =40
filename = 'life-expectancy-years-vs-real-gdp-per-capita-2011us.csv'
df = pd.read_csv(filename)
print(df)


'''
js comments
-----------
 - You've got a pretty clean dataset, which is good. But you have a bit of 
   cleaning up to do with this dataset. You have the tools to do this, it's
   just a matter of doing it. 

 - 9/10

'''
