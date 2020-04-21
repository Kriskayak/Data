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