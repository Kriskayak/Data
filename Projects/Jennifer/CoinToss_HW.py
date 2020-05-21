#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jennifer Yim

Coin Toss HW

Created on Thu May 21 19:11:08 2020

@author: jennifer
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("/Users/jennifer/Jen_Data_Science/Coin Toss Experiment_ DataSci 2020 - Sheet1.csv")

def likelyhood(data,p):
       
    likely = 1
       
    for i in data:
           
        if i == 'H':
               likely = likely * p
           
        if i == 'T':
               likely = likely * (1-p)
               
    return likely

plotIndex = 221

for name in df.columns[1:]:
    xValues = []
    yValues = []

    for i in np.arange(0, 1, 0.01):
        xValues.append(i)
        yValues.append(likelyhood(df[name],i))
    
    plt.subplot(plotIndex)
    plotIndex += 1
    
    plt.plot(xValues,yValues)
    plt.title(name)
    plt.xlabel('p')
    plt.ylabel('likelyhood')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.5, wspace=0.5)
