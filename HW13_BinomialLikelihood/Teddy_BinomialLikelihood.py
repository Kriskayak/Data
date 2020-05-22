#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:01:18 2020

@author: teddyrosenbaum
"""

import matplotlib.pyplot as plt
import numpy as np
flips = 1000
bias = 0.5
def flip(flips = 100, bias = 0.5):
    h = 0
    for i in range(flips):
        r = np.random.rand()
        if(r<bias):
            h = h + 1
            
    x = np.array(range(100))/100.0
    y = (x**h) * (1-x)**(flips-h)
    w = y.sum()
    y = y/w
    return np.average(x,weights = y)

def biasDist(flips = 100, bias = 0.5): 
    h = 0
    for i in range(flips):
        r = np.random.rand()
        if(r<bias):
            h = h + 1
            
    x = np.array(range(100))/100.0
    y = (x**h) * (1-x)**(flips-h)
    w = y.sum()
    y = y/w
    plt.plot(x,y)
    plt.xlabel('bias')
    plt.ylabel('probability')
    print(np.average(x,weights=y))
#I kind of went down a rabit hole but thought it was cool
def stdOfBias(bias = 0.5):
    std = []
    for i in range(200):
        vals = []
        for j in range(100):
            vals.append(flip(i,bias))
        std.append(np.std(vals))  
    flips = np.array(range(200))
    plt.plot(flips,std)
            
            