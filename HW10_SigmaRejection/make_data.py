import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv

# Set parameters of the parent distribution
mn = 0
sigma = 5
sz = 100

# Set the probability limit of "outlier"
prob = 0.04

# Fix seed for reproducible results 
np.random.seed(333)
x = np.random.normal(mn,sigma,sz-1)

# Aopend value corresponding to the probility limit
val  = erfinv(1-prob/sz)*np.sqrt(2)*sigma
x = np.append(x,-1*val)

# Quick plot, just to take a look
plt.ion()
plt.figure(1)
plt.clf()
plt.hist(x,bins=30)

# Randomize values in the array (so it is not so obvious)
inds = np.argsort(np.random.uniform(0,1,sz))
x = x[inds]

# Write out data
np.savetxt('test_distribution.txt',x)






































































































# Sigma Rejection
from statsmodels.robust.scale import mad
data = np.copy(x)
m    = 0.05
n    = np.sqrt(2.)*erfinv(1-(m/len(data)))
sig  = mad(data)
r    = [np.median(data)-(n*sig),np.median(data)+(n*sig)]
newx = np.array([i for i in data if r[0]<i<r[1]])
