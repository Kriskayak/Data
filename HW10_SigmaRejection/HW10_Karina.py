import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels
from statsmodels import robust

'''
Looking at the data
'''

data = pd.read_csv('test_distribution.txt', header=None, names=['data'])
data = data['data']

plt.ion()
plt.figure()
plt.clf()

hist = plt.hist(data, alpha=0.5, bins=10)
# I don't notice any significant outliers, though perhaps some of those that are less than -15 are skewing the distribution a bit

'''
Sigma rejection
'''

# Median and MAD of original data
median = np.median(data)
mad = statsmodels.robust.scale.mad(data)

# Inverse erf: I don't understand how to get the highest/lowest value with this function or how to do sigma rejection with this function
# What are you supposed to do if with it if the only domain it creates useful outputs for is -1 to 1 and the rest are infinity?
erf = scipy.special.erfinv(data)

# I understand sigma rejection, I don't understand inverse error
# I'm going to use this function even though I know it won't be accurate since it uses the mean to create the sigma rejected array
sigma_rejected, lower, upper = scipy.stats.sigmaclip(data,low=2.0,high=2.0)

# Median and MAD of sigma rejected
median_sr = np.median(sigma_rejected)
mad_sr = statsmodels.robust.scale.mad(data)

# Comparing original and sigma rejected
print median, median_sr
print mad, mad_sr

'''
 I think that the sigma_rejected values for mad and median are better estimates
 for the mean and std as they exclude the outliers outside of 2 sigma, which 
 allows the median and mad calculations to be more representative of the majority
 of the data.
'''
