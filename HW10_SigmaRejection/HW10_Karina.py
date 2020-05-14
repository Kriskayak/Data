import numpy as np
from scipy.special import erf, erfinv
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
mad = robust.scale.mad(data)

dist_sig = erfinv(0.99)*np.sqrt(2)   # Solved 1 = 100 * (1 - erf(x/np.sqrt(2)))
highest_expected = median + (mad * dist_sig)
lowest_expected = median - (mad * dist_sig)

sigreg = erfinv(1 - (0.05/100))*np.sqrt(2) # Solved 0.05 = 100 * (1 - erf(x/np.sqrt(2)))
lower_sigbound = median - (mad * sigreg)
upper_sigbound = median + (mad * sigreg)

filtered_data = [] # Will hold data that passes sigma rejection

for x in data:
  if lower_sigbound < x < upper_sigbound:
    filtered_data.append(x)

mean = np.mean(data)
std = np.std(data)
mean_f = np.mean(filtered_data)
std_f = np.std(filtered_data)


'''
 I think that the sigma_rejected values for mean and std are better estimates
 for the "true" mean and std since they reject the outliers and then take the traditional
 mean/std, which is more representative of the data than the median/mad.
'''


'''
js comments
-----------
 - You are not understanding what the error function is and how it is used. It
   returns a probability. The inverse error function is needed. 

 - There are sigma clip algorithms already written, your task was to understand
   how it works by writing your own. 

 - The sigmaclip parameters were way too stringent, and did not conform to the 
   directions

 - The sigma-rejected array is not valid

5/20

'''


