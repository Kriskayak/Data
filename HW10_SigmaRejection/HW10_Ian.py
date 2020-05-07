
#import needed libraries and create gaussian formula

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sy
import statsmodels
import statsmodels.robust
from astropy.stats import sigma_clip

def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

# load in data (I still can't get the data to be read without a full filepath)
df = pd.read_csv('D:\Code\Thacher\datasci2020\HW10_SigmaRejection\\test_distribution.txt',names=["value"])


# create histogram
plt.hist(df,bins=10)
#plt.show()

# there aren't any points that look to be too far out, but I would
# probably trim off any values past +- 10

median = np.median(df)
print(median)
mad = statsmodels.robust.scale.mad(df,c=median)
print(mad)

# sigma rejection using function from astropy
# set sigma=2 to reject everything past 2 stdevs
df2 = sigma_clip(df,sigma=2)
#print(df2)

# removed 24 points


''' didn't work because the values in the new array don't match anything, ended up scrapping due to use of sigma_clip

# removes values from df which match those remaining in dfremove
cond = df['value'].isin(dfremove['value'])
df.drop(df[cond].index, inplace = True)

# see if deletion of outliers worked
print(df)
'''

# find and compare the mean and stdev of the two distributions, since the median and therefore MAD haven't changed
print(np.mean(df))
print(np.mean(df2))
print(np.std(df))
print(np.std(df2))
# both the mean and stdev decreased with the improved distribution, which means
# that the outliers were both skewing the data and making it less precise

# comparing the orignal median and MAD to the new mean and stdev, it shows that
# median and MAD are much more accurate than mean and stdev withOUT sigma rejection,
# so they can be seen as a better estimation for the original dataset