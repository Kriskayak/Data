# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.stats.diagnostic import normal_ad
import pandas as pd
import scipy as sy

# for plotting the histograms
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

# %%
# load in the data
# (for some reason would not load without the full file address)

# js - Probably because your console was being run from somewhere else. Check
# this by typing "pwd" in your ipython console.
pd.options.display.max_columns =None
pd.options.display.max_rows =40
filename = 'D:\Code\Thacher\datasci2020\HW9_Dataset\Ian_data.csv'
df = pd.read_csv(filename)
df
# distribution 1 is called "pless"
# distribution 2 is called "ordna"
# data also includes binary "bin" value
# will need to remove the values with fault 'n' instead of 'y'
# data has 555 points, minus the null values for pless

# %%
# set up distributions
dist1 = df.pless
dist2 = df.ordna

# %%
# filtering out the null values
# js - slick! :) Pandas also has this capability. 
dist1 = list(filter((-999.0).__ne__, dist1))


# %%
# first plot of data
# js - normed -> density
hist = plt.hist(dist1,bins=100,normed=True,edgecolor='none')

plt.hist(dist2,bins=100,normed=True,edgecolor='none')

# %%
# ordna has outliers at 300 and 80, will remove
dist2 = list(filter((300.0).__ne__, dist2))
dist2 = list(filter((80.0).__ne__, dist2))
# not most effective but couldn't make auto outlier removal work
# would use median and stdev to filter anything lying too far out

# new plot:
# js - binning not exactly the same, only the number of bins!
hist = plt.hist(dist1,bins=80,normed=True,edgecolor='none',label='pless')
plt.hist(dist2,bins=80,normed=True,edgecolor='none',label='ordna')
plt.xlabel('values')
plt.ylabel('frequency')
#plt.savefig('Ian_hist.png',dpi=300)

# %%
print('pless data:')
print('mean')
print(np.mean(dist1))
print('median')
print(np.median(dist1))
print('stdev')
print(np.std(dist1))

print('ordna data:')
print('mean')
print(np.nanmean(dist2))
print('median')
print(np.nanmedian(dist2))
print('stdev')
print(np.nanstd(dist2))
# data looks very similar

# %%
# invdividual a-d tests to see if ditributions are normal
#pless
stat,crit,sig = sy.stats.anderson(dist1, dist='norm')
print(crit)
print(stat)
# stat is lower than the critical values so the ditribution does follow the normal

#ordna
stat2,crit2,sig2 = sy.stats.anderson(dist2, dist='norm')
print(crit2)
print(stat2)
# distribution seems to have null values but I can't figure to remove them
# I would assume with their similarities this is also a normal distribution


# %%
# Kolmogorov-smirnov to compare the two lists
from scipy.stats import ks_2samp
ks_2samp(dist1,dist2)

#the low p-value suggests that the two are not taken from the same distribution
# these two distributions have similar values but are obviously from different sources
# I would say that it's a coincedence, and a good way to show that even if data
# look the same, they may be entirely different



'''
js comments
-----------
 - Great commenting!

 - Legend on your histogram?

 - Solid analysis and conclusion

 98/100

'''

