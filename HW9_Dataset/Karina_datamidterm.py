import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from scipy import stats
import statsmodels
from statsmodels.stats.diagnostic import normal_ad

# First pass over data
data = pd.read_csv('Karina_data.csv')
data.shape
data.isna().sum()

'''
555 rows x 6 columns
Columns: "Unnamed:0", "pless", "pfault", "ordna", "ofault", "bin"
The first column just looks like an index
"pless" and "ordna" are useful for plotting histograms
"pfault", "ofault", and "bin" are just y/n or 0/1 values, not useful for the hist
It looks like there are just a couple rows of "ordna" that are missing a useful value,
so might be easiest just to drop those rows since there are so few of them
Going to also just drop the non-useful columns and reset the titles to clean up the data
'''

# Cleaning up the data
data = pd.read_csv('Karina_data.csv',
                    usecols=[1,3],
                    skiprows=[0],
                    header=0,
                    names=['pless','ordna'])

data = data.dropna(axis=0, how='any')
data.describe()

'''
Looks like there are some outliers in both sets looking at the quartiles, mean, and minimum/maximum values, so I
will remove them by calculating the z-scores of each value and creating a boolean array as to whether the z-scores
are less than 1 which I will use to index the original data (I also plotted this before removing them just to visualize 
the outliers but didn't keep that in this code)
'''

# Extracting each data set
pless = data['pless']
ordna = data['ordna']

# Removing the p-less outliers
pless_z = scipy.stats.zscore(pless)
pless_z = np.abs(pless_z)
pless_filtered = (pless_z < 1)
new_pless = pless[pless_filtered]

# Removing the ordna outliers
ordna_z = scipy.stats.zscore(ordna)
ordna_z = np.abs(ordna_z)
ordna_filtered = (ordna_z < 1)
new_ordna = ordna[ordna_filtered]


# Making the histograms
plt.ion()
plt.figure()
plt.clf()

pless_hist = plt.hist(new_pless,alpha=0.5)
ordna_hist = plt.hist(new_ordna,alpha=0.5)

'''
Well at least there are no outliers now
Going to make it look pretty
'''

pless_hist = plt.hist(new_pless,alpha=0.5,bins=np.arange(60)*1,label='pless')
ordna_hist = plt.hist(ordna,alpha=0.5,bins=np.arange(60)*1,label='ordna')
plt.xlim(30,55)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig('Karina_hist',dpi=300)

'''
Basic Data Analysis
'''

# Mean, med, and std for p-less
pless_mean = np.mean(new_pless)
pless_med = np.median(new_pless)
pless_std = np.std(new_pless)

# Mean, med, and std for ordna
ordna_mean = np.mean(new_ordna)
ordna_med = np.median(new_ordna)
ordna_std = np.std(new_ordna)

# Printing the previous values
print "pless: Mean = " + str(pless_mean) + "; Median = " + str(pless_med) + "; Std = " + str(pless_std)
print "ordna: Mean = " + str(ordna_mean) + "; Median = " + str(ordna_med) + "; Std = " + str(ordna_std)

# A-D test on both pless and ordna -- I couldn't figure out how to print the name and then the statement, so it outputs in the order in categories
categories = [new_pless, new_ordna]
for x in categories:
    ad, p_ad = statsmodels.stats.diagnostic.normal_ad(x)
    print p_ad
    if p < 0.05:
        print "p is less than 0.05, so the data are not consistent with a normal model"
    else:
        print "p is greater than 0.05, so the data are consistent with a normal model"

'''
Because they are normal distributions, isn't the precision of the mean just the error? (#12)
So 1 std on each is 68% confidence. Not sure if there's supposed to be something else.
'''

# K-S test
ks, p_ks = scipy.stats.ks_2samp(new_pless,new_ordna)
print p_ks
if p_ks < 0.05:
    print "The data are not likely drawn from the same parent distribution"
else:
    print "The data may likely be drawn from the same parent distribution"

# Student's T-test
ttest, p_t = scipy.stats.ttest_ind(new_pless, new_ordna) # I don't have time to figure out how to calculate the confidence right now
print p_t
if p_t < 0.05:
    print "The means are not consistent via the Student's t test"
if p_t > 0.05:
    print "The means are consistent via the Student's t test"


'''
The two distributions are both consistent with normal distributions (proved by the Andersen-Darling test), though
they are not drawn from the same parent distribution (as shown by the Komolgov-Smirov test). The Student's t-test
also shows that the means are not consistent. 
'''