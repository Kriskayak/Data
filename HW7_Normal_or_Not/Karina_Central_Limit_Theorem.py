
# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels
from statsmodels.stats.diagnostic import normal_ad

""" Making the distributions """
# Gaussian/Normal Function
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

# The size of each distribution
size = 200

# The number of distributions that we will average
ndist = 100000

# Create an array of zeros and then accumulate the values from each draw.
dist =  np.zeros(size)
pbar = tqdm(desc='Averaging distributions',unit='dists',total=ndist)
for i in range(ndist):
    dist += np.random.lognormal(0,1,size)
    pbar.update(1)
pbar.close()

# Now divide by the number of distributions to find the average values
dist /= np.float(ndist)

# Plot the resultant distribution
hist = plt.hist(dist,bins=100,normed=True,edgecolor='none')


# Overplot gaussian
hist = plt.hist(dist,bins=100,normed=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.plot(x,y,'r--')
xlim = plt.xlim(1.56,1.74)

""" Andersen-Darling Test """
ad, p = statsmodels.stats.diagnostic.normal_ad(dist)
print(p)
