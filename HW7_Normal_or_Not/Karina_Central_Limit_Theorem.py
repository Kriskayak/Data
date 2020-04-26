
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
#--------------------------------------------------
# js -  You've got some depreciated keywords. Also, you probably want to do some
#hist = plt.hist(dist,bins=100,normed=True,edgecolor='none')
# interactive plotting
plt.ion()
plt.figure(1)
plt.clf()
hist = plt.hist(dist,bins=30,density=True,edgecolor='none')


# Overplot gaussian
#--------------------------------------------------
# js - Again some depreciated kwargs
#hist = plt.hist(dist,bins=100,normed=True,edgecolor='none')
hist = plt.hist(dist,bins=30,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.title("Central Limit Theorem")
plt.xlabel("x values")
plt.ylabel("Frequency")
plt.plot(x,y,'r--')
xlim = plt.xlim(1.56,1.74)
#--------------------------------------------------
# js - Make some labels! :)

""" Andersen-Darling Test """
ad, p = statsmodels.stats.diagnostic.normal_ad(dist)
print(p)
