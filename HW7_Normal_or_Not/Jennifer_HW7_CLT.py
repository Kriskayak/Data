'DataScience HW: Chisquared Curve'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels
from statsmodels.stats.diagnostic import normal_ad

def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

dist = np.random.chisquare(1,100000)
plt.hist(dist[dist<10],bins=100,density=True)
plt.axvline(x=np.sqrt(np.e),linestyle='-',color='red')
plt.axvline(x=np.mean(dist),linestyle='--',color='green')

print(np.sqrt(np.e),np.mean(dist))


size = 10000
plt.figure(1,figsize=(8,6))
dist1 = np.random.chisquare(1,100000)
dist2 = np.random.chisquare(1,100000)
dist3 = np.random.chisquare(1,100000)
dist = (dist1+dist2+dist3)/3.0
hist = plt.hist(dist1,bins=200,density=True,edgecolor='none',alpha=0.35,color='red',label='$d_1$')
hist = plt.hist(dist2,bins=200,density=True,edgecolor='none',alpha=0.35,color='blue',label='$d_2$')
hist = plt.hist(dist3,bins=200,density=True,edgecolor='none',alpha=0.35,color='yellow',label='$d_3$')
hist = plt.hist(dist,bins=200,density=True,edgecolor='none',alpha=0.35,color='green',label='$d_1+d_2+d_3$')
plt.xlim(0,10)
leg = plt.legend()


size = 10000
ndist = 100000
dist =  np.zeros(size)
pbar = tqdm(desc='Averaging distributions',unit='dists',total=ndist)
for i in range(ndist):
    dist += np.random.chisquare(1,size)
    pbar.update(1)
pbar.close()

dist /= np.float(ndist)

hist = plt.hist(dist,bins=100,density=True,edgecolor='none')


hist = plt.hist(dist,bins=100,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.plot(x,y,'r--')
xlim = plt.xlim(0.98,1.02)


ad, p = statsmodels.stats.diagnostic.normal_ad(dist)
print(p)
