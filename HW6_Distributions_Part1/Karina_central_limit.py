import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Plotting Single Binomial Distribution
#--------------------------------------------------
# js - Use interactive plotting!
plt.ion()
plt.figure()
dist = np.random.binomial(200,0.5, size=40000)
plt.hist(dist,bins=40,density=True)
#plt.show()

# The size of each distribution
size = 40000

# The number of distributions that we will average
ndist = 100000

# Create an array of zeros and then accumulate the values from each draw.
dist =  np.zeros(size)
pbar = tqdm(desc='Averaging distributions',unit='dists',total=ndist)
for i in range(ndist):
    dist += np.random.binomial(200,0.5,size)
    pbar.update(1)
pbar.close()

# Now divide by the number of distributions to find the average values
dist /= np.float(ndist)

# Plot the resultant distribution
hist = plt.hist(dist,bins=100,density=True,edgecolor='none')

# Defining gaussian function
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

# Overplot gaussian on top of average distribution
hist = plt.hist(dist,bins=100,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.plot(x,y,'r--')

'''
js comments
-----------
- Change use of normed to density for histograms

- Nice commenting... I love it!

- Use interactive plotting to promote flow in a script. Also good
  in general.

- Your results look very nice!

10/10


