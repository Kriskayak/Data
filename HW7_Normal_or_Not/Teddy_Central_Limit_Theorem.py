import statsmodels.stats.diagnostic as test
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#--------------------------------------------------
# js - These lines are for jupyter only. 
#get_ipython().magic(u"config InlineBackend.figure_format = 'retina'")
#get_ipython().magic(u'matplotlib inline')


size = 10000

# The number of distributions that we will average
ndist = 100000

# Create an array of zeros and then accumulate the values from each draw.
dist =  np.zeros(size)
pbar = tqdm(desc='Averaging distributions',unit='dists',total=ndist)
for i in range(ndist):
    dist += np.random.rayleigh(5,size)
    pbar.update(1)
pbar.close()

# Now divide by the number of distributions to find the average values
dist /= np.float(ndist)

# Plot the resultant distribution
hist = plt.hist(dist,bins=100,normed=True,edgecolor='none')

a,b = test.normal_ad(dist,axis=0)
print("\n" + str(a) + " = ad a2 statistic")
print(str(b) + " = p value data comes from nd")

'''
js comments
-----------
- Could be a bit better commenting

- Plotting was a bit sparse. Use labels, use interactive plotting, 
  make use of the great visualization tools!

- Use density instead of normed... it's been depreciated

- Your distribution did not pass the AD test!

7/10
'''
