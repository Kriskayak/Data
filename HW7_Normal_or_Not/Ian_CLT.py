# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#--------------------------------------------------
# js - Don't need this for standard console only jupyter notebook
#from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.stats.diagnostic import normal_ad
#--------------------------------------------------
# js - These following lines are only for jupyter notebooks
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#get_ipython().run_line_magic('matplotlib', 'inline')


# %%
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y


# %%
rand = np.random.normal(5,2,100)
print(rand)


# %%
size = 10000

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


# %%
hist = plt.hist(dist,bins=100,normed=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.plot(x,y,'r--')
xlim = plt.xlim(1.56,1.74)

# %%
ad2,pval = normal_ad(dist,axis=0)
print(pval)
# %%


'''
js comments
-----------
- You can get rid of the residual comments from the other assigment. 

- Comments could be more comprehensive

- Use density instead of normed for histogram call

- Results look good. Where is the threshold of number of averages before you
  are consistent with Normal?

8/10

'''

