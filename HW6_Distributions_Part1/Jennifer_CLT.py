#!/usr/bin/env python
# coding: utf-8


#--------------------------------------------------
# js - You might want to use block commenting for
# this part. A block comment is started and ended
# with a triple quote. For example
'''
This
is
a 
block
comment
'''

# # The Central Limit Theorem
# In this exercise we will explore and demonstrate the Central Limit Theorem—a very important idea that arises in much of science and data analysis. The theorem states that the average of a series of numbers will always follow a Gaussian, or "normal," probability distribution regardless of the probability distribution that the data points were drawn from.
# 
# A Gaussian probability distribution (or density) function (PDF) is given by 
# $$ p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\frac{(x-x_0)^2}{2\sigma^2}\right] $$
# where $x_0$ is the mean of the distribution, and $\sigma$ is the standard deviation of the distribution.

# Let's first take a look at a gaussian function

#--------------------------------------------------
# js - Get rid of these comments...
# In[12]:


# Load in useful packages and make plots display inline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#--------------------------------------------------
# js - These commands are only relevant for jupyter
# notebooks
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#get_ipython().run_line_magic('matplotlib', 'inline')


# To make life easy, let's first make a function that creates a Gaussian curve

# In[13]:


def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y


# In[14]:


x,y = gauss(sig=5,x0=12)
plt.plot(x,y)
plt.title('Gaussian Curve')


# Go ahead and change the mean and standard deviation and see how the curve changes...
# 
# Now, what does it mean to say that we "draw" numbers from a normal distribution? It means that numbers are drawn randomly with probability equal to the value of a Gaussian function with a given mean and standard deviation. Luckily, there is a very nice numpy function that will do this for us.

# In[15]:


# This draws 100 samples from a normal distribution with a mean of "0" and a standard deviation of "1"
rand = np.random.normal(0,1,100)
print(rand)


# Hmm. Ok, it's just a bunch of values. But what are the frequency of the values?

# In[16]:


hist = plt.hist(rand,bins=20,density=True)


# Interesting! It (sort of) looks like a normal distribution. Why don't you try upping the number of samples drawn to 1000, or more! You can also increase the number of bins in your histogram. 
# 
# Next, let's overplot a Gaussian curve with the same mean and standard deviation.

# In[17]:


hist = plt.hist(np.random.normal(0,1,10000),bins=50, density=True,edgecolor='none')
x,y = gauss(sig=1,x0=0)
plt.plot(x,y,'r--')


# You can see that it follows the curve very well.
# 
# Now let's look at what it looks like to draw a bunch of samples from a different distribution. We'll use a lognormal form. Don't worry if we have not gone over what a lognormal PDF is yet, you can look <a href='https://en.wikipedia.org/wiki/Log-normal_distribution'>here</a> to get some quick info about it, or just take it as fact that it is an asymmetric distribution. 

# In[18]:


dist = np.random.chisquare(1,100000)
plt.hist(dist[dist<10],bins=100,density=True)
plt.axvline(x=np.sqrt(np.e),linestyle='-',color='red')
plt.axvline(x=np.mean(dist),linestyle='--',color='green')

print(np.sqrt(np.e),np.mean(dist))


# Ok, so, there is a well defined mean. But this sure looks different than a normal distribution.
# 
# Now on to the Central Limit Theorem (the point of this whole exercise!). The CLT states that if we average many lognormal distributions that the result should be Gaussian! 

# In[20]:

#--------------------------------------------------
# js - you don't need any of this!
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


# Hmmm. It looks like the peak of the resultant distribution has moved to the right some, but it sure doesn't look Gaussian!
# 
# Let's average many, many lognormal distributions and see what happens...

# In[22]:


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


# Holy moly! It sure looks Gaussian. But is it really?

# In[30]:


hist = plt.hist(dist,bins=100,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(dist),sig=np.std(dist))
plt.plot(x,y,'r--')
xlim = plt.xlim(0.98,1.02)


# ## Homework
# Demonstrate the CLT with a different distribution and save it in a python script or module. When complete you will push that code to the same directory as this notebook on the repo. Please call your script "YourName_CLT.py". There are many distributions you can choose from <a href="https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html">here</a>. Also, you may (are expected to) copy and edit code from this notebook to commplete this task.


'''
js comments
-----------
 - You should really clean up this code more. Text editing is a good skill that needs practice

 - Some of your commands are only relevant for jupyter notebooks

 - 8/10


'''
