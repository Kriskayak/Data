
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from statsmodels.stats.diagnostic import normal_ad
from scipy import stats

import pandas as pd
pd.options.display.max_columns =None
pd.options.display.max_rows =40
filename = 'D:\Code\Thacher\datasci2020\Projects\Ian\country_patience.csv'
df = pd.read_csv(filename)


#for plotting the histogram
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y


df

df.columns = ['Country', 'Code','Year','Patience']

# plotting data as histogram
x = df.Patience
hist = plt.hist(x,bins=45,density=True,edgecolor='none')

# find mean and stdev
print(np.mean(x))
print(np.std(x))

# plotting another histogram with gaussian over it
hist = plt.hist(x,bins=50,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(x),sig=np.std(x))
plt.plot(x,y,'r--')
xlim = plt.xlim(-1.5,1.5)
plt.title('Patience of all countries')
plt.xlabel('Patience')
plt.ylabel('Occurance')
plt.savefig('Ian_ModernData.png',dpi=300)

# Anderson-Darling Test
ad2,pval = normal_ad(x,axis=0)
print(pval)
# this pval is tiny, which shows that this distribution is not normal
# I would assume this is true because each country has a different culture,
# so their value would not line up like a set of data from one source



#using the kde to do something, but it doesn't work

#separate the data somehow
#error can only tuple index with a multiindex
d1 = df.Patience[df['Country'] == 'United States','Mexico','Canada']
d2 = df.Patience[df['Year'] != 'United States','Mexico','Canada']

def measure():
    return d1+d2, d1-d2

xmin = d1.min()
xmax = d1.max()
ymin = d2.min()
ymax = d2.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([d1, d2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
           extent=[xmin, xmax, ymin, ymax])
ax.plot(d1, d2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()