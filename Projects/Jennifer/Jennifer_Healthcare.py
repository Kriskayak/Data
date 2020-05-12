"""
GDP per capita, PPP:  based on purchasing power parity (PPP)
1990 – 2017
2011 international dollars
Published by: World Bank – World Development Indicators

Healthcare Expenditure per Capita, PPP: sum of public and private health expenditures as a ratio of total population
1995 – 2014
Published by: World Bank – World Development Indicators
Source: World Health Organization Global Health Expenditure database
"Limitations and exceptions: Country data may differ in terms of definitions, data collection methods, population coverage and estimation methods used."

Total Population: based on Gapminder data, HYDE, and UN Population Division (2019) estimates
10000 BCE – 2019
Published by: Gapminder, HYDE (2016) and United Nations Population Division (2019)

What we want to investigate: Population vs. Health Care Expenditure / GDP per capita 

"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy import stats
from scipy.stats import norm
from scipy.stats import kstest
from pandas.plotting import register_matplotlib_converters
import statsmodels
from statsmodels.stats.diagnostic import normal_ad

#-----------
# js comment
# keep path separate. if you are running this from the same directory as the script
# a simple filename will suffice.
df= pd.read_csv("/Users/jennifer/Jen_Data_Science/datasci2020/Projects/Jennifer/healthcare-expenditure-vs-gdp.csv")

df.columns=['Country','Code','Year','GDP per Capita','Healthcare Expenditure','Population']

# Over here we see the columns names and their data types
df.dtypes

#We want to look at the year of 2004 for every country.
df = df.loc[df['Year'] == '2004']

df.isna()
df.isna().sum() 
df = df.dropna(axis = 0, how = 'any')  #drop rows with any column having np.nan values

len(df) #184 countries' data available for 2004

df['hG'] =(df['Healthcare Expenditure']/df['GDP per Capita'])*100

df = df[['Country','Population','hG']]

df.sort_values(by='hG') #The United States clearly stands out

#------------
# js comments
# best to do plotting interactively when exploring data.
plt.ion()
plt.figure()
plt.clf()
plt.hist(df['hG'], bins=20, density=True)
plt.xlabel('Health Care Expenditure / GDP per capita(%)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Health Care Expenditure / GDP per capita')

#looks somewhat Gaussian but both datasets look to be skewed to the left

#------------
# js comments
# buggy print statement
df.describe()
##Both the mean and standard deviation vary greatly between the two data sets.
#Mean: 41.9 vs. 42.5
#Std.:3.0 vs. 3.6
df.median()
##The median seem similar for the two data sets.
#41.8 vs. 42.5

#Normal?
#------------
# js comments
# This next line is buggy... you needed an hG key...
# Also this plotting did not work out for me...
"""it did for me..?"""
mean,std=norm.fit(df['hG'])
plt.hist(df['hG'], bins=np.arange(50), density=True)
# Take out the arguments of the xlim call...
""" 'xlim:(0,20) -> why not?"""
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean, std)
plt.plot(x, y)

##p-values Interpretation
ad, p = statsmodels.stats.diagnostic.normal_ad(df['hG'])
print(p) #p=1.70753e-06 
"""
Conclusion: As p is inarguably close to 0, I can reject hypothesis that the data comes from a normal distribution.
"""


'''
js comments
-----------
 - Save plot to PNG file?

 - I appreciate the block comment at the top of this script. Very useful!

 - Your plotting has several bugs. See above. Your approach looks sound, but 
   a bit is lacking in the execution. 

 - Conclusions?

15/20
'''

"""
HW11
Jennifer Yim:
'Produce a KDE of some subset of the data that you are working with and push a 
PNG version of a plot into your Projects/Name directory in the repo.''
"""
df= pd.read_csv("/Users/jennifer/Jen_Data_Science/datasci2020/Projects/Jennifer/healthcare-expenditure-vs-gdp.csv")
df.columns=['Country','Code','Year','GDP per Capita','Healthcare Expenditure','Population']
df = df.dropna(axis = 0, how = 'any')
df['hG'] =(df['Healthcare Expenditure']/df['GDP per Capita'])*100
d1 = df.loc[df['Year'] == '2004']
d2 = df.loc[df['Year'] == '2005']
d1 = d1['hG']
d2 = d2['hG']

from scipy import stats
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

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
           extent=[xmin, xmax, ymin, ymax])
ax.plot(d1, d2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_title('2004 vs. 2005 Healthcare expenditure/GDP per capita (%)')
ax.set_xlabel('2004 H.E./GDP (%)')
ax.set_ylabel('2005 H.E./GDP (%)')
plt.show()



