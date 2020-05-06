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

df= pd.read_csv("/Users/jennifer/Jen_Data_Science/datasci2020/Projects/Jennifer/healthcare-expenditure-vs-gdp.csv", sep=',')

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

df.sort_values(by='hG')

plt.hist(df['hG'], bins=20, density=True)
plt.xlabel('Health Care Expenditure / GDP per capita(%)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Health Care Expenditure / GDP per capita')

#looks somewhat Gaussian but both datasets look to be skewed to the left

print (df.describe())
##Both the mean and standard deviation vary greatly between the two data sets.
#Mean: 41.9 vs. 42.5
#Std.:3.0 vs. 3.6
print (df.median())
##The median seem similar for the two data sets.
#41.8 vs. 42.5

#Normal?
mean,std=norm.fit(df)
plt.hist(df, bins=np.arange(50), density=True)
xmin, xmax = plt.xlim(20,60)
x = np.linspace(xmin, xmax, 100)
y = norm.pdf(x, mean, std)
plt.plot(x, y)
plt.show()

##p-values Interpretation
ad, p = statsmodels.stats.diagnostic.normal_ad(df)
print(p)

