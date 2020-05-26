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
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv("/Users/jennifer/Jen_Data_Science/datasci2020/Projects/Jennifer/healthcare-expenditure-vs-gdp.csv")
df.columns=['Country','Code','Year','GDP per Capita','Healthcare Expenditure','Population']
df = df.dropna(axis = 0, how = 'any')
df['hG'] =(df['Healthcare Expenditure']/df['GDP per Capita'])*100
df = df.loc[df['Year'] == '2004']

plt.ion() #opens interactive plot
plt.figure()
plt.clf()
kde = gaussian_kde(df['hG'])
xvals = np.linspace(np.min(df['hG']),np.max(df['hG']))
plt.hist(df['hG'],bins=20,density=True)
plt.plot(xvals,kde.pdf(xvals),'r--')
plt.title('2004 Healthcare expenditure/GDP per capita (%)')
plt.xlabel('2004 H.E./GDP (%)')
plt.ylabel('Frequency')
plt.show()

"""
HW12
Fitting a Line 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import erf, erfinv
import statsmodels
from statsmodels import robust

df = pd.read_csv("/Users/jennifer/Jen_Data_Science/datasci2020/Projects/Jennifer/healthcare-expenditure-vs-gdp.csv")
df.columns=['Country','Code','Year','GDP per Capita','Healthcare Expenditure','Population']
df = df.dropna(axis = 0, how = 'any')
df = df.loc[df['Year'] == '2004']
xvals = df['GDP per Capita']
yvals = df['Healthcare Expenditure']
yerror = np.std(df['Healthcare Expenditure']) # how do you find error bars for each datapoint?

# Create a plot with the two variables & the data points' error bar (1 sigma?: over-calculated): GDP per capita and Healthcare Expenditure 
plt.ion() 
plt.figure()
plt.clf()
plt.errorbar(xvals,yvals,yerr=yerror,fmt='o',color='black')
plt.xlabel('GDP per Capita')
plt.ylabel('Healthcare Expenditure')

fit = np.polyfit(xvals,yvals,1,full=False,cov=True)
fitparams = fit[0]
slope = fitparams[0]
intercept = fitparams[1]

#Covariance, parameter, slope, and intercept errors
cov = fit[1]
param_error = np.sqrt(np.diagonal(cov))
slope_error = param_error[0]
intercept_error = param_error[1]

#Plot with the Line of Best Fit
plt.clf()
plt.errorbar(xvals,yvals,yerr=yerror,fmt='o',color='black')
plt.xlabel('GDP per Capita')
plt.ylabel('Healthcare Expenditure')
xfit = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit = intercept + slope*xfit
plt.plot(xfit,yfit,'r--')

#Shifting data (I don't fully understand what I am supposed to do for this part)
median = np.median(df['GDP per Capita'])
newx  = xvals - np.median(df['GDP per Capita'])
fit = np.polyfit(newx,yvals,1,full=False,cov=True)
fitparams = fit[0]
slope = fitparams[0]
intercept = fitparams[1]

plt.clf()
plt.errorbar(newx,yvals,yerr=yerror,fmt='o',color='black')
plt.xlabel('GDP per Capita')
plt.ylabel('Healthcare Expenditure')
xfit = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit = intercept + slope*xfit
plt.plot(xfit,yfit,'r--')

"""
New subset of data that I'm interested in
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats
from scipy.stats import norm

f = open("/Users/jennifer/Jen_Data_Science/datasci2020/Projects/Jennifer/healthcare-expenditure-vs-gdp.csv")
	
data = []
for i in f.readlines()[1:]:
	if i.split(",")[3] != '' and i.split(",")[4] != '':
		data.append(i.split(","))

data1 = [] # 1~1000
data2 = [] # 1001~10000
data3 = [] # 10001~

for i in data:
	if float(i[3]) < 1000:
		data1.append([float(i[3]), float(i[4]) / float(i[3])])
	elif float(i[3]) < 10000:
		data2.append([float(i[3]), float(i[4]) / float(i[3])])
	else:
		data3.append([float(i[3]), float(i[4]) / float(i[3])])
        
len(data1) #172
len(data2) #2309
len(data3) #2044

xValues = []
yValues = []

for i in data1:
	xValues.append(i[0])
	yValues.append(i[1])
	

plt.scatter(xValues, yValues)
plt.xlabel("GDP per capita (int.-$)")
plt.ylabel("Healthcare Expenditure per capita / GDP per capita (int.-$) ")
plt.show() #Hmm I can't really see a trend


for i in np.arange(0, 0.25, 0.01):
	xValues.append(i)
	cnt = 0
	for j in data3:
		if j[1] >= i and j[1] <= i+0.01:
			cnt = cnt + 1
	yValues.append(cnt)

"""
Likelihood Functions:
    "You will need to choose a model and construct a log-likelihood function 
    for your data given your model. You will add this function in your script 
    in the Projects directory of our repo."
"""

def logLikelihood(x, mean):
	return norm(loc=mean).logpdf(x)

L1 = logLikelihood(data1, np.mean(data1)) #-> array?
np.amax(L1) #-92.2979
#IDK what I should do with this... hm











