#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 19:03:59 2020

@author: jennifer
"""
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

What we want to investigate: Health Care Expenditure vs. GDP
"""
"""
DATA SUPER MESSY RIGHT NOW SINCE I WAS WORKING ON IT UNTIL THE LAST MOMENT :(
NEEDS SOME CLEANING UP + CLARIFYING
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.stats import norm
from scipy.stats import gaussian_kde
from tqdm import tqdm
from scipy.stats import kstest
from pandas.plotting import register_matplotlib_converters
import statsmodels
from statsmodels.stats.diagnostic import normal_ad
from scipy.special import erf, erfinv
from statsmodels import robust
from scipy.optimize import minimize, rosen, rosen_der
from scipy.stats import chisquare

"""
"""
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
        
xValues = []
yValues = []

for i in data1:
	xValues.append(i[0])
	yValues.append(i[1])

for i in np.arange(0, 0.25, 0.01): # repeat for data1, data2, data3 -> plots
	xValues.append(i)
	cnt = 0
	for j in data3:
		if j[1] >= i and j[1] <= i+0.01:
			cnt = cnt + 1
	yValues.append(cnt)

plt.ion()
plt.figure()
plt.plot(xValues, yValues)
plt.title("gdp10001~")
plt.xlabel('Healthcare expenditure/GDP')
plt.ylabel('Frequency')
plt.show()

"""
"""
from scipy.optimize import minimize

def neg_log_lhood(s, x):
    # zero mean
    # add the "-" sign because we want to maximize but are using the minimize method
    return -(-0.5 * len(x) * np.log(2*np.pi*s**2) - 1./(2.*s**2)*sum([xx**2 for xx in x]))

def grad(s, x):
    # perform a central difference numerical derivative with h = 1e-8
    return (neg_log_lhood(s+1e-8, x) - neg_log_lhood(s-1e-8, x)) / (2e-8) 

# scipy's minimize function returns a solution struct, which contains the solution (if one was found)
# and a message (and other things, check the docs)

sigma_guess = 0.1
samples = []

for i in data1:
	samples.append(float(i[1]))

samples = np.array(samples)

sol = minimize(neg_log_lhood, sigma_guess, args=(samples), jac=grad, method='bfgs')

print(sol.message)

if sol.success:
    print('mean_final = {}'.format(sol.x[0]))
else:
    print('No solution found')
"""
data1 : mean_final = 0.0546841781771
data2 : mean_final = 0.0546088178282
data3 : mean_final = 0.0645437296197
"""
f_obs = []
f_exp = []

mean = data3[int(len(data3)/2)][1]

for i in range(0, len(yValues)):
	yValues[i] = yValues[i] / float(len(data3))

for i in np.arange(0, 0.25, 0.01):
	f_obs.append(yValues[int(i/0.01)])
	f_exp.append(norm.pdf(i, loc=mean, scale=0.0645437296197))

from scipy.stats import chisquare

print (f_obs)
print (f_exp)

chisquare(f_obs, f_exp)
"""
data1 = Power_divergenceResult(statistic=86.88991259430887, pvalue=4.686796196986366e-09)
data2 = Power_divergenceResult(statistic=86.74714792926366, pvalue=4.945908679510005e-09)
data3 = Power_divergenceResult(statistic=70.6190518718353, pvalue=1.7612409032468476e-06)
"""
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

samples = []

for i in data1:
	samples.append(float(i[1]))

samples2 = []

for i in data2:
	samples2.append(float(i[1]))

samples3 = []

for i in data3:
	samples3.append(float(i[1]))

from scipy.stats import gaussian_kde
import numpy as np

def estimate_maxima(data):

      kde = gaussian_kde(data)

      no_dp = 172

      dp = np.linspace(0, 10, no_dp)

      probs = kde.evaluate(dp)

      maxima_index = probs.argmax()

      maxima = dp[maxima_index]

      return maxima
  
estimate_maxima(samples)
#data1 = 0.05847953216374269
#data2 = 0.05847953216374269
#data3 = 0.05847953216374269

"""
With Sigma Rejection
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.special import erf, erfinv
import statsmodels
from statsmodels import robust
from scipy.optimize import minimize

def neg_log_lhood(s, x):
    # zero mean
    # add the "-" sign because we want to maximize but are using the minimize method
    return -(-0.5 * len(x) * np.log(2*np.pi*s**2) - 1./(2.*s**2)*sum([xx**2 for xx in x]))

def grad(s, x):
    # perform a central difference numerical derivative with h = 1e-8
    return (neg_log_lhood(s+1e-8, x) - neg_log_lhood(s-1e-8, x)) / (2e-8) 

# scipy's minimize function returns a solution struct, which contains the solution (if one was found)
# and a message (and other things, check the docs)

sigma_guess = 0.1
samples = []

for i in data1:
	samples.append(float(i[1]))
    
# Median and MAD 
median = np.median(samples)
mad = statsmodels.robust.scale.mad(samples, center=median)

df_s = erfinv(0.99)*np.sqrt(2)   # 1 = 100 * (1 - erf(x/np.sqrt(2)))
exp_high = median + (mad * df_s)
exp_low = median - (mad * df_s)

reg_s = erfinv(1 - (0.05/100))*np.sqrt(2) # 0.05 = 100 * (1 - erf(x/np.sqrt(2)))
sbound_lower = median - (mad * reg_s)
sbound_upper = median + (mad * reg_s)

filtered_data = [] # Data that is left after sigma rejection

for x in samples:
  if sbound_lower  < x < sbound_upper:
    filtered_data.append(x)

mean = np.mean(samples)
std = np.std(samples)
mean_new = np.mean(filtered_data)
std_new = np.std(filtered_data)

def logLikelihood(x, mean_new, std_new):
	return norm(loc=mean_new, scale=std_new).logpdf(x)

mean = data1[int(len(data1)/2)][1]

maxVal = 0.0
maxScale = 0

for scaleVal in np.arange(0.001, 0.2, 0.001):

	mle = 0.0
	for i in data1:
		mle = mle + logLikelihood(float(i[1]), mean_new, scaleVal) 

	if maxVal < mle:
		maxVal = mle
		maxScale = scaleVal

	print (scaleVal, mle)


print ("maxScale is : ", maxScale)

"""
data1 maxscale = 0.024
data2 maxscale = 0.024
data3 maxsxale = 0.039
"""
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

samples = []

for i in data1:
	samples.append(float(i[1]))
    
plt.ion() #opens interactive plot
plt.figure()
plt.clf()
kde = gaussian_kde(samples)
xvals = np.linspace(np.min(samples),np.max(samples))
plt.hist(samples,bins=20,density=True,label='data')
plt.plot(xvals,kde.pdf(xvals),'r--',label='best fit')
plt.title('GDP1000~ Healthcare expenditure/GDP per capita')
plt.xlabel('H.E./GDP')
plt.ylabel('Frequency')
plt.legend()
plt.show()
"""
"""
import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

def order(elem):
	return elem[0]

data1.sort(key=order)	
data2.sort(key=order)	
data3.sort(key=order)	

samples1 = []

for i in data1:
	samples1.append(float(i[1]))

samples2 = []

for i in data2:
	samples2.append(float(i[1]))
    
samples3 = []

for i in data3:
	samples3.append(float(i[1]))

# Median and MAD 
median1 = np.median(samples1)
mad1 = statsmodels.robust.scale.mad(samples1, center=median1)
median2 = np.median(samples2)
mad2 = statsmodels.robust.scale.mad(samples2, center=median2)
median3 = np.median(samples3)
mad3 = statsmodels.robust.scale.mad(samples3, center=median3)

xValues = []
yValues = []

mean = np.mean(samples3)
std = np.std(samples3)


for i in np.arange(0, 0.25, 0.01):
	xValues.append(i)
	cnt = 0
	for j in data3:
		if j[1] >= i and j[1] <= i+0.01:
			cnt = cnt + 1
	yValues.append(cnt)


x = xValues
y = yValues

n = len(x)                        
mean = mean            
sigma = std  

def gaus(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])

plt.plot(x,y,'b+:',label='data')
plt.plot(x,gaus(x,*popt),'ro:',label='fit')
plt.legend()
plt.title('GDP10001~ H.E./GDP')
plt.xlabel('H.E./GDP')
plt.ylabel('Frequency')
plt.show()
