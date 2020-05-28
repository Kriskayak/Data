
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
hist = plt.hist(x,bins=45,density=True,edgecolor='none')
x,y = gauss(x0=np.mean(x),sig=np.std(x))
plt.plot(x,y,'r--')
xlim = plt.xlim(-1.5,1.5)
plt.title('Patience of all countries')
plt.xlabel('Patience')
plt.ylabel('Occurance')
plt.show()
#plt.savefig('Ian_ModernData.png',dpi=300)

# Anderson-Darling Test
ad2,pval = normal_ad(x,axis=0)
print(pval)
# this pval is tiny, which shows that this distribution is not normal
# I would assume this is true because each country has a different culture,
# so their value would not line up like a set of data from one source


from scipy import stats
# using the gaussian_kde to smooth

kde = stats.gaussian_kde(x)
x_plot = np.linspace(np.min(x), np.max(x), 1000)
plt.plot(x_plot,kde.pdf(x_plot))
#only plots a horizontal line, but histogram has density=True ??
#plt.show()



plt.clf()
# bringing in the second set of data for line fitting
# have to test out different sets for correlation
filename = 'D:\Code\Thacher\datasci2020\Projects\Ian\country_risk.csv'
df2 = pd.read_csv(filename)
df2.columns = ['Country', 'Code','Year','Risk']
# none of the different sets line up well, but risk fits the best
# it probably doesn't help that the measurements are in an unknown scale/context

# first plot attempt
patience = df.Patience
risk = df2.Risk
riskError = np.std(risk)

# plot up data with errorbars
plt.errorbar(patience,risk,yerr=riskError,fmt='o',color='black')
plt.xlabel('patience')
plt.ylabel('risk-taking')
#plt.show()

# line fitting
riskWeights = 1.0/(riskError**2)

# Fit a line to the data:
# can't figure out weights, wants 1-d array for w but only have stdev
fit = np.polyfit(patience,risk,1,full=False,cov=True)

# Fit parameters are the first element in the returned "tuple"
fitparams = fit[0]
slope = fitparams[0]
intercept = fitparams[1]

print('The best estimate of the slope is %.3f as compared to the "real" slope of ??' % slope)
print('The best estimate of the intercept %.3f as compared to the "real" intercept of ??' % intercept)

# getting covariance matrix
cov = fit[1]
print(cov)
first_cov = cov[0][0]
second_cov = cov[1][1]
print('Covariance of first parameter: %.3f' % first_cov)
print('Second parameter: %.3f' % second_cov)

# getting errors out of covariance
param_error = np.sqrt(np.diagonal(cov))
slope_error = param_error[0]
intercept_error = param_error[1]

print('The slope is %.3f +/- %0.3f' %(slope,slope_error))
print('The intercept is %.3f +/- %0.3f' %(intercept,intercept_error))

# plotting the line of fit
xfit = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit = intercept + slope*xfit
plt.plot(xfit,yfit,'r--')
plt.show()
# plot looks much cleaner without error bars, but looks closer to the fit line with them


# x-shifting
patience2 = patience - np.median(patience)

# new fit and covariance
fit = np.polyfit(patience,risk,1,full=False,cov=True)


# getting covariance matrix
cov = fit[1]
print(cov)
first_cov = cov[0][0]
second_cov = cov[1][1]
print('Covariance of first parameter: %.3f' % first_cov)
print('Second parameter: %.3f' % second_cov)

# my covariance values are the exact same, I suspect it has to due with the lack
# of weighting the points earlier when I couldn't figure it out


'''
log-likelihood function
'''
# can't find anything useful on google
# trying to adapt the chi-squared function from slides
# need to figure out summation and different variables

def chisqr(xbar,sigma):
    finalsum = 0
     
    for x in patience:
        y = (x - xbar)**2
        z = y/(sigma**2)
        finalsum += z
    
    return(finalsum)

print(chisqr(np.mean(patience),np.std(patience)))
#DOF = 74, expected chi2 value is 46

# table says that probability of 75.999 is about 25%
# therefore we can say this data is not gaussian