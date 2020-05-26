# Imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import statsmodels
from statsmodels.stats.diagnostic import normal_ad

# Reducing and cleaning up the data
data = pd.read_csv('co-emissions-per-capita-vs-gdp-per-capita-international-.csv',
                    usecols=[0,2,3,4,5],
                    skiprows=[0], header=0,
                    names=['Country', 'Year', 'CO2_emissions', 'GDP', 'Total_population'],
                    na_values='--')

data_2015 = data.loc[data['Year'] == '2015']
data_2015 = data_2015.dropna()
data_2015 = data_2015.reset_index()
del data_2015['index']
data_2015_simple = data_2015[['Country','CO2_emissions','GDP','Total_population']]
N = data_2015_simple.shape[0]

# Preliminary scatter plot of data - Dif colors for each point, size of point corresponds to population
np.random.seed(521)
colors = np.random.rand(N)
size = np.sqrt(np.sqrt(data_2015_simple['Total_population']))

x = list(data_2015_simple.GDP)
y = list(data_2015_simple.CO2_emissions)
plt.xlabel('GDP per Capita')
plt.ylabel('CO2 Emissions per Capita')
plt.title('CO2 Emissions Based on GDP')
plt.xscale('log')
#plt.annotate(list(data_2015_simple.Country), ((data_2015_simple.GDP), (data_2015_simple.CO2_emissions)))
plt.scatter(x, y, s=size, c=colors, alpha=0.5)
plt.savefig('Scatter_CO2_Emissions_vs_GDP')

# Preliminary fitting a line 
fit = np.polyfit(x, y, 1, full=False,cov=True )
fitparams = fit[0]
slope = fitparams[0]
intercept = fitparams[1]

cov = fit[1]
param_error = np.sqrt(np.diagonal(cov))
slope_error = param_error[0]
intercept_error = param_error[1]

xfit = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit = intercept + slope*xfit
plt.plot(xfit,yfit,'r--') # Looks pretty good ngl
plt.savefig('Preliminary_Line_Fit')

# Basix data values
std = np.std(data_2015_simple.CO2_emissions)
mean = np.mean(data_2015_simple.CO2_emissions)
print(std, mean)

# Seeing how closely related the data is to a Gaussian
def gauss(sig=1,x0=0):
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

# Histogram of CO2 Emissions
hist = plt.hist(data_2015_simple.CO2_emissions,bins=50,density=True)

x_gauss,y_gauss = gauss(sig=std,x0=mean)
plt.plot(x_gauss,y_gauss,'r--')
plt.savefig('Attempted_Gaussian_Fit')

ad, p = normal_ad(data_2015_simple.CO2_emissions) # A-D test
print(p) # Basically 0

# KDE Estimation
kde = gaussian_kde(data_2015_simple.CO2_emissions)
xvals = np.linspace(0, 50, 10000)

plt.plot(xvals,kde.pdf(xvals),'r--')
plt.xlim(0,50)
plt.hist(data_2015_simple.CO2_emissions,bins=np.arange(50),density=True)
plt.ylabel('Frequency',fontsize=12)
plt.xlabel('CO2 Emissions Per Capita',fontsize=12)
plt.title('KDE Plot')
plt.savefig('CO2_Emissions_KDE.png')

# Log Likelihood for Linear Fit
