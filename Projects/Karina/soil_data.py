# Imports 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import statsmodels
from statsmodels.stats.diagnostic import normal_ad
from scipy.optimize import minimize
import emcee


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

x = data_2015_simple.GDP
y = data_2015_simple.CO2_emissions
names = data_2015_simple.Country

fig,ax = plt.subplots()
plt.xlabel('GDP per Capita')
plt.ylabel('CO2 Emissions per Capita (tonnes)')
plt.title('CO2 Emissions Based on GDP')
plt.xscale('log')
sc = plt.scatter(x, y, s=size, c=colors, alpha=0.5,label='_nolegend_')

annot = ax.annotate("", xy=(0,0), xytext=(20,20),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)

def update_annot(ind):

    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
                           " ".join([names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_facecolor('g')
    annot.get_bbox_patch().set_alpha(0.4)

def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
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
plt.plot(xfit,yfit,'r--',label='Polyfit') # Looks pretty good ngl
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
#plt.savefig('Attempted_Gaussian_Fit')

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
#plt.savefig('CO2_Emissions_KDE.png')

# Log Likelihood for Linear Fit
def log_likelihood(theta, x, y, yerr=1):
    m, b = theta
    model = m * x + b
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

np.random.seed(42)
nll = lambda *args: -log_likelihood(*args)
initial = np.array([slope, intercept]) + 0.1 * np.random.randn(2)
soln = minimize(nll, initial, args=(x, y))
m_ml, b_ml = soln.x

xfit_nll = np.linspace(plt.xlim()[0],plt.xlim()[1],100)
yfit_nll = b_ml + m_ml*xfit
plt.plot(xfit_nll,yfit_nll,'g--',label='Log Likelihood')

# Played around with emcee, it didn't do much as expected
pos = soln.x + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(x, y))
sampler.run_mcmc(pos, 1000, progress=True);

fig, axes = plt.subplots(2, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["m", "b"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number")