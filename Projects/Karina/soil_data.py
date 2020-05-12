import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels
from statsmodels.stats.diagnostic import normal_ad



#--------------------------------------------------
# js - I've suggested a bit of fomatting...
data = pd.read_csv('Soil_Microbial_Biomass_C_N_P_spatial.csv',encoding='latin1',
                   usecols=[0,1,2,3,4,5,8,9,10,11,12,13,14,15,16,17,18],
                    skiprows=[0], header=0,
                   names=['Country','Biome','Latitude','Longitude','Elevation',
                          'Vegetation','Soil_microbial_biomass_carbon',
                          'Soil_microbial_biomass_nitrogen','Soil_microbial_biomass_phosphorus',
                          'Soil_organic_carbon','Total_nitrogen','Total_organic_phosphorus',
                          'pH','Date','Upper_depth','Lower_depth','Depth'],
                   na_values='--')

#data = data.fillna(0)

categories = ['Soil_microbial_biomass_carbon', 'Soil_microbial_biomass_nitrogen','Soil_microbial_biomass_phosphorus','Total_nitrogen','Total_organic_phosphorus']
# Create a for loop to do this for all the categories
# Not sure of the most effective way to do that b/c each category has a different optimal range of values for the query command below and dif bins
# Maybe make a dictionary with category and range? and loop using that


microbial_carbon = data['Soil_microbial_biomass_carbon'] # Extract the category and get rid of na values
microbial_carbon = microbial_carbon.dropna()


plt.ion() # Setting up interactive plot
plt.figure()
plt.clf() 


hist = plt.hist(microbial_carbon, alpha=0.5,density=True,bins=np.arange(50)*6, label='Soil Microbial Carbon') # Make hist of data
plt.xlim(0,300)
plt.xlabel('Soil Microbial Carbon (mmol/kg)')
plt.ylabel('Frequency')
plt.title('Soil Microbial Carbon')
#------------
# js comment
# Don't use spaces in file names. You must specify tag (such as .png)
plt.savefig('Soil Microbial Carbon Plot')


microbial_carbon = microbial_carbon.to_frame() # Get rid of the outliers and get std/mean
microbial_carbon = microbial_carbon.query('0 <= Soil_microbial_biomass_carbon <= 300')
microbial_carbon = microbial_carbon.to_numpy()
std = np.std(microbial_carbon)
mean = np.mean(microbial_carbon)
print(std, mean)


def gauss(sig=1,x0=0): # Gaussian function
    x = np.linspace(x0-10*sig,x0+10*sig,1000)
    y = 1.0/(np.sqrt(2*np.pi)*sig)*np.exp(-(x-x0)**2/(2*sig**2))
    return x,y

x,y = gauss(sig=std,x0=mean) # Obviously not a Gaussian, I think it could be lognormal but I don't know how to fit that
plt.plot(x,y,'r--')

ad, p = statsmodels.stats.diagnostic.normal_ad(microbial_carbon) # A-D test anyways, it gets 0.0 lol
print(p)


'''
js comments
-----------
 - PNG file of your plot?

 - Couple buggy things (see above comments)

 - Otherwise, good work!

18/20


'''
