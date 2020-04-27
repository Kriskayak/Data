import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------
# js - The data file lives in the same directory
# as this script, so no path designation is needed
path = '/home/karina/Desktop/'
#--------------------------------------------------
# js - Looks like you figured out the formatting
# in the next line. So, you don't need to keep this
#data = pd.read_csv('Soil_Microbial_Biomass_C_N_P_spatial.csv',encoding='latin1')


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

data = data.fillna(0)

'''
js comments
-----------
 - Well done, Karina!

 - 10/10

'''

