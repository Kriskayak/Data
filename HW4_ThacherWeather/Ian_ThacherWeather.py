

# # Thacher Observatory Weather Data
# This is an introductory exploration into data wrangling, visualization, and analysis. This notebook will step you through some rudimentary steps leaving the bulk of the exploration up to you.

# Note: 2015 Data starts mid-April

#--------------------------------------------------
# js - There are a lot of extraneous comments in
# here, like the Input numbers from your notebook

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt


import numpy as np

# In[4]:

data = pd.read_csv('WS_data_2014.txt',sep='\t')

# In[8]:

data = pd.read_csv('WeatherLink_Data_2015.txt',sep='\t',
                     skiprows=[0,1],header=0,names=['Date','Time',
                                                     'Temp Out','Hi Temp',
                                                     'Low Temp','Hum Out','Dew Pt.','Wind Speed',
                                                     'Wind Dir','Wind Run','Hi Speed','Hi Dir','Wind Chill',
                                                     'Heat Index','THW Index','THSW Index','Bar','Rain',
                                                     'Rain Rate','Solar Rad','Solar Energy','Hi Rad','Solar Index',
                                                     'UV Dose','UV UV','Hi D-D','Heat D-D','Cool Temp','In Hum',
                                                     'In Dew','In Heat','In EMC','In Density','In Air ET','Samp',
                                                     'Wind TX','Wind Recept','ISS Int.'],
                    na_values='---')

# In[10]:

#--------------------------------------------------
# js - Do you really need this as you have specified
# all the header keys explicitly?
data.keys()

# In[13]:


# Get just the values using this method
temp = data['Temp Out'].values

#--------------------------------------------------
# js - These are already floats.
temp.astype('float')
# In[17]:

#--------------------------------------------------
# js - Remenant comment. Make this code your own!
# You can do fun and useful appending of dataframe columns to make new dataframes
dtDF = data['Date']+' '+data['Time']

# In[18]:
#--------------------------------------------------
# js - Ditto above...
# datetime objects are something that we will work with a lot in this class. More to come
dt = pd.to_datetime(dtDF).values


# In[21]:

#--------------------------------------------------
# js - This is a remnant as well and not needed.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
fig,ax = plt.subplots()
plt.plot_date(dt,temp,'k-')
plt.title('Outside Temperature 2015')
fig.autofmt_xdate()
plt.ylabel('Temperature')


# In[22]:

#--------------------------------------------------
# js - This is a remnant as well and not needed.
# looks like all the data are there, but you can't see any detail
fig,ax = plt.subplots()
plt.plot_date(dt,temp,'k-')
import datetime
plt.xlim([datetime.date(2015, 5, 26), datetime.date(2015, 5, 30)])
fig.autofmt_xdate()
plt.ylabel('Temperature')


# How do we select out only one month of data? Make a new data frame indexed by the date, then it's easy!

# In[23]:


dtDFI = pd.DataFrame(temp,columns=['Temp Out'],index=dt)


# In[24]:


dtDFI['2015-5':'2015-6'].values


# In[25]:


fig,ax = plt.subplots()
plt.plot(dtDFI['2015-5-1':'2015-5-10'])
fig.autofmt_xdate()


# Now let's find all the high and low daily values in January 2015

# In[26]:


maxs = []
mins = []
for i in np.arange(31)+1:
    date = '2015-5-'+str(i)
    vals = dtDFI[date].values
    try:
        maxs = np.append(maxs,np.max(vals))
        mins = np.append(mins,np.min(vals))
    except:
        pass


# In[29]:

#--------------------------------------------------
# js - This does not work as there are "NaN"s in
# the data arrays.
maxhist = plt.hist(maxs,alpha=0.5)
minhist = plt.hist(mins,alpha=0.5)


# In[30]:

#--------------------------------------------------
# js - this is well done!, but what about a y axis
# label?
maxhist = plt.hist(maxs,alpha=0.5,bins=np.arange(50)*3,label='Daily Highs')
minhist = plt.hist(mins,alpha=0.5,bins=np.arange(50)*3,label='Daily Lows')
plt.xlim(30,100)
plt.xlabel('Outside Temperature ($^\circ\!$F)')
plt.legend()
plt.title('Temperature Highs/Lows May 2015')


# %%
'''
js comments
-----------
 - Commenting is sparse and there is a paucity of useful comments

 - Could have looked closer at the code to understand what it is
   doing.

 - A lot of extraneous code that is not working toward the primary goal
   of the assignment

 - Bug in the plotting of the data due to NaNs

 - Final plot looks good!

 - 12/15

'''
