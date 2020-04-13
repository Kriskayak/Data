#!/usr/bin/env python
# coding: utf-8

# # Thacher Observatory Weather Data
# This is an introductory exploration into data wrangling, visualization, and analysis. This notebook will step you through some rudimentary steps leaving the bulk of the exploration up to you.
# 
# First order of duty is to import all the amazing code that other people have written for our benefit.

# In[1]:


# To clear output
# jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --inplace

# Pandas is a very useful python package (read: library, or module) for data acquisition and analysis
import pandas as pd

# Matplotlib is a large and powerful library for visualizing data
import matplotlib.pyplot as plt
# These commands force plots to be displayed in-line
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

# Numpy is another large and powerful library for dealing with math and arithmetic
import numpy as np


# In[2]:


# To look at the attributes and methods of any object use the "dir" function.
dir(pd)


# Now let us "load" a dataset. Loading a dataset is not always as straightforward as one might hope, and the process of getting data into machine readable format, reading the data, and otherwise preparing the data for further analysis is called "data wrangling."
# 
# In dealing with a dataset that you've never seen before, you might want to start with determining the kind of dataset you are working with. Here are some tips:
# 
# <ol><li>What kind of extension does the dataset have?</li>
# <li>Is it binary or ASCII (American Standard Code for Information Interchange)?</li>
# <li>If it is ASCII, is it a common form: CSV, TXT, XML?</li>
# <li>If not you may use your favorite editor to look at it.</li>
#     <ol><li>What delimiters are being used: tab, comma, pipe, etc.?</li></ol>
# <li>If it is binary, does the extension signify what binary format is being used?</li>
# <ol><li>Look at online documentation about binary file format.</li>
#      <li>Look for python package that will help you read that data.</li></ol>
# </ol>
#     
# To start out, let's look at a pretty well behaved dataset. It is the file called WS_data_2014.txt. You can open it with your favorite text editor: TextEdit (Mac), Emacs (cross platform), VI (oldie but goodie), nano (UNIX-like), pico (UNIX-like), etc. This file is simply a text file, so you can even open it from the Home page of you Jupyter session.
# 
# To read this dataset into python using pandas use the following command

# In[3]:


data = pd.read_csv('WS_data_2014.txt',sep='\t')


# In[4]:


# Let's take a look at what the function "read_table" returned into the variable 
# (is is actually a pandas DataFrame, which is very much like a python dictionary)
type(data)


# In[5]:


len(data)


# In[6]:


data


# In[7]:


# Knowing our data a little bit better, we could have directed the read_table function a little bit better
data = pd.read_csv('WS_data_2014.txt',sep='\t',
                     usecols=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                     skiprows=[0,1],header=0,names=['Date','Time','Heat Index',
                                                     'Temp Out','Wind Chill','Hi Temp',
                                                     'Low Temp','Hum Out','Dew Pt.','Wind Speed',
                                                     'Wind Hi','Wind Dir','Rain','Barometer',
                                                     'Temp In','Hum In','Archive'],
                    na_values='--')


# In[8]:


data


# Now it looks like we have all the data we want in a nice, neat DataFrame. How do we access the data?

# In[9]:


data.keys()


# In[11]:


# This will create a variable that is a pandas series
temp = data['Temp Out']


# In[12]:


type(temp)


# In[13]:


# Get just the values using this method
temp = data['Temp Out'].values


# In[18]:


type(temp)


# In[19]:


temp


# In[20]:


temp.astype('float')


# In[21]:


# You can do fun and useful appending of dataframe columns to make new dataframes
dtDF = data['Date']+' '+data['Time']


# In[22]:


dtDF


# In[23]:


# datetime objects are something that we will work with a lot in this class. More to come
dt = pd.to_datetime(dtDF).values


# In[24]:


dt


# In[25]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
fig,ax = plt.subplots()
plt.plot_date(dt,temp,'k-')
plt.title('Outside Temperature 2014')
fig.autofmt_xdate()
plt.ylabel('Temperature')


# In[26]:


# looks like all the data are there, but you can't see any detail
fig,ax = plt.subplots()
plt.plot_date(dt,temp,'k-')
import datetime
plt.xlim([datetime.date(2014, 1, 26), datetime.date(2014, 1, 31)])
fig.autofmt_xdate()
plt.ylabel('Temperature')


# How do we select out only one month of data? Make a new data frame indexed by the date, then it's easy!

# In[27]:


dtDFI = pd.DataFrame(temp,columns=['Temp'],index=dt)


# In[30]:


dtDFI['2014-1':'2014-2'].values


# In[32]:


fig,ax = plt.subplots()
plt.plot(dtDFI['2014-1-1':'2014-1-10'])
fig.autofmt_xdate()
plt.xlabel ('Date')
plt.ylabel ('Outside Temperature')
plt.title('Cool Plot')


# Now let's find all the high and low daily values in January 2014

# In[33]:


maxs = []
mins = []
for i in np.arange(31)+1:
    date = '2014-1-'+str(i)
    vals = dtDFI[date].values
    try:
        maxs = np.append(maxs,np.max(vals))
        mins = np.append(mins,np.min(vals))
    except:
        pass


# In[34]:


maxs


# In[35]:


mins


# In[36]:


maxhist = plt.hist(maxs,alpha=0.5)
minhist = plt.hist(mins,alpha=0.5)


# In[37]:


maxhist = plt.hist(maxs,alpha=0.5,bins=np.arange(50)*3,label='Daily Highs')
minhist = plt.hist(mins,alpha=0.5,bins=np.arange(50)*3,label='Daily Lows')
plt.xlim(30,100)
plt.xlabel('Outside Temperature ($^\circ\!$F)')
plt.legend()
plt.title('Temperature Highs/Lows January 2014')


# # Exercises
# <ol><li>Load the data from 2015 (WeatherLink_Data_2015.txt). Notice that this is data from another weather station on campus, and is formatted differently.</li>
# <li>Plot the equivalent plot of the high and low temperatures in January</li>
# <li>Determine if the mean high and mean low are significantly different from year to year (Hint: You might want to use a <a href='https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.stats.ttest_ind.html' target='_blank'>Welch's T-test</a>, though there are other techniques that we can explore).</li>
# <li>Export the code in this notebook as python code (File --> Download as --> Python (.py)). Then open that code in spyder (or equivalent IDE), clean it up and make it modular so that weather data from any year can be loaded and visualized</li>
# </ol>

# In[ ]:




