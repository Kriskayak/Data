'''
Data Science HW for 4/14:
Part 1 graphs the daily highs and lows for a given time period from the data.
Part 2 takes given time periods from two data sets and compares the mean highs or mean lows 
and states whether they are significantly different (p value greater than 0.05).
'''

# Imports for both parts #
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#--------------------------------------------------
# js - since you are only using the t-test from
# scipy, you probably should import it this way...
from scipy.stats import ttest_ind

#from scipy import stats
#import scipy

### Part 1: Graphing the highs and lows ###

# Look at the data set to see how to clean it up

#--------------------------------------------------
# js - Changing path to work locally add path
# variable to file name if needed
path = '/home/karina/Desktop/Data_Science_Spring_2020/datasci2020/2_DataWrangling/ThacherWeather/'
data = pd.read_csv('WeatherLink_Data_2015.txt',sep='\t')

# Clean up the needed columns, relabel data
data = pd.read_csv('WeatherLink_Data_2015.txt',sep='\t', usecols=[0,1,2], skiprows=[0,1],
                   header=0, names=['Date','Time','Temp Out'], na_values='--') 
#--------------------------------------------------
# js - Getting error
# Get the temperature values as a nparray
temp = data['Temp Out'].values

# Create df of date and time, index temperature nparray with datetime conversion
dtDF = data['Date']+' '+data['Time']
dt = pd.to_datetime(dtDF).values
dtDFI = pd.DataFrame(temp,columns=['Temp'],index=dt)

# Get the maxs and mins for each day with viable data
maxs = []
mins = []
for i in np.arange(31)+1:
    date = '2015-12-'+str(i) # Choose time period
    vals = dtDFI[date].values
    try:
        maxs = np.append(maxs,np.max(vals))
        mins = np.append(mins,np.min(vals))
    except:
        pass

# Make sure the values are float
maxs = maxs.astype('float')
mins = mins.astype('float')

# Plot histogram
#--------------------------------------------------
# js - best to get used to the interactive plot
# feature. This way you don't usurp your prompt
# every time you make a plot.
plt.ion()
plt.figure()
plt.clf()
maxhist = plt.hist(maxs,alpha=0.5,bins=np.arange(50)*3,label='Daily Highs')
minhist = plt.hist(mins,alpha=0.5,bins=np.arange(50)*3,label='Daily Lows')
plt.xlim(30,100)
plt.xlabel('Outside Temperature ($^\circ\!$F)')
#--------------------------------------------------
# js - y axis label?
plt.legend()
plt.title('Temperature Highs/Lows January 2014')
#plt.show()




#--------------------------------------------------
# js - create more of a break here for flow




### Part 2: Taking two data sets and comparing to see if the temps are significantly different ###
# Note, I reload the same data as in Part 1 just for the sake of clarity, it can just be reused, might have to rename some stuff tho

# Load the two data sets, relabeling and cleaning up the dataframes
data1 = pd.read_csv('WeatherLink_Data_2015.txt',
                  sep='\t', usecols=[0,1,2], skiprows=[0,1], header=0, names=['Date','Time','Temp Out'], na_values='--')

data2 = pd.read_csv('WS_data_2014.txt',sep='\t',
                        usecols=[0,2,3], skiprows=[0,1], header=0, names=['Date', 'Time', 'Temp Out'], na_values = '--')

# Pull out the temperatures
temp1 = data1['Temp Out'].values
temp2 = data2['Temp Out'].values

#--------------------------------------------------
# js - the next two tasks could easily be
# compressed into a for loop or modular functions


# Index the temperatures with datetimes
dtDF1 = data1['Date']+' '+data1['Time']
dt1 = pd.to_datetime(dtDF1).values
dtDFI1 = pd.DataFrame(temp1,columns=['Temp'],index=dt1)

dtDF2 = data2['Date']+' '+data2['Time']
dt2 = pd.to_datetime(dtDF2).values
dtDFI2 = pd.DataFrame(temp2,columns=['Temp'],index=dt2)

# Get the maxes and mins for both temp ranges
maxs1 = []
mins1 = []
for i in np.arange(31)+1:
    date = '2015-12-'+str(i) # Choose date range
    vals = dtDFI1[date].values
    try:
        maxs1 = np.append(maxs1,np.max(vals))
        mins1 = np.append(mins1,np.min(vals))
    except:
        pass

maxs2 = []
mins2 = []
for i in np.arange(31)+1:
    date = '2014-1-'+str(i) # Choose date range
    vals = dtDFI2[date].values
    try:
        maxs2 = np.append(maxs2,np.max(vals))
        mins2 = np.append(mins2,np.min(vals))
    except:
        pass

# Make sure they're all floats
maxs1 = maxs1.astype('float')
mins1 = mins1.astype('float')

maxs2 = maxs2.astype('float')
mins2 = mins2.astype('float')

# Compare the maxes
statistic_max, pvalue_max = ttest_ind(maxs1, maxs2)

#--------------------------------------------------
# js - I think you got this logic backwards. If the
# p value is small it means the values ARE
# significantly different. Also, your variable
# "pvalue" is not defined. I fixed... 

if pvalue_max > 0.05:
    print("The mean high is not significantly different between years")
else:
    print("The mean highs are significantly different")

# Compare the mins
statistic_min, pvalue_min = ttest_ind(mins1, mins2)

if pvalue_min > 0.05:
    print("The mean low is not significantly different between years")
else:
    print("The mean lows are significantly different")


#--------------------------------------------------
# js - one logical thing to do here would be to
# visualize the two datasets


'''
--------------------------------------------------
js comments
-----------
 - Well commented!

 - Good variable naming

 - Aside from the flipped logic and bug with your p-value output
   this was a solid piece of code. 

 - 15/15

'''
