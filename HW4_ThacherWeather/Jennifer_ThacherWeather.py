'''
DataScience HW (Again)
No Data for January? -> December
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#--------------------------------------------------
# js - best to use a path designation to both
# simplify and make flexible calls like this
path = '/Users/jennifer/Jen_Data_Science/datasci2020/2_DataWrangling/ThacherWeather/'
data = pd.read_csv('WeatherLink_Data_2015.txt',sep='\t')

data = pd.read_csv('WeatherLink_Data_2015.txt',sep='\t',
                     usecols=[0,1,2], skiprows=[0,1], header=0, names=['Date','Time','Temp Out'], na_values='--')


temp = data['Temp Out'].values
dtDF = data['Date']+' '+data['Time']
dt = pd.to_datetime(dtDF).values
dtDFI = pd.DataFrame(temp,columns=['Temp'],index=dt)

maxs = []
mins = []
for i in np.arange(31)+1:
    date = '2015-12-'+str(i)
    vals = dtDFI[date].values
    try:
        maxs = np.append(maxs,np.max(vals))
        mins = np.append(mins,np.min(vals))
    except:
        pass

maxs = maxs.astype('float')
mins = mins.astype('float')

#--------------------------------------------------
# js - get familiar with the interactive plotting
# features of matplotlib. They will help you out
# in the future. I added the functionality here
plt.ion()
plt.figure()
plt.clf()
maxhist = plt.hist(maxs,alpha=0.5,bins=np.arange(50)*3,label='Daily Highs')
minhist = plt.hist(mins,alpha=0.5,bins=np.arange(50)*3,label='Daily Lows')

plt.xlim(30,100)
plt.xlabel('Outside Temperature')
plt.legend()
plt.title('Temperature Highs/Lows December 2015')
#--------------------------------------------------
# js - y axis label?

#--------------------------------------------------
# js - Don't need this when you are plotting
# interactively (plot always "show"s).
#plt.show()

'''
js comments
-----------
 - Well done pairing down the code into the essential elements!

 - There is actually NO commenting. This will be something to 
   work on.

 - 13/15

'''
