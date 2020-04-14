'''
DataScience HW
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%config InlineBackend.figure_format = 'retina'
%matplotlib inline


data = pd.read_csv('/Users/jennifer/Jen_Data_Science/datasci2020/2_DataWrangling/ThacherWeather/WeatherLink_Data_2015.txt',sep='\t')


data = pd.read_csv('/Users/jennifer/Jen_Data_Science/datasci2020/2_DataWrangling/ThacherWeather/WeatherLink_Data_2015.txt',sep='\t',
                     usecols=[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],
                     skiprows=[0,1],header=0,names=['Date','Time','Heat Index',
                                                     'Temp Out','Wind Chill','Hi Temp',
                                                     'Low Temp','Hum Out','Dew Pt.','Wind Speed',
                                                     'Wind Hi','Wind Dir','Rain','Barometer',
                                                     'Temp In','Hum In','Archive'],
                    na_values='--')


temp = data['Temp Out'].values

temp.astype('float')

dtDF = data['Date']+' '+data['Time']

dt = pd.to_datetime(dtDF).values

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
fig,ax = plt.subplots()
plt.plot_date(dt,temp,'k-')
plt.title('Outside Temperature 2015')
fig.autofmt_xdate()
plt.ylabel('Temperature')

fig,ax = plt.subplots()
plt.plot_date(dt,temp,'k-')
import datetime
plt.xlim([datetime.date(2015, 1, 1), datetime.date(2015, 12, 31)])
fig.autofmt_xdate()
plt.ylabel('Temperature')

dtDFI = pd.DataFrame(temp,columns=['Temp'],index=dt)

dtDFI['2015-1':'2015-2'].values

fig,ax = plt.subplots()
plt.plot(dtDFI['2015-1-1':'2014-1-31'])
fig.autofmt_xdate()
plt.xlabel ('Date')
plt.ylabel ('Outside Temperature')
plt.title('2015 January Weather')

maxs = []
mins = []
for i in np.arange(31)+1:
    date = '2015-1-'+str(i)
    vals = dtDFI[date].values
    try:
        maxs = np.append(maxs,np.max(vals))
        mins = np.append(mins,np.min(vals))
    except:
        pass


maxhist = plt.hist(maxs,alpha=0.5)
minhist = plt.hist(mins,alpha=0.5)

maxhist = plt.hist(maxs,alpha=0.5,bins=np.arange(50)*3,label='Daily Highs')
minhist = plt.hist(mins,alpha=0.5,bins=np.arange(50)*3,label='Daily Lows')
plt.xlim(30,100)
plt.xlabel('Outside Temperature ($^\circ\!$F)')
plt.legend()
plt.title('Temperature Highs/Lows January 2015')

