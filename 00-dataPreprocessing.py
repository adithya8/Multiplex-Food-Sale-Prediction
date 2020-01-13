
import os
from collections import Counter
from six.moves import xrange
from pprint import pprint
from math import sqrt
import pickle as pkl

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import random as rn

from sklearn.utils import check_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import tensorflow as tf
from keras import losses
from keras import callbacks
from keras import regularizers
from keras import optimizers
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.layers import Dense,Dropout,BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization as BN
from keras.models import Sequential


np.random.seed(1)
rn.seed(1234)


correctdf1=pd.read_csv("../New Data/notNiteeshData.csv")
correctdf1.drop("holiday",axis=1,inplace=True)
correctdf1.head()

hol = pd.read_csv("../New Data/holiday.csv",index_col=0)
hol['Date'] = pd.to_datetime(hol['Date'])
hol = hol[hol['Date'].isin(correctdf1['Date'])]
hol.set_index('Date',inplace=True)


print ("Distribution of Holidays: ",Counter(hol['holiday']))

print ("Printing columns in the Dataset along with the column number....")
for i in range(correctdf1.shape[1]):
    print(i ,":", correctdf1.columns[i])

'''
0-24 columns contained:
['Date', 'dayOfWeek', 'lanuage 1 occupancy', 'language 2 occupancy', 'language 3 occupancy', 'other language occupancy', 
'language 1 allotted occupancy', 'language 2 allotted occupancy', 'language 3 allotted occupancy', 'other language allotted occupancy',
Daily Number of transactions for 14 food items(14 cols)]

80-94 columns contained
[Daily sale tally for 14 food items(14 cols)]
'''

l1=[range(0,24), range(80,94)]
list1=[item for sublist in l1 for item in sublist]

tempdf=pd.DataFrame()
tempdf=correctdf1[correctdf1.columns[list1]]
tempdf.Date = pd.to_datetime(tempdf.Date)
data_npm = tempdf

'''
Generating the food/occupancy attribute (combined Language occupancy).
'''
for i in data_npm.columns[24:38]:
    data_npm["food per "+i] = ( (data_npm[i] / (data_npm["language 1"] + data_npm["language 2"] + data_npm["language 3"] + data_npm["others"])) )                          


'''
Generating language wise food per occupant.
'''

for i in data_npm.columns[24:38]:
        data_npm["food per language 1 for "+i] =  (data_npm[i] / data_npm["language 1"])                        

for i in data_npm.columns[24:38]:
        data_npm["food per language 2 for "+i] =  (data_npm[i] / data_npm["language 2"])  

for i in data_npm.columns[24:38]:
        data_npm["food per language 3 for "+i] =  (data_npm[i] / data_npm["language 3"])                        

for i in data_npm.columns[24:38]:
        data_npm["food per Others for "+i] =  (data_npm[i] / data_npm["others"])                        

#Get the indices where 'inf' is present.
'''
When the language occupancy is 0, the food per occ pertaining to that language is 0.
'''
data_npm.fillna(0,inplace=True,axis=0)
data_npm.replace([np.inf],[0],inplace=True,axis=0)


'''
Dropping days '05-12-2016' and '23-01-2017' from the data.
It was decided to do so, based on the analysis of outlier graphs of the food sales.
The indices would be continuous. No need to reset the indices.
'''

t = []
index1 = tempdf[tempdf["Date"] == pd.to_datetime("2016-12-05")].index[0]

index2 = tempdf[tempdf["Date"] == pd.to_datetime("2017-01-23")].index[0]

t.append(index1)
t.append(index2)

print(t)

data_npm.drop(t,axis=0,inplace=True)
data_npm.reset_index(drop=True,inplace=True)

dropper = list(data_npm[data_npm['food1'] == 0].index)

'''
Removing all the days when the food history is zero. (for 12-3PM Slot)
'''
print (data_npm.shape)
data_npm.drop(dropper,axis=0,inplace=True)
data_npm.reset_index(drop=True,inplace=True)
print (data_npm.shape)


'''
Function that returns the Moving Average from both the 12PM data as well as the 3PM data.
Arguments : 
(1) Number of days for MA.
(2) Current date.

Returns : 
A list of values for all the food items.
'''

def moving_average(window,currentdate,start=38,end=52):
    hybrid = pd.DataFrame()
    #MOVING AVERAGE OF TODAY AS WELL, ONLY FOR 12-12 MODEL.
    hybrid = data_npm[data_npm['Date']<=currentdate].iloc[-(window):]
    return list(np.mean(hybrid.iloc[:,range(start,end)]))

def moving_maxmin(window,currentdate,tp):
    hybrid = pd.DataFrame()
    t = pd.DataFrame()
    hybrid = data_npm[data_npm['Date']<currentdate].iloc[-(window):]
    if(tp == 1): return list(np.amax(hybrid.iloc[:,range(24,38)]))
    else: return list(np.amin(hybrid.iloc[:,range(24,38)]))

def moving_sd(window,currentdate):
    hybrid = pd.DataFrame()
    t = pd.DataFrame()
    hybrid = data_npm[data_npm['Date']<currentdate].iloc[-(window):]
    return list(np.std(hybrid.iloc[:,range(24,38)]))

newhol = hol.reset_index().copy()
newhol['Date'] = pd.to_datetime(newhol['Date'])
newhol = newhol[newhol['Date'].isin(data_npm['Date'])]
newhol.drop(['Date'],axis=1,inplace=True)
newhol.reset_index(drop=True,inplace=True)
print newhol.shape
print data_npm.shape
data_npm = pd.concat([data_npm,newhol],axis=1)
print data_npm.shape 


counter = 0
inputlist = []
outputlist = []
templist = []

'''
The loop starts from 11, because our target day is day 'i'.
'''

for i in range(11,data_npm.shape[0]):

    currentdate = data_npm.iloc[i-1][0]

    current_hol = data_npm.iloc[i][-1]
    currentdow  = data_npm.iloc[i-1][1]
    '''
    (1) 7 initial columns.
    (2) 14 columns for Moving Average of food / occ.
    '''
    templist = [0]*484
    oplist = [0]*14

    templist[0] = data_npm.iloc[i-1][0]
    templist[1] = data_npm.iloc[i][1]
    
    templist[2:6] = data_npm.iloc[0,2:6]
    
    #Moving Averages of food per occ, for each food item.
    

    templist[6:20] = moving_average(2,currentdate)

    templist[20:34] = moving_average(3,currentdate)

    templist[34:48] = moving_average(7,currentdate)

    templist[48:62] = moving_average(10,currentdate)
    
    '''
    Getting the history of the previous 7 days.
    '''
    #The history will also be obtained from the 12-12 CSV file.
    newdf = data_npm[data_npm['Date'] < currentdate].iloc[-7:]
    
    templist[62:76] = moving_maxmin(10,currentdate,1)
    templist[76:90] = moving_maxmin(10,currentdate,0)
    templist[90:104] = moving_sd(10,currentdate)
    
    skipper=0
    skippercount=0
    while skippercount < 6:
        #History starting from Day - 6.......day - 1.
        templist[104+skipper:118+skipper] = newdf[newdf.columns[24:38]].iloc[skippercount+1]
        skipper += 14
        if (skippercount == 5):
            #Include current day's food sales history. (12PM-3PM).
            templist[104+skipper:118+skipper] = data_npm[data_npm['Date']==currentdate][data_npm.columns[24:38]].iloc[0]
        skippercount+=1
        
    templist[202:216] = moving_average(2,currentdate,52,66)
    templist[216:230] = moving_average(3,currentdate,52,66)
    templist[230:244] = moving_average(7,currentdate,52,66)
    templist[244:258] = moving_average(10,currentdate,52,66)
    
    templist[258:272] = moving_average(2,currentdate,66,80)
    templist[272:286] = moving_average(3,currentdate,66,80)
    templist[286:300] = moving_average(7,currentdate,66,80)
    templist[300:314] = moving_average(10,currentdate,66,80)
    
    templist[314:328] = moving_average(2,currentdate,80,94)
    templist[328:342] = moving_average(3,currentdate,80,94)
    templist[342:356] = moving_average(7,currentdate,80,94)
    templist[356:370] = moving_average(10,currentdate,80,94)
    
    templist[370:384] = moving_average(2,currentdate,94,108)
    templist[384:398] = moving_average(3,currentdate,94,108)
    templist[398:412] = moving_average(7,currentdate,94,108)
    templist[412:426] = moving_average(10,currentdate,94,108)
    
    #MA of food per occ, for today (between n PM)
    templist[426:440] = data_npm.iloc[i-1,38:52]
    
    # MAKE CHANGES ACCORDINGLY. THIS HAS BEEN DONE FOR 12-12 MODEL IN MIND.
    #Construct day type specific history, i.e. holiday history and not holiday history.
    nt = data_npm[(data_npm['Date']<=currentdate) & (data_npm['holiday']==current_hol)].iloc[-3:]
    nt.reset_index(drop=True,inplace=True)
    templist[440:454] = list(nt.iloc[0,range(24,38)])
    templist[454:468] = list(nt.iloc[1,range(24,38)])
    templist[468:482] = list(nt.iloc[2,range(24,38)])
    
    #Holiday/Not Holiday for day 'i-1'.
    templist[482] = hol.loc[currentdate][0]
    #Holiday/Not Holiday for day 'i'.
    templist[483] = hol.loc[data_npm.iloc[i][0]][0]
    
    oplist[0:14] = data_npm.iloc[i,24:38]
    
    inputlist.append(templist)
    outputlist.append(oplist)
print ("Shape of inputlist : ",len(inputlist))


'''
Generating column names for the dataframe.
'''
cnames = []
cnames = tempdf.columns[0:2]

cnames = list(cnames)

for i in tempdf.columns[2:6]:
    cnames.append(str(i))

cnames = list(cnames)

#MA of food/occ.
for i in tempdf.columns[38:52]:
        cnames.append("Moving Average of 2 days of " + i)

for i in tempdf.columns[38:52]:
        cnames.append("Moving Average of 3 days of " + i)

for i in tempdf.columns[38:52]:
        cnames.append("Moving Average of 7 days of " + i)

for i in tempdf.columns[38:52]:
        cnames.append("Moving Average of 10 days of " + i)

for i in tempdf.columns[24:38]:
        cnames.append("Moving Max of 10 days of " + i)
        
for i in tempdf.columns[24:38]:
        cnames.append("Moving Min of 10 days of " + i)

for i in tempdf.columns[24:38]:
        cnames.append("Moving SD of 10 days of " + i)

for i in range(6,-1,-1):
    for col in tempdf.columns[24:38]:
        cnames.append("History of "+col+" - "+str(i))

        
#MA of food/occ.
for i in tempdf.columns[52:66]:
        cnames.append("Moving Average of 2 days of " + i)

for i in tempdf.columns[52:66]:
        cnames.append("Moving Average of 3 days of " + i)

for i in tempdf.columns[52:66]:
        cnames.append("Moving Average of 7 days of " + i)

for i in tempdf.columns[52:66]:
        cnames.append("Moving Average of 10 days of " + i)
        
#MA of food/occ.
for i in tempdf.columns[66:80]:
        cnames.append("Moving Average of 2 days of " + i)

for i in tempdf.columns[66:80]:
        cnames.append("Moving Average of 3 days of " + i)

for i in tempdf.columns[66:80]:
        cnames.append("Moving Average of 7 days of " + i)

for i in tempdf.columns[66:80]:
        cnames.append("Moving Average of 10 days of " + i)
        
#MA of food/occ.
for i in tempdf.columns[80:94]:
        cnames.append("Moving Average of 2 days of " + i)

for i in tempdf.columns[80:94]:
        cnames.append("Moving Average of 3 days of " + i)

for i in tempdf.columns[80:94]:
        cnames.append("Moving Average of 7 days of " + i)

for i in tempdf.columns[80:94]:
        cnames.append("Moving Average of 10 days of " + i)
        
#MA of food/occ.
for i in tempdf.columns[94:108]:
        cnames.append("Moving Average of 2 days of " + i)

for i in tempdf.columns[94:108]:
        cnames.append("Moving Average of 3 days of " + i)

for i in tempdf.columns[94:108]:
        cnames.append("Moving Average of 7 days of " + i)

for i in tempdf.columns[94:108]:
        cnames.append("Moving Average of 10 days of " + i)

for i in tempdf.columns[38:52]:
        cnames.append("Food per occ of " + i)

for i in range(2,-1,-1):
    for col in tempdf.columns[24:38]:
        cnames.append("Relevant History of "+col+" - "+str(i))
        
cnames.append("Holiday/Not Holiday of today")
cnames.append("Holiday/Not Holiday of tomorrow")

print (cnames)    


inputdf = pd.DataFrame(inputlist,columns = cnames)
inputdf.Date = pd.to_datetime(inputdf.Date)
outputdf = pd.DataFrame(outputlist,columns = data_npm.columns[24:38])
print (inputdf.shape , outputdf.shape)

#History of foodsale for food1
d=[]
for i in range(104,202,14):
    d.append(i)

inputdf.iloc[:,d]

#CHECK IF HISTORY HAS BEEN CONSTRUCTED CORRECTLY.
curr_hol = 0
currentdate = pd.to_datetime("2017-11-22")
t = data_npm[data_npm['Date']<=currentdate]
nt = t[t['holiday'] == curr_hol].iloc[-3:]
nt.reset_index(drop=True,inplace=True)
nt.iloc[:,[0]+range(24,38)+[-1]]


'''
Generating the list of indices to be dropped from inputdf and outputdf , because of incorrect MA of 'food/occ' for them.
They encompass the missing period between '2015-07-23' and '2015-09-30'.

The below range of dates denote the 7 day period to be dropped after '2015-09-30'.
'211' is the index corresponding to '2015-09-30'.
'''

start = inputdf[inputdf["Date"] > pd.to_datetime("2015-07-23")].index[0]
#drop_indices = list(range(inputdf[inputdf['Date'] > pd.to_datetime("2017-01-01")].index[0]))
drop_indices = []
for i in range(start,start+11,1):
    drop_indices.append(i)

print (drop_indices, len(drop_indices))


'''
Actually dropping the rows from the required dataframes.
'''
inputdf = inputdf.drop(inputdf.index[drop_indices])
inputdf = inputdf.reset_index(drop=True)

outputdf = outputdf.drop(outputdf.index[drop_indices])
outputdf = outputdf.reset_index(drop=True)
print(inputdf.shape,outputdf.shape)

l = []
for i in range(6,482,14):
    print (i," - ",inputdf.columns[i])
    l.append(i)

print (l)

inputdf['day_of_week']  = ( (inputdf['Date'] + pd.Timedelta(1,unit='D') ).dt.dayofweek )
print inputdf.shape


inputdf = pd.concat([inputdf,pd.get_dummies(inputdf['dow'])],axis=1).copy()
inputdf.columns = list(inputdf.columns[:-3]) + ["dow1","dow2","dow3"]
print(inputdf.columns)

'''
Correcting the dow attribute.
DoW Classes : 
Class 1 -  Wednesday , Friday
Class 2 - Monday , Tuesday, Thursday
Class 3 - Saturday , Sunday
'''
for i in range(inputdf.shape[0]):
    if((inputdf.iloc[i,-4] in [5,6])):
        inputdf.iloc[i,1] = 3
    elif((inputdf.iloc[i,-4] in [0,1,3])):
        inputdf.iloc[i,1] = 2
    elif((inputdf.iloc[i,-4] in [2,4])):
        inputdf.iloc[i,1] = 1


df_total_input = inputdf.copy()
df_total_output = outputdf.copy()
print (df_total_input.shape,df_total_output.shape)


analysisdf = pd.concat([inputdf['Date'],outputdf,inputdf['dow']],axis=1)
analysisdf

for t in range(14):
    print "-----------------------------------------------"
    print outputdf.columns[t]
    for j in range(7):
        lop = list(inputdf[inputdf['day_of_week']==j].index)
        print ("Day ",j," : ",np.max(outputdf.iloc[lop,t],axis=0))
        print ("Day ",j," : ",np.min(outputdf.iloc[lop,t],axis=0))
        print ("Day ",j," : ",np.average(outputdf.iloc[lop,t],axis=0)),"\n"


inputdf.to_csv("./New Data/inputdf.csv", index=False)
outputdf.to_csv("./New Data/output.csv", index=False)

#----------------------------------------------------------- END OF GENERATION OF INPUT & OUTPUT DF.
