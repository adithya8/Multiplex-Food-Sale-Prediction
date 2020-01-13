import pandas as pd
from pandas import np
import os
import json
from collections import Counter
import pickle as pkl
from datetime import datetime, timedelta

class generateDf:
    """ Class with object generatingFoodDf to generate language wise occupancy and total food sale for each day. Call for help() if need be :P """
    
    def __init__(self):
        self.defaultSlotList = ["06:00:00","12:00:00","15:00:00","18:00:00","21:00:00","30:00:00"]
        self.defaultOccPath = "/Data/OccAtTimes.csv"
        self.langPathName = "/Data/film_features.pkl"
        self.itemLookUp = {}
        self.itemId = []
        self.columnName = []
        self.foodColumnName = []
        self.dataTotalSlot = []
        self.occData = []
        self.missingDates = []
        self.occDf = pd.read_csv(self.defaultOccPath)
        self.langDict = pkl.load(open(self.langPathName,'rb'))
        self.occDf['DateTimeTime']=pd.to_datetime(self.occDf['DateTimeTime'])
        self.occDf=self.occDf[self.occDf['DateTimeTime']>=pd.to_datetime('1-Jan-15 00:00:00')]        
        
    def generateTransactionData(self):
        """ Function that extracts raw csv and cleans it. Creates itemId, itemLookUp and reqData """
        print ("Collecting raw CSV data")
        # import data and drop duplicate records
        csvData = pd.read_csv("/Data/allFoodTransactions.csv", low_memory=False)
        csvData.drop_duplicates(inplace=True)
        csvData = csvData.reset_index(drop=True)
        print ("CSV data loaded. Good to go.")
    
        # Taking out the top n food items that got sold
        j = list(Counter(csvData.ItemId).keys())
        k = list(Counter(csvData.ItemId).values())

        j = np.reshape(j, (-1, 1))
        k = np.reshape(k, (-1, 1))
        temp = pd.DataFrame(np.hstack((j, k)), columns=['id', 'c']).sort_values(by='c', ascending=False)

        for i in temp.id:
            self.itemLookUp[i] = csvData[csvData.ItemId == i].iloc[1,3]

        self.itemId = list(self.itemLookUp)

        #Pruning 
        self.reqData = csvData[csvData.ItemId.isin(self.itemLookUp.keys())]

        #changing the datatype of DateTime column 
        self.reqData['DateTime'] = pd.DatetimeIndex(self.reqData['DateTime'])

        #Dropping
        self.reqData.drop(labels='ItemDescription',axis=1,inplace=True)

        print ("reqData shape: ",self.reqData.shape)

    def getSlotTransaction(self,df):
        """ returns the number of transactions for the sale of a particular item passed in the argument df. Internally called by getDataForSlot(). """
        noOfTrans = 0
        unique = list(set(df.lgnNumber))
        for i in unique:
            val = np.sum(df[df.lgnNumber == i].NoOfItems.values)
            if(val > 0):
                noOfTrans += 1
            elif(val < 0):
                noOfTrans -= 1
        return noOfTrans

    def getDataForSlot(self,slotData):
        """ returns a list of number of transactions and number of food sold for all the itemId from the slotDataDf for each day. Internally called by generateSlotData(). """
        temp = []
        
        for i in self.itemId:
            foodSaleforSlot = slotData[slotData.ItemId == i]
            temp.append(self.getSlotTransaction(foodSaleforSlot))

        for i in self.itemId:
            foodSaleforSlot = slotData[slotData.ItemId == i]
            temp.append(np.sum(foodSaleforSlot.NoOfItems.values))

        return temp

    def generateSlotData(self,slotList= None):
        """ generates slotwise number of transactions and number of food sold for each of itemId for everyday according to the slot specified by the argument(list). This method calls generateTransactionData(), getSlotTransaction(), getDataForSlot() and geenrateOcc() internally. This method needs to be called.
        Example : slotList = ['06:00:00','11:00:00','16:00:00','30:00:00']. """
        
        if slotList is None:
            slotList = self.defaultSlotList
        
        self.generateTransactionData()
        
        minDate, maxDate = self.reqData.DateTime.min().date(), self.reqData.DateTime.max().date()
        today = pd.to_datetime(minDate) + pd.Timedelta("06:00:00")
        i = 0
        slots = slotList
        count = 0
        dfDates = [x.date() for x in self.reqData.DateTime]
        dfDates = pd.DataFrame(dfDates)

        
        print ("Generating Slotwise data")
        
        while(1):
            if(today > pd.to_datetime(maxDate)):
                break

            tomorrow = today + pd.Timedelta("24:00:00")
            currentDay = pd.to_datetime(today.date())

            if(np.sum((dfDates == today.date()).values) == 0):
                self.missingDates.append(today.date())
                count += 1
                today = tomorrow
                continue

            for i in range(len(slots)-1):

                tempSlotData = [0]*(2*len(self.itemId) + 2)
                slotData = self.reqData[(self.reqData.DateTime >= currentDay + pd.Timedelta(slots[i])) &
                                   (self.reqData.DateTime < currentDay + pd.Timedelta(slots[i+1]))]
                tempSlotData[0] = (today.date())
                tempSlotData[1] = (i+1)
                tempSlotData[2:] = self.getDataForSlot(slotData)
                self.dataTotalSlot.append(tempSlotData)
                del tempSlotData
                
            today = tomorrow

        print ("Generating Slotwise DF")
        
        self.foodColumnName.append('DateTime')
        self.foodColumnName.append('daySlot')
        for i in self.itemId:
            self.foodColumnName.append(str(self.itemLookUp[i])+'_NoOfTrans')
        for i in self.itemId:
            self.foodColumnName.append(str(self.itemLookUp[i]))

        self.foodSaleDf = pd.DataFrame(np.array(self.dataTotalSlot),columns=self.foodColumnName)
        self.foodSaleDf.DateTime= pd.DatetimeIndex(self.foodSaleDf.DateTime)
        self.foodSaleDf = self.foodSaleDf.set_index(self.foodSaleDf.DateTime)
        self.foodSaleDf = self.foodSaleDf.resample('d').sum()
        self.foodSaleDf = self.foodSaleDf.dropna(axis=0)
        self.foodSaleDf = self.foodSaleDf.reset_index()

        del self.foodSaleDf['daySlot']
        print ("Slotwise Food data shape: ", self.foodSaleDf.shape)
        
        self.generateOcc(slotList)

    def getLangCategory(self,movie):
        """ Returns language category of a particular movie. Internally invoked by generateOcc()."""
        return int(self.langDict[movie]['Language_Param'] if movie in self.langDict else 4) 

    def generateOcc(self,slotList = None):
        """  Generates language wise occDataDf containing language wise occupancy and allotted. Internally invoked by generateSlotData().  """
        self.occDataDf = []
        
        if slotList is None:
            slotList = self.defaultSlotList
        print ("Generating Occ Data")
        for currentDate in self.occDf['Date'].unique():
            currentDate = (pd.to_datetime(currentDate).date())          
            for i in range(len(slotList)-1):
                tempDf = self.occDf[self.occDf['Date'] == str(currentDate)]
                tempDf = tempDf[tempDf.DateTimeTime >= pd.to_datetime(currentDate) + pd.Timedelta(slotList[i])]
                tempDf = tempDf[tempDf.DateTimeTime < pd.to_datetime(currentDate) + pd.Timedelta(slotList[i+1])]
                temp = [0]*10
                
                temp[0] = (currentDate)
                temp[1] = (i + 1)

                for record in tempDf.index:
                    langCategory = self.getLangCategory(tempDf.loc[record]['Movie'])
                    temp[langCategory + 1] += int(tempDf.loc[record]['OccAtShow'])
                    temp[langCategory + 5] += int(tempDf.loc[record]['Capacity'])
                self.occDataDf.append(temp)
                del temp,tempDf
        self.occDataL = self.occDataDf
        self.occDataDf = pd.DataFrame(self.occDataDf,columns=['DateTime','dayslot','langugae1','langugae2','langugae3','others','langugae1_allotted','langugae2_allotted','langugae3_allotted','others_allotted'])
        print ("Generated Occpuncy DF")

        self.occDataDf = self.occDataDf.sort_values(by=['DateTime','dayslot'],ascending=[True,True])

        print ("Cleaning Occupancy DF...")
        self.occDataDf = self.occDataDf.reset_index()
        del self.occDataDf["index"]
        self.occDataDf.DateTime = pd.to_datetime(self.occDataDf.DateTime)

        self.occDataDf = self.occDataDf.set_index(self.occDataDf.DateTime)
        self.occDataDf = self.occDataDf.resample('d').sum()
        self.occDataDf = self.occDataDf.dropna(axis=0)
        self.occDataDf = self.occDataDf.reset_index()

        del self.occDataDf['dayslot']            
        print ("Occupancy Df shape: ", self.occDataDf.shape)

    def generateColumnNames(self):
        """ Generates column Names for finalDf. Internally invoked by generateDF(). """
        self.columnName = []
        self.columnName.append('DateTime')
        self.columnName.append('dow')
        for i in self.occDataDf.columns[1:]:
            self.columnName.append(i)
        for i in self.foodColumnName[2:]:
            self.columnName.append(i)
    
    def generateFinalDf(self):
        """ Creates the finalDf integrating both occData and foodSaleDf. Invokes generateColumnNames() internally. Needs to be invoked."""
        
        print ("Integrating Food Sale and Occupancy DF ....")
        notInOcc = list(set(self.foodSaleDf.DateTime) - set(self.occDataDf.DateTime))
        notInFood = list(set(self.occDataDf.DateTime) - set(self.foodSaleDf.DateTime))
        for i in notInOcc:
            self.foodSaleDf = self.foodSaleDf[self.foodSaleDf.DateTime != i]
        for i in notInFood:
            self.occDataDf = self.occDataDf[self.occDataDf.DateTime != i]    

        self.foodSaleDf = self.foodSaleDf[self.foodSaleDf.DateTime >= pd.to_datetime("2015-01-01")]
        self.occDataDf = self.occDataDf[self.occDataDf.DateTime >= pd.to_datetime("2015-01-01")]

        self.foodSaleDf = self.foodSaleDf.reset_index(drop=True)
        self.occDataDf = self.occDataDf.reset_index(drop=True)
        
        self.generateColumnNames()
        
        del self.occDataDf['DateTime']


        self.finalDf = pd.concat([self.foodSaleDf,self.occDataDf],axis=1)

        self.finalDf['dow'] = [pd.to_datetime(x).dayofweek for x in self.finalDf.DateTime]


        
        self.finalDf = self.finalDf[self.columnName]
        print ("finalDf shape: ", self.finalDf.shape)
        print ("DF created. Object name : finalDf")

    def help(self):
        """ Go for it :P Expand the docstring for an example.
                                    Example usage of class
                                    
        > from ClassModelPreprocessing import generatingFoodDf as gf
        > gf.generateSlotData(['06:00:00','18:00:00'])
        > gf.generateFinalDf()
        > print (gf.finalDf.head())"""

        print ("DATA MEMBERS")
        print ("\n")
        print ("defaultSlotList - List of time demarcating slots")
        print ("defaultOccPath - String holding default path to language Occupancy pkl file")
        print ("itemId - list of all the item Ids")
        print ("itemLookUp - dict containing itemId as keys and Item Name as corresponding values")
        print ("foodColumnName - list containg the column name for foodSaleDf")
        print ("columnName - list of column names for finalDf")
        print ("missingDates - list of missing Dates")
        print ("reqData - Df containing Transaction data")
        print ("foodSaleDf - Df containing daywise food sale and number of transactions")
        print ("occDataDf - Df containing language wise occupancy and allotted for each day")
        print ("finalDf - Df integrating occupancy and foodSaleDf")
        print ("\n")
        print ("MEMBER FUNCTIONS")
        print ("\n")
        print ("generateTransactionData() - extracts raw csv and cleans it. Creates itemId, itemLookUp and reqData")
        print ("getSlotTransaction(df) - returns the number of transactions for the sale of a particular item passed in df")
        print ("getDataForSlot(slotData) - returns a list of number of transactions and number of food sold for all the itemId from the slotDataDf")
        print ("generateSlotData(slotList) - generates slotwise number of transactions and number of food sold for each of itemId for everyday according to the slot specified by the argument(list)")
        print ("generateOcc(slotList) - generates occData Df for the slotList given.")
        print ("generateColumnNames() - generates column Names for finalDf")
        print ("generateFinalDf() - creates the finalDf integrating both occData and foodSaleDf")
        print ('help() - I HOPE THIS HELPED :P xD')
        print ("\n")
        print ("OBJECT")
        print ("\n")
        print ("generatingFoodDf - use this object to create a class object of generateDf")
        print ("\n")
        print ('''Example usage of class
                                    
        > from ClassModelPreprocessing import generatingFoodDf as gf
        > gf.generateSlotData(['06:00:00','18:00:00'])
        > gf.generateFinalDf()
        > print (gf.finalDf.head())''')

generatingFoodDf = generateDf()

