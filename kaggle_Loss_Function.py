
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from copy import copy
from sklearn.ensemble import RandomForestRegressor
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

dateparse=lambda x:pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
train=pd.read_csv('train.csv',parse_dates=['datetime'],date_parser=dateparse)
test=pd.read_csv('test.csv',parse_dates=['datetime'],date_parser=dateparse)

#test['windspeed']=np.log(test['windspeed']+1)
#train['windspeed']=np.log(train['windspeed']+1)
print 'train.shape is ',train.shape,' and test.shape is ',test.shape


def extractFeaturesTrain(data):
    #print 'data is ',data
    data['Hour']=data.datetime.dt.hour
    labels=data['count']
    train_years=data.datetime.dt.year
    train_months=data.datetime.dt.month
    data=data.drop(['datetime','count','casual','registered'], axis = 1)
    
    return np.array(data),np.array(labels),np.array(train_years),np.array(train_months),(data.columns.values)

def extractFeaturesTest(data):
    #print 'data is \n',data
    data['Hour']=data.datetime.dt.hour
    test_years=data.datetime.dt.year
    test_months=data.datetime.dt.month
    data=data.drop(['datetime'], axis = 1)
    return np.array(data),np.array(test_years),np.array(test_months)
    
train2=copy(train)
test2=copy(test)
test=np.array(test)
#print 'train2 is ',train2
traind,labelsTrain,train_years,train_months,headers=extractFeaturesTrain(train2)
testd,test_years,test_months=extractFeaturesTest(test2)
print 'traind.shape is ',traind.shape,' and testd.shape is ',testd.shape


# In[2]:

enc=OneHotEncoder(categorical_features=[0,1,2,3,8],sparse=False)
traind2=enc.fit_transform(traind)
print traind2.shape
testd2=enc.fit_transform(testd)
print testd2.shape
ones1=np.ones((traind.shape[0],1))
ones2=np.ones((testd.shape[0],1))
traind2=copy(np.hstack((traind2,ones1)))
testd2=copy(np.hstack((testd2,ones2)))
print traind2.shape
print testd2.shape


# In[3]:

train=np.array(train)

def getSplits(years,months):
    locsTrain=[]
    locsTest=[]
    print 'in getSplits ,train is \n',train
    for i in range(0,train.shape[0]):
            if (train[i,0].year==years[0] or train[i,0].year==years[1]) and (train[i,0].month in months):
                locsTest.append(i)
            else:
                locsTrain.append(i) 
    
    return locsTrain,locsTest

def getCustomLocsTest(year,month,data):
    locs=[]
    for i in range(0,data.shape[0]):
        if data[i][0].year==year and data[i][0].month==month:
            locs.append(i)
    return locs

def calculateGradientAndLoss(weights,x,y):
    y2=y.reshape(-1,1)
    weights=weights.reshape(1,-1)
    print 'weights are \n',weights.shape
    print 'y2 is ',y2.shape
    for i in range(100):
        h=np.dot(x,weights.T)
        print 'h is \n',h
        err=np.log(y2+1)-np.log(h+1)
        print 'err is ',err.shape
        loss=np.sum(err**2)
        loss=np.sqrt(loss/x.shape[0])
        grad1=(err)/(h+1)
        print 'grad1 is ',grad1.shape
        
        grad=np.dot(grad1.T,x)
        print 'Loss at iteration ',i,' is ',loss
        print 'grad is ',grad.shape
        weights=weights+grad
    
    return weights
    
def TrainFucntion(x,y):
    weights=np.random.rand(1,x.shape[1])
    weights=calculateGradientAndLoss(weights,x,y)
    return weights
    

def Predict(weights,test):
    
    return np.dot(test,weight.T)

def crossValidate():
        months=[12]
        locsTrain,locsTest=getSplits([2011,2012],months)
        
        testSubset=traind2[locsTest]
        testSubset2=train[locsTest]
        testLabels=labelsTrain[locsTest]
        #rf3=RandomForestRegressor(20) 
        
        trainSubset=traind2[locsTrain]
        trainSubset2=train[locsTrain]
        trainLabels=labelsTrain[locsTrain]
        
        for i in [2011,2012]:
            for j in months:
                testLocs=getCustomLocsTest(i,j,testSubset2)
                testSubset3=testSubset2[testLocs]
                testSubset4=testSubset[testLocs]
                testLabels4=testLabels[testLocs]
                
                trainLocs2=np.where(trainSubset2[:,0]<=min(testSubset3[:,0]))
                
                trainSubset3=trainSubset[trainLocs2]
                trainLabels3=trainLabels[trainLocs2]
                x1=trainSubset2[trainLocs2]
                x2=testSubset2[testLocs]
                
                print 'trainSubset min is  ', min(x1[:,0]),' and max is ',max(x1[:,0])
                print 'testSubset  min is  ', min(x2[:,0]),' and max is ',max(x2[:,0])
                
                #rf3.fit(trainSubset3,trainLabels3)change here to program new function to train
                weights=TrainFucntion(trainSubset3,trainLabels3)
                
                ypred=Predict(weights,testSubset4)
                
                print 'loss with year =',i,' and month = ',j,' is ',findLoss(testLabels4,ypred)
                
                
                
crossValidate() 


