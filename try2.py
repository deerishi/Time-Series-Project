
# coding: utf-8

# In[1]:

import numpy as np

from copy import copy
from sklearn.ensemble import RandomForestRegressor
import csv
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns



dateparse=lambda x:pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
train=pd.read_csv('train.csv',parse_dates=['datetime'],date_parser=dateparse)
test=pd.read_csv('test.csv',parse_dates=['datetime'],date_parser=dateparse)

#test['windspeed']=np.log(test['windspeed']+1)
#train['windspeed']=np.log(train['windspeed']+1)
print train.shape


# In[2]:




def extractFeaturesTrain(data):
    #print 'data is ',data
    #data['Hour']=data.datetime.dt.hour
    labels=data['count']
    train_years=data.datetime.dt.year
    train_months=data.datetime.dt.month
    data=data.drop(['datetime','count','casual','registered'], axis = 1)
    
    return np.array(data),np.array(labels),np.array(train_years),np.array(train_months),(data.columns.values)

def extractFeaturesTest(data):
    
    #data['Hour']=data.datetime.dt.hour
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

submit=np.array((test.shape[0],2))

#train.to_csv('Remodeled Train.csv')
train=np.array(train)
print 'train is \n',traind.shape
print 'labels train are \n',labelsTrain.shape
print 'test is \n',testd.shape

l=[]
print train_years
print train_months
for i in range(train_months.shape[0]):
    l.append(str(train_years[i]) + '_' + str(train_months[i]))
   
l2=[]
for i in set(l):
        l2.append(i)

print '\nl2 is \n',l2


randomState=10
q=[0,1,2,3,4,5,6,7,8,9]
#data2 = TSNE(random_state=randomState).fit_transform(traind)
#np.save('Original data by TSNE',data2)
data2=np.load('Original data by TSNE.npy')
print 'data2 is ',data2[:,0],' and ',data2[:,1]
print 'data2 shape is ',data2.shape
fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
ax=fig.add_subplot(111)
#ax.scatter(data2[:,0],data2[:,1],labelsTrain)
palette=(sns.color_palette("hls",13))
s=zip(palette,q)
print 's i
s ',s

palette=np.array(palette)
ax.scatter(data2[:,0],data2[:,1],c=palette[train_months.astype(np.int)])

#plt.legend(palette.tolist(),[0,1,2,3,4,5,6,7,8,9,10,11,12])
p1=[mpatches.Patch(color=palette[i],label=i) for i in range(1,13)]
#ax.set_zlabel('Count Hourly')
plt.legend(handles=p1)
plt.title('Original Monthly data reduced to 2D by TSNE')
plt.show()

