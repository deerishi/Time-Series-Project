import numpy as np
import pandas as pd

dateparse=lambda x:pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
train=pd.read_csv('train.csv',parse_dates=['datetime'],date_parser=dateparse)


train['year']=train.datetime.dt.year
train['month']=train.datetime.dt.month
train['dayofyear'] = train.datetime.dt.dayofyear
train['dayofweek'] = train.datetime.dt.dayofweek
train['day'] = train.datetime.dt.day
train['Hour']=train.datetime.dt.hour
train['week']=train.datetime.dt.week

print 'train is \n',train
train.to_csv('Remodeled Train.csv')
