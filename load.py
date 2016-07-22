import numpy as np
import pandas as pd

dateparse=lambda x:pd.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
datatrain=pd.read_csv('train.csv',parse_dates=['datetime'],date_parser=dateparse)
print 'train is \n',datatrain

datatrain['year']=datatrain.datetime.dt.year
