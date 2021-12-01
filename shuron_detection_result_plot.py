#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:57:07 2021

@author: yuichiro
"""
import glob
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
event_date_list = []
event_time_list = []
files = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/*/*/*peak.png')

for file in files:
    event_date = datetime.datetime(int(file.split('/')[-1].split('_')[0][:4]), int(file.split('/')[-1].split('_')[0][4:6]), int(file.split('/')[-1].split('_')[0][6:]))
    event_time = datetime.datetime(2013, 1, 1, int(file.split('/')[-1].split('_')[1][:2]), int(file.split('/')[-1].split('_')[1][2:4]), int(file.split('/')[-1].split('_')[1][4:6]))+ datetime.timedelta(seconds=int(round((int(file.split('/')[-1].split('_')[5])+int(file.split('/')[-1].split('_')[5]))/2,1)))
    event_date_list.append(event_date)
    event_time_list.append(event_time)
    
event_date_list = np.array(event_date_list)
event_time_list = np.array(event_time_list)

    
date_in=[20120101, 20201231]

start_day, end_day=date_in

edate=datetime.datetime.strptime(str(end_day), '%Y%m%d')
DATE=datetime.datetime.strptime(str(start_day), '%Y%m%d')

event_num = []
event_num_time = []
while DATE <= edate:
    print(DATE)
    try:
        idx = np.where((event_date_list >= DATE) & (event_date_list <= DATE + relativedelta(months=1, days=-1)))[0]
        event_num.append(len(event_date_list[idx]))
        event_num_time.append(DATE+ relativedelta(days=15))
    except:
        print('Plot error: ',DATE)
    DATE += relativedelta(months=1)

figure_=plt.figure(1,figsize=(10,8))
labels = ['2012', '2013', '2014', '2017', '2018', '2019', '2020']
event_num_time2 = np.arange(0,108,1)
plt.bar(event_num_time2, event_num, width=0.8, color='white', edgecolor='k')
plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Number of events', fontsize = 18)
plt.show()


event_num = []
event_num_times = np.arange(6,19,1)
for event_num_time in event_num_times:
    start = datetime.datetime(2013,1,1,event_num_time)
    end = datetime.datetime(2013,1,1,event_num_time+1)
    idx = np.where((event_time_list >= start) & (event_time_list <= end))[0]
    event_num.append(len(event_date_list[idx]))



figure_=plt.figure(1,figsize=(10,8))
# labels = ['2012', '2013', '2014', '2017', '2018', '2019', '2020']
event_num_time2 = np.arange(0,108,1)
plt.bar(event_num_times, event_num, width=0.8, color='white', edgecolor='k')
# plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xlabel('Year', fontsize = 18)
plt.ylabel('Number of events', fontsize = 18)
plt.show()

