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
import sys
import matplotlib.pyplot as plt


# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'


file_gain =Parent_directory + '/hinode_catalog/SN_d_tot_V2.0.csv'

sunspot_obs_times = []
sunspot_obs_num_list = []
print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
    # if (BG_obs_time_event >= datetime.datetime(2008, 12, 1)) & (BG_obs_time_event <= datetime.datetime(2019, 12, 31)):
    if (BG_obs_time_event >= datetime.datetime(1995, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2021, 1, 1)):
        sunspot_num = csv_input['sunspot_number'][i]
        if not sunspot_num == -1:
            sunspot_obs_times.append(BG_obs_time_event)
            # Frequency_list = csv_input['Frequency'][i]
            sunspot_obs_num_list.append(sunspot_num)
        else:
            print (BG_obs_time_event)

sunspot_obs_times = np.array(sunspot_obs_times)
sunspot_obs_num_list = np.array(sunspot_obs_num_list)

sunspots_min_idx = np.where(((sunspot_obs_times >= datetime.datetime(2017,1,1)) & (sunspot_obs_times <= datetime.datetime(2020,12,31))))[0]
sunspots_max_idx = np.where(((sunspot_obs_times >= datetime.datetime(2012,1,1)) & (sunspot_obs_times <= datetime.datetime(2014,12,31))))[0]










event_date_list = []
event_time_list = []
event_month_list = []
files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/afjpgusimpleselect/*/*/*peak.png')

for file in files:
    event_date = datetime.datetime(int(file.split('/')[-1].split('_')[0][:4]), int(file.split('/')[-1].split('_')[0][4:6]), int(file.split('/')[-1].split('_')[0][6:]))
    event_time = datetime.datetime(2013, 1, 1, int(file.split('/')[-1].split('_')[1][:2]), int(file.split('/')[-1].split('_')[1][2:4]), int(file.split('/')[-1].split('_')[1][4:6]))+ datetime.timedelta(seconds=int(round((int(file.split('/')[-1].split('_')[5])+int(file.split('/')[-1].split('_')[5]))/2,1)))
    event_date_list.append(event_date)
    event_time_list.append(event_time)
    event_month = datetime.datetime(2013, int(file.split('/')[-1].split('_')[0][4:6]), 1, 0,0)
    event_month_list.append(event_month)
    
event_date_list = np.array(event_date_list)
event_time_list = np.array(event_time_list)
event_month_list = np.array(event_month_list)


date_in=[20120101, 20210101]

start_day, end_day=date_in

edate=datetime.datetime.strptime(str(end_day), '%Y%m%d')
sdate=datetime.datetime.strptime(str(start_day), '%Y%m%d')
DATE=datetime.datetime.strptime(str(start_day), '%Y%m%d')

freq_start_list = []
freq_end_list = []
obs_start_list = []
obs_end_list = []
hour_list = np.zeros(23)
month_list = np.zeros(12)

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/nda_obs_2012_2020.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
for j in range(len(csv_input_final)):
    obs_start = datetime.datetime(int(csv_input_final['obs_start'][j].split('-')[0]), int(csv_input_final['obs_start'][j].split('-')[1]), int(csv_input_final['obs_start'][j].split(' ')[0][-2:]), int(csv_input_final['obs_start'][j].split(' ')[1][:2]), int(csv_input_final['obs_start'][j].split(':')[1]), int(csv_input_final['obs_start'][j].split(':')[2][:2]))
    obs_end = datetime.datetime(int(csv_input_final['obs_end'][j].split('-')[0]), int(csv_input_final['obs_end'][j].split('-')[1]), int(csv_input_final['obs_end'][j].split(' ')[0][-2:]), int(csv_input_final['obs_end'][j].split(' ')[1][:2]), int(csv_input_final['obs_end'][j].split(':')[1]), int(csv_input_final['obs_end'][j].split(':')[2][:2]))
    if (obs_start >= sdate) and (obs_start <= edate):
        freq_start_list.append(csv_input_final["freq_start"][j])
        freq_end_list.append(csv_input_final["freq_end"][j])
        obs_start_list.append(obs_start)
        obs_end_list.append(obs_end)
        start_hour = obs_start.hour
        end_hour = obs_end.hour
        if end_hour - start_hour >= 2:
            for i in range(start_hour+1,end_hour,1):
                hour_list[i] += 1
            hour_list[start_hour] += 1 - (obs_start.minute/60 + obs_start.second/3600)
            hour_list[end_hour] += (obs_end.minute/60 + obs_end.second/3600)
        elif end_hour - start_hour == 1:
            hour_list[start_hour] += 1 - (obs_start.minute/60 + obs_start.second/3600)
            hour_list[end_hour] += (obs_end.minute/60 + obs_end.second/3600)
        elif end_hour - start_hour== 0:
            (obs_end.minute/60 + obs_end.second/3600) - (obs_start.minute/60 + obs_start.second/3600)
        else:
            sys.exit()
        month = obs_start.month - 1
        month_list[month] += end_hour - start_hour + (obs_end.minute/60 + obs_end.second/3600) - (obs_start.minute/60 + obs_start.second/3600)



freq_start_list = np.array(freq_start_list)
freq_end_list = np.array(freq_end_list)
obs_start_list = np.array(obs_start_list)
obs_end_list = np.array(obs_end_list)



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

figure_=plt.figure(1,figsize=(10,16))
labels = ['2012', '2013', '2014', '2017', '2018', '2019', '2020']
event_num_time2 = np.arange(0,len(event_num),1)
ax1 = figure_.add_subplot(212)
ax1.bar(event_num_time2, event_num, width=1, color='white', edgecolor='k')
plt.xlabel('Year', fontsize = 30)
plt.ylabel('Number of events', fontsize = 30)
plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)

ax2 = figure_.add_subplot(211)

ax2.plot((np.arange(0,len(sunspot_obs_times[sunspots_min_idx]),1)/len(sunspot_obs_times[sunspots_min_idx])*48)+60, sunspot_obs_num_list[sunspots_min_idx], '.')
ax2.plot((np.arange(0,len(sunspot_obs_times[sunspots_max_idx]),1)/len(sunspot_obs_times[sunspots_max_idx])*36), sunspot_obs_num_list[sunspots_max_idx], '.')
ax1.set_ylim(0, 250)
ax2.set_ylim(0, 250)

plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xlabel('Year', fontsize = 30)
plt.ylabel('Sunspot Number', fontsize = 30)
plt.xticks(fontsize= 20)
plt.yticks(fontsize= 20)


plt.show()

print (np.sum(event_num))



event_num = []
event_num_times = np.arange(0,23,1)
for event_num_time in event_num_times:
    start = datetime.datetime(2013,1,1,event_num_time)
    end = datetime.datetime(2013,1,1,event_num_time+1)
    idx = np.where((event_time_list >= start) & (event_time_list <= end))[0]
    event_num.append(len(event_date_list[idx]))



figure_=plt.figure(1,figsize=(10,8))
# labels = ['2012', '2013', '2014', '2017', '2018', '2019', '2020']
event_num_time2 = np.arange(0,108,1)
plt.bar(event_num_times, event_num/hour_list, width=1, color='white', edgecolor='k')
# plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xlabel('UT(Hour)', fontsize = 18)
plt.ylabel('Events/Hour', fontsize = 18)
plt.xlim(6,17)
plt.show()

figure_=plt.figure(1,figsize=(10,8))
# labels = ['2012', '2013', '2014', '2017', '2018', '2019', '2020']
event_num_time2 = np.arange(0,108,1)
plt.bar(event_num_times, hour_list, width=1, color='white', edgecolor='k')
# plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xlabel('UT(Hour)', fontsize = 18)
plt.ylabel('Hour', fontsize = 18)
plt.xlim(6,17)
plt.show()


print (hour_list)


event_num = []
event_num_months = np.arange(0,12,1) + 1
for event_num_month in event_num_months:
    start = datetime.datetime(2013,event_num_month,1,0,0)
    idx = np.where((event_month_list == start))[0]
    event_num.append(len(event_month_list[idx]))



figure_=plt.figure(1,figsize=(10,8))
# labels = ['2012', '2013', '2014', '2017', '2018', '2019', '2020']
plt.bar(event_num_months, event_num/month_list, width=1, color='white', edgecolor='k')
# plt.xticks([0,12,24,60, 72, 84, 96], labels)
plt.xlabel('Month', fontsize = 18)
plt.ylabel('Events/Hour', fontsize = 18)
plt.show()

