#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 12:28:50 2021

@author: yuichiro
"""
import glob
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import sys
import matplotlib.pyplot as plt


Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
# Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'


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


    # if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23))) | ((obs_time >= datetime.datetime(1995,1,1)) & (obs_time <= datetime.datetime(1997,12,31,23)))):
    #     if sunspots_num <= 36:


    # if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
    #     if sunspots_num >= 36:



date_in=[19950101, 20210101]

start_day, end_day=date_in

edate=datetime.datetime.strptime(str(end_day), '%Y%m%d')
sdate=datetime.datetime.strptime(str(start_day), '%Y%m%d')
DATE=datetime.datetime.strptime(str(start_day), '%Y%m%d')

freq_start_list = []
freq_end_list = []
obs_start_list = []
obs_end_list = []
obs_time_list = []
hour_list = np.zeros(23)
month_list = np.zeros(12)

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/nda_obs_1995_2020.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_start'][j].split('-')[0]), int(csv_input_final['obs_start'][j].split('-')[1]), int(csv_input_final['obs_start'][j].split(' ')[0][-2:]), int(csv_input_final['obs_start'][j].split(' ')[1][:2]), int(csv_input_final['obs_start'][j].split(':')[1]), int(csv_input_final['obs_start'][j].split(':')[2][:2]))
    obs_end = datetime.datetime(int(csv_input_final['obs_end'][j].split('-')[0]), int(csv_input_final['obs_end'][j].split('-')[1]), int(csv_input_final['obs_end'][j].split(' ')[0][-2:]), int(csv_input_final['obs_end'][j].split(' ')[1][:2]), int(csv_input_final['obs_end'][j].split(':')[1]), int(csv_input_final['obs_end'][j].split(':')[2][:2]))
    if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23))) | ((obs_time >= datetime.datetime(1995,1,1)) & (obs_time <= datetime.datetime(1997,12,31,23)))):
        sunspots_idx = np.where(sunspot_obs_times == datetime.datetime.strptime(obs_end.strftime("%Y%m%d"), '%Y%m%d'))[0][0]
        if sunspot_obs_num_list[sunspots_idx] <= 36:
            freq_start_list.append(csv_input_final["freq_start"][j])
            freq_end_list.append(csv_input_final["freq_end"][j])
            obs_start_list.append(obs_time)
            obs_end_list.append(obs_end)
            obs_gap = float(str(obs_end -obs_time).split(':')[0]) + float(str(obs_end -obs_time).split(':')[1])/60 + float(str(obs_end -obs_time).split(':')[2])/3600
            obs_time_list.append(obs_gap)

    if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
        sunspots_idx = np.where(sunspot_obs_times == datetime.datetime.strptime(obs_end.strftime("%Y%m%d"), '%Y%m%d'))[0][0]
        if sunspot_obs_num_list[sunspots_idx] >= 36:
            freq_start_list.append(csv_input_final["freq_start"][j])
            freq_end_list.append(csv_input_final["freq_end"][j])
            obs_start_list.append(obs_time)
            obs_end_list.append(obs_end)
            obs_gap = float(str(obs_end -obs_time).split(':')[0]) + float(str(obs_end -obs_time).split(':')[1])/60 + float(str(obs_end -obs_time).split(':')[2])/3600
            obs_time_list.append(obs_gap)

obs_start_list = np.array(obs_start_list)
obs_time_list = np.array(obs_time_list)
obs_time_2_list = []

select_periods = [[datetime.datetime(1995,1,1), datetime.datetime(1997,12,31,23)],[datetime.datetime(2007,1,1), datetime.datetime(2009,12,31,23)],[datetime.datetime(2017,1,1), datetime.datetime(2020,12,31,23)]]
for select_period in select_periods:
    start = select_period[0]
    end = select_period[1]
    select_idx = np.where((obs_start_list >= start) & (obs_start_list <= end))[0]
    obs_time_2 = np.sum(obs_time_list[select_idx])
    obs_time_2_list.append(obs_time_2)

x = np.arange(0,3,1)
values = ['1995-1997', '2007-2009', '2017-2020'] 
plt.bar(x, obs_time_2_list, align="center")
plt.xticks(x,values)
plt.xlabel("Analysis period")
plt.ylabel("Observation time (Hour)")
plt.show()


x = np.arange(0,3,1)
values = ['1995-1997', '2007-2009', '2017-2020'] 
plt.bar(x, [0.0014383214788342004, 0.00014531599326024424, 0.0017764833574351133], align="center")
plt.xticks(x,values)
plt.xlabel("Analysis period")
plt.ylabel("Events/Hour")
plt.show()

