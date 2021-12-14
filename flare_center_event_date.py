#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:14:07 2021

@author: yuichiro
"""
import pandas as pd
import sys
import os
import glob
import shutil
import datetime
import numpy as np

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/hinode_catalog/Hinode Flare Catalogue new.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")



flare_times = []
flare_locations = []
for i in range (len(flare_csv['peak'])):
    yyyy = flare_csv['peak'][i].split('/')[0]
    if int(yyyy) < 2012:
        pass
    elif int(yyyy) > 2014:
        pass
    else:
        mm = flare_csv['peak'][i].split('/')[1]
        dd = flare_csv['peak'][i].split('/')[2].split(' ')[0]
        str_date = yyyy + mm + dd
        HH = flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0]
        MM = flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1]
        location = flare_csv['AR location'][i]
        if location[4] == '0':
            peak_time = datetime.datetime(int(flare_csv['peak'][i].split('/')[0]), int(flare_csv['peak'][i].split('/')[1]), int(flare_csv['peak'][i].split('/')[2].split(' ')[0]), int(flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0]), int(flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1]))
            # pd.to_datetime(flare_csv['peak'][i].split('/')[0] + flare_csv['peak'][i].split('/')[1] + flare_csv['peak'][i].split('/')[2].split(' ')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
            if (peak_time.hour >= 8) and (peak_time.hour <= 16):
                # pd_start_time = pd.to_datetime(flare_csv['start'][i].split('/')[0] + flare_csv['start'][i].split('/')[1] + flare_csv['start'][i].split('/')[2].split(' ')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                # pd_end_time = pd.to_datetime(flare_csv['end'][i].split('/')[0] + flare_csv['end'][i].split('/')[1] + flare_csv['end'][i].split('/')[2].split(' ')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                flare_times.append(peak_time)
                flare_locations.append(location)
flare_times = np.array(flare_times)
flare_locations = np.array(flare_locations)

file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'

sunspot_obs_times = []
sunspot_num_list = []


print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
    if (BG_obs_time_event >= datetime.datetime(2012, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2015, 1, 1)):
        sunspot_num = csv_input['sunspot_number'][i]
        if not sunspot_num == -1:
            sunspot_obs_times.append(BG_obs_time_event)
            # Frequency_list = csv_input['Frequency'][i]
            sunspot_num_list.append(sunspot_num)
        else:
            print (BG_obs_time_event)

sunspot_obs_times = np.array(sunspot_obs_times)
sunspot_num_list = np.array(sunspot_num_list)

flare_days = []
for flare_time in flare_times:
    sunspot_idx = np.where(sunspot_obs_times == datetime.datetime(flare_time.year, flare_time.month, flare_time.day))[0][0]
    if sunspot_num_list[sunspot_idx] >= 36:
        # if sunspot_num_list[sunspot_idx] < 50:
            flare_days.append(flare_time.strftime("%Y%m%d"))



    
    
files_list = []
for flare_day in flare_days:
    print(flare_day)
    burst_start_time_list = []
    file_names = []
    files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/nonclearevent_analysis/'+flare_day[:4]+'/'+flare_day+'_*peak.png')
    # files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/test_flare_related/'+flare_day[:4]+'/*/'+flare_day+'_*compare.png')
    print(len(files))
    for file in files:
        print (file)
        file_name = file.split('/')[-1]
        yyyy = int(flare_day[:4])
        mm = int(flare_day[4:6])
        dd = int(flare_day[6:8])
        stime = file_name.split('_')[1]
        shour = int(stime[:2])
        smin = int(stime[2:4])
        ssec = int(stime[4:6])
        event_sec = int(file_name.split('_')[5])
        burst_start_time = datetime.datetime(yyyy, mm, dd, shour, smin, ssec) + datetime.timedelta(seconds=event_sec)
        burst_start_time_list.append(burst_start_time_list)
        file_names.append(file)
    if len(files) > 0:
        file_names = np.array(file_names)
        burst_start_time_list = np.array(burst_start_time_list)
        flare_idx = np.where((flare_times >= datetime.datetime(yyyy, mm, dd)) and (flare_times <= datetime.datetime(yyyy, mm, dd)+datetime.timedelta(days = 1)))[0]
        selected_flare_times = flare_times[flare_idx]
        if len(selected_flare_times) == 0:
            print ('Problem')
            sys.exit()
        else:
            for selected_flare_time in selected_flare_times:
                selected_burst_idx = np.where((burst_start_time_list >= selected_flare_time - datetime.timedelta(minutes = 10)) and (burst_start_time_list <= selected_flare_time + datetime.timedelta(minutes = 10)))[0]
                if len(file_names[selected_burst_idx]) > 0:
                    files_list.extend(file_names[selected_burst_idx])
            # if pd_peak_time + pd.to_timedelta(10,unit='minute') >= pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'):
                # if pd_peak_time - pd.to_timedelta(10,unit='minute') <= pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'):

# # if pd_end_time < pd.to_datetime('20170101'):
# #     sys.exit()

#     flare_day = flare_days[0]
#     print(flare_day)
#     burst_start_time_list = []
#     file_names = []
#     files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/test_flare_related/'+flare_day[:4]+'/*/'+flare_day+'_*compare.png')
#     print(len(files))
#     for file in files:
#         print (file)
#         file_name = file.split('/')[-1]
#         yyyy = int(flare_day[:4])
#         mm = int(flare_day[4:6])
#         dd = int(flare_day[6:8])
#         stime = file_name.split('_')[1]
#         shour = int(stime[:2])
#         smin = int(stime[2:4])
#         ssec = int(stime[4:6])
#         event_sec = int(file_name.split('_')[5])
#         burst_start_time = datetime.datetime(yyyy, mm, dd, shour, smin, ssec) + datetime.timedelta(seconds=event_sec)
#         burst_start_time_list.append(burst_start_time_list)
#         file_names.append(file)
#     file_names = np.array(file_names)
#     burst_start_time_list = np.array(burst_start_time_list)
#     flare_idx = np.where((flare_times >= datetime.datetime(yyyy, mm, dd)) and (flare_times <= datetime.datetime(yyyy, mm, dd)+datetime.timedelta(days = 1)))[0]
#     selected_flare_times = flare_times[flare_idx]
#     if len(selected_flare_times) == 0:
#         print ('Problem')
#         sys.exit()
#     else:
#         for selected_flare_time in selected_flare_times:
#             selected_burst_idx = np.where((burst_start_time_list >= selected_flare_time - datetime.timedelta(minutes = 10)) and (burst_start_time_list <= selected_flare_time + datetime.timedelta(minutes = 10)))[0]
#             if len(file_names[selected_burst_idx]) > 0:
#                 print (file_names[selected_burst_idx])
#                 files_list.extend(file_names[selected_burst_idx])
