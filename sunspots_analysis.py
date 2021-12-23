#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 11:43:54 2021

@author: yuichiro
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pynverse import inversefunc
from dateutil import relativedelta
from matplotlib import dates as mdates
from datetime import date
import scipy
from scipy import stats
import sys
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from dateutil import relativedelta

# file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'
file_gain = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/hinode_catalog/SN_d_tot_V2.0.csv'

sunspot_obs_times = []
sunspot_num_list = []


print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
    if (BG_obs_time_event >= datetime.datetime(1995, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2021, 1, 1)):
        sunspot_num = csv_input['sunspot_number'][i]
        if not sunspot_num == -1:
            sunspot_obs_times.append(BG_obs_time_event)
            # Frequency_list = csv_input['Frequency'][i]
            sunspot_num_list.append(sunspot_num)
        else:
            print (BG_obs_time_event)
        # BG = csv_input['BG_40[dB]'][i]
        # BG_40_list.append(BG)

sunspot_obs_times = np.array(sunspot_obs_times)
sunspot_num_list = np.array(sunspot_num_list)

# solar_min_idx = np.where(((sunspot_obs_times >= datetime.datetime(1995, 1, 1)) & (sunspot_obs_times <= datetime.datetime(1997, 12, 31))) | ((sunspot_obs_times >= datetime.datetime(2007, 1, 1)) & (sunspot_obs_times <= datetime.datetime(2009, 12, 31))) | ((sunspot_obs_times >= datetime.datetime(2017, 1, 1)) & (sunspot_obs_times <= datetime.datetime(2020, 12, 31))))[0]
# solar_max_idx = np.where(((sunspot_obs_times >= datetime.datetime(2012, 1, 1)) & (sunspot_obs_times <= datetime.datetime(2014, 12, 31))))[0]

fig = plt.figure(figsize = (14, 4))
top = 50
bottom = 50
# print ('Bottom: '+ str(np.percentile(sunspot_num_list, bottom)))
# print ('Top: '+ str(np.percentile(sunspot_num_list, top)))

# topidx = np.where(sunspot_num_list >= np.percentile(sunspot_num_list, top))[0]
topidx = np.where(sunspot_num_list >= 36)[0]
plt.plot(sunspot_obs_times[topidx], sunspot_num_list[topidx], '.', color = 'k')
# bottomidx = np.where(sunspot_num_list <= np.percentile(sunspot_num_list, bottom))[0]
bottomidx = np.where(sunspot_num_list <= 36)[0]
plt.plot(sunspot_obs_times[bottomidx], sunspot_num_list[bottomidx], '.', color = 'k')
idx = np.where((sunspot_num_list > np.percentile(sunspot_num_list, bottom)) & (sunspot_num_list < np.percentile(sunspot_num_list, top)))[0]
plt.plot(sunspot_obs_times[idx], sunspot_num_list[idx], '.', color = 'k')
plt.axhline(36, ls = "-.", color = "r", label = 'Median value')
print ('filelen top: ' + str(len(sunspot_obs_times[topidx])), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])))
print ('filerate top: ' + str(len(sunspot_obs_times[topidx])/len(sunspot_obs_times)), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])/len(sunspot_obs_times)))
# plt.plot(sunspot_obs_times[solar_min_idx], sunspot_num_list[solar_min_idx], '.', color = 'b', label = 'Around the solar minimum')
# plt.plot(sunspot_obs_times[solar_max_idx], sunspot_num_list[solar_max_idx], '.', color = 'r', label = 'Around the solar maximum')
# plt.axvline(datetime.datetime(2017,1,1), ls = "--", color = "navy")
# plt.axvline(datetime.datetime(2012,1,1), ls = "--", color = "navy")
# plt.axvline(datetime.datetime(2015,1,1), ls = "--", color = "navy")
# plt.axvline(datetime.datetime(2010,1,1), ls = "--", color = "navy")
plt.xlim(datetime.datetime(1995, 1, 1), datetime.datetime(2021, 1, 1))
plt.ylabel('Sunspot number '+' $S_n$', fontsize = 20)
plt.ylim(0,400)
plt.legend(fontsize = 20)
plt.tick_params(labelsize=18)
plt.show()
plt.close()

# bursts_obs_times_micro, bursts_obs_times_od
count = 0
for sunspot_obs_time in sunspot_obs_times[topidx]:
    if ((sunspot_obs_time >= datetime.datetime(2012,1,1)) & (sunspot_obs_time <= datetime.datetime(2015,1,1))):
        if len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Ordinary III maximum '+ str(count))
count = 0
for sunspot_obs_time in sunspot_obs_times[bottomidx]:
    if ((sunspot_obs_time >= datetime.datetime(2017,1,1)) & (sunspot_obs_time <= datetime.datetime(2021,1,1))):
        if len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
    elif ((sunspot_obs_time >= datetime.datetime(2007,1,1)) & (sunspot_obs_time <= datetime.datetime(2010,1,1))):
        if len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Ordinary III minimum '+ str(count))

count = 0
for sunspot_obs_time in sunspot_obs_times[topidx]:
    if ((sunspot_obs_time >= datetime.datetime(2012,1,1)) & (sunspot_obs_time <= datetime.datetime(2015,1,1))):
        if len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Micro III maximum '+ str(count))
count = 0
for sunspot_obs_time in sunspot_obs_times[bottomidx]:
    if ((sunspot_obs_time >= datetime.datetime(2017,1,1)) & (sunspot_obs_time <= datetime.datetime(2021,1,1))):
        if len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
    elif ((sunspot_obs_time >= datetime.datetime(2007,1,1)) & (sunspot_obs_time <= datetime.datetime(2010,1,1))):
        if len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Micro III minimum '+ str(count))





##############################################

# file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'
file_gain = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/hinode_catalog/SN_d_tot_V2.0.csv'

sunspot_obs_times = []
sunspot_num_list = []


print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
    if (BG_obs_time_event >= datetime.datetime(2007, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2021, 1, 1)):
        sunspot_num = csv_input['sunspot_number'][i]
        if not sunspot_num == -1:
            sunspot_obs_times.append(BG_obs_time_event)
            # Frequency_list = csv_input['Frequency'][i]
            sunspot_num_list.append(sunspot_num)
        else:
            print (BG_obs_time_event)
        # BG = csv_input['BG_40[dB]'][i]
        # BG_40_list.append(BG)



fig = plt.figure(figsize = (14, 4))
top_num = 53
bottom_num = 23
print ('Bottom: '+ str(bottom_num))
print ('Top: '+ str(top_num))
sunspot_obs_times = np.array(sunspot_obs_times)
sunspot_num_list = np.array(sunspot_num_list)
topidx = np.where(sunspot_num_list >= top_num)[0]
plt.plot(sunspot_obs_times[topidx], sunspot_num_list[topidx], '.', color = 'r')
bottomidx = np.where(sunspot_num_list <= bottom_num)[0]
plt.plot(sunspot_obs_times[bottomidx], sunspot_num_list[bottomidx], '.', color = 'b')
idx = np.where((sunspot_num_list > bottom_num) & (sunspot_num_list < top_num))[0]
plt.plot(sunspot_obs_times[idx], sunspot_num_list[idx], '.', color = 'k')
print ('filelen top: ' + str(len(sunspot_obs_times[topidx])), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])))
print ('filerate top: ' + str(len(sunspot_obs_times[topidx])/len(sunspot_obs_times)), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])/len(sunspot_obs_times)))
plt.axvline(datetime.datetime(2017,1,1), ls = "--", color = "navy")
plt.axvline(datetime.datetime(2012,1,1), ls = "--", color = "navy")
plt.axvline(datetime.datetime(2015,1,1), ls = "--", color = "navy")
plt.axvline(datetime.datetime(2010,1,1), ls = "--", color = "navy")
plt.show()
plt.close()

# bursts_obs_times_micro, bursts_obs_times_od
count = 0
for sunspot_obs_time in sunspot_obs_times[topidx]:
    if ((sunspot_obs_time >= datetime.datetime(2012,1,1)) & (sunspot_obs_time <= datetime.datetime(2015,1,1))):
        if len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Ordinary III maximum '+ str(count))
count = 0
for sunspot_obs_time in sunspot_obs_times[bottomidx]:
    if ((sunspot_obs_time >= datetime.datetime(2017,1,1)) & (sunspot_obs_time <= datetime.datetime(2021,1,1))):
        if len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
    elif ((sunspot_obs_time >= datetime.datetime(2007,1,1)) & (sunspot_obs_time <= datetime.datetime(2010,1,1))):
        if len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_od >= sunspot_obs_time) & (bursts_obs_times_od <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Ordinary III minimum '+ str(count))

count = 0
for sunspot_obs_time in sunspot_obs_times[topidx]:
    if ((sunspot_obs_time >= datetime.datetime(2012,1,1)) & (sunspot_obs_time <= datetime.datetime(2015,1,1))):
        if len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Micro III maximum '+ str(count))
count = 0
for sunspot_obs_time in sunspot_obs_times[bottomidx]:
    if ((sunspot_obs_time >= datetime.datetime(2017,1,1)) & (sunspot_obs_time <= datetime.datetime(2021,1,1))):
        if len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
    elif ((sunspot_obs_time >= datetime.datetime(2007,1,1)) & (sunspot_obs_time <= datetime.datetime(2010,1,1))):
        if len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])>0:
            count += len(np.where((bursts_obs_times_micro >= sunspot_obs_time) & (bursts_obs_times_micro <= sunspot_obs_time + datetime.timedelta(days=1)))[0])
print ('Micro III minimum '+ str(count))




