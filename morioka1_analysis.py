#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:41:03 2021

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

file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_analysis1/20070101_20201231.csv'

burst_obs_times = []
intensity_list = []
BG_40_list = []

print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=",")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(int(csv_input['Time_list_35MHz'][i].split('-')[0]), int(csv_input['Time_list_35MHz'][i].split('-')[1]), int(csv_input['Time_list_35MHz'][i].split(' ')[0][-2:]), int(csv_input['Time_list_35MHz'][i].split(' ')[1][:2]), int(csv_input['Time_list_35MHz'][i].split(':')[1]), int(csv_input['Time_list_35MHz'][i].split(':')[2][:2]))
    if not BG_obs_time_event in burst_obs_times:
        burst_obs_times.append(BG_obs_time_event)
        # Frequency_list = csv_input['Frequency'][i]
        intensity = np.log10((10**(csv_input['intensity'][i]/10))/2)* 10
        intensity_list.append(intensity)
        # BG = csv_input['BG_40[dB]'][i]
        # BG_40_list.append(BG)
    else:
        print ('yes')

burst_obs_times = np.array(burst_obs_times)
intensity_list = np.array(intensity_list)
BG_40_list = np.array(BG_40_list)

idx = np.where((burst_obs_times >= datetime.datetime(2007,1,1)) & (burst_obs_times <= datetime.datetime(2010,1,1)))[0]
plt.hist(intensity_list[idx], bins = 20)
plt.xlabel('from background[dB]')
plt.yscale('log')
plt.show()
plt.close()

# flux_idxes = np.where((intensity_list >= 20) & (intensity_list <= 22))[0]
flux_idxes = np.where((intensity_list >= 35) & (intensity_list <= 40) & (burst_obs_times >= datetime.datetime(2012,1,1)) & (burst_obs_times <= datetime.datetime(2013,1,1)))[0]
for flux_idx in flux_idxes:
    print (burst_obs_times[flux_idx])
