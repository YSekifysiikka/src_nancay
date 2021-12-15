#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:12:01 2021

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
import datetime
from dateutil.relativedelta import relativedelta
import sys
freq_check = 40
move_ave = 12
move_ave_analysis = 12
move_plot = 4
#赤の線の変動を調べる
analysis_move_ave = 12
average_threshold = 1
error_threshold = 1
solar_maximum = [datetime.datetime(2000, 1, 1), datetime.datetime(2003, 1, 1)]
solar_minimum = [datetime.datetime(2007, 1, 1), datetime.datetime(2010, 1, 1)]
solar_maximum_1 = [datetime.datetime(2012, 1, 1), datetime.datetime(2015, 1, 1)]
solar_minimum_1 = [datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 1)]
analysis_period = [solar_maximum, solar_minimum, solar_maximum_1, solar_minimum_1]

def getNearestValue(list, num):

    # 昇順に挿入する際のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]

def numerical_diff_allen_velocity_fp(factor, r):
    h = 1e-2
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_newkirk_velocity_fp(factor, r):
    h = 1e-2
    
    ne_1 = np.log(factor * 4.2 * 10 ** (4+4.32/((r+h)/69600000000)))
    ne_2 = np.log(factor * 4.2 * 10 ** (4+4.32/((r-h)/69600000000)))
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_allen_velocity_2fp(factor, r):
    h = 1e-2
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_allen(factor, velocity, t, h_start):
    h = 1e-3
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))

def func(x, a, b):
    return a * x + b

labelsize = 18
fontsize = 20
factor_velocity = 1
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

import csv







# burst_types = ['storm'] 
# csv_names = ['shuron_LL_RR_micro_dB_cycle23.csv']
with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_ordinary_LL_RR_withcycle22.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list", "peak_RR_40MHz", "peak_LL_40MHz", "drift_rate_40MHz", "sunspots_num"])
    w.writeheader()
    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_LL_RR_flare_associated_ordinary_dB_cycle23.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

    for j in range(len(csv_input_final)):
        obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
        freq_start = csv_input_final["freq_start"][j]

        freq_end = csv_input_final["freq_end"][j]
        event_start = csv_input_final["event_start"][j]
        event_end = csv_input_final["event_end"][j]
        slope = csv_input_final["drift_rate_40MHz"][j]
        sunspots_num = csv_input_final["sunspots_num"][j]
        time_rate_final = csv_input_final["velocity"][j]
        residual_list = csv_input_final["residual"][j]
        best_factor = csv_input_final["factor"][j]
        time_list = csv_input_final["peak_time_list"][j]
        freq_list = csv_input_final["peak_freq_list"][j]

        freq_list = csv_input_final["peak_freq_list"][j]
        w.writerow({'obs_time':obs_time,'velocity':time_rate_final, 'residual':residual_list, 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':best_factor, 'peak_time_list':time_list, 'peak_freq_list':freq_list, "drift_rate_40MHz":slope, "sunspots_num":sunspots_num})





    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_ordinary_LL_RR_withcycle22.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    for j in range(len(csv_input_final)):
        obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
        freq_start = csv_input_final["freq_start"][j]

        freq_end = csv_input_final["freq_end"][j]
        event_start = csv_input_final["event_start"][j]
        event_end = csv_input_final["event_end"][j]
        if (freq_start >= 40) & (freq_end <= 40):
            slope = csv_input_final["drift_rate_40MHz"][j]
        else:
            slope = np.nan
        sunspots_num = csv_input_final["sunspots_num"][j]
        time_rate_final = csv_input_final["velocity"][j]
        residual_list = csv_input_final["residual"][j]
        best_factor = csv_input_final["factor"][j]
        time_list = csv_input_final["peak_time_list"][j]
        freq_list = csv_input_final["peak_freq_list"][j]
        RR_peak = csv_input_final["peak_RR_40MHz"][j]
        LL_peak = csv_input_final["peak_LL_40MHz"][j]
        freq_list = csv_input_final["peak_freq_list"][j]
        
        w.writerow({'obs_time':obs_time,'velocity':time_rate_final, 'residual':residual_list, 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':best_factor, 'peak_time_list':time_list, 'peak_freq_list':freq_list, "peak_RR_40MHz":RR_peak, "peak_LL_40MHz":LL_peak, "drift_rate_40MHz":slope, "sunspots_num":sunspots_num})
    
    
    
    
    
