#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:50:51 2020

@author: yuichiro
"""

#Newversion
# file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/ver2_30dB_40dB_gain_analysis.csv'
# file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20120101_20141231.csv'
# file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20170101_20201231.csv'
# file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import glob
import cdflib
import os
import sys
import math

import astropy.time
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation

def solar_cos_list(obs_datetime):
    koko = EarthLocation(lat='47 22 24.00',lon='2 11 50.00', height = '150')
    obs_time = obs_datetime
    toki = astropy.time.Time(obs_time)
    taiyous = get_sun(toki).transform_to(AltAz(obstime=toki,location=koko))
    # print(taiyou)
    # print(taiyou.az) # 天球での方位角
    # print(taiyou.alt) # 天球での仰俯角
    # print(taiyou.distance) # 距離
    # print(taiyou.distance.au) # au単位での距離
    cos_list = []
    for taiyou in taiyous:
        azimuth = float(str(taiyou.az).split('d')[0] + '.' + str(taiyou.az).split('d')[1].split('m')[0])
        altitude = float(str(taiyou.alt).split('d')[0] + '.' + str(taiyou.alt).split('d')[1].split('m')[0])
        # print (azimuth, altitude)
        
        solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
                                math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
                                math.sin(math.radians(altitude))])
        machine_place = np.array([math.cos(math.radians(70)) * math.cos(math.radians(180)),
                                math.cos(math.radians(70)) * math.sin(math.radians(180)),
                                math.sin(math.radians(70))])
        cos_list.append(np.dot(solar_place, machine_place))
    return cos_list


def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]

def plot_arange_gain(ax_list, ymin, ymax):
    for ax in ax_list:
        if plot_days == 1:
            Minute_fmt = mdates.DateFormatter('%H:%M')  
            ax.xaxis.set_major_formatter(Minute_fmt)
            ax.set_xlim(gain_obs_time[full_idxes_gain][0] - datetime.timedelta(minutes=5), gain_obs_time[full_idxes_gain][-1]+ datetime.timedelta(minutes=5))
            ax.set_ylim(ymin*0.95, ymax*1.05)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                            labelright=False,
                            labeltop=False)
        else:
            fmt = mdates.DateFormatter('%m/%d %H') 
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlim(gain_obs_time[full_idxes_gain][0] - datetime.timedelta(minutes=45), gain_obs_time[full_idxes_gain][-1]+ datetime.timedelta(minutes=45))
            ax.set_ylim(ymin*0.95, ymax*1.05)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                            labelright=False,
                            labeltop=False)
    return



# def plot_arange_dB_year(ax_list, idxes):
#     for ax in ax_list:
#         ax.set_xlim(obs_time[idxes][0] - datetime.timedelta(days=1), obs_time[idxes][-1]+ datetime.timedelta(days=1))
#         ax.set_ylim(ymin-0.5, ymax+0.5)
#         ax.legend(fontsize = 12)
#         ax.tick_params(labelbottom=False,
#                         labelright=False,
#                         labeltop=False)
#     return
def plot_arange_dB_day(ax_list, ymin, ymax):
    for ax in ax_list:
        if plot_days == 1:
            Minute_fmt = mdates.DateFormatter('%H:%M')  
            ax.xaxis.set_major_formatter(Minute_fmt)
            ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=5), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=5))
            ax.set_ylim(ymin-0.5, ymax+0.5)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                            labelright=False,
                            labeltop=False)
        else:
            fmt = mdates.DateFormatter('%m/%d %H') 
            ax.xaxis.set_major_formatter(fmt)
            ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=45), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=45))
            ax.set_ylim(ymin-0.5, ymax+0.5)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                            labelright=False,
                            labeltop=False)
    return

def check_BG(date, HH, MM, event_check_days):
    event_date = str(date)
    yyyy = event_date[:4]
    MM = event_date[4:6]
    dd = event_date[6:8]
    hh = HH
    mm = MM
    select_date = datetime.datetime(int(yyyy),int(MM),int(dd),int(hh),int(mm))
    check_decibel_l = []
    check_decibel_r = []
    check_obs_time = []
    if event_check_days == 1:
        start_date = select_date
        for i in range(1):
            check_date = start_date
            obs_index = np.where(BG_obs_time == getNearestValue(BG_obs_time,check_date))[0][0]
            if abs(BG_obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel_l.append(decibel_list_l[obs_index])
                check_decibel_r.append(decibel_list_r[obs_index])
                check_obs_time.append(BG_obs_time[obs_index])
    else:
        start_date = select_date - datetime.timedelta(days=event_check_days/2)
        for i in range(event_check_days+1):
            check_date = start_date + datetime.timedelta(days=i)
            obs_index = np.where(BG_obs_time == getNearestValue(BG_obs_time,check_date))[0][0]
            if abs(BG_obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel_l.append(decibel_list_l[obs_index])
                check_decibel_r.append(decibel_list_r[obs_index])
                check_obs_time.append(BG_obs_time[obs_index])

    check_decibel_l = np.array(check_decibel_l)
    check_decibel_r = np.array(check_decibel_r)
    check_obs_time = np.array(check_obs_time)

    return np.percentile(check_decibel_l, 12.5, axis = 0), np.percentile(check_decibel_r, 12.5, axis = 0)

# def cali_BG:



Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = '/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_antenna/RR_LL_all_freq_under5_20070101_20201231.csv'
# file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_final.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

BG_obs_times = []
decibel_list_r = []
decibel_list_l = []
for i in range(len(antenna1_csv)):
    BG_obs_times.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    # decibel_list.append(antenna1_csv['decibel'][i])
    RR = antenna1_csv['BG_RR'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list_r.append([float(s) for s in RR if s != ''])
    LL = antenna1_csv['BG_LL'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list_l.append([float(s) for s in LL if s != ''])
    # for j in range(len(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' '))):
    #     if not antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j] == '':
    #         decibel_list.append(float(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j]))

BG_obs_times = np.array(BG_obs_times)
decibel_list_r = np.array(decibel_list_r)
decibel_list_l = np.array(decibel_list_l)



file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain/new_ver2_gain_analysis_20070101_20201231.csv'

gain_obs_times = []
Frequency = []
gain_RR = []
Trx_RR = []
hot_dB_RR = []
cold_dB_RR = []
gain_LL = []
Trx_LL = []
hot_dB_LL = []
cold_dB_LL = []
print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=",")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    gain_obs_time_event = datetime.datetime(int(csv_input['obs_time'][i].split('-')[0]), int(csv_input['obs_time'][i].split('-')[1]), int(csv_input['obs_time'][i].split(' ')[0][-2:]), int(csv_input['obs_time'][i].split(' ')[1][:2]), int(csv_input['obs_time'][i].split(':')[1]), int(csv_input['obs_time'][i].split(':')[2][:2]))
    gain_obs_times.append(gain_obs_time_event)
    # Frequency_list = csv_input['Frequency'][i]
    Frequency.append([float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    gain_RR.append([float(k) for k in csv_input['Right-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    Trx_RR.append([float(k) for k in csv_input['Right-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    hot_dB_RR.append([float(k) for k in csv_input['Right-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    cold_dB_RR.append([float(k) for k in csv_input['Right-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    # Frequency.append([float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    gain_LL.append([float(k) for k in csv_input['Left-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    Trx_LL.append([float(k) for k in csv_input['Left-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    hot_dB_LL.append([float(k) for k in csv_input['Left-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    cold_dB_LL.append([float(k) for k in csv_input['Left-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])

gain_obs_times = np.array(gain_obs_times)
Frequency = np.array(Frequency)
gain_RR = np.array(gain_RR)
Trx_RR = np.array(Trx_RR)
hot_dB_RR = np.array(hot_dB_RR)
cold_dB_RR = np.array(cold_dB_RR)
gain_LL = np.array(gain_LL)
Trx_LL = np.array(Trx_LL)
hot_dB_LL = np.array(hot_dB_LL)
cold_dB_LL = np.array(cold_dB_LL)

import csv

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_BG/RR_LL_gain_BG_analysis.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "Frequency", "Right_BG", "Left-BG", "Right-gain", "Right-Trx", "Right-hot_dB", "Right-cold_dB", "Left-gain", "Left-Trx", "Left-hot_dB", "Left-cold_dB"])
    w.writeheader()
    for i in range(len(BG_obs_times)):
        if np.abs(BG_obs_times[i] - getNearestValue(gain_obs_times, BG_obs_times[i])) <= datetime.timedelta(seconds=3600):
            gain_idx = np.where(gain_obs_times==gain_obs_times[0])[0][0]
            
    
    
    
    
    
    