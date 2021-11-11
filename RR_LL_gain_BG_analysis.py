#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:50:51 2020

@author: yuichiro
"""

#Newversion
# file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
# file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/ver2_30dB_40dB_gain_analysis.csv'
# file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20120101_20141231.csv'
# file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20170101_20201231.csv'
# file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'
#最初はBGは移動平均で下位1/8のみ
#現在はBGは2σを超えるものは除去、その後median値使用

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
import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
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



def check_BG(data_RR, data_LL, BG_obs_times_selected, event_check_days):
    check_decibel_all_l = []
    check_decibel_all_r = []
    check_obs_time_list = []
    for BG_obs_time_selected in BG_obs_times_selected:
        check_decibel_l = []
        check_decibel_r = []
        check_obs_time = []
        if event_check_days == 1:
            start_date = BG_obs_time_selected
            for i in range(1):
                obs_index = np.where(BG_obs_times_selected == getNearestValue(BG_obs_times_selected,BG_obs_time_selected))[0][0]
                if abs(BG_obs_times_selected[obs_index] - BG_obs_time_selected) <= datetime.timedelta(seconds=60*90):
                    check_decibel_l.append(data_LL[obs_index])
                    check_decibel_r.append(data_RR[obs_index])
                    check_obs_time.append(BG_obs_times_selected[obs_index])
        else:
            start_date = BG_obs_time_selected - datetime.timedelta(days=event_check_days/2)
            for i in range(event_check_days+1):
                check_date = start_date + datetime.timedelta(days=i)
                obs_index = np.where(BG_obs_times_selected == getNearestValue(BG_obs_times_selected,check_date))[0][0]
                if abs(BG_obs_times_selected[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                    check_decibel_l.append(data_LL[obs_index])
                    check_decibel_r.append(data_RR[obs_index])
                    check_obs_time.append(BG_obs_times_selected[obs_index])
        if len(check_decibel_l) >= 15:
            check_decibel_l = np.array(check_decibel_l)
            check_decibel_r = np.array(check_decibel_r)
            # check_obs_time = np.array(check_obs_time)
            check_obs_time_list.append(BG_obs_time_selected)
            # check_decibel_all_l.append(np.percentile(check_decibel_l, 12.5))
            # check_decibel_all_r.append(np.percentile(check_decibel_r, 12.5))
            check_decibel_all_l.append(np.nanmedian(check_decibel_l))
            check_decibel_all_r.append(np.nanmedian(check_decibel_r))
        else:
            # delete_date.append(BG_obs_time_selected)
            pass
    return np.array(check_decibel_all_r), np.array(check_decibel_all_l), np.array(check_obs_time_list)




file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_BG/under25_RR_LL_gain_BG_analysis_calibrated_gainfixed.csv'

BG_obs_times = []
decibel_list_r = []
decibel_list_l = []
cali_decibel_list_r = []
cali_decibel_list_l = []
Frequency = []
gain_RR = []
Trx_RR = []
hot_dB_RR = []
cold_dB_RR = []
gain_LL = []
Trx_LL = []
hot_dB_LL = []
cold_dB_LL = []
Used_db_median = []
print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=",")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(int(csv_input['obs_time'][i].split('-')[0]), int(csv_input['obs_time'][i].split('-')[1]), int(csv_input['obs_time'][i].split(' ')[0][-2:]), int(csv_input['obs_time'][i].split(' ')[1][:2]), int(csv_input['obs_time'][i].split(':')[1]), int(csv_input['obs_time'][i].split(':')[2][:2]))
    BG_obs_times.append(BG_obs_time_event)
    # Frequency_list = csv_input['Frequency'][i]
    cali_RR = csv_input['Right-BG_Calibrated'][i].replace('\n', '')[1:-1].split(' ')
    cali_decibel_list_r.append([float(s) for s in cali_RR if s != ''])
    cali_LL = csv_input['Left-BG_Calibrated'][i].replace('\n', '')[1:-1].split(' ')
    cali_decibel_list_l.append([float(s) for s in cali_LL if s != ''])
    RR = csv_input['Right-BG'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list_r.append([float(s) for s in RR if s != ''])
    LL = csv_input['Left-BG'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list_l.append([float(s) for s in LL if s != ''])

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
    Used_db_median.append([float(k) for k in csv_input['Used_dB_median'][i][1:-1].replace('\n', '').split(' ') if k != ''])

BG_obs_times = np.array(BG_obs_times)
BG_r = np.array(decibel_list_r)
BG_l = np.array(decibel_list_l)
cali_BG_r = np.array(cali_decibel_list_r)
cali_BG_l = np.array(cali_decibel_list_l)
Frequency = np.array(Frequency)
gain_RR = np.array(gain_RR)
Trx_RR = np.array(Trx_RR)
hot_dB_RR = np.array(hot_dB_RR)
cold_dB_RR = np.array(cold_dB_RR)
gain_LL = np.array(gain_LL)
Trx_LL = np.array(Trx_LL)
hot_dB_LL = np.array(hot_dB_LL)
cold_dB_LL = np.array(cold_dB_LL)
Used_db_median = np.array(Used_db_median)



idxes_BG_2007_20080701 = np.where((BG_obs_times>datetime.datetime(int(2007),int(1),int(1)))&(BG_obs_times<datetime.datetime(int(2008),int(7),int(1))))[0]
idxes_BG_20080808_2009 = np.where((BG_obs_times>datetime.datetime(int(2008),int(8),int(8)))&(BG_obs_times<datetime.datetime(int(2010),int(1),int(1))))[0]
idxes_BG_2012_2014 = np.where((BG_obs_times>datetime.datetime(int(2012),int(1),int(1)))&(BG_obs_times<datetime.datetime(int(2015),int(1),int(1))))[0]
idxes_BG_2017_20171101 = np.where((BG_obs_times>datetime.datetime(int(2017),int(1),int(1)))&(BG_obs_times<datetime.datetime(int(2017),int(11),int(1))))[0]
idxes_BG_20171101_2020 = np.where((BG_obs_times>datetime.datetime(int(2017),int(11),int(1)))&(BG_obs_times<datetime.datetime(int(2021),int(1),int(1))))[0]
idxes_BG_selected = np.hstack((idxes_BG_2007_20080701, idxes_BG_20080808_2009, idxes_BG_2012_2014, idxes_BG_2017_20171101, idxes_BG_20171101_2020))

Frequency_setting = 40
fontsize_title = 20
fontsize_axis = 20
fmt = mdates.DateFormatter('%Y-%m-%d') 

fig = plt.figure(figsize=(18.0, 12.0))
gs = gridspec.GridSpec(260, 2)
ax_r = plt.subplot(gs[0:40, :1])
ax_l = plt.subplot(gs[0:40, 1:])
ax1_r = plt.subplot(gs[55:95, :1])
ax1_l = plt.subplot(gs[55:95, 1:])
ax2_r = plt.subplot(gs[110:150, :1])
ax2_l = plt.subplot(gs[110:150, 1:])
ax3_r = plt.subplot(gs[165:205, :1])
ax3_l = plt.subplot(gs[165:205, 1:])
ax4_r = plt.subplot(gs[220:260, :1])
ax4_l = plt.subplot(gs[220:260, 1:])


for idxes_BG in [idxes_BG_2007_20080701, idxes_BG_20080808_2009, idxes_BG_2012_2014, idxes_BG_2017_20171101, idxes_BG_20171101_2020]:
    if np.all(np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1] == np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]):
        idx_freq = np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]
        

        ax_r.set_title('BG_RR (Calibrated)', fontsize = fontsize_title)
        ax_r.plot(BG_obs_times[idxes_BG], cali_BG_r[idxes_BG,idx_freq], color = 'b')   
        # ax_r.xaxis.set_major_formatter(fmt)
        # xticklabels = ax_r.get_xticklabels()
        # yticklabels = ax_r.get_yticklabels()
        # ax_r.set_xticklabels(xticklabels,fontsize= fontsize_axis, rotation=45)
        # ax_r.set_xticklabels(yticklabels, fontsize= fontsize_axis, rotation=45)

        ax_l.set_title('BG_LL (Calibrated)', fontsize = fontsize_title)
        ax_l.plot(BG_obs_times[idxes_BG], cali_BG_l[idxes_BG,idx_freq], color = 'b')
        # ax_l.xaxis.set_major_formatter(fmt)
        # xticklabels = ax_l.get_xticklabels()
        # yticklabels = ax_l.get_yticklabels()
        # ax_l.set_xticklabels(xticklabels,fontsize= fontsize_axis, rotation=45)
        # ax_l.set_xticklabels(yticklabels,fontsize= fontsize_axis, rotation=45)

        ax1_r.set_title('BG_RR', fontsize = fontsize_title)
        ax1_r.plot(BG_obs_times[idxes_BG], BG_r[idxes_BG,idx_freq], color = 'b')

        ax1_l.set_title('BG_LL', fontsize = fontsize_title)
        ax1_l.plot(BG_obs_times[idxes_BG], BG_l[idxes_BG,idx_freq], color = 'b')

        ax2_r.set_title('Gain_RR', fontsize = fontsize_title)
        ax2_r.plot(BG_obs_times[idxes_BG], gain_RR[idxes_BG,idx_freq], color = 'b')

        ax2_l.set_title('Gain_LL', fontsize = fontsize_title)
        ax2_l.plot(BG_obs_times[idxes_BG], gain_LL[idxes_BG,idx_freq], color = 'b')

        ax3_r.set_title('Trx_RR', fontsize = fontsize_title)
        ax3_r.plot(BG_obs_times[idxes_BG], Trx_RR[idxes_BG,idx_freq], color = 'b')

        ax3_l.set_title('Trx_LL', fontsize = fontsize_title)
        ax3_l.plot(BG_obs_times[idxes_BG], Trx_LL[idxes_BG,idx_freq], color = 'b')

        ax4_r.set_title('Gain_RR/Gain_LL', fontsize = fontsize_title)
        ax4_r.plot(BG_obs_times[idxes_BG], gain_RR[idxes_BG,idx_freq]/ gain_LL[idxes_BG,idx_freq], color = 'b')

        ax4_l.set_title('Trx_RR/Trx_LL', fontsize = fontsize_title)
        ax4_l.plot(BG_obs_times[idxes_BG], Trx_RR[idxes_BG,idx_freq]/Trx_LL[idxes_BG,idx_freq], color = 'b')

# print ('a')
plt.show()
plt.close()


# idxes_BG_selected = np.hstack((idxes_BG_2007_20080701, idxes_BG_20080808_2009, idxes_BG_2012_2014, idxes_BG_2017_20171101, idxes_BG_20171101_2020))
Frequency_setting = 40
fontsize_title = 20
fontsize_axis = 20
fmt = mdates.DateFormatter('%Y-%m-%d') 

fig = plt.figure(figsize=(18.0, 12.0))
gs = gridspec.GridSpec(260, 2)
ax_r = plt.subplot(gs[0:40, :1])
ax_l = plt.subplot(gs[0:40, 1:])
ax1_r = plt.subplot(gs[55:95, :1])
ax1_l = plt.subplot(gs[55:95, 1:])
ax2_r = plt.subplot(gs[110:150, :1])
ax2_l = plt.subplot(gs[110:150, 1:])
ax3_r = plt.subplot(gs[165:205, :1])
ax3_l = plt.subplot(gs[165:205, 1:])
ax4_r = plt.subplot(gs[220:260, :1])
ax4_l = plt.subplot(gs[220:260, 1:])


for idxes_BG in [idxes_BG_2007_20080701, idxes_BG_20080808_2009, idxes_BG_2012_2014, idxes_BG_2017_20171101, idxes_BG_20171101_2020]:
    if np.all(np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1] == np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]):
        idx_freq = np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]
        mean_l = np.nanmean(cali_BG_l[idxes_BG,idx_freq])
        std_l = np.nanstd(cali_BG_l[idxes_BG,idx_freq])
        mean_r = np.nanmean(cali_BG_r[idxes_BG,idx_freq])
        std_r = np.nanstd(cali_BG_r[idxes_BG,idx_freq])
        threshold_l = [mean_l - 2*std_l, mean_r + 2*std_r]
        threshold_r = [mean_r - 2*std_r, mean_l + 2*std_l]
        
        idx_select_r = np.where((cali_BG_r[:,idx_freq]>=threshold_r[0])&(cali_BG_r[:,idx_freq]<=threshold_r[1]))[0]
        idx_select_l = np.where((cali_BG_l[:,idx_freq]>=threshold_l[0])&(cali_BG_l[:,idx_freq]<=threshold_l[1]))[0]


        ax_r.set_title('BG_RR (Calibrated)', fontsize = fontsize_title)
        ax_r.plot(BG_obs_times[idxes_BG], cali_BG_r[idxes_BG,idx_freq], color = 'b')   
        # ax_r.xaxis.set_major_formatter(fmt)
        # xticklabels = ax_r.get_xticklabels()
        # yticklabels = ax_r.get_yticklabels()
        # ax_r.set_xticklabels(xticklabels,fontsize= fontsize_axis, rotation=45)
        # ax_r.set_xticklabels(yticklabels, fontsize= fontsize_axis, rotation=45)

        ax_l.set_title('BG_LL (Calibrated)', fontsize = fontsize_title)
        ax_l.plot(BG_obs_times[idxes_BG], cali_BG_l[idxes_BG,idx_freq], color = 'b')
        # ax_l.xaxis.set_major_formatter(fmt)
        # xticklabels = ax_l.get_xticklabels()
        # yticklabels = ax_l.get_yticklabels()
        # ax_l.set_xticklabels(xticklabels,fontsize= fontsize_axis, rotation=45)
        # ax_l.set_xticklabels(yticklabels,fontsize= fontsize_axis, rotation=45)

        ax1_r.set_title('BG_RR', fontsize = fontsize_title)
        ax1_r.plot(BG_obs_times[idxes_BG], BG_r[idxes_BG,idx_freq], color = 'b')

        ax1_l.set_title('BG_LL', fontsize = fontsize_title)
        ax1_l.plot(BG_obs_times[idxes_BG], BG_l[idxes_BG,idx_freq], color = 'b')

        ax2_r.set_title('Gain_RR', fontsize = fontsize_title)
        ax2_r.plot(BG_obs_times[idxes_BG], gain_RR[idxes_BG,idx_freq], color = 'b')

        ax2_l.set_title('Gain_LL', fontsize = fontsize_title)
        ax2_l.plot(BG_obs_times[idxes_BG], gain_LL[idxes_BG,idx_freq], color = 'b')

        ax3_r.set_title('Trx_RR', fontsize = fontsize_title)
        ax3_r.plot(BG_obs_times[idxes_BG], Trx_RR[idxes_BG,idx_freq], color = 'b')

        ax3_l.set_title('Trx_LL', fontsize = fontsize_title)
        ax3_l.plot(BG_obs_times[idxes_BG], Trx_LL[idxes_BG,idx_freq], color = 'b')

        ax4_r.set_title('Gain_RR/Gain_LL', fontsize = fontsize_title)
        ax4_r.plot(BG_obs_times[idxes_BG], gain_RR[idxes_BG,idx_freq]/ gain_LL[idxes_BG,idx_freq], color = 'b')

        ax4_l.set_title('Trx_RR/Trx_LL', fontsize = fontsize_title)
        ax4_l.plot(BG_obs_times[idxes_BG], Trx_RR[idxes_BG,idx_freq]/Trx_LL[idxes_BG,idx_freq], color = 'b')

# print ('a')
plt.show()
plt.close()



# idxes_BG_2007_2009_selected = np.where((idxes_BG_selected>datetime.datetime(int(2007),int(1),int(1)))&(idxes_BG_selected<datetime.datetime(int(2010),int(1),int(1))))[0]
# idxes_BG_2012_2014_selected = np.where((idxes_BG_selected>datetime.datetime(int(2012),int(1),int(1)))&(idxes_BG_selected<datetime.datetime(int(2015),int(1),int(1))))[0]
# idxes_BG_2017_2020_selected = np.where((idxes_BG_selected>datetime.datetime(int(2017),int(1),int(1)))&(idxes_BG_selected<datetime.datetime(int(2021),int(1),int(1))))[0]



event_check_days = 30
Frequency_setting = 37.5
fontsize_title = 20
fontsize_axis = 20
fontsize_ylabel = 16

fmt = mdates.DateFormatter('%Y-%m-%d') 

# fig = plt.figure(figsize=(18.0, 12.0))
# gs = gridspec.GridSpec(260, 2)
# ax_r = plt.subplot(gs[0:40, :1])
# ax_l = plt.subplot(gs[0:40, 1:])
# ax1_r = plt.subplot(gs[55:95, :1])
# ax1_l = plt.subplot(gs[55:95, 1:])
# ax2_r = plt.subplot(gs[110:150, :1])
# ax2_l = plt.subplot(gs[110:150, 1:])
# ax3_r = plt.subplot(gs[165:205, :1])
# ax3_l = plt.subplot(gs[165:205, 1:])
# ax4_r = plt.subplot(gs[220:260, :1])
# ax4_l = plt.subplot(gs[220:260, 1:])

# for idxes_BG in [idxes_BG_2007_20080701, idxes_BG_20080808_2009, idxes_BG_2012_2014, idxes_BG_2017_20171101, idxes_BG_20171101_2020]:

# for i in [2007,2008,2009,2012,2013,2014,2017,2018,2019,2020]:
#     for j in range(4):
#         idxes_BG = np.where((BG_obs_times>datetime.datetime(i,int(j+1),int(1),0,0))&(BG_obs_times<datetime.datetime(i,int(j+1),1,0,0)+datetime.timedelta(days=10)))[0]
#         if np.all(np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1] == np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]):
#             fig = plt.figure(figsize=(18.0, 12.0))
#             gs = gridspec.GridSpec(300, 2)
#             ax_r = plt.subplot(gs[0:40, :1])
#             ax_l = plt.subplot(gs[0:40, 1:])
#             ax1_r = plt.subplot(gs[65:105, :1])
#             ax1_l = plt.subplot(gs[65:105, 1:])
#             ax2_r = plt.subplot(gs[130:170, :1])
#             ax2_l = plt.subplot(gs[130:170, 1:])
#             ax3_r = plt.subplot(gs[195:235, :1])
#             ax3_l = plt.subplot(gs[195:235, 1:])
#             ax4_r = plt.subplot(gs[260:300, :1])
#             ax4_l = plt.subplot(gs[260:300, 1:])
#             idx_freq = np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]
            
#             idx_nonTrx_RR = np.where((Trx_RR[idxes_BG,idx_freq] >= 9000) | (Trx_RR[idxes_BG,idx_freq] <= 0))[0]
#             idx_nonTrx_LL = np.where((Trx_LL[idxes_BG,idx_freq] >= 9000) | (Trx_LL[idxes_BG,idx_freq] <= 0))[0]
#             idx_nonTrx_LL_RR = np.where(((Trx_LL[idxes_BG,idx_freq]/Trx_RR[idxes_BG,idx_freq]) >= 4) |((Trx_LL[idxes_BG,idx_freq]/Trx_RR[idxes_BG,idx_freq]) <= 0.25))[0]
#             idx_nongain_LL_RR = np.where((gain_LL[idxes_BG,idx_freq]/gain_RR[idxes_BG,idx_freq] <= 0.5) | (gain_LL[idxes_BG,idx_freq]/gain_RR[idxes_BG,idx_freq] >= 2))[0]
#             idx_nonTrx = np.unique(np.concatenate([idx_nonTrx_RR, idx_nonTrx_LL, idx_nonTrx_LL_RR, idx_nongain_LL_RR], 0))
            
            
#             BG_obs_times_selected = np.delete(BG_obs_times[idxes_BG], idx_nonTrx, 0)
#             Trx_RR_selected = np.delete(Trx_RR[idxes_BG], idx_nonTrx, 0)
#             BG_r_selected = np.delete(BG_r[idxes_BG], idx_nonTrx, 0)
#             cali_BG_r_selected = np.delete(cali_BG_r[idxes_BG], idx_nonTrx, 0)
#             gain_RR_selected = np.delete(gain_RR[idxes_BG], idx_nonTrx, 0)
            
            
#             Trx_LL_selected = np.delete(Trx_LL[idxes_BG], idx_nonTrx, 0)
#             BG_l_selected = np.delete(BG_l[idxes_BG], idx_nonTrx, 0)
#             cali_BG_l_selected = np.delete(cali_BG_l[idxes_BG], idx_nonTrx, 0)
#             gain_LL_selected = np.delete(gain_LL[idxes_BG], idx_nonTrx, 0)
    
#             ax_r.set_title('BG_RR (Calibrated)', fontsize = fontsize_title)
#             ax_r.plot(BG_obs_times_selected, cali_BG_r_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')   
#             ax_r.set_ylabel('BG [dB]', fontsize = fontsize_ylabel)
#             # ax_r.set_ylim(15,30)
#             ax_r.tick_params(axis='x', labelrotation=15)
#             # ax_r.xaxis.set_major_formatter(fmt)
#             # xticklabels = ax_r.get_xticklabels()
#             # yticklabels = ax_r.get_yticklabels()
#             # ax_r.set_xticklabels(xticklabels,fontsize= fontsize_axis, rotation=45)
#             # ax_r.set_xticklabels(yticklabels, fontsize= fontsize_axis, rotation=45)
    
#             ax_l.set_title('BG_LL (Calibrated)', fontsize = fontsize_title)
#             ax_l.plot(BG_obs_times_selected, cali_BG_l_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax_l.set_ylabel('BG [dB]', fontsize = fontsize_ylabel)
#             # ax_l.set_ylim(15,30)
#             ax_l.tick_params(axis='x', labelrotation=15)
#             # ax_l.xaxis.set_major_formatter(fmt)
#             # xticklabels = ax_l.get_xticklabels()
#             # yticklabels = ax_l.get_yticklabels()
#             # ax_l.set_xticklabels(xticklabels,fontsize= fontsize_axis, rotation=45)
#             # ax_l.set_xticklabels(yticklabels,fontsize= fontsize_axis, rotation=45)
    
#             ax1_r.set_title('BG_RR', fontsize = fontsize_title)
#             ax1_r.plot(BG_obs_times_selected, BG_r_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax1_r.set_ylabel('BG [dB]', fontsize = fontsize_ylabel)
#             ax1_r.tick_params(axis='x', labelrotation=15)
#             # ax1_r.tick_params(axis='x', labelrotation=45)
    
#             ax1_l.set_title('BG_LL', fontsize = fontsize_title)
#             ax1_l.plot(BG_obs_times_selected, BG_l_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax1_l.set_ylabel('BG [dB]', fontsize = fontsize_ylabel)
#             ax1_l.tick_params(axis='x', labelrotation=15)
    
#             ax2_r.set_title('Gain_RR', fontsize = fontsize_title)
#             ax2_r.plot(BG_obs_times_selected, gain_RR_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax2_r.set_ylabel('Gain', fontsize = fontsize_ylabel)
#             ax2_r.tick_params(axis='x', labelrotation=15)
    
#             ax2_l.set_title('Gain_LL', fontsize = fontsize_title)
#             ax2_l.plot(BG_obs_times_selected, gain_LL_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax2_l.set_ylabel('Gain', fontsize = fontsize_ylabel)
#             ax2_l.tick_params(axis='x', labelrotation=15)
    
#             ax3_r.set_title('Trx_RR', fontsize = fontsize_title)
#             ax3_r.plot(BG_obs_times_selected, Trx_RR_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax3_r.set_ylabel('Temperature [K]', fontsize = fontsize_ylabel)
#             ax3_r.tick_params(axis='x', labelrotation=15)
    
#             ax3_l.set_title('Trx_LL', fontsize = fontsize_title)
#             ax3_l.plot(BG_obs_times_selected, Trx_LL_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax3_l.set_ylabel('Temperature [K]', fontsize = fontsize_ylabel)
#             ax3_l.tick_params(axis='x', labelrotation=15)
    
#             ax4_r.set_title('Gain_RR/Gain_LL', fontsize = fontsize_title)
#             ax4_r.plot(BG_obs_times_selected, gain_RR_selected[:,idx_freq]/ gain_LL_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax4_r.set_ylabel('Ratio', fontsize = fontsize_ylabel)
#             ax4_r.tick_params(axis='x', labelrotation=15)
    
#             ax4_l.set_title('Trx_RR/Trx_LL', fontsize = fontsize_title)
#             ax4_l.plot(BG_obs_times_selected, Trx_RR_selected[:,idx_freq]/Trx_LL_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
#             ax4_l.set_ylabel('Ratio', fontsize = fontsize_ylabel)
#             ax4_l.tick_params(axis='x', labelrotation=15)
    
#             # move_cali_BG_r_selected, move_cali_BG_l_selected = check_BG(cali_BG_r_selected[:,idx_freq], cali_BG_l_selected[:,idx_freq], BG_obs_times_selected, event_check_days)
    
#             print ('a')
#             plt.show()
#             plt.close()

# idxes_BG_2007_20080701 = np.where((BG_obs_times>datetime.datetime(int(2007),int(1),int(1)))&(BG_obs_times<datetime.datetime(int(2008),int(7),int(1))))[0]
# idxes_BG_20080808_2009 = np.where((BG_obs_times>datetime.datetime(int(2008),int(8),int(8)))&(BG_obs_times<datetime.datetime(int(2010),int(1),int(1))))[0]
# idxes_BG_2012_2014 = np.where((BG_obs_times>datetime.datetime(int(2012),int(1),int(1)))&(BG_obs_times<datetime.datetime(int(2015),int(1),int(1))))[0]
# idxes_BG_2017_20171101 = np.where((BG_obs_times>datetime.datetime(int(2017),int(1),int(1)))&(BG_obs_times<datetime.datetime(int(2017),int(11),int(1))))[0]
# idxes_BG_20171101_2020 = np.where((BG_obs_times>datetime.datetime(int(2017),int(11),int(1)))&(BG_obs_times<datetime.datetime(int(2021),int(1),int(1))))[0]



if Frequency_setting == 30:
    dir_name = str(Frequency_setting)+'MHz'
    dir_name_1 = str(Frequency_setting)+'MHz'
elif Frequency_setting == 35:
    dir_name = str(Frequency_setting)+'MHz'
    dir_name_1 = str(Frequency_setting)+'MHz'
elif Frequency_setting == 40:
    dir_name = str(Frequency_setting)+'MHz'
    dir_name_1 = str(Frequency_setting)+'MHz'
elif Frequency_setting == 32.5:
    dir_name = str(32.5)+'MHz'
    dir_name_1 = str(32_5)+'MHz'
elif Frequency_setting == 37.5:
    dir_name = str(37.5)+'MHz' 
    dir_name_1 = str(37_5)+'MHz'

##csvmaker
with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/'+dir_name_1+'/under25_RR_LL_gain_movecaliBG_MHz_analysis_calibrated_median.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "Frequency", "Right-BG_move_Calibrated", "Left-BG_move_Calibrated", "Right-gain", "Left-gain", "Right-Trx", "Left-Trx", "Used_dB_median"])
    w.writeheader()
    # fig = plt.figure(figsize=(18.0, 12.0))
    # gs = gridspec.GridSpec(260, 2)
    # ax_r = plt.subplot(gs[0:40, :])
    # ax_l = plt.subplot(gs[55:95, :])
    for idxes_BG in [idxes_BG_2007_20080701, idxes_BG_20080808_2009, idxes_BG_2012_2014, idxes_BG_2017_20171101, idxes_BG_20171101_2020]:
        fig = plt.figure(figsize=(18.0, 12.0))
        gs = gridspec.GridSpec(260, 2)
        ax_r = plt.subplot(gs[0:40, :1])
        ax_l = plt.subplot(gs[55:95, :1])
        ax1_r = plt.subplot(gs[0:40, 1:])
        ax1_l = plt.subplot(gs[55:95, 1:])
        idx_freq = np.where(Frequency[idxes_BG] == getNearestValue(Frequency[idxes_BG][0], Frequency_setting))[1][0]
    
    
        idx_nonTrx_RR = np.where((Trx_RR[idxes_BG,idx_freq] >= 9000) | (Trx_RR[idxes_BG,idx_freq] <= 0))[0]
        idx_nonTrx_LL = np.where((Trx_LL[idxes_BG,idx_freq] >= 9000) | (Trx_LL[idxes_BG,idx_freq] <= 0))[0]
        idx_nonTrx_LL_RR = np.where(((Trx_LL[idxes_BG,idx_freq]/Trx_RR[idxes_BG,idx_freq]) >= 4) |((Trx_LL[idxes_BG,idx_freq]/Trx_RR[idxes_BG,idx_freq]) <= 0.25))[0]
        idx_nongain_LL_RR = np.where((gain_LL[idxes_BG,idx_freq]/gain_RR[idxes_BG,idx_freq] <= 0.5) | (gain_LL[idxes_BG,idx_freq]/gain_RR[idxes_BG,idx_freq] >= 2))[0]
    
    
    
    
        # idx_nonTrx = np.unique(np.concatenate([idx_nonTrx_RR, idx_nonTrx_LL, idx_nonTrx_LL_RR, idx_nongain_LL_RR], 0))
        idx_nonTrx = np.unique(np.concatenate([idx_nonTrx_RR, idx_nonTrx_LL, idx_nonTrx_LL_RR, idx_nongain_LL_RR], 0))
        BG_obs_times_selected = np.delete(BG_obs_times[idxes_BG], idx_nonTrx, 0)
        Trx_RR_selected = np.delete(Trx_RR[idxes_BG], idx_nonTrx, 0)
        BG_r_selected = np.delete(BG_r[idxes_BG], idx_nonTrx, 0)
        cali_BG_r_selected = np.delete(cali_BG_r[idxes_BG], idx_nonTrx, 0)
        gain_RR_selected = np.delete(gain_RR[idxes_BG], idx_nonTrx, 0)
        




        
        Trx_LL_selected = np.delete(Trx_LL[idxes_BG], idx_nonTrx, 0)
        BG_l_selected = np.delete(BG_l[idxes_BG], idx_nonTrx, 0)
        cali_BG_l_selected = np.delete(cali_BG_l[idxes_BG], idx_nonTrx, 0)
        gain_LL_selected = np.delete(gain_LL[idxes_BG], idx_nonTrx, 0)
        # delete_date_idx = []
        # for i in range(len(delete_date)):
        #     BG_obs_times_selected = BG_obs_times_selected[BG_obs_times_selected != delete_date[i]]
    
        # if not len(BG_obs_times_selected) == len(move_cali_BG_l_selected):
        #     sys.exit()
        mean_l = np.mean(cali_BG_l_selected[:,idx_freq])
        std_l = np.std(cali_BG_l_selected[:,idx_freq])
        mean_r = np.mean(cali_BG_r_selected[:,idx_freq])
        std_r = np.std(cali_BG_r_selected[:,idx_freq])
        threshold_l = [mean_l - 2*std_l, mean_r + 2*std_r]
        threshold_r = [mean_r - 2*std_r, mean_l + 2*std_l]

        idx_nonBG_RR = np.where(((cali_BG_r_selected[:,idx_freq]) >= threshold_r[1]) |((cali_BG_r_selected[:,idx_freq]) <= threshold_r[0]))[0]
        idx_nonBG_LL = np.where(((cali_BG_l_selected[:,idx_freq]) >= threshold_l[1]) |((cali_BG_l_selected[:,idx_freq]) <= threshold_l[0]))[0]
        idx_nonBG = np.unique(np.concatenate([idx_nonBG_RR, idx_nonBG_LL], 0))
    
        # BG_obs_times[idx_nonBG] = np.nan
        # Trx_RR[idx_nonBG] = np.nan
        cali_BG_r_selected[idx_nonBG,idx_freq] = np.nan
        # gain_RR[idx_nonBG] = np.nan
      
        # Trx_LL[idx_nonBG] = np.nan
        # cali_BG_l[idx_nonBG] = np.nan
        cali_BG_l_selected[idx_nonBG,idx_freq] = np.nan
        # gain_LL[idx_nonBG] = np.nan

        move_cali_BG_r_selected, move_cali_BG_l_selected, check_obs_time_list = check_BG(cali_BG_r_selected[:,idx_freq], cali_BG_l_selected[:,idx_freq], BG_obs_times_selected, event_check_days)


        for i in range(len(check_obs_time_list)):
            idx_selected = np.where(BG_obs_times==check_obs_time_list[i])[0][0]
            gain_LL_40MHz = gain_LL[idx_selected, idx_freq]
            gain_RR_40MHz = gain_RR[idx_selected, idx_freq]
            Trx_RR_40MHz = Trx_RR[idx_selected, idx_freq]
            Trx_LL_40MHz = Trx_LL[idx_selected, idx_freq]
    
            Frequency_40MHz = Frequency[idx_selected, idx_freq]
    
            Used_db_median_40MHz = Used_db_median[idx_selected, idx_freq]
            w.writerow({ "obs_time": check_obs_time_list[i], "Frequency": Frequency_40MHz, "Right-BG_move_Calibrated": move_cali_BG_r_selected[i], "Left-BG_move_Calibrated": move_cali_BG_l_selected[i], "Right-gain": gain_RR_40MHz, "Left-gain": gain_LL_40MHz, "Right-Trx": Trx_RR_40MHz, "Left-Trx": Trx_LL_40MHz, "Used_dB_median": Used_db_median_40MHz})
           
    
        ax_r.plot(check_obs_time_list, move_cali_BG_r_selected, color = 'b', marker='o', linestyle='dashdot')
        # plt.xticks(rotation=45)
    
        ax_l.plot(check_obs_time_list, move_cali_BG_l_selected, color = 'b', marker='o', linestyle='dashdot')
        # plt.xticks(rotation=45)
        ax1_r.plot(BG_obs_times_selected, cali_BG_r_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
        ax1_l.plot(BG_obs_times_selected, cali_BG_l_selected[:,idx_freq], color = 'b', marker='o', linestyle='dashdot')
        plt.show()
        plt.close()
#     # move_cali_BG_r_selected, move_cali_BG_l_selected
    
    