#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:16:56 2021

@author: yuichiro
"""


import pandas as pd
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
import shutil
import os
import cdflib
import math
import datetime as dt
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from scipy import signal


import astropy.time
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation

def solar_cos(obs_datetime):
    koko = EarthLocation(lat='47 22 24.00',lon='2 11 50.00', height = '150')
    obs_time = obs_datetime
    toki = astropy.time.Time(obs_time)
    taiyou = get_sun(toki).transform_to(AltAz(obstime=toki,location=koko))
    # print(taiyou)
    # print(taiyou.az) # 天球での方位角
    # print(taiyou.alt) # 天球での仰俯角
    # print(taiyou.distance) # 距離
    # print(taiyou.distance.au) # au単位での距離
    
    azimuth = float(str(taiyou.az).split('d')[0] + '.' + str(taiyou.az).split('d')[1].split('m')[0])
    altitude = float(str(taiyou.alt).split('d')[0] + '.' + str(taiyou.alt).split('d')[1].split('m')[0])
    
    solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
                            math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
                            math.sin(math.radians(altitude))])
    machine_place = np.array([-math.cos(math.radians(0)),
                              0,
                              math.sin(math.radians(0))])
    cos = np.dot(solar_place, machine_place)
    return cos


def color_setting_lib(burst_type):
    if burst_type == 'flare_associated_ordinary':
        name = 'fao'
        color = '#e41a1c'
    elif burst_type == 'flare_related_storm':
        name = 'faμ'
        color = '#4daf4a'
    elif burst_type == 'marginal':
        name = 'non-fam'
        color = "0.8"
    elif burst_type == 'maybe_ordinary':
        name = 'fam'
        color = "0.8"
    elif burst_type == 'ordinary':
        name = 'non-fao'
        color = '#f781bf'
    elif burst_type == 'storm':
        name = 'μ'
        color = '#a65628'
    else:
        sys.exit()
    return name, color
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




Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_final.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

obs_time = []
decibel_list = []
for i in range(len(antenna1_csv)):
    obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    # decibel_list.append(antenna1_csv['decibel'][i])
    l = antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list.append([float(s) for s in l if s != ''])
    # for j in range(len(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' '))):
    #     if not antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j] == '':
    #         decibel_list.append(float(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j]))

obs_time = np.array(obs_time)
decibel_list = np.array(decibel_list)

def check_BG_1(event_date, event_check_days, file_name):
    event_date = str(event_date)
    yyyy = event_date[:4]
    MM = event_date[4:6]
    dd = event_date[6:8]
    select_date = datetime.datetime(int(yyyy),int(MM),int(dd))
    check_decibel = []
    check_obs_time = []
    if event_check_days == 1:
        start_date = select_date
        for i in range(1):
            obs_indexes = np.where((obs_time >= select_date) & (obs_time <= select_date + datetime.timedelta(days=1)))[0]
            for obs_index in obs_indexes:
            # if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])
    else:
        pass
        # start_date = select_date - datetime.timedelta(days=event_check_days/2)
        # for i in range(event_check_days+1):
        #     check_date = start_date + datetime.timedelta(days=i)
        #     obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
        #     if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
        #         check_decibel.append(decibel_list[obs_index])
        #         check_obs_time.append(obs_time[obs_index])
    
    check_decibel = np.array(check_decibel)
    check_obs_time = np.array(check_obs_time)

    return np.median(check_decibel, axis=0)

def check_BG(event_date, event_hour, event_minite, event_check_days, file_name):
    event_date = str(event_date)
    yyyy = event_date[:4]
    MM = event_date[4:6]
    dd = event_date[6:8]
    hh = event_hour
    mm = event_minite
    select_date = datetime.datetime(int(yyyy),int(MM),int(dd),int(hh),int(mm))
    check_decibel = []
    check_obs_time = []
    if event_check_days == 1:
        start_date = select_date
        for i in range(1):
            check_date = start_date
            obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
            if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])
    else:
        start_date = select_date - datetime.timedelta(days=event_check_days/2)
        for i in range(event_check_days+1):
            check_date = start_date + datetime.timedelta(days=i)
            obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
            if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])

    check_decibel = np.array(check_decibel)
    check_obs_time = np.array(check_obs_time)

    return np.median(check_decibel, axis=0)



def radio_plot(diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs, burst_type, detected_files, save_dir, width, Status_sep):
    # freq_list = [Frequency_start, Frequency_end]
    # NDA_freq = Frequency_start/Frequency_end
    # p_data_2 = 20*np.log10(data_2)
    # vmin_2 = 0
    # vmax_2 = 2.5
    
    # freq_axis = freq_setting(receiver_2)
    # y_lim_2 = [freq_axis[0], freq_axis[-1]]
    # rad_2_freq = freq_axis[-1]/freq_axis[0]
    # # freq_list.append(freq_axis[0])
    # # freq_list.append(freq_axis[-1])
    
    # time_axis = data_2.index
    # x_lim_2 = [time_axis[0], time_axis[-1]]
    # x_lim_2 = mdates.date2num(x_lim_2)

    # p_data_1 = 20*np.log10(data_1)
    # vmin_1 = 0
    # vmax_1 = 8
    
    # freq_axis = freq_setting(receiver_1)
    # y_lim_1 = [freq_axis[0], freq_axis[-1]]
    # rad_1_freq = freq_axis[-1]/freq_axis[1]
    # # freq_list.append(freq_axis[0])
    # # freq_list.append(freq_axis[-1])
    
    # time_axis = data_1.index
    # x_lim_1 = [time_axis[0], time_axis[-1]]
    # x_lim_1 = mdates.date2num(x_lim_1)


    plt.close()
    year=date_OBs[0:4]
    month=date_OBs[4:6]
    day=date_OBs[6:8]
    # if type(data_1) != pd.core.frame.DataFrame:
    #     print('radio_plot \n Type error: data type must be DataFrame')
    #     sys.exit()

    # NDA_gs = int(round((NDA_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    # rad_2_gs = int(round((rad_2_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    # rad_1_gs = int(round((rad_1_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    fig = plt.figure(figsize=(18.0, 12.0))
    gs = gridspec.GridSpec(101, 1)
    ax = plt.subplot(gs[0:40, :])
    ax2 = plt.subplot(gs[45:60, :])
    ax3 = plt.subplot(gs[65:80, :])
    ax1 = plt.subplot(gs[85:100, :])
    # ax2 = plt.subplot(gs[NDA_gs+rad_2_gs+1:NDA_gs+rad_2_gs+rad_1_gs+1, :])
    # fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12.0, 12.0))
    plt.subplots_adjust(hspace=0.001)

    y_lims = [Frequency_end, Frequency_start]
    ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=5,vmax=30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(labelsize=10)
    # ax.set_xlabel('Time [UT]')
    # ax.set_ylabel('Frequency [MHz]')
    # ax.set_yscale('log')

    cali_starts = np.where(Status_sep == 17)[0]
    cali_ends = np.where(Status_sep == 0)[0]
    cali_idx = []
    if len(cali_starts) > 0:
        for cali_start in cali_starts:
            cali_idx.extend(np.arange(cali_start-1, cali_start+12,1))
    if len(cali_ends) > 0:
        for cali_end in cali_ends:
            cali_idx.extend(np.arange(cali_end-11, cali_end+2,1))
    cali_idx = np.unique(np.array(cali_idx))
    cali_idx = cali_idx[(cali_idx>=0) & (cali_idx<time_band+time_co)]
    
    # x_idx = np.setdiff1d(np.arange(0,time_band+time_co,1), cali_idx)





    db_40 = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
    # around_40 = np.log10(((np.sum(10 ** ((diff_db_plot_sep[int(db_40-((check_data-1)/2)):int(db_40+((check_data-1)/2)+1)])/10), axis =0))/check_data)) * 10
    around_40 = np.median(diff_db_plot_sep[int(db_40-((check_data-1)/2)):int(db_40+((check_data-1)/2)+1)], axis =0)
    # around_over0 = around_40[around_40>0]
    # under_dBsetting_val = np.percentile(around_over0, 25)
    # under_dB_threshold_around = around_over0[around_over0<=under_dBsetting_val]
    # dB_BG_40 = np.mean(under_dB_threshold_around) + 2*np.std(under_dB_threshold_around)
    under_dBsetting_val = np.nanpercentile(around_40, dB_threshold)
    under_dB_threshold_around = around_40[~np.isnan(around_40)][(around_40[~np.isnan(around_40)]<=under_dBsetting_val)]
    dB_BG_40 = np.mean(under_dB_threshold_around) + 2*np.std(under_dB_threshold_around)

    db_35 = np.where(Frequency == getNearestValue(Frequency,35))[0][0]
    # around_50 = np.log10(((np.sum(10 ** ((diff_db_plot_sep[int(db_50-((check_data-1)/2)):int(db_50+((check_data-1)/2)+1)])/10), axis =0))/check_data)) * 10
    around_35 = np.median(diff_db_plot_sep[int(db_35-((check_data-1)/2)):int(db_35+((check_data-1)/2)+1)], axis =0)
    # around_over0 = around_50[around_50>0]
    # under_dBsetting_val = np.percentile(around_over0, 25)
    # under_dB_threshold_around = around_over0[around_over0<=under_dBsetting_val]
    # dB_BG_50 = np.mean(under_dB_threshold_around) + 2*np.std(under_dB_threshold_around)

    under_dBsetting_val = np.nanpercentile(around_35, dB_threshold)
    under_dB_threshold_around = around_35[~np.isnan(around_35)][(around_35[~np.isnan(around_35)]<=under_dBsetting_val)]
    dB_BG_35 = np.mean(under_dB_threshold_around) + 2*np.std(under_dB_threshold_around)


    db_30 = np.where(Frequency == getNearestValue(Frequency,30))[0][0]
    # around_30 = np.log10(((np.sum(10 ** ((diff_db_plot_sep[int(db_30-((check_data-1)/2)):int(db_30+((check_data-1)/2)+1)])/10), axis =0))/check_data)) * 10
    around_30 = np.median(diff_db_plot_sep[int(db_30-((check_data-1)/2)):int(db_30+((check_data-1)/2)+1)], axis =0)
    # around_over0 = around_30[around_30>0]
    # under_dBsetting_val = np.percentile(around_over0, 25)
    # under_dB_threshold_around = around_over0[around_over0<=under_dBsetting_val]
    # dB_BG__dB_threshold = np.mean(under_dB_threshold_around) + 2*np.std(under_dB_threshold_around)
    under_dBsetting_val = np.nanpercentile(around_30, dB_threshold)
    under_dB_threshold_around = around_30[~np.isnan(around_30)][(around_30[~np.isnan(around_30)]<=under_dBsetting_val)]
    dB_BG_30 = np.mean(under_dB_threshold_around) + 2*np.std(under_dB_threshold_around)
    

    if len(cali_idx) > 0:
        
        around_40[cali_idx]=np.nan
        around_35[cali_idx]=np.nan
        around_30[cali_idx]=np.nan

    maxid = signal.argrelmax(around_35, order=width)
    ax3.plot(np.arange(0,time_band+time_co,1), around_35, '.-', linewidth=0.5, markersize=1, label = '35MHz')
    # ax3.axhline(dB_line, ls = "-.", color = "magenta", label = str(dB_line)+'dB')
    ax3.axhline(dB_BG_35, ls = "-.", color = "r", label = str(round(dB_BG_35,1))+'dB')
    if plot_type == 'rgb':
        ax3.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], around_35[maxid[0]], color = "blue")
    maxid3_list = []
    for maxidx in maxid[0]:
        if threshold_time % 2 == 1:
            if (maxidx - int(threshold_time/2) >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around_35[int(maxidx-int(threshold_time/2)):int(maxidx+int(threshold_time/2)+1)] > dB_BG_35)[0]) == threshold_time:
                    maxid3_list.append(maxidx)
                    if plot_type == 'rgb':
                        ax3.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_35[maxidx], color = "g")


        else:
            if (maxidx - int(threshold_time/2) + 1 >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around_35[int(maxidx-int(threshold_time/2)+1):int(maxidx+int(threshold_time/2)+1)] > dB_BG_35)[0]) == threshold_time:
                    maxid3_list.append(maxidx)
                    if plot_type == 'rgb':
                        ax3.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_35[maxidx], color = "g")


    maxid = signal.argrelmax(around_40, order=width)
    ax2.plot(np.arange(0,time_band+time_co,1), around_40, '.-', linewidth=0.5, markersize=1, label = '40MHz')
    # ax2.axhline(dB_line, ls = "-.", color = "magenta", label = str(dB_line)+'dB')
    ax2.axhline(dB_BG_40, ls = "-.", color = "r", label = str(round(dB_BG_40,1))+'dB')
    if plot_type == 'rgb':
        ax2.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], around_40[maxid[0]], color = "blue")
    maxid2_list = []
    for maxidx in maxid[0]:
        if threshold_time % 2 == 1:
            if (maxidx - int(threshold_time/2) >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around_40[int(maxidx-int(threshold_time/2)):int(maxidx+int(threshold_time/2)+1)] > dB_BG_40)[0]) == threshold_time:
                    maxid2_list.append(maxidx)
                    if plot_type == 'rgb':
                        ax2.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_40[maxidx], color = "g")

        else:
            if (maxidx - int(threshold_time/2) + 1 >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around_40[int(maxidx-int(threshold_time/2)+1):int(maxidx+int(threshold_time/2)+1)] > dB_BG_40)[0]) == threshold_time:
                    maxid2_list.append(maxidx)
                    if plot_type == 'rgb':
                        ax2.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_40[maxidx], color = "g")
    # ax2.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], around_40[maxid[0]])


    maxid = signal.argrelmax(around_30, order=width)
    ax1.plot(np.arange(0,time_band+time_co,1), around_30, '.-', linewidth=0.5, markersize=1, label = '30MHz')
    # ax1.axhline(dB_line, ls = "-.", color = "magenta", label = str(dB_line)+'dB')
    ax1.axhline(dB_BG_30, ls = "-.", color = "r", label = str(round(dB_BG_30,1))+'dB')
    if plot_type == 'rgb':
        ax1.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], around_30[maxid[0]], color = "blue")
    maxid1_list = []
    for maxidx in maxid[0]:
        if threshold_time % 2 == 1:
            if (maxidx - int(threshold_time/2) >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around_30[int(maxidx-int(threshold_time/2)):int(maxidx+int(threshold_time/2)+1)] > dB_BG_30)[0]) == threshold_time:
                    maxid1_list.append(maxidx)
                    if plot_type == 'rgb':
                        ax1.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_30[maxidx], color = "g")

        else:
            if (maxidx - int(threshold_time/2) + 1 >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around_30[int(maxidx-int(threshold_time/2)+1):int(maxidx+int(threshold_time/2)+1)] > dB_BG_30)[0]) == threshold_time:
                    maxid1_list.append(maxidx)
                    if plot_type == 'rgb':
                        ax1.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_30[maxidx], color = "g")
    #30MHz
    maxid1_list = np.array(maxid1_list)
    #40MHz
    maxid2_list = np.array(maxid2_list)
    #35MHz
    maxid3_list = np.array(maxid3_list)
    for maxid2 in maxid2_list:
        if len(maxid3_list[(maxid3_list >= maxid2) & (maxid3_list <= maxid2 + 3)]) >= 0:
            for maxid3 in maxid3_list[(maxid3_list >= maxid2) & (maxid3_list <= maxid2 + 3)]:
                if len(maxid1_list[(maxid1_list >= maxid3) & (maxid1_list <= maxid3 + 4)]) >= 0:
                    for maxid1 in maxid1_list[(maxid1_list >= maxid3) & (maxid1_list <= maxid3 + 4)]:
                        if maxid1 >= maxid2 + 1:
                            ax3.scatter(np.arange(0,time_band+time_co,1)[maxid3], around_35[maxid3], color = "r")
                            ax2.scatter(np.arange(0,time_band+time_co,1)[maxid2], around_40[maxid2], color = "r")
                            ax1.scatter(np.arange(0,time_band+time_co,1)[maxid1], around_30[maxid1], color = "r")
    ax3.set_xlim(0,time_band+time_co-1)
    ax3.legend(fontsize = 12)
    ax3.tick_params(labelbottom=False,
                   labelright=False,
                   labeltop=False)
    ax2.set_xlim(0,time_band+time_co-1)
    ax2.legend(fontsize = 12)
    ax2.tick_params(labelbottom=False,
                   labelright=False,
                   labeltop=False)
                    # ax1.scatter(np.arange(0,time_band+time_co,1)[maxidx], around_30[maxidx], color = "blue")
    # ax2.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], around_40[maxid[0]])
    # ax1.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], around_30[maxid[0]])
    ax1.set_xlim(0,time_band+time_co-1)
    ax1.legend(fontsize = 12)
    ax1.tick_params(labelbottom=False,
                   labelright=False,
                   labeltop=False)




    if len(detected_files) > 0:
        for detected_file in detected_files:
            print (detected_file.split('/')[-3])
            name, color = color_setting_lib(detected_file.split('/')[-3])
            # ax3.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
            # ax3.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.max(around_50) - np.min(around_50))*4/5+np.min(around_50), name, size=10, horizontalalignment="center")
            # ax2.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
            # ax2.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2,(np.max(around_40) - np.min(around_40))*4/5+np.min(around_40), name, size=10, horizontalalignment="center")
            # ax1.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
            # ax1.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.max(around_30) - np.min(around_30))*4/5+np.min(around_30), name, size=10, horizontalalignment="center")
            #nanversion
            ax3.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
            ax3.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.nanmax(around_35) - np.nanmin(around_35))*4/5+np.nanmin(around_35), name, size=10, horizontalalignment="center")
            ax2.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
            ax2.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2,(np.nanmax(around_40) - np.nanmin(around_40))*4/5+np.nanmin(around_40), name, size=10, horizontalalignment="center")
            ax1.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
            ax1.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.nanmax(around_30) - np.nanmin(around_30))*4/5+np.nanmin(around_30), name, size=10, horizontalalignment="center")

# 
    # ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
    #             extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
    #             vmin=vmin_2, vmax=vmax_2)
    # ax1.xaxis_date()
    # ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax1.tick_params(labelsize=10)
    # ax1.set_xlabel('Time [UT]')
    # ax1.set_ylabel('Frequency [MHz]')
    # ax1.set_yscale('log')
    

    # ax2.imshow(p_data_1.T, origin='lower', aspect='auto', cmap='jet',
    #             extent=[x_lim_1[0],x_lim_1[1],y_lim_1[0],y_lim_1[1]],
    #             vmin=vmin_1, vmax=vmax_1)
    # ax2.xaxis_date()
    # ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax2.tick_params(labelsize=10)
    # ax2.set_xlabel('Time [UT]')
    # # ax2.set_ylabel('Frequency [MHz]')
    # ax2.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.xaxis.tick_top()
    # ax.tick_params(labeltop=False) 
    # ax1.spines['bottom'].set_visible(False)
    # ax1.xaxis.tick_top()
    # ax1.tick_params(labeltop=False) 
    # ax2.xaxis.tick_bottom()
    # ax2.set_yscale('log')

    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+save_dir+'/'+burst_type+'/'+year + '/' + month+ '/'+year+month+day):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+save_dir+'/'+burst_type+'/'+year + '/' + month+ '/'+year+month+day)
    filename = Parent_directory + '/solar_burst/Nancay/plot/'+save_dir+'/'+burst_type+'/'+year + '/' + month+ '/'+year+month+day + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    plt.savefig(filename)
    plt.show()
    return


# 
# import os


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

def file_generator(file):
    with open(file, encoding="utf-8") as f:
        for line in f:
            yield line

def final_txt_make(Parent_directory, Parent_lab, year, start, end):
    start = format(start, '04') 
    end = format(end, '04') 
    year = str(year)
    start = int(year + start)
    end = int(year + end)
    path = Parent_directory + '/solar_burst/Nancay/data/' + year + '/*/*'+'.cdf'
    File = glob.glob(path, recursive=True)
    File = sorted(File)
    i = open(Parent_directory + '/solar_burst/Nancay/final.txt', 'w')
    for cstr in File:
        a = cstr.split('/')
        line = a[Parent_lab + 6]+'\n'
        a1 = line.split('_')
        if (int(a1[5][:8])) >= start:
          if (int(a1[5][:8])) <= end:
            i.write(line)
    i.close()
    return start, end



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

def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res


def read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end):
    file_name_separate =file_name.split('_')
    Date_start = file_name_separate[5]
    date_OBs=str(Date_start)
    year=date_OBs[0:4]
    month=date_OBs[4:6]

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
    Status = cdf_file['Status']
    Frequency = cdf_file['Frequency']
    #for_return_frequency_range
    Frequency_start = round(float(getNearestValue(Frequency, Freq_start)), 5)
    Frequency_end = round(float(getNearestValue(Frequency, Freq_end)), 5)
    resolution = round(float(Frequency[1] - Frequency[0]), 5)

    freq_start_idx = np.where(np.flipud(Frequency) == getNearestValue(Frequency, Freq_start))[0][0]
    freq_end_idx = np.where(np.flipud(Frequency) == getNearestValue(Frequency, Freq_end))[0][0]
    Frequency = np.flipud(Frequency)[freq_start_idx:freq_end_idx + 1]


    
    LL = cdf_file['LL'] 
    RR = cdf_file['RR'] 
    
    data_r_0 = RR
    data_l_0 = LL
    diff_r_last =(data_r_0).T
    diff_r_last = np.flipud(diff_r_last)[freq_start_idx:freq_end_idx + 1] * 0.3125
    diff_l_last =(data_l_0).T
    diff_l_last = np.flipud(diff_l_last)[freq_start_idx:freq_end_idx + 1] * 0.3125

    diff_P = (((10 ** ((diff_r_last[:,:])/10)) + (10 ** ((diff_l_last[:,:])/10)))/2)
    diff_db = np.log10(diff_P) * 10
    ####################
    y_power = []
    num = int(move_ave)
    b = np.ones(num)/num
    for i in range (diff_db.shape[0]):
        y_power.append(np.convolve(diff_P[i], b, mode='valid'))
    y_power = np.array(y_power)
    diff_move_db = np.log10(y_power) * 10
    ####################
    # min_power = np.amin(y_power, axis=1)
    # min_db_LL = np.amin(y_power, axis=1)
    # min_db = np.log10(min_power) * 10
    # min_db_LL = np.amin(diff_r_last, axis=1)
    # min_db_RR = np.amin(diff_l_last, axis=1)
    # diff_db_min_med = (diff_move_db.T - min_db).T
    # LL_min = (diff_l_last.T - min_db_LL).T
    # RR_min = (diff_r_last.T - min_db_RR).T
    return diff_l_last, diff_r_last, diff_move_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)

def separated_data_LL_RR(LL, RR, diff_move_db, epoch, time_co, time_band, t, Status):
    if t == math.floor((diff_move_db.shape[1]-time_co)/time_band):
        t = (diff_move_db.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    time = round(time_band*t)
    print (time)
    if time < 0:
        time = 0
    #+1 is due to move_average
    start = epoch[time + 1]
    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time = round(time_band*(t+1) + time_co)
    if time > len(epoch):
        time = len(epoch) - 1
    end = epoch[time + 1]
    end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
    # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)
    start = start.timestamp()
    end = end.timestamp()
    
    # print (start, end)

    x_lims = []
    x_lims.append(dt.datetime.fromtimestamp(start))
    x_lims.append(dt.datetime.fromtimestamp(end))

    x_lims = mdates.date2num(x_lims)



    diff_db_plot_sep = diff_move_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    LL_sep = LL[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    RR_sep = RR[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    Status_sep = Status[time - time_band - time_co:time]
    return LL_sep, RR_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t, Status_sep





sigma_value = 2
after_plot = str('af_sgepss')
time_band = 340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
freq_check_range = 20
threshold_frequency_final = 10.5
cnn_plot_time = 50
file_path = Parent_directory + '/solar_burst/Nancay/final.txt'
save_place = 'cnn_af_sgepss'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 1.35
db_setting = 40
Freq_start = 80
Freq_end = 10
event_check_days = 30
check_data = 7
width = 3
threshold_time = 4
dB_line = 20
dB_threshold = 25
burst_type = 'fao'
save_dir = 'freq_check_around'+str(check_data)+'_width'+str(width)+'_mediancos_dBthreshold'+str(dB_threshold)+'_threshold_time'+str(threshold_time)
plot_type = 'rgb1'
#basic:check_data7, witdh5







date_list = []
if burst_type =='micro':
    files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/storm/*/*.png')
elif burst_type =='fao':
    files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_associated_ordinary/*/*.png')
else:
    pass
for file in files:
    date_list.append(file.split('/')[-1].split('_')[0])
date_list = sorted(list(set(date_list)))

count = 0
for date_in in date_list:
    count+= 1
    if count > 1:
        sys.exit()
    
# date_in=[20090707,20090707]
    start_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
# edate=pd.to_datetime(end_day,format='%Y%m%d')

    DATE=sdate
# while DATE <= edate:
    date=DATE.strftime(format='%Y%m%d')
    print(date)
    year = date[:4]
    mm = date[4:6]
    try:
        file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+year+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[-1]
        print (file_name)
# start_date, end_date = final_txt_make(Parent_directory, Parent_lab, int(year), 101,1231)
# gen = file_generator(file_path)
# for file in gen:
#     file_name = file[:-1]
        LL, RR, diff_move_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end)
        for t in range (math.floor(((diff_move_db.shape[1]-time_co)/time_band) + 1)):
            LL_db_sep, RR_db_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t_1, Status_sep = separated_data_LL_RR(LL, RR, diff_move_db, epoch, time_co, time_band, t, Status)
            obs_datetime = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]), int(Time_start.split(':')[2])) + datetime.timedelta(seconds=time_band+time_co)
            HH = obs_datetime.strftime(format='%H')
            MM = obs_datetime.strftime(format='%M')
            cos = solar_cos(obs_datetime)
            BG_decibel = check_BG(date, HH, MM, event_check_days, file_name)
            diff_db_plot_sep_power = (((10 ** ((diff_db_plot_sep[:,:])/10)).T - (10 ** ((BG_decibel)/10))).T)/cos
            diff_db_plot_sep = np.where(diff_db_plot_sep_power <= 1, np.nan, np.log10(diff_db_plot_sep_power) * 10)
            detected_files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/*/'+year+'/'+date+'_'+Time_start.replace(':','')+'_'+Time_end.replace(':','')+'*.png')
            # if len(detected_files)>0:
            #     sys.exit()


            # # quick_look(date_OBs, x_lims, LL_db_sep, RR_db_sep, diff_db_plot_sep, Frequency_start, Frequency_end, Time_start, Time_end)
            # if int(Time_start.split(':')[2]) >= 30:
            #     wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=1)
            # else:
            #     wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]))
            # waves_setting_1 = {'receiver'   : 'rad1',
            #                  'date_in'    : int(date_OBs[:8]),#yyyymmdd
            #                  'HH'         : str(wind_Time.hour).zfill(2),#hour
            #                  'MM'         : str(wind_Time.minute).zfill(2),#minute
            #                  'SS'         : '00',#second
            #                  'duration'   :  30,#min
            #                  'freq_band'  :  [0.3, 0.7],
            #                  'init_param' : [0, 0],
            #                  'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
            #                  'directry'   : '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R1/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
            #                  }
            # waves_setting_2 = {'receiver'   : 'rad2',
            #                  'date_in'    : int(date_OBs[:8]),#yyyymmdd
            #                  'HH'         : str(wind_Time.hour).zfill(2),#hour
            #                  'MM'         : str(wind_Time.minute).zfill(2),#minute
            #                  'SS'         : '00',#second
            #                  'duration'   :  30,#min
            #                  'freq_band'  :  [0.3, 0.7],
            #                  'init_param' : [0, 0],
            #                  'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
            #                  'directry'   : '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R2/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
            #                  }
            # try:
            #     rw_1 = read_waves(**waves_setting_1)
            #     rw_2 = read_waves(**waves_setting_2)
                
            #     receiver_1 = waves_setting_1['receiver']
            #     receiver_2 = waves_setting_2['receiver']
            #     data_1 = rw_1.read_rad(receiver_1)
            #     data_2 = rw_2.read_rad(receiver_2)
            try:
                radio_plot(diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs[:8], burst_type, detected_files, save_dir, width, Status_sep)
                # sys.exit()
            except:
                print ('No data: wind' + date_OBs[:8])
                # sys.exit()
    except:
        print('Plot error: ',date)
        # sys.exit()
    # DATE+=pd.to_timedelta(1,unit='day')



