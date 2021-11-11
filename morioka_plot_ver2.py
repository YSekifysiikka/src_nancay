#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 11:48:44 2021

@author: yuichiro
"""
import func_small
import func_read_detection

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from scipy import signal
import os
import glob
import math
import astropy.time
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import cdflib
import csv

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


# file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_final.csv"
# antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

# obs_time = []
# decibel_list = []
# for i in range(len(antenna1_csv)):
#     obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
#     # decibel_list.append(antenna1_csv['decibel'][i])
#     l = antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')
#     decibel_list.append([float(s) for s in l if s != ''])
#     # for j in range(len(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' '))):
#     #     if not antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j] == '':
#     #         decibel_list.append(float(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j]))

# obs_time = np.array(obs_time)
# decibel_list = np.array(decibel_list)

def plot_green(around, ax, dB_BG, check_freq):
    # maxid = signal.argrelmax(around, order=width)
    maxid = np.array([int(k) for k in np.arange(1,len(around)-2,1) if (around[k] >= around[k-1]) & (around[k] >= around[k+1]) & (around[k+1] >= around[k+2])])
    if plot_setting == 'yes':
        ax.plot(np.arange(0,time_band+time_co,1), around, '.-', linewidth=0.5, markersize=1, label = check_freq)
    # ax1.axhline(dB_line, ls = "-.", color = "magenta", label = str(dB_line)+'dB')
        ax.axhline(dB_BG, ls = "-.", color = "r", label = str(round(dB_BG,1))+'dB')
        if plot_type == 'rgb':
            ax.scatter(np.arange(0,time_band+time_co,1)[maxid], around[maxid], color = "blue")
    maxid_list = []
    for maxidx in maxid:
        if threshold_time % 2 == 1:
            if (maxidx - int(threshold_time/2) >= 0) & (maxidx + int(threshold_time/2) < time_band + time_co):
                if len(np.where(around[int(maxidx-int(threshold_time/2)-1):int(maxidx+int(threshold_time/2)+1)] > dB_BG)[0]) == threshold_time:
                    maxid_list.append(maxidx)
                    if plot_setting == 'yes':
                        if plot_type == 'rgb':
                            ax.scatter(np.arange(0,time_band+time_co,1)[maxidx], around[maxidx], color = "g")
        else:
            if (maxidx - int(threshold_time/2)+1 >= 0) & (maxidx + int(threshold_time/2) <= time_band + time_co):
                if len(np.where(around[int(maxidx-int(threshold_time/2)+1):int(maxidx+int(threshold_time/2)+1)] > dB_BG)[0]) == threshold_time:
                    maxid_list.append(maxidx)
                    if plot_setting == 'yes':
                        if plot_type == 'rgb':
                            ax.scatter(np.arange(0,time_band+time_co,1)[maxidx], around[maxidx], color = "g")
    return maxid_list

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

    diff_P_RR = 10 ** ((diff_r_last[:,:])/10)
    diff_db_RR = np.log10(diff_P_RR) * 10
    diff_P_LL = 10 ** ((diff_l_last[:,:])/10)
    diff_db_LL = np.log10(diff_P_LL) * 10
    
    RR_power = []
    num = int(move_ave)
    b = np.ones(num)/num
    for i in range (diff_db_RR.shape[0]):
        RR_power.append(np.convolve(diff_P_RR[i], b, mode='valid'))
    RR_power = np.array(RR_power)
    diff_move_RR_db = np.log10(RR_power) * 10

    LL_power = []
    num = int(move_ave)
    b = np.ones(num)/num
    for i in range (diff_db_LL.shape[0]):
        LL_power.append(np.convolve(diff_P_LL[i], b, mode='valid'))
    LL_power = np.array(LL_power)
    diff_move_LL_db = np.log10(LL_power) * 10



    return Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs, diff_move_RR_db, diff_move_LL_db

def separated_data_antenna(diff_move_RR_db, diff_move_LL_db, epoch, time_co, time_band, t, Status):
    if t == math.floor((diff_move_RR_db.shape[1]-time_co)/time_band):
        t = (diff_move_RR_db.shape[1] + (-1*(time_band+time_co)))/time_band
    if t >= 0:
    # if t >  36:
    #     sys.exit()
        time = round(time_band*t)
        print (time)
        #+1 is due to move_average
        start = epoch[time + 1]
        start = datetime.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
        # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        time = round(time_band*(t+1) + time_co)
        end = epoch[time + 1]
        end = datetime.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
        # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        Time_start = start.strftime('%H:%M:%S')
        Time_end = end.strftime('%H:%M:%S')
        print (start)
        print(Time_start+'-'+Time_end)
        start = start.timestamp()
        end = end.timestamp()
        
        # print (start, end)
    
        x_lims = []
        x_lims.append(datetime.datetime.fromtimestamp(start))
        x_lims.append(datetime.datetime.fromtimestamp(end))
    
        x_lims = mdates.date2num(x_lims)
    
    
    
        # diff_db_plot_sep = diff_db_min_med[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
        # diff_move_db_sep = diff_move_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
        diff_move_db_RR_sep = diff_move_RR_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
        diff_move_db_LL_sep = diff_move_LL_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
        Status_sep = Status[time - time_band - time_co:time]
    else:
        time = 0
        print (time)
        #+1 is due to move_average
        start = epoch[time + 1]
        start = datetime.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
        # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        time = diff_move_RR_db.shape[1] - 1
        end = epoch[time + 1]
        end = datetime.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
        # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        Time_start = start.strftime('%H:%M:%S')
        Time_end = end.strftime('%H:%M:%S')
        print (start)
        print(Time_start+'-'+Time_end)
        start = start.timestamp()
        end = end.timestamp()
        
        # print (start, end)
    
        x_lims = []
        x_lims.append(datetime.datetime.fromtimestamp(start))
        x_lims.append(datetime.datetime.fromtimestamp(end))
    
        x_lims = mdates.date2num(x_lims)
    
    
        diff_move_db_RR_sep = diff_move_RR_db[freq_start_idx:freq_end_idx + 1]
        diff_move_db_LL_sep = diff_move_LL_db[freq_start_idx:freq_end_idx + 1]
        # diff_db_plot_sep = diff_db_min_med[freq_start_idx:freq_end_idx + 1]
        # diff_move_db_sep = diff_move_db[freq_start_idx:freq_end_idx + 1]
        Status_sep = Status
        


    return diff_move_db_RR_sep, diff_move_db_LL_sep, x_lims, time, Time_start, Time_end, Status_sep


def plot_arange(ax_list):
    for ax in ax_list:
        ax.set_xlim(0,time_band+time_co-1)
        ax.legend(fontsize = 12)
        ax.tick_params(labelbottom=False,
                       labelright=False,
                       labeltop=False)
    return

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
    # print (azimuth, altitude)
    
    solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
                            math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
                            math.sin(math.radians(altitude))])
    machine_place = np.array([math.cos(math.radians(70)) * math.cos(math.radians(180)),
                            math.cos(math.radians(70)) * math.sin(math.radians(180)),
                            math.sin(math.radians(70))])
    cos = np.dot(solar_place, machine_place)
    return cos




Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1





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
check_data = 1
check_data_plot = 3
width = 3
threshold_time = 4
dB_line = 20
dB_threshold = 25
skip_rate = 1
v_list = [19, 35]
save_dir = 'LL_RR_freq_check_around'+str(check_data)+'_width'+str(width)+'_mediancos_dBthreshold'+str(dB_threshold)+'_threshold_time'+str(threshold_time)+'_skiprate'+str(skip_rate) +'_ver2'
plot_type = 'rgb'
#basic:check_data7, witdh5
dB_list_final_r = []
dB_list_final_l = []
dB_list_final_total = []
burst_time_list_final = []
dB_BG_40_list_r = []
dB_BG_40_list_l = []



date_in=[20120726,20120731]
plot_setting =  'yes'
with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/morioka_analysis1/'+str(date_in[0])+'_'+str(date_in[1])+'.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["Time_list_35MHz", "intensity", "intensity_RR","intensity_LL", "BG_40_RR[dB]", "BG_40_LL[dB]", "cos"])
    w.writeheader()

    file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG_5freq/under25_RR_LL_gain_movecaliBG_MHz_analysis_calibrated_median.csv'
    
    BG_obs_times = []
    decibel_list_r = []
    decibel_list_l = []
    cali_decibel_list_r = []
    cali_decibel_list_l = []
    Frequency_list = []
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
        cali_RR = csv_input['Right-BG_move_Calibrated'][i].replace('\n', '')[1:-1].split(' ')
        cali_decibel_list_r.append([float(s) for s in cali_RR if s != ''])
        cali_LL = csv_input['Left-BG_move_Calibrated'][i].replace('\n', '')[1:-1].split(' ')
        cali_decibel_list_l.append([float(s) for s in cali_LL if s != ''])
        # RR = csv_input['Right-BG'][i].replace('\n', '')[1:-1].split(' ')
        # decibel_list_r.append([float(s) for s in RR if s != ''])
        # LL = csv_input['Left-BG'][i].replace('\n', '')[1:-1].split(' ')
        # decibel_list_l.append([float(s) for s in LL if s != ''])
    
        Frequency_list.append([float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        gain_RR.append([float(k) for k in csv_input['Right-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        Trx_RR.append([float(k) for k in csv_input['Right-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # hot_dB_RR.append([float(k) for k in csv_input['Right-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # cold_dB_RR.append([float(k) for k in csv_input['Right-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # Frequency.append([float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        gain_LL.append([float(k) for k in csv_input['Left-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        Trx_LL.append([float(k) for k in csv_input['Left-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # hot_dB_LL.append([float(k) for k in csv_input['Left-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # cold_dB_LL.append([float(k) for k in csv_input['Left-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        Used_db_median.append([float(k) for k in csv_input['Used_dB_median'][i][1:-1].replace('\n', '').split(' ') if k != ''])
    
    BG_obs_times = np.array(BG_obs_times)
    cali_move_BG_r = np.array(cali_decibel_list_r)
    cali_move_BG_l = np.array(cali_decibel_list_l)
    Frequency_list = np.array(Frequency_list)
    gain_RR = np.array(gain_RR)
    Trx_RR = np.array(Trx_RR)
    gain_LL = np.array(gain_LL)
    Trx_LL = np.array(Trx_LL)
    Used_db_median = np.array(Used_db_median)
    
    
    
    start_day,end_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    DATE=sdate
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        year = date[:4]
        mm = date[4:6]
        # try:
        if len(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+year+'/'+mm+'/*'+ date +'*cdf')) >0:
            file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+year+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[-1]
            print (file_name)
            Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs, diff_move_RR_db, diff_move_LL_db= read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end)
            for t in range (math.floor(((diff_move_RR_db.shape[1]-time_co)/time_band) + 1)):
                diff_move_db_RR_sep, diff_move_db_LL_sep, x_lims, time, Time_start, Time_end, Status_sep = separated_data_antenna(diff_move_RR_db, diff_move_LL_db, epoch, time_co, time_band, t, Status)
                obs_datetime = datetime.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]), int(Time_start.split(':')[2])) + datetime.timedelta(seconds=(time_band+time_co)/2)
                if np.abs(getNearestValue(BG_obs_times, obs_datetime) - obs_datetime) <= datetime.timedelta(seconds=5400):
                    BG_idx = np.where(BG_obs_times == getNearestValue(BG_obs_times, obs_datetime))[0][0]
                    BG_r = cali_move_BG_r[BG_idx]
                    BG_l = cali_move_BG_l[BG_idx]
                    check_Frequency = Frequency_list[BG_idx]
                    gain_r = gain_RR[BG_idx]
                    gain_l = gain_LL[BG_idx]
                    fixed_gain = Used_db_median[BG_idx]
                    cos = solar_cos(obs_datetime)
        
                    # diff_move_P = (((10 ** ((diff_move_db_RR_sep[:,:])/10)) + (10 ** ((diff_move_db_LL_sep[:,:])/10)))/2)
                    # diff_db_plot_sep = np.log10(diff_move_P) * 10
        
                    idx30 = np.where(Frequency == check_Frequency[0])[0][0]
                    idx32_5 = np.where(Frequency == check_Frequency[1])[0][0]
                    idx35 = np.where(Frequency == check_Frequency[2])[0][0]
                    idx37_5 = np.where(Frequency == check_Frequency[3])[0][0]
                    idx40 = np.where(Frequency == check_Frequency[4])[0][0]
        
                    diff_move_30db_RR = diff_move_db_RR_sep[idx30,:]+ (np.log10(fixed_gain[0]/gain_r[0]) * 10)
                    diff_move_30db_LL = diff_move_db_LL_sep[idx30,:]+ (np.log10(fixed_gain[0]/gain_l[0]) * 10)
                    # diff_move_db_30 = np.log10(((10 ** ((diff_move_30db_RR)/10)) + (10 ** ((diff_move_30db_LL)/10)))/2) * 10
                    # dB_BG_30 = np.log10(((10 ** ((BG_r[0])/10)) + (10 ** ((BG_l[0])/10)))/2) * 10
        
                    # diff_move_30db_BG_RR_P = (((10 ** ((diff_move_30db_RR)/10)).T - (10 ** ((BG_r[0])/10))).T)
                    diff_move_32_5db_RR = diff_move_db_RR_sep[idx32_5,:]+ (np.log10(fixed_gain[1]/gain_r[1]) * 10)
                    diff_move_32_5db_LL = diff_move_db_LL_sep[idx32_5,:]+ (np.log10(fixed_gain[1]/gain_l[1]) * 10)
                    # diff_move_db_32_5 = np.log10(((10 ** ((diff_move_32_5db_RR)/10)) + (10 ** ((diff_move_32_5db_LL)/10)))/2) * 10
                    # dB_BG_32_5 = np.log10(((10 ** ((BG_r[1])/10)) + (10 ** ((BG_l[1])/10)))/2) * 10
                    
                    diff_move_35db_RR = diff_move_db_RR_sep[idx35,:]+ (np.log10(fixed_gain[2]/gain_r[2]) * 10)
                    diff_move_35db_LL = diff_move_db_LL_sep[idx35,:]+ (np.log10(fixed_gain[2]/gain_l[2]) * 10)
                    # diff_move_db_35 = np.log10(((10 ** ((diff_move_35db_RR)/10)) + (10 ** ((diff_move_35db_LL)/10)))/2) * 10
                    # dB_BG_35 = np.log10(((10 ** ((BG_r[2])/10)) + (10 ** ((BG_l[2])/10)))/2) * 10
                    
                    diff_move_37_5db_RR = diff_move_db_RR_sep[idx37_5,:]+ (np.log10(fixed_gain[3]/gain_r[3]) * 10)
                    diff_move_37_5db_LL = diff_move_db_LL_sep[idx37_5,:]+ (np.log10(fixed_gain[3]/gain_l[3]) * 10)
                    # diff_move_db_37_5 = np.log10(((10 ** ((diff_move_37_5db_RR)/10)) + (10 ** ((diff_move_37_5db_LL)/10)))/2) * 10
                    # dB_BG_37_5 = np.log10(((10 ** ((BG_r[3])/10)) + (10 ** ((BG_l[3])/10)))/2) * 10
                    
                    diff_move_40db_RR = diff_move_db_RR_sep[idx40,:]+ (np.log10(fixed_gain[4]/gain_r[4]) * 10)
                    diff_move_40db_LL = diff_move_db_LL_sep[idx40,:]+ (np.log10(fixed_gain[4]/gain_l[4]) * 10)
                    # diff_move_db_40 = np.log10(((10 ** ((diff_move_40db_RR)/10)) + (10 ** ((diff_move_40db_LL)/10)))/2) * 10
                    # dB_BG_40 = np.log10(((10 ** ((BG_r[4])/10)) + (10 ** ((BG_l[4])/10)))/2) * 10
        
        
        
                    detected_files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/*/'+year+'/'+date+'_'+Time_start.replace(':','')+'_'+Time_end.replace(':','')+'*.png')
                    stime = time - time_band - time_co
        

                
                
                    plt.close()
                    year=date_OBs[0:4]
                    month=date_OBs[4:6]
                    day=date_OBs[6:8]
        
                    fig = plt.figure(figsize=(36.0, 12.0))
                    gs = gridspec.GridSpec(191, 2)
                    for i in range(2):
                        if i == 0: #RR
                            if plot_setting == 'yes':
                                ax = plt.subplot(gs[0:40, :1])
                                ax4 = plt.subplot(gs[50:65, :1])
                                ax2 = plt.subplot(gs[95:110, :1])#40MHz
                                ax3 = plt.subplot(gs[115:130, :1])#37.5MHz
                                ax1 = plt.subplot(gs[135:150, :1])#35MHz
                                ax5 = plt.subplot(gs[155:170, :1])#32.5MHz
                                ax6 = plt.subplot(gs[175:190, :1])#30MHz
                            else:
                                ax = 'test'
                                ax4 = 'test'
                                ax2 = 'test'
                                ax3 = 'test'
                                ax1 = 'test'
                                ax5 = 'test'
                                ax6 = 'test'
                            diff_move_db_40 = diff_move_40db_RR
                            diff_move_db_37_5 = diff_move_37_5db_RR
                            diff_move_db_35 = diff_move_35db_RR
                            diff_move_db_32_5 = diff_move_32_5db_RR
                            diff_move_db_30 = diff_move_30db_RR
                            dB_BG_30 = BG_r[0]
                            dB_BG_32_5 = BG_r[1]
                            dB_BG_35 = BG_r[2]
                            dB_BG_37_5 = BG_r[3]
                            dB_BG_40 = BG_r[4]
                            dB_list_RR = []
                            burst_time_list_RR = []
                            diff_db_plot_sep = diff_move_db_RR_sep

                        else: #LL
                            if plot_setting == 'yes':
                                ax = plt.subplot(gs[0:40, 1:])
                                ax4 = plt.subplot(gs[50:65, 1:])
                                ax2 = plt.subplot(gs[95:110, 1:])#40MHz
                                ax3 = plt.subplot(gs[115:130, 1:])#37.5MHz
                                ax1 = plt.subplot(gs[135:150, 1:])#35MHz
                                ax5 = plt.subplot(gs[155:170, 1:])#32.5MHz
                                ax6 = plt.subplot(gs[175:190, 1:])#30MHz
                            else:
                                ax = 'test'
                                ax4 = 'test'
                                ax2 = 'test'
                                ax3 = 'test'
                                ax1 = 'test'
                                ax5 = 'test'
                                ax6 = 'test'
                            diff_move_db_40 = diff_move_40db_LL
                            diff_move_db_37_5 = diff_move_37_5db_LL
                            diff_move_db_35 = diff_move_35db_LL
                            diff_move_db_32_5 = diff_move_32_5db_LL
                            diff_move_db_30 = diff_move_30db_LL
                            dB_BG_30 = BG_l[0]
                            dB_BG_32_5 = BG_l[1]
                            dB_BG_35 = BG_l[2]
                            dB_BG_37_5 = BG_l[3]
                            dB_BG_40 = BG_l[4]
                            dB_list_LL = []
                            burst_time_list_LL = []
                            diff_db_plot_sep = diff_move_db_LL_sep
                        if plot_setting == 'yes':
                            plt.subplots_adjust(hspace=0.001)
                        
                            y_lims = [Frequency_end, Frequency_start]
                            ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                      aspect='auto',cmap='jet',vmin=v_list[0],vmax=v_list[1])
                            ax.xaxis_date()
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            ax.tick_params(labelsize=10)
                    
                    
                    
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
                    
                    
                    
                        select_array_empty = np.ones((400,400))
                        select_array_empty = np.where(select_array_empty == 1, np.nan, np.nan)
                    
                        db_40 = np.where(Frequency == func_small.getNearestValue(Frequency,40))[0][0]
                        # around_40 = np.log10(((np.sum(10 ** ((diff_db_plot_sep[int(db_40-((check_data-1)/2)):int(db_40+((check_data-1)/2)+1)])/10), axis =0))/check_data)) * 10
                        around_40 = diff_move_db_40
                        select_array_empty[int(db_40-((check_data_plot-1)/2)):int(db_40+((check_data_plot-1)/2)+1)] = diff_db_plot_sep[int(db_40-((check_data_plot-1)/2)):int(db_40+((check_data_plot-1)/2)+1)]
            
            
            
            
                        db_35 = np.where(Frequency == func_small.getNearestValue(Frequency,35))[0][0]
                        around_35 = diff_move_db_35
                        select_array_empty[int(db_35-((check_data_plot-1)/2)):int(db_35+((check_data_plot-1)/2)+1)] = diff_db_plot_sep[int(db_35-((check_data_plot-1)/2)):int(db_35+((check_data_plot-1)/2)+1)]      
                    
                    
                        db_30 = np.where(Frequency == func_small.getNearestValue(Frequency,30))[0][0]
                        around_30 = diff_move_db_30
                        select_array_empty[int(db_30-((check_data_plot-1)/2)):int(db_30+((check_data_plot-1)/2)+1)] = diff_db_plot_sep[int(db_30-((check_data_plot-1)/2)):int(db_30+((check_data_plot-1)/2)+1)]
                    
                    ########################################
                        db_37_5 = np.where(Frequency == func_small.getNearestValue(Frequency,37.5))[0][0]
                        around_37_5 = diff_move_db_37_5
                        select_array_empty[int(db_37_5-((check_data_plot-1)/2)):int(db_37_5+((check_data_plot-1)/2)+1)] = diff_db_plot_sep[int(db_37_5-((check_data_plot-1)/2)):int(db_37_5+((check_data_plot-1)/2)+1)]
                    
                    
                        db_32_5 = np.where(Frequency == func_small.getNearestValue(Frequency,32.5))[0][0]
                        around_32_5 = diff_move_db_32_5
                        select_array_empty[int(db_32_5-((check_data_plot-1)/2)):int(db_32_5+((check_data_plot-1)/2)+1)] = diff_db_plot_sep[int(db_32_5-((check_data_plot-1)/2)):int(db_32_5+((check_data_plot-1)/2)+1)]

                    ########################################
                        if plot_setting == 'yes':
                            ax4.imshow(select_array_empty[int(db_40-((check_data_plot-1)/2))-2:int(db_30+((check_data_plot-1)/2)+2)], extent = [x_lims[0], x_lims[1],  Frequency[int(db_30+((check_data_plot-1)/2)+1)], Frequency[int(db_40-((check_data_plot-1)/2))]], 
                                          aspect='auto',cmap='jet',vmin = v_list[0],vmax=v_list[1])
                            ax4.xaxis_date()
                            ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                            ax4.tick_params(labelsize=10)
                        # plt.show()
                    
                    
                    
                    
                        if len(cali_idx) > 0:
                            
                            around_40[cali_idx]=np.nan
                            around_35[cali_idx]=np.nan
                            around_30[cali_idx]=np.nan
                            around_37_5[cali_idx]=np.nan
                            around_32_5[cali_idx]=np.nan
                        maxid3_list = plot_green(around_35, ax1, dB_BG_35, '35MHz')
                        maxid2_list = plot_green(around_40, ax2, dB_BG_40, '40MHz')
                        maxid1_list = plot_green(around_30, ax6, dB_BG_30, '30MHz')
                        maxid4_list = plot_green(around_37_5, ax3, dB_BG_37_5, '37.5MHz')
                        maxid5_list = plot_green(around_32_5, ax5, dB_BG_32_5, '32.5MHz')
                    
                        #30MHz
                        maxid1_list = np.array(maxid1_list)
                        #40MHz
                        maxid2_list = np.array(maxid2_list)
                        #35MHz
                        maxid3_list = np.array(maxid3_list)
                        #37.5MHz
                        maxid4_list = np.array(maxid4_list)
                        #32.5MHz
                        maxid5_list = np.array(maxid5_list)
                        for maxid2 in maxid2_list:
                            #40MHzと35MHzの時間差が0秒以上2秒以下
                            if len(maxid3_list[(maxid3_list >= maxid2) & (maxid3_list <= maxid2 + 2)]) >= 0:
                                for maxid3 in maxid3_list[(maxid3_list >= maxid2) & (maxid3_list <= maxid2 + 2)]:
                                    #30MHzと35MHzの時間差が0秒以上2秒以下
                                    if len(maxid1_list[(maxid1_list >= maxid3) & (maxid1_list <= maxid3 + 2)]) >= 0:
                                        for maxid1 in maxid1_list[(maxid1_list >= maxid3) & (maxid1_list <= maxid3 + 2)]:
                                            #40MHzと37.5MHzの時間差が0秒以上1秒以下
                                            if maxid1 >= maxid2 + 1:
                                                if len(maxid4_list[(maxid4_list >= maxid2) & (maxid4_list <= maxid2 + 1)]) >= 0:
                                                    for maxid4 in maxid4_list[(maxid4_list >= maxid2) & (maxid4_list <= maxid2 + 1)]:
                                                        #30MHzと32.5MHzの時間差が0秒以上1秒以下
                                                        if len(maxid5_list[(maxid5_list >= maxid3) & (maxid5_list <= maxid3 + 1)]) >= 0:
                                                            for maxid5 in maxid5_list[(maxid5_list >= maxid3) & (maxid5_list <= maxid3 + 1)]:
                                                                if plot_setting == 'yes':
                                                                    ax1.scatter(np.arange(0,time_band+time_co,1)[maxid3], around_35[maxid3], color = "r")
                                                                    ax2.scatter(np.arange(0,time_band+time_co,1)[maxid2], around_40[maxid2], color = "r")
                                                                    ax6.scatter(np.arange(0,time_band+time_co,1)[maxid1], around_30[maxid1], color = "r")
                                                                    ax3.scatter(np.arange(0,time_band+time_co,1)[maxid4], around_37_5[maxid4], color = "r")
                                                                    ax5.scatter(np.arange(0,time_band+time_co,1)[maxid5], around_32_5[maxid5], color = "r")
                                                                burst_intensity = np.log10((((10 ** ((around_40[maxid2])/10))/cos) - (10 ** ((dB_BG_40)/10))))* 10
                                                                if i == 0:
                                                                    dB_list_RR.append(burst_intensity)
                                                                    bstime = epoch[stime+np.arange(0,time_band+time_co,1)[maxid3]+1]
                                                                    burst_time_list_RR.append(datetime.datetime(bstime[0], bstime[1], bstime[2], bstime[3], bstime[4], bstime[5], bstime[6]))
                                                                else:
                                                                    dB_list_LL.append(burst_intensity)
                                                                    bstime = epoch[stime+np.arange(0,time_band+time_co,1)[maxid3]+1]
                                                                    burst_time_list_LL.append(datetime.datetime(bstime[0], bstime[1], bstime[2], bstime[3], bstime[4], bstime[5], bstime[6]))
                        if plot_setting == 'yes':
                            plot_arange([ax1, ax2, ax3, ax5, ax6])
                            if len(detected_files) > 0:
                                for detected_file in detected_files:
                                    # print (detected_file.split('/')[-3])
                                    name, color = func_small.color_setting_lib(detected_file.split('/')[-3])
                        
                                    ax1.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
                                    ax1.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.nanmax(around_35) - np.nanmin(around_35))*4/5+np.nanmin(around_35), name, size=10, horizontalalignment="center")
                                    ax2.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
                                    ax2.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2,(np.nanmax(around_40) - np.nanmin(around_40))*4/5+np.nanmin(around_40), name, size=10, horizontalalignment="center")
                                    ax6.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
                                    ax6.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.nanmax(around_30) - np.nanmin(around_30))*4/5+np.nanmin(around_30), name, size=10, horizontalalignment="center")
                                    ax3.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
                                    ax3.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2,(np.nanmax(around_37_5) - np.nanmin(around_37_5))*4/5+np.nanmin(around_37_5), name, size=10, horizontalalignment="center")
                                    ax5.axvspan(int(detected_file.split('/')[-1].split('_')[5]), int(detected_file.split('/')[-1].split('_')[6]), color = color, alpha = 0.3)
                                    ax5.text((int(detected_file.split('/')[-1].split('_')[5])+int(detected_file.split('/')[-1].split('_')[6]))/2, (np.nanmax(around_32_5) - np.nanmin(around_32_5))*4/5+np.nanmin(around_32_5), name, size=10, horizontalalignment="center")
                    if plot_setting == 'yes':
                        if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+save_dir+'/'+year + '/' + month+ '/'+year+month+day):
                            os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+save_dir+'/'+year + '/' + month+ '/'+year+month+day)
                        filename = Parent_directory + '/solar_burst/Nancay/plot/'+save_dir+'/'+year + '/' + month+ '/'+year+month+day + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
                        plt.savefig(filename)
                        plt.show()
        
        

        
        
        
# dB_list_final_rr = []
# dB_list_final_ll = []
# dB_list_final_total = []
        
        
        
        
                    # sys.exit()
                    # try:
                        # dB_list, burst_time_list, dB_BG_40 = radio_plot(diff_move_db, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs[:8], burst_type, detected_files, save_dir, width, Status_sep, time - time_band - time_co)
                    u_RR, indices_RR = np.unique(burst_time_list_RR, return_index=True)
                    u_LL, indices_LL = np.unique(burst_time_list_LL, return_index=True)
                    
                    if ((len(u_LL) > 0) & (len(u_RR) > 0)):
                        for u_r in u_RR:
                            if len(np.where(np.abs(getNearestValue(u_LL, u_r) - u_r) < datetime.timedelta(seconds=2))[0]) > 0:
                                l_idx = np.where(u_LL == getNearestValue(u_LL, u_r))[0][0]
                                r_idx = np.where(u_RR == u_r)[0][0]
                                burst_time_list_final.append(u_r)
                                dB_list_final_r.append(dB_list_RR[indices_RR[r_idx]])
                                dB_list_final_l.append(dB_list_LL[indices_LL[l_idx]])
                                # np.log10((((10 ** ((around_40[maxid2])/10))/cos) - (10 ** ((dB_BG_40)/10))))* 10
                                intensity = np.log10((10**(dB_list_RR[indices_RR[r_idx]]/10))+(10**(dB_list_LL[indices_LL[l_idx]]/10)))* 10
                                w.writerow({'Time_list_35MHz': u_r, 'intensity': intensity, 'intensity_RR': dB_list_RR[indices_RR[r_idx]], 'intensity_LL': dB_list_LL[indices_LL[l_idx]], 'BG_40_RR[dB]': BG_r[4], 'BG_40_LL[dB]': BG_l[4], 'cos': cos})


                    # dB_list_final.extend(np.array(dB_list_RR)[indices_RR].tolist())
                    # dB_BG_40_list.append(dB_BG_40)
                    # if len(u.tolist())>0:
                    #     for i in range(len(u.tolist())):
                    #         w.writerow({'Time_list': u.tolist()[i], 'intensity_list': np.array(dB_list)[indices].tolist()[i], 'BG_40_RR[dB]': BG_r[4], 'BG_40_LL[dB]': BG_l[4]})
                    # sys.exit()
                # except:
                    # print ('No data:' + date_OBs[:8])
        
        DATE+=pd.to_timedelta(skip_rate,unit='day')
        
        
        
        
        # #             # sys.exit()
        # # except:
        # #     print('Plot error: ',date)
        # #     # sys.exit()
        # plt.hist(dB_list_final)
        # plt.xlabel('dB')
        # plt.yscale('log')
        # plt.show()
        # plt.close()
        
        
            
            
