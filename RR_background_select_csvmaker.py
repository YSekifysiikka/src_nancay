#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 12:14:39 2021

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
import datetime
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from scipy import signal


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

def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res


def read_data(Parent_directory, file_name, move_ave, Freq_start, Freq_end):
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
    min_power = np.amin(y_power, axis=1)
    min_db = np.log10(min_power) * 10
    diff_db_min_med = (diff_move_db.T - min_db).T
    return diff_move_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs, diff_move_RR_db, diff_move_LL_db

def separated_data_antenna(diff_move_RR_db, epoch, time_co, time_band, sep_place, Status):

    time = sep_place - 1
    print (time)
    #+1 is due to move_average
    start = epoch[time + 1]
    start = datetime.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time += 1799
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


    diff_move_db_RR_sep = diff_move_RR_db[freq_start_idx:freq_end_idx + 1, time - 1799:time]
    Status_sep = Status[time - 1799:time]

    return diff_move_db_RR_sep, x_lims, time, Time_start, Time_end, Status_sep

def threshold_array(diff_move_db_RR_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration, db_threshold_percentail,db_range, Status_sep):
    quartile_db_r = []
    mean_r_list = []
    stdev_sub = []
    quartile_power = []
    
    cali_place = []
    for cali17 in (np.where(Status_sep== 17)[0]):
        if cali17 >= 1:
            if cali17 + 11 <= diff_db_min_med.shape[1] - 1:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17-1,cali17 + 12,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17-1,diff_db_min_med.shape[1],1)
        elif cali17 == 0:
            if cali17 + 12 <= diff_db_min_med.shape[1]:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17,cali17 + 12,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17,diff_db_min_med.shape[1],1)
        else:
            sys.exit()
            pass

    for cali0 in np.where(Status_sep== 0)[0]:
        if cali0 >= 41:
            if cali0 + 1 <= diff_db_min_med.shape[1] - 1:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali0-41,cali0 + 2,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali0-41, diff_db_min_med.shape[1],1)
        elif cali0 >= 0 and cali0 < 41:
            if cali0 + 2 <= diff_db_min_med.shape[1]:
                cali_place[len(cali_place):len(cali_place)] = np.arange(0,cali0 + 2,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(0,diff_db_min_med.shape[1],1)
        else:
            sys.exit()
            pass
    cali_place = list(set(cali_place))
    diff_move_db_RR_sep = np.delete(diff_move_db_RR_sep, cali_place, 1)


    for i in range(diff_move_db_RR_sep.shape[0]):
    #    for i in range(0, 357, 1):
        quartile_db_25_1 = np.percentile(diff_move_db_RR_sep[i], db_threshold_percentail)
        quartile_db_25_2 = np.percentile(diff_move_db_RR_sep[i], db_threshold_percentail-db_range)
        quartile_db_each_r = []
        for k in range(diff_move_db_RR_sep[i].shape[0]):
            if diff_move_db_RR_sep[i][k] <= quartile_db_25_1:
                if diff_move_db_RR_sep[i][k] >= quartile_db_25_2:
                    diff_power_quartile_r = (10 ** ((diff_move_db_RR_sep[i][k])/10))
                    quartile_db_each_r.append(diff_power_quartile_r)
        #                quartile_db_each.append(math.log10(diff_power) * 10)
        m_r = np.mean(quartile_db_each_r)
        stdev_r = np.std(quartile_db_each_r)
        sigma_r = m_r + sigma_value * stdev_r
        sigma_db_r = (math.log10(sigma_r) * 10)
        quartile_power.append(sigma_r)
        quartile_db_r.append(sigma_db_r)
        mean_r_list.append(math.log10(m_r) * 10)
    quartile_power = np.array(quartile_power)
    quartile_db_r = np.array(quartile_db_r)
    mean_r_list = np.array(mean_r_list)
    stdev_sub = quartile_db_r - mean_r_list
    diff_power_last_r = ((10 ** ((diff_move_db_RR_sep)/10)).T - quartile_power).T
    
    arr_threshold_1 = np.where(diff_power_last_r > 1, diff_power_last_r, 1)
    arr_threshold = np.log10(arr_threshold_1) * 10
    return arr_threshold, mean_r_list, quartile_db_r, quartile_power, diff_power_last_r, stdev_sub


    
#rawデータに移動平均をかけている
sigma_value = 2
after_plot = str('af_sgepss')
time_band = 5340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
freq_check_range = 20
threshold_frequency_final = 10.5
cnn_plot_time = 50
save_place = 'cnn_af_sgepss'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 1.35
db_setting = 40
db_range=5

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
db_threshold_percentails = np.arange(db_range,db_range*2,db_range)
db_check_mean_list = []
db_threshold_percentail = db_threshold_percentails[0]
#0-5%の範囲でBGを決定している
file_path = Parent_directory + '/solar_burst/Nancay/final_1.txt'

import csv
date_in=[20131229,20131229]
# years = ['2013']
with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/RR/antenna_RR_all_freq_under5_'+str(date_in[0])+'_'+str(date_in[1])+'.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "freq_start", "freq_end", "decibel"])
    w.writeheader()


    start_day,end_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    
    DATE=sdate
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        try:
        # with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_3.csv', 'w') as f:
        #     w = csv.DictWriter(f, fieldnames=["obs_time", "freq_start", "freq_end", "decibel"])
        #     w.writeheader()
            yyyy = date[:4]
            mm = date[4:6]
            file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[10]

            diff_move_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs, diff_move_RR_db, diff_move_LL_db= read_data(Parent_directory, file_name, 3, 80, 10)
            date_list=[datetime.datetime(epoch[1][0], epoch[1][1], epoch[1][2], epoch[1][3], epoch[1][4], epoch[1][5]) + datetime.timedelta(seconds=i) for i in range((datetime.datetime(epoch[-2][0], epoch[-2][1], epoch[-2][2], epoch[-2][3], epoch[-2][4], epoch[-2][5])-datetime.datetime(epoch[1][0], epoch[1][1], epoch[1][2], epoch[1][3], epoch[1][4], epoch[1][5])).seconds)]
            cos_list = np.array(solar_cos_list(date_list))
            max_index = int(np.median(np.where(cos_list==np.max(cos_list))[0]))
            start = max_index - 1800*int((14353-900)/1800)
            sep_places = np.arange(start, len(epoch)-901-2,1800) - 900
            mean_list = []
            time_list_oneday = []
            for sep_place in sep_places:
                diff_move_db_RR_sep, x_lims, time, Time_start, Time_end, Status_sep = separated_data_antenna(diff_move_RR_db, epoch, time_co, time_band, sep_place, Status)
                arr_threshold, mean_r_list, quartile_db_r, quartile_power, diff_power_last_r, stdev_sub = threshold_array(diff_move_db_RR_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration, db_threshold_percentail, db_range, Status_sep)
                # plot_array_threshold(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, db_setting, min_db, quartile_db_l)
                mean_list.append(mean_r_list)
                time_list_oneday.append(np.mean(x_lims))
                obs_time= datetime.datetime(int(date_OBs[:4]), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]), int(Time_start.split(':')[2])) + (datetime.datetime(int(date_OBs[:4]), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_end.split(':')[0]), int(Time_end.split(':')[1]), int(Time_end.split(':')[2])) - datetime.datetime(int(date_OBs[:4]), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]), int(Time_start.split(':')[2])))/2
                db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
                # np.array(mean_list)[:,db_standard[0][0]]
                w.writerow({'obs_time':obs_time, 'freq_start':Frequency_start, 'freq_end':Frequency_end, 'decibel':mean_r_list})
            db_check_mean_list.append(np.array(mean_list))

        
            if len(mean_list) == 1:
                mean_list = []
                time_list_oneday = []
                for t in range(2):
                    mean_list.append(mean_r_list)
                    time_list_oneday.append(x_lims[t])
            year = date_OBs[0:4]
            month = date_OBs[4:6]
            day = date_OBs[6:8]
            db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(time_list_oneday, np.array(mean_list)[:,db_standard[0][0]])
            ln1=ax1.plot(time_list_oneday, np.array(mean_list)[:,db_standard[0][0]],'C0',label='parcentail: ' + str(db_threshold_percentails[0]-db_range) + '-' + str(db_threshold_percentails[0])+'%', linestyle='dashdot')
            # ax1.plot(time_list_oneday, np.array(mean_list[i+1])[:,db_standard[0][0]],'C1',label='parcentail: ' + str(db_threshold_percentails[i+1]-db_range) + '-' + str(db_threshold_percentails[i+1])+'%', linestyle='dashdot')
            
            # ax2 = ax1.twinx()
            
            # ln2=ax2.plot(time_list_oneday, np.array(mean_list[i+1][:,db_standard[0][0]] - np.array(mean_list[i])[:,db_standard[0][0]]),'C2',label='Gap')
            
            h1, l1 = ax1.get_legend_handles_labels()
            # h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1, l1, loc='upper right')
            ax1.xaxis_date()
            date_format = mdates.DateFormatter('%H:%M')
            ax1.xaxis.set_major_formatter(date_format)
            ax1.set_xlabel('t')
            ax1.set_ylabel('Decibel [dB]',fontsize=20)
            ax1.grid(True)
            # ax1.set_ylim([min(np.array(mean_list)[:,db_standard[0][0]]), min(np.array(mean_list)[:,db_standard[0][0]]) + 10])
            # ax2.set_ylabel('Decibel [dB]',fontsize=20)
            # ax2.set_ylim([0, 2])
            plt.title('Nancay: '+year+'-'+month+'-'+day+': dB threshold '+ str(db_threshold_percentails[0]-db_range) + '-' + str(db_threshold_percentails[0]))
            # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_'+str(db_setting)+'/'+year):
                # os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_'+str(db_setting)+'/'+year)
            # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_'+str(db_setting)+'/'+year+'/'+year+month+day+'.png'
            # plt.savefig(filename)
            plt.show()
            plt.close()
            # sys.exit()
        
                

        except:
            print('Plot error: ',date)
        DATE+=pd.to_timedelta(1,unit='day')