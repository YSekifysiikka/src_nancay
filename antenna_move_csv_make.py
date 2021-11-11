#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:48:59 2021

@author: yuichiro
"""

import pandas as pd
import datetime
import numpy as np
import glob
import cdflib
import astropy.time
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import math
import csv
import matplotlib.pyplot as plt

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

def check_BG(event_date, event_hour, event_minite, event_check_days):
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

    return np.percentile(check_decibel, 25, axis = 0)

# def solar_cos(obs_datetime):
#     koko = EarthLocation(lat='47 22 24.00',lon='2 11 50.00', height = '150')
#     obs_time = obs_datetime
#     toki = astropy.time.Time(obs_time)
#     taiyou = get_sun(toki).transform_to(AltAz(obstime=toki,location=koko))
#     # print(taiyou)
#     # print(taiyou.az) # 天球での方位角
#     # print(taiyou.alt) # 天球での仰俯角
#     # print(taiyou.distance) # 距離
#     # print(taiyou.distance.au) # au単位での距離
    
#     azimuth = float(str(taiyou.az).split('d')[0] + '.' + str(taiyou.az).split('d')[1].split('m')[0])
#     altitude = float(str(taiyou.alt).split('d')[0] + '.' + str(taiyou.alt).split('d')[1].split('m')[0])
    
#     solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
#                             math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
#                             math.sin(math.radians(altitude))])
#     machine_place = np.array([-math.cos(math.radians(0)),
#                               0,
#                               math.sin(math.radians(0))])
#     cos = np.dot(solar_place, machine_place)
#     return cos


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



with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/test/all_antenna_move30days_under25.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "30MHz[dB]", "32.5MHz[dB]", "35MHz[dB]", "37.5MHz[dB]", "40MHz[dB]"])
    w.writeheader()
#     for i in range(len(obs_time)):
#         decibel_list[i] = decibel_list[i]/solar_cos(obs_time[i])
#         print ('Preparing ' + str(obs_time[i]))
#         w.writerow({'obs_time': obs_time[i], 'freq_start': antenna1_csv['freq_start'][i], 'freq_end': antenna1_csv['freq_end'][i], 'decibel_with_cos': decibel_list[i]})


    check_40_dB = []
    check_35_dB = []
    check_30_dB = []
    check_37_5_dB = []
    check_32_5_dB = []
    check_data = 7
    event_check_days = 30
    
    
    
    date_in = [20070101,20201231]
    start_day,end_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    
        
    DATE=sdate
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        yyyy = date[:4]
        mm = date[4:6]
        dd = date[6:8]
        try:
            file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[-1]
            print (file_name)
            file = Parent_directory + '/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/'+file_name
            cdf_file = cdflib.CDF(file)
            # epoch = cdf_file['Epoch'] 
            # epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
            # Status = cdf_file['Status']
            Frequency = list(reversed(cdf_file['Frequency']))
            check_40_dB_day = []
            check_35_dB_day = []
            check_30_dB_day = []
            check_40_dB_day = []
            check_37_5_dB_day = []
            check_32_5_dB_day = []
        
        
        
            for obs_time_each in obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=1))]:
                print (obs_time_each)
                # cos = solar_cos(obs_time_each)
                HH = obs_time_each.strftime(format='%H')
                MM = obs_time_each.strftime(format='%M')
                BG_list = check_BG(date, HH, MM, event_check_days)
                db_40 = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
                around_40 = np.nanmedian(BG_list[int(db_40-((check_data-1)/2)):int(db_40+((check_data-1)/2)+1)], axis =0)
                check_40_dB.append(around_40)
                check_40_dB_day.append(around_40)
                db_35 = np.where(Frequency == getNearestValue(Frequency,35))[0][0]
                around_35 = np.nanmedian(BG_list[int(db_35-((check_data-1)/2)):int(db_35+((check_data-1)/2)+1)], axis =0)
                check_35_dB.append(around_35)
                check_35_dB_day.append(around_35)
                db_30 = np.where(Frequency == getNearestValue(Frequency,30))[0][0]
                around_30 = np.nanmedian(BG_list[int(db_30-((check_data-1)/2)):int(db_30+((check_data-1)/2)+1)], axis =0)
                check_30_dB.append(around_30)
                check_30_dB_day.append(around_30)
                db_37_5 = np.where(Frequency == getNearestValue(Frequency,37.5))[0][0]
                around_37_5 = np.nanmedian(BG_list[int(db_37_5-((check_data-1)/2)):int(db_37_5+((check_data-1)/2)+1)], axis =0)
                check_37_5_dB.append(around_37_5)
                check_37_5_dB_day.append(around_37_5)
                db_32_5 = np.where(Frequency == getNearestValue(Frequency,32.5))[0][0]
                around_32_5 = np.nanmedian(BG_list[int(db_32_5-((check_data-1)/2)):int(db_32_5+((check_data-1)/2)+1)], axis =0)
                check_32_5_dB.append(around_32_5)
                check_32_5_dB_day.append(around_32_5)
                w.writerow({'obs_time': obs_time_each, '30MHz[dB]': around_30, '32.5MHz[dB]':around_32_5, '35MHz[dB]': around_35, '37.5MHz[dB]':around_37_5, '40MHz[dB]': around_40})
        except:
            print ('Error: '+str(date))
        # plt.plot()
        # print ('a')
        DATE+=pd.to_timedelta(1,unit='day')


    
    
    
    
    
    
    
    