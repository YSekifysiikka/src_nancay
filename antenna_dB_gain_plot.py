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
    check_decibel = []
    check_obs_time = []
    if event_check_days == 1:
        start_date = select_date
        for i in range(1):
            check_date = start_date
            obs_index = np.where(BG_obs_time == getNearestValue(BG_obs_time,check_date))[0][0]
            if abs(BG_obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(BG_obs_time[obs_index])
    else:
        start_date = select_date - datetime.timedelta(days=event_check_days/2)
        for i in range(event_check_days+1):
            check_date = start_date + datetime.timedelta(days=i)
            obs_index = np.where(BG_obs_time == getNearestValue(BG_obs_time,check_date))[0][0]
            if abs(BG_obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(BG_obs_time[obs_index])

    check_decibel = np.array(check_decibel)
    check_obs_time = np.array(check_obs_time)

    return np.percentile(check_decibel, 25, axis = 0)

# def cali_BG:



Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = '/solar_burst/Nancay/af_sgepss_analysis_data/RR/antenna_RR_all_freq_under5_20131229_20131229.csv'
# file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_final.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

BG_obs_time = []
decibel_list = []
for i in range(len(antenna1_csv)):
    BG_obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    # decibel_list.append(antenna1_csv['decibel'][i])
    l = antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list.append([float(s) for s in l if s != ''])
    # for j in range(len(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' '))):
    #     if not antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j] == '':
    #         decibel_list.append(float(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j]))

BG_obs_time = np.array(BG_obs_time)
decibel_list = np.array(decibel_list)



dB_30_list = []
dB_32_5_list = []
dB_35_list = []
dB_37_5_list = []
dB_40_list = []



check_data = 7
event_check_days = 1


date_in = [20131229,20131229]
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

    
        # if len(obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]) > 0:
        for obs_time_each in BG_obs_time[(BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=1))]:
                # print (obs_time_each)
                # cos = solar_cos(obs_time_each)
            HH = obs_time_each.strftime(format='%H')
            MM = obs_time_each.strftime(format='%M')
            BG_list = check_BG(date, HH, MM, event_check_days)
            db_40 = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
            around_40 = np.nanmedian(BG_list[int(db_40-((check_data-1)/2)):int(db_40+((check_data-1)/2)+1)], axis =0)
            dB_40_list.append(around_40)
            db_35 = np.where(Frequency == getNearestValue(Frequency,35))[0][0]
            around_35 = np.nanmedian(BG_list[int(db_35-((check_data-1)/2)):int(db_35+((check_data-1)/2)+1)], axis =0)
            dB_35_list.append(around_35)
            db_30 = np.where(Frequency == getNearestValue(Frequency,30))[0][0]
            around_30 = np.nanmedian(BG_list[int(db_30-((check_data-1)/2)):int(db_30+((check_data-1)/2)+1)], axis =0)
            dB_30_list.append(around_30)
            db_37_5 = np.where(Frequency == getNearestValue(Frequency,37.5))[0][0]
            around_37_5 = np.nanmedian(BG_list[int(db_37_5-((check_data-1)/2)):int(db_37_5+((check_data-1)/2)+1)], axis =0)
            dB_37_5_list.append(around_37_5)
            db_32_5 = np.where(Frequency == getNearestValue(Frequency,32.5))[0][0]
            around_32_5 = np.nanmedian(BG_list[int(db_32_5-((check_data-1)/2)):int(db_32_5+((check_data-1)/2)+1)], axis =0)
            dB_32_5_list.append(around_32_5)
    except:
        print ('Error: '+str(date))
        # sys.exit()
    # plt.plot()
    # print ('a')
    DATE+=pd.to_timedelta(1,unit='day')

print ('Done')

dB_40_list = np.array(dB_40_list)
dB_37_5_list = np.array(dB_37_5_list)
dB_35_list = np.array(dB_35_list)
dB_32_5_list = np.array(dB_32_5_list)
dB_30_list = np.array(dB_30_list)





gain_obs_time = []
gain_40_list = []
gain_37_5_list = []
gain_35_list = []
gain_32_5_list = []
gain_30_list = []
hot_40_list = []
cold_40_list = []
Trx_40 = []


file_list = [file2]

for file in file_list:
    print (file)

    csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
    # print(csv_input['Time_list'])
    for i in range(len(csv_input)):
        obs_time_event = datetime.datetime(int(csv_input['obs_time'][i].split('-')[0]), int(csv_input['obs_time'][i].split('-')[1]), int(csv_input['obs_time'][i].split(' ')[0][-2:]), int(csv_input['obs_time'][i].split(' ')[1][:2]), int(csv_input['obs_time'][i].split(':')[1]), int(csv_input['obs_time'][i].split(':')[2][:2]))
        # Frequency_list = csv_input['Frequency'][i]
        Frequency = np.array([float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        gain = np.array([float(k) for k in csv_input['Right-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        Trx = np.array([float(k) for k in csv_input['Right-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        hot_dB = np.array([float(k) for k in csv_input['Right-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        cold_dB = np.array([float(k) for k in csv_input['Right-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # Frequency = np.array(Frequency)
        # gain = []
        # for j in range(len(csv_input['gain'][i][1:-1].replace('\n', '').split(' '))):
            # if not csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j] == '':
                # gain.append(float(csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j]))
        # Gain_list = csv_input['gain'][i][1:-1].replace('\n', '').split(' ')
        # gain = csv_input['Right-gain'][i]

        # Trx_list = csv_input['Right-Trx'][i]
        # hot_dB_list = csv_input['Right-hot_dB'][i]
        # cold_dB_list = csv_input['Right-cold_dB'][i]

        
        # print ('a')
        # if not len(Gain_list) == len(Frequency):
        #     print ('c')
        #     sys.exit()
        # print ('b')

        gain_obs_time.append(obs_time_event)
        gain_40_list.append(gain[0])
        gain_37_5_list.append(gain[1])
        gain_35_list.append(gain[2])
        gain_32_5_list.append(gain[3])
        gain_30_list.append(gain[4])
        hot_40_list.append(hot_dB[0])
        cold_40_list.append(cold_dB[0])
        Trx_40.append(Trx[0])

print ('Done')
plt.plot(gain_obs_time, Trx_40)
plt.ylim(1000,3000)
plt.show()
plt.close()

gain_obs_time = np.array(gain_obs_time)
gain_40_list = np.array(gain_40_list)
gain_37_5_list = np.array(gain_37_5_list)
gain_35_list = np.array(gain_35_list)
gain_32_5_list = np.array(gain_32_5_list)
gain_30_list = np.array(gain_30_list)
hot_40_list = np.array(hot_40_list)
cold_40_list = np.array(cold_40_list)

cali_dB_40_list = []
cali_dB_37_5_list = []
cali_dB_35_list = []
cali_dB_32_5_list = []
cali_dB_30_list = []

for i in range(len(BG_obs_time)):
    check_date = BG_obs_time[i]
    gain_index = np.where(gain_obs_time == getNearestValue(gain_obs_time,check_date))[0][0]
    if abs(gain_obs_time[gain_index] - check_date) <= datetime.timedelta(seconds=60*90):
        cali_dB_40_list.append(dB_40_list[i]+(np.log10(np.nanmedian(gain_40_list)/gain_40_list[gain_index]) * 10))
        cali_dB_37_5_list.append(dB_37_5_list[i]+(np.log10(np.nanmedian(gain_37_5_list)/gain_37_5_list[gain_index]) * 10))
        cali_dB_35_list.append(dB_35_list[i]+(np.log10(np.nanmedian(gain_35_list)/gain_35_list[gain_index]) * 10))
        cali_dB_32_5_list.append(dB_32_5_list[i]+(np.log10(np.nanmedian(gain_32_5_list)/gain_32_5_list[gain_index]) * 10))
        cali_dB_30_list.append(dB_30_list[i]+(np.log10(np.nanmedian(gain_30_list)/gain_30_list[gain_index]) * 10))
    else:
        cali_dB_40_list.append(np.nan)
        cali_dB_37_5_list.append(np.nan)
        cali_dB_35_list.append(np.nan)
        cali_dB_32_5_list.append(np.nan)
        cali_dB_30_list.append(np.nan)
cali_dB_40_list = np.array(cali_dB_40_list)
cali_dB_37_5_list = np.array(cali_dB_37_5_list)
cali_dB_35_list = np.array(cali_dB_35_list)
cali_dB_32_5_list = np.array(cali_dB_32_5_list)
cali_dB_30_list = np.array(cali_dB_30_list)




#日変化
check_data = 7
plot_days = 1


date_in = [20131229,20131229]
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
        # epoch = cdf_file['Epoch'] 
        # epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
        # Status = cdf_file['Status']
        # obs_time_list_each = []
        # dB_40_list_each = []
    
    
        if len(BG_obs_time[(BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]) > 0:
            # for obs_time_each in obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]:
            # idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)))[0]

            # ax1 = plt.subplot(gs[:, :])#30MHz
            fig = plt.figure(figsize=(36.0, 12.0))
            gs = gridspec.GridSpec(121, 6)
            ax1 = plt.subplot(gs[0:20, :2])#40MHz
            ax2 = plt.subplot(gs[25:45, :2])#37.5MHz
            ax3 = plt.subplot(gs[50:70, :2])#35MHz
            ax4 = plt.subplot(gs[75:95, :2])#32.5MHz
            ax5 = plt.subplot(gs[100:120, :2])#30MHz



            for i in range(plot_days):
                idxes_BG = np.where((BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                if i == 0:
                    ax1.plot(BG_obs_time[idxes_BG], dB_40_list[idxes_BG], '.--', label = '40MHz')
                    ax2.plot(BG_obs_time[idxes_BG], dB_37_5_list[idxes_BG], '.--', label = '37.5MHz')
                    ax3.plot(BG_obs_time[idxes_BG], dB_35_list[idxes_BG], '.--', label = '35MHz')
                    ax4.plot(BG_obs_time[idxes_BG], dB_32_5_list[idxes_BG], '.--', label = '32.5MHz')
                    ax5.plot(BG_obs_time[idxes_BG], dB_30_list[idxes_BG], '.--', label = '30MHz')

                else:
                    ax1.plot(BG_obs_time[idxes_BG], dB_40_list[idxes_BG], '.--')
                    ax2.plot(BG_obs_time[idxes_BG], dB_37_5_list[idxes_BG], '.--')
                    ax3.plot(BG_obs_time[idxes_BG], dB_35_list[idxes_BG], '.--')
                    ax4.plot(BG_obs_time[idxes_BG], dB_32_5_list[idxes_BG], '.--')
                    ax5.plot(BG_obs_time[idxes_BG], dB_30_list[idxes_BG], '.--')


            full_idxes_BG = np.where((BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)))[0]
            ymax_BG = np.max([np.max(dB_40_list[full_idxes_BG]), np.max(dB_37_5_list[full_idxes_BG]), np.max(dB_35_list[full_idxes_BG]), np.max(dB_32_5_list[full_idxes_BG]), np.max(dB_30_list[full_idxes_BG])])
            ymin_BG = np.min([np.min(dB_40_list[full_idxes_BG]), np.min(dB_37_5_list[full_idxes_BG]), np.min(dB_35_list[full_idxes_BG]), np.min(dB_32_5_list[full_idxes_BG]), np.min(dB_30_list[full_idxes_BG])])

            # plot_arange_dB_day([ax1, ax2, ax3, ax4], ymin_BG, ymax_BG)
            if plot_days == 1:
                Minute_fmt = mdates.DateFormatter('%H:%M')  
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(Minute_fmt)
                    ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=5), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=5))
                    ax.set_ylim(ymin_BG-0.5, ymax_BG+0.5)
                    ax.legend(fontsize = 12)
                    if not ax == ax5:
                        ax.tick_params(labelbottom=False,
                                        labelright=False,
                                        labeltop=False)
                # ax5.xaxis.set_major_formatter(Minute_fmt)
                # ax5.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=5), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=5))
            else:
                fmt = mdates.DateFormatter('%m/%d %H') 
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(fmt)
                    ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=45), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=45))
                    ax.set_ylim(ymin_BG-0.5, ymax_BG+0.5)
                    ax.legend(fontsize = 12)
                    if not ax == ax5:
                        ax.tick_params(labelbottom=False,
                                        labelright=False,
                                        labeltop=False)


            # ax5.set_ylim(ymin_BG-0.5, ymax_BG+0.5)
            # ax5.legend(fontsize = 12)
            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.legend()
            #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
            # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
            plt.xlabel('Time', fontsize = 20)
            if plot_days == 1:
                ax1.set_title('Background analysis: ' + date, fontsize = 25)
            else:
                ax1.set_title('Background analysis: ' + date + ' - ' + (datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)).strftime(format='%Y%m%d'), fontsize = 25)
            ax3.set_ylabel('Intensity[dB]', fontsize = 20)
            plt.tick_params(axis='x', which='major', labelsize=15)
            # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm):
            #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm)
            # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm + '/'+date+'.png'
            # plt.savefig(filename)
            # plt.show()
            # plt.close()

            ax1 = plt.subplot(gs[0:20, 2:4])#40MHz
            ax2 = plt.subplot(gs[25:45, 2:4])#37.5MHz
            ax3 = plt.subplot(gs[50:70, 2:4])#35MHz
            ax4 = plt.subplot(gs[75:95, 2:4])#32.5MHz
            ax5 = plt.subplot(gs[100:120, 2:4])#30MHz
            for i in range(plot_days):
                idxes_gain = np.where((gain_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(gain_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                if i == 0:
                    ax1.plot(gain_obs_time[idxes_gain], gain_40_list[idxes_gain], '.--', label = '40MHz')
                    ax2.plot(gain_obs_time[idxes_gain], gain_37_5_list[idxes_gain], '.--', label = '37.5MHz')
                    ax3.plot(gain_obs_time[idxes_gain], gain_35_list[idxes_gain], '.--', label = '35MHz')
                    ax4.plot(gain_obs_time[idxes_gain], gain_32_5_list[idxes_gain], '.--', label = '32.5MHz')
                    ax5.plot(gain_obs_time[idxes_gain], gain_30_list[idxes_gain], '.--', label = '30MHz')

                else:
                    ax1.plot(gain_obs_time[idxes_gain], gain_40_list[idxes_gain], '.--')
                    ax2.plot(gain_obs_time[idxes_gain], gain_37_5_list[idxes_gain], '.--')
                    ax3.plot(gain_obs_time[idxes_gain], gain_35_list[idxes_gain], '.--')
                    ax4.plot(gain_obs_time[idxes_gain], gain_32_5_list[idxes_gain], '.--')
                    ax5.plot(gain_obs_time[idxes_gain], gain_30_list[idxes_gain], '.--')


            full_idxes_gain = np.where((gain_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(gain_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)))[0]
            ymax_gain = np.max([np.nanmax(gain_40_list[full_idxes_gain]), np.nanmax(gain_37_5_list[full_idxes_gain]), np.nanmax(gain_35_list[full_idxes_gain]), np.nanmax(gain_32_5_list[full_idxes_gain]), np.nanmax(gain_30_list[full_idxes_gain])])
            ymin_gain = np.min([np.nanmin(gain_40_list[full_idxes_gain]), np.nanmin(gain_37_5_list[full_idxes_gain]), np.nanmin(gain_35_list[full_idxes_gain]), np.nanmin(gain_32_5_list[full_idxes_gain]), np.nanmin(gain_30_list[full_idxes_gain])])
            # plot_arange_gain([ax1, ax2, ax3, ax4], ymin_gain, ymax_gain)
            if plot_days == 1:
                Minute_fmt = mdates.DateFormatter('%H:%M')  
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(Minute_fmt)
                    ax.set_xlim(gain_obs_time[full_idxes_gain][0] - datetime.timedelta(minutes=5), gain_obs_time[full_idxes_gain][-1]+ datetime.timedelta(minutes=5))
                    ax.set_ylim(ymin_gain*0.95, ymax_gain*1.05)
                    ax.legend(fontsize = 12)
                    if not ax == ax5:
                        ax.tick_params(labelbottom=False,
                                        labelright=False,
                                        labeltop=False)

            else:
                fmt = mdates.DateFormatter('%m/%d %H') 
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(fmt)
                    ax.set_xlim(gain_obs_time[full_idxes_gain][0] - datetime.timedelta(minutes=45), gain_obs_time[full_idxes_gain][-1]+ datetime.timedelta(minutes=45))
                    ax.set_ylim(ymin_gain*0.95, ymax_gain*1.05)
                    ax.legend(fontsize = 12)
                    if not ax == ax5:
                        ax.tick_params(labelbottom=False,
                                        labelright=False,
                                        labeltop=False)
                # ax5.xaxis.set_major_formatter(fmt)
                # ax5.set_xlim(gain_obs_time[full_idxes_gain][0] - datetime.timedelta(minutes=45), gain_obs_time[full_idxes_gain][-1]+ datetime.timedelta(minutes=45))
                
            # ax5.set_ylim(ymin_gain*0.95, ymax_gain*1.05)
            # ax5.legend(fontsize = 12)
            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.legend()
            #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
            # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
            plt.xlabel('Time', fontsize = 20)
            if plot_days == 1:
                ax1.set_title('Antenna analysis: ' + date, fontsize = 25)
            else:
                ax1.set_title('Antenna analysis: ' + date + ' - ' + (datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)).strftime(format='%Y%m%d'), fontsize = 25)
            ax3.set_ylabel('Gain[dB]', fontsize = 20)
            # ax.set_yscale('log')
            plt.tick_params(axis='x', which='major', labelsize=15)
            # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_gain_BG_60days/'+yyyy + '/' + mm):
                # os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_gain_BG_60days/'+yyyy + '/' + mm)
            # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_gain_BG_60days/'+yyyy + '/' + mm + '/'+date+'.png'
            # plt.savefig(filename)
            ax1 = plt.subplot(gs[0:20, 4:])#40MHz
            ax2 = plt.subplot(gs[25:45, 4:])#37.5MHz
            ax3 = plt.subplot(gs[50:70, 4:])#35MHz
            ax4 = plt.subplot(gs[75:95, 4:])#32.5MHz
            ax5 = plt.subplot(gs[100:120, 4:])#30MHz


            for i in range(plot_days):
                idxes_BG = np.where((BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                if i == 0:
                    ax1.plot(BG_obs_time[idxes_BG], cali_dB_40_list[idxes_BG], '.--', label = '40MHz')
                    ax2.plot(BG_obs_time[idxes_BG], cali_dB_37_5_list[idxes_BG], '.--', label = '37.5MHz')
                    ax3.plot(BG_obs_time[idxes_BG], cali_dB_35_list[idxes_BG], '.--', label = '35MHz')
                    ax4.plot(BG_obs_time[idxes_BG], cali_dB_32_5_list[idxes_BG], '.--', label = '32.5MHz')
                    ax5.plot(BG_obs_time[idxes_BG], cali_dB_30_list[idxes_BG], '.--', label = '30MHz')

                else:
                    ax1.plot(BG_obs_time[idxes_BG], cali_dB_40_list[idxes_BG], '.--')
                    ax2.plot(BG_obs_time[idxes_BG], cali_dB_37_5_list[idxes_BG], '.--')
                    ax3.plot(BG_obs_time[idxes_BG], cali_dB_35_list[idxes_BG], '.--')
                    ax4.plot(BG_obs_time[idxes_BG], cali_dB_32_5_list[idxes_BG], '.--')
                    ax5.plot(BG_obs_time[idxes_BG], cali_dB_30_list[idxes_BG], '.--')


            full_idxes_BG = np.where((BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)))[0]
            ymax_cali_BG = np.nanmax([np.nanmax(cali_dB_40_list[full_idxes_BG]), np.nanmax(cali_dB_37_5_list[full_idxes_BG]), np.nanmax(cali_dB_35_list[full_idxes_BG]), np.nanmax(cali_dB_32_5_list[full_idxes_BG]), np.nanmax(cali_dB_30_list[full_idxes_BG])])
            ymin_cali_BG = np.nanmin([np.nanmin(cali_dB_40_list[full_idxes_BG]), np.nanmin(cali_dB_37_5_list[full_idxes_BG]), np.min(cali_dB_35_list[full_idxes_BG]), np.nanmin(cali_dB_32_5_list[full_idxes_BG]), np.nanmin(cali_dB_30_list[full_idxes_BG])])

            # plot_arange_dB_day([ax1, ax2, ax3, ax4], ymin_cali_BG, ymax_cali_BG)
            if plot_days == 1:
                Minute_fmt = mdates.DateFormatter('%H:%M')  
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(Minute_fmt)
                    ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=5), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=5))
                    ax.set_ylim(ymin_cali_BG-0.5, ymax_cali_BG+0.5)
                    ax.legend(fontsize = 12)
                    if not ax == ax5:
                        ax.tick_params(labelbottom=False,
                                        labelright=False,
                                        labeltop=False)
                # ax5.xaxis.set_major_formatter(Minute_fmt)
                # ax5.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=5), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=5))
            else:
                fmt = mdates.DateFormatter('%m/%d %H') 
                for ax in [ax1, ax2, ax3, ax4, ax5]:
                    ax.xaxis.set_major_formatter(fmt)
                    ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=45), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=45))
                    ax.set_ylim(ymin_cali_BG-0.5, ymax_cali_BG+0.5)
                    ax.legend(fontsize = 12)
                    if not ax == ax5:
                        ax.tick_params(labelbottom=False,
                                        labelright=False,
                                        labeltop=False)
            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.legend()
            #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
            # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
            plt.xlabel('Time', fontsize = 20)
            if plot_days == 1:
                ax1.set_title('Background analysis: ' + date, fontsize = 25)
            else:
                ax1.set_title('Background analysis: ' + date + ' - ' + (datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)).strftime(format='%Y%m%d'), fontsize = 25)
            ax3.set_ylabel('Intensity[dB]', fontsize = 20)
            plt.tick_params(axis='x', which='major', labelsize=15)
            plt.show()
            plt.close()

            if plot_days == 1:
                fig = plt.figure(figsize=(12.0, 12.0))
                gs = gridspec.GridSpec(121, 6)
    
                ax1 = plt.subplot(gs[0:20, :])#40MHz
                ax2 = plt.subplot(gs[25:45, :])#37.5MHz
                ax3 = plt.subplot(gs[50:70, :])#35MHz
                ax4 = plt.subplot(gs[75:95, :])#32.5MHz
                ax5 = plt.subplot(gs[100:120, :])#30MHz
    
                for i in range(plot_days):
                    idxes_BG = np.where((BG_obs_time>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(BG_obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                    cos_list = solar_cos_list(BG_obs_time[idxes_BG])
                    if i == 0:
                        ax1.plot(cos_list, cali_dB_40_list[idxes_BG], '.--', label = '40MHz')
                        ax2.plot(cos_list, cali_dB_37_5_list[idxes_BG], '.--', label = '37.5MHz')
                        ax3.plot(cos_list, cali_dB_35_list[idxes_BG], '.--', label = '35MHz')
                        ax4.plot(cos_list, cali_dB_32_5_list[idxes_BG], '.--', label = '32.5MHz')
                        ax5.plot(cos_list, cali_dB_30_list[idxes_BG], '.--', label = '30MHz')
    
                    # else:
                    #     ax1.plot(cos_list[idxes_BG], cali_dB_40_list[idxes_BG], '.--')
                    #     ax2.plot(cos_list[idxes_BG], cali_dB_37_5_list[idxes_BG], '.--')
                    #     ax3.plot(cos_list[idxes_BG], cali_dB_35_list[idxes_BG], '.--')
                    #     ax4.plot(cos_list[idxes_BG], cali_dB_32_5_list[idxes_BG], '.--')
                    #     ax5.plot(cos_list[idxes_BG], cali_dB_30_list[idxes_BG], '.--')
                
                if plot_days == 1:
                    # Minute_fmt = mdates.DateFormatter('%H:%M')  
                    for ax in [ax1, ax2, ax3, ax4, ax5]:
                        # ax.xaxis.set_major_formatter(Minute_fmt)
                        # ax.set_xlim(BG_obs_time[full_idxes_BG][0] - datetime.timedelta(minutes=5), BG_obs_time[full_idxes_BG][-1]+ datetime.timedelta(minutes=5))
                        ax.set_ylim(ymin_cali_BG-0.5, ymax_cali_BG+0.5)
                        ax.legend(fontsize = 12)
                        if not ax == ax5:
                            ax.tick_params(labelbottom=False,
                                            labelright=False,
                                            labeltop=False)
                        ax5.set_xlabel('cos', fontsize = '25')
                plt.show()
                            
                    

            
            
            
            
            
            # sys.exit()
    except:
        print ('Error: '+str(date))
        # sys.exit()
    # plt.plot()
    # print ('a')
    DATE+=pd.to_timedelta(3000,unit='day')


# def change_BG_40dB(BG_obs_time_each, event_check_days):
#     select_date = BG_obs_time_each
#     check_decibel = []
#     check_obs_time = []
#     if event_check_days == 1:
#         start_date = select_date
#         for i in range(1):
#             check_date = start_date
#             obs_index = np.where(BG_obs_time == getNearestValue(BG_obs_time,check_date))[0][0]
#             if abs(BG_obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
#                 check_decibel.append(cali_dB_40_list[obs_index])
#                 check_obs_time.append(BG_obs_time[obs_index])
#     else:
#         start_date = select_date - datetime.timedelta(days=event_check_days/2)
#         for i in range(event_check_days+1):
#             check_date = start_date + datetime.timedelta(days=i)
#             obs_index = np.where(BG_obs_time == getNearestValue(BG_obs_time,check_date))[0][0]
#             if abs(BG_obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
#                 check_decibel.append(cali_dB_40_list[obs_index])
#                 check_obs_time.append(BG_obs_time[obs_index])

#     # check_decibel = np.array(check_decibel)
#     # check_obs_time = np.array(check_obs_time)
#     # plt.plot(check_obs_time, check_decibel, '.')
#     # plt.axhline(np.nanpercentile(check_decibel, 50, axis = 0), ls = "-.", color = "magenta")
#     # plt.show()
#     # plt.close()

#     return np.nanpercentile(check_decibel, 50, axis = 0)

# BG_40dB_move = []
# event_check_days = 30
# for i in range(len(BG_obs_time)):
#     print (BG_obs_time[i])
#     # if BG_obs_time[i] > datetime.datetime(int(2020),int(1),int(10)):
#         # if BG_obs_time[i] < datetime.datetime(int(2020),int(1),int(20)):
#             # BG_40dB_each = change_BG_40dB(BG_obs_time[i], event_check_days)
#     BG_40dB_move.append(change_BG_40dB(BG_obs_time[i], event_check_days))
# BG_40dB_move = np.array(BG_40dB_move)

# for i in [0,1,2,5,6,7,10,11,12,13]:
# # for i in [6]:
#     # for j in range(366):
#     fig = plt.figure(figsize=(18.0, 12.0))
#     gs = gridspec.GridSpec(121, 2)
#     for j in range(12):
#         if j <= 5:
#             ax = plt.subplot(gs[20*j:11+20*j, :1])
#         else:
#             ax = plt.subplot(gs[0+20*(j-6):11+20*(j-6), 1:])
#         idx = np.where((BG_obs_time >= datetime.datetime(int(2007+i),int(1+j),int(1))) & (BG_obs_time <= datetime.datetime(int(2007+i),int(1+j),int(2))))[0]
#         if len(idx)>0:
#             ax.plot(BG_obs_time[idx], BG_40dB_move[idx], '.-', label = 'Move')
#             max_index = np.where(BG_40dB_move[idx]==np.max(BG_40dB_move[idx]))[0]
#             ax.scatter(BG_obs_time[idx][max_index], BG_40dB_move[idx][max_index], c = 'r')
#             print (BG_obs_time[idx][max_index], BG_40dB_move[idx][max_index])
#             # plt.plot(BG_obs_time[idx], cali_dB_40_list[idx], '.-', label = 'Nonmove')
#             ax.set_title(datetime.datetime(int(2007+i),int(1+j),int(1)).strftime(format='%Y%m%d'))
#             ax.set_xlim(datetime.datetime(int(2007+i),int(1+j),int(1),7), datetime.datetime(int(2007+i),int(1+j),int(1),18))
#             # plt.legend()
#             ax.tick_params(axis='x', labelrotation=20)
#             ax.set_ylim(20.2,23.8)
#     plt.show()
#     plt.close()


# year_list = [0,1,2,5,6,7,10,11,12,13]
# for j in range(12):
#     fig = plt.figure(figsize=(18.0, 12.0))
#     gs = gridspec.GridSpec(101, 2)
#     for i in range(len(year_list)):
# # for i in [6]:
#     # for j in range(366):
    
#         if i <= 4:
#             ax = plt.subplot(gs[20*i:11+20*i, :1])
#         else:
#             ax = plt.subplot(gs[0+20*(i-5):11+20*(i-5), 1:])
#         idx = np.where((BG_obs_time >= datetime.datetime(int(2007+year_list[i]),int(1+j),int(15))) & (BG_obs_time <= datetime.datetime(int(2007+year_list[i]),int(1+j),int(16))))[0]
#         if len(idx)>0:
#             ax.plot(BG_obs_time[idx], BG_40dB_move[idx], '.-', label = 'Move')
#             max_index = np.where(BG_40dB_move[idx]==np.max(BG_40dB_move[idx]))[0]
#             ax.scatter(BG_obs_time[idx][max_index], BG_40dB_move[idx][max_index], c = 'r')
#             # print (BG_obs_time[idx][max_index], BG_40dB_move[idx][max_index])
#             # plt.plot(BG_obs_time[idx], cali_dB_40_list[idx], '.-', label = 'Nonmove')
#             ax.set_title(datetime.datetime(int(2007+year_list[i]),int(1+j),int(15)).strftime(format='%Y%m%d'))
#             ax.set_xlim(datetime.datetime(int(2007+year_list[i]),int(1+j),int(15),7), datetime.datetime(int(2007+year_list[i]),int(1+j),int(15),18))
#             # plt.legend()
#             ax.tick_params(axis='x', labelrotation=20)
#             ax.set_ylim(20.2,23.8)
#     plt.show()
#     plt.close()
