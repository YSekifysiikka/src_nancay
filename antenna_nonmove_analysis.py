#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:50:32 2021

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
import sys
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import os

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



def plot_arange_year(ax_list, idxes):
    for ax in ax_list:
        ax.set_xlim(obs_time[idxes][0] - datetime.timedelta(days=1), obs_time[idxes][-1]+ datetime.timedelta(days=1))
        ax.set_ylim(ymin-0.5, ymax+0.5)
        ax.legend(fontsize = 12)
        ax.tick_params(labelbottom=False,
                       labelright=False,
                       labeltop=False)
    return
def plot_arange_day(ax_list):
    for ax in ax_list:
        if plot_days == 1:
            ax.set_xlim(obs_time_list_each[0] - datetime.timedelta(minutes=5), obs_time_list_each[-1]+ datetime.timedelta(minutes=5))
            ax.set_ylim(ymin-0.5, ymax+0.5)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                           labelright=False,
                           labeltop=False)
        else:
            ax.set_xlim(obs_time_list_each[0] - datetime.timedelta(minutes=45), obs_time_list_each[-1]+ datetime.timedelta(minutes=45))
            ax.set_ylim(ymin-0.5, ymax+0.5)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                           labelright=False,
                           labeltop=False)
    return

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



dB_30_list = []
dB_32_5_list = []
dB_35_list = []
dB_37_5_list = []
dB_40_list = []



#日変化
check_data = 7
event_check_days = 1
plot_days = 1


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
        obs_time_list_each = []
        dB_30_list_each = []
        dB_32_5_list_each = []
        dB_35_list_each = []
        dB_37_5_list_each = []
        dB_40_list_each = []
    
    
        if len(obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]) > 0:
            for obs_time_each in obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]:
                # print (obs_time_each)
                obs_time_list_each.append(obs_time_each)
                # cos = solar_cos(obs_time_each)
                HH = obs_time_each.strftime(format='%H')
                MM = obs_time_each.strftime(format='%M')
                BG_list = check_BG(date, HH, MM, event_check_days)
                db_40 = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
                around_40 = np.nanmedian(BG_list[int(db_40-((check_data-1)/2)):int(db_40+((check_data-1)/2)+1)], axis =0)
                dB_40_list.append(around_40)
                dB_40_list_each.append(around_40)
                db_35 = np.where(Frequency == getNearestValue(Frequency,35))[0][0]
                around_35 = np.nanmedian(BG_list[int(db_35-((check_data-1)/2)):int(db_35+((check_data-1)/2)+1)], axis =0)
                dB_35_list.append(around_35)
                dB_35_list_each.append(around_35)
                db_30 = np.where(Frequency == getNearestValue(Frequency,30))[0][0]
                around_30 = np.nanmedian(BG_list[int(db_30-((check_data-1)/2)):int(db_30+((check_data-1)/2)+1)], axis =0)
                dB_30_list.append(around_30)
                dB_30_list_each.append(around_30)
                db_37_5 = np.where(Frequency == getNearestValue(Frequency,37.5))[0][0]
                around_37_5 = np.nanmedian(BG_list[int(db_37_5-((check_data-1)/2)):int(db_37_5+((check_data-1)/2)+1)], axis =0)
                dB_37_5_list.append(around_37_5)
                dB_37_5_list_each.append(around_37_5)
                db_32_5 = np.where(Frequency == getNearestValue(Frequency,32.5))[0][0]
                around_32_5 = np.nanmedian(BG_list[int(db_32_5-((check_data-1)/2)):int(db_32_5+((check_data-1)/2)+1)], axis =0)
                dB_32_5_list.append(around_32_5)
                dB_32_5_list_each.append(around_32_5)
            obs_time_list_each = np.array(obs_time_list_each)
            dB_30_list_each = np.array(dB_30_list_each )
            dB_32_5_list_each = np.array(dB_32_5_list_each)
            dB_35_list_each = np.array(dB_35_list_each)
            dB_37_5_list_each = np.array(dB_37_5_list_each)
            dB_40_list_each = np.array(dB_40_list_each)
            fig = plt.figure(figsize=(18.0, 12.0))
            gs = gridspec.GridSpec(121, 1)
            ax5 = plt.subplot(gs[0:20, :])#40MHz
            ax4 = plt.subplot(gs[25:45, :])#37.5MHz
            ax3 = plt.subplot(gs[50:70, :])#35MHz
            ax2 = plt.subplot(gs[75:95, :])#32.5MHz
            ax1 = plt.subplot(gs[100:120, :])#30MHz
            
            # if plot_days == 1:
            #     ax1.plot(obs_time_list_each, dB_30_list_each, '.--', label = '30MHz')
            #     ax2.plot(obs_time_list_each, dB_32_5_list_each, '.--', label = '32.5MHz')
            #     ax3.plot(obs_time_list_each, dB_35_list_each, '.--', label = '35MHz')
            #     ax4.plot(obs_time_list_each, dB_37_5_list_each, '.--', label = '37.5MHz')
            #     ax5.plot(obs_time_list_each, dB_40_list_each, '.--', label = '40MHz')
            # else:
            for i in range(plot_days):
                idxes = np.where((obs_time_list_each>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(obs_time_list_each<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                if i == 0:
                    ax1.plot(obs_time_list_each[idxes], dB_30_list_each[idxes], '.--', label = '30MHz')
                    ax2.plot(obs_time_list_each[idxes], dB_32_5_list_each[idxes], '.--', label = '32.5MHz')
                    ax3.plot(obs_time_list_each[idxes], dB_35_list_each[idxes], '.--', label = '35MHz')
                    ax4.plot(obs_time_list_each[idxes], dB_37_5_list_each[idxes], '.--', label = '37.5MHz')
                    ax5.plot(obs_time_list_each[idxes], dB_40_list_each[idxes], '.--', label = '40MHz')
                else:
                    ax1.plot(obs_time_list_each[idxes], dB_30_list_each[idxes], '.--')
                    ax2.plot(obs_time_list_each[idxes], dB_32_5_list_each[idxes], '.--')
                    ax3.plot(obs_time_list_each[idxes], dB_35_list_each[idxes], '.--')
                    ax4.plot(obs_time_list_each[idxes], dB_37_5_list_each[idxes], '.--')
                    ax5.plot(obs_time_list_each[idxes], dB_40_list_each[idxes], '.--')
            ymax = np.max([np.max(dB_30_list_each), np.max(dB_32_5_list_each), np.max(dB_35_list_each), np.max(dB_37_5_list_each), np.max(dB_40_list_each)])
            ymin = np.min([np.min(dB_30_list_each), np.min(dB_32_5_list_each), np.min(dB_35_list_each), np.min(dB_37_5_list_each), np.min(dB_40_list_each)])
            plot_arange_day([ax2, ax3, ax4, ax5])
            if plot_days == 1:
                Minute_fmt = mdates.DateFormatter('%H:%M')  
                ax1.xaxis.set_major_formatter(Minute_fmt)
                ax1.set_xlim(obs_time_list_each[0] - datetime.timedelta(minutes=5), obs_time_list_each[-1]+ datetime.timedelta(minutes=5))
            else:
                fmt = mdates.DateFormatter('%m-%d %H') 
                ax1.xaxis.set_major_formatter(fmt)
                ax1.set_xlim(obs_time_list_each[0] - datetime.timedelta(minutes=45), obs_time_list_each[-1]+ datetime.timedelta(minutes=45))
                
            ax1.set_ylim(ymin-0.5, ymax+0.5)
            ax1.legend(fontsize = 12)
            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.legend()
            #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
            # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
            plt.xlabel('Time', fontsize = 20)
            if plot_days == 1:
                ax5.set_title('Antenna analysis: ' + date, fontsize = 25)
            else:
                ax5.set_title('Antenna analysis: ' + date + ' - ' + (datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)).strftime(format='%Y%m%d'), fontsize = 25)
            ax3.set_ylabel('Intensity[dB]', fontsize = 20)
            plt.tick_params(axis='x', which='major', labelsize=15)
            if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm):
                os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm)
            filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm + '/'+date+'.png'
            plt.savefig(filename)
            plt.show()
            plt.close()
    except:
        print ('Error: '+str(date))
    # plt.plot()
    # print ('a')
    DATE+=pd.to_timedelta(1,unit='day')



# #年変化
date_in=[20070101,20201231]
start_day,end_day=date_in

dB_30_list = np.array(dB_30_list)
dB_32_5_list = np.array(dB_32_5_list)
dB_35_list = np.array(dB_35_list)
dB_37_5_list = np.array(dB_37_5_list)
dB_40_list = np.array(dB_40_list)


while start_day <= end_day:
    
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    DATE=sdate
    date=DATE.strftime(format='%Y%m%d')
    print(date)
    yyyy = date[:4]
    mm = date[4:6]
    dd = date[6:8]
    try:
        idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy)+1,int(mm),int(dd)) - datetime.timedelta(days=1)))[0]
        if len(idxes) > 0:
            fig = plt.figure(figsize=(18.0, 12.0))
            gs = gridspec.GridSpec(121, 1)
            ax5 = plt.subplot(gs[0:20, :])#40MHz
            ax4 = plt.subplot(gs[25:45, :])#37.5MHz
            ax3 = plt.subplot(gs[50:70, :])#35MHz
            ax2 = plt.subplot(gs[75:95, :])#32.5MHz
            ax1 = plt.subplot(gs[100:120, :])#30MHz

            ax1.plot(obs_time[idxes], dB_30_list[idxes], '.--', label = '30MHz')
            ax2.plot(obs_time[idxes], dB_32_5_list[idxes], '.--', label = '32.5MHz')
            ax3.plot(obs_time[idxes], dB_35_list[idxes], '.--', label = '35MHz')
            ax4.plot(obs_time[idxes], dB_37_5_list[idxes], '.--', label = '37.5MHz')
            ax5.plot(obs_time[idxes], dB_40_list[idxes], '.--', label = '40MHz')
            
            ymax = np.max([np.max(dB_30_list[idxes]), np.max(dB_32_5_list[idxes]), np.max(dB_35_list[idxes]), np.max(dB_37_5_list[idxes]), np.max(dB_40_list[idxes])])
            ymin = np.min([np.min(dB_30_list[idxes]), np.min(dB_32_5_list[idxes]), np.min(dB_35_list[idxes]), np.min(dB_37_5_list[idxes]), np.min(dB_40_list[idxes])])
            plot_arange_year([ax2, ax3, ax4, ax5],idxes)

            fmt = mdates.DateFormatter('%m/%d') 
            ax1.xaxis.set_major_formatter(fmt)
            ax1.set_xlim(obs_time[idxes][0] - datetime.timedelta(days=1), obs_time[idxes][-1]+ datetime.timedelta(days=1))
            ax1.set_ylim(ymin-0.5, ymax+0.5)
            ax1.legend(fontsize = 12)
            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.legend()
            #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
            # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
            plt.xlabel('Time', fontsize = 20)
            ax5.set_title('Antenna analysis: ' + date + ' - ' + str((datetime.datetime(int(yyyy)+1,int(mm),int(dd))- datetime.timedelta(days=1)).strftime(format='%Y%m%d')), fontsize = 25)
            ax3.set_ylabel('Intensity[dB]', fontsize = 20)
            plt.tick_params(axis='x', which='major', labelsize=15)
            # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm):
            #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm)
            # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm + '/'+date+'.png'
            # plt.savefig(filename)
            plt.show()
            plt.close()
    except:
        pass
    start_day+=10000


