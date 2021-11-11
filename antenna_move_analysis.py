#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:12:31 2021

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

def plot_arange_year(ax_list):
    for ax in ax_list:
        ax.set_xlim(obs_time[idxes][0] - datetime.timedelta(days=1), obs_time[idxes][-1]+ datetime.timedelta(days=1))
        ax.set_ylim(ymin-0.5, ymax+0.5)
        ax.legend(fontsize = 12)
        ax.tick_params(labelbottom=False,
                       labelright=False,
                       labeltop=False)
    return


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/test/all_antenna_move30days.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

obs_time = []
dB_30_list = []
dB_32_5_list = []
dB_35_list = []
dB_37_5_list = []
dB_40_list = []

for i in range(len(antenna1_csv)):
    obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    dB_30_list.append(antenna1_csv['30MHz[dB]'][i])
    dB_32_5_list.append(antenna1_csv['32.5MHz[dB]'][i])
    dB_35_list.append(antenna1_csv['35MHz[dB]'][i])
    dB_37_5_list.append(antenna1_csv['37.5MHz[dB]'][i])
    dB_40_list.append(antenna1_csv['40MHz[dB]'][i])

obs_time = np.array(obs_time)
dB_30_list = np.array(dB_30_list)
dB_32_5_list = np.array(dB_32_5_list)
dB_35_list = np.array(dB_35_list)
dB_37_5_list = np.array(dB_37_5_list)
dB_40_list = np.array(dB_40_list)


# #年変化
# date_in=[20070101,20201231]
# start_day,end_day=date_in
# while start_day <= end_day:
    
#     sdate=pd.to_datetime(start_day,format='%Y%m%d')
#     edate=pd.to_datetime(end_day,format='%Y%m%d')
#     DATE=sdate
#     date=DATE.strftime(format='%Y%m%d')
#     print(date)
#     yyyy = date[:4]
#     mm = date[4:6]
#     dd = date[6:8]
#     try:
#         idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy)+1,int(mm),int(dd)) - datetime.timedelta(days=1)))[0]
#         if len(idxes) > 0:
#             fig = plt.figure(figsize=(18.0, 12.0))
#             gs = gridspec.GridSpec(121, 1)
#             ax5 = plt.subplot(gs[0:20, :])#40MHz
#             ax4 = plt.subplot(gs[25:45, :])#37.5MHz
#             ax3 = plt.subplot(gs[50:70, :])#35MHz
#             ax2 = plt.subplot(gs[75:95, :])#32.5MHz
#             ax1 = plt.subplot(gs[100:120, :])#30MHz

#             ax1.plot(obs_time[idxes], dB_30_list[idxes], '.--', label = '30MHz')
#             ax2.plot(obs_time[idxes], dB_32_5_list[idxes], '.--', label = '32.5MHz')
#             ax3.plot(obs_time[idxes], dB_35_list[idxes], '.--', label = '35MHz')
#             ax4.plot(obs_time[idxes], dB_37_5_list[idxes], '.--', label = '37.5MHz')
#             ax5.plot(obs_time[idxes], dB_40_list[idxes], '.--', label = '40MHz')
            
#             ymax = np.max([np.max(dB_30_list[idxes]), np.max(dB_32_5_list[idxes]), np.max(dB_35_list[idxes]), np.max(dB_37_5_list[idxes]), np.max(dB_40_list[idxes])])
#             ymin = np.min([np.min(dB_30_list[idxes]), np.min(dB_32_5_list[idxes]), np.min(dB_35_list[idxes]), np.min(dB_37_5_list[idxes]), np.min(dB_40_list[idxes])])
#             plot_arange_year([ax2, ax3, ax4, ax5])

#             fmt = mdates.DateFormatter('%m/%d') 
#             ax1.xaxis.set_major_formatter(fmt)
#             ax1.set_xlim(obs_time[idxes][0] - datetime.timedelta(days=1), obs_time[idxes][-1]+ datetime.timedelta(days=1))
#             ax1.set_ylim(ymin-0.5, ymax+0.5)
#             ax1.legend(fontsize = 12)
#             # for ax in [ax1, ax2, ax3, ax4, ax5]:
#             #     ax.legend()
#             #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
#             # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
#             plt.xlabel('Time', fontsize = 20)
#             ax5.set_title('Antenna analysis: ' + date + ' - ' + str((datetime.datetime(int(yyyy)+1,int(mm),int(dd))- datetime.timedelta(days=1)).strftime(format='%Y%m%d')), fontsize = 25)
#             ax3.set_ylabel('Intensity[dB]', fontsize = 20)
#             plt.tick_params(axis='x', which='major', labelsize=15)
#             # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm):
#             #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm)
#             # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm + '/'+date+'.png'
#             # plt.savefig(filename)
#             plt.show()
#             plt.close()
#     except:
#         pass
#     start_day+=10000


# 日変化
plot_days = 10

date_in=[20200222,20200310]
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
        obs_time_list_each = []
        dB_30_list_each = []
        dB_32_5_list_each = []
        dB_35_list_each = []
        dB_37_5_list_each = []
        dB_40_list_each = []
        if len(obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]) > 0:
            for obs_time_each in obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]:
                idx = np.where(obs_time == obs_time_each)[0][0]
                obs_time_list_each.append(obs_time[idx])
                dB_30_list_each.append(dB_30_list[idx])
                dB_32_5_list_each.append(dB_32_5_list[idx])
                dB_35_list_each.append(dB_35_list[idx])
                dB_37_5_list_each.append(dB_37_5_list[idx])
                dB_40_list_each.append(dB_40_list[idx])
            fig = plt.figure(figsize=(18.0, 12.0))
            gs = gridspec.GridSpec(121, 1)
            ax5 = plt.subplot(gs[0:20, :])#40MHz
            ax4 = plt.subplot(gs[25:45, :])#37.5MHz
            ax3 = plt.subplot(gs[50:70, :])#35MHz
            ax2 = plt.subplot(gs[75:95, :])#32.5MHz
            ax1 = plt.subplot(gs[100:120, :])#30MHz
            
            if plot_days == 1:
                ax1.plot(obs_time_list_each, dB_30_list_each, '.--', label = '30MHz')
                ax2.plot(obs_time_list_each, dB_32_5_list_each, '.--', label = '32.5MHz')
                ax3.plot(obs_time_list_each, dB_35_list_each, '.--', label = '35MHz')
                ax4.plot(obs_time_list_each, dB_37_5_list_each, '.--', label = '37.5MHz')
                ax5.plot(obs_time_list_each, dB_40_list_each, '.--', label = '40MHz')
            else:
                for i in range(plot_days):
                    idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                    if i == 0:
                        ax1.plot(obs_time[idxes], dB_30_list[idxes], '.--', label = '30MHz')
                        ax2.plot(obs_time[idxes], dB_32_5_list[idxes], '.--', label = '32.5MHz')
                        ax3.plot(obs_time[idxes], dB_35_list[idxes], '.--', label = '35MHz')
                        ax4.plot(obs_time[idxes], dB_37_5_list[idxes], '.--', label = '37.5MHz')
                        ax5.plot(obs_time[idxes], dB_40_list[idxes], '.--', label = '40MHz')
                    else:
                        ax1.plot(obs_time[idxes], dB_30_list[idxes], '.--')
                        ax2.plot(obs_time[idxes], dB_32_5_list[idxes], '.--')
                        ax3.plot(obs_time[idxes], dB_35_list[idxes], '.--')
                        ax4.plot(obs_time[idxes], dB_37_5_list[idxes], '.--')
                        ax5.plot(obs_time[idxes], dB_40_list[idxes], '.--')
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
            if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm):
                os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm)
            filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days/'+yyyy + '/' + mm + '/'+date+'.png'
            plt.savefig(filename)
            plt.show()
            plt.close()

    except:
        pass
    DATE+=pd.to_timedelta(10,unit='day')







