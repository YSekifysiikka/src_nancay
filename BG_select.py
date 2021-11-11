#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 13:23:49 2021

@author: yuichiro
"""

import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dateutil.relativedelta import relativedelta
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

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_40MHz_final.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

obs_time = []
decibel_list = []
for i in range(len(antenna1_csv)):
    obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    decibel_list.append(antenna1_csv['decibel'][i])

obs_time = np.array(obs_time)
decibel_list = np.array(decibel_list)
select_date = datetime.datetime(2012,1,15,12,30)


check_decibel = []
check_obs_time = []

event_check_days = 30
start_date = select_date - datetime.timedelta(days=event_check_days/2)
for i in range(event_check_days+1):
    check_date = start_date + datetime.timedelta(days=i)
    obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
    if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
        check_decibel.append(decibel_list[obs_index])
        check_obs_time.append(obs_time[obs_index])

check_decibel = np.array(check_decibel)
check_obs_time = np.array(check_obs_time)

plot_range = 2
min_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date - datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
max_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date + datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
dB_max = max(decibel_list[min_obs_index:max_obs_index + 1])
dB_min = min(decibel_list[min_obs_index:max_obs_index + 1])
plt.close()
fig=plt.figure(1,figsize=(8,4))
ax1 = fig.add_subplot(311) 
ax1.plot(obs_time, decibel_list,'.')

ax1.xaxis_date()
date_format = mdates.DateFormatter('%m-%d')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
ax1.set_ylim([dB_min, dB_max])
ax1.set_ylabel('Decibel [dB]',fontsize=10)
ax1.set_title('Time varidation of BG around ' + select_date.strftime("%Y/%m/%d"))


obs_index_final = np.where(check_decibel == getNearestValue(check_decibel,np.median(check_decibel)))[0][0]
ax2 = fig.add_subplot(313)
ax2.plot(check_obs_time,check_decibel,'.')
ax2.axhline(np.median(check_decibel), ls = "--", color = "magenta", label = check_obs_time[obs_index_final].strftime("%Y/%m/%d %H:%M") + ' :' + str(np.median(check_decibel))[:4] + '[dB]')
ax2.xaxis_date()
date_format = mdates.DateFormatter('%m-%d')
ax2.xaxis.set_major_formatter(date_format)
ax2.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
ax2.set_ylim([dB_min, dB_max])
ax2.set_ylabel('Decibel [dB]',fontsize=10)
ax2.set_title('Time varidation of BG around ' + select_date.strftime("%H:%M"))
ax2.legend(fontsize = 8, loc = 'upper right')
# plt.show()
plt.close()






































# for j in range(30):
#     obs_time = np.array(obs_time)
#     decibel_list = np.array(decibel_list)
#     select_date = datetime.datetime(2012,1,15,12,30) + relativedelta(months=j)

    
#     check_decibel = []
#     check_obs_time = []
    
#     event_check_days = 30
#     start_date = select_date - datetime.timedelta(days=event_check_days/2)
#     for i in range(event_check_days+1):
#         check_date = start_date + datetime.timedelta(days=i)
#         obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
#         if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
#             check_decibel.append(decibel_list[obs_index])
#             check_obs_time.append(obs_time[obs_index])

#     check_decibel = np.array(check_decibel)
#     check_obs_time = np.array(check_obs_time)

#     plot_range = 2
#     min_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date - datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
#     max_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date + datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
#     dB_max = max(decibel_list[min_obs_index:max_obs_index + 1])
#     dB_min = min(decibel_list[min_obs_index:max_obs_index + 1])
#     plt.close()
#     fig=plt.figure(1,figsize=(8,4))
#     ax1 = fig.add_subplot(311) 
#     ax1.plot(obs_time, decibel_list,'.')
    
#     ax1.xaxis_date()
#     date_format = mdates.DateFormatter('%m-%d')
#     ax1.xaxis.set_major_formatter(date_format)
#     ax1.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
#     ax1.set_ylim([dB_min, dB_max])
#     ax1.set_ylabel('Decibel [dB]',fontsize=10)
#     ax1.set_title('Time varidation of BG around ' + select_date.strftime("%Y/%m/%d"))
    
    
#     obs_index_final = np.where(check_decibel == getNearestValue(check_decibel,np.median(check_decibel)))[0][0]
#     ax2 = fig.add_subplot(313)
#     ax2.plot(check_obs_time,check_decibel,'.')
#     ax2.axhline(np.median(check_decibel), ls = "--", color = "magenta", label = check_obs_time[obs_index_final].strftime("%Y/%m/%d %H:%M") + ' :' + str(np.median(check_decibel))[:4] + '[dB]')
#     ax2.xaxis_date()
#     date_format = mdates.DateFormatter('%m-%d')
#     ax2.xaxis.set_major_formatter(date_format)
#     ax2.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
#     ax2.set_ylim([dB_min, dB_max])
#     ax2.set_ylabel('Decibel [dB]',fontsize=10)
#     ax2.set_title('Time varidation of BG around ' + select_date.strftime("%H:%M"))
#     ax2.legend(fontsize = 8, loc = 'upper right')
#     plt.show()


#test


obs_time = []
decibel_list = []
for i in range(len(antenna1_csv)):
    obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    decibel_list.append(antenna1_csv['decibel'][i])


day_night_median_list = []
day_night = [10,12]
for day_night_check in day_night:
    median_time_list = []
    medial_decibel_list = []
    for j in range(36):
        obs_time = np.array(obs_time)
        decibel_list = np.array(decibel_list)
        select_date = datetime.datetime(2007,1,15,day_night_check,30) + relativedelta(months=j)
    
        
        check_decibel = []
        check_obs_time = []
    
        event_check_days = 30
        start_date = select_date - datetime.timedelta(days=event_check_days/2)
        for i in range(event_check_days+1):
            check_date = start_date + datetime.timedelta(days=i)
            obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
            if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])
        if len(check_decibel)>0:
            check_decibel = np.array(check_decibel)
            check_obs_time = np.array(check_obs_time)
            
            plot_range = 2
            min_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date - datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
            max_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date + datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
            dB_max = max(decibel_list[min_obs_index:max_obs_index + 1])
            dB_min = min(decibel_list[min_obs_index:max_obs_index + 1])
            plt.close()
            fig=plt.figure(1,figsize=(8,4))
            ax1 = fig.add_subplot(311) 
            ax1.plot(obs_time, decibel_list,'.')
            
            ax1.xaxis_date()
            date_format = mdates.DateFormatter('%m-%d')
            ax1.xaxis.set_major_formatter(date_format)
            ax1.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
            ax1.set_ylim([dB_min, dB_max])
            ax1.set_ylabel('Decibel [dB]',fontsize=10)
            ax1.set_title('Time varidation of BG around ' + select_date.strftime("%Y/%m/%d"))
            
            
            obs_index_final = np.where(check_decibel == getNearestValue(check_decibel,np.median(check_decibel)))[0][0]
            ax2 = fig.add_subplot(313)
            ax2.plot(check_obs_time,check_decibel,'.')
            ax2.axhline(np.median(check_decibel), ls = "--", color = "magenta", label = check_obs_time[obs_index_final].strftime("%Y/%m/%d %H:%M") + ' :' + str(np.median(check_decibel))[:4] + '[dB]')
            ax2.xaxis_date()
            date_format = mdates.DateFormatter('%m-%d')
            ax2.xaxis.set_major_formatter(date_format)
            ax2.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
            ax2.set_ylim([dB_min, dB_max])
            ax2.set_ylabel('Decibel [dB]',fontsize=10)
            ax2.set_title('Time varidation of BG around ' + select_date.strftime("%H:%M"))
            ax2.legend(fontsize = 8, loc = 'upper right')
            # plt.show()
            median_time_list.append(check_obs_time[obs_index_final])
            medial_decibel_list.append(np.median(check_decibel))

        else:
            check_decibel = np.nan
            check_obs_time = np.nan
            median_time_list.append(select_date)
            medial_decibel_list.append(check_decibel)
    

    
    plt.close()
    fig=plt.figure(1,figsize=(8,4))
    ax1 = fig.add_subplot(111) 
    ax1.plot(median_time_list, medial_decibel_list)
    ax1.axhline(22, ls = "--", color = "magenta", label = '22' + '[dB]')
    date_format = mdates.DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax1.set_ylabel('Decibel [dB]',fontsize=10)
    ax1.set_title('Time varidation of BG around ' + select_date.strftime("%H:%M"))
    ax1.legend(fontsize = 8, loc = 'upper right')
    plt.xticks(rotation=45)
    plt.show()
    day_night_median_list.append(medial_decibel_list)
    plt.close()

day_night_median_list = np.array(day_night_median_list)
plt.close()
fig=plt.figure(1,figsize=(8,4))
ax1 = fig.add_subplot(111) 
ax1.plot(median_time_list, np.array(day_night_median_list[0])-np.array(day_night_median_list[1]))
# ax1.axhline(22, ls = "--", color = "magenta", label = '22' + '[dB]')
date_format = mdates.DateFormatter('%Y-%m-%d')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_ylabel('Decibel [dB]',fontsize=10)
ax1.set_title('Time varidation of BG Gap between day and night')
plt.xticks(rotation=45)
plt.show()
    
    
    
