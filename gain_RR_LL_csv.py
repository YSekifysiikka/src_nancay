#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:50:51 2020

@author: yuichiro
"""
import sys
sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
import cdflib
import scipy
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy import signal
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
import datetime
from astropy.time import TimeDelta, Time
from astropy import units as u
import os
import math
import glob
import matplotlib.gridspec as gridspec
import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


###############
sigma_range = 1
sigma_start = 2
sigma_mul = 1
#when you only check one threshold
sigma_value = 2
after_plot = str('drift_check')
after_after_plot = str('time_check')
#x_start_range = 10
#x_end_range = 79.825
#x_whole_range = 400
x_start_range = 29.95
x_end_range = 79.825
x_whole_range = 286
#check_frequency_range = 400
check_frequency_range = 286
time_band = 340
time_co = 60
move_ave = 3
median_size = 1
duration = 7
threshold_frequency = 20
freq_check_range = 20
threshold_frequency_final = 3 * freq_check_range
###############

def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res

def file_generator(file):
    with open(file, encoding="utf-8") as f:
        for line in f:
            yield line

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
date_in=[20140101, 20141231]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')
DATE=sdate

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/LL_gain/new_ver2_gain_analysis_'+str(date_in[0])+'_'+str(date_in[1])+'.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "Frequency", "Right-gain", "Right-Trx", "Right-hot_dB", "Right-cold_dB", "Left-gain", "Left-Trx", "Left-hot_dB", "Left-cold_dB"])
    w.writeheader()
    
        
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        try:
            yyyy = date[:4]
            mm = date[4:6]
            file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[10]
            file_name_separate =file_name.split('_')
            Date_start = file_name_separate[5]
            date_OBs=str(Date_start)
            year=date_OBs[0:4]
            month=date_OBs[4:6]
            day=date_OBs[6:8]
            start_h = date_OBs[8:10]
            start_m = date_OBs[10:12]
            Date_stop = file_name_separate[6]
            end_h = Date_stop[8:10]
            end_m = Date_stop[10:12]
            file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
            cdf_file = cdflib.CDF(file)
            epoch = cdf_file['Epoch'] 
            epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
            Frequency = cdf_file['Frequency']
            resolution = round((Frequency[1]-Frequency[0]),3)
            LL = cdf_file['LL'] 
            RR = cdf_file['RR'] 
            
            data_r_0 = RR
            data_l_0 = LL
            diff_r_last =(data_r_0).T
            diff_r_last = np.flipud(diff_r_last) * 0.3125
            diff_l_last =(data_l_0).T
            diff_l_last = np.flipud(diff_l_last) * 0.3125
        
            # diff_P = (((10 ** ((diff_r_last[:,:])/10)) + (10 ** ((diff_l_last[:,:])/10)))/2)
            # diff_db = np.log10(diff_P) * 10
            ####################
            # y_power = []
            # num = int(move_ave)
            # b = np.ones(num)/num
            # for i in range (diff_db.shape[0]):
            #     y_power.append(np.convolve(diff_P[i], b, mode='valid'))
            # y_power = np.array(y_power)
            # diff_move_db = np.log10(y_power) * 10
            ####################
            # min_power = np.amin(y_power, axis=1)
            # min_db = np.log10(min_power) * 10
            # diff_db_min_med = (diff_move_db.T - min_db).T
        
        
        
            status = np.where(cdf_file['Status'] == 0)[0]
            for i in status:
                if i < 50:
                    if i > 0:
                        cali_list = []
                        cali_time = []
                        cali_1 = np.where((cdf_file['Status'][0:i + 2] == 17))
                        cali_2 = np.where((cdf_file['Status'][0:i + 2] == 0))
                        if len(cali_1[0]) > 0:
                            for j in range (len(cali_1[0])):
                                cali_list.append(cdf_file['Status'][0:i + 2][cali_1[0][j]][cali_1[1][j]])
                                cali_time.append(cali_1[0][j])
            
                        if len(cali_2[0]) > 0:
                            for j in range (len(cali_2[0])):
                                cali_list.append(cdf_file['Status'][0:i + 2][cali_2[0][j]][cali_2[1][j]])
                                cali_time.append(cali_2[0][j])
            
                        cali_list = np.array(cali_list)
                        cali_time = np.array(cali_time)
                        
                        cali_list_f = []
            
                        for j in range(len(cali_time)):
                            cali_list_f.append(cali_list[np.argsort(cali_time)[j]])
                        cali_time_f = np.sort(cali_time)
                        cali_list_f = np.array(cali_list_f)
            
            
                        # print (cdf_file['Status'][0:i+2])
                        time = 0
                        # print (time)
    
                        #+1 is due to move_average
                        start = epoch[time + 1]
                        start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
                        # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
                        time = i
                        end = epoch[time + 1]
                        end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
                        # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
                        Time_start = start.strftime('%H:%M:%S')
                        Time_end = end.strftime('%H:%M:%S')
    
                        # print (start)
                        print(Time_start+'-'+Time_end)
                        start = start.timestamp()
                        end = end.timestamp()
                        x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
                        x_lims = mdates.date2num(x_lims)
                
                        # Set some generic y-limits.
                        y_lims = [Frequency[0], Frequency[-1]]
                        
            
                        RR_sep = diff_r_last[:, 0:i + 2]
                        LL_sep = diff_l_last[:, 0:i + 2]
        
        
                else:
                    cali_list = []
                    cali_time = []
                    cali_1 = np.where((cdf_file['Status'][i-41:i+2] == 17))
                    cali_2 = np.where((cdf_file['Status'][i-41:i+2] == 0))
                    if len(cali_1[0]) > 0:
                        for j in range (len(cali_1[0])):
                            cali_list.append(cdf_file['Status'][i-41:i+2][cali_1[0][j]][cali_1[1][j]])
                            cali_time.append(cali_1[0][j])
        
                    if len(cali_2[0]) > 0:
                        for j in range (len(cali_2[0])):
                            cali_list.append(cdf_file['Status'][i-41:i+2][cali_2[0][j]][cali_2[1][j]])
                            cali_time.append(cali_2[0][j])
        
                    cali_list = np.array(cali_list)
                    cali_time = np.array(cali_time)
                    
                    cali_list_f = []
        
                    for j in range(len(cali_time)):
                        cali_list_f.append(cali_list[np.argsort(cali_time)[j]])
                    cali_time_f = np.sort(cali_time)
                    cali_list_f = np.array(cali_list_f)
        
                    # print (cdf_file['Status'][i-41:i + 2])
                    time = i - 41
                    # print (time)
                    #+1 is due to move_average
                    start = epoch[time]
                    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
                    time = i + 1
                    end = epoch[time]
                    end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
                    time_obs = i - 20
                    obs = epoch[time_obs]
                    obs_time = dt.datetime(obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6])
                    Time_start = start.strftime('%H:%M:%S')
                    Time_end = end.strftime('%H:%M:%S')
                    # print (start)
                    print(Time_start+'-'+Time_end)
                    start = start.timestamp()
                    end = end.timestamp()
                    x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
                    x_lims = mdates.date2num(x_lims)
            
                    # Set some generic y-limits.
                    y_lims = [Frequency[0], Frequency[-1]]
        
                    # RR_sep = diff_r_last[:, i - 39:i]
                    # LL_sep = diff_l_last[:, i - 39:i]
                    RR_sep = diff_r_last[:, i - 41:i + 2]
                    LL_sep = diff_l_last[:, i - 41:i + 2]
    
    
        
                    if RR_sep.shape[1] >= 40:
                        if len(cali_time_f ) == 5:
                            hot_power_r_30 = np.median(RR_sep[285][cali_time_f[-1] - 10:cali_time_f[-1] + 1])
                            cold_power_r_30 = np.median(RR_sep[285][cali_time_f[0]:cali_time_f[0] + 10])
                            hot_power_l_30 = np.median(LL_sep[285][cali_time_f[-1] - 10:cali_time_f[-1] + 1])
                            cold_power_l_30 = np.median(LL_sep[285][cali_time_f[0]:cali_time_f[0] + 10])
                            hot_power_r_80 = np.median(RR_sep[0][cali_time_f[-1] - 10:cali_time_f[-1] + 1])
                            cold_power_r_80 = np.median(RR_sep[0][cali_time_f[0]:cali_time_f[0] + 10])
                            hot_power_l_80 = np.median(LL_sep[0][cali_time_f[-1] - 10:cali_time_f[-1] + 1])
                            cold_power_l_80 = np.median(LL_sep[0][cali_time_f[0]:cali_time_f[0] + 10])
        
                            delta_f = 30000
        
                            hot_power_all_r = (10 ** ((np.median(RR_sep[:, cali_time_f[-1] - 10:cali_time_f[-1] + 1], axis = 1))/10)) * delta_f / 50#50は抵抗値
                            cold_power_all_r = (10 ** ((np.median(RR_sep[:, cali_time_f[0]:cali_time_f[0] + 10], axis = 1))/10)) * delta_f / 50
                            hot_power_all_l = (10 ** ((np.median(LL_sep[:, cali_time_f[-1] - 10:cali_time_f[-1] + 1], axis = 1))/10)) * delta_f / 50
                            cold_power_all_l = (10 ** ((np.median(LL_sep[:, cali_time_f[0]:cali_time_f[0] + 10], axis = 1))/10)) * delta_f / 50
            
                            kb = 1.380649e-23
                            Thot = 290*(10 ** (42/10))
                            Tcold = 290*(10 ** ((42-30)/10))
                            # Thot = -72
                            # Tcold = -102
                            Thot_Tcold = Thot - Tcold
                            # (10 ** ((hot_power_all_r)/10)) + (10 ** ((cold_power_all_r)/10))
                            Phot_Pcold_r = hot_power_all_r - cold_power_all_r
                            Phot_Pcold_r = np.where(Phot_Pcold_r <= 0, np.nan, Phot_Pcold_r)
                            delta_f = 30000#bandwidth
                            gain_r = Phot_Pcold_r/(kb*delta_f*Thot_Tcold)
                            Y_r = hot_power_all_r/cold_power_all_r
                            Y_r = np.where(Y_r <= 1, np.nan, Y_r)
                            Trx_r = (Thot - (Y_r*Tcold))/(Y_r-1)
                            x = np.round(np.arange(Frequency[0], Frequency[-1]+resolution, resolution), decimals=3)
                            x = x[::-1]
                            Phot_Pcold_l = hot_power_all_l - cold_power_all_l
                            Phot_Pcold_l = np.where(Phot_Pcold_l <= 0, np.nan, Phot_Pcold_l)
                            gain_l = Phot_Pcold_l/(kb*delta_f*Thot_Tcold)
                            Y_l = hot_power_all_l/cold_power_all_l
                            Y_l = np.where(Y_l <= 1, np.nan, Y_l)
                            Trx_l = (Thot - (Y_l*Tcold))/(Y_l-1)
                            if (np.count_nonzero(np.isnan(gain_r))+np.count_nonzero(np.isnan(gain_l)))/(len(gain_r)*2)*100 < 10:
                                w.writerow({'obs_time': obs_time, 'Frequency': x, 'Right-gain':gain_r, 'Right-Trx': Trx_r, 'Right-hot_dB':np.median(RR_sep[:, cali_time_f[-1] - 10:cali_time_f[-1] + 1], axis = 1), 'Right-cold_dB': np.median(RR_sep[:, cali_time_f[0]:cali_time_f[0] + 10], axis = 1), 'Left-gain':gain_l, 'Left-Trx': Trx_l, 'Left-hot_dB':np.median(LL_sep[:, cali_time_f[-1] - 10:cali_time_f[-1] + 1], axis = 1), 'Left-cold_dB': np.median(LL_sep[:, cali_time_f[0]:cali_time_f[0] + 10], axis = 1)})

    
        
                                figure_=plt.figure(1,figsize=(25,8))
                                gs = gridspec.GridSpec(40, 24)
                    
                                axes_1 = figure_.add_subplot(gs[:5, 0:10])
                        #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                ax1 = axes_1.imshow(RR_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                          aspect='auto',cmap='jet',vmin= 18.75,vmax = 62.5)
                                axes_1.xaxis_date()
                                date_format = mdates.DateFormatter('%H:%M:%S')
                                axes_1.xaxis.set_major_formatter(date_format)
                                plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=15)
                                plt.xlabel('Time (UT)',fontsize=10)
                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                plt.colorbar(ax1,label='dB[V^2/Hz]')
                
                                figure_.autofmt_xdate()
                        
                                axes_2 = figure_.add_subplot(gs[11:16, 0:10])
                        #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                ax2 = axes_2.imshow(LL_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                          aspect='auto',cmap='jet',vmin= 18.75,vmax = 62.5)
                                axes_2.xaxis_date()
                                date_format = mdates.DateFormatter('%H:%M:%S')
                                axes_2.xaxis.set_major_formatter(date_format)
                                # plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=15)
                                plt.xlabel('Time (UT)',fontsize=10)
                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                plt.colorbar(ax2,label='dB[V^2/Hz]')
                                figure_.autofmt_xdate()
                        
                                # plt.show()
                                # plt.close()
                
            
                
                
                                # figure_=plt.figure(1,figsize=(18,5))
                                # gs = gridspec.GridSpec(20, 12)
                                axes_3 = figure_.add_subplot(gs[23:29, 0:8])
                                x_sep = np.arange((RR_sep).shape[1])
                                axes_3.plot(x_sep, RR_sep[285], color = 'r')
                                axes_3.plot(x_sep, LL_sep[285], color = 'b')
                                for j in range(len(cali_time_f)):
                                    # axes_3.axvline(cali_time_f[j], ls = "--", label = str(cali_list_f[j]) + ',' + str(cali_time_f[j])+ 'sec')
                                    axes_3.axvline(cali_time_f[j], ls = "--")
                
                                axes_3.axhline(hot_power_r_30, ls = "-.", label = "hot_power_r", color = 'r')
                                axes_3.axhline(hot_power_l_30, ls = "-.", label = "hot_power_l", color = 'b')
                                axes_3.axhline(cold_power_r_30, ls = "-.", label = "cold_power_r", color = 'r')
                                axes_3.axhline(cold_power_l_30, ls = "-.", label = "cold_power_l", color = 'b')
                
                                plt.title('Calibration at 30MHz')
                                plt.xlabel('Time (sec)',fontsize=10)
                                plt.ylabel('dB[V^2/Hz]',fontsize=10)
                                plt.legend(bbox_to_anchor=(1.3, 1.1), loc='upper right', borderaxespad=2) 
                        
                                axes_4 = figure_.add_subplot(gs[34:40, 0:8])
                                axes_4.plot(x_sep, RR_sep[0], color = 'r')
                                axes_4.plot(x_sep, LL_sep[0], color = 'b')
                                for j in range(len(cali_time_f)):
                                    # axes_4.axvline(cali_time_f[j], ls = "--", label = str(cali_list_f[j]) + ',' + str(cali_time_f[j]) + 'sec')
                                    axes_4.axvline(cali_time_f[j], ls = "--")
                                axes_4.axhline(hot_power_r_80, ls = "-.", label = "hot_power_r", color = 'r')
                                axes_4.axhline(hot_power_l_80, ls = "-.", label = "hot_power_l", color = 'b')
                                axes_4.axhline(cold_power_r_80, ls = "-.", label = "cold_power_r", color = 'r')
                                axes_4.axhline(cold_power_l_80, ls = "-.", label = "cold_power_l", color = 'b')
                                plt.title('Calibration at 80MHz')
                                plt.xlabel('Time (sec)',fontsize=10)
                                plt.ylabel('dB[V^2/Hz]',fontsize=10)
                                plt.legend(bbox_to_anchor=(1.3, 1.1), loc='upper right', borderaxespad=2) 
                                # plt.show()
                                # plt.close()
                
            
                                # figure_=plt.figure(1,figsize=(18,5))
                                # gs = gridspec.GridSpec(20, 12)
                                axes_3 = figure_.add_subplot(gs[:5, 11:17])
            
                
                                axes_3.plot(x, hot_power_all_r, color = 'r', label = 'hot')
                                axes_3.plot(x, cold_power_all_r, color = 'b', label = 'cold')
                                axes_3.set_yscale('log')
                
                
                                plt.title('Right_pol_Calibration')
                                plt.xlabel('Frequency [MHz]',fontsize=10)
                                plt.ylabel('dB[V^2/Hz]',fontsize=10)
                                plt.legend(bbox_to_anchor=(1.2, 1.1), loc='upper right', borderaxespad=2) 
                        
                                axes_4 = figure_.add_subplot(gs[10:15, 11:17])
                                axes_4.plot(x, hot_power_all_l, color = 'r', label = 'hot')
                                axes_4.plot(x, cold_power_all_l, color = 'b', label = 'cold')
                                axes_4.set_yscale('log')
                                plt.title('Left_pol_Calibration')
                                plt.xlabel('Frequency [MHz]',fontsize=10)
                                plt.ylabel('dB[V^2/Hz]',fontsize=10)
                                plt.legend(bbox_to_anchor=(1.2, 1.1), loc='upper right', borderaxespad=2) 
                                # plt.show()
                                # plt.close()
                                
                
            
                                # print
                                # print (gain)
                
                                # figure_=plt.figure(1,figsize=(18,5))
                                # gs = gridspec.GridSpec(20, 12)
                                axes_3 = figure_.add_subplot(gs[20:27, 11:17])
                                # x = np.round(np.arange(Frequency[0], Frequency[-1]+resolution, resolution), decimals=3)
                                # x = x[::-1]
                                index_40dB = np.where(x == getNearestValue(x,40))[0][0]
                
                                axes_3.plot(x, gain_r, color = 'r', label = 'gain')
                                print (gain_r[index_40dB])
                                axes_3.set_yscale('log')
                
                
                                plt.title('Gain(Right)')
                                plt.xlabel('Frequency [MHz]',fontsize=10)
                                plt.ylabel('Gain[dB]',fontsize=10)
                                plt.legend(bbox_to_anchor=(1.2, 1.1), loc='upper right', borderaxespad=2) 
                
                                
            
                
                                # figure_=plt.figure(1,figsize=(18,5))
                                # gs = gridspec.GridSpec(20, 12)
                                axes_3 = figure_.add_subplot(gs[33:40, 11:17])
                
                                
                                axes_3.plot(x, Trx_r, color = 'r', label = 'Trx')
                                axes_3.set_yscale('log')
                                
                                plt.title('Trx(Right)')
                                plt.xlabel('Frequency [MHz]',fontsize=10)
                                plt.ylabel('Trx[K]',fontsize=10)
                                plt.legend(bbox_to_anchor=(1.2, 1.1), loc='upper right', borderaxespad=2) 
                                if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/gain_plot/'+yyyy+'/'+mm):
                                    os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/gain_plot/'+yyyy+'/'+mm)
                                plt.savefig('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/gain_plot/'+yyyy+'/'+mm+'/'+obs_time.strftime(format='%Y%m%d%H%M'))
                                plt.show()
                                plt.close()
                          # if i == 2597:
                          #   sys.exit()
        except:
            pass
        DATE+=pd.to_timedelta(1,unit='day')
    
    
    
    #Trxは100前後Nancayでは1000ぐらい？
    #Gainは100dB〜数十デシベルぐらいが普通