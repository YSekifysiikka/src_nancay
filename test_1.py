#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:36:49 2020

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
from pynverse import inversefunc

sigma_range = 1
sigma_start = 2
sigma_mul = 1
#when you only check one threshold
sigma_value = 2
after_plot = str('drift_check')
after_after_check = str('std_check_1')
after_after_plot = str('test')
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
Parent_directory = '/Volumes/HDPH-UT/lab'

def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res
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
#####################
#plot_date_select
Obs_date = []
path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' +str(sigma_value)+ '/*/*/*'
#path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/*/*'
File = glob.glob(path, recursive=True)
#print(File)
File1=len(File)
print (File1)
for cstr in File:
    a=cstr.split('/')
#            line=a[9]+'\n'
    line = a[12]
#    line = a[10]
#    print(line)
    file_name_separate = line.split('_')
    if not file_name_separate[4] == 'sigma':
        Obs_date.append(file_name_separate[0])
Obs_date_final = np.unique(Obs_date)
#####################
#plot_date_time_select
for x in range(len(Obs_date_final)):
    Obs_time_start = []
    Obs_burst_start = []
    Obs_burst_end = []
    year = Obs_date_final[x][:4]
    month = Obs_date_final[x][4:6]
    day = Obs_date_final[x][6:8]
    path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' +str(sigma_value)+ '/' + year + '/'+ month +'/'+ Obs_date_final[x] +'*'
#    path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' + year + '/'+ Obs_date_final[x] +'*'
    File = glob.glob(path, recursive=True)
    File1=len(File)
    print (File1)
    for cstr in File:
        a = cstr.split('/')
        line = a[12]
#        line = a[10]
    #    print(line)
        file_name_separate = line.split('_')
        if not file_name_separate[4] == 'sigma':
            Obs_time_start.append(int(file_name_separate[3]))
            Obs_burst_start.append(int(file_name_separate[5]))
            Obs_burst_end.append(int(file_name_separate[6]))
#####################
#data_import_and_editation
    path = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/*' + year + month + day + '*_' + year + month + day + '*' + '.cdf'
    File = glob.glob(path, recursive=True)
    print (File)
    for cstr in File:
        a=cstr.split('/')
        line=a[9]
        print (line)
        file_name = line
        file_name_separate =file_name.split('_')
        Date_start = file_name_separate[5]
        date_OBs=str(Date_start)
        year = date_OBs[0:4]
        month = date_OBs[4:6]
        day = date_OBs[6:8]
        start_h = date_OBs[8:10]
        start_m = date_OBs[10:12]
        Date_stop = file_name_separate[6]
        end_h = Date_stop[8:10]
        end_m = Date_stop[10:12]
        file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
        cdf_file = cdflib.CDF(file)
        epoch = cdf_file['Epoch'] 

        file = Parent_directory + '/solar_burst/Nancay/csv/min_med/' + year + '/' + month + '/' + file_name[:-4] + '.csv'
        diff_db_min_med_0 = pd.read_csv(file, header=None)
        diff_db_min_med = diff_db_min_med_0.values
        file = Parent_directory + '/solar_burst/Nancay/csv/min/' + year + '/' + month + '/' + file_name[:-4] + '.csv'
        min_db_0 = pd.read_csv(file, header=None)
        min_db_1 = min_db_0.values
        min_db = min_db_1[0]
        file = Parent_directory + '/solar_burst/Nancay/csv/db/' + year + '/' + month + '/' + file_name[:-4] + '.csv'
        diff_db_0 = pd.read_csv(file, header=None)
        diff_db = diff_db_0.values
#        LL = cdf_file['LL'] 
#        RR = cdf_file['RR'] 
#    
#        data_r_0 = RR
#        data_l_0 = LL
#        diff_r_last =(data_r_0).T
#        diff_r_last = np.flipud(diff_r_last)
#        diff_l_last =(data_l_0).T
#        diff_l_last = np.flipud(diff_l_last)
#    
#    
#        diff_db_pre = []
#        diff_power_check = []
#        for i in range(diff_l_last.shape[0]):
#            l = diff_l_last[i]
#            r = diff_r_last[i]
#            for k in range(diff_l_last.shape[1]):
#                diff_power = (((10 ** ((r[k])/10)) + (10 ** ((l[k])/10)))/2)
#                diff_power_check.append(diff_power)
#                diff_db_pre.append(math.log10(diff_power) * 10)
#        diff_P = np.array(diff_power_check).reshape([diff_l_last.shape[0], diff_l_last.shape[1]])
#        diff_db = np.array(diff_db_pre).reshape([diff_l_last.shape[0], diff_l_last.shape[1]])
#        #diff_db = diff_db_pre.reshape([diff_l_last_1.shape[0], diff_l_last_1.shape[1]])
#        ####################
#        y_power = []
#        y_db = []
#        num = int(move_ave)
#        b = np.ones(num)/num
#        for i in range (diff_db.shape[0]):
#            y_power.append(np.convolve(diff_P[i], b, mode='valid'))
#        for i in range (diff_db.shape[0]):
#            for k in range(len(y_power[i])):
#                y_db.append(math.log10(y_power[i][k]) * 10)
#        diff_move_db = np.array(y_db).reshape([diff_l_last.shape[0], diff_l_last.shape[1] -2 ])
#        ####################
#        min_db = []
#        min_power = np.amin(y_power, axis=1)
#        for i in range(min_power.shape[0]):
#            min_db.append(math.log10(min_power[i]) * 10)
#        diff_db_min_med = (diff_move_db.T - min_db).T
        ####################
        for z in range(len(Obs_time_start)):
            if int(Obs_time_start[z]) % time_band == 0:
                t = round(int(Obs_time_start[z])/time_band)
            else:
                t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
            time = round(time_band*t)
            print (time)
            #+1 is due to move_average
            start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
            time = round(time_band*(t+1) + time_co)
            end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
            Time_start = start.strftime('%H:%M:%S')
            Time_end = end.strftime('%H:%M:%S')
            print (start)
            print(Time_start+'-'+Time_end)
            start = start.timestamp()
            end = end.timestamp()
    
            x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
            x_lims = mdates.date2num(x_lims)
    
            # Set some generic y-limits.
            y_lims = [10, 80]
    
    
            diff_db_plot_l = diff_db_min_med[:, time - time_band - time_co:time]
            diff_db_l = diff_db[:, time - time_band - time_co:time]
    
            y_over_analysis_time_l = []
            y_over_analysis_data_l = []
    
            TYPE3 = []
            TYPE3_group = [] 
            quartile_db_l = []
            mean_l_list = []
            stdev_sub = []
            quartile_power = []
    
    
    
            for i in range(0, check_frequency_range, 1):
            #    for i in range(0, 357, 1):
                quartile_db_25_l = np.percentile(diff_db_plot_l[i], 25)
                quartile_db_each_l = []
                for k in range(diff_db_plot_l[i].shape[0]):
                    if diff_db_plot_l[i][k] <= quartile_db_25_l:
                        diff_power_quartile_l = (10 ** ((diff_db_plot_l[i][k])/10))
                        quartile_db_each_l.append(diff_power_quartile_l)
            #                quartile_db_each.append(math.log10(diff_power) * 10)
                m_l = np.mean(quartile_db_each_l)
                stdev_l = np.std(quartile_db_each_l)
                sigma_l = m_l + sigma_value*stdev_l
                sigma_db_l = (math.log10(sigma_l) * 10)
                quartile_power.append(sigma_l)
                quartile_db_l.append(sigma_db_l)
                mean_l_list.append(math.log10(m_l) * 10)
            for i in range(len(quartile_db_l)):
                stdev_sub.append(quartile_db_l[i] - mean_l_list[i])
    
    
            #        y3_db = diff_db_plot[0:400, j+2]
            x = np.flip(np.linspace(x_start_range, x_end_range, x_whole_range))
            x2 = np.flip(np.linspace(10, x_end_range, 400))
            x1 = x
            for j in range(0, time_band + time_co, 1):
                y1_db_l = diff_db_plot_l[0:check_frequency_range, j]
    
                y_le = []
                for i in range(len(y1_db_l)):
                    diff_power_last_l = (10 ** ((y1_db_l[i])/10)) - quartile_power[i]
                    if diff_power_last_l > 0:
                        y_le.append(math.log10(diff_power_last_l) * 10)
                    else:
                        y_le.append(0)
                y_over_l = []
                for i in range(len(y_le)):
                    if y_le[i] > 0:
                        y_over_l.append(i)
                    else:
                        pass
                y_over_final_l = []
                if len(y_over_l) > 0:
                    y_over_group_l = groupSequence(y_over_l)
                    for i in range(len(y_over_group_l)):
                        if len(y_over_group_l[i]) >= threshold_frequency:
                            y_over_final_l.append(y_over_group_l[i])
                            y_over_analysis_time_l.append(round(time_band*t + j))
                            y_over_analysis_data_l.append(y_over_group_l[i])
    
                if len(y_over_final_l) > 0:
                    TYPE3.append(round(time_band*t + j))
    
    
            TYPE3_final = np.unique(TYPE3)
            if len(TYPE3_final) > 0:
                TYPE3_group.append(groupSequence(TYPE3_final))
    
    
            if len(TYPE3_group)>0:
                for m in range (len(TYPE3_group[0])):
                    if len(TYPE3_group[0][m]) >= duration:
                        if TYPE3_group[0][m][0] >= 0 and TYPE3_group[0][m][-1] <= diff_db_min_med.shape[1]:
                            arr = [[-10 for i in range(400)] for j in range(400)]
    #                        for i in range(400):
    #                            arr[i][time_co] = 20
    #                            arr[i][400-time_co] = 20
    #                            arr[286][i] = 20
                            for i in range(len(TYPE3_group[0][m])):
                                check_start_time_l = ([y for y, x in enumerate(y_over_analysis_time_l) if x == TYPE3_group[0][m][i]])
                                for q in range(len(check_start_time_l)):
                                    for p in range(len(y_over_analysis_data_l[check_start_time_l[q]])):
                                        arr[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = diff_db_plot_l[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)]
    #frequency_sequence
                            frequency_sequence = []
                            frequency_separate = []
                            for i in range(check_frequency_range):
                                for j in range(400):
                                    if arr[i][j] > -10:
                                        frequency_sequence.append(i)
                            frequency_sequence_final = np.unique(frequency_sequence)
                            if len(frequency_sequence_final) > 0:
                                frequency_separate.append(groupSequence(frequency_sequence_final))
                                for i in range(len(frequency_separate[0])):
                                    arr1 = [[-10 for l in range(400)] for n in range(400)]
                                    for j in range(len(frequency_separate[0][i])):
                                        for k in range(400):
                                            if arr[frequency_separate[0][i][j]][k] > -10:
                                                arr1[frequency_separate[0][i][j]][k] = arr[frequency_separate[0][i][j]][k]
    #check_time_sequence
                                    time_sequence = []
                                    time_sequence_final_group = []
                                    for j in range(400):
                                        for k in range(check_frequency_range):
                                            if not arr1[k][j] == -10:
                                                time_sequence.append(j)
                                    time_sequence_final = np.unique(time_sequence)
                                    if len(time_sequence) > 0:
                                        time_sequence_final_group.append(groupSequence(time_sequence_final))
                                        for j in range(len(time_sequence_final_group[0])):
                                            if len(time_sequence_final_group[0][j]) >= duration:
                                                arr2 = [[-10 for l in range(400)] for n in range(400)]
                                                for k in range(len(time_sequence_final_group[0][j])):
                                                    for l in range(check_frequency_range):
                                                        if arr1[l][time_sequence_final_group[0][j][k]] > -10:
                                                            arr2[l][time_sequence_final_group[0][j][k]] = arr1[l][time_sequence_final_group[0][j][k]]
    #check_drift
                                                drift_freq = []
                                                drift_time_end = []
                                                drift_time_start = []
                                                for k in range(check_frequency_range):
                                                    drift_time = []
                                                    for l in range(400):
                                                        if arr2[k][l] > -10:
                                                            drift_freq.append(k)
                                                            drift_time.append(l)
                                                    if len(drift_time) > 0:
                                                        drift_time_end.append(drift_time[-1])
                                                        drift_time_start.append(drift_time[0])
                                                if len(drift_time_end) >= threshold_frequency_final:
    #                                                if drift_time_end[0] < drift_time_end[-1]:
    #                                                if not max(drift_freq) == 285:
    #                                                    if not min(drift_freq) ==0:
    #make_arr3_for_dB_setting
                                                    freq_each_max = np.max(arr2, axis=1)
                                                    arr3 = []
                                                    for l in range(len(freq_each_max)):
                                                        if freq_each_max[l] > -10:
                                                            arr3.append(freq_each_max[l])
                                                    arr4 = []
                                                    for l in range(len(arr2)):
                                                        for k in range(len(arr2[l])):
                                                            if arr2[l][k] > -10:
                                                                arr4.append(arr2[l][k])
                                                            
                                                    freq_list = []
                                                    time_list = []
                                                    freq_list_use = []
                                                    time_list_use = []
                                                    freq_list_nonuse = []
                                                    time_list_nonuse = []
                                                    freq_list_plot = []
                                                    time_list_plot = []
                                                    time_list_error = []
                                                    drift_ave_time = []
                                                    drift_1_time = []
                                                    drift_2_time = []
                                                    drift_ave_freq = []
                                                    drift_1_freq = []
                                                    drift_2_freq = []
                                                    for k in range(check_frequency_range):
                                                        if max(arr2[k]) > -10:
                                                            if (len([l for l in arr2[k] if l == max(arr2[k])])) == 1:
                                                                freq_list.append(cdf_file['Frequency'][399 - k])
                                                                time_list.append(np.argmax(arr2[k]))
    
                                                    freq_list_plot.append((freq_list[int(freq_check_range/2)] + freq_list[int((freq_check_range/2) + 1)])/2)
                                                    freq_list_plot.append(freq_list[round(len(freq_list)/2)])
                                                    freq_list_plot.append((freq_list[int(-(freq_check_range/2)-1)] + freq_list[int(-(freq_check_range/2) -2)])/2)
                                                    time_list_plot.append(np.mean(time_list[0: freq_check_range]))
                                                    time_list_plot.append(np.mean(time_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))]))
                                                    time_list_plot.append(np.mean(time_list[len(time_list) - 20: len(time_list)]))
                                                    time_list_error.append(np.std(time_list[0:freq_check_range]))
                                                    time_list_error.append(np.std(time_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))]))
                                                    time_list_error.append(np.std(time_list[len(time_list) - 20: len(time_list)]))
                                                    drift_ave_freq.append((freq_list[int(freq_check_range/2)] + freq_list[int((freq_check_range/2) + 1)])/2)
                                                    drift_ave_freq.append((freq_list[int(-(freq_check_range/2)-1)] + freq_list[int(-(freq_check_range/2) -2)])/2)
                                                    drift_ave_time.append(np.mean(time_list[0: freq_check_range]))
                                                    drift_ave_time.append(np.mean(time_list[len(time_list) - 20: len(time_list)]))
                                                    drift_1_freq.append((freq_list[int(freq_check_range/2)] + freq_list[int((freq_check_range/2) + 1)])/2)
                                                    drift_1_freq.append(freq_list[round(len(freq_list)/2)])
                                                    drift_1_time.append(np.mean(time_list[0: freq_check_range]))
                                                    drift_1_time.append(np.mean(time_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))]))
                                                    drift_2_freq.append(freq_list[round(len(freq_list)/2)])
                                                    drift_2_freq.append((freq_list[int(-(freq_check_range/2)-1)] + freq_list[int(-(freq_check_range/2) -2)])/2)
                                                    drift_2_time.append(np.mean(time_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))]))
                                                    drift_2_time.append(np.mean(time_list[len(time_list) - 20: len(time_list)]))
                                                    p_1 = np.polyfit(freq_list, time_list, 1) #determine coefficients
                                                    p_2 = np.polyfit(freq_list, time_list, 3) #determine coefficients
                                                    drift_std = str(np.std((np.polyval(p_2, freq_list) - time_list) ** 2))
                                                    for k in range(len(time_list)):
                                                        if ((np.polyval(p_2, freq_list) - time_list) ** 2)[k] <= np.mean((np.polyval(p_2, freq_list) - time_list) ** 2) + 2 * np.std((np.polyval(p_2, freq_list) - time_list) ** 2):
                                                            freq_list_use.append(freq_list[k])
                                                            time_list_use.append(time_list[k])
                                                        else:
                                                            freq_list_nonuse.append(freq_list[k])
                                                            time_list_nonuse.append(time_list[k])
                                                    p_3 = np.polyfit(freq_list_use, time_list_use, 3)
                                                    p_4 = np.polyfit(freq_list_use, time_list_use, 1)
                                                    drift_std_new = str(np.std((np.polyval(p_3, freq_list_use) - time_list_use) ** 2))
                                                    print ('old:' + drift_std + 'New:' + drift_std_new)
                                                    if 1/p_4[0] < 0:
                                                        #drift_allen_model_*1_80_69.5
                                                        if 1/p_4[0] > -107.22096882538592:
                                                            if np.count_nonzero(cdf_file['Status'][time - time_band - time_co + 1:time + 1] == 0) == 0:
                                                                if np.count_nonzero(cdf_file['Status'][time - time_band - time_co + 1:time + 1] == 17) == 0:
    #                                                                p_2 = np.polyfit(time_list_plot, freq_list_plot, 2) #determine coefficients
                #                                                            if p_2[0] > 0:
                                                                    print (freq_list_plot)
                                                                    print (time_list_plot)
                                                                    drift_ave = np.polyfit(drift_ave_time, drift_ave_freq, 1)
                                                                    if not drift_1_time[0] == drift_1_time[1]:
                                                                        drift_1 = np.polyfit(drift_1_time, drift_1_freq, 1)
                                                                    if not drift_2_time[0] == drift_2_time[1]:
                                                                        drift_2 = np.polyfit(drift_2_time, drift_2_freq, 1)
                                                                    event_start = str(min(drift_time_start))
                                                                    event_end = str(max(drift_time_end))
    ######################################################################################
                                                                    if int(event_start) == int(Obs_burst_start[z]):
                                                                        if int(event_end) == int(Obs_burst_end[z]):
                                                                            freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                                            freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                                            event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                                            freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                                            y_lims = [10, 80]
                                                                            plt.close(1)
                        #                                                            figure_=plt.figure(1,figsize=(16,10))
                                                                            figure_=plt.figure(1,figsize=(10,16))
                                                                            gs = gridspec.GridSpec(8, 6)
                                                                            axes_2 = figure_.add_subplot(gs[:4, :])
                                                                #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                            ax2 = axes_2.imshow(arr2, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                      aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                            axes_2.xaxis_date()
                                                                            date_format = mdates.DateFormatter('%H:%M:%S')
                                                                            axes_2.xaxis.set_major_formatter(date_format)
                                                                            plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
                                                                            plt.xlabel('Time',fontsize=20)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=20)
                                                                            plt.colorbar(ax2,label='from Background [dB]')
                        
                                                                            plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                            figure_.autofmt_xdate()
                        #                                                            if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day):
                        #                                                                os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day)
                        #                                                            filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end
                        #                                                            filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'/'+filename1+'.png'
                        #                                                            plt.savefig(filename)
                        #                                                            plt.show()
                        #                                                            plt.close()
                                                                            ax1 = figure_.add_subplot(gs[5:, :2])
                                                                            xx_1 = np.linspace(min(time_list) - 5, max(time_list) + 5, 100)
                                                                            yy_1 = np.polyval(drift_ave, xx_1)
                                                                            ax1.errorbar(time_list_plot, freq_list_plot, xerr=time_list_error, ecolor  = 'r', markersize=3) #Data plot
                                                                            ax1.plot(xx_1, yy_1, 'r', label = 'ave_freq_drift') #Interpolation
                                                                            ax1.plot(time_list[0:freq_check_range], freq_list[0:freq_check_range], "bo", label = 'Used_Data') #Interpolation
                                                                            ax1.plot(time_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))], freq_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))], "bo") #Interpolation
                                                                            ax1.plot(time_list[len(time_list) - 20: len(time_list)], freq_list[len(time_list) - 20: len(time_list)], "bo") #Interpolation
                                                                            ax1.plot(time_list[freq_check_range:int(round((len(freq_list))/2)-(freq_check_range/2))], freq_list[freq_check_range:int(round((len(freq_list))/2)-(freq_check_range/2))], "ko", label = 'Non_used_Data') #Interpolation
                                                                            ax1.plot(time_list[int(round((len(freq_list))/2)+(freq_check_range/2)):len(time_list) - 20], freq_list[int(round((len(freq_list))/2)+(freq_check_range/2)):len(time_list) - 20], "ko") #Interpolation
                                                                            
            #                                                                plt.title('freq_drift_rate')
                                                                            plt.xlabel('Time[sec]',fontsize=20)
                                                                            plt.ylabel('Frequency[MHz]',fontsize=20)
                                                                            plt.legend(fontsize=12)
                                                                            plt.tick_params(labelsize=18)
                                                                            plt.xlim(min(time_list) - 5, max(time_list) + 5)
                                                                            plt.ylim(10, 80)
                                                                            print (drift_ave)
                        #                                                            plt.show()
                        #                                                            plt.close()
            #            
                                                                            ax2 = figure_.add_subplot(gs[5:, 3:5])
                                                                            xx_2 = np.linspace(min(freq_list) - 5, max(freq_list) + 5, 100)
                                                                            yy_1 = np.polyval(p_4, xx_2)
                                                                            yy_2 = np.polyval(p_2, xx_2)
                                                                            yy_3 = np.polyval(p_3, xx_2)
            #                                                                ax2.errorbar(time_list_plot, freq_list_plot, xerr=time_list_error, label = 'drift', markersize=3) #Data plot
                                                                            ax2.plot(time_list_use, freq_list_use, "bo", label = 'Used_Data') #Interpolation
                                                                            ax2.plot(time_list_nonuse, freq_list_nonuse, "ko", label = 'Non_used_Data') #Interpolation
                                                                            ax2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
                                                                            ax2.plot(yy_2, xx_2, 'b')
                                                                            ax2.plot(yy_3, xx_2, 'r', label = 'freq_drift_new(third_order)')
######################################################################################
#                                                                            h5 = np.arange(0, 1183200/(time_rate3 * 300000), 69600/(time_rate3 * 300000))
#                                                                            x5 = 3 * 9 * 10 * np.sqrt((2.99*(1+(h5/696000)*(time_rate3 * 300000))**(-16)+1.55*(1+(h5/696000)*(time_rate3 * 300000))**(-6)+0.036*(1+(h5/696000)*(time_rate3 * 300000))**(-1.5)))
#                                                                            h5 = h5[7:]
#                                                                            x5 = x5[7:]
#                                                                            h5 = h5 + 350
#                                                                            p_5 = np.polyfit(x5, h5, 2)
#                                                                            xx_5 = np.linspace(min(x5) - 5, max(x5) + 5, 100)
#                                                                            yy_5 = np.polyval(p_5, xx_5)
#                                                                            ax2.plot(yy_5, xx_5, 'k', label = 'freq_drift(linear)')
#                                                                            ax2.plot(h5, x5, '.', label = 'allen_model')
######################################################################################
            #                                                                plt.title('Used_data')
                                                                            plt.xlabel('Time[sec]',fontsize=20)
                        #                                                            plt.ylabel('Frequency[MHz]')
                                                                            plt.legend(fontsize=12)
                                                                            plt.tick_params(labelsize=18)
                                                                            plt.xlim(min(time_list) - 5, max(time_list) + 5)
                                                                            plt.ylim(10, 80)
            #                                                                print (p_2)
                                                                            if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                                os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                            filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'_'+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'_'+ str(1/p_4[0]) +'_'+ drift_std_new
                                                                            filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+filename1+'.png'
                                                                            plt.savefig(filename)
                                                                            plt.show()
                                                                            plt.close()
                        #                                                    for l in range(285):
                        #                                                        x = np.linspace(0, 399, num=400)
                        #                                                        plt.plot(x,arr2[l],alpha=0.7,linewidth=1,c="red",label="original wave")
                        #                                                        plt.show()
                        #                                                        plt.close()
                                                                            
                                                                            ###########################################
                                                                                
                                                                            
                        
                                                                            figure_=plt.figure(1,figsize=(18,5))
                                                                            gs = gridspec.GridSpec(20, 12)
                                                                            axes_1 = figure_.add_subplot(gs[:8, 6:11])
                                                                    #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                            ax1 = axes_1.imshow(diff_db_plot_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                      aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[228] + 10)
                                                                            axes_1.xaxis_date()
                                                                            date_format = mdates.DateFormatter('%H:%M:%S')
                                                                            axes_1.xaxis.set_major_formatter(date_format)
                                                                            plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=15)
                                                                            plt.xlabel('Time (UT)',fontsize=10)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                            plt.colorbar(ax1,label='from Background [dB]')
                                                                            figure_.autofmt_xdate()
                                                                    
                                                                    
                                                                    
                                                                    
                                                                    
                                                                            axes_1 = figure_.add_subplot(gs[:8, :5])
                                                                    #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                            ax1 = axes_1.imshow(diff_db_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                      aspect='auto',cmap='jet',vmin= -5 + min_db[228],vmax = quartile_db_l[228] + min_db[228] + 5)
                                                                            axes_1.xaxis_date()
                                                                            date_format = mdates.DateFormatter('%H:%M:%S')
                                                                            axes_1.xaxis.set_major_formatter(date_format)
                                                                            plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=15)
                                                                            plt.xlabel('Time (UT)',fontsize=10)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                            plt.colorbar(ax1,label='from Background [dB]')
                                                                            figure_.autofmt_xdate()
                                                                    
                                                                            axes_5 = figure_.add_subplot(gs[13:, 0:1])
                                                                            plt.title('min',fontsize=10)
                                                                            axes_5.plot(min_db, x2)
                                                                            axes_5.set_xlim(0, 80)
                                                                            axes_5.set_ylim(10, 80)
                                                                            plt.xlabel('Decibel [dB]',fontsize=8)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=8)
                                                                    
                                                                            axes_4 = figure_.add_subplot(gs[13:, 2:3])
                                                                            plt.title('mean',fontsize=10)
                                                                            axes_4.plot(mean_l_list, x1)
                                                                            axes_4.set_xlim(0, 50)
                                                                            axes_4.set_ylim(30, 80)
                                                                            plt.xlabel('Decibel [dB]',fontsize=8)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=8)
                                                                    
                                                                    
                                                                            axes_1 = figure_.add_subplot(gs[13:, 6:7])
                                                                            plt.title('m_'+str(sigma_value)+'sigma',fontsize=10)
                                                                            axes_1.plot(quartile_db_l, x1)
                                                                            axes_1.set_xlim(0, 50)
                                                                            axes_1.set_ylim(30, 80)
                                                                            plt.xlabel('Decibel [dB]',fontsize=8)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=8)
                                                                    
                                                                    
                                                                            axes_3 = figure_.add_subplot(gs[13:, 4:5])
                                                                            plt.title('stdev_'+str(sigma_value)+'sigma',fontsize=10)
                                                                            axes_3.plot(stdev_sub, x1)
                                                                            axes_3.set_xlim(0, 10)
                                                                            axes_3.set_ylim(30, 80)
                                                                            plt.xlabel('Decibel [dB]',fontsize=8)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=8)
                                                                    
                                                                            if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                                os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                            filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'__sigma_l_r.png'
                                                                            plt.savefig(filename)
                                                                            plt.show()
                                                                            plt.close()


                                                                            distance = []
                                                                            time_x = []
                                                                            fitting = []
                                                                            time_rate_result = []
                                                                            slide_result = []
                                                                            fitting_new = []
                                                                            time_rate_result_new = []
                                                                            slide_result_new = []
                                                                            
                                                                            freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                                            freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                                            event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                                            freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                                            y_lims = [10, 80]
                                                                            plt.close(1)
                                                                            #                   
                                                                            figure_=plt.figure(1,figsize=(20,10))
                                                                            gs = gridspec.GridSpec(20, 10)
                                                                            axes_2 = figure_.add_subplot(gs[:, :])
                                                                            #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                            ax2 = axes_2.imshow(arr2, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                      aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                            axes_2.xaxis_date()
                                                                            date_format = mdates.DateFormatter('%H:%M:%S')
                                                                            axes_2.xaxis.set_major_formatter(date_format)
                                                                            plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
                                                                            plt.xlabel('Time',fontsize=20)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=20)
                                                                            plt.colorbar(ax2,label='from Background [dB]')
                                                                            
                                                                            plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                            figure_.autofmt_xdate()
                                                                            plt.show()
                                                                            plt.close()
                                                                            #                                                            if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day):
                                                                            #                                                                os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day)
                                                                            #                                                            filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end
                                                                            #                                                            filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'/'+filename1+'.png'
                                                                            #                                                            plt.savefig(filename)
                                                                            #                                                            plt.show()
                                                                            #        
                                                                            figure_=plt.figure(1,figsize=(20,10))
                                                                            gs = gridspec.GridSpec(20, 10)
                                                                            ax1 = figure_.add_subplot(gs[:, :4])
                                                                            xx_1 = np.linspace(min(time_list) - 5, max(time_list) + 5, 100)
                                                                            yy_1 = np.polyval(drift_ave, xx_1)
                                                                            ax1.errorbar(time_list_plot, freq_list_plot, xerr=time_list_error, ecolor  = 'r', markersize=3) #Data plot
                                                                            ax1.plot(xx_1, yy_1, 'r', label = 'ave_freq_drift') #Interpolation
                                                                            ax1.plot(time_list[0:freq_check_range], freq_list[0:freq_check_range], "bo", label = 'Used_Data') #Interpolation
                                                                            ax1.plot(time_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))], freq_list[int(round((len(freq_list))/2)-(freq_check_range/2)):int(round((len(freq_list))/2)+(freq_check_range/2))], "bo") #Interpolation
                                                                            ax1.plot(time_list[len(time_list) - 20: len(time_list)], freq_list[len(time_list) - 20: len(time_list)], "bo") #Interpolation
                                                                            ax1.plot(time_list[freq_check_range:int(round((len(freq_list))/2)-(freq_check_range/2))], freq_list[freq_check_range:int(round((len(freq_list))/2)-(freq_check_range/2))], "ko", label = 'Non_used_Data') #Interpolation
                                                                            ax1.plot(time_list[int(round((len(freq_list))/2)+(freq_check_range/2)):len(time_list) - 20], freq_list[int(round((len(freq_list))/2)+(freq_check_range/2)):len(time_list) - 20], "ko") #Interpolation
                                                                            
                                                                            #                                                                plt.title('freq_drift_rate')
                                                                            plt.xlabel('Time[sec]',fontsize=20)
                                                                            plt.ylabel('Frequency[MHz]',fontsize=20)
                                                                            plt.legend(fontsize=12)
                                                                            plt.tick_params(labelsize=18)
                                                                            plt.xlim(min(time_list) - 5, max(time_list) + 5)
                                                                            plt.ylim(10, 80)
                                                                            print (drift_ave)
                                                                            #                                                            plt.show()
                                                                            #                                                            plt.close()
                                                                            #            
                                                                            ax2 = figure_.add_subplot(gs[:, 5:])
                                                                            xx_2 = np.linspace(min(freq_list) - 5, max(freq_list) + 5, 100)
                                                                            yy_1 = np.polyval(p_4, xx_2)
                                                                            yy_2 = np.polyval(p_2, xx_2)
                                                                            yy_3 = np.polyval(p_3, xx_2)
                                                                            #                                                                ax2.errorbar(time_list_plot, freq_list_plot, xerr=time_list_error, label = 'drift', markersize=3) #Data plot
                                                                            ax2.plot(time_list, freq_list, "bo", label = 'Data') #Interpolation
                                                                            #ax2.plot(time_list_nonuse, freq_list_nonuse, "ko", label = 'Non_used_Data') #Interpolation
                                                                            #ax2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
                                                                            #ax2.plot(yy_2, xx_2, 'b')
                                                                            #ax2.plot(yy_3, xx_2, 'r', label = 'freq_drift_new(third_order)')
                                                                            ######################################################################################
                                                                            fitting.append(100)
                                                                            time_rate_result.append(100)
                                                                            slide_result.append(100)
                                                                            
                                                                            for i in range(10, 100, 10):
                                                                                time_rate3 = i/100
                                                                                for j in range((min(time_list) - 10) * 10, max(time_list) * 10, 10):
                                                                                    slide = j/10
                                                                                    time_x = []
                                                                                    for k in range(len(freq_list)):
                                                                                        i = freq_list[k]
                                                                                        cube_3 = (lambda h5: 3 * 9 * 10 * np.sqrt(2.99 * ((1+(h5/696000)*(time_rate3 * 300000))**(-16)) + 1.55 * ((1+(h5/696000)*(time_rate3 * 300000))**(-6)) + 0.036 * ((1+(h5/696000)*(time_rate3 * 300000))**(-1.5))))
                                                                                        invcube_3 = inversefunc(cube_3, y_values=i)
                                                                                        time_x.append(invcube_3)
                                                                                    if min(fitting) > sum((((time_x + np.array(slide)) - time_list) ** 2))/len(freq_list):
                                                                                        fitting.append(sum((((time_x + np.array(slide)) - time_list) ** 2))/len(freq_list))
                                                                                        time_rate_result.append(time_rate3)
                                                                                        slide_result.append(slide)
                                                                            
                                                                            fitting_new.append(100)
                                                                            time_rate_result_new.append(100)
                                                                            slide_result_new.append(100)
                                                                            time_new = int(time_rate_result[-1]*100)
                                                                            slide_new = int(slide_result[-1])
                                                                            for i in range(time_new -10, time_new + 10, 1):
                                                                                time_rate3 = i/100
                                                                                for j in range(slide_new * 100 - 100, slide_new * 100 + 100, 1):
                                                                                    slide = j/100
                                                                                    time_x = []
                                                                                    for k in range(len(freq_list)):
                                                                                        i = freq_list[k]
                                                                                        cube_3 = (lambda h5: 3 * 9 * 10 * np.sqrt(2.99 * ((1+(h5/696000)*(time_rate3 * 300000))**(-16)) + 1.55 * ((1+(h5/696000)*(time_rate3 * 300000))**(-6)) + 0.036 * ((1+(h5/696000)*(time_rate3 * 300000))**(-1.5))))
                                                                                        invcube_3 = inversefunc(cube_3, y_values=i)
                                                                                        time_x.append(invcube_3)
                                                                                    if min(fitting_new) > sum((((time_x + np.array(slide)) - time_list) ** 2))/len(freq_list):
                                                                                        fitting_new.append(sum((((time_x + np.array(slide)) - time_list) ** 2))/len(freq_list))
                                                                                        time_rate_result_new.append(time_rate3)
                                                                                        slide_result_new.append(slide)
                                                                            print (fitting_new[-1])
                                                                            slide = slide_result_new[-1]
                                                                            time_rate3 = time_rate_result_new[-1]
                                                                            print (slide_result_new[-1], time_rate_result_new[-1])
                                                                            h5 = np.arange(min(time_list) - 5, max(time_list) + 5, 0.01)
                                                                            x5 = 3 * 9 * 10 * np.sqrt(2.99 * ((1+((h5 - slide)/696000)*(time_rate3 * 300000))**(-16)) + 1.55* ((1+((h5 - slide)/696000)*(time_rate3 * 300000))**(-6)) + 0.036 * ((1+((h5 - slide)/696000)*(time_rate3 * 300000))**(-1.5)))
                                                                            #h5 = h5[6:]
                                                                            #x5 = x5[6:]
                                                                            #p_5 = np.polyfit(x5, h5, 5)
                                                                            #xx_5 = np.linspace(min(x5) - 5, max(x5) + 5, 100)
                                                                            #yy_5 = np.polyval(p_5, xx_5)
                                                                            #ax2.plot(yy_5, xx_5, 'k', label = 'freq_drift')
                                                                            ax2.plot(h5, x5, '-', label = 'allen_model')
                                                                            #plt.ylim(10, 80)
                                                                                    
                                                                                    
                                                                            
                                                                            
                                                                            
                                                                            #周波数方向に最小二乗
                                                                            #import numpy as np
                                                                            #import scipy.optimize
                                                                            #import matplotlib.pylab as plt
                                                                            #
                                                                            # data which you want to fit
                                                                            time_x = []
                                                                            xdata = np.array(time_list)
                                                                            ydata = np.array(freq_list)
                                                                            # initial guess for the parameters
                                                                            parameter_initial = np.array([float(np.average(time_list)) - 5, 0.5]) #a, b
                                                                            # function to fit
                                                                            def func(x, a, b):
                                                                                return 3 * 9 * 10 * np.sqrt((2.99*(1+((x - a)/696000)*(b * 300000))**(-16) + 1.55*(1+((x - a)/696000)*(b * 300000))**(-6) + 0.036*(1+((x - a)/696000)*(b * 300000))**(-1.5)))
                                                                            
                                                                            paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata, ydata, p0=parameter_initial)
                                                                            #print ("paramater =", paramater_optimal)
                                                                            print ('velocity =', paramater_optimal[1])
                                                                            final_xdata = np.arange(min(xdata) - 5, max(xdata) + 6, 0.01)
                                                                            y = func(final_xdata,paramater_optimal[0],paramater_optimal[1])
                                                                            plt.plot(xdata, ydata, 'o')
                                                                            plt.plot(final_xdata, y, '-')
                                                                            plt.ylim(10, 80)
                                                                            
                                                                            for k in range(len(freq_list)):
                                                                            #    print(k)
                                                                                i = freq_list[k]
                                                                                cube_3 = (lambda h5: 3 * 9 * 10 * np.sqrt((2.99*(1+(h5/696000)*(paramater_optimal[1] * 300000))**(-16)+1.55*(1+(h5/696000)*(paramater_optimal[1] * 300000))**(-6)+0.036*(1+(h5/696000)*(paramater_optimal[1] * 300000))**(-1.5))))
                                                                                invcube_3 = inversefunc(cube_3, y_values=i)
                                                                                time_x.append(invcube_3)
                                                                            print (sum((((time_x + np.array(paramater_optimal[0])) - time_list) ** 2))/len(freq_list))
                                                                            ######################################################################################
                                                                            ##                                                                plt.title('Used_data')
                                                                            plt.xlabel('Time[sec]',fontsize=20)
                                                                            #                                                            plt.ylabel('Frequency[MHz]')
                                                                            plt.legend(fontsize=12)
                                                                            plt.tick_params(labelsize=18)
                                                                            plt.xlim(min(time_list) - 5, max(time_list) + 5)
                                                                            plt.ylim(10, 80)
                                                                            #                                                                print (p_2)
                                                                            #if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                            #    os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                            #filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'_'+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'_'+ str(1/p_4[0]) +'_'+ drift_std_new
                                                                            #filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+filename1+'.png'
                                                                            if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                                os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                            filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'compare.png'
                                                                            plt.savefig(filename)
                                                                            plt.show()
                                                                            plt.close()
                                                                            ###########################################
                                                                            arr5 = arr2
                                                                            arr6 = arr2
                                                                            for i in range (len(final_xdata)):
                                                                                if y[i] < 79.825:
                                                                                    if y[i] > 10:
                                                                                        freq_y_0 = getNearestValue(cdf_file['Frequency'], y[i])
                                                                                        freq_y = int(399 - (freq_y_0 - 10)/0.175)
                                                                                        freq_time = int(round(final_xdata[i]))
                                                                                        arr5[freq_y][freq_time] = 100
                                                                            for i in range (len(h5)):
                                                                                if x5[i] < 79.825:
                                                                                    if x5[i] > 10:
                                                                                        freq_y_0 = getNearestValue(cdf_file['Frequency'], x5[i])
                                                                                        freq_y = int(399 - (freq_y_0 - 10)/0.175)
                                                                                        freq_time = int(round(h5[i]))
                                                                                        arr6[freq_y][freq_time] = 100
                                                                            
                                                                            plt.close(1)
                                                                            ###########################################
                                                                            #                                                            figure_=plt.figure(1,figsize=(16,10))
                                                                            figure_=plt.figure(1,figsize=(20,10))
                                                                            gs = gridspec.GridSpec(20, 10)
                                                                            axes_2 = figure_.add_subplot(gs[:, :])
                                                                            #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                            ax2 = axes_2.imshow(arr6, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                      aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                            axes_2.xaxis_date()
                                                                            date_format = mdates.DateFormatter('%H:%M:%S')
                                                                            axes_2.xaxis.set_major_formatter(date_format)
                                                                            plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
                                                                            plt.xlabel('Time',fontsize=20)
                                                                            plt.ylabel('Frequency [MHz]',fontsize=20)
                                                                            plt.colorbar(ax2,label='from Background [dB]')
                                                                            
                                                                            plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                            figure_.autofmt_xdate()
                                                                            if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                                os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                            filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'fv' + str(paramater_optimal[1]) + 'tv' + str(time_rate3)+ '.png'
                                                                            plt.savefig(filename)
                                                                            plt.show()
                                                                            plt.close()