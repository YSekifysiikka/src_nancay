#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 16:27:50 2019

@author: yuichiro
"""
#median, duration, sigma

import sys
sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
import cdflib
import scipy
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy import signal
import csv
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



###############
sigma_range = 1
sigma_start = 2
sigma_mul = 1
#when you only check one threshold
sigma_value = 2
after_plot = str('flare_check')
after_after_plot = str('no3')
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
duration = 4
threshold_frequency = 20
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


file_path = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/final.txt'
gen = file_generator(file_path)
for file in gen:

    file_name = file[:-1]
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
    file = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    LL = cdf_file['LL'] 
    RR = cdf_file['RR'] 
    
    data_r_0 = RR
    data_l_0 = LL
    diff_r_last =(data_r_0).T
    diff_r_last = np.flipud(diff_r_last)
    diff_l_last =(data_l_0).T
    diff_l_last = np.flipud(diff_l_last)


    diff_db_pre = []
    diff_power_check = []
    for i in range(diff_l_last.shape[0]):
        l = diff_l_last[i]
        r = diff_r_last[i]
        for k in range(diff_l_last.shape[1]):
            diff_power = (((10 ** ((r[k])/10)) + (10 ** ((l[k])/10)))/2)
            diff_power_check.append(diff_power)
            diff_db_pre.append(math.log10(diff_power) * 10)
    diff_P = np.array(diff_power_check).reshape([diff_l_last.shape[0], diff_l_last.shape[1]])
    diff_db = np.array(diff_db_pre).reshape([diff_l_last.shape[0], diff_l_last.shape[1]])
    #diff_db = diff_db_pre.reshape([diff_l_last_1.shape[0], diff_l_last_1.shape[1]])
    ####################
    y_power = []
    y_db = []
    num = int(move_ave)
    b = np.ones(num)/num
    for i in range (diff_db.shape[0]):
        y_power.append(np.convolve(diff_P[i], b, mode='valid'))
    for i in range (diff_db.shape[0]):
        for k in range(len(y_power[i])):
            y_db.append(math.log10(y_power[i][k]) * 10)
    diff_move_db = np.array(y_db).reshape([diff_l_last.shape[0], diff_l_last.shape[1] -2 ])
    ####################
    min_db = []
    min_power = np.amin(y_power, axis=1)
    for i in range(min_power.shape[0]):
        min_db.append(math.log10(min_power[i]) * 10)
    diff_db_min_med = (diff_move_db.T - min_db).T
#    diff_db_min = (diff_move_db.T - min_db).T
#    
#    
#    l_r = []
#    for change in range(diff_db_min.shape[1]):
#        l_r.append(scipy.signal.medfilt(diff_db_min[:,change], kernel_size = median_size))
#    diff_db_min_med = np.array(l_r).T
    
   
    
     for t in range (math.floor((diff_db_min_med.shape[1]-time_co)/time_band)):
#                ################################
#                for l in range(sigma_range):
#                    sigma_value = sigma_start + sigma_mul*l
#                ################################


        time = time_band*t
        print (time)
        #+1 is due to move_average
        start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        time = time_band*(t+1) + time_co
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
                        y_over_analysis_time_l.append(time_band*t + j)
                        y_over_analysis_data_l.append(y_over_group_l[i])

            if len(y_over_final_l) > 0:
                TYPE3.append(time_band*t + j)

        plt.close(1)
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

        if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day):
            os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day)
        filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'_sigma_l_r.png'
        plt.savefig(filename)
        plt.show()
        plt.close()


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
#check_2sec&frequency_sequence
                        frequency_sequence = []
                        frequency_separate = []
                        for i in range(check_frequency_range):
                            for j in range(400):
                                if arr[i][j] > -10:
                                    if j == 0:
                                        if arr[i][j+1] == -10:
                                            arr[i][j] = -10
                                        else:
                                            frequency_sequence.append(i)
                                    elif j == 399:
                                        if arr[i][j-1] == -10:
                                            arr[i][j] = -10
                                        else:
                                            frequency_sequence.append(i)
                                    else:
                                        if arr[i][j+1] == -10 and arr[i][j-1]== -10:
                                            arr[i][j] = -10
                                        else:
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
                                            if drift_time_end[7] < drift_time_end[-8]:
                                                if not max(drift_freq) == 285:
                                                    if not min(drift_freq) ==0:
                                                        event_start = str(min(drift_time_start))
                                                        event_end = str(max(drift_time_end))
                                                        freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                        freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                        event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                        freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                        y_lims = [10, 80]
                                                        plt.close(1)
                                                        figure_=plt.figure(1,figsize=(16,10))
                                                        axes_2=figure_.add_subplot(111)
                                            #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                        ax2 = axes_2.imshow(arr2, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                  aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[228] + 10)
                                                        axes_2.xaxis_date()
                                                        date_format = mdates.DateFormatter('%H:%M:%S')
                                                        axes_2.xaxis.set_major_formatter(date_format)
                                                        plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
                                                        plt.xlabel('Time (UT)',fontsize=20)
                                                        plt.ylabel('Frequency [MHz]',fontsize=20)
                                                        plt.colorbar(ax2,label='from Background [dB]')
    
                                                        plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                        figure_.autofmt_xdate()
                                                        if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day):
                                                            os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day)
                                                        filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end
                                                        filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'/'+filename1+'.png'
                                                        plt.savefig(filename)
                                                        plt.show()
                                                        plt.close()


#####################################
    for t in range (-1, -2, -1):
#                ################################
#                for l in range(sigma_range):
#                    sigma_value = sigma_start + sigma_mul*l
#                ################################

    
    
        time = diff_db_min_med.shape[1] + t*(time_band+time_co)
        print (time)
        start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time -1]/1000000000)
        time = diff_db_min_med.shape[1] - 1
        end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time -1]/1000000000)
        Time_start = start.strftime('%H:%M:%S')
        Time_end = end.strftime('%H:%M:%S')
        print (start)
        print(Time_start+'-'+Time_end)
        start = start.timestamp()
        end = end.timestamp()
        #    diff_r_db = diff_r_min[:, time - time_band - time_co:time]
        #    diff_l_db = diff_l_min[:, time - time_band - time_co:time]

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
                        y_over_analysis_time_l.append(time_band*t + j)
                        y_over_analysis_data_l.append(y_over_group_l[i])

            if len(y_over_final_l) > 0:
                TYPE3.append(time_band*t + j)

        plt.close(1)
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

        if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day):
            os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day)
        filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'_sigma_l_r.png'
        plt.savefig(filename)
        plt.show()
        plt.close()


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
#check_2sec&frequency_sequence
                        frequency_sequence = []
                        frequency_separate = []
                        for i in range(check_frequency_range):
                            for j in range(400):
                                if arr[i][j] > -10:
                                    if j == 0:
                                        if arr[i][j+1] == -10:
                                            arr[i][j] = -10
                                        else:
                                            frequency_sequence.append(i)
                                    elif j == 399:
                                        if arr[i][j-1] == -10:
                                            arr[i][j] = -10
                                        else:
                                            frequency_sequence.append(i)
                                    else:
                                        if arr[i][j+1] == -10 and arr[i][j-1]== -10:
                                            arr[i][j] = -10
                                        else:
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
                                            if drift_time_end[7] < drift_time_end[-8]:
                                                if not max(drift_freq) == 285:
                                                    if not min(drift_freq) ==0:
                                                        event_start = str(min(drift_time_start))
                                                        event_end = str(max(drift_time_end))
                                                        freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                        freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                        event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                        freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                        y_lims = [10, 80]
                                                        plt.close(1)
                                                        figure_=plt.figure(1,figsize=(16,10))
                                                        axes_2=figure_.add_subplot(111)
                                            #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                        ax2 = axes_2.imshow(arr2, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                  aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[228] + 10)
                                                        axes_2.xaxis_date()
                                                        date_format = mdates.DateFormatter('%H:%M:%S')
                                                        axes_2.xaxis.set_major_formatter(date_format)
                                                        plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
                                                        plt.xlabel('Time (UT)',fontsize=20)
                                                        plt.ylabel('Frequency [MHz]',fontsize=20)
                                                        plt.colorbar(ax2,label='from Background [dB]')
    
                                                        plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                        figure_.autofmt_xdate()
                                                        if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day):
                                                            os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day)
                                                        filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end
                                                        filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'/'+filename1+'.png'
                                                        plt.savefig(filename)
                                                        plt.show()
                                                        plt.close()