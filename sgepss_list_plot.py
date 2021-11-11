#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:19:15 2020

@author: yuichiro
"""

import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
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
after_plot = str('residual_test_1')
after_after_check = str('_135')
plot_dic = str('sgepss_list')
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
intensity_range = 10
threshold_frequency_final = 3 * freq_check_range
Parent_lab = len(Parent_directory.split('/')) - 1


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

def allen_model(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5)))))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    
#        print ('aaa')
#        print(fitting)
    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    print (h_start)
#        print ('aaa')
#        print (fitting_new[-1])
    return (-slide_result_new[-1], time_rate_result_new[-1], fitting_new[-1], h_start)



#####################
#plot_date_select
Obs_date = []
#path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' +str(sigma_value)+ '/*/*/*'
path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/*/*compare.png'
File = glob.glob(path, recursive=True)
#print(File)
File1=len(File)
print (File1)
for cstr in File:
    a=cstr.split('/')
#            line=a[9]+'\n'
#    line = a[12]
    line = a[Parent_lab + 7]
#    print(line)
    file_name_separate = line.split('_')
    # if not file_name_separate[4] == 'sigma':
    if int(file_name_separate [0][:8]) >= 20140125:
        if int(file_name_separate [0][:8]) <= 20141231:
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
#    path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' +str(sigma_value)+ '/' + year + '/'+ month +'/'+ Obs_date_final[x] +'*'
    path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' + year + '/'+ Obs_date_final[x] +'*compare.png'
    File = glob.glob(path, recursive=True)
    File1=len(File)
    print (File1)
    for cstr in File:
        a = cstr.split('/')
#        line = a[12]
        line = a[Parent_lab + 7]
#        print(line)
        file_name_separate = line.split('_')
        if not file_name_separate[4] == 'sigma':
            Obs_time_start.append(int(file_name_separate[3]))
            Obs_burst_start.append(int(file_name_separate[5]))
            Obs_burst_end.append(int(file_name_separate[6]))
#####################
#data_import_and_editation
    path = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/*' + year + month + day + '*_'  + '*' + '.cdf'
    file_name = glob.glob(path, recursive=True)[0].split('/')[Parent_lab + 6]
    # print (File)
    # print (Obs_time_start)

    file_name_separate = file_name.split('_')
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

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
    # cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
    LL = cdf_file['LL'] 
    RR = cdf_file['RR'] 
    
    data_r_0 = RR
    data_l_0 = LL
    diff_r_last =(data_r_0).T
    diff_r_last = np.flipud(diff_r_last)[0:286] * 0.3125
    diff_l_last =(data_l_0).T
    diff_l_last = np.flipud(diff_l_last)[0:286] * 0.3125

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




    ####################
    for z in range(len(Obs_time_start)):
        if int(Obs_time_start[z]) % time_band == 0:
            t = round(int(Obs_time_start[z])/time_band)
        else:
            t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
        time = round(time_band*t)
        print (time)
        #+1 is due to move_average
        start = epoch[time + 1]
        start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
        # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        time = round(time_band*(t+1) + time_co)
        end = epoch[time + 1]
        end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
        # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        Time_start = start.strftime('%H:%M:%S')
        Time_end = end.strftime('%H:%M:%S')
        print (start)
        print(Time_start+'-'+Time_end)
        start_1 = start.timestamp()
        end_1 = end.timestamp()

        x_lims_1 = list(map(dt.datetime.fromtimestamp, [start_1, end_1]))
        x_lims = mdates.date2num(x_lims_1)

        # Set some generic y-limits.
        y_lims = [10, 80]


        diff_db_plot_l = diff_db_min_med[:, time - time_band - time_co:time]
        diff_db_l = diff_db[:, time - time_band - time_co:time]
        diff_move_db_new = diff_move_db[:, time - time_band - time_co:time]


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
                                                for k in range(check_frequency_range):
                                                    if max(arr2[k]) > -10:
                                                        if (len([l for l in arr2[k] if l == max(arr2[k])])) == 1:
                                                            freq_list.append(cdf_file['Frequency'][399 - k])
                                                            time_list.append(np.argmax(arr2[k]))

                                                p_1 = np.polyfit(freq_list, time_list, 1) #determine coefficients


                                                if 1/p_1[0] < 0:
                                                    #drift_allen_model_*1_80_69.5
                                                    if 1/p_1[0] > -107.22096882538592:
                                                        if np.count_nonzero(cdf_file['Status'][time - time_band - time_co + 1:time + 1] == 0) == 0:
                                                            if np.count_nonzero(cdf_file['Status'][time - time_band - time_co + 1:time + 1] == 17) == 0:
                                                                middle = round((min(drift_time_start)+max(drift_time_end))/2) - 25
                                                                arr5 = [[-10 for l in range(50)] for n in range(400)]
                                                                try_list = [[-10 for l in range(50)] for n in range(400)]
                                                                sss = 0
                                                                for k in range(50):
                                                                    middle_k = middle + k
                                                                    if middle_k >= 0:
                                                                        if middle_k <= 399:
                                                                            start_number = 0
                                                                            for l in range(check_frequency_range):
                                                                                if arr2[l][middle_k] > -10:
                                                                                    arr5[l][k] = arr2[l][middle_k]
                                                                                    try_list[l][k] = diff_move_db_new[l][min(drift_time_start) + sss]
                                                                                    start_number += 1
                                                                            if start_number > 0:
                                                                                sss += 1

                                                                try_list = np.array(try_list)
                                                                event_start = str(min(drift_time_start))
                                                                event_end = str(max(drift_time_end))
######################################################################################
                                                                if int(event_start) == int(Obs_burst_start[z]):
                                                                    if int(event_end) == int(Obs_burst_end[z]):
                                                                        ###########################################
                                                                        freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                                        freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                                        event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                                        freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                                        fontsize = 110
                                                                        ticksize = 100
                                                                        y_lims = [30, 80]
                                                                        plt.close(1)
                                                                        #                   
                                                                        figure_=plt.figure(1,figsize=(140,50))
                                                                        gs = gridspec.GridSpec(140, 50)
                                                                        # axes_2 = figure_.add_subplot(gs[:, :])
                                                                        # #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                        # ax2 = axes_2.imshow(arr5[0:286], extent = [0, 50, 30, y_lims[1]], 
                                                                        #           aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                        # # ax2.set_label('from Background [dB]',size=32)
                                                                        # # axes_2.xaxis_date()
                                                                        # # date_format = mdates.DateFormatter('%H:%M:%S')
                                                                        # # axes_2.xaxis.set_major_formatter(date_format)
                                                                        # plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=32)
                                                                        # plt.xlabel('Time',fontsize=32)
                                                                        # plt.ylabel('Frequency [MHz]',fontsize=32)
                                                                        # cbar = plt.colorbar(ax2)
                                                                        # cbar.ax.tick_params(labelsize=32)
                                                                        # cbar.set_label('from Background [dB]', size=32)
                                                                        # axes_2.tick_params(labelsize=25)
                                                                        
                                                                        # plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                        # figure_.autofmt_xdate()
                                                                        # # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+plot_dic):
                                                                        #     # os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+plot_dic)
                                                                        # # filename = Parent_directory + '/solar_burst/Nancay/plot/'+plot_dic+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'compare.png'
                                                                        # # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                        #     # os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                        # # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8] + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
                                                                        # # plt.savefig(filename)
                                                                        # plt.show()
                                                                        # plt.close()

                                                                        axes_1 = figure_.add_subplot(gs[:55, :20])
                                                                #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                        ax1 = axes_1.imshow(diff_db_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                  aspect='auto',cmap='jet',vmin= -5 + min_db[228],vmax = quartile_db_l[228] + min_db[228] + 5)
                                                                        axes_1.xaxis_date()
                                                                        date_format = mdates.DateFormatter('%H:%M:%S')
                                                                        axes_1.xaxis.set_major_formatter(date_format)
                                                                        plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize )
                                                                        plt.xlabel('Time (UT)',fontsize=fontsize )
                                                                        plt.ylabel('Frequency [MHz]',fontsize=fontsize)
                                                                        cbar = plt.colorbar(ax1)
                                                                        cbar.ax.tick_params(labelsize=ticksize)
                                                                        cbar.set_label('Decibel [dB]', size=fontsize)
                                                                        axes_1.tick_params(labelsize=ticksize)
                                                                        # plt.colorbar(ax1,label='Decibel [dB]')
                                                                        figure_.autofmt_xdate()

                                                                        # figure_=plt.figure(1,figsize=(18,5))
                                                                        # gs = gridspec.GridSpec(20, 12)
                                                                        axes_1 = figure_.add_subplot(gs[85:, :20])
                                                                #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                        ax1 = axes_1.imshow(diff_db_plot_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                                  aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[228] + 5)
                                                                        axes_1.xaxis_date()
                                                                        date_format = mdates.DateFormatter('%H:%M:%S')
                                                                        axes_1.xaxis.set_major_formatter(date_format)
                                                                        plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
                                                                        plt.xlabel('Time (UT)',fontsize=fontsize)
                                                                        plt.ylabel('Frequency [MHz]',fontsize=fontsize)
                                                                        cbar = plt.colorbar(ax1)
                                                                        cbar.ax.tick_params(labelsize=ticksize)
                                                                        cbar.set_label('from Background [dB]', size=fontsize)
                                                                        axes_1.tick_params(labelsize=ticksize)
                                                                        # plt.colorbar(ax1,label='from Background [dB]')
                                                                        figure_.autofmt_xdate()
                                                                
                                                                
                                                                
                                                                
                                                                

                                                                
                                                                        axes_2 = figure_.add_subplot(gs[:, 21:34])
                                                                        ax2 = axes_2.imshow(arr5[0:286], extent = [0, 50, 30, y_lims[1]], 
                                                                                  aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                        # ax2.set_label('from Background [dB]',size=32)
                                                                        # axes_2.xaxis_date()
                                                                        # date_format = mdates.DateFormatter('%H:%M:%S')
                                                                        # axes_2.xaxis.set_major_formatter(date_format)
                                                                        plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
                                                                        plt.xlabel('Time[sec]',fontsize=fontsize)
                                                                        plt.ylabel('Frequency [MHz]',fontsize=fontsize)
                                                                        cbar = plt.colorbar(ax2)
                                                                        cbar.ax.tick_params(labelsize=ticksize)
                                                                        cbar.set_label('from Background [dB]', size=fontsize)
                                                                        axes_2.tick_params(labelsize=ticksize)
                                                                        
                                                                        # plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                        figure_.autofmt_xdate()


                                                                        axes_2 = figure_.add_subplot(gs[:,38:51])
                                                                        # ax2.plot(time_list, freq_list, "bo", label = 'Peak data') #Interpolation
                                                                        
                                                                        ######################################################################################
                                                                        time_rate_final = []
                                                                        residual_list = []
                                                                        slide_list = []
                                                                        x_time = []
                                                                        y_freq = []
                                                                        factor_list = [1,2,3,4,5]
                                                                        for factor in factor_list:
                                                                            slide, time_rate5, residual, h_start = allen_model(factor, time_list, freq_list)
                                                                            time_rate_final.append(time_rate5)
                                                                            residual_list.append(residual)
                                                                            slide_list.append(slide)
                                                                            h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
                                                                            h5_0 = h5_0/1000
                                                                            x_time.append(h5_0)
                                                                            y_freq.append(9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))

                                                                        # axes_2 = figure_.add_subplot(gs[:, 5:])
                                                                        axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=30)
#                                                                                figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                        x_cmap = np.arange(10, 80, 0.175)
                                                                        y_cmap = np.arange(0, 400, 1)
                                                                        cs = axes_2.contourf(y_cmap, x_cmap, arr2[::-1], levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[228] + 10)
                                                                        cs.cmap.set_over('red')
                                                                        cs.cmap.set_under('blue')
                                                                        cycle = 0
                                                                        for factor in factor_list:
                                                                            ###########################################
                                                                            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 30.0)
                                                                            cycle += 1
#                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
                                                                        plt.xlim(min(time_list) - 10, max(time_list) + 10)
                                                                        plt.ylim(min(freq_list), max(freq_list))
                                                                        plt.title('Nancay: '+year+'-'+month+'-'+day+ ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize  + 10)
                                                                        plt.xlabel('Time[sec]',fontsize=fontsize)
                                                                        plt.ylabel('Frequency [MHz]',fontsize=fontsize)
                                                                        plt.tick_params(labelsize=ticksize)
                                                                        plt.legend(fontsize=ticksize - 40)
                                                                        # plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                        figure_.autofmt_xdate()
                                                                
                                                                        if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+plot_dic+'/'+year):
                                                                            os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+plot_dic+'/'+year)
                                                                        filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+plot_dic+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
                                                                        plt.savefig(filename)
                                                                        plt.show()
                                                                        plt.close()
