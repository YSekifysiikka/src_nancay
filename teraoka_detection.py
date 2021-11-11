#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 15:26:29 2021

@author: yuichiro
"""
#!/usr/bin/env python3

import glob
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cdflib

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
    yyyy=date_OBs[0:4]
    mm=date_OBs[4:6]

    file = Parent_directory + '/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/'+file_name
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
    return diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs


import math
import datetime as dt
import matplotlib.dates as mdates
import datetime

def separated_data(diff_db, diff_db_min_med, epoch, time_co, time_band, t):
    if t == math.floor((diff_db_min_med.shape[1]-time_co)/time_band):
        t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    time = round(time_band*t)
    print (time)
    #+1 is due to move_average
    start = epoch[time + 1]
    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    time = round(time_band*(t+1) + time_co)
    end = epoch[time + 1]
    end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)
    start = start.timestamp()
    end = end.timestamp()
    

    x_lims = []
    x_lims.append(dt.datetime.fromtimestamp(start))
    x_lims.append(dt.datetime.fromtimestamp(end))
    x_lims = mdates.date2num(x_lims)

    diff_db_plot_sep = diff_db_min_med[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    diff_db_sep = diff_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    return diff_db_plot_sep, diff_db_sep, x_lims, time, Time_start, Time_end, t
    
def threshold_array(diff_db_plot_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration):
    quartile_db_l = []
    mean_l_list = []
    stdev_sub = []
    quartile_power = []



    for i in range(diff_db_plot_sep.shape[0]):
    #    for i in range(0, 357, 1):
        quartile_db_25_l = np.percentile(diff_db_plot_sep[i], 25)
        quartile_db_each_l = []
        for k in range(diff_db_plot_sep[i].shape[0]):
            if diff_db_plot_sep[i][k] <= quartile_db_25_l:
                diff_power_quartile_l = (10 ** ((diff_db_plot_sep[i][k])/10))
                quartile_db_each_l.append(diff_power_quartile_l)
    #                quartile_db_each.append(math.log10(diff_power) * 10)
        m_l = np.mean(quartile_db_each_l)
        stdev_l = np.std(quartile_db_each_l)
        sigma_l = m_l + sigma_value * stdev_l
        sigma_db_l = (math.log10(sigma_l) * 10)
        quartile_power.append(sigma_l)
        quartile_db_l.append(sigma_db_l)
        mean_l_list.append(math.log10(m_l) * 10)
    quartile_power = np.array(quartile_power)
    quartile_db_l = np.array(quartile_db_l)
    mean_l_list = np.array(mean_l_list)
    stdev_sub = quartile_db_l - mean_l_list
    diff_power_last_l = ((10 ** ((diff_db_plot_sep)/10)).T - quartile_power).T
    
    arr_threshold_1 = np.where(diff_power_last_l > 1, diff_power_last_l, 1)
    arr_threshold = np.log10(arr_threshold_1) * 10

    return arr_threshold, mean_l_list, quartile_db_l, quartile_power, diff_power_last_l, stdev_sub


def plot_array(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    figure_=plt.figure(1,figsize=(6,4))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    ax2 = axes_2.imshow(arr_threshold[freq_start_idx:freq_end_idx + 1], extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= 0 ,vmax = 10)
    # ax2 = axes_2.imshow(arr_threshold[freq_start_idx:freq_end_idx + 1], 
    #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day, fontsize=20)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Decibel [dB]', size=20)
    axes_2.tick_params(labelsize=18)

    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()

    plt.ylim(30, 80)
    plt.show()
    plt.close()





def sep_array(arr_threshold, diff_db_plot_sep, Frequency, time_band, time_co,  quartile_power, time, duration, resolution, Status, cnn_plot_time, t_1):
    y_over_analysis_time_l = []
    y_over_analysis_data_l = []
    TYPE3 = []
    TYPE3_group = []
    y_over_l = []
    arr_5_list = []
    event_start_list = []
    event_end_list = []
    freq_start_list = []
    freq_end_list = []
    event_time_gap_list = []
    freq_gap_list = []
    vmin_1_list = []
    vmax_1_list = []
    freq_list_1 = []
    time_list_1 = []
    arr_sep_time_list = []
    for j in range(0, time_band + time_co, 1):
        y_le = arr_threshold[:, j]
        y_over_l = []
        for i in range(y_le.shape[0]):
            if y_le[i] > 0:
                y_over_l.append(i)
            else:
                pass
        y_over_final_l = []
        if len(y_over_l) > 0:
            y_over_group_l = groupSequence(y_over_l)
            for i in range(len(y_over_group_l)):
                if len(y_over_group_l[i]) >= (round(threshold_frequency/resolution,1)):
                    y_over_final_l.append(y_over_group_l[i])
                    y_over_analysis_time_l.append(round(time_band*t_1 + j))
                    y_over_analysis_data_l.append(y_over_group_l[i])

        if len(y_over_final_l) > 0:
            TYPE3.append(round(time_band * t_1 + j))


    TYPE3_final = np.unique(TYPE3)
    if len(TYPE3_final) > 0:
        TYPE3_group.append(groupSequence(TYPE3_final))


    if len(TYPE3_group)>0:
        for m in range (len(TYPE3_group[0])):
            if len(TYPE3_group[0][m]) >= duration:
                if TYPE3_group[0][m][0] >= 0 and TYPE3_group[0][m][-1] <= diff_db_min_med.shape[1]:  
                    arr = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                    for i in range(len(TYPE3_group[0][m])):
    
                        check_start_time_l = ([y for y, Frequency in enumerate(y_over_analysis_time_l) if Frequency == TYPE3_group[0][m][i]])
                        for q in range(len(check_start_time_l)):
                            for p in range(len(y_over_analysis_data_l[check_start_time_l[q]])):
                                arr[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = diff_db_plot_sep[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)]
                    arr_freq_time = np.array(arr)

                    frequency_sequence = []
                    frequency_separate = []
                    for i in range(arr_freq_time.shape[0]):
                        for j in range(arr_freq_time.shape[1]):
                            if arr_freq_time[i][j] > -10:
                                frequency_sequence.append(i)
                    frequency_sequence_final = np.unique(frequency_sequence)
                    if len(frequency_sequence_final) > 0:
                        frequency_separate.append(groupSequence(frequency_sequence_final))
                        for i in range(len(frequency_separate[0])):
                            arr_1 = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                            for j in range(len(frequency_separate[0][i])):
                                for k in range(arr_freq_time.shape[1]):
                                    if arr_freq_time[frequency_separate[0][i][j]][k] > -10:
                                        arr_1[frequency_separate[0][i][j]][k] = arr_freq_time[frequency_separate[0][i][j]][k]
                            arr_sep_freq = np.array(arr_1)


                            time_sequence = []
                            time_sequence_final_group = []
                            for j in range(arr_sep_freq.shape[1]):
                                for k in range(arr_sep_freq.shape[0]):
                                    if not arr_sep_freq[k][j] == -10:
                                        time_sequence.append(j)
                            time_sequence_final = np.unique(time_sequence)
                            if len(time_sequence) > 0:
                                time_sequence_final_group.append(groupSequence(time_sequence_final))
                                for j in range(len(time_sequence_final_group[0])):
                                    if len(time_sequence_final_group[0][j]) >= duration:
                                        arr_2 = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                                        for k in range(len(time_sequence_final_group[0][j])):
                                            for l in range(arr_freq_time.shape[0]):
                                                if arr_sep_freq[l][time_sequence_final_group[0][j][k]] > -10:
                                                    arr_2[l][time_sequence_final_group[0][j][k]] = arr_sep_freq[l][time_sequence_final_group[0][j][k]]
                                        arr_sep_time = np.array(arr_2)
            
                                        drift_freq = []
                                        drift_time_end = []
                                        drift_time_start = []
                                        for k in range(arr_freq_time.shape[0]):
                                            drift_time = []
                                            for l in range(arr_freq_time.shape[1]):
                                                if arr_sep_time[k][l] > -10:
                                                    drift_freq.append(k)
                                                    drift_time.append(l)
                                            if len(drift_time) > 0:
                                                drift_time_end.append(drift_time[-1])
                                                drift_time_start.append(drift_time[0])
                                        if len(drift_time_end) >= (round(threshold_frequency_final/resolution,1)):
                                            freq_each_max = np.max(arr_sep_time, axis=1)
                                            arr3 = []
                                            for l in range(len(freq_each_max)):
                                                if freq_each_max[l] > -10:
                                                    arr3.append(freq_each_max[l])
                                            arr4 = []
                                            for l in range(len(arr_sep_time)):
                                                for k in range(len(arr_sep_time[l])):
                                                    if arr_sep_time[l][k] > -10:
                                                        arr4.append(arr_sep_time[l][k])
                                                    
                                            freq_list = []
                                            time_list = []
                                            for k in range(arr_freq_time.shape[0]):
                                                if max(arr_sep_time[k]) > -10:
                                                    if (len([l for l in arr_sep_time[k] if l == max(arr_sep_time[k])])) == 1:
                                                        freq_list.append(Frequency[k])
                                                        time_list.append(np.argmax(arr_sep_time[k]))
            
                                            p_1 = np.polyfit(freq_list, time_list, 1) #determine coefficients
            
            
                                            # if 1/p_1[0] < 0:
                                            #     if 1/p_1[0] > -107.22096882538592:
                                            if np.count_nonzero(Status[time - time_band - time_co + 1:time + 1] == 0) == 0:
                                                if np.count_nonzero(Status[time - time_band - time_co + 1:time + 1] == 17) == 0:
                                                    middle = round((min(drift_time_start)+max(drift_time_end))/2) - 25
                                                    arr_5 = [[-10 for l in range(cnn_plot_time)] for n in range(Frequency.shape[0])]
                                                    for k in range(cnn_plot_time):
                                                        middle_k = middle + k
                                                        if middle_k >= 0:
                                                            if middle_k <= 399:
                                                                for l in range(arr_sep_time.shape[0]):
                                                                    if arr_2[l][middle_k] > -10:
                                                                        arr_5[l][k] = arr_2[l][middle_k]
                                                    arr_5_list.append(arr_5)      
                                                    event_start_list.append(str(min(drift_time_start)))
                                                    event_end_list.append(str(max(drift_time_end)))
                                                    freq_start_list.append(str(Frequency[min(drift_freq)]))
                                                    freq_end_list.append(str(Frequency[max(drift_freq)]))
                                                    event_time_gap_list.append(str(max(drift_time_end) - min(drift_time_start) + 1))
                                                    freq_gap_list.append(str(round(Frequency[min(drift_freq)] - Frequency[max(drift_freq)] + resolution, 3)))
                                                    vmin_1_list.append(min(arr4))
                                                    vmax_1_list.append(np.percentile(arr3, 75))
                                                    freq_list_1.append(freq_list)
                                                    time_list_1.append(time_list)
                                                    arr_sep_time_list.append(arr_sep_time)
    return arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list_1, time_list_1, arr_sep_time_list

def prepare_cnn(arr_5, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, date_OBs, Time_start, Time_end, Frequency):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    y_lims = [Frequency[-1], Frequency[0]]
    plt.close(1)

    figure_=plt.figure(1,figsize=(10,10))
    axes_2 = figure_.add_subplot(1, 1, 1)
    ax2 = axes_2.imshow(arr_5[freq_start_idx:freq_end_idx + 1], extent = [0, 50,  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1-2 ,vmax = vmax_1)
    # plt.axis('off')
    # axes_2.xaxis_date()
    # date_format = mdates.DateFormatter('%H:%M')
    # axes_2.xaxis.set_major_formatter(date_format)

    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Decibel [dB]', size=20)

    plt.title('Nancay: '+year+'-'+month+'-'+day, fontsize=20)
    plt.xlabel('Time [sec]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    axes_2.tick_params(labelsize=18)
    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    # figure_.autofmt_xdate()
    plt.show()
    plt.close()
    return





sigma_value = 2
after_plot = str('af_sgepss')
time_band = 340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
freq_check_range = 20
threshold_frequency_final = 10.5
cnn_plot_time = 50
file_path = Parent_directory + '/solar_burst/Nancay/final.txt'
save_place = 'cnn_af_sgepss'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 1.35
db_setting = 40




import pandas as pd
date_in=[20130405,20130405]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')

DATE=sdate
while DATE <= edate:
    date=DATE.strftime(format='%Y%m%d')
    print(date)
    try:
        yyyy = date[:4]
        mm = date[4:6]
        file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[10]
        diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs = read_data(Parent_directory, file_name, 3, 80, 30)
        for t in range (math.floor(((diff_db_min_med.shape[1]-time_co)/time_band) + 1)):
            diff_db_plot_sep, diff_db_sep, x_lims, time, Time_start, Time_end, t_1 = separated_data(diff_db, diff_db_min_med, epoch, time_co, time_band, t)
            plot_array(diff_db_plot_sep, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)

            arr_threshold, mean_l_list, quartile_db_l, quartile_power, diff_power_last_l, stdev_sub = threshold_array(diff_db_plot_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration)
            plot_array(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)

            arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list, time_list, arr_sep_time_list = sep_array(arr_threshold, diff_db_plot_sep, Frequency, time_band, time_co,  quartile_power, time, duration, resolution, Status, cnn_plot_time, t_1)
            if len(arr_5_list) == 0:
                pass
            else:
                for i in range(len(arr_5_list)):
                    prepare_cnn(arr_5_list[i], event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], date_OBs, Time_start, Time_end, Frequency)
    except:
        print('Plot error: ',date)
    DATE+=pd.to_timedelta(1,unit='day')



