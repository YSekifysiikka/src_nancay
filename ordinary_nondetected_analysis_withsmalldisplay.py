#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:46:17 2021

@author: yuichiro
"""
import glob
# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


def file_generator(file):
    with open(file, encoding="utf-8") as f:
        for line in f:
            yield line

def final_txt_make(Parent_directory, Parent_lab, year, start, end):
    start = format(start, '04') 
    end = format(end, '04') 
    year = str(year)
    start = int(year + start)
    end = int(year + end)
    path = Parent_directory + '/solar_burst/Nancay/data/' + year + '/*/*'+'.cdf'
    File = glob.glob(path, recursive=True)
    File = sorted(File)
    i = open(Parent_directory + '/solar_burst/Nancay/final.txt', 'w')
    for cstr in File:
        a = cstr.split('/')
        line = a[Parent_lab + 6]+'\n'
        a1 = line.split('_')
        if (int(a1[5][:8])) >= start:
          if (int(a1[5][:8])) <= end:
            i.write(line)
    i.close()
    return start, end

# start_date, end_date = final_txt_make(Parent_directory, Parent_lab, 2013, 401, 401)


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def load_model_flare(Parent_directory, file_name, color_setting, image_size, fw, strides, fn_conv2d, output_size):
    color_setting = color_setting  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
    image_size = 128 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
    fw = 3
    strides = 1
    fn_conv2d = 16
    output_size = 2
    model = Sequential()
    model.add(Conv2D(fn_conv2d, (fw, fw), padding='same', strides=strides,
              input_shape=(image_size, image_size, color_setting), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))               
    model.add(Conv2D(128, (fw, fw), padding='same', strides=strides, activation='relu'))
    model.add(Conv2D(256, (fw, fw), padding='same', strides=strides, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))                
    model.add(Dropout(0.2))                                   
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))                                 
    model.add(Dense(output_size, activation='softmax'))
    model.load_weights(Parent_directory + file_name)
    return model


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
    year=date_OBs[0:4]
    month=date_OBs[4:6]

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
    obs_time_list = []
    for i in range(len(epoch[1:-1])):
        time = epoch[i+1]
        obs_time_list.append(dt.datetime(time[0], time[1], time[2], time[3], time[4], time[5], time[6]))
    obs_time_list = np.array(obs_time_list)
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
    return diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, obs_time_list, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)


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
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time = round(time_band*(t+1) + time_co)
    end = epoch[time + 1]
    end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
    # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)
    start = start.timestamp()
    end = end.timestamp()
    
    # print (start, end)

    x_lims = []
    x_lims.append(dt.datetime.fromtimestamp(start))
    x_lims.append(dt.datetime.fromtimestamp(end))

    x_lims = mdates.date2num(x_lims)



    diff_db_plot_sep = diff_db_min_med[:, time - time_band - time_co:time]
    diff_db_sep = diff_db[:, time - time_band - time_co:time]
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


def plot_array_threshold(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    figure_=plt.figure(1,figsize=(12,8))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    ax2 = axes_2.imshow(arr_threshold, extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= 0 ,vmax = 1)
    # ax2 = axes_2.imshow(arr_threshold ,
    #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day, fontsize = 20)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Decibel [dB]', size=20)
    axes_2.tick_params(labelsize=18)

    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()

    plt.ylim(Frequency[-1], Frequency[0])
    plt.show()
    plt.close()
    return

def plot_array_threshold_2(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, min_db, quartile_db_l):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    figure_=plt.figure(1,figsize=(12,8))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    ax2 = axes_2.imshow(arr_threshold, extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 10)
    # ax2 = axes_2.imshow(arr_threshold ,
    #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day, fontsize = 20)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Decibel [dB]', size=20)
    axes_2.tick_params(labelsize=18)

    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()

    plt.ylim(Frequency[-1], Frequency[0])
    plt.show()
    plt.close()
    return



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
                    y_over_analysis_time_l.append(int(round(time_band*t_1 + j)))
                    y_over_analysis_data_l.append(y_over_group_l[i])

        if len(y_over_final_l) > 0:
            TYPE3.append(int(round(time_band * t_1 + j)))


    TYPE3_final = np.unique(TYPE3)
    if len(TYPE3_final) > 0:
        TYPE3_group.append(groupSequence(TYPE3_final))


    if len(TYPE3_group)>0:
        arr_shuron = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
        arr_shuron_1 = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
        for m in range (len(TYPE3_group[0])):
            if len(TYPE3_group[0][m]) >= duration:
                if TYPE3_group[0][m][0] >= 0 and TYPE3_group[0][m][-1] <= diff_db_min_med.shape[1]:  
                    arr = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                    for i in range(len(TYPE3_group[0][m])):
    
                        check_start_time_l = ([y for y, Frequency in enumerate(y_over_analysis_time_l) if Frequency == TYPE3_group[0][m][i]])
                        for q in range(len(check_start_time_l)):
                            for p in range(len(y_over_analysis_data_l[check_start_time_l[q]])):
                                arr[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = diff_db_plot_sep[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)]
                                arr_shuron_1[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = 100
                            # print (TYPE3_group[0][m][i] - (time - time_band - time_co))
                    arr_freq_time = np.array(arr)
                    # plot_array_threshold(arr_shuron_1, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)
                    



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
                                        arr_shuron[frequency_separate[0][i][j]][k] = 100
                            arr_sep_freq = np.array(arr_1)
                            # plot_array_threshold(arr_shuron, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)
                            # plot_array_threshold(arr_sep_freq, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)

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
            
            
                                            if 1/p_1[0] < 0:
                                                #drift_allen_model_*1_80_69.5
                                                if 1/p_1[0] > -107.22096882538592:
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
    # print(event_start_list, event_end_list)

    return arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list_1, time_list_1, arr_sep_time_list

import matplotlib.pyplot as plt
import os



from pynverse import inversefunc
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
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start)


def residual_detection(factor_list, freq_list, time_list, Frequency):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    for factor in factor_list:
        slide, time_rate5, residual, h_start = allen_model(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)

        
        cube_4 = (lambda h5_0: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=Frequency)
        x_time.append(invcube_4 + slide)
        y_freq.append(Frequency)

    return residual_list, x_time, y_freq, time_rate_final


import matplotlib.gridspec as gridspec
def plot_data(diff_db_plot_sep, diff_db_sep, freq_list, time_list, arr_5, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(140,50))
    gs = gridspec.GridSpec(140, 50)


    axes_1 = figure_.add_subplot(gs[:55, :20])
#                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
    ax1 = axes_1.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 5)
    axes_1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_1.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
    plt.xlabel('Time (UT)',fontsize=fontsize )
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('Decibel [dB]', size=fontsize)
    axes_1.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()


    axes_1 = figure_.add_subplot(gs[85:, :20])
    ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
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
    figure_.autofmt_xdate()


    axes_2 = figure_.add_subplot(gs[:, 21:34])
    ax2 = axes_2.imshow(arr_5, extent = [0, 50, 30, y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1 -2 ,vmax = vmax_1)
    plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('from Background [dB]', size=fontsize)
    axes_2.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()
    

    axes_2 = figure_.add_subplot(gs[:,38:51])
    
    ######################################################################################


    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=30)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    # cycle = 0
    # for factor in factor_list:
    #     ###########################################
    #     axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 30.0)
    #     cycle += 1
#                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
    plt.xlim(min(time_list) - 10, max(time_list) + 10)
    plt.ylim(min(freq_list), max(freq_list))
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize  + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.legend(fontsize=ticksize - 40)
    figure_.autofmt_xdate()

    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
    #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
    # plt.savefig(filename)
    plt.show()
    plt.close()

    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        ###########################################
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == 2:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
            # plt.xlim(min(time_list) - 10, max(time_list) + 10)
            # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
            plt.xlim(np.median(time_list)-10, np.median(time_list)+40)
            plt.ylim(min(freq_list), max(freq_list))
            plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ 12:00',fontsize=20)
            plt.xlabel('Time[sec]',fontsize=20)
            plt.ylabel('Frequency [MHz]',fontsize=20)
            plt.tick_params(labelsize=18)
            plt.legend(fontsize=18)
            # figure_.autofmt_xdate()
        
            values =np.arange(0,50,5)
            x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
            plt.xticks(x,values)
        cycle += 1
    
    
        # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
        #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
        # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
        # plt.savefig(filename)
    plt.show()
    plt.close()

    return

def plot_data_non_clear(diff_db_plot_sep, diff_db_sep, freq_list, time_list, arr_5, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(140,50))
    gs = gridspec.GridSpec(140, 50)


    axes_1 = figure_.add_subplot(gs[:55, :20])
#                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
    ax1 = axes_1.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 5)
    axes_1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_1.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
    plt.xlabel('Time (UT)',fontsize=fontsize )
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('Decibel [dB]', size=fontsize)
    axes_1.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()


    axes_1 = figure_.add_subplot(gs[85:, :20])
    ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
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
    figure_.autofmt_xdate()


    axes_2 = figure_.add_subplot(gs[:, 21:34])
    ax2 = axes_2.imshow(arr_5, extent = [0, 50, 30, y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1 -2 ,vmax = vmax_1)
    plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('from Background [dB]', size=fontsize)
    axes_2.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()
    

    axes_2 = figure_.add_subplot(gs[:,38:51])
    
    ######################################################################################


    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=30)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    # cycle = 0
    # for factor in factor_list:
    #     ###########################################
    #     axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 30.0)
    #     cycle += 1
#                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
    plt.xlim(min(time_list) - 10, max(time_list) + 10)
    plt.ylim(min(freq_list), max(freq_list))
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize  + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.legend(fontsize=ticksize - 40)
    figure_.autofmt_xdate()

    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
    #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
    # plt.savefig(filename)
    plt.show()
    plt.close()


    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == 2:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
            # plt.xlim(min(time_list) - 10, max(time_list) + 10)
            # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
            # plt.xlim(np.median(time_list)-10, np.median(time_list)+40)
            plt.ylim(min(freq_list), max(freq_list))
            plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ 12:00',fontsize=20)
            plt.xlabel('Time[sec]',fontsize=20)
            plt.ylabel('Frequency [MHz]',fontsize=20)
            plt.tick_params(labelsize=18)
            plt.legend(fontsize=18)
            # figure_.autofmt_xdate()
        
            # values =np.arange(0,50,5)
            # x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
                
            # plt.xticks(x,values)
        cycle += 1
    
    plt.xlim(min(time_list) - 10, max(time_list) + 10)
    plt.show()
    plt.close()

    return


def selected_event_plot(freq_list, time_list, x_time, y_freq, time_rate_final, date_OBs, arr_sep_time, quartile_db_l, db_setting, s_event_time, e_event_time, selected_Frequency, resi_idx, date_event_hour, date_event_minute):

    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    gs = gridspec.GridSpec(140, 50)
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    y_cmap = selected_Frequency
    x_cmap = np.arange(s_event_time, e_event_time + 1, 1)
    cs = axes_2.contourf(x_cmap, y_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == resi_idx+1:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
        cycle += 1
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ '+date_event_hour+':'+date_event_minute,fontsize=20)
    plt.xlabel('Time[sec]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.ylim(selected_Frequency[-1], selected_Frequency[0])
    plt.xlim(s_event_time, e_event_time)
    plt.show()
    plt.close()

    return

def numerical_diff_df_dn(ne):
    h = 1e-5
    f_1 = 9*np.sqrt(ne+h)/1e+3
    f_2 = 9*np.sqrt(ne-h)/1e+3
    return ((f_1 - f_2)/(2*h))

def numerical_diff_allen_dn_dr(factor, r):
    h = 1e-1
    ne_1 = factor * 10**8 * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))
    ne_2 = factor * 10**8 * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))
    return ((ne_1 - ne_2)/(2*h))

def selected_event_plot_2(freq_list, time_list, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, s_event_time, e_event_time, s_event_freq, e_event_freq, selected_Frequency, resi_idx, delete_idx, selected_idx, date_event_hour, date_event_minute):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    gs = gridspec.GridSpec(140, 50)
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(np.array(time_list)[selected_idx], np.array(freq_list)[selected_idx], "wo", label = 'Peak data(selected)', markersize=4)
    axes_2.plot(np.array(time_list)[delete_idx], np.array(freq_list)[delete_idx], "ko", label = 'Peak data(deleted)', markersize=4)
    y_cmap = selected_Frequency
    x_cmap = np.arange(s_event_time, e_event_time + 1, 1)
    cs = axes_2.contourf(x_cmap, y_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == resi_idx+1:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
            # plt.xlim(min(time_list) - 10, max(time_list) + 10)
            # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
            # plt.xlim(np.median(time_list)-10, np.median(time_list)+40)

            # figure_.autofmt_xdate()
        
            # values =np.arange(0,50,5)
            # x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
                
            # plt.xticks(x,values)
        cycle += 1
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ '+date_event_hour+':'+date_event_minute,fontsize=20)
    plt.xlabel('Time[sec]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.ylim(selected_Frequency[-1], selected_Frequency[0])
    plt.xlim(s_event_time, e_event_time)
    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+burst_type+'/'+year):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+burst_type+'/'+year)
    filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+burst_type+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + str(np.min(np.array(time_list)[selected_idx]))+'_'+str(np.max(np.array(time_list)[selected_idx]))+'_'+str(np.max(np.array(freq_list)[selected_idx]))+'_'+str(np.min(np.array(freq_list)[selected_idx]))+ 'peak.png'
    plt.savefig(filename)
    plt.show()
    plt.close()

    return

def choice():
    i = 0
    while i < 1:
        choice = input("Continue drift rate analysis? [y/n/m]: ").lower()
        if choice in ['y']:
            i += 1
            return 'Yes'
        elif choice in ['n']:
            i += 1
            return 'No'
        elif choice in ['m']:
            i += 1
            return 'move'

def fitting_check():
    i = 0
    while i < 1:
        choice = input("How is a fitting? [y/n]: ").lower()
        if choice in ['y']:
            i += 1
            return 'Yes'
        elif choice in ['n']:
            i += 1
            return 'No'



def change_time_range():
    choice = input("Input time range: ").lower()
    return choice

def change_freq_range():
    choice = input("Input freq range: ").lower()
    if type(choice) == list:
        return choice

def change_check_formula():
    choice = input("Input [t, f, b]: ").lower()
    return choice

def change_event_time_range():
    choice = input("Input time range: ").lower()
    time_list = [int(choice.split(',')[0]),int(choice.split(',')[1])]
    return time_list

def change_event_freq_range():
    choice = input("Input freq range: ").lower()
    freq_list = [float(choice.split(',')[0]),float(choice.split(',')[1])]
    return freq_list

def select_time():
    i = 0
    while i < 1:
        choice = input("Select time[t for change time range & f for freq change & p for exit]: ").lower()
        if choice in ['p']:
            i += 1
            return 'exit'
        elif choice in ['t']:
            time_range = change_time_range()
            i += 1
            return str(time_range)
        elif choice in ['f']:
            freq_range = change_freq_range()
            i += 1
            return freq_range
        elif len(str(choice)) == 8:
            i += 1
        elif len(str(choice).split('_')) > 1:
            i += 1
            return str(choice)

def sunspots_num_check(yyyy, mm, dd):
    file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'
    print (file_gain)
    csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
    # print(csv_input['Time_list'])
    for i in range(len(csv_input)):
        BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
        if ((BG_obs_time_event >= datetime.datetime(int(yyyy), int(mm), int(dd))) & (BG_obs_time_event <= datetime.datetime(int(yyyy), int(mm), int(dd)) + datetime.timedelta(days=1))):
            sunspot_num = csv_input['sunspot_number'][i]
            if sunspot_num <= 36:
                return True
            else:
                return False

def wind_geotail_flare(yyyy, mm, dd, WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4):
    if os.path.isfile(Parent_directory + "/solar_burst/wind/plot_QL/" + yyyy + "/wav_summary_" + yyyy + mm + dd + ".png") == True:
        img_wind = cv2.imread(Parent_directory + "/solar_burst/wind/plot_QL/" + yyyy + "/wav_summary_" + yyyy + mm + dd + ".png", cv2.IMREAD_COLOR)
        img_wind_1 = img_wind[50:380, 20:790]
        image_wind = cv2.resize(img_wind_1, dsize=None, fx=1.24*efactor, fy=1.24*efactor)
        cv2.namedWindow(WINDOW_NAME_0, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME_0, image_wind)
        cv2.moveWindow(WINDOW_NAME_0, 1450, 0)
    else:
        print('No data: wind')
    if os.path.isfile(Parent_directory + '/solar_burst/geotail/plot_QL/' + yyyy + '/' +yyyy[2:4] + mm + dd +'00.gif') == True:
        gif_geotail = cv2.VideoCapture(Parent_directory + '/solar_burst/geotail/plot_QL/' + yyyy + '/' +yyyy[2:4] + mm + dd +'00.gif')
        is_success, img_geotail = gif_geotail.read()
        img_geotail_1 = img_geotail[0:400, 40:1105]
        image_geotail = cv2.resize(img_geotail_1, dsize=None, fx=0.9*efactor, fy=0.9*efactor)
        cv2.namedWindow(WINDOW_NAME_1, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME_1, image_geotail)
        cv2.moveWindow(WINDOW_NAME_1, 1450, 250)
    else:
        print('No data: Geotail')
    # if len(glob.glob(Parent_directory + '/hinode_catalog/flare_plot/' + yyyy+mm+dd +'.png')) == 1:
    #     img_flare = cv2.imread(Parent_directory + '/hinode_catalog/flare_plot/' + yyyy+mm+dd +'.png', cv2.IMREAD_COLOR)
    #     # img_flare_1 = img_flare[30:280, 30:700]
    #     # image_flare = cv2.resize(img_flare_1, dsize=None, fx=0.95, fy=0.95)
    #     # cv2.namedWindow(WINDOW_NAME_4, cv2.WINDOW_AUTOSIZE)
    #     # cv2.imshow(WINDOW_NAME_4, image_flare)
    #     # cv2.moveWindow(WINDOW_NAME_4, 0, 503)
    #     img_flare_1 = img_flare[30:280, 40:700]
    #     image_flare = cv2.resize(img_flare_1, dsize=None, fx=1.455*efactor, fy=1.455*efactor)
    #     cv2.namedWindow(WINDOW_NAME_4, cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow(WINDOW_NAME_4, image_flare)
    #     # cv2.moveWindow(WINDOW_NAME_4, 1450, 451)
    #     cv2.moveWindow(WINDOW_NAME_4, 1450, 825)
    #     # cv2.moveWindow(WINDOW_NAME_4, 1450, 825*factor)
    # else:
    #     print('No data: Flare')
    cv2.waitKey(1)
    return


WINDOW_NAME_0 = "wind"
WINDOW_NAME_1 = "geotail"
WINDOW_NAME_2 = "detected type 3 burst  "
WINDOW_NAME_3 = "Nancay QL"
WINDOW_NAME_4 = "Flare"
# WINDOW_NAME_5 = 'SDO_HMIB'
WINDOW_NAME_6 = 'Nancay Wind'
WINDOW_NAME_7  = 'SDO_0193'
WINDOW_NAME_8 = 'SDO_0211'

sigma_value = 2
after_plot = str('nonclearevent_analysis')
burst_type = 'burst_type'
time_band = 340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
threshold_frequency_final = 10.5
cnn_plot_time = 50
save_place = 'cnn_used_data/shuron_plot_random'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 100
db_setting = 40
sun_to_earth = 150000000
sun_radius = 696000
light_v = 300000 #[km/s]
time_rate = 0.13
import csv
import pandas as pd
import shutil
import os
import sys
import cv2
efactor = 0.5


def flare_check(yyyy):
    file_final = "/hinode_catalog/Hinode Flare Catalogue new.csv"
    flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    
    flare_times = []
    for i in range (len(flare_csv['peak'])):
        year = flare_csv['peak'][i].split('/')[0]
        if int(year) != int(yyyy):
            pass
        else:
            peak_time = datetime.datetime(int(flare_csv['peak'][i].split('/')[0]), int(flare_csv['peak'][i].split('/')[1]), int(flare_csv['peak'][i].split('/')[2].split(' ')[0]), int(flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0]), int(flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1]))
            # pd.to_datetime(flare_csv['peak'][i].split('/')[0] + flare_csv['peak'][i].split('/')[1] + flare_csv['peak'][i].split('/')[2].split(' ')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
            if (peak_time.hour >= 8) and (peak_time.hour <= 16):
                # pd_start_time = pd.to_datetime(flare_csv['start'][i].split('/')[0] + flare_csv['start'][i].split('/')[1] + flare_csv['start'][i].split('/')[2].split(' ')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                # pd_end_time = pd.to_datetime(flare_csv['end'][i].split('/')[0] + flare_csv['end'][i].split('/')[1] + flare_csv['end'][i].split('/')[2].split(' ')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                flare_times.append(peak_time)
    flare_times = np.array(flare_times)
    return flare_times


cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

#20140708
#黒点36以下

    
t = 0
selected_time = '10:55:30'
selected_time_range = 30
freq_list_setting = [80,30]
little_move = 0
while t == 0:
    cv2.waitKey(1)
    cv2.destroyWindow(WINDOW_NAME_6)
    cv2.waitKey(1)
    select = select_time()
    if select == 'exit':
        t += 1
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        sys.exit()
    elif len(select) == 8:
        selected_time = select
        print (selected_time)
    elif len(str(select).split('_')) > 1:
        filename = str(select)
        selected_date = str(select).split('_')[0]
        selected_stime = datetime.datetime.strptime(str(select).split('_')[0]+str(select).split('_')[1], '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(str(select).split('_')[5]))
        selected_time  = selected_stime.strftime("%H:%M:%S")
        print (selected_time)
        selected_idx_2 = np.arange(int(str(select).split('_')[3]), int(str(select).split('_')[4]), 1)
        s_event_time, e_event_time = [int(str(select).split('_')[5]),int(str(select).split('_')[6])]
        selected_time_range = int(str(select).split('_')[6]) - int(str(select).split('_')[5]) + 1

        if (int(str(select).split('_')[3])/time_band).is_integer() is True:
            t_1 = int(int(str(select).split('_')[3])/time_band)
        else:
            # t_1 = math.floor((diff_db_min_med.shape[1]-time_co)/time_band)
            t_1 = 'No'
    else:
        selected_time_range = int(select)



    yyyy = selected_date[:4]
    mm = selected_date[4:6]
    dd = selected_date[6:8]
    
    wind_geotail_flare(yyyy, mm, dd, WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4)
    nancay_wind_QL_list = sorted(glob.glob(Parent_directory + '/solar_burst/Nancaywind2/' + yyyy + '/' + mm + '/' + selected_date + '*'))
    
    if sunspots_num_check(yyyy, mm, dd) == False:
        print ('Sunspots error')
    else:
        print ('Over 36 sunspots')
        flare_times = flare_check(yyyy)
        flare_idx = np.where((flare_times >= datetime.datetime(int(yyyy),int(mm), int(dd),8)) & (flare_times <= datetime.datetime(int(yyyy),int(mm), int(dd),17)))[0]
        
        print ('_______________________\n')
        print ('Flare time')
        
        for flare_time in np.flip(flare_times[flare_idx]):
            
            print ((flare_time + dt.timedelta(minutes = -10)).strftime("%H:%M"), ' - ', (flare_time + dt.timedelta(minutes = 10)).strftime("%H:%M"))
        print ('_______________________\n')
            
            
            
            
        file_names = glob.glob(Parent_directory + '/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ selected_date +'*cdf')
        for file_name in file_names:
            file_name = file_name.split('/')[10]
        
            if int(yyyy) <= 1997:
                diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, obs_time_list, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data(Parent_directory, file_name, 3, 70, 30)
            else:
                diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, obs_time_list, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data(Parent_directory, file_name, 3, 80, 30)
            print (Frequency_start, Frequency_end)
            if t_1 == 'No':
                t_1 = math.floor((diff_db_min_med.shape[1]-time_co)/time_band)



            # if (os.path.isfile(Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/micro/done/'+filename) and os.path.isfile(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/micro/' + filename.split('.png')[0]+'.csv')) == True:
            #     print ('Analyzed data')
            # else:
            int_start_time = int(selected_time[:2]+selected_time[3:5]+selected_time[6:8])
            Nancay_wind_QL = []
            for nancay_wind_QL in nancay_wind_QL_list:
                if int(nancay_wind_QL.split('.')[0].split('_')[2]) <= int_start_time:
                    if int(nancay_wind_QL.split('.')[0].split('_')[3]) >= int_start_time:
                        Nancay_wind_QL.append(nancay_wind_QL)
            if len(Nancay_wind_QL) == 0:
                print('Find error: No Nancay and Wind QL is found   filename_' + filename)
            else:
                if len(Nancay_wind_QL) == 1:
                    plot_nancay_wind_QL = Nancay_wind_QL[0]
                else:
                    stime_list = []
                    for NWQL in Nancay_wind_QL:
                        stime_list.append(int(NWQL.split('.')[0].split('_')[2]))
                    plot_nancay_wind_QL = glob.glob(Parent_directory + '/solar_burst/Nancaywind2/' + yyyy + '/' + mm + '/' + selected_date +  '_*' + str(min(stime_list)) +'_*')[0]
                    # plot_nancay_wind_QL = glob.glob(Parent_directory + '/solar_burst/Nancaywind/' + yyyy + '/' + mm + '/' + str_date +  '_*' + str(min(stime_list)) +'_*')[0]
                img_nancay_wind_QL = cv2.imread(plot_nancay_wind_QL, cv2.IMREAD_COLOR)
                img_nancay_wind_QL_1 = img_nancay_wind_QL[100:800, 50:800]
                image_nancay_wind_QL = cv2.resize(img_nancay_wind_QL_1, dsize=None, fx=1.75*efactor, fy=1.75*efactor)
                # img_nancay_QL_1 = img_nancay_QL[0:400, 120:1800]
                # image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.72, fy=0.72)
                cv2.namedWindow(WINDOW_NAME_6, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(WINDOW_NAME_6, image_nancay_wind_QL)
                # cv2.moveWindow(WINDOW_NAME_6, 1928, 0)
                cv2.moveWindow(WINDOW_NAME_6, 2409, 0)
                cv2.waitKey(1)




                stime = dt.datetime.strptime(selected_date + ' ' + selected_time, '%Y%m%d %H:%M:%S')+ dt.timedelta(seconds = little_move)
                etime = dt.datetime.strptime(selected_date + ' ' + selected_time, '%Y%m%d %H:%M:%S') + dt.timedelta(seconds = selected_time_range)+ dt.timedelta(seconds = little_move)
                midtime = dt.datetime.strptime(selected_date + ' ' + selected_time, '%Y%m%d %H:%M:%S') + dt.timedelta(seconds = selected_time_range/2)+ dt.timedelta(seconds = little_move)
                if len(np.where((flare_times[flare_idx] <= midtime + dt.timedelta(minutes = 10)) & (flare_times[flare_idx] >= midtime - dt.timedelta(minutes = 10)))[0]) == 0:
                    print ('There is no flares')
                    print (flare_times[flare_idx][np.where((flare_times[flare_idx] <= midtime + dt.timedelta(minutes = 10)) & (flare_times[flare_idx] >= midtime - dt.timedelta(minutes = 10)))[0]])
                else:
                    selected_idx = np.where((obs_time_list >= stime) & (obs_time_list <= etime))[0]
                    if len(selected_idx) < selected_time_range:
                        print ('Selected range error')
                        sys.exit()

                    # selected_idx_2 = np.arange(center-200, center+200, 1)
                    diff_db_plot_sep = diff_db_min_med[:, selected_idx_2]
                    diff_db_sep = diff_db[:, selected_idx_2]
                    arr_threshold, mean_l_list, quartile_db_l, quartile_power, diff_power_last_l, stdev_sub = threshold_array(diff_db_plot_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration)
                    time = int(round(time_band*(t_1+1) + time_co))
                    start = obs_time_list[selected_idx_2][0]
                    end = obs_time_list[selected_idx_2][-1]
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
            
            
                    aaa = np.where(arr_threshold > 0, 100, 0)
                    plot_array_threshold(aaa, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)
            
            
                    arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list, time_list, arr_sep_time_list = sep_array(arr_threshold, diff_db_plot_sep, Frequency, time_band, time_co,  quartile_power, time, duration, resolution, Status, cnn_plot_time, t_1)
                    plot_array_threshold_2(diff_db_sep, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, min_db, quartile_db_l)
                    if len(arr_5_list) == 0:
                        print ('No arr_5_list')
                        sys.exit()
                        pass
                    else:
                        print ('Select' + str(s_event_time)+' - ' + str(e_event_time))
                        print (event_start_list, event_end_list)
                        for i in range(len(arr_5_list)):
                            check_arr_time = np.arange(int(event_start_list[i]), int(event_end_list[i])+1,1)
                            if len(np.where((check_arr_time >= s_event_time) & (check_arr_time <= e_event_time))[0])>0:
                                # if len(freq_list) == 0:
                                #     s_event_freq, e_event_freq = [int(np.max(Frequency)), int(np.min(Frequency))]
                                # else:
                                s_event_freq, e_event_freq = freq_list_setting
                                freq_start_idx = np.where(Frequency == getNearestValue(Frequency, s_event_freq))[0][0]
                                freq_end_idx = np.where(Frequency == getNearestValue(Frequency, e_event_freq))[0][0]
                                sep_arr_sep_time_list =  arr_sep_time_list[i][freq_start_idx:freq_end_idx + 1, selected_idx - selected_idx_2[0]]
                                selected_Frequency = Frequency[freq_start_idx:freq_end_idx + 1]
                                
                                
                                z = 0
                                while z >= 0:
                                    if z == 0:
                                        freq_list_new = []
                                        time_list_new = []
                                        for k in range(sep_arr_sep_time_list.shape[0]):
                                            if max(sep_arr_sep_time_list[k]) > -10:
                                                if (len([l for l in sep_arr_sep_time_list[k] if l == max(sep_arr_sep_time_list[k])])) == 1:
                                                    freq_list_new.append(selected_Frequency[k])
                                                    time_list_new.append(np.argmax(sep_arr_sep_time_list[k]) + (selected_idx - selected_idx_2[0])[0])
                                        time_event = dt.timedelta(seconds=(int(event_end_list[i]) + int(event_start_list[i]))/2) + dt.datetime(int(date_OBs[0:4]), int(date_OBs[4:6]), int(date_OBs[6:8]),int(Time_start[0:2]), int(Time_start[3:5]), int(Time_start[6:8]))
                                        date_event = str(time_event.date())[0:4] + str(time_event.date())[5:7] + str(time_event.date())[8:10]
                                        date_event_hour = str(time_event.hour)
                                        date_event_minute = str(time_event.minute)
                                        residual_list, x_time, y_freq, time_rate_final = residual_detection(factor_list, freq_list_new, time_list_new, freq_list_new)
                                        resi_idx = np.argmin(residual_list)
                                        s_event_time, e_event_time = [(selected_idx - selected_idx_2[0])[0], (selected_idx - selected_idx_2[0])[-1]]
                                        selected_event_plot(freq_list_new, time_list_new, x_time, y_freq, time_rate_final, date_OBs, sep_arr_sep_time_list, quartile_db_l, db_setting, s_event_time, e_event_time, selected_Frequency, resi_idx, date_event_hour, date_event_minute)
                                        time_gap_arr = x_time[resi_idx][np.where(y_freq[resi_idx] == freq_list_new[0])[0][0]:np.where(y_freq[resi_idx] == freq_list_new[-1])[0][0] + 1] - np.array(time_list_new)
                                        delete_idx = np.where(np.abs(time_gap_arr) >= 2 * residual_list[resi_idx])[0]
                                        selected_idx_3 = np.where(np.abs(time_gap_arr) < 2 * residual_list[resi_idx])[0]
                                        if len(np.array(freq_list_new)[selected_idx_3]) == 0:
                                            print ('freq_time error')
                                        else:
                                            residual_list, x_time, y_freq, time_rate_final = residual_detection(factor_list, np.array(freq_list_new)[selected_idx_3], np.array(time_list_new)[selected_idx_3], freq_list_new)
                                            resi_idx = np.argmin(residual_list)
                                            selected_event_plot_2(freq_list_new, time_list_new, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], sep_arr_sep_time_list, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, s_event_time, e_event_time, s_event_freq, e_event_freq, selected_Frequency, resi_idx, delete_idx, selected_idx_3, date_event_hour, date_event_minute)
                                    elif z >= 1:
                                        change_check = change_check_formula()
                                        if change_check == 't':
                                            time_list_setting = change_event_time_range()
                                            s_event_start, e_event_end = np.array(time_list_setting) + int(str(select).split('_')[3])
                                            selected_idx = np.arange(s_event_start, e_event_end+1, 1)
                                        elif change_check == 'f':
                                            freq_list_setting = change_event_freq_range()
                                            s_event_freq, e_event_freq = np.array(freq_list_setting)
                                        elif change_check == 'b':
                                            time_list_setting = change_event_time_range()
                                            freq_list_setting = change_event_freq_range()
                                            s_event_freq, e_event_freq = np.array(freq_list_setting)
                                            s_event_start, e_event_end = np.array(time_list_setting) + int(str(select).split('_')[3])
                                            selected_idx = np.arange(s_event_start, e_event_end+1, 1)
                                        freq_start_idx = np.where(Frequency == getNearestValue(Frequency, s_event_freq))[0][0]
                                        freq_end_idx = np.where(Frequency == getNearestValue(Frequency, e_event_freq))[0][0]
                                        sep_arr_sep_time_list =  arr_sep_time_list[i][freq_start_idx:freq_end_idx + 1, selected_idx - selected_idx_2[0]]
                                        selected_Frequency = Frequency[freq_start_idx:freq_end_idx + 1]
                                        freq_list_new = []
                                        time_list_new = []
                                        for k in range(sep_arr_sep_time_list.shape[0]):
                                            if max(sep_arr_sep_time_list[k]) > -10:
                                                if (len([l for l in sep_arr_sep_time_list[k] if l == max(sep_arr_sep_time_list[k])])) == 1:
                                                    freq_list_new.append(selected_Frequency[k])
                                                    time_list_new.append(np.argmax(sep_arr_sep_time_list[k]) + (selected_idx - selected_idx_2[0])[0])
                                        time_event = dt.timedelta(seconds=(int(event_end_list[i]) + int(event_start_list[i]))/2) + dt.datetime(int(date_OBs[0:4]), int(date_OBs[4:6]), int(date_OBs[6:8]),int(Time_start[0:2]), int(Time_start[3:5]), int(Time_start[6:8]))
                                        date_event = str(time_event.date())[0:4] + str(time_event.date())[5:7] + str(time_event.date())[8:10]
                                        date_event_hour = str(time_event.hour)
                                        date_event_minute = str(time_event.minute)
                                        residual_list, x_time, y_freq, time_rate_final = residual_detection(factor_list, freq_list_new, time_list_new, freq_list_new)
                                        resi_idx = np.argmin(residual_list)
                                        s_event_time, e_event_time = [(selected_idx - selected_idx_2[0])[0], (selected_idx - selected_idx_2[0])[-1]]
                                        selected_event_plot(freq_list_new, time_list_new, x_time, y_freq, time_rate_final, date_OBs, sep_arr_sep_time_list, quartile_db_l, db_setting, s_event_time, e_event_time, selected_Frequency, resi_idx, date_event_hour, date_event_minute)
                                        time_gap_arr = x_time[resi_idx][np.where(y_freq[resi_idx] == freq_list_new[0])[0][0]:np.where(y_freq[resi_idx] == freq_list_new[-1])[0][0] + 1] - np.array(time_list_new)
                                        delete_idx = np.where(np.abs(time_gap_arr) >= 2 * residual_list[resi_idx])[0]
                                        selected_idx_3 = np.where(np.abs(time_gap_arr) < 2 * residual_list[resi_idx])[0]
                                        residual_list, x_time, y_freq, time_rate_final = residual_detection(factor_list, np.array(freq_list_new)[selected_idx_3], np.array(time_list_new)[selected_idx_3], freq_list_new)
                                        resi_idx = np.argmin(residual_list)
                                        selected_event_plot_2(freq_list_new, time_list_new, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], sep_arr_sep_time_list, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, s_event_time, e_event_time, s_event_freq, e_event_freq, selected_Frequency, resi_idx, delete_idx, selected_idx_3, date_event_hour, date_event_minute)
                                    if z >= 0:
                                        fitting = fitting_check()
                                        if fitting == 'Yes':
                                            z -= 100
                                            if np.min(residual_list) > 1.35:
                                                print ('Residual error')
                                                file_dir = Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/flare_associated_ordinary/moved'
                                                if not os.path.isfile(file_dir+'/'+filename):
                                                    shutil.move(Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/flare_associated_ordinary/'+yyyy+'/'+filename, file_dir)
                                            else:
                                                yes_or_not = choice()
                                                if yes_or_not == 'Yes':
                                                    csvfiles = Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/ordinary/'+filename.split('.png')[0]+'*.csv'
                                                    csvfiles_len = glob.glob(csvfiles)
                                                    if csvfiles_len == 0:
                                                        filename_2 = filename.split('.png')[0]+'.csv'
                                                    else:
                                                        filename_2 = filename.split('.png')[0]+str(len(csvfiles_len))+'.csv'
                                                    
                                                    with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/ordinary/' + filename_2, 'w') as f:
                                                        obs_time = obs_time_list[selected_idx_2[0]] + datetime.timedelta(seconds = int(np.min(np.array(time_list_new)[selected_idx_3])))
                                                        w = csv.DictWriter(f, fieldnames=["obs_time", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list", "drift_rate_40MHz"])
                                                        w.writeheader()
                                                        factor = resi_idx+1
                                                        time_rate = time_rate_final[resi_idx]
                                                        cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
                                                        r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                                                        r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                                                        ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
                                                        # print ('\n'+str(factor)+'×B-A model' + 'emission fp')
                                                        drift_rates = (-1)*numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
                                                        w.writerow({'obs_time':obs_time,'velocity':time_rate_final, 'residual':residual_list, 'event_start': np.min(np.array(time_list_new)[selected_idx_3]),'event_end': np.max(np.array(time_list_new)[selected_idx_3]),'freq_start': np.max(np.array(freq_list_new)[selected_idx_3]),'freq_end':np.min(np.array(freq_list_new)[selected_idx_3]), 'factor':resi_idx+1, 'peak_time_list':np.array(time_list_new)[selected_idx_3], 'peak_freq_list':np.array(freq_list_new)[selected_idx_3], 'drift_rate_40MHz':drift_rates})
            
                                                        file_dir = Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/flare_associated_ordinary/done'
                                                        if not os.path.isfile(file_dir+'/'+filename):
                                                            shutil.copy(Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/flare_associated_ordinary/'+yyyy+'/'+filename, file_dir)
                                                elif yes_or_not == 'move':
                                                    file_dir = Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/flare_associated_ordinary/moved'
                                                    if not os.path.isfile(file_dir+'/'+filename):
                                                        shutil.move(Parent_directory + '/solar_burst/Nancay/plot/afjpgunonsimpleselect/flare_associated_ordinary/'+yyyy+'/'+filename, file_dir)
                                        elif fitting == 'No':
                                            z += 1


