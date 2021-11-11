#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:48:59 2021

@author: yuichiro
"""

import glob
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


def plot_array_threshold(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    figure_=plt.figure(1,figsize=(6,4))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    ax2 = axes_2.imshow(arr_threshold, extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= 0 ,vmax = 1)
    # ax2 = axes_2.imshow(arr_threshold ,
    #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day)
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
import random
from keras.preprocessing.image import load_img, img_to_array
import shutil
def cnn_detection(arr_5, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, date_OBs, Time_start, Time_end, color_setting, image_size, img_rows, img_cols, model, save_place, Frequency, x_lims):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    y_lims = [Frequency[-1], Frequency[0]]
    plt.close(1)

    figure_=plt.figure(1,figsize=(10,10))
    axes_2 = figure_.add_subplot(1, 1, 1)
    axes_2.imshow(arr_5, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1-2 ,vmax = vmax_1)
    plt.axis('off')
    
    extent = axes_2.get_window_extent().transformed(figure_.dpi_scale_trans.inverted())
    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all'):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all')
    filename = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
    if os.path.isfile(filename):
      os.remove(filename)
    plt.savefig(filename, bbox_inches=extent)
    # plt.show()
    plt.close()

    files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
    print('--- 読み込んだデータセットは', Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', 'です。')

    for i, file in enumerate(random.sample(files,len(files))):  
      if color_setting == 1:
        img = load_img(file, color_mode = 'grayscale' ,target_size=(image_size, image_size))  
      elif color_setting == 3:
        img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
      array = img_to_array(img)
      x_test = np.array(array)

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(1, 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(1, img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    predict = model.predict(x_test)
    print ('flare: ' + str(predict[0][0]) + ' , others: '+ str(predict[0][1]))
    if predict[0][0] >= 0.5:
      save_directory = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/flare'
      filename2 = save_directory + '/simple/' +year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
      if os.path.isfile(filename2):
        os.remove(filename2)
      shutil.move(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory + '/simple/')
    else:
      save_directory = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/others'
      filename2 = save_directory + '/simple/' +year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
      if os.path.isfile(filename2):
        os.remove(filename2)
      shutil.move(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory + '/simple/')

    return save_directory

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


def residual_detection(Parent_directory, save_directory, factor_list, freq_list, time_list, save_place, residual_threshold, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
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
        h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        h5_0 = h5_0/1000
        x_time.append(h5_0)
        y_freq.append(9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))
    if min(residual_list) <= residual_threshold:
        save_directory_1 = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/flare_clear'
        filename2 = save_directory_1 + '/simple/' +year + month + day +'_' + Time_start[0:2] + Time_start[3:5] + Time_start[6:8] + '_' + Time_end[0:2] + Time_end[3:5] + Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
        if os.path.isfile(filename2):
          os.remove(filename2)
        shutil.move(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/flare/simple/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory_1 + '/simple/')
    else:
        save_directory_1 = save_directory
    return residual_list, save_directory_1, x_time, y_freq, time_rate_final


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
    figure_.autofmt_xdate()

    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
    plt.savefig(filename)
    plt.show()
    plt.close()
    return filename

def read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end):
    file_name_separate =file_name.split('_')
    Date_start = file_name_separate[5]
    date_OBs=str(Date_start)
    year=date_OBs[0:4]
    month=date_OBs[4:6]

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
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
    # min_power = np.amin(y_power, axis=1)
    # min_db_LL = np.amin(y_power, axis=1)
    # min_db = np.log10(min_power) * 10
    # min_db_LL = np.amin(diff_r_last, axis=1)
    # min_db_RR = np.amin(diff_l_last, axis=1)
    # diff_db_min_med = (diff_move_db.T - min_db).T
    # LL_min = (diff_l_last.T - min_db_LL).T
    # RR_min = (diff_r_last.T - min_db_RR).T
    return diff_l_last, diff_r_last, diff_move_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)

def separated_data_LL_RR(LL, RR, diff_move_db, epoch, time_co, time_band, t, Status):
    if t == math.floor((diff_move_db.shape[1]-time_co)/time_band):
        t = (diff_move_db.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    time = round(time_band*t)
    print (time)
    if time < 0:
        time = 0
    #+1 is due to move_average
    start = epoch[time + 1]
    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time = round(time_band*(t+1) + time_co)
    if time > len(epoch):
        time = len(epoch) - 1
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



    diff_db_plot_sep = diff_move_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    LL_sep = LL[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    RR_sep = RR[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    Status_sep = Status[time - time_band - time_co:time]
    return LL_sep, RR_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t, Status_sep

class read_waves:
    
    def __init__(self, **kwargs):
        self.date_in  = kwargs['date_in']
        self.HH = kwargs['HH']
        self.MM = kwargs['MM']
        self.SS = kwargs['SS']
        self.duration = kwargs['duration']
        self.directry = kwargs['directry']
        
        self.str_date = str(self.date_in)
        self.time_axis = pd.date_range(start=self.str_date, periods=1440, freq='T')
        self.yyyy = self.str_date[0:4]
        self.mm   = self.str_date[4:6]
        self.dd   = self.str_date[6:8]
        
        start = pd.to_datetime(self.str_date + self.HH + self.MM + self.SS,
                               format='%Y%m%d%H%M%S')
        
        end = start + pd.to_timedelta(self.duration, unit='min')

        self.time_range = [start, end]
        
        return
    
    def tlimit(self, df):
        if type(df) != pd.core.frame.DataFrame:
            print('tlimit \n Type error: data type must be DataFrame')
            sys.exit()
        
        

        tl_df =    df[   df.index >= self.time_range[0]]
        tl_df = tl_df[tl_df.index  <= self.time_range[1]]
        
        return tl_df
    
    
    def read_rad(self, receiver):
        if type(receiver) != str:
            print('read_rad \n Keyword error: reciever must be a string')
            sys.exit()
        
        if receiver == 'rad1':
            extension = '.R1'
        elif receiver == 'rad2':
            extension = '.R2'
        else:
            print('read_rad \n Name error: receiver name')
            sys.exit()
        file_path = self.directry + self.str_date + extension
        sav_data = sio.readsav(file_path)
        data = sav_data['arrayb'][:, 0:1440]
        BG   = sav_data['arrayb'][:, 1440]
        
        rad_data = np.where(data==0, np.nan, data)
        rad_data = rad_data.T
        rad = pd.DataFrame(rad_data)
        
        rad.index = self.time_axis
        rad = self.tlimit(rad)
        return rad, BG
    
    def read_waves(self):
        rad1 = self.read_rad('rad1')
        rad2 = self.read_rad('rad2')
        waves = pd.concat([rad1, rad2], axis=1)
        
        return waves



def waves_peak_finder(data):
    if type(data) != pd.core.frame.DataFrame:
        print('waves_peak_finder \n Type error: data type must be DataFrame')
        sys.exit()
    data = data.reset_index(drop=True)
    peak = data.max(axis=0)
    idx  = data.idxmax(axis=0)
    result = pd.concat([idx, peak], axis=1)
    result.columns = ['index', 'peak']
    return result

def freq_setting(receiver):
    if receiver == 'rad1':
        freq = 0.02 + 0.004*np.arange(256)
    elif receiver == 'rad2':
        freq = 1.075 + 0.05*np.arange(256)
    elif receiver == 'waves':
        freq1 = 0.02 + 0.004*np.arange(256)
        freq2 = 1.075 + 0.05*np.arange(256)
        freq  = np.hstack([freq1, freq2])
    else:
        print('freq_setting \n Name error: receiver name')
    return freq

def linear_fit(data, receiver='rad1', freq_band=[0.02, 1.04],
               p0=[0,0], bounds = ([-np.inf, -np.inf], [np.inf, np.inf])):
    
    def linear_func(x, a, b):
        return a*x + b
    
    peak_data = waves_peak_finder(data)
    index = peak_data['index']
    peak  = peak_data['peak']
    
    freq = freq_setting(receiver)
    freq = pd.DataFrame(freq)
    
    cat_data = pd.concat([peak_data,freq], axis=1)
    cat_data.columns = ['index', 'peak', 'freq']
    
    flimit_df =  cat_data[ cat_data['freq'] >= freq_band[0]]
    flimit_df = flimit_df[flimit_df['freq'] <= freq_band[1]]
    
    l_index = flimit_df['index'].values
    l_peak  = flimit_df['peak'].values
    l_freq = flimit_df['freq'].values
    
    if len(l_index) == 0:
        print('linear_fit \n Value error: freq_band range are illegal values for fitting')
        sys.exit()
    
    x = []
    y = []
    
    for i in range(len(l_index)):
        if np.isnan(l_peak[i]) != True:
            x.append(l_index[i])
            y.append(l_freq[i])
    
    popt, pcov = curve_fit(linear_func,x, y, p0=p0, bounds=bounds)
    error = np.sqrt(np.diag(pcov))
    
    plt.figure()
    plt.plot(cat_data['index'], cat_data['freq'], 'ro')
    plt.plot(index, linear_func(index, popt[0], popt[1]))
    plt.axhline(freq_band[0], xmin=0, xmax=1, color='blue', linestyle='dashed')
    plt.axhline(freq_band[1], xmin=0, xmax=1, color='blue', linestyle='dashed')
    plt.ylim(freq.iloc[0][0], freq.iloc[-1][0])
    plt.xlabel('Time [min]')
    plt.ylabel('Frequency [MHz]')
    return popt, error
def radio_plot(data_1, receiver_1, BG_1, data_2, receiver_2, BG_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs):
    # freq_list = [Frequency_start, Frequency_end]
    NDA_freq = Frequency_start/Frequency_end
    p_data_2 = 20*np.log10(data_2)
    vmin_2 = 0
    vmax_2 = 2.5
    
    freq_axis = freq_setting(receiver_2)
    y_lim_2 = [freq_axis[0], freq_axis[-1]]
    rad_2_freq = freq_axis[-1]/freq_axis[0]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_2.index
    x_lim_2 = [time_axis[0], time_axis[-1]]
    x_lim_2 = mdates.date2num(x_lim_2)

    p_data_1 = 20*np.log10(data_1)
    vmin_1 = 0
    vmax_1 = 8
    
    freq_axis = freq_setting(receiver_1)
    y_lim_1 = [freq_axis[0], freq_axis[-1]]
    rad_1_freq = freq_axis[-1]/freq_axis[1]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_1.index
    x_lim_1 = [time_axis[0], time_axis[-1]]
    x_lim_1 = mdates.date2num(x_lim_1)


    plt.close()
    year=date_OBs[0:4]
    month=date_OBs[4:6]
    day=date_OBs[6:8]
    if type(data_1) != pd.core.frame.DataFrame:
        print('radio_plot \n Type error: data type must be DataFrame')
        sys.exit()

    NDA_gs = int(round((NDA_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_2_gs = int(round((rad_2_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_1_gs = int(round((rad_1_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    fig = plt.figure(figsize=(12.0, 12.0))
    gs = gridspec.GridSpec(131, 1)
    ax = plt.subplot(gs[0:NDA_gs, :])
    ax1 = plt.subplot(gs[NDA_gs+1:NDA_gs+rad_2_gs+1, :])
    ax2 = plt.subplot(gs[NDA_gs+rad_2_gs+1:NDA_gs+rad_2_gs+rad_1_gs+1, :])
    ax3 = plt.subplot(gs[108:118, :])
    ax4 = plt.subplot(gs[118:128, :])
    # fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12.0, 12.0))
    plt.subplots_adjust(hspace=0.001)

    y_lims = [Frequency_end, Frequency_start]
    ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=5,vmax=30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(labelsize=10)
    # ax.set_xlabel('Time [UT]')
    # ax.set_ylabel('Frequency [MHz]')
    ax.set_yscale('log')

    

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')
    

    ax2.imshow(p_data_1.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_1[0],x_lim_1[1],y_lim_1[0],y_lim_1[1]],
                vmin=vmin_1, vmax=vmax_1)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(labelsize=10)
    ax2.set_xlabel('Time [UT]')
    # ax2.set_ylabel('Frequency [MHz]')
    ax2.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False) 
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False) 
    ax2.xaxis.tick_bottom()
    ax2.set_yscale('log')

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')

    ax3.plot(np.arange(0,len(p_data_2.T.values.tolist()[247]),1),p_data_2.T.values.tolist()[247], '-', label = '13.425MHz')
    maxid = signal.argrelmax(np.array(p_data_2.T.values.tolist()[247]), order=width)
    ax3.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], np.array(p_data_2.T.values.tolist()[247])[maxid[0]])
    ax3.set_xlim(0,len(p_data_2.T.values.tolist()[247])-1)
    ax3.legend()
    ax3.spines['bottom'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop=False) 
    ax4.plot(np.arange(0,len(p_data_1.T.values.tolist()[224]),1),p_data_1.T.values.tolist()[224], '-', label = '916kHz')
    maxid = signal.argrelmax(np.array(p_data_1.T.values.tolist()[224]), order=width)
    ax4.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], np.array(p_data_1.T.values.tolist()[224])[maxid[0]])
    ax4.set_xlim(0,len(p_data_1.T.values.tolist()[224])-1)
    ax4.legend()
    # ax4.spines['bottom'].set_visible(False)
    ax4.xaxis.tick_top()
    ax4.tick_params(labeltop=False) 


    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancaywind_4/'+year + '/' + month):
        # os.makedirs(Parent_directory + '/solar_burst/Nancaywind_4/'+year + '/' + month)
    # filename = Parent_directory + '/solar_burst/Nancaywind_4/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    # plt.savefig(filename)
    plt.show()
    return


def radio_plot_1(data_1, receiver_1, BG_1, data_2, receiver_2, BG_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs):
    # freq_list = [Frequency_start, Frequency_end]
    NDA_freq = Frequency_start/Frequency_end
    p_data_2 = 20*np.log10(data_2)
    vmin_2 = 0
    # vmax_2 = 2.5
    vmax_2 = 7

    freq_axis = freq_setting(receiver_2)
    y_lim_2 = [freq_axis[0], freq_axis[-1]]
    rad_2_freq = freq_axis[-1]/freq_axis[0]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_2.index
    x_lim_2 = [time_axis[0], time_axis[-1]]
    x_lim_2 = mdates.date2num(x_lim_2)

    p_data_1 = 20*np.log10(data_1)
    vmin_1 = 0
    # vmax_1 = 8
    vmax_1 = 20
    
    freq_axis = freq_setting(receiver_1)
    y_lim_1 = [freq_axis[0], freq_axis[-1]]
    rad_1_freq = freq_axis[-1]/freq_axis[1]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_1.index
    x_lim_1 = [time_axis[0], time_axis[-1]]
    x_lim_1 = mdates.date2num(x_lim_1)


    plt.close()
    year=date_OBs[0:4]
    month=date_OBs[4:6]
    day=date_OBs[6:8]
    if type(data_1) != pd.core.frame.DataFrame:
        print('radio_plot \n Type error: data type must be DataFrame')
        sys.exit()

    NDA_gs = int(round((NDA_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_2_gs = int(round((rad_2_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_1_gs = int(round((rad_1_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    fig = plt.figure(figsize=(18.0, 11.0))
    gs = gridspec.GridSpec(151, 1)
    ax = plt.subplot(gs[0:NDA_gs, :])
    ax1 = plt.subplot(gs[NDA_gs+1:NDA_gs+rad_2_gs+1, :])
    ax2 = plt.subplot(gs[NDA_gs+rad_2_gs+1:NDA_gs+rad_2_gs+rad_1_gs+1, :])
    ax3 = plt.subplot(gs[108:128, :])
    ax4 = plt.subplot(gs[128:148, :])
    # fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12.0, 12.0))
    plt.subplots_adjust(hspace=0.001)

    y_lims = [Frequency_end, Frequency_start]
    ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=5,vmax=30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(labelsize=10)
    # ax.set_xlabel('Time [UT]')
    # ax.set_ylabel('Frequency [MHz]')
    ax.set_yscale('log')

    

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')
    

    ax2.imshow(p_data_1.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_1[0],x_lim_1[1],y_lim_1[0],y_lim_1[1]],
                vmin=vmin_1, vmax=vmax_1)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(labelsize=10)
    ax2.set_xlabel('Time [UT]')
    # ax2.set_ylabel('Frequency [MHz]')
    ax2.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False) 
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False) 
    ax2.xaxis.tick_bottom()
    ax2.set_yscale('log')

    num=10
    b=np.ones(num)/num
    y_2=np.convolve(p_data_2.T.values.tolist()[247], b, mode='valid')
    ax3.plot(np.arange(0,len(y_2),1),y_2, '-', label = '13.425MHz')
    ax3.set_ylim(min(y_2),np.percentile([x for x in y_2 if str(x) != 'nan'], 90))
    y_1=np.convolve(p_data_1.T.values.tolist()[224], b, mode='valid')
    ax4.plot(np.arange(0,len(y_1),1),y_1, '-', label = '916kHz')
    ax4.set_ylim(min(y_1),np.percentile([x for x in y_1 if str(x) != 'nan'], 90))

    ax3.legend()
    ax3.spines['bottom'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop=False) 
    ax4.legend()
    # ax4.spines['bottom'].set_visible(False)
    ax4.xaxis.tick_top()
    ax4.tick_params(labeltop=False) 

    if not os.path.isdir(Parent_directory + '/solar_burst/Nancaywind_all/'+year + '/' + month):
        os.makedirs(Parent_directory + '/solar_burst/Nancaywind_all/'+year + '/' + month)
    filename = Parent_directory + '/solar_burst/Nancaywind_all/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    plt.savefig(filename)
    plt.show()
    return

def radio_plot_rad2(data, receiver='rad2'):
    plt.close()
    if type(data) != pd.core.frame.DataFrame:
        print('radio_plot \n Type error: data type must be DataFrame')
        sys.exit()
    
    p_data = 20*np.log10(data)
    vmin = 0
    vmax = 4
    
    freq_axis = freq_setting(receiver)
    y_lim = [freq_axis[0], freq_axis[-1]]
    
    time_axis = data.index
    x_lim = [time_axis[0], time_axis[-1]]
    x_lim = mdates.date2num(x_lim)
    
    fig = plt.figure(figsize=[8,6])
    
    axes = fig.add_subplot(111)
    axes.imshow(p_data.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim[0],x_lim[1],y_lim[0],y_lim[1]],
                vmin=vmin, vmax=vmax)
    axes.xaxis_date()
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes.tick_params(labelsize=10)
    axes.set_xlabel('Time [UT]')
    axes.set_ylabel('Frequency [MHz]')
    plt.show()
    return


