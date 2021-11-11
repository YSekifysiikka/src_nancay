#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 01:41:09 2019

@author: yuichiro
"""

import glob
import sys
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
    i = open(Parent_directory + '/solar_burst/Nancay/final_1.txt', 'w')
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


# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D

# def load_model_flare(Parent_directory, file_name, color_setting, image_size, fw, strides, fn_conv2d, output_size):
#     color_setting = color_setting  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
#     image_size = 128 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
#     fw = 3
#     strides = 1
#     fn_conv2d = 16
#     output_size = 2
#     model = Sequential()
#     model.add(Conv2D(fn_conv2d, (fw, fw), padding='same', strides=strides,
#               input_shape=(image_size, image_size, color_setting), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))               
#     model.add(Conv2D(128, (fw, fw), padding='same', strides=strides, activation='relu'))
#     model.add(Conv2D(256, (fw, fw), padding='same', strides=strides, activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=2))                
#     model.add(Dropout(0.2))                                   
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dropout(0.2))                                 
#     model.add(Dense(output_size, activation='softmax'))
#     model.load_weights(Parent_directory + file_name)
#     return model


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
    return diff_move_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)


import math
import datetime as dt
import matplotlib.dates as mdates
import datetime

def separated_data(diff_move_db, diff_db_min_med, epoch, time_co, time_band, t, Status):
    if t == math.floor((diff_db_min_med.shape[1]-time_co)/time_band):
        t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
    if t >= 0:
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
        diff_move_db_sep = diff_move_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
        Status_sep = Status[time - time_band - time_co:time]
    else:
        time = 0
        print (time)
        #+1 is due to move_average
        start = epoch[time + 1]
        start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
        # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
        time = diff_db_min_med.shape[1] - 1
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
    
    
    
        diff_db_plot_sep = diff_db_min_med[freq_start_idx:freq_end_idx + 1]
        diff_move_db_sep = diff_move_db[freq_start_idx:freq_end_idx + 1]
        Status_sep = Status

    return diff_db_plot_sep, diff_move_db_sep, x_lims, time, Time_start, Time_end, t, Status_sep
    
def threshold_array(diff_move_db_plot_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration, db_threshold_percentail,db_range, Status_sep):
    quartile_db_l = []
    mean_l_list = []
    stdev_sub = []
    quartile_power = []
    
    cali_place = []
    for cali17 in (np.where(Status_sep== 17)[0]):
        if cali17 >= 1:
            if cali17 + 11 <= diff_db_min_med.shape[1] - 1:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17-1,cali17 + 12,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17-1,diff_db_min_med.shape[1],1)
        elif cali17 == 0:
            if cali17 + 12 <= diff_db_min_med.shape[1]:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17,cali17 + 12,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali17,diff_db_min_med.shape[1],1)
        else:
            sys.exit()
            pass

    for cali0 in np.where(Status_sep== 0)[0]:
        if cali0 >= 41:
            if cali0 + 1 <= diff_db_min_med.shape[1] - 1:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali0-41,cali0 + 2,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(cali0-41, diff_db_min_med.shape[1],1)
        elif cali0 >= 0 and cali0 < 41:
            if cali0 + 2 <= diff_db_min_med.shape[1]:
                cali_place[len(cali_place):len(cali_place)] = np.arange(0,cali0 + 2,1)
            else:
                cali_place[len(cali_place):len(cali_place)] = np.arange(0,diff_db_min_med.shape[1],1)
        else:
            sys.exit()
            pass
    cali_place = list(set(cali_place))
    diff_move_db_plot_sep = np.delete(diff_move_db_plot_sep, cali_place, 1)


    for i in range(diff_move_db_plot_sep.shape[0]):
    #    for i in range(0, 357, 1):
        quartile_db_25_l = np.percentile(diff_move_db_plot_sep[i], db_threshold_percentail)
        quartile_db_25_2 = np.percentile(diff_move_db_plot_sep[i], db_threshold_percentail-db_range)
        quartile_db_each_l = []
        for k in range(diff_move_db_plot_sep[i].shape[0]):
            if diff_move_db_plot_sep[i][k] <= quartile_db_25_l:
                if diff_move_db_plot_sep[i][k] >= quartile_db_25_2:
                    diff_power_quartile_l = (10 ** ((diff_move_db_plot_sep[i][k])/10))
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
    diff_power_last_l = ((10 ** ((diff_move_db_plot_sep)/10)).T - quartile_power).T
    
    arr_threshold_1 = np.where(diff_power_last_l > 1, diff_power_last_l, 1)
    arr_threshold = np.log10(arr_threshold_1) * 10
    return arr_threshold, mean_l_list, quartile_db_l, quartile_power, diff_power_last_l, stdev_sub


def plot_array_threshold(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, db_setting, min_db, quartile_db_l):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    figure_=plt.figure(1,figsize=(6,4))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    ax2 = axes_2.imshow(arr_threshold[freq_start_idx:freq_end_idx + 1], extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] - 10)
    # ax2 = axes_2.imshow(arr_threshold[freq_start_idx:freq_end_idx + 1], 
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

    plt.ylim(Frequency[-1], Frequency[0])
    plt.show()
    plt.close()
    return
    # print(event_start_list, event_end_list)

    # return arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list_1, time_list_1, arr_sep_time_list

import matplotlib.pyplot as plt
import os



import matplotlib.gridspec as gridspec
# def plot_data(diff_db_plot_sep, diff_db_sep, freq_list, time_list, arr_5, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot):
#     year = date_OBs[0:4]
#     month = date_OBs[4:6]
#     day = date_OBs[6:8]
#     fontsize = 110
#     ticksize = 100
#     y_lims = [Frequency[-1], Frequency[0]]
#     db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
#     plt.close()
#     figure_=plt.figure(1,figsize=(140,50))
#     gs = gridspec.GridSpec(140, 50)


#     axes_1 = figure_.add_subplot(gs[:55, :20])
# #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
#     ax1 = axes_1.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 5)
#     axes_1.xaxis_date()
#     date_format = mdates.DateFormatter('%H:%M:%S')
#     axes_1.xaxis.set_major_formatter(date_format)
#     plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
#     plt.xlabel('Time (UT)',fontsize=fontsize )
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     cbar = plt.colorbar(ax1)
#     cbar.ax.tick_params(labelsize=ticksize)
#     cbar.set_label('Decibel [dB]', size=fontsize)
#     axes_1.tick_params(labelsize=ticksize)
#     figure_.autofmt_xdate()


#     axes_1 = figure_.add_subplot(gs[85:, :20])
#     ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
#     axes_1.xaxis_date()
#     date_format = mdates.DateFormatter('%H:%M:%S')
#     axes_1.xaxis.set_major_formatter(date_format)
#     plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
#     plt.xlabel('Time (UT)',fontsize=fontsize)
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     cbar = plt.colorbar(ax1)
#     cbar.ax.tick_params(labelsize=ticksize)
#     cbar.set_label('from Background [dB]', size=fontsize)
#     axes_1.tick_params(labelsize=ticksize)
#     figure_.autofmt_xdate()


#     axes_2 = figure_.add_subplot(gs[:, 21:34])
#     ax2 = axes_2.imshow(arr_5[freq_start_idx:freq_end_idx + 1], extent = [0, 50, 30, y_lims[1]], 
#               aspect='auto',cmap='jet',vmin= vmin_1 -2 ,vmax = vmax_1)
#     plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
#     plt.xlabel('Time[sec]',fontsize=fontsize)
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     cbar = plt.colorbar(ax2)
#     cbar.ax.tick_params(labelsize=ticksize)
#     cbar.set_label('from Background [dB]', size=fontsize)
#     axes_2.tick_params(labelsize=ticksize)
#     figure_.autofmt_xdate()
    

#     axes_2 = figure_.add_subplot(gs[:,38:51])
    
#     ######################################################################################


#     axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=30)
#     x_cmap = Frequency
#     y_cmap = np.arange(0, time_band + time_co, 1)
#     cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
#     cs.cmap.set_over('red')
#     cs.cmap.set_under('blue')
#     cycle = 0
#     for factor in factor_list:
#         ###########################################
#         axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 30.0)
#         cycle += 1
# #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
#     plt.xlim(min(time_list) - 10, max(time_list) + 10)
#     plt.ylim(min(freq_list), max(freq_list))
#     plt.title('Nancay: '+year+'-'+month+'-'+day+ ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize  + 10)
#     plt.xlabel('Time[sec]',fontsize=fontsize)
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     plt.tick_params(labelsize=ticksize)
#     plt.legend(fontsize=ticksize - 40)
#     figure_.autofmt_xdate()

#     if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
#         os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
#     filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
#     plt.savefig(filename)
#     plt.show()
#     plt.close()
#     return filename


#rawデータに移動平均をかけている
sigma_value = 2
after_plot = str('af_sgepss')
time_band = 5340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
freq_check_range = 20
threshold_frequency_final = 10.5
cnn_plot_time = 50
file_path = Parent_directory + '/solar_burst/Nancay/final_1.txt'
save_place = 'cnn_af_sgepss'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 1.35
db_setting = 15
db_range = 5

    
db_threshold_percentails = np.arange(db_range,db_range*2,db_range)
db_check_mean_list = []

for db_threshold_percentail in db_threshold_percentails:
    import csv
    years = ['2013']
    with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_2.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=["obs_time", "freq_start", "freq_end", "decibel"])
        w.writeheader()
        for year in years:
            start_date, end_date = final_txt_make(Parent_directory, Parent_lab, int(year), 303, 1231)
            gen = file_generator(file_path)
            for file in gen:
                file_name = file[:-1]
                diff_move_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data(Parent_directory, file_name, 3, 80, 10)
            
                mean_list = []
                time_list_oneday = []
                
                for t in range (math.floor(((diff_db_min_med.shape[1]-time_co)/time_band) + 1)):
                    diff_db_plot_sep, diff_move_db_sep, x_lims, time, Time_start, Time_end, t_1, Status_sep = separated_data(diff_move_db, diff_db_min_med, epoch, time_co, time_band, t, Status)
                    arr_threshold, mean_l_list, quartile_db_l, quartile_power, diff_power_last_l, stdev_sub = threshold_array(diff_move_db_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration, db_threshold_percentail, db_range, Status_sep)
                    # plot_array_threshold(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, db_setting, min_db, quartile_db_l)
                    mean_list.append(mean_l_list)
                    time_list_oneday.append(np.mean(x_lims))
                    obs_time= datetime.datetime(int(date_OBs[:4]), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]), int(Time_start.split(':')[2])) + (datetime.datetime(int(date_OBs[:4]), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_end.split(':')[0]), int(Time_end.split(':')[1]), int(Time_end.split(':')[2])) - datetime.datetime(int(date_OBs[:4]), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]), int(Time_start.split(':')[2])))/2
                    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
                    # np.array(mean_list)[:,db_standard[0][0]]
                    w.writerow({'obs_time':obs_time, 'freq_start':Frequency_start, 'freq_end':Frequency_end, 'decibel':mean_l_list})
                db_check_mean_list.append(np.array(mean_list))

            
                if len(mean_list) == 1:
                    mean_list = []
                    time_list_oneday = []
                    for t in range(2):
                        mean_list.append(mean_l_list)
                        time_list_oneday.append(x_lims[t])
                year = date_OBs[0:4]
                month = date_OBs[4:6]
                day = date_OBs[6:8]
                # figure_=plt.figure(1,figsize=(6,4))
                # gs = gridspec.GridSpec(6, 4)
                # axes_2 = figure_.add_subplot(gs[:, :])
                
                # ax2 = axes_2.imshow(np.array(mean_list).T,extent = [min(time_list_oneday), max(time_list_oneday),  Frequency[-1], Frequency[0]],aspect='auto',cmap='jet',vmin=15,vmax = 30)
                # # ax2 = axes_2.imshow(arr_threshold[freq_start_idx:freq_end_idx + 1], 
                # #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
                # axes_2.xaxis_date()
                # date_format = mdates.DateFormatter('%H:%M:%S')
                # axes_2.xaxis.set_major_formatter(date_format)
                # plt.title('Nancay: '+year+'-'+month+'-'+day)
                # plt.xlabel('Time',fontsize=20)
                # plt.ylabel('Frequency [MHz]',fontsize=20)
                # cbar = plt.colorbar(ax2)
                # cbar.ax.tick_params(labelsize=18)
                # cbar.set_label('Decibel [dB]', size=20)
                # axes_2.tick_params(labelsize=18)
                
                # plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                # figure_.autofmt_xdate()
                
                # plt.ylim(Frequency[-1], Frequency[0])
                # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_all/'+year):
                #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_all/'+year)
                # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_all/'+year+'/'+year+month+day+'.png'
                # plt.savefig(filename)
                # # plt.show()
                # plt.close()
                db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
                fig = plt.figure()
                ax1 = fig.add_subplot(111)
                ax1.scatter(time_list_oneday, np.array(mean_list)[:,db_standard[0][0]])
                ln1=ax1.plot(time_list_oneday, np.array(mean_list)[:,db_standard[0][0]],'C0',label='parcentail: ' + str(db_threshold_percentails[0]-db_range) + '-' + str(db_threshold_percentails[0])+'%', linestyle='dashdot')
                # ax1.plot(time_list_oneday, np.array(mean_list[i+1])[:,db_standard[0][0]],'C1',label='parcentail: ' + str(db_threshold_percentails[i+1]-db_range) + '-' + str(db_threshold_percentails[i+1])+'%', linestyle='dashdot')
                
                # ax2 = ax1.twinx()
                
                # ln2=ax2.plot(time_list_oneday, np.array(mean_list[i+1][:,db_standard[0][0]] - np.array(mean_list[i])[:,db_standard[0][0]]),'C2',label='Gap')
                
                h1, l1 = ax1.get_legend_handles_labels()
                # h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1, l1, loc='upper right')
                ax1.xaxis_date()
                date_format = mdates.DateFormatter('%H:%M')
                ax1.xaxis.set_major_formatter(date_format)
                ax1.set_xlabel('t')
                ax1.set_ylabel('Decibel [dB]',fontsize=20)
                ax1.grid(True)
                # ax1.set_ylim([min(np.array(mean_list)[:,db_standard[0][0]]), min(np.array(mean_list)[:,db_standard[0][0]]) + 10])
                # ax2.set_ylabel('Decibel [dB]',fontsize=20)
                # ax2.set_ylim([0, 2])
                plt.title('Nancay: '+year+'-'+month+'-'+day+': dB threshold '+ str(db_threshold_percentails[0]-db_range) + '-' + str(db_threshold_percentails[0]))
                if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_15/'+year):
                    os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_15/'+year)
                filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_15/'+year+'/'+year+month+day+'.png'
                plt.savefig(filename)
                plt.show()
                plt.close()
                # sys.exit()
            
                    
                    # arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list, time_list, arr_sep_time_list = sep_array(arr_threshold, diff_db_plot_sep, Frequency, time_band, time_co,  quartile_power, time, duration, resolution, Status, cnn_plot_time, t_1)
                    # if len(arr_5_list) == 0:
                    #     pass
                    # else:
                    #     for i in range(len(arr_5_list)):
                    #         save_directory = cnn_detection(arr_5_list[i], event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], date_OBs, Time_start, Time_end, color_setting, image_size, img_rows, img_cols, cnn_model, save_place, Frequency, x_lims)
                    #         if save_directory.split('/')[-1] == 'flare':
                    #             residual_list, save_directory_1, x_time, y_freq, time_rate_final = residual_detection(Parent_directory, save_directory, factor_list, freq_list[i], time_list[i], save_place, residual_threshold, date_OBs, Time_start, Time_end, event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i])
                    #             if min(residual_list) <= residual_threshold:
                    #                 plot_data(diff_db_plot_sep, diff_db_sep, freq_list[i], time_list[i], arr_5_list[i], x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], arr_sep_time_list[i], quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot)
                    #                 best_factor = np.argmin(residual_list) + 1
                    #                 time_event = dt.timedelta(seconds=(int(event_end_list[i]) + int(event_start_list[i]))/2) + dt.datetime(int(date_OBs[0:4]), int(date_OBs[4:6]), int(date_OBs[6:8]),int(Time_start[0:2]), int(Time_start[3:5]), int(Time_start[6:8]))
                    #                 date_event = str(time_event.date())[0:4] + str(time_event.date())[5:7] + str(time_event.date())[8:10]
                    #                 date_event_hour = str(time_event.hour)
                    #                 date_event_minute = str(time_event.minute)
                    #                 print (time_rate_final)
                                    # w.writerow({'event_date':date_event, 'event_hour':date_event_hour, 'event_minite':date_event_minute,'velocity':time_rate_final, 'residual':residual_list, 'event_start': event_start_list[i],'event_end': event_end_list[i],'freq_start': freq_start_list[i],'freq_end':freq_end_list[i], 'factor':best_factor, 'peak_time_list':time_list[i], 'peak_freq_list':freq_list[i]})
                                    # sys.exit()
        
