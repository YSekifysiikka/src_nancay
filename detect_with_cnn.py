#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:16:09 2019

@author: yuichiro
"""

import scipy
import cdflib
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
import random
from keras.preprocessing.image import load_img, img_to_array
import shutil

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_path = '/Users/yuichiro/Desktop/lab/solar_burst/Nancay/final.txt'

###############
sigma_range = 1
sigma_start = 2
sigma_mul = 1
#when you only check one threshold
sigma_value = 2
after_plot = str('drift_check')
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




color_setting = 1  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
image_size = 128 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
img_rows, img_cols = image_size, image_size
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
input_shape=(image_size, image_size, 1)
fw = 3
strides = 1
fn_conv2d = 50
hidden_size = 150
output_size = 2
batch_size = 32
epochs = 14
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', strides=strides,
          input_shape=(image_size, image_size, color_setting), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))               
model.add(Conv2D(128, (3, 3), padding='same', strides=strides, activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', strides=strides, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))                
model.add(Dropout(0.2))                                   
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))                                 
model.add(Dense(output_size, activation='softmax'))

model.load_weights(Parent_directory + '/solar_burst/Nancay/data/keras/pkl_file_new/keras_param_128_0.9945.hdf5')
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

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
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


    for t in range (math.floor(((diff_db_min_med.shape[1]-time_co)/time_band) + 1)):
#                ################################
#                for l in range(sigma_range):
#                    sigma_value = sigma_start + sigma_mul*l
#                ################################
        if t == math.floor((diff_db_min_med.shape[1]-time_co)/time_band):
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
        x2 = np.flip(np.linspace(29.95, x_end_range, 286))
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
                                                                    for k in range(50):
                                                                        middle_k = middle + k
                                                                        if middle_k >= 0:
                                                                            if middle_k <= 399:
                                                                                for l in range(check_frequency_range):
                                                                                    if arr2[l][middle_k] > -10:
                                                                                        arr5[l][k] = arr2[l][middle_k]
                                                                    y_lims = [30, 80]
                                                                    event_start = str(min(drift_time_start))
                                                                    event_end = str(max(drift_time_end))
                                                                    freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                                    freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                                    event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                                    freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                                    plt.close(1)

                                                                    figure_=plt.figure(1,figsize=(10,10))
                                                                    axes_2 = figure_.add_subplot(1, 1, 1)
                                                                    ax2 = axes_2.imshow(arr5[0:286], extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                              aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                    plt.axis('off')
                                                                    
                                                                    extent = axes_2.get_window_extent().transformed(figure_.dpi_scale_trans.inverted())
                                                                    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all'):
                                                                        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all')
                                                                    filename = Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
                                                                    if os.path.isfile(filename):
                                                                      os.remove(filename)
                                                                    plt.savefig(filename, bbox_inches=extent)
                                                                    plt.show()
                                                                    plt.close()

                                                                    files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
                                                                    print('--- 読み込んだデータセットは', Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', 'です。')

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
                                                                      save_directory = Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/flare'
                                                                      filename2 = save_directory + '/simple/' +year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
                                                                      if os.path.isfile(filename2):
                                                                        os.remove(filename2)
                                                                      shutil.move(Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory + '/simple/')
                                                                    else:
                                                                      save_directory = Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/others'
                                                                      filename2 = save_directory + '/simple/' +year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
                                                                      if os.path.isfile(filename2):
                                                                        os.remove(filename2)
                                                                      shutil.move(Parent_directory + '/solar_burst/Nancay/plot/cnn_simple/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory + '/simple/')

                                                                    figure_=plt.figure(1,figsize=(10,16))
                                                                    gs = gridspec.GridSpec(8, 6)
                                                                    axes_2 = figure_.add_subplot(gs[:4, :])
                                                        #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                    ax2 = axes_2.imshow(arr2[0:286], extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
                                                                              aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                    axes_2.xaxis_date()
                                                                    date_format = mdates.DateFormatter('%H:%M:%S')
                                                                    axes_2.xaxis.set_major_formatter(date_format)
                                                                    plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
                                                                    plt.xlabel('Time',fontsize=20)
                                                                    plt.ylabel('Frequency [MHz]',fontsize=20)
                                                                    cbar = plt.colorbar(ax2)
                                                                    cbar.ax.tick_params(labelsize=18)
                                                                    cbar.set_label('Decibel [dB]', size=20)
                                                                    axes_2.tick_params(labelsize=18)
                
                                                                    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                    figure_.autofmt_xdate()

                                                                    ax2 = figure_.add_subplot(gs[5:, :])
                                                                    ax2.plot(time_list, freq_list, "bo", label = 'Used_Data') #Interpolation
                                                                    plt.title('Used_data', fontsize = 20)
                                                                    plt.xlabel('Time[sec]',fontsize=20)
                                                                    plt.ylabel('Frequency[MHz]', fontsize=20)
                                                                    plt.legend(fontsize=12)
                                                                    plt.tick_params(labelsize=18)
                                                                    plt.xlim(min(time_list) - 5, max(time_list) + 5)
                                                                    plt.ylim(30, 80)
                                                                    if not os.path.isdir(save_directory + '/'+year+'/'+month):
                                                                        os.makedirs(save_directory + '/'+year+'/'+month)
                                                                    filename1 = year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'_'+str(time - time_band - time_co)+'_'+str(time)+'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end
                                                                    filename = save_directory + '/'+year+'/'+month+'/'+filename1+'.png'
                                                                    if os.path.isfile(filename):
                                                                      os.remove(filename)
                                                                    plt.savefig(filename)
                                                                    plt.show()
                                                                    plt.close()

                                                                    
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
                                                                    cbar = plt.colorbar(ax1)
                                                                    cbar.ax.tick_params(labelsize=10)
                                                                    cbar.set_label('Decibel [dB]', size=10)
                                                                    figure_.autofmt_xdate()
                                                            
                                                            
                                                                    axes_1 = figure_.add_subplot(gs[:8, :5])
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
                                                                    axes_5.set_xlim(0, 80*0.3125)
                                                                    axes_5.set_ylim(30, 80)
                                                                    plt.xlabel('Decibel [dB]',fontsize=8)
                                                                    plt.ylabel('Frequency [MHz]',fontsize=8)
                                                            
                                                                    axes_4 = figure_.add_subplot(gs[13:, 2:3])
                                                                    plt.title('mean',fontsize=10)
                                                                    axes_4.plot(mean_l_list, x1)
                                                                    axes_4.set_xlim(0, 50*0.3125)
                                                                    axes_4.set_ylim(30, 80)
                                                                    plt.xlabel('Decibel [dB]',fontsize=8)
                                                                    plt.ylabel('Frequency [MHz]',fontsize=8)
                                                            
                                                            
                                                                    axes_1 = figure_.add_subplot(gs[13:, 6:7])
                                                                    plt.title('m_'+str(sigma_value)+'sigma',fontsize=10)
                                                                    axes_1.plot(quartile_db_l, x1)
                                                                    axes_1.set_xlim(0, 50*0.3125)
                                                                    axes_1.set_ylim(30, 80)
                                                                    plt.xlabel('Decibel [dB]',fontsize=8)
                                                                    plt.ylabel('Frequency [MHz]',fontsize=8)
                                                            
                                                            
                                                                    axes_3 = figure_.add_subplot(gs[13:, 4:5])
                                                                    plt.title('stdev_'+str(sigma_value)+'sigma',fontsize=10)
                                                                    axes_3.plot(stdev_sub, x1)
                                                                    axes_3.set_xlim(0, 10*0.3125)
                                                                    axes_3.set_ylim(30, 80)
                                                                    plt.xlabel('Decibel [dB]',fontsize=8)
                                                                    plt.ylabel('Frequency [MHz]',fontsize=8)
                                                            
                                                                    if not os.path.isdir(save_directory + '/'+year+'/'+month):
                                                                        os.makedirs(save_directory + '/'+year+'/'+month)
                                                                    filename = save_directory+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+'__sigma_l_r.png'
                                                                    if os.path.isfile(filename):
                                                                      os.remove(filename)
                                                                    plt.savefig(filename)
                                                                    plt.show()
                                                                    plt.close()