#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 00:01:35 2021

@author: yuichiro
"""
# import glob
# import random
# import os 
# import shutil
# import math

# INPUT_DIR = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/train_cycle23/70/flare/'
# OUTPUT_DIR = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/train_cycle23_test/70/flare/'
# #ランダムで抽出する割合
# SAMPLING_RATIO = 0.2

# def random_sample_file():
#     files = glob.glob(INPUT_DIR + '/*.png')

#     random_sample_file = random.sample(files,math.ceil(len(files)*SAMPLING_RATIO))
#     os.makedirs(OUTPUT_DIR,exist_ok=True)

#     for file in random_sample_file:
#         shutil.move(file,OUTPUT_DIR)

# if __name__ == '__main__':
#     random_sample_file()

# 20130102_081307_081947_1020_1420_167_176_79.825_45.7compare.png
# /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/train_cycle23_test/80/noise/20130105_092206_092846_5100_5500_68_81_47.275_29.95compare.png
# import glob
# # import sys
# import shutil
# files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_af_jpgu/flare_clear/simple/*compare.png')
# for file in files:
#     file_name = file.split('/')[11]
#     if int(file_name.split('_')[0]) >= 20070101:
#         if int(file_name.split('_')[0]) <= 20100101:
#             search_file = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_af_sgepss/flare_clear/simple/'+file_name.split('_')[0] + '_' + file_name.split('_')[1] + '_' + file_name.split('_')[2] + '_' + file_name.split('_')[3] + '_' + file_name.split('_')[4] + '_' + file_name.split('_')[5] + '_' + file_name.split('_')[6] + '_*_' + file_name.split('_')[8])
#             savedir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/test_1/'
#             if len(search_file) == 0:
#                 print ('yes')
#                 shutil.copy(file, savedir)
#             else:
#                 pass



import random
color_setting = 1
import glob
import matplotlib.pyplot as plt
import os
from keras.preprocessing.image import load_img, img_to_array
import shutil
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
sigma_value = 2
after_plot = str('flare_event')
time_band = 340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
freq_check_range = 20
threshold_frequency_final = 10.5
cnn_plot_time = 50
save_place = 'cnn_af_sgepss'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 1.35
db_setting = 40

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

count_flare_1 = 0
count_noise_1 = 0
count_flare_2 = 0
count_noise_2 = 0
file_list = []

# /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/keras/pkl_file_af_jpgu_80/af_jpgu_inputdata_name128.csv
model = load_model_flare(Parent_directory, '/solar_burst/Nancay/data/keras/pkl_file_af_jpgu_80/af_jpgu_keras_param.hdf5', color_setting = 1, image_size = 128, fw = 3, strides = 1, fn_conv2d = 16, output_size = 2)
files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/train_cycle23_test/80/flare/*compare.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
# print('--- 読み込んだデータセットは', Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', 'です。')

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
        count_flare_1 += 1
    else:
        count_noise_1 += 1
        file_list.append(file)








noise_list = []

# model = load_model_flare(Parent_directory, '/solar_burst/Nancay/data/keras/af_jpgu_keras_param.hdf5', color_setting = 1, image_size = 128, fw = 3, strides = 1, fn_conv2d = 16, output_size = 2)
files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/train_cycle23_test/80/noise/*compare.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
# print('--- 読み込んだデータセットは', Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', 'です。')

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
        count_flare_2 += 1
        noise_list.append(file)
    else:
        count_noise_2 += 1

print (count_flare_1, count_noise_1)
print (count_flare_2, count_noise_2)
print ((count_flare_1+count_noise_2)/(count_flare_1+count_noise_1+count_flare_2+count_noise_2) )
