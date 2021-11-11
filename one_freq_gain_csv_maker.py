#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:01:21 2021

@author: yuichiro
"""

import numpy as np
import pandas as pd
import datetime as dt
import sys
import matplotlib.pyplot as plt
import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
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

Calibration_time_list = []
gain_list = []


# file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_ver2_gain_analysis_20070101_20091231.csv'
file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_ver2_gain_analysis_20120101_20141231.csv'
file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_ver2_gain_analysis_20170101_20201231.csv'
# file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'


with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/test/ver2_30dB_40dB_gain_analysis.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "Frequency", "Right-gain", "Right-Trx", "Right-hot_dB", "Right-cold_dB"])
    w.writeheader()
    file_list = [file2, file3, file4]
    
    for file in file_list:
        print (file)
        each_Calibration_time_list = []
        each_gain_list = []
        csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
        # print(csv_input['Time_list'])
        for i in range(len(csv_input)):
            obs_time = dt.datetime(int(csv_input['obs_time'][i].split('-')[0]), int(csv_input['obs_time'][i].split('-')[1]), int(csv_input['obs_time'][i].split(' ')[0][-2:]), int(csv_input['obs_time'][i].split(' ')[1][:2]), int(csv_input['obs_time'][i].split(':')[1]), int(csv_input['obs_time'][i].split(':')[2][:2]))
            # Frequency_list = csv_input['Frequency'][i]
            Frequency = [float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != '']
            Frequency = np.array(Frequency)
            freq_idx_40 = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
            freq_idx_37_5 = np.where(Frequency == getNearestValue(Frequency,37.5))[0][0]
            freq_idx_35 = np.where(Frequency == getNearestValue(Frequency,35))[0][0]
            freq_idx_32_5 = np.where(Frequency == getNearestValue(Frequency,32.5))[0][0]
            freq_idx_30 = np.where(Frequency == getNearestValue(Frequency,30))[0][0]
            # gain = []
            # for j in range(len(csv_input['gain'][i][1:-1].replace('\n', '').split(' '))):
                # if not csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j] == '':
                    # gain.append(float(csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j]))
            # Gain_list = csv_input['gain'][i][1:-1].replace('\n', '').split(' ')
            Gain_list = np.array([float(k) for k in csv_input['Right-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
            gain = Gain_list[[freq_idx_40, freq_idx_37_5, freq_idx_35, freq_idx_32_5, freq_idx_30]]
    
            Trx_list = np.array([float(k) for k in csv_input['Right-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
            hot_dB_list = np.array([float(k) for k in csv_input['Right-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
            cold_dB_list = np.array([float(k) for k in csv_input['Right-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
            Trx = Trx_list[[freq_idx_40, freq_idx_37_5, freq_idx_35, freq_idx_32_5, freq_idx_30]]
            hot_dB = hot_dB_list[[freq_idx_40, freq_idx_37_5, freq_idx_35, freq_idx_32_5, freq_idx_30]]
            cold_dB = cold_dB_list[[freq_idx_40, freq_idx_37_5, freq_idx_35, freq_idx_32_5, freq_idx_30]]

            # # print ('a')
            # if not len(Gain_list) == len(Frequency):
            #     print ('c')
            #     sys.exit()
            w.writerow({'obs_time': obs_time, 'Frequency': Frequency[[freq_idx_40, freq_idx_37_5, freq_idx_35, freq_idx_32_5, freq_idx_30]], 'Right-gain':gain, 'Right-Trx': Trx, 'Right-hot_dB':hot_dB, 'Right-cold_dB': cold_dB})
