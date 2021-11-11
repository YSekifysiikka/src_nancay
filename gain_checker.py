#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 17:01:21 2021

@author: yuichiro
"""
# # file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
# file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120118_20121231.csv'
# file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140101_20140202.csv'
# file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140401_20140424.csv'
# file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'

# import numpy as np
# import pandas as pd
# import datetime as dt
# import sys
# import matplotlib.pyplot as plt

# obs_time_list = []
# intensity_list = []
# bg_list = []

# file_list = [file2, file3, file4, file5, file6, file7]

# for file in file_list:
#     print (file)
#     each_obs_time_list = []
#     each_intensity_list = []
#     each_bg_list = []
#     csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
#     # print(csv_input['Time_list'])
#     for i in range(len(csv_input)):
#         obs_time = dt.datetime(int(csv_input['Time_list'][i].split('-')[0]), int(csv_input['Time_list'][i].split('-')[1]), int(csv_input['Time_list'][i].split(' ')[0][-2:]), int(csv_input['Time_list'][i].split(' ')[1][:2]), int(csv_input['Time_list'][i].split(':')[1]), int(csv_input['Time_list'][i].split(':')[2][:2]))
#         intensity_list.append(csv_input['intensity_list'][i])
#         obs_time_list.append(obs_time)
#         bg_list.append(csv_input['BG_40[dB]'][i])
#         each_intensity_list.append(csv_input['intensity_list'][i])
#         each_obs_time_list.append(obs_time)
#         each_bg_list.append(csv_input['BG_40[dB]'][i])
#     plt.title(file.split('/')[-1].split('.')[0].split('_')[2][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[2][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[2][6:8]+'-'+file.split('/')[-1].split('.')[0].split('_')[3][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[3][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[3][6:8])
#     plt.hist(each_intensity_list, bins = 20)
#     plt.yscale('log')
#     # plt.xscale('log')
#     plt.xlabel('dB from background[V^2/Hz]')
#     plt.ylabel('Occurence Number')
#     plt.show()
#     plt.close()

#     plt.title(file.split('/')[-1].split('.')[0].split('_')[2][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[2][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[2][6:8]+'-'+file.split('/')[-1].split('.')[0].split('_')[3][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[3][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[3][6:8])
#     plt.plot(each_obs_time_list, each_bg_list, '.')
#     # plt.yscale('log')
#     # plt.xscale('log')
#     plt.xlabel('Time')
#     plt.ylabel('BG[dB]')
#     plt.xticks(rotation=45)
#     plt.show()
#     plt.close()
#     # sys.exit()

# plt.title('Total')
# plt.hist(intensity_list, bins = 20)
# plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('dB from background[V^2/Hz]')
# plt.ylabel('Occurence Number')
# plt.show()
# plt.close()

# plt.title('Total')
# plt.plot(obs_time_list, bg_list, '.')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('Time')
# plt.ylabel('BG[dB]')
# plt.xticks(rotation=45)
# plt.show()
# plt.close()

#Newversion
# file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20070101_20091231.csv'
file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20120101_20141231.csv'
file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20170101_20201231.csv'
# file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'

import numpy as np
import pandas as pd
import datetime as dt
import sys
import matplotlib.pyplot as plt
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
        freq_idx = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
        # gain = []
        # for j in range(len(csv_input['gain'][i][1:-1].replace('\n', '').split(' '))):
            # if not csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j] == '':
                # gain.append(float(csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j]))
        # Gain_list = csv_input['gain'][i][1:-1].replace('\n', '').split(' ')
        Gain_list = [float(k) for k in csv_input['Right-gain'][i][1:-1].replace('\n', '').split(' ') if k != '']
        gain = float(Gain_list[freq_idx])

        Trx_list = [float(k) for k in csv_input['Right-Trx'][i][1:-1].replace('\n', '').split(' ') if k != '']
        hot_dB_list = [float(k) for k in csv_input['Right-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != '']
        cold_dB_list = [float(k) for k in csv_input['Right-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != '']
        Trx = float(Trx_list[freq_idx])
        hot_dB = float(hot_dB_list[freq_idx])
        cold_dB = float(cold_dB_list[freq_idx])
        
        # print ('a')
        if not len(Gain_list) == len(Frequency):
            print ('c')
            sys.exit()
        # print ('b')
        Calibration_time_list.append(obs_time)
        each_Calibration_time_list.append(obs_time)
        gain_list.append(gain)
        each_gain_list.append(gain)


    plt.title(file.split('/')[-1].split('.')[0].split('_')[2][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[2][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[2][6:8]+'-'+file.split('/')[-1].split('.')[0].split('_')[3][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[3][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[3][6:8])
    plt.plot(each_Calibration_time_list, each_gain_list, '.')
    # plt.yscale('log')
    # plt.xscale('log')
    plt.xlabel('Time')
    plt.ylabel('Gain[dB]')
    plt.xticks(rotation=45)
    plt.show()
    plt.close()
    # sys.exit()

plt.title('Total')
plt.hist(gain_list, bins = 20)
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Gain[dB]')
plt.ylabel('Occurence Number')
# plt.xticks(rotation=45)
plt.show()
plt.close()


plt.title('Total')
plt.plot(Calibration_time_list, gain_list, '.')
# plt.yscale('log')
# plt.xscale('log')
plt.xlabel('Time')
plt.ylabel('Gain[dB]')
plt.xticks(rotation=45)
plt.show()
plt.close()




# #10days_Oldversion
# # file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
# file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/gain_analysis_20070101_20091231.csv'
# file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/gain_analysis_20120101_20141231.csv'
# file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/gain_analysis_20170101_20201231.csv'
# # file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# # file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# # file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'

# import numpy as np
# import pandas as pd
# import datetime as dt
# import sys
# import matplotlib.pyplot as plt
# def getNearestValue(list, num):
#     """
#     概要: リストからある値に最も近い値を返却する関数
#     @param list: データ配列
#     @param num: 対象値
#     @return 対象値に最も近い値
#     """

#     # リスト要素と対象値の差分を計算し最小値のインデックスを取得
#     idx = np.abs(np.asarray(list) - num).argmin()
#     return list[idx]

# Calibration_time_list = []
# gain_list = []


# file_list = [file2, file3, file4]

# for file in file_list:
#     print (file)
#     each_Calibration_time_list = []
#     each_gain_list = []
#     csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
#     # print(csv_input['Time_list'])
#     for i in range(len(csv_input)):
#         obs_time = dt.datetime(int(csv_input['Calibration_time'][i].split('-')[0]), int(csv_input['Calibration_time'][i].split('-')[1]), int(csv_input['Calibration_time'][i].split(' ')[0][-2:]), int(csv_input['Calibration_time'][i].split(' ')[1][:2]), int(csv_input['Calibration_time'][i].split(':')[1]), int(csv_input['Calibration_time'][i].split(':')[2][:2]))
#         Frequency_list = csv_input['Freq_list'][i]
#         Frequency = []
#         for j in range(len(Frequency_list[1:-1].replace('\n', '').split(' '))):
#             # print (j)
#             if not Frequency_list[1:-1].replace('\n', '').split(' ')[j] == '':
#                 Frequency.append(float(Frequency_list[1:-1].replace('\n', '').split(' ')[j]))
#         Frequency = np.array(Frequency)
#         freq_idx = np.where(Frequency == getNearestValue(Frequency,40))[0][0]
#         # gain = []
#         # for j in range(len(csv_input['gain'][i][1:-1].replace('\n', '').split(' '))):
#             # if not csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j] == '':
#                 # gain.append(float(csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j]))
#         # Gain_list = csv_input['gain'][i][1:-1].replace('\n', '').split(' ')
#         Gain_list = [k for k in csv_input['gain'][i][1:-1].replace('\n', '').split(' ') if k != '']
#         gain = float(Gain_list[freq_idx])
#         # print ('a')
#         if not len(Gain_list) == len(Frequency):
#             print ('c')
#             sys.exit()
#         # print ('b')
#         Calibration_time_list.append(obs_time)
#         each_Calibration_time_list.append(obs_time)
#         gain_list.append(gain)
#         each_gain_list.append(gain)


#     plt.title(file.split('/')[-1].split('.')[0].split('_')[2][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[2][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[2][6:8]+'-'+file.split('/')[-1].split('.')[0].split('_')[3][:4] +'/'+file.split('/')[-1].split('.')[0].split('_')[3][4:6]+'/'+file.split('/')[-1].split('.')[0].split('_')[3][6:8])
#     plt.plot(each_Calibration_time_list, each_gain_list, '.')
#     # plt.yscale('log')
#     # plt.xscale('log')
#     plt.xlabel('Time')
#     plt.ylabel('Gain[dB]')
#     plt.xticks(rotation=45)
#     plt.show()
#     plt.close()
#     # sys.exit()

# plt.title('Total')
# plt.hist(gain_list, bins = 20)
# # plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('Gain[dB]')
# plt.ylabel('Occurence Number')
# # plt.xticks(rotation=45)
# plt.show()
# plt.close()


# plt.title('Total')
# plt.plot(Calibration_time_list, gain_list, '.')
# # plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('Time')
# plt.ylabel('Gain[dB]')
# plt.xticks(rotation=45)
# plt.show()
# plt.close()