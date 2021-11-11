#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 11:50:00 2021

@author: yuichiro
"""




import pandas as pd
import datetime
import numpy as np
import math
import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


BG_30_csv_file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/30MHz/under25_RR_LL_gain_movecaliBG_30MHz_analysis_calibrated_median.csv'
csv_input_30 = pd.read_csv(filepath_or_buffer= BG_30_csv_file, sep=",")
obs_time_30dB = []
BG_30dB_list_RR = []
BG_30dB_list_LL = []
for i in range(len(csv_input_30)):
    BG_obs_time_event = datetime.datetime(int(csv_input_30['obs_time'][i].split('-')[0]), int(csv_input_30['obs_time'][i].split('-')[1]), int(csv_input_30['obs_time'][i].split(' ')[0][-2:]), int(csv_input_30['obs_time'][i].split(' ')[1][:2]), int(csv_input_30['obs_time'][i].split(':')[1]), int(csv_input_30['obs_time'][i].split(':')[2][:2]))
    obs_time_30dB.append(BG_obs_time_event)
    BG_30dB_list_RR.append(csv_input_30['Right-BG_move_Calibrated'][i])
    BG_30dB_list_LL.append(csv_input_30['Left-BG_move_Calibrated'][i])

BG_32_5_csv_file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/32.5MHz/under25_RR_LL_gain_movecaliBG_32_5MHz_analysis_calibrated_median.csv'
csv_input_32_5 = pd.read_csv(filepath_or_buffer= BG_32_5_csv_file, sep=",")
obs_time_32_5dB = []
BG_32_5dB_list_RR = []
BG_32_5dB_list_LL = []
for i in range(len(csv_input_32_5)):
    BG_obs_time_event = datetime.datetime(int(csv_input_32_5['obs_time'][i].split('-')[0]), int(csv_input_32_5['obs_time'][i].split('-')[1]), int(csv_input_32_5['obs_time'][i].split(' ')[0][-2:]), int(csv_input_32_5['obs_time'][i].split(' ')[1][:2]), int(csv_input_32_5['obs_time'][i].split(':')[1]), int(csv_input_32_5['obs_time'][i].split(':')[2][:2]))
    obs_time_32_5dB.append(BG_obs_time_event)
    BG_32_5dB_list_RR.append(csv_input_32_5['Right-BG_move_Calibrated'][i])
    BG_32_5dB_list_LL.append(csv_input_32_5['Left-BG_move_Calibrated'][i])

BG_35_csv_file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/35MHz/under25_RR_LL_gain_movecaliBG_35MHz_analysis_calibrated_median.csv'
csv_input_35 = pd.read_csv(filepath_or_buffer= BG_35_csv_file, sep=",")
obs_time_35dB = []
BG_35dB_list_RR = []
BG_35dB_list_LL = []
for i in range(len(csv_input_35)):
    BG_obs_time_event = datetime.datetime(int(csv_input_35['obs_time'][i].split('-')[0]), int(csv_input_35['obs_time'][i].split('-')[1]), int(csv_input_35['obs_time'][i].split(' ')[0][-2:]), int(csv_input_35['obs_time'][i].split(' ')[1][:2]), int(csv_input_35['obs_time'][i].split(':')[1]), int(csv_input_35['obs_time'][i].split(':')[2][:2]))
    obs_time_35dB.append(BG_obs_time_event)
    BG_35dB_list_RR.append(csv_input_35['Right-BG_move_Calibrated'][i])
    BG_35dB_list_LL.append(csv_input_35['Left-BG_move_Calibrated'][i])

BG_37_5csv_file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/37.5MHz/under25_RR_LL_gain_movecaliBG_37_5MHz_analysis_calibrated_median.csv'
csv_input_37_5 = pd.read_csv(filepath_or_buffer= BG_37_5csv_file, sep=",")
obs_time_37_5dB = []
BG_37_5dB_list_RR = []
BG_37_5dB_list_LL = []
for i in range(len(csv_input_37_5)):
    BG_obs_time_event = datetime.datetime(int(csv_input_37_5['obs_time'][i].split('-')[0]), int(csv_input_37_5['obs_time'][i].split('-')[1]), int(csv_input_37_5['obs_time'][i].split(' ')[0][-2:]), int(csv_input_37_5['obs_time'][i].split(' ')[1][:2]), int(csv_input_37_5['obs_time'][i].split(':')[1]), int(csv_input_37_5['obs_time'][i].split(':')[2][:2]))
    obs_time_37_5dB.append(BG_obs_time_event)
    BG_37_5dB_list_RR.append(csv_input_37_5['Right-BG_move_Calibrated'][i])
    BG_37_5dB_list_LL.append(csv_input_37_5['Left-BG_move_Calibrated'][i])

BG_40_csv_file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/40MHz/under25_RR_LL_gain_movecaliBG_40MHz_analysis_calibrated_median.csv'
csv_input_40 = pd.read_csv(filepath_or_buffer= BG_40_csv_file, sep=",")
obs_time_40dB = []
BG_40dB_list_RR = []
BG_40dB_list_LL = []
for i in range(len(csv_input_40)):
    BG_obs_time_event = datetime.datetime(int(csv_input_40['obs_time'][i].split('-')[0]), int(csv_input_40['obs_time'][i].split('-')[1]), int(csv_input_40['obs_time'][i].split(' ')[0][-2:]), int(csv_input_40['obs_time'][i].split(' ')[1][:2]), int(csv_input_40['obs_time'][i].split(':')[1]), int(csv_input_40['obs_time'][i].split(':')[2][:2]))
    obs_time_40dB.append(BG_obs_time_event)
    BG_40dB_list_RR.append(csv_input_40['Right-BG_move_Calibrated'][i])
    BG_40dB_list_LL.append(csv_input_40['Left-BG_move_Calibrated'][i])

obs_time_40dB = np.array(obs_time_40dB)
obs_time_37_5dB = np.array(obs_time_37_5dB)
obs_time_35dB = np.array(obs_time_35dB)
obs_time_32_5dB = np.array(obs_time_32_5dB)
obs_time_30dB = np.array(obs_time_30dB)
    

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG_5freq/under25_RR_LL_gain_movecaliBG_MHz_analysis_calibrated_median.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "Frequency", "Right-BG_move_Calibrated", "Left-BG_move_Calibrated", "Right-gain", "Left-gain", "Right-Trx", "Left-Trx", "Used_dB_median"])
    w.writeheader()
    for i in range(len(csv_input_40)):
        # if (obs_time_40dB[i]<=datetime.datetime(2008,12,30)) & (obs_time_40dB[i]>=datetime.datetime(2008,12,29)):
        print (obs_time_40dB[i])
        if all([(obs_time_40dB[i] in obs_time_30dB), (obs_time_40dB[i] in obs_time_35dB), (obs_time_40dB[i] in obs_time_32_5dB), obs_time_40dB[i] in obs_time_37_5dB]) == True:
            idx_37_5 = np.where(obs_time_37_5dB == obs_time_40dB[i])[0][0]
            idx_35 = np.where(obs_time_35dB == obs_time_40dB[i])[0][0]
            idx_32_5 = np.where(obs_time_32_5dB == obs_time_40dB[i])[0][0]
            idx_30 = np.where(obs_time_30dB == obs_time_40dB[i])[0][0]
            if any([math.isnan(BG_40dB_list_RR[i]), math.isnan(BG_40dB_list_LL[i]), math.isnan(BG_37_5dB_list_RR[idx_37_5]), math.isnan(BG_37_5dB_list_LL[idx_37_5]), math.isnan(BG_35dB_list_RR[idx_35]), math.isnan(BG_35dB_list_LL[idx_35]), math.isnan(BG_32_5dB_list_RR[idx_32_5]), math.isnan(BG_32_5dB_list_LL[idx_32_5]), math.isnan(BG_30dB_list_LL[idx_30]), math.isnan(BG_30dB_list_RR[idx_30])]):
                pass
            else:
                # pass
                print ('True')
                Frequency_list = np.array([csv_input_30['Frequency'][idx_30], csv_input_32_5['Frequency'][idx_32_5], csv_input_35['Frequency'][idx_35], csv_input_37_5['Frequency'][idx_37_5], csv_input_40['Frequency'][i]])
                Right_BG_list = np.array([csv_input_30['Right-BG_move_Calibrated'][idx_30], csv_input_32_5['Right-BG_move_Calibrated'][idx_32_5], csv_input_35['Right-BG_move_Calibrated'][idx_35], csv_input_37_5['Right-BG_move_Calibrated'][idx_37_5], csv_input_40['Right-BG_move_Calibrated'][i]])
                Left_BG_list = np.array([csv_input_30['Left-BG_move_Calibrated'][idx_30], csv_input_32_5['Left-BG_move_Calibrated'][idx_32_5], csv_input_35['Left-BG_move_Calibrated'][idx_35], csv_input_37_5['Left-BG_move_Calibrated'][idx_37_5], csv_input_40['Left-BG_move_Calibrated'][i]])
                Right_gain_list = np.array([csv_input_30['Right-gain'][idx_30], csv_input_32_5['Right-gain'][idx_32_5], csv_input_35['Right-gain'][idx_35], csv_input_37_5['Right-gain'][idx_37_5], csv_input_40['Right-gain'][i]])
                Left_gain_list = np.array([csv_input_30['Left-gain'][idx_30], csv_input_32_5['Left-gain'][idx_32_5], csv_input_35['Left-gain'][idx_35], csv_input_37_5['Left-gain'][idx_37_5], csv_input_40['Left-gain'][i]])
                Right_Trx_list = np.array([csv_input_30['Right-Trx'][idx_30], csv_input_32_5['Right-Trx'][idx_32_5], csv_input_35['Right-Trx'][idx_35], csv_input_37_5['Right-Trx'][idx_37_5], csv_input_40['Right-Trx'][i]])
                Left_Trx_list = np.array([csv_input_30['Left-Trx'][idx_30], csv_input_32_5['Left-Trx'][idx_32_5], csv_input_35['Left-Trx'][idx_35], csv_input_37_5['Left-Trx'][idx_37_5], csv_input_40['Left-Trx'][i]])
                Used_db_list = np.array([csv_input_30['Used_dB_median'][idx_30], csv_input_32_5['Used_dB_median'][idx_32_5], csv_input_35['Used_dB_median'][idx_35], csv_input_37_5['Used_dB_median'][idx_37_5], csv_input_40['Used_dB_median'][i]])
                w.writerow({ "obs_time": obs_time_40dB[i], "Frequency": Frequency_list, "Right-BG_move_Calibrated": Right_BG_list, "Left-BG_move_Calibrated": Left_BG_list, "Right-gain": Right_gain_list, "Left-gain": Left_gain_list, "Right-Trx": Right_Trx_list, "Left-Trx": Left_Trx_list, "Used_dB_median":Used_db_list})
