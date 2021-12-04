#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 15:04:46 2020

@author: yuichiro
"""

from PIL import Image
import numpy as np
import pandas as pd
import datetime as dt
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import glob
import os
import sys
import shutil
import csv
labelsize = 18
fontsize = 20
Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/solar_burst/Nancay/analysis_data/residual_test_final.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

# with open(Parent_directory+ '/solar_burst/Nancay/analysis_data/residual_test_max_10.csv', 'w') as f:
    # w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor"])
    # w.writeheader()

factor_list = []
y_residual_0 = []
y_residual_1 = []
file_name = []

for i in range(len(csv_input_final)):
    velocity = csv_input_final[["velocity"][0]][i].lstrip("['")[:-1].split(',')
    residual = csv_input_final[["residual"][0]][i].lstrip("['")[:-1].split(',')
    velocity_list = [float(s) for s in velocity]
    residual_list_1 = np.array([float(s) for s in residual])
    freq_start = csv_input_final['freq_start'][i]
    freq_end = csv_input_final['freq_end'][i]
    freq_range = ((freq_start - freq_end)/0.175) + 1
    # residual_list_1 = np.sqrt(residual_list_1/freq_range)
    residual_list_1 = np.sqrt(residual_list_1)
    y_residual_0.append(min(residual_list_1))
    if min(residual_list_1)  < 0.3:
    #     # if min(residual_list_1) > 0.2:
    #         # print('b')
        date_event = csv_input_final['event_date'][i]
        date_event_hour = csv_input_final['event_hour'][i]
        date_event_minute = csv_input_final['event_minite'][i]
        time_rate_final = csv_input_final['velocity'][i]
        residual_list = min(residual_list_1)
        event_start = csv_input_final['event_start'][i]
        event_end = csv_input_final['event_end'][i]
        # freq_start = csv_input_final['freq_start'][i]
        # freq_end = csv_input_final['freq_end'][i]
        best_factor = csv_input_final[["factor"][0]][i]
        # factor_list.append(best_factor)
        y_residual_1.append(min(residual_list_1))
        #     w.writerow({'event_date':date_event, 'event_hour':date_event_hour, 'event_minite':date_event_minute,'velocity':time_rate_final, 'residual':residual_list, 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':best_factor})

        # #     # ファイルの確認コード
        #     path = Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/' + str(date_event)[:4] + '/' + str(date_event) + '*' + str(event_start) + '_' + str(event_end) + '_' + str(freq_start) + '_' + str(freq_end)+ 'compare.png'
        #     File = glob.glob(path, recursive=True)
        #     File1=len(File)
        #     print (File1)
        #     if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/10_/' + str(date_event)[:4]):
        #         os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/10_/' + str(date_event)[:4] )
        #     shutil.copy(File[0],  Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/10_/' + str(date_event)[:4] + '/'+ File[0].split('/')[Parent_lab + 6])
        #     path = Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/' + str(date_event)[:4] + '/' + str(date_event) + '*' + str(event_start) + '_' + str(event_end) + '_' + str(freq_start) + '_' + str(freq_end)+ 'peak.png'
        #     File = glob.glob(path, recursive=True)
        #     shutil.copy(File[0],  Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/10_/'+ str(date_event)[:4] + '/' + File[0].split('/')[Parent_lab + 6])
# #誤差のplot
y_residual_0 = np.array(y_residual_0)
print(len(np.where(y_residual_0<=1.35)[0])/len(y_residual_0))
plt.close(1)
fig = plt.figure(figsize=(20,10),dpi=80)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1.hist(y_residual_0, bins = 200)
ax1.tick_params(axis='x', labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.set_xlabel('Fitting error', fontsize=fontsize)
ax1.set_ylabel('The number of events', fontsize=fontsize)
ax1.set_xlim(0,8)
ax1.axvline(1.35, ls = "--", color = "navy")
ax2.hist(y_residual_1, bins = 200)
ax2.tick_params(axis='x', labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.set_xlabel('Fitting error', fontsize=fontsize)
ax2.set_ylabel('The number of events', fontsize=fontsize)
plt.show()
plt.close()



#ファイルの確認コード
# path = Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/' + str(date_event)[:4] + '/' + str(date_event) + '*' + str(event_start) + '_' + str(event_end) + '_' + str(freq_start) + '_' + str(freq_end)+ 'compare.png'
# File = glob.glob(path, recursive=True)
# File1=len(File)
# print (File1)
# shutil.copy(File[0],  Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/2_3/' + File[0].split('/')[Parent_lab + 6])
# path = Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/' + str(date_event)[:4] + '/' + str(date_event) + '*' + str(event_start) + '_' + str(event_end) + '_' + str(freq_start) + '_' + str(freq_end)+ 'peak.png'
# File = glob.glob(path, recursive=True)
# shutil.copy(File[0],  Parent_directory + '/solar_burst/Nancay/plot/residual_test_1/2_3/' + File[0].split('/')[Parent_lab + 6])




# #過去解析イベントの状況確認
# Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
# file = "/solar_burst/Nancay/analysis_data/velocity_factor_jpgu_pparc11.csv"
# # file = "velocity_factor_jpgu_1.csv"
# #file = "velocity_factor1.csv"
# csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file, sep=",")

# factor_list = []
# y_residual_0 = []
# y_residual_1 = []
# file_name = []

# for i in range(len(csv_input_final)):
#     velocity = csv_input_final[["velocity"][0]][i].lstrip("['")[:-1].split(',')
#     residual = csv_input_final[["residual"][0]][i].lstrip("['")[:-1].split(',')
#     velocity_list = [float(s) for s in velocity]
#     residual_list_1 = np.array([float(s) for s in residual])
#     freq_start = csv_input_final['freq_start'][i]
#     freq_end = csv_input_final['freq_end'][i]
#     freq_range = ((freq_start - freq_end)/0.175) + 1
#     residual_list_1 = np.sqrt(residual_list_1/freq_range)
#     y_residual_0.append(min(residual_list_1))
#     if min(residual_list_1)  < 0.2:
#     #     # if min(residual_list_1) > 0.2:
#     #         # print('b')
#         date_event = csv_input_final['event_date'][i]
#         date_event_hour = csv_input_final['event_hour'][i]
#         date_event_minute = csv_input_final['event_minite'][i]
#         time_rate_final = csv_input_final['velocity'][i]
#         residual_list = min(residual_list_1)
#         event_start = csv_input_final['event_start'][i]
#         event_end = csv_input_final['event_end'][i]
#         # freq_start = csv_input_final['freq_start'][i]
#         # freq_end = csv_input_final['freq_end'][i]
#         best_factor = csv_input_final[["factor"][0]][i]
#         # factor_list.append(best_factor)
#         y_residual_1.append(min(residual_list_1))
# plt.close(1)
# fig = plt.figure(figsize=(20,10),dpi=80)
# ax1 = fig.add_subplot(1, 2, 1)
# ax2 = fig.add_subplot(1, 2, 2)
# ax1.hist(y_residual_0, bins = 200)
# ax1.tick_params(axis='x', labelsize=labelsize)
# ax1.tick_params(axis='y', labelsize=labelsize)
# ax1.set_xlabel('Fitting error', fontsize=fontsize)
# ax1.set_ylabel('The number of events', fontsize=fontsize)
# ax2.hist(y_residual_1, bins = 200)
# ax2.tick_params(axis='x', labelsize=labelsize)
# ax2.tick_params(axis='y', labelsize=labelsize)
# ax2.set_xlabel('Fitting error', fontsize=fontsize)
# ax2.set_ylabel('The number of events', fontsize=fontsize)
# plt.show()