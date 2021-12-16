#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:12:01 2021

@author: yuichiro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pynverse import inversefunc
from dateutil import relativedelta
from matplotlib import dates as mdates
from datetime import date
import datetime
from dateutil.relativedelta import relativedelta
import sys
import glob


file_final = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_micro_withnonclear.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")

freq_drift_micro = []
obs_time_micro = []

for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    obs_time_micro.append(obs_time)
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    freq_drift_micro.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')]
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')]
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]


freq_drift_micro = np.array(freq_drift_micro)
obs_time_micro = np.array(obs_time_micro)


micro_solar_max_idx = np.where((obs_time_micro >= datetime.datetime(2012,1,1)) & (obs_time_micro <= datetime.datetime(2014,12,31,23)))[0]
micro_solar_min_idx = np.where(((obs_time_micro >= datetime.datetime(2007,1,1)) & (obs_time_micro <= datetime.datetime(2009,12,31,23))) | ((obs_time_micro >= datetime.datetime(2017,1,1)) & (obs_time_micro <= datetime.datetime(2020,12,31,23))))[0]




file_final = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_ordinary_withnonclear.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")

freq_drift_ordinary = []
obs_time_ordinary = []

for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    obs_time_ordinary.append(obs_time)
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    freq_drift_ordinary.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')]
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')]
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]


freq_drift_ordinary = np.array(freq_drift_ordinary)
obs_time_ordinary = np.array(obs_time_ordinary)
ordinary_solar_max_idx = np.where((obs_time_ordinary >= datetime.datetime(2012,1,1)) & (obs_time_ordinary <= datetime.datetime(2014,12,31,23)))[0]
ordinary_solar_min_idx = np.where(((obs_time_ordinary >= datetime.datetime(2007,1,1)) & (obs_time_ordinary <= datetime.datetime(2009,12,31,23))) | ((obs_time_ordinary >= datetime.datetime(2017,1,1)) & (obs_time_ordinary <= datetime.datetime(2020,12,31,23))) | ((obs_time_ordinary >= datetime.datetime(1995,1,1)) & (obs_time_ordinary <= datetime.datetime(1997,12,31,23))))[0]

freq_check = 40

#event_study
date_in = ['201802111141']
# target = '20040907'
# burst = 'Micro type Ⅲ burst'
burst = 'Ordinary type Ⅲ burst'
if burst == 'Micro type Ⅲ burst':
    start, end = date_in
    micro_idx = np.where((obs_time_micro >= datetime.datetime(int(start[:4]),int(start[4:6]),int(start[6:8]))) & (obs_time_micro <= datetime.datetime(int(end[:4]),int(end[4:6]),int(end[6:8]),23,0)))[0]
    micro_idx2 = np.where((obs_time_micro >= datetime.datetime(int(target[:4]),int(target[4:6]),int(target[6:8]))) & (obs_time_micro <= datetime.datetime(int(target[:4]),int(target[4:6]),int(target[6:8]),23,0)))[0]


    fig=plt.figure(1,figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(obs_time_micro[micro_idx],freq_drift_micro[micro_idx] , '.')
    
    # format your data to desired format. Here I chose YYYY-MM-DD but you can set it to whatever you want.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    plt.xticks(rotation=20)
    plt.title(burst + '\n' + start + ' - ' + end, fontsize = 15)
    plt.ylabel('Frequency drift rates[MHz/s]', fontsize = 15)
    plt.xlabel('Time (UT)', fontsize = 15)
    plt.ylim(1.3,16.2)
    plt.show()
    
    
    text = ''
    if burst == 'Micro type Ⅲ burst':
        freq_drift_solar_maximum = freq_drift_micro[micro_solar_max_idx]
        color_1 = "r"
        freq_drift_solar_minimum = freq_drift_micro[micro_solar_min_idx]
        color_2 = "b"
    
    elif burst == 'Ordinary type Ⅲ burst':
        freq_drift_solar_maximum = freq_drift_ordinary[ordinary_solar_max_idx]
        color_1 = "orange"
        freq_drift_solar_minimum = freq_drift_ordinary[ordinary_solar_min_idx]
        color_2 = "deepskyblue"
    
    if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
        max_val = max(freq_drift_solar_minimum)
    else:
        max_val = max(freq_drift_solar_maximum)
    
    if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
        min_val = min(freq_drift_solar_minimum)
    else:
        min_val = min(freq_drift_solar_maximum)
    bin_size = 11
    
    # x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    # y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
    # x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    # y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
    
    x_hist = (plt.hist(freq_drift_micro[micro_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    y_hist = (plt.hist(freq_drift_micro[micro_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_micro[micro_idx2]))
    # x_hist_1 = (plt.hist(freq_drift_micro[micro_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    # y_hist_1 = (plt.hist(freq_drift_micro[micro_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_micro[micro_idx2]))
    plt.close()
    width = x_hist[1]-x_hist[0]
    for i in range(len(y_hist)):
        if i == 0:
            # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
            # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
            # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
        else:
            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3)
            # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
        plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst + text + '\n' + target[:4] + '/'+target[4:6]+ '/'+target[6:8])
    # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
        plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
        plt.ylabel('Occurrence rate',fontsize=15)
        # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
        plt.xticks(rotation = 20)
    plt.show()
    plt.close()
    
    print ('イベント数:' + str(len(freq_drift_micro[micro_idx2])))
    print ('平均値:' + str(round(np.nanmean(freq_drift_micro[micro_idx2]),1)))
    print ('標準偏差:' + str(round(np.nanstd(freq_drift_micro[micro_idx2]),2)))
    print ('最小値:' + str(round(np.nanmin(freq_drift_micro[micro_idx2]),1)))
    print ('最大値:' + str(round(np.nanmax(freq_drift_micro[micro_idx2]),1)))

if burst == 'Ordinary type Ⅲ burst':
    start = date_in[0]
    ordinary_idx = np.where((obs_time_ordinary >= datetime.datetime(int(start[:4]),int(start[4:6]),int(start[6:8]))) & (obs_time_ordinary <= datetime.datetime(int(start[:4]),int(start[4:6]),int(start[6:8]),23,0)))[0]
    ordinary_idx2 = np.where((obs_time_ordinary >= datetime.datetime(int(start[:4]),int(start[4:6]),int(start[6:8]), int(start[8:10]), int(start[10:12])) - datetime.timedelta(minutes =10))  & (obs_time_ordinary <= datetime.datetime(int(start[:4]),int(start[4:6]),int(start[6:8]), int(start[8:10]), int(start[10:12])) + datetime.timedelta(minutes =10)))[0]
    


    fig=plt.figure(1,figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.plot(obs_time_ordinary[ordinary_idx],freq_drift_ordinary[ordinary_idx] , '.')
    
    # format your data to desired format. Here I chose YYYY-MM-DD but you can set it to whatever you want.
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # rotate and align the tick labels so they look better
    fig.autofmt_xdate()
    plt.xticks(rotation=20)
    plt.title(burst + '\n' + start, fontsize = 15)
    plt.ylabel('Frequency drift rates[MHz/s]', fontsize = 15)
    plt.xlabel('Time (UT)', fontsize = 15)
    plt.ylim(1.3,16.2)
    plt.show()
    
    
    text = ''
    if burst == 'Micro type Ⅲ burst':
        freq_drift_solar_maximum = freq_drift_micro[micro_solar_max_idx]
        color_1 = "r"
        freq_drift_solar_minimum = freq_drift_micro[micro_solar_min_idx]
        color_2 = "b"
    
    elif burst == 'Ordinary type Ⅲ burst':
        freq_drift_solar_maximum = freq_drift_ordinary[ordinary_solar_max_idx]
        color_1 = "orange"
        freq_drift_solar_minimum = freq_drift_ordinary[ordinary_solar_min_idx]
        color_2 = "deepskyblue"
    
    if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
        max_val = max(freq_drift_solar_minimum)
    else:
        max_val = max(freq_drift_solar_maximum)
    
    if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
        min_val = min(freq_drift_solar_minimum)
    else:
        min_val = min(freq_drift_solar_maximum)
    bin_size = 11
    
    # x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    # y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
    # x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    # y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
    
    x_hist = (plt.hist(freq_drift_ordinary[ordinary_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    y_hist = (plt.hist(freq_drift_ordinary[ordinary_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_ordinary[ordinary_idx2]))
    x_hist_1 = (plt.hist(freq_drift_ordinary[ordinary_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
    y_hist_1 = (plt.hist(freq_drift_ordinary[ordinary_idx2], bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_ordinary[ordinary_idx2]))
    plt.close()
    width = x_hist[1]-x_hist[0]
    for i in range(len(y_hist)):
        if i == 0:
            # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
            # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_ordinary[ordinary_idx2])) + ' events')
            # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_ordinary[ordinary_idx2])) + ' events')
        else:
            # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3)
            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
        plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst + text + '\n' + start[:4] + '/'+start[4:6]+ '/'+start[6:8] + ' ' + start[8:10] + ':' + start[10:12])
    # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
        plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
        plt.ylabel('Occurrence rate',fontsize=15)
        # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
        plt.xticks(rotation = 20)
    plt.show()
    plt.close()
    
    print ('イベント数:' + str(len(freq_drift_ordinary[ordinary_idx2])))
    print ('平均値:' + str(round(np.nanmean(freq_drift_ordinary[ordinary_idx2]),1)))
    print ('標準偏差:' + str(round(np.nanstd(freq_drift_ordinary[ordinary_idx2]),2)))
    print ('最小値:' + str(round(np.nanmin(freq_drift_ordinary[ordinary_idx2]),1)))
    print ('最大値:' + str(round(np.nanmax(freq_drift_ordinary[ordinary_idx2]),1)))