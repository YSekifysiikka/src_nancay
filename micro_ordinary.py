#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:37:00 2021

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
from pynverse import inversefunc
import scipy
from dateutil import relativedelta
from datetime import date,timedelta
import pandas.tseries.offsets as offsets
from matplotlib import dates as mdates
import matplotlib.ticker as ticker
import scipy
freq_check = 35
move_ave = 12
move_plot = 4

def numerical_diff_allen_velocity(factor, r):
    h = 1e-2
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))
def numerical_diff_allen(factor, velocity, t, h_start):
    h = 1e-3
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))

def func(x, a, b):
    return a * x + b

labelsize = 18
fontsize = 20
factor_velocity = 2
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1




obs_burst = 'Ordinary type Ⅲ bursts'
obs_burst_1 = 'Micro-type Ⅲ bursts'




def analysis_bursts(DATE, FDATE, csv_input_final):
    date_start = date(DATE.year, DATE.month, DATE.day)
    FDATE = date(FDATE.year, FDATE.month, FDATE.day)
    SDATE = date_start
    # print (SDATE)
    obs_time = []
    freq_drift_final = []
    start_check_list = []
    end_check_list = []
    duration_list = []
    repeat_num = 0
    while SDATE < FDATE:
        try:
            sdate = int(str(SDATE.year)+str(SDATE.month).zfill(2)+str(SDATE.day).zfill(2))
            EDATE = SDATE + relativedelta.relativedelta(months=move_ave) - relativedelta.relativedelta(days=1)
            edate = int(str(EDATE.year)+str(EDATE.month).zfill(2)+str(EDATE.day).zfill(2))
            print (sdate, '-', edate)
            

            y_factor = []
            y_velocity = []
            freq_drift = []
            freq = []
            start_check = []
            time_gap = []
            separate_num = 1
        
            for j in range(len(csv_input_final)):
                if csv_input_final['event_date'][j] >= sdate and csv_input_final['event_date'][j] <= edate:
                    start_check_list.append(csv_input_final[["freq_start"][0]][j])
                    end_check_list.append(csv_input_final[["freq_end"][0]][j])
                    duration_list.append(csv_input_final[["event_end"][0]][j]-csv_input_final[["event_start"][0]][j]+1)
                    if csv_input_final[["freq_start"][0]][j] >= freq_check:
                        if csv_input_final[["freq_end"][0]][j] <= freq_check:
                            # print (csv_input_final['event_date'][j])
                            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
                            best_factor = csv_input_final["factor"][j]
                            for z in range(separate_num):
                                year = int(str(csv_input_final['event_date'][j])[0:4])
                                if str(csv_input_final['event_date'][j])[4:5] == '0':
                                    month = int(str(csv_input_final['event_date'][j])[5:6])
                                else:
                                    month = int(str(csv_input_final['event_date'][j])[4:6])
                                if str(csv_input_final['event_date'][j])[6:7] == '0':
                                    day = int(str(csv_input_final['event_date'][j])[7:8])
                                else:
                                    day = int(str(csv_input_final['event_date'][j])[6:8])
                                obs_date = date(year, month, day)
                                    
                                    
                                obs_time.append(obs_date)
                                freq.append(freq_check)
                                start_check.append(csv_input_final[["freq_start"][0]][j])
                                y_factor.append(best_factor)
                                y_velocity.append(float(velocity[best_factor-1]))
                                time_gap.append(csv_input_final[["event_end"][0]][j] - csv_input_final[["event_start"][0]][j] + 1)
        
            
            t = np.arange(0, 2000, 1)
            t = (t+1)/100
            if len(y_factor) > 0:
                repeat_num += 1
                for i in range (len(y_factor)):
                    factor = y_factor[i]
                    velocity = y_velocity[i]
                    freq_max = start_check[i]
                    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
                    invcube_4 = inversefunc(cube_4, y_values = freq_max)
                    h_start = invcube_4/69600000000 + 1
                    # print (h_start)
                    cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
                    invcube_3 = inversefunc(cube_3, y_values=freq[i])
                    
                    slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
                    freq_drift.append(-slope)
                    freq_drift_final.append(-slope)


            
        except:
            print ('Plot error: ' + SDATE)
        SDATE+= relativedelta.relativedelta(months=move_ave)
    return obs_time, freq_drift_final, repeat_num, start_check_list, end_check_list, duration_list
        



    
date_in=[20120101, 20210101]

if __name__=='__main__':
    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')
    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    
    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time_ordinary, freq_drift_ordinary, repeat_num, freq_start_ordinary, freq_end_ordinary, duration_ordinary = analysis_bursts(DATE, FDATE, csv_input_final)
    except:
        print('DL error: ',DATE)


    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')
    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/storm_burst_cycle24.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    
    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time_micro, freq_drift_micro, repeat_num, freq_start_micro, freq_end_micro, duration_micro = analysis_bursts(DATE, FDATE, csv_input_final)
    except:
        print('DL error: ',DATE)




if max(freq_drift_micro) >= max(freq_drift_ordinary):
    max_val = max(freq_drift_micro)
else:
    max_val = max(freq_drift_ordinary)

if min(freq_drift_micro) <= min(freq_drift_ordinary):
    min_val = min(freq_drift_micro)
else:
    min_val = min(freq_drift_ordinary)
bin_size = 25

x_hist = (plt.hist(freq_drift_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
y_hist = (plt.hist(freq_drift_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_ordinary))
x_hist_1 = (plt.hist(freq_drift_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
y_hist_1 = (plt.hist(freq_drift_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_micro))
plt.close()
width = x_hist[1]-x_hist[0]
for i in range(len(y_hist)):
    if i == 0:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(freq_drift_ordinary)) + 'events')
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(freq_drift_micro)) + 'events')
    else:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
    plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 14)
    plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()



##############################################
#開始周波数
bin_size = 19
#(79.825-29.95)/2.625


x_hist = (plt.hist(freq_start_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[1])
y_hist = (plt.hist(freq_start_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_start_ordinary))
x_hist_1 = (plt.hist(freq_start_micro, bins = bin_size, range = (29.95, 79.825), density= None)[1])
y_hist_1 = (plt.hist(freq_start_micro, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_start_micro))
plt.close()
width = x_hist[1]-x_hist[0]
for i in range(len(y_hist)):
    if i == 0:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(freq_start_ordinary)) + 'events')
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(freq_start_micro)) + 'events')
    else:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
    # plt.title('Start Frequency')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
    plt.xlabel('Start Frequency[MHz]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()

##############################################
#終了周波数
bin_size = 19


x_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[1])
y_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_end_ordinary))
x_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.95, 79.825), density= None)[1])
y_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_end_micro))
plt.close()
width = x_hist[1]-x_hist[0]
for i in range(len(y_hist)):
    if i == 0:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(freq_end_ordinary)) + 'events')
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(freq_end_micro)) + 'events')
    else:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
    # plt.title('Start Frequency')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
    plt.xlabel('End Frequency[MHz]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()


#終了周波数2
bin_size = 19


x_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.96, 79.825), density= None)[1])
y_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.96, 79.825), density= None)[0]/len(freq_end_ordinary))
x_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.96, 79.825), density= None)[1])
y_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.96, 79.825), density= None)[0]/len(freq_end_micro))
plt.close()
width = x_hist[1]-x_hist[0]
for i in range(len(y_hist)):
    if i == 0:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(freq_end_ordinary)) + 'events')
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(freq_end_micro)) + 'events')
    else:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
    # plt.title('Start Frequency')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
    plt.xlabel('End Frequency[MHz]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()



##############################################
#継続時間
if max(duration_micro) >= max(duration_ordinary):
    max_val = max(duration_micro)
else:
    max_val = max(duration_ordinary)

if min(duration_micro) <= min(duration_ordinary):
    min_val = min(duration_micro)
else:
    min_val = min(duration_ordinary)
bin_size = int((max_val-min_val)/2)


x_hist = (plt.hist(duration_ordinary, bins = bin_size, range = (min_val,max_val), density= None)[1])
y_hist = (plt.hist(duration_ordinary, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_ordinary))
x_hist_1 = (plt.hist(duration_micro, bins = bin_size, range = (min_val,max_val), density= None)[1])
y_hist_1 = (plt.hist(duration_micro, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_micro))
plt.close()
width = x_hist[1]-x_hist[0]
for i in range(len(y_hist)):
    if i == 0:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(duration_ordinary)) + 'events')
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(duration_micro)) + 'events')
    else:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
    # plt.title('Start Frequency')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
    plt.xlabel('Duration[sec]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()
