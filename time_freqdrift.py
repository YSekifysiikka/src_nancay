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
from scipy.stats import norm
move_ave = 12
freq_check = 35


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
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

if file_final.split('_')[4].split('/')[1] == 'original':
    obs_burst = 'Ordinary type Ⅲ bursts'
else:
    obs_burst = 'Micro-type Ⅲ bursts'

# function to fit
def func(f, a, b):
    return a * (f ** b)

def monthly_analysis(DATE, FDATE):
    date_start = date(DATE.year, DATE.month, DATE.day)
    FDATE = date(FDATE.year, FDATE.month, FDATE.day)
    SDATE = date_start
    # print (SDATE)
    obs_time = []
    obs_velocity = []
    freq_drift_final = []
    repeat_num = 0
    while SDATE < FDATE:
        try:

            # print (SDATE)
            sdate = int(str(SDATE.year)+str(SDATE.month).zfill(2)+str(SDATE.day).zfill(2))
            EDATE = SDATE + relativedelta.relativedelta(months=move_ave) - relativedelta.relativedelta(days=1)
            edate = int(str(EDATE.year)+str(EDATE.month).zfill(2)+str(EDATE.day).zfill(2))
            # year_list_1.append(sdate)
            # year_list_2.append(edate)
            print (sdate, '-', edate)
            
            month_num = []
            month_num.append(np.count_nonzero((csv_input_final['event_date'] >= sdate) & (csv_input_final['event_date']<= edate)))
            # print(month_num)
            x_range = []
            y_range = []
            mean_val = []
            std_val = []
        
            y_factor = []
            y_velocity = []
            freq_drift = []
            freq = []
            start_check = []
            time_gap = []
            separate_num = 1
        
            for j in range(len(csv_input_final)):
                if csv_input_final['event_date'][j] >= sdate and csv_input_final['event_date'][j] <= edate:
                    if csv_input_final[["freq_start"][0]][j] >= freq_check:
                        if csv_input_final[["freq_end"][0]][j] <= freq_check:
                            # print (csv_input_final['event_date'][j])
                            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
                            best_factor = 2
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
                                obs_velocity.append(float(velocity[best_factor-1]))
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
                    # freq_drift_list.append()
                    # velocity_real.append(2/numerical_diff_allen_velocity(factor_velocity, h_radio)/freq[i]*slope/29979245800)

                plt.close()
                x_range.append(plt.hist(freq_drift, bins = 20, range = (0,20), density= None)[1])
                
                y_range.append(plt.hist(freq_drift, bins = 20, range = (0,20), density= None)[0]/len(freq_drift))
                mean_val.append(round(np.mean(freq_drift), 1))
                std_val.append(round(np.std(freq_drift), 1))
                print (len(freq_drift))
                plt.close()
                    # plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)
                
                
                label = []
                for i in range(x_range[0].shape[0] - 1):
                    label.append(str('{:.00f}'.format(abs(round(x_range[0][i], 0)))) + '-' + str('{:.00f}'.format(round(x_range[0][i+1], 0))))
                
                
                width = 0.5
                x_range = np.array(x_range)
                for i in range(len(y_range)):
                    plt.bar(x_range[i][:20] + i * width, y_range[i], width= width, label = str(sdate)[:4]+'/'+ str(sdate)[4:6]+ ' ARV: ' + str('{:.01f}'.format(mean_val[i])) + ' SD: ' + str('{:.01f}'.format(std_val[i]) + '\nEvent Number: ' + str(len(freq_drift))), color = color_list[i])
                    # plt.title(str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' velocity')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
                    plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)

                param = norm.fit(freq_drift)
                print (param)
                x = np.linspace(min(x_range) - 2,max(x_range) + 2,100)
                pdf_fitted = norm.pdf(x,loc=param[0], scale=param[1])
                # pdf = norm.pdf(x)
                plt.plot(x, pdf_fitted, 'r-')
                plt.show()
                plt.close()
            

            
        except:
            print ('Plot error: ' + SDATE)
        SDATE+= relativedelta.relativedelta(months=move_ave)
    return obs_time, freq_drift_final, repeat_num

    
    
    
    
date_in=[20120101, 20210101]

if __name__=='__main__':
    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')
    
    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time, freq_drift_final, repeat_num = monthly_analysis(DATE, FDATE)
    except:
        print('DL error: ',DATE)

sobs_time = []
eobs_time = []
mobs_time = []
month_frequency_mean = []
month_frequency_std = []
for i in range(40):
    num = 0
    month_velocities = []
    for j in range(len(obs_time)):
        if obs_time[j] >= date(int(str(date_in[0])[:4]), 1, 1)+relativedelta.relativedelta(months=move_ave*i) and obs_time[j] <= date(int(str(date_in[0])[:4]), 1, 1)+ relativedelta.relativedelta(months=move_ave*(i+1)) - relativedelta.relativedelta(days=1):
            month_velocities.append(freq_drift_final[j])
            num += 1
    if num >= 1:
        sobs_time.append(date(int(str(date_in[0])[:4]), 1, 1) + relativedelta.relativedelta(months=move_ave*i))
        eobs_time.append(date(int(str(date_in[0])[:4]), 1, 1) + relativedelta.relativedelta(months=move_ave*(i+1)) - relativedelta.relativedelta(days=1))
        month_frequency_mean.append(np.mean(month_velocities))
        month_frequency_std.append(np.std(month_velocities))
        # mobs_time.append(date(int(str(date_in[0])[:4]), 1, 15) + relativedelta.relativedelta(months=move_ave*i))


fig, ax = plt.subplots(figsize=(14, 6))
ax.scatter(obs_time, freq_drift_final, label = 'f = ' + str(freq_check) + '[MHz]')

for i in range (len(sobs_time)):
    # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
    if i == 0:
        ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(move_ave) + ' months average')
    else:
        ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r')
    
for i in range (len(sobs_time)):
    if eobs_time[i] + relativedelta.relativedelta(days=1) in sobs_time:
        ax.plot([eobs_time[i], sobs_time[i+1]], [month_frequency_mean[i], month_frequency_mean[i+1]], color = 'r')



ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=None, interval=6, tz=None))
ax.set_xlim(datetime.datetime(int(str(date_in[0])[:4]), 1, 1, 00), datetime.datetime(int(str(date_in[1])[:4])-1, 12, 31, 23))

# ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
plt.title(obs_burst, fontsize = 20)
plt.xticks(rotation=90)
plt.ylim(0,20)
plt.ylabel('Frequency drift rates[MHz/s]',fontsize=21)
plt.legend(bbox_to_anchor=(1.02, 1.3), loc='upper right', borderaxespad=1, fontsize = 20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.show()
plt.close()


x_hist = (plt.hist(freq_drift_final, bins = 15, range = (min(freq_drift_final)-1,max(freq_drift_final)+1), density= None)[1])

y_hist = (plt.hist(freq_drift_final, bins = 15, range = (min(freq_drift_final)-1,max(freq_drift_final)+1), density= None)[0]/len(freq_drift_final))
plt.close()
width = 1.24
for i in range(len(y_hist)):
    plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b')
    # plt.title(str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' velocity')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
    plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()


