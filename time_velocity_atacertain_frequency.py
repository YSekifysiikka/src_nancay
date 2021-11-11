#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 12:51:44 2021

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
            velocity_real = []
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
                    y = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5))))
                    cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
                    invcube_3 = inversefunc(cube_3, y_values=freq[i])
                    
                    slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
                    y_slope = slope * (t - invcube_3) + freq[i]
                    freq_drift.append(-slope)
            
                    cube_5 = (lambda h: 9 * 10 * np.sqrt(factor_velocity * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
                    invcube_5 = inversefunc(cube_5, y_values = freq[i])
                    h_radio = invcube_5 + 69600000000
                    velocity_real.append(2/numerical_diff_allen_velocity(factor_velocity, h_radio)/freq[i]*slope/29979245800)
            
                
                plt.close()
                x_range.append(plt.hist(velocity_real, bins = 20, range = (0,1), density= None)[1])
                y_range.append(plt.hist(velocity_real, bins = 20, range = (0,1), density= None)[0]/len(velocity_real))
                mean_val.append(round(np.mean(velocity_real), 3))
                std_val.append(round(np.std(velocity_real), 4))
                print (len(velocity_real))
                plt.close()
                    # plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)
                
                
                label = []
                for i in range(x_range[0].shape[0] - 1):
                    label.append(str('{:.02f}'.format(round(x_range[0][i], 3))) + '-' + str('{:.02f}'.format(round(x_range[0][i+1], 3))))
                
                
                width = 0.022
                x_range = np.array(x_range) - (width/2)
                for i in range(len(y_range)):
                    plt.bar(x_range[i][:20] + i * width, y_range[i], width= width, label = str(sdate)[:4]+'/'+ str(sdate)[4:6]+ ' ARV: ' + str('{:.03f}'.format(mean_val[i])) + 'c' + ' SD: ' + str('{:.04f}'.format(std_val[i])) + 'c', color = color_list[i])
                    # plt.title(str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' velocity')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
                    plt.xlabel('Velocity[c]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    plt.xticks([0,0.2,0.4,0.6, 0.8], ['0.00 - 0.05','0.20 - 0.25','0.40 - 0.45', '0.60 - 0.65', '0.80 - 0.85']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
            

            
        except:
            print ('Plot error: ' + SDATE)
        SDATE+= relativedelta.relativedelta(months=move_ave)
    return obs_time, obs_velocity, repeat_num
        

    
    
    
    
    
    
    
    
date_in=[20120101, 20210101]

if __name__=='__main__':
    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')
    
    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time, obs_velocity, repeat_num = monthly_analysis(DATE, FDATE)
    except:
        print('DL error: ',DATE)

sobs_time = []
eobs_time = []
mobs_time = []
month_velocity_mean = []
month_velocity_std = []
for i in range(40):
    num = 0
    month_velocities = []
    for j in range(len(obs_time)):
        if obs_time[j] >= date(int(str(date_in[0])[:4]), 1, 1)+relativedelta.relativedelta(months=move_ave*i) and obs_time[j] <= date(int(str(date_in[0])[:4]), 1, 1)+ relativedelta.relativedelta(months=move_ave*(i+1)) - relativedelta.relativedelta(days=1):
            month_velocities.append(obs_velocity[j])
            num += 1
    if num >= 1:
        sobs_time.append(date(int(str(date_in[0])[:4]), 1, 1) + relativedelta.relativedelta(months=move_ave*i))
        eobs_time.append(date(int(str(date_in[0])[:4]), 1, 1) + relativedelta.relativedelta(months=move_ave*(i+1)) - relativedelta.relativedelta(days=1))
        month_velocity_mean.append(np.mean(month_velocities))
        month_velocity_std.append(np.std(month_velocities))
        mobs_time.append(date(int(str(date_in[0])[:4]), 1, 15) + relativedelta.relativedelta(months=move_ave*i))


fig, ax = plt.subplots(figsize=(14, 6))

for i in range (len(sobs_time)):
    # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_velocity_std[i], capsize=3, color = 'r')
    ax.plot([sobs_time[i], eobs_time[i]], [month_velocity_mean[i], month_velocity_mean[i]], color = 'r')
for i in range (len(sobs_time)):
    if eobs_time[i] + relativedelta.relativedelta(days=1) in sobs_time:
        ax.plot([eobs_time[i], sobs_time[i+1]], [month_velocity_mean[i], month_velocity_mean[i+1]], color = 'r')


ax.scatter(obs_time, obs_velocity)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=None, interval=6, tz=None))
ax.set_xlim(datetime.datetime(int(str(date_in[0])[:4]), 1, 1, 00), datetime.datetime(int(str(date_in[1])[:4])-1, 12, 31, 23))
# ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
plt.xticks(rotation=90)
plt.ylim(0,0.4)

plt.show()