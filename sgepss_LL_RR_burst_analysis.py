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
freq_check = 40
move_ave = 12
move_ave_analysis = 12
move_plot = 4
#赤の線の変動を調べる
analysis_move_ave = 12
average_threshold = 1
error_threshold = 1
solar_maximum = [datetime.datetime(2000, 1, 1), datetime.datetime(2003, 1, 1)]
solar_minimum = [datetime.datetime(2007, 1, 1), datetime.datetime(2010, 1, 1)]
solar_maximum_1 = [datetime.datetime(2012, 1, 1), datetime.datetime(2015, 1, 1)]
solar_minimum_1 = [datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 1)]
analysis_period = [solar_maximum, solar_minimum, solar_maximum_1, solar_minimum_1]

def getNearestValue(list, num):

    # 昇順に挿入する際のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]

def numerical_diff_allen_velocity_fp(factor, r):
    h = 1e-2
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_newkirk_velocity_fp(factor, r):
    h = 1e-2
    
    ne_1 = np.log(factor * 4.2 * 10 ** (4+4.32/((r+h)/69600000000)))
    ne_2 = np.log(factor * 4.2 * 10 ** (4+4.32/((r-h)/69600000000)))
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_allen_velocity_2fp(factor, r):
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
factor_velocity = 1
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1





def analysis_bursts(DATE, FDATE, csv_input_final, burst_type):
    date_start = date(DATE.year, DATE.month, DATE.day)
    FDATE = date(FDATE.year, FDATE.month, FDATE.day)
    SDATE = date_start
    # print (SDATE)
    obs_time = []
    freq_drift_final = []
    start_check_list = []
    end_check_list = []
    duration_list = []
    intensity_list = []
    cos_list = []
    pol_list = []
    sunspot_list = []
    velocity_fp_list = []
    velocity_2fp_list = []
    while SDATE < FDATE:
        try:
            sdate = int(str(SDATE.year)+str(SDATE.month).zfill(2)+str(SDATE.day).zfill(2))
            EDATE = SDATE + relativedelta(months=move_ave) - relativedelta(days=1)
            edate = int(str(EDATE.year)+str(EDATE.month).zfill(2)+str(EDATE.day).zfill(2))
            print (sdate, '-', edate)
        
            for j in range(len(csv_input_final)):
                obs_date = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
                if int(obs_date.strftime('%Y%m%d')) >= sdate and int(obs_date.strftime('%Y%m%d')) <= edate:
                    start_check_list.append(csv_input_final["freq_start"][j])
                    end_check_list.append(csv_input_final["freq_end"][j])
                    duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
                    obs_date = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
                    obs_time.append(obs_date)
                    freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
                    intensity_list.append(csv_input_final["peak_intensity_calibrated"][j])
                    cos_list.append(csv_input_final["cos"][j])
                    pol_list.append(csv_input_final["Polarization"][j])
                    sunspot_list.append(csv_input_final["sunspots_num"][j])
                    if burst_type == 'ordinary':
                        if ~np.isnan(csv_input_final["drift_rate_40MHz"][j]):
                            #極小期
                            if (((analysis_period[1][0] <= obs_date) & (analysis_period[1][1] >= obs_date)) | ((analysis_period[3][0] <= obs_date) & (analysis_period[3][1] >= obs_date))):
                                factor_velocity = 1
                                freq = csv_input_final["40MHz"][j]
                                # fp_cube = (lambda h: 9 * 10 * np.sqrt(factor_velocity * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
                                fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
                                h_radio = inversefunc(fp_cube, y_values = freq) + 69600000000
                                # velocity_fp_list.append(2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                velocity_fp_list.append(2/numerical_diff_newkirk_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                # print (2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                
                                s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
                                e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
                                velocity_2fp_list.append((e_radio-s_radio)/0.02/29979245800)
                            #極大期
                            elif (((analysis_period[0][0] <= obs_date) & (analysis_period[0][1] >= obs_date)) | ((analysis_period[2][0] <= obs_date) & (analysis_period[2][1] >= obs_date))):
                                factor_velocity = 1
                                freq = csv_input_final["40MHz"][j]
                                # fp_cube = (lambda h: 9 * 10 * np.sqrt(factor_velocity * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
                                fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
                                h_radio = inversefunc(fp_cube, y_values = freq) + 69600000000
                                # velocity_fp_list.append(2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                velocity_fp_list.append(2/numerical_diff_newkirk_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                # print (2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                
                                s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
                                e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
                                velocity_2fp_list.append((e_radio-s_radio)/0.02/29979245800)
                        else:
                            velocity_fp_list.append(np.nan)
                            velocity_2fp_list.append(np.nan)
                    else:
    # h2_1 = np.arange(1, 4, 0.1)
    # allen_model = factor * 10**8 * (2.99*(h2_1)**(-16)+1.55*(h2_1)**(-6)+0.036*(h2_1)**(-1.5))
    # x2_1 = allen_model
    # ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×B-A model')
    # h2_1 = np.arange(1, 4, 0.1)
    # wang_min = 353766/h2_1 + 1.03359e+07/(h2_1)**2 - 5.46541e+07/(h2_1)**3 + 8.24791e+07/(h2_1)**4
    # x2_1 = wang_min 
    # ln1 = ax1.plot(h2_1, x2_1, label = 'Wang model(solar min)')

    # h2_1 = np.arange(1, 4, 0.1)
    # wang_max = -4.42158e+06/h2_1 + 5.41656e+07/(h2_1)**2 - 1.86150e+08 /(h2_1)**3 + 2.13102e+08/(h2_1)**4
    # x2_1 = wang_max
    # ln1 = ax1.plot(h2_1, x2_1, label = 'Wang model(solar max)')
                        if ~np.isnan(csv_input_final["drift_rate_40MHz"][j]):
                            #極小期
                            if (((analysis_period[1][0] <= obs_date) & (analysis_period[1][1] >= obs_date)) | ((analysis_period[3][0] <= obs_date) & (analysis_period[3][1] >= obs_date))):
                                freq = csv_input_final["40MHz"][j]
                                fp_cube = (lambda h: 9 * 1e-3 * np.sqrt((353766*((1+(h/69600000000))**(-1))+1.03359e+07*((1+(h/69600000000))**(-2))- 5.46541e+07*((1+(h/69600000000))**(-3))+ 8.24791e+07*((1+(h/69600000000))**(-4)))))
                                s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)
                                e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)
                                s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
                                e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
                                velocity_fp_list.append((e_radio-s_radio)/0.02/29979245800)
                                # print (2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                
                                s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
                                e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
                                velocity_2fp_list.append((e_radio-s_radio)/0.02/29979245800)
                            #極大期
                            elif (((analysis_period[0][0] <= obs_date) & (analysis_period[0][1] >= obs_date)) | ((analysis_period[2][0] <= obs_date) & (analysis_period[2][1] >= obs_date))):
                                freq = csv_input_final["40MHz"][j]
                                fp_cube = (lambda h: 9 * 1e-3 * np.sqrt((-4.42158e+06*((1+(h/69600000000))**(-1))+5.41656e+07*((1+(h/69600000000))**(-2))- 1.86150e+08*((1+(h/69600000000))**(-3))+ 2.13102e+08*((1+(h/69600000000))**(-4)))))
                                s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)
                                e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)
                                s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
                                e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
                                velocity_fp_list.append((e_radio-s_radio)/0.02/29979245800)
                                # print (2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                                
                                s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
                                s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
                                e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
                                velocity_2fp_list.append((e_radio-s_radio)/0.02/29979245800)
                            else:
                                sys.exit()
                        else:
                            velocity_fp_list.append(np.nan)
                            velocity_2fp_list.append(np.nan)

                    # freq = csv_input_final["40MHz"][j]/2
                    # fp2_cube = (lambda h: 9 * 10 * np.sqrt(factor_velocity * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
                    # h_radio = inversefunc(fp2_cube, y_values = freq) + 69600000000
                    # print (2/numerical_diff_allen_velocity_2fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
                    # velocity_fp_list.append(2/numerical_diff_allen_velocity_2fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
            
        except:
            print ('Plot error: ' + str(SDATE))
        SDATE += relativedelta(months = move_ave)
    obs_time = np.array(obs_time)
    freq_drift_final = np.array(freq_drift_final)
    start_check_list = np.array(start_check_list)
    end_check_list = np.array(end_check_list)
    duration_list = np.array(duration_list)
    intensity_list = np.array(intensity_list)
    cos_list = np.array(cos_list)
    pol_list = np.array(pol_list)
    sunspot_list = np.array(sunspot_list)
    velocity_fp_list = np.array(velocity_fp_list)
    velocity_2fp_list = np.array(velocity_2fp_list)
    return obs_time, freq_drift_final, start_check_list, end_check_list, duration_list, intensity_list, cos_list, pol_list, sunspot_list, velocity_fp_list, velocity_2fp_list

def frequency_hist_analysis_solar_cycle_dependence():
    for file in [each_freq_drift]:
        if len(file) == 4:
            for burst in ['Micro type Ⅲ burst', 'Ordinary type Ⅲ burst']:
                if file == each_freq_drift:
                    text = ''
                # elif file == freq_drift_day_list:
                #     text = ' observed more than ' +  str(average_threshold) + ' events a day'
                if burst == 'Micro type Ⅲ burst':
                    freq_drift_solar_maximum = file[0]
                    color_1 = "r"
                    freq_drift_solar_minimum = file[1]
                    color_2 = "b"

                elif burst == 'Ordinary type Ⅲ burst':
                    freq_drift_solar_maximum = file[2]
                    color_1 = "orange"
                    freq_drift_solar_minimum = file[3]
                    color_2 = "deepskyblue"
            
                if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
                    max_val = max(freq_drift_solar_minimum)
                else:
                    max_val = max(freq_drift_solar_maximum)
                
                if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
                    min_val = min(freq_drift_solar_minimum)
                else:
                    min_val = min(freq_drift_solar_maximum)
                bin_size = 12
                
                x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
                y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
                x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
                y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
                    plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst + text)
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
                    plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
    return


def frequency_hist_analysis_micro_ordinary_solar_cycle_dependence():
    for file in [each_freq_drift]:
        if len(file) == 4:
            for period in ['Around the solar maximum', 'Around the solar minimum']:
                if file == each_freq_drift:
                    text = ''
                # elif file == freq_drift_day_list:
                    # text = ' observed more than ' +  str(average_threshold) + ' events a day'

                if period == 'Around the solar maximum':
                    freq_drift_micro = file[0]
                    color_2 = "r"
                    freq_drift_ordinary = file[2]
                    color_1 = "orange"
                    
                elif period == 'Around the solar minimum':
                    freq_drift_micro = file[1]
                    color_2 = "b"
                    freq_drift_ordinary = file[3]
                    color_1 = "deepskyblue"
                if len(freq_drift_micro) > 0 and len(freq_drift_ordinary) > 0:
                    if max(freq_drift_micro) >= max(freq_drift_ordinary):
                        max_val = max(freq_drift_micro)
                    else:
                        max_val = max(freq_drift_ordinary)
                    
                    if min(freq_drift_micro) <= min(freq_drift_ordinary):
                        min_val = min(freq_drift_micro)
                    else:
                        min_val = min(freq_drift_ordinary)
                    bin_size = 8
                    
                    x_hist = (plt.hist(freq_drift_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
                    y_hist = (plt.hist(freq_drift_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_ordinary))
                    x_hist_1 = (plt.hist(freq_drift_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
                    y_hist_1 = (plt.hist(freq_drift_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_micro))
                    plt.close()
                    width = x_hist[1]-x_hist[0]
                    for i in range(len(y_hist)):
                        if i == 0:
                            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Ordinary type Ⅲ burst\n'+ str(len(freq_drift_ordinary)) + ' events')
                            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(freq_drift_micro)) + ' events')
                        else:
                            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3)
                            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
                        plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + period + text)
                    # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
                        plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
                        plt.ylabel('Occurrence rate',fontsize=15)
                        # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                        plt.xticks(rotation = 20)
                    plt.show()
                    plt.close()
                else:
                    print ('No data ' + 'freq_drift_micro: ' + str(len(freq_drift_micro)) + ' freq_drift_ordinary' + str(len(freq_drift_ordinary)))
    return

def velocity_hist_analysis_solar_cycle_dependence(velocity_1fp, velocity_2fp):
    for velocity_list in [velocity_1fp, velocity_2fp]:
        if len(velocity_list) == 4:
            for burst in ['Micro type Ⅲ burst', 'Ordinary type Ⅲ burst']:
                if velocity_list == velocity_1fp:
                    text = ''
                    emission_type = 'fp'
                elif velocity_list == velocity_2fp:
                    text = ''
                    emission_type = '2fp'
                # elif file == freq_drift_day_list:
                #     text = ' observed more than ' +  str(average_threshold) + ' events a day'
                if burst == 'Micro type Ⅲ burst':
                    velocity_list_solar_maximum = velocity_list[0]
                    velocity_list_solar_minimum = velocity_list[1]
                elif burst == 'Ordinary type Ⅲ burst':
                    velocity_list_solar_maximum = velocity_list[2]
                    velocity_list_solar_minimum = velocity_list[3]
            
                if max(velocity_list_solar_minimum) >= max(velocity_list_solar_maximum):
                    max_val = max(velocity_list_solar_minimum)
                else:
                    max_val = max(velocity_list_solar_maximum)
                
                if min(velocity_list_solar_minimum) <= min(velocity_list_solar_maximum):
                    min_val = min(velocity_list_solar_minimum)
                else:
                    min_val = min(velocity_list_solar_maximum)
                bin_size = 12
                
                x_hist = (plt.hist(velocity_list_solar_maximum, bins = bin_size, range = (0,1), density= None)[1])
                y_hist = (plt.hist(velocity_list_solar_maximum, bins = bin_size, range = (0,1), density= None)[0]/len(velocity_list_solar_maximum))
                x_hist_1 = (plt.hist(velocity_list_solar_minimum, bins = bin_size, range = (0,1), density= None)[1])
                y_hist_1 = (plt.hist(velocity_list_solar_minimum, bins = bin_size, range = (0,1), density= None)[0]/len(velocity_list_solar_minimum))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(velocity_list_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(velocity_list_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                    plt.title('Velocity @ ' + str(freq_check) + '[MHz/s]' + '\n' + burst + text + ': '+emission_type+' emission')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
                    plt.xlabel('Radial velocity [c]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
    return


def velocity_hist_analysis_micro_ordinary_solar_cycle_dependence(velocity_1fp, velocity_2fp):
    for velocity_list in [velocity_1fp, velocity_2fp]:
        if len(velocity_list) == 4:
            for period in ['Around the solar maximum', 'Around the solar minimum']:
                if velocity_list == velocity_1fp:
                    text = ''
                    emission_type = 'fp'
                elif velocity_list == velocity_2fp:
                    text = ''
                    emission_type = '2fp'
                if period == 'Around the solar maximum':
                    velocity_list_micro = velocity_list[0]
                    velocity_list_ordinary = velocity_list[2]
                    
                elif period == 'Around the solar minimum':
                    velocity_list_micro = velocity_list[1]
                    velocity_list_ordinary = velocity_list[3]
            
                if len(velocity_list_micro) > 0 and len(velocity_list_ordinary) > 0:
                #     if max(velocity_list_micro) >= max(velocity_list_ordinary):
                #         max_val = max(velocity_list_micro)
                #     else:
                #         max_val = max(velocity_list_ordinary)
                    
                #     if min(velocity_list_micro) <= min(velocity_list_ordinary):
                #         min_val = min(velocity_list_micro)
                #     else:
                #         min_val = min(velocity_list_ordinary)
                    bin_size = 8
                    
                    x_hist = (plt.hist(velocity_list_ordinary, bins = bin_size, range = (0,1), density= None)[1])
                    y_hist = (plt.hist(velocity_list_ordinary, bins = bin_size, range = (0,1), density= None)[0]/len(velocity_list_ordinary))
                    x_hist_1 = (plt.hist(velocity_list_micro, bins = bin_size, range = (0,1), density= None)[1])
                    y_hist_1 = (plt.hist(velocity_list_micro, bins = bin_size, range = (0,1), density= None)[0]/len(velocity_list_micro))
                    plt.close()
                    width = x_hist[1]-x_hist[0]
                    for i in range(len(y_hist)):
                        if i == 0:
                            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Ordinary type Ⅲ burst\n'+ str(len(velocity_list_ordinary)) + ' events')
                            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro type Ⅲ burst\n'+ str(len(velocity_list_micro)) + ' events')
                        else:
                            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                        plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + period + text + ': '+emission_type+' emission')
                    # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
                        plt.xlabel('Radial velocity [c]',fontsize=15)
                        plt.ylabel('Occurrence rate',fontsize=15)
                        # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                        plt.xticks(rotation = 20)
                    plt.show()
                    plt.close()
                else:
                    print ('No data ' + 'freq_drift_micro: ' + str(len(freq_drift_micro)) + ' freq_drift_ordinary' + str(len(freq_drift_ordinary)))
    return

def start_end_duration_hist_analysis_solar_cycle_dependence():
    for file in [each_start_frequency]:
        if len(file) == 4:
            for burst in ['Micro type Ⅲ burst', 'Ordinary type Ⅲ burst']:
                if file == each_start_frequency:
                    file1 = each_end_frequency
                    # file2 = duration_each_active_list
                    text = ''
                # elif file == start_frequency_day_list:
                #     file1 = end_frequency_day_list
                #     file2 = duration_day_list
                #     text = ' observed more than ' +  str(average_threshold) + ' events a day'
                else:
                    break
                if burst == 'Micro type Ⅲ burst':
                    freq_start_solar_maximum = file[0]
                    freq_start_solar_minimum = file[1]
                    freq_end_solar_maximum = file1[0]
                    freq_end_solar_minimum = file1[1]
                    # duration_solar_maximum = file2[0]
                    # duration_solar_minimum = file2[1]
                elif burst == 'Ordinary type Ⅲ burst':
                    freq_start_solar_maximum = file[2]
                    freq_start_solar_minimum = file[3]
                    freq_end_solar_maximum = file1[2]
                    freq_end_solar_minimum = file1[3]
                    # duration_solar_maximum = file2[2]
                    # duration_solar_minimum = file2[3]
    
                ##############################################
                #開始周波数
                bin_size = 19
                # bin_size = 1
                #(79.825-29.95)/2.625
                
                
                x_hist = (plt.hist(freq_start_solar_maximum, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist = (plt.hist(freq_start_solar_maximum, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_start_solar_maximum))
                x_hist_1 = (plt.hist(freq_start_solar_minimum, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist_1 = (plt.hist(freq_start_solar_minimum, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_start_solar_minimum))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_start_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_start_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                    # plt.title('Start Frequency')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                    plt.title(burst + text, fontsize=15)
                    plt.xlabel('Start Frequency[MHz]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
                
                ##############################################
                #終了周波数
                bin_size = 19
                
                
                x_hist = (plt.hist(freq_end_solar_maximum, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist = (plt.hist(freq_end_solar_maximum, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_end_solar_maximum))
                x_hist_1 = (plt.hist(freq_end_solar_minimum, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist_1 = (plt.hist(freq_end_solar_minimum, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_end_solar_minimum))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_end_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_end_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                    # plt.title('Start Frequency')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                    plt.title(burst + text, fontsize=15)
                    plt.xlabel('End Frequency[MHz]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
                
                
                # # #終了周波数2
                # # bin_size = 19
                
                
                # # x_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.96, 79.825), density= None)[1])
                # # y_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.96, 79.825), density= None)[0]/len(freq_end_ordinary))
                # # x_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.96, 79.825), density= None)[1])
                # # y_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.96, 79.825), density= None)[0]/len(freq_end_micro))
                # # plt.close()
                # # width = x_hist[1]-x_hist[0]
                # # for i in range(len(y_hist)):
                # #     if i == 0:
                # #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(freq_end_ordinary)) + 'events')
                # #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(freq_end_micro)) + 'events')
                # #     else:
                # #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                # #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                # #     # plt.title('Start Frequency')
                # # # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                # #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                # #     plt.title(text, fontsize=15)
                # #     plt.xlabel('End Frequency[MHz]',fontsize=15)
                # #     plt.ylabel('Occurrence rate',fontsize=15)
                # #     # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                # #     plt.xticks(rotation = 20)
                # # plt.show()
                # # plt.close()
                
                
                
                # ##############################################
                # #継続時間
                # if max(duration_solar_minimum) >= max(duration_solar_maximum):
                #     max_val = max(duration_solar_minimum)
                # else:
                #     max_val = max(duration_solar_maximum)
                
                # if min(duration_solar_minimum) <= min(duration_solar_maximum):
                #     min_val = min(duration_solar_minimum)
                # else:
                #     min_val = min(duration_solar_maximum)
                # bin_size = int((max_val-min_val)*2)
                
                
                # x_hist = (plt.hist(duration_solar_maximum, bins = bin_size, range = (min_val,max_val), density= None)[1])
                # y_hist = (plt.hist(duration_solar_maximum, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_solar_maximum))
                # x_hist_1 = (plt.hist(duration_solar_minimum, bins = bin_size, range = (min_val,max_val), density= None)[1])
                # y_hist_1 = (plt.hist(duration_solar_minimum, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_solar_minimum))
                # plt.close()
                # width = x_hist[1]-x_hist[0]
                # for i in range(len(y_hist)):
                #     if i == 0:
                #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Around solar maximum\n'+ str(len(duration_solar_maximum)) + ' events')
                #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around solar minimum\n'+ str(len(duration_solar_minimum)) + ' events')
                #     else:
                #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                #     # plt.title('Start Frequency')
                # # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                #     plt.title(burst + text, fontsize=15)
                #     plt.xlabel('Duration[sec]',fontsize=15)
                #     plt.ylabel('Occurrence rate',fontsize=15)
                #     # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                #     plt.xticks(rotation = 20)
                # plt.show()
                # plt.close()
    return

def start_end_duration_hist_analysis_micro_ordinary_solar_cycle_dependence():
    for file in [each_start_frequency]:
        if len(file) == 4:
            for period in ['Around the solar maximum', 'Around sthe olar minimum']:
                if file == each_start_frequency:
                    file1 = each_end_frequency
                    # file1 = end_frequency_each_active_list
                    # file2 = duration_each_active_list
                    text = ''
                else:
                    break
                if period == 'Around the solar maximum':
                    freq_start_micro = file[0]
                    freq_start_ordinary = file[2]
                    freq_end_micro = file1[0]
                    freq_end_ordinary = file1[2]
                    # duration_micro = file2[0]
                    # duration_ordinary = file2[2]
                elif period == 'Around the solar minimum':
                    freq_start_micro = file[1]
                    freq_start_ordinary = file[3]
                    freq_end_ordinary = file1[1]
                    freq_end_ordinary = file1[3]
                    # duration_micro = file2[1]
                    # duration_ordinary = file2[3]
    
                ##############################################
                #開始周波数
                bin_size = 19
                #(79.825-29.95)/2.625
                
                
                x_hist = (plt.hist(freq_start_micro, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist = (plt.hist(freq_start_micro, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_start_micro))
                x_hist_1 = (plt.hist(freq_start_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist_1 = (plt.hist(freq_start_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_start_ordinary))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n'+ str(len(freq_start_ordinary)) + ' events')
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(freq_start_micro)) + ' events')
                        
                    else:
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3)
                        
                    # plt.title('Start Frequency')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                    plt.title(period + text, fontsize=15)
                    plt.xlabel('Start Frequency[MHz]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
                
                ##############################################
                #終了周波数
                bin_size = 19
                
                
                x_hist = (plt.hist(freq_end_micro, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist = (plt.hist(freq_end_micro, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_end_micro))
                x_hist_1 = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[1])
                y_hist_1 = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.95, 79.825), density= None)[0]/len(freq_end_ordinary))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n'+ str(len(freq_end_ordinary)) + ' events')
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(freq_end_micro)) + ' events')
                        
                    else:
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3)
                        
                    # plt.title('Start Frequency')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                    plt.title(period + text, fontsize=15)
                    plt.xlabel('End Frequency[MHz]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
                
                
                
                # ##############################################
                # #継続時間
                # if max(duration_ordinary) >= max(duration_micro):
                #     max_val = max(duration_ordinary)
                # else:
                #     max_val = max(duration_micro)
                
                # if min(duration_ordinary) <= min(duration_micro):
                #     min_val = min(duration_ordinary)
                # else:
                #     min_val = min(duration_micro)
                # bin_size = int((max_val-min_val)*2)
                
                
                # x_hist = (plt.hist(duration_micro, bins = bin_size, range = (min_val,max_val), density= None)[1])
                # y_hist = (plt.hist(duration_micro, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_micro))
                # x_hist_1 = (plt.hist(duration_ordinary, bins = bin_size, range = (min_val,max_val), density= None)[1])
                # y_hist_1 = (plt.hist(duration_ordinary, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_ordinary))
                # plt.close()
                # width = x_hist[1]-x_hist[0]
                # for i in range(len(y_hist)):
                #     if i == 0:
                #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n'+ str(len(duration_ordinary)) + ' events')
                #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(duration_micro)) + ' events')
                        
                #     else:
                #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3)
                #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3)
                        
                #     # plt.title('Start Frequency')
                # # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                #     plt.title(period + text, fontsize=15)
                #     plt.xlabel('Duration[sec]',fontsize=15)
                #     plt.ylabel('Occurrence rate',fontsize=15)
                #     # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                #     plt.xticks(rotation = 20)
                # plt.show()
                # plt.close()
    return



        

date_in=[20000101, 20201231]

if __name__=='__main__':
    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')

    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/sgepss_ordinary_LL_RR.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


    try:
        obs_time_ordinary, freq_drift_ordinary, freq_start_ordinary, freq_end_ordinary, duration_ordinary, intensity_list_ordinary, cos_list_ordinary, pol_list_ordinary, sunspot_list_ordinary, velocity_fp_list_ordinary, velocity_2fp_list_ordinary = analysis_bursts(DATE, FDATE, csv_input_final, 'ordinary')
    except:
        print('DL error: ',DATE)


    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')

    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/sgepss_micro_LL_RR.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    
    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time_micro, freq_drift_micro, freq_start_micro, freq_end_micro, duration_micro, intensity_list_micro, cos_list_micro, pol_list_micro, sunspot_list_micro, velocity_fp_list_micro, velocity_2fp_list_micro = analysis_bursts(DATE, FDATE, csv_input_final, 'micro')
    except:
        print('DL error: ',DATE)
        


file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'

sunspot_obs_times = []
sunspot_obs_num_list = []


print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
    # if (BG_obs_time_event >= datetime.datetime(2008, 12, 1)) & (BG_obs_time_event <= datetime.datetime(2019, 12, 31)):
    if (BG_obs_time_event >= datetime.datetime(2007, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2021, 1, 1)):
        sunspot_num = csv_input['sunspot_number'][i]
        if not sunspot_num == -1:
            sunspot_obs_times.append(BG_obs_time_event)
            # Frequency_list = csv_input['Frequency'][i]
            sunspot_obs_num_list.append(sunspot_num)
        else:
            print (BG_obs_time_event)

sunspot_obs_times = np.array(sunspot_obs_times)
sunspot_obs_num_list = np.array(sunspot_obs_num_list)

fig = plt.figure(figsize = (6, 4))
plot_type = 'b'

if plot_type == 'a':
    sunspot_threshold_top = 50
    sunspot_threshold_bottom = 50
    print ('Bottom: '+ str(np.percentile(sunspot_obs_num_list, sunspot_threshold_bottom)))
    print ('Top: '+ str(np.percentile(sunspot_obs_num_list, sunspot_threshold_top)))
    
    sunspot_threshold = [np.percentile(sunspot_obs_num_list, sunspot_threshold_bottom), np.percentile(sunspot_obs_num_list, sunspot_threshold_top)]
    sunspot_obs_times = np.array(sunspot_obs_times)
    sunspot_num_list = np.array(sunspot_obs_num_list)
    topidx = np.where(sunspot_num_list >= np.percentile(sunspot_num_list, sunspot_threshold_top))[0]
    plt.plot(sunspot_obs_times[topidx], sunspot_num_list[topidx], '.', color = 'r')
    bottomidx = np.where(sunspot_num_list <= np.percentile(sunspot_num_list, sunspot_threshold_bottom))[0]
    plt.plot(sunspot_obs_times[bottomidx], sunspot_num_list[bottomidx], '.', color = 'b')
    idx = np.where((sunspot_num_list > np.percentile(sunspot_num_list, sunspot_threshold_bottom)) & (sunspot_num_list < np.percentile(sunspot_num_list, sunspot_threshold_top)))[0]
    plt.plot(sunspot_obs_times[idx], sunspot_num_list[idx], '.', color = 'k')
    print ('filelen top: ' + str(len(sunspot_obs_times[topidx])), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])))
    print ('filerate top: ' + str(len(sunspot_obs_times[topidx])/len(sunspot_obs_times)), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])/len(sunspot_obs_times)))
    plt.axvline(datetime.datetime(2017,1,1), ls = "--", color = "navy")
    plt.axvline(datetime.datetime(2012,1,1), ls = "--", color = "navy")
    plt.axvline(datetime.datetime(2015,1,1), ls = "--", color = "navy")
    plt.axvline(datetime.datetime(2010,1,1), ls = "--", color = "navy")
    plt.ylabel('Sunspots number')
    plt.show()
    plt.close()
elif plot_type == 'b':
    sunspot_threshold_top = 36
    sunspot_threshold_bottom = 36
    print ('Bottom: '+ str(sunspot_threshold_top))
    print ('Top: '+ str(sunspot_threshold_bottom))
    
    sunspot_threshold = [sunspot_threshold_bottom, sunspot_threshold_top]
    sunspot_obs_times = np.array(sunspot_obs_times)
    sunspot_num_list = np.array(sunspot_obs_num_list)
    topidx = np.where(sunspot_num_list >= sunspot_threshold_top)[0]
    plt.plot(sunspot_obs_times[topidx], sunspot_num_list[topidx], '.', color = 'r')
    bottomidx = np.where(sunspot_num_list <= sunspot_threshold_bottom)[0]
    plt.plot(sunspot_obs_times[bottomidx], sunspot_num_list[bottomidx], '.', color = 'b')
    idx = np.where((sunspot_num_list > sunspot_threshold_bottom) & (sunspot_num_list < sunspot_threshold_top))[0]
    plt.plot(sunspot_obs_times[idx], sunspot_num_list[idx], '.', color = 'k')
    print ('filelen top: ' + str(len(sunspot_obs_times[topidx])), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])))
    print ('filerate top: ' + str(len(sunspot_obs_times[topidx])/len(sunspot_obs_times)), 'bottom: ' + str(len(sunspot_obs_times[bottomidx])/len(sunspot_obs_times)))
    # plt.axvline(datetime.datetime(2017,1,1), ls = "--", color = "navy")
    # plt.axvline(datetime.datetime(2012,1,1), ls = "--", color = "navy")
    # plt.axvline(datetime.datetime(2015,1,1), ls = "--", color = "navy")
    # plt.axvline(datetime.datetime(2010,1,1), ls = "--", color = "navy")
    plt.ylabel('Sunspots number')
    plt.show()
    plt.close()


obs_burst = 'Ordinary type Ⅲ bursts'
obs_burst_1 = 'Micro-type Ⅲ bursts'
each_freq_drift_1 = []
each_start_frequency_1 = []
each_end_frequency_1 = []
each_duration_1 = []
each_obs_time_1 = []
each_freq_drift = []
each_start_frequency = []
each_end_frequency = []
each_duration = []
each_obs_time = []
velocity_fp_1 = []
velocity_fp_2 = []
velocity_1fp = []
velocity_2fp = []


burst_types = ["Micro-type Ⅲ burst", "Ordinary type Ⅲ burst"]
for burst_type in burst_types:
    if burst_type == "Micro-type Ⅲ burst":
        obs_time = obs_time_micro
        freq_drift = freq_drift_micro
        start_freq = freq_start_micro
        end_freq = freq_end_micro
        sunspot_list = sunspot_list_micro
        velocity_fp_list = velocity_fp_list_micro
        velocity_2fp_list = velocity_2fp_list_micro
    elif burst_type == "Ordinary type Ⅲ burst":
        obs_time = obs_time_ordinary
        freq_drift = freq_drift_ordinary
        start_freq = freq_start_ordinary
        end_freq = freq_end_ordinary
        sunspot_list = sunspot_list_ordinary
        velocity_fp_list = velocity_fp_list_ordinary
        velocity_2fp_list = velocity_2fp_list_ordinary
    else:
        break

    for period in analysis_period:
        if (period[0] == datetime.datetime(2007, 1, 1)) or (period[0] == datetime.datetime(2017, 1, 1)):
            drift_idx = np.where((period[0] <= obs_time) & (period[1] >= obs_time) & (~np.isnan(freq_drift)) & (sunspot_list <= sunspot_threshold[0]))[0]
        else:
            drift_idx = np.where((period[0] <= obs_time) & (period[1] >= obs_time) & (~np.isnan(freq_drift)) & (sunspot_list >= sunspot_threshold[1]))[0]
        each_freq_drift_1.append(freq_drift[drift_idx].tolist())
        each_obs_time_1.append(obs_time[drift_idx].tolist())
        velocity_fp_1.append(velocity_fp_list[drift_idx].tolist())
        velocity_fp_2.append(velocity_2fp_list[drift_idx].tolist())
        freq_idx = np.where((period[0] <= obs_time) & (period[1] >= obs_time))[0]
        each_start_frequency_1.append(start_freq[freq_idx].tolist())
        each_end_frequency_1.append(end_freq[freq_idx].tolist())


        # print (each_end_frequency_list)
for num in [0,1,4,5]:
    each_freq_drift_1[num].extend(each_freq_drift_1[num+2])
    each_start_frequency_1[num].extend(each_start_frequency_1[num+2])
    each_end_frequency_1[num].extend(each_end_frequency_1[num+2])
    each_obs_time_1[num].extend(each_obs_time_1[num+2])
    velocity_fp_1[num].extend(velocity_fp_1[num+2])
    velocity_fp_2[num].extend(velocity_fp_2[num+2])
    each_freq_drift.append(each_freq_drift_1[num])
    each_start_frequency.append(each_start_frequency_1[num])
    each_end_frequency.append(each_end_frequency_1[num])
    each_obs_time.append(each_obs_time_1[num])
    velocity_1fp.append(velocity_fp_1[num])
    velocity_2fp.append(velocity_fp_2[num])
#0: Micro-極大期
#1: Micro-極小期
#2: FAO 極大期
#3: FAO 極小期


frequency_hist_analysis_solar_cycle_dependence()
frequency_hist_analysis_micro_ordinary_solar_cycle_dependence()
start_end_duration_hist_analysis_solar_cycle_dependence()
# start_end_duration_hist_analysis_micro_ordinary_solar_cycle_dependence()
# velocity_hist_analysis_micro_ordinary_solar_cycle_dependence()
velocity_hist_analysis_solar_cycle_dependence(velocity_1fp, velocity_2fp)
velocity_hist_analysis_micro_ordinary_solar_cycle_dependence(velocity_1fp, velocity_2fp)


# print ('\nMicro-solar maximum: Event Num ' +str(len(each_freq_drift[0])) + '\nFrequency drift rates\nMean DR '+str(np.nanmean(each_freq_drift[0]))+'  STD DR '+str(np.nanstd(each_freq_drift[0]))+'\nDR Max ' + str(np.nanmax(each_freq_drift[0]))+'  DR Min '+str(np.nanmin(each_freq_drift[0]))+'\nRadial velocity(fp)\nMean RV(fp) '+str(np.nanmean(velocity_1fp[0]))+'  STD RV(fp) '+str(np.nanstd(velocity_1fp[0]))+'\nRV(fp) Max ' + str(np.nanmax(velocity_1fp[0]))+'  RV(fp) Min '+str(np.nanmin(velocity_1fp[0]))+'\nRadial velocity(2fp)\nMean RV(2fp) '+str(np.nanmean(velocity_2fp[0]))+'  STD RV(2fp) '+str(np.nanstd(velocity_2fp[0]))+'\nRV(2fp) Max ' + str(np.nanmax(velocity_2fp[0]))+'  RV(2fp) Min '+str(np.nanmin(velocity_2fp[0])))
# print ('\nMicro-solar minimum: Event Num ' +str(len(each_freq_drift[1])) + '\nFrequency drift rates\nMean DR '+str(np.nanmean(each_freq_drift[1]))+'  STD DR '+str(np.nanstd(each_freq_drift[1]))+'\nDR Max ' + str(np.nanmax(each_freq_drift[1]))+'  DR Min '+str(np.nanmin(each_freq_drift[1]))+'\nRadial velocity(fp)\nMean RV(fp) '+str(np.nanmean(velocity_1fp[1]))+'  STD RV(fp) '+str(np.nanstd(velocity_1fp[1]))+'\nRV(fp) Max ' + str(np.nanmax(velocity_1fp[1]))+'  RV(fp) Min '+str(np.nanmin(velocity_1fp[1]))+'\nRadial velocity(2fp)\nMean RV(2fp) '+str(np.nanmean(velocity_2fp[1]))+'  STD RV(2fp) '+str(np.nanstd(velocity_2fp[1]))+'\nRV(2fp) Max ' + str(np.nanmax(velocity_2fp[1]))+'  RV(2fp) Min '+str(np.nanmin(velocity_2fp[1])))
# print ('\nOrdinary-solar maximum: Event Num ' +str(len(each_freq_drift[2])) + '\nFrequency drift rates\nMean DR '+str(np.nanmean(each_freq_drift[2]))+'  STD DR '+str(np.nanstd(each_freq_drift[2]))+'\nDR Max ' + str(np.nanmax(each_freq_drift[2]))+'  DR Min '+str(np.nanmin(each_freq_drift[2]))+'\nRadial velocity(fp)\nMean RV(fp) '+str(np.nanmean(velocity_1fp[2]))+'  STD RV(fp) '+str(np.nanstd(velocity_1fp[2]))+'\nRV(fp) Max ' + str(np.nanmax(velocity_1fp[2]))+'  RV(fp) Min '+str(np.nanmin(velocity_1fp[2]))+'\nRadial velocity(2fp)\nMean RV(2fp) '+str(np.nanmean(velocity_2fp[2]))+'  STD RV(2fp) '+str(np.nanstd(velocity_2fp[2]))+'\nRV(2fp) Max ' + str(np.nanmax(velocity_2fp[2]))+'  RV(2fp) Min '+str(np.nanmin(velocity_2fp[2])))
# print ('\nOrdinary-solar minimum: Event Num ' +str(len(each_freq_drift[3])) + '\nFrequency drift rates\nMean DR '+str(np.nanmean(each_freq_drift[3]))+'  STD DR '+str(np.nanstd(each_freq_drift[3]))+'\nDR Max ' + str(np.nanmax(each_freq_drift[3]))+'  DR Min '+str(np.nanmin(each_freq_drift[3]))+'\nRadial velocity(fp)\nMean RV(fp) '+str(np.nanmean(velocity_1fp[3]))+'  STD RV(fp) '+str(np.nanstd(velocity_1fp[3]))+'\nRV(fp) Max ' + str(np.nanmax(velocity_1fp[3]))+'  RV(fp) Min '+str(np.nanmin(velocity_1fp[3]))+'\nRadial velocity(2fp)\nMean RV(2fp) '+str(np.nanmean(velocity_2fp[3]))+'  STD RV(2fp) '+str(np.nanstd(velocity_2fp[3]))+'\nRV(2fp) Max ' + str(np.nanmax(velocity_2fp[3]))+'  RV(2fp) Min '+str(np.nanmin(velocity_2fp[3])))


#95%信頼区間は母数によって値かえる
#http://kogolab.chillout.jp/elearn/hamburger/chap2/sec3.html
print ('DR analysis')
df = pd.DataFrame([[len(each_freq_drift[0]), np.nanmean(each_freq_drift[0]), np.nanstd(each_freq_drift[0]), np.nanmin(each_freq_drift[0]), np.nanmax(each_freq_drift[0]), np.nanstd(each_freq_drift[0])/np.sqrt(len(each_freq_drift[0])), np.nanmean(each_freq_drift[0]) - np.nanstd(each_freq_drift[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(each_freq_drift[0]) + np.nanstd(each_freq_drift[0])/np.sqrt(len(each_freq_drift[0])), 1.96*np.nanstd(each_freq_drift[0])/np.sqrt(len(each_freq_drift[0])), np.nanmean(each_freq_drift[0]) - 1.96*np.nanstd(each_freq_drift[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(each_freq_drift[0]) + 1.96*np.nanstd(each_freq_drift[0])/np.sqrt(len(each_freq_drift[0]))],
                   [len(each_freq_drift[1]), np.nanmean(each_freq_drift[1]), np.nanstd(each_freq_drift[1]), np.nanmin(each_freq_drift[1]), np.nanmax(each_freq_drift[1]), np.nanstd(each_freq_drift[1])/np.sqrt(len(each_freq_drift[1])), np.nanmean(each_freq_drift[1]) - np.nanstd(each_freq_drift[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(each_freq_drift[1]) + np.nanstd(each_freq_drift[1])/np.sqrt(len(each_freq_drift[1])), 1.98*np.nanstd(each_freq_drift[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(each_freq_drift[1]) - 1.98*np.nanstd(each_freq_drift[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(each_freq_drift[1]) + 1.98*np.nanstd(each_freq_drift[1])/np.sqrt(len(each_freq_drift[1]))],
                   [len(each_freq_drift[2]), np.nanmean(each_freq_drift[2]), np.nanstd(each_freq_drift[2]), np.nanmin(each_freq_drift[2]), np.nanmax(each_freq_drift[2]), np.nanstd(each_freq_drift[2])/np.sqrt(len(each_freq_drift[2])), np.nanmean(each_freq_drift[2]) - np.nanstd(each_freq_drift[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(each_freq_drift[2]) + np.nanstd(each_freq_drift[2])/np.sqrt(len(each_freq_drift[2])), 2*np.nanstd(each_freq_drift[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(each_freq_drift[2]) - 2*np.nanstd(each_freq_drift[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(each_freq_drift[2]) + 2*np.nanstd(each_freq_drift[2])/np.sqrt(len(each_freq_drift[2]))],
                   [len(each_freq_drift[3]), np.nanmean(each_freq_drift[3]), np.nanstd(each_freq_drift[3]), np.nanmin(each_freq_drift[3]), np.nanmax(each_freq_drift[3]), np.nanstd(each_freq_drift[3])/np.sqrt(len(each_freq_drift[3])), np.nanmean(each_freq_drift[3]) - np.nanstd(each_freq_drift[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(each_freq_drift[3]) + np.nanstd(each_freq_drift[3])/np.sqrt(len(each_freq_drift[3])), 2.197*np.nanstd(each_freq_drift[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(each_freq_drift[3]) - 2.197*np.nanstd(each_freq_drift[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(each_freq_drift[3]) + 2.197*np.nanstd(each_freq_drift[3])/np.sqrt(len(each_freq_drift[3]))]], 
index=['Micro solar maximum', 'Micro solar minimum', 'Ordinary solar maximum', 'Ordinary solar minimum'], 
columns=['Event Num', 'AVE DR','STD DR','DR Min', 'DR Max', 'DR SE', 'AVE - SE', 'AVE + SE', '95%信頼区間', '95%信頼区間下限', '95%信頼区間上限'])
pd.set_option('display.max_columns', 150)
print (df)



print ('\nRV analysis(fp)')
df = pd.DataFrame([[len(each_freq_drift[0]), np.nanmean(each_freq_drift[0]), np.nanmean(velocity_1fp[0]), np.nanstd(velocity_1fp[0]), np.nanmin(velocity_1fp[0]), np.nanmax(velocity_1fp[0]), np.nanstd(velocity_1fp[0])/np.sqrt(len(each_freq_drift[0])), np.nanmean(velocity_1fp[0]) - np.nanstd(velocity_1fp[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(velocity_1fp[0]) + np.nanstd(velocity_1fp[0])/np.sqrt(len(each_freq_drift[0])), 1.96*np.nanstd(velocity_1fp[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(velocity_1fp[0]) - 1.96*np.nanstd(velocity_1fp[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(velocity_1fp[0]) + 1.96*np.nanstd(velocity_1fp[0])/np.sqrt(len(each_freq_drift[0]))],
                   [len(each_freq_drift[1]), np.nanmean(each_freq_drift[1]), np.nanmean(velocity_1fp[1]), np.nanstd(velocity_1fp[1]), np.nanmin(velocity_1fp[1]), np.nanmax(velocity_1fp[1]), np.nanstd(velocity_1fp[1])/np.sqrt(len(each_freq_drift[1])), np.nanmean(velocity_1fp[1]) - np.nanstd(velocity_1fp[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(velocity_1fp[1]) + np.nanstd(velocity_1fp[1])/np.sqrt(len(each_freq_drift[1])), 1.98*np.nanstd(velocity_1fp[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(velocity_1fp[1]) - 1.98*np.nanstd(velocity_1fp[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(velocity_1fp[1]) + 1.98*np.nanstd(velocity_1fp[1])/np.sqrt(len(each_freq_drift[1]))],
                   [len(each_freq_drift[2]), np.nanmean(each_freq_drift[2]), np.nanmean(velocity_1fp[2]), np.nanstd(velocity_1fp[2]), np.nanmin(velocity_1fp[2]), np.nanmax(velocity_1fp[2]), np.nanstd(velocity_1fp[2])/np.sqrt(len(each_freq_drift[2])), np.nanmean(velocity_1fp[2]) - np.nanstd(velocity_1fp[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(velocity_1fp[2]) + np.nanstd(velocity_1fp[2])/np.sqrt(len(each_freq_drift[2])), 2*np.nanstd(velocity_1fp[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(velocity_1fp[2]) - 2*np.nanstd(velocity_1fp[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(velocity_1fp[2]) + 2*np.nanstd(velocity_1fp[2])/np.sqrt(len(each_freq_drift[2]))],
                   [len(each_freq_drift[3]), np.nanmean(each_freq_drift[3]), np.nanmean(velocity_1fp[3]), np.nanstd(velocity_1fp[3]), np.nanmin(velocity_1fp[3]), np.nanmax(velocity_1fp[3]), np.nanstd(velocity_1fp[3])/np.sqrt(len(each_freq_drift[3])), np.nanmean(velocity_1fp[3]) - np.nanstd(velocity_1fp[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(velocity_1fp[3]) + np.nanstd(velocity_1fp[3])/np.sqrt(len(each_freq_drift[3])), 2.197*np.nanstd(velocity_1fp[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(velocity_1fp[3]) - 2.197*np.nanstd(velocity_1fp[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(velocity_1fp[3]) + 2.197*np.nanstd(velocity_1fp[3])/np.sqrt(len(each_freq_drift[3]))]], 
index=['Micro solar maximum', 'Micro solar minimum', 'Ordinary solar maximum', 'Ordinary solar minimum'], 
columns=['Event Num', 'AVE DR','AVE RD(fp)','STD RD(fp)','RD(fp) Min', 'RD(fp) Max', 'RD(fp) SE', 'AVE - SE', 'AVE + SE', '95%信頼区間', '95%信頼区間下限', '95%信頼区間上限'])
pd.set_option('display.max_columns', 150)
print (df)

print ('\nRV analysis(2fp)')
df = pd.DataFrame([[len(each_freq_drift[0]), np.nanmean(each_freq_drift[0]), np.nanmean(velocity_2fp[0]), np.nanstd(velocity_2fp[0]), np.nanmin(velocity_2fp[0]), np.nanmax(velocity_2fp[0]), np.nanstd(velocity_2fp[0])/np.sqrt(len(each_freq_drift[0])), np.nanmean(velocity_2fp[0]) - np.nanstd(velocity_2fp[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(velocity_2fp[0]) + np.nanstd(velocity_2fp[0])/np.sqrt(len(each_freq_drift[0])), 1.96*np.nanstd(velocity_2fp[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(velocity_2fp[0]) - 1.96*np.nanstd(velocity_2fp[0])/np.sqrt(len(each_freq_drift[0])),np.nanmean(velocity_2fp[0]) + 1.96*np.nanstd(velocity_2fp[0])/np.sqrt(len(each_freq_drift[0]))],
                   [len(each_freq_drift[1]), np.nanmean(each_freq_drift[1]), np.nanmean(velocity_2fp[1]), np.nanstd(velocity_2fp[1]), np.nanmin(velocity_2fp[1]), np.nanmax(velocity_2fp[1]), np.nanstd(velocity_2fp[1])/np.sqrt(len(each_freq_drift[1])), np.nanmean(velocity_2fp[1]) - np.nanstd(velocity_2fp[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(velocity_2fp[1]) + np.nanstd(velocity_2fp[1])/np.sqrt(len(each_freq_drift[1])), 1.98*np.nanstd(velocity_2fp[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(velocity_2fp[1]) - 1.98*np.nanstd(velocity_2fp[1])/np.sqrt(len(each_freq_drift[1])),np.nanmean(velocity_2fp[1]) + 1.98*np.nanstd(velocity_2fp[1])/np.sqrt(len(each_freq_drift[1]))],
                   [len(each_freq_drift[2]), np.nanmean(each_freq_drift[2]), np.nanmean(velocity_2fp[2]), np.nanstd(velocity_2fp[2]), np.nanmin(velocity_2fp[2]), np.nanmax(velocity_2fp[2]), np.nanstd(velocity_2fp[2])/np.sqrt(len(each_freq_drift[2])), np.nanmean(velocity_2fp[2]) - np.nanstd(velocity_2fp[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(velocity_2fp[2]) + np.nanstd(velocity_2fp[2])/np.sqrt(len(each_freq_drift[2])), 2*np.nanstd(velocity_2fp[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(velocity_2fp[2]) - 2*np.nanstd(velocity_2fp[2])/np.sqrt(len(each_freq_drift[2])),np.nanmean(velocity_2fp[2]) + 2*np.nanstd(velocity_2fp[2])/np.sqrt(len(each_freq_drift[2]))],
                   [len(each_freq_drift[3]), np.nanmean(each_freq_drift[3]), np.nanmean(velocity_2fp[3]), np.nanstd(velocity_2fp[3]), np.nanmin(velocity_2fp[3]), np.nanmax(velocity_2fp[3]), np.nanstd(velocity_2fp[3])/np.sqrt(len(each_freq_drift[3])), np.nanmean(velocity_2fp[3]) - np.nanstd(velocity_2fp[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(velocity_2fp[3]) + np.nanstd(velocity_2fp[3])/np.sqrt(len(each_freq_drift[3])), 2.197*np.nanstd(velocity_2fp[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(velocity_2fp[3]) - 2.197*np.nanstd(velocity_2fp[3])/np.sqrt(len(each_freq_drift[3])),np.nanmean(velocity_2fp[3]) + 2.197*np.nanstd(velocity_2fp[3])/np.sqrt(len(each_freq_drift[3]))]], 
index=['Micro solar maximum', 'Micro solar minimum', 'Ordinary solar maximum', 'Ordinary solar minimum'], 
columns=['Event Num', 'AVE DR','AVE RD(2fp)','STD RD(2fp)','RD(2fp) Min', 'RD(2fp) Max', 'RD(2fp) SE', 'AVE - SE', 'AVE + SE', '95%信頼区間', '95%信頼区間下限', '95%信頼区間上限'])
pd.set_option('display.max_columns', 150)
print (df)


# freq_drift_min = np.nanmin([np.min(each_freq_drift[0]), np.nanmin(each_freq_drift[1]), np.nanmin(each_freq_drift[2]), np.nanmin(each_freq_drift[3])]) - 0.5
# freq_drift_max = np.nanmax([np.max(each_freq_drift[0]), np.nanmax(each_freq_drift[1]), np.nanmax(each_freq_drift[2]), np.nanmax(each_freq_drift[3])]) + 0.5



# # each_obs_time = np.array(each_obs_time)
# for i in range(len(each_obs_time)):
#     nd_obs_time = np.array(each_obs_time[i])
#     if i == 0 or i == 1:
#         burst_type = "Micro type Ⅲ burst"
#         if i == 0:
#             burst_period = "around the solar maximum"
#         else:
#             burst_period = "around the solar minimum"
#     else:
#         burst_type = "Ordinary type Ⅲ burst"
#         if i == 2:
#             burst_period = "around the solar maximum"
#         else:
#             burst_period = "around the solar minimum"
#     for period in analysis_period:
#         sdate,edate=period
#         if len(nd_obs_time[(nd_obs_time <= edate) & (nd_obs_time >= sdate)]) > 0:
#             DATE=sdate
#             while DATE <= edate:
#                 if len(nd_obs_time[(nd_obs_time <= datetime.datetime.combine((DATE+relativedelta(months=move_ave_analysis)).date(), datetime.datetime.min.time())) & (nd_obs_time >= DATE)]) > 0:
#                     index = np.where((nd_obs_time <= datetime.datetime.combine((DATE+relativedelta(months=move_ave_analysis)).date(), datetime.datetime.min.time())) & (nd_obs_time >= DATE))[0]
#                     ax = plt.subplot()
#                     ax.plot(nd_obs_time[index[0]:index[-1]+1], each_freq_drift[i][index[0]:index[-1]+1], '.')
                    
#                     xfmt = mdates.DateFormatter("%m/%d")
#                     xloc = mdates.DayLocator(interval=60)
#                     ax.xaxis.set_major_locator(xloc)
#                     ax.xaxis.set_major_formatter(xfmt)
                    
#                     ax.set_ylim(freq_drift_min, freq_drift_max)
#                     ax.set_title(burst_type + ' observed '+burst_period + '\n'+DATE.strftime(format='%Y%m%d')+' - ' + (datetime.datetime.combine((DATE+relativedelta(months=move_ave_analysis)).date(), datetime.datetime.min.time())-datetime.timedelta(days=1)).strftime(format='%Y%m%d'), fontsize = 14)
#                     ax.set_ylabel("DRs[MHz/s]", fontsize = 12)
#                     ax.set_xlabel("Time", fontsize = 12)
#                     plt.show()
#                     plt.close()

                




#                 DATE+=relativedelta(months = move_ave_analysis)
#                 DATE = datetime.datetime.combine(DATE.date(), datetime.datetime.min.time())

# solar_maximum = [datetime.datetime(2000, 1, 1), datetime.datetime(2003, 1, 1)]
# solar_minimum = [datetime.datetime(2007, 1, 1), datetime.datetime(2010, 1, 1)]
# solar_maximum_1 = [datetime.datetime(2012, 1, 1), datetime.datetime(2015, 1, 1)]
# solar_minimum_1 = [datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 1)]
# analysis_period = [solar_maximum, solar_minimum, solar_maximum_1, solar_minimum_1]




# micro_min_idx = np.where((((analysis_period[1][0] <= obs_time_micro) & (analysis_period[1][1] >= obs_time_micro)) | ((analysis_period[3][0] <= obs_time_micro) & (analysis_period[3][1] >= obs_time_micro))) & (~np.isnan(freq_drift_micro)) & (sunspot_list_micro <= sunspot_threshold[0]) & (~np.isnan(intensity_list_micro)))[0]
# micro_max_idx = np.where((((analysis_period[0][0] <= obs_time_micro) & (analysis_period[0][1] >= obs_time_micro)) | ((analysis_period[2][0] <= obs_time_micro) & (analysis_period[2][1] >= obs_time_micro))) & (~np.isnan(freq_drift_micro)) & (sunspot_list_micro >= sunspot_threshold[1]) & (~np.isnan(intensity_list_micro)))[0]
# ordinary_min_idx = np.where((((analysis_period[1][0] <= obs_time_ordinary) & (analysis_period[1][1] >= obs_time_ordinary)) | ((analysis_period[3][0] <= obs_time_ordinary) & (analysis_period[3][1] >= obs_time_ordinary))) & (~np.isnan(freq_drift_ordinary)) & (sunspot_list_ordinary <= sunspot_threshold[0]) & (~np.isnan(intensity_list_ordinary)))[0]
# ordinary_max_idx = np.where((((analysis_period[0][0] <= obs_time_ordinary) & (analysis_period[0][1] >= obs_time_ordinary)) | ((analysis_period[2][0] <= obs_time_ordinary) & (analysis_period[2][1] >= obs_time_ordinary))) & (~np.isnan(freq_drift_ordinary)) & (sunspot_list_ordinary >= sunspot_threshold[1]) & (~np.isnan(intensity_list_ordinary)))[0]


# x_lims = [np.min([np.nanmin(intensity_list_micro), np.nanmin(intensity_list_ordinary)]),np.max([np.nanmax(intensity_list_micro), np.nanmax(intensity_list_ordinary)])]
# y_lims = [np.min([np.nanmin(freq_drift_micro), np.nanmin(freq_drift_ordinary)]),np.max([np.nanmax(freq_drift_micro), np.nanmax(freq_drift_ordinary)])]

# plt.title('Micro type III burst')

# plt.scatter(intensity_list_micro[micro_max_idx], freq_drift_micro[micro_max_idx], c = 'r',marker='.')
# plt.scatter(intensity_list_micro[micro_min_idx], freq_drift_micro[micro_min_idx], c = 'b',marker='.')
# plt.xlabel('from background [dB]')
# plt.ylabel('Drift rates @ 40MHz [MHz/s]')
# plt.xlim(x_lims[0], x_lims[1])
# plt.ylim(y_lims[0], y_lims[1])
# plt.show()
# plt.close()

# plt.title('Ordinary type III burst')

# plt.scatter(intensity_list_ordinary[ordinary_max_idx], freq_drift_ordinary[ordinary_max_idx], c = 'r',marker='.')
# plt.scatter(intensity_list_ordinary[ordinary_min_idx], freq_drift_ordinary[ordinary_min_idx], c = 'b',marker='.')
# plt.xlabel('from background [dB]')
# plt.ylabel('Drift rates @ 40MHz [MHz/s]')
# plt.xlim(x_lims[0], x_lims[1])
# plt.ylim(y_lims[0], y_lims[1])
# plt.show()
# plt.close()


# plt.title('Micro type III burst')

# plt.scatter(freq_drift_micro[micro_max_idx], np.abs(pol_list_micro[micro_max_idx]), c = 'r',marker='.')
# plt.scatter(freq_drift_micro[micro_min_idx], np.abs(pol_list_micro[micro_min_idx]), c = 'b',marker='.')
# plt.xlabel('Drift rates @ 40MHz [MHz/s]')
# plt.ylabel('polarization')
# plt.ylim(0, 1)
# plt.xlim(y_lims[0], y_lims[1])
# plt.show()
# plt.close()

# plt.title('Ordinary type III burst')

# plt.scatter(freq_drift_ordinary[ordinary_max_idx], np.abs(pol_list_ordinary[ordinary_max_idx]), c = 'r',marker='.')
# plt.scatter(freq_drift_ordinary[ordinary_min_idx], np.abs(pol_list_ordinary[ordinary_min_idx]), c = 'b',marker='.')
# plt.xlabel('Drift rates @ 40MHz [MHz/s]')
# plt.ylabel('polarization')
# plt.ylim(0, 1)
# plt.xlim(y_lims[0], y_lims[1])
# plt.show()
# plt.close()

# plt.title('Micro type III burst')

# plt.scatter(sunspot_list_micro[micro_max_idx], freq_drift_micro[micro_max_idx], c = 'r',marker='.')
# plt.scatter(sunspot_list_micro[micro_min_idx], freq_drift_micro[micro_min_idx], c = 'b',marker='.')
# plt.xlabel('Sunspots number')
# plt.ylabel('Drift rates @ 40MHz [MHz/s]')
# # plt.xlim(x_lims[0], x_lims[1])
# plt.ylim(y_lims[0], y_lims[1])
# plt.show()
# plt.close()

# plt.title('Ordinary type III burst')

# plt.scatter(sunspot_list_ordinary[ordinary_max_idx], freq_drift_ordinary[ordinary_max_idx], c = 'r',marker='.')
# plt.scatter(sunspot_list_ordinary[ordinary_min_idx], freq_drift_ordinary[ordinary_min_idx], c = 'b',marker='.')
# plt.xlabel('Sunspots number')
# plt.ylabel('Drift rates @ 40MHz [MHz/s]')
# # plt.xlim(x_lims[0], x_lims[1])
# plt.ylim(y_lims[0], y_lims[1])
# plt.show()
# plt.close()



        # for period in ['Around the solar maximum']:
        #     if velocity_list == velocity_1fp:
        #         text = ''
        #         emission_type = 'fp'
        #     elif velocity_list == velocity_2fp:
        #         text = ''
        #         emission_type = '2fp'
        #     if period == 'Around the solar maximum':
        #         velocity_list_micro = velocity_1fp[0]
        #         velocity_list_ordinary =  velocity_2fp[2]
                
        #     elif period == 'Around the solar minimum':
        #         velocity_list_micro = velocity_1fp[1]
        #         velocity_list_ordinary = velocity_2fp[3]
        
        #     if len(velocity_list_micro) > 0 and len(velocity_list_ordinary) > 0:
        #         bin_size = 8
                
        #         x_hist = (plt.hist(velocity_list_ordinary, bins = bin_size, range = (0,0.6), density= None)[1])
        #         y_hist = (plt.hist(velocity_list_ordinary, bins = bin_size, range = (0,0.6), density= None)[0]/len(velocity_list_ordinary))
        #         x_hist_1 = (plt.hist(velocity_list_micro, bins = bin_size, range = (0,0.6), density= None)[1])
        #         y_hist_1 = (plt.hist(velocity_list_micro, bins = bin_size, range = (0,0.6), density= None)[0]/len(velocity_list_micro))
        #         plt.close()
        #         width = x_hist[1]-x_hist[0]
        #         for i in range(len(y_hist)):
        #             if i == 0:
        #                 plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Ordinary type Ⅲ burst'+' (f=2fp)\n'+ str(len(velocity_list_ordinary)) + ' events')
        #                 plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro type Ⅲ burst'+' (f=fp)\n'+ str(len(velocity_list_micro)) + ' events')
        #             else:
        #                 plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
        #                 plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
        #             plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + period)
        #         # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
        #             plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
        #             plt.xlabel('Radial velocity[c]',fontsize=15)
        #             plt.ylabel('Occurrence rate',fontsize=15)
        #             # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
        #             plt.xticks(rotation = 20)
        #         plt.show()
        #         plt.close()
