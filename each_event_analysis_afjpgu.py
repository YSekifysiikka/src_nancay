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
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


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


def analysis_bursts(DATE, FDATE, csv_input_final, burst_type):
    date_start = date(DATE.year, DATE.month, DATE.day)
    FDATE = date(FDATE.year, FDATE.month, FDATE.day)
    SDATE = date_start
    # print (SDATE)
    obs_time = []
    freq_drift_final = []
    start_check_list = []
    end_check_list = []
    obs_time_all = []
    duration_list = []
    time_list = []
    repeat_num = 0
    while SDATE < FDATE:
        try:
            sdate = int(str(SDATE.year)+str(SDATE.month).zfill(2)+str(SDATE.day).zfill(2))
            EDATE = SDATE + relativedelta(months=move_ave) - relativedelta(days=1)
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
                    year = int(str(csv_input_final['event_date'][j])[0:4])
                    if str(csv_input_final['event_date'][j])[4:5] == '0':
                        month = int(str(csv_input_final['event_date'][j])[5:6])
                    else:
                        month = int(str(csv_input_final['event_date'][j])[4:6])
                    if str(csv_input_final['event_date'][j])[6:7] == '0':
                        day = int(str(csv_input_final['event_date'][j])[7:8])
                    else:
                        day = int(str(csv_input_final['event_date'][j])[6:8])
                    # obs_date = date(year, month, day)
                    obs_date = datetime.datetime(year, month, day, int(str(csv_input_final['event_hour'][j])), int(str(csv_input_final['event_minite'][j])))
                    obs_time_all.append(obs_date)
                    if csv_input_final[["freq_start"][0]][j] >= freq_check:
                        if csv_input_final[["freq_end"][0]][j] <= freq_check:
                            # print (csv_input_final['event_date'][j])
                            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
                            best_factor = csv_input_final["factor"][j]
                            for z in range(separate_num):
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
                    # plt.plot(freq[i], invcube_3)
                    # plt.show()
                    # plt.close()
                    
                    slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
                    freq_drift.append(-slope)
                    freq_drift_final.append(-slope)
                    time_list.append([inversefunc(cube_3, y_values=37.5)-inversefunc(cube_3, y_values=40), inversefunc(cube_3, y_values=32.5)-inversefunc(cube_3, y_values=35)])
                    
                    # if burst_type == 'ordinary':
                    #     if -slope >= 10:
                    #         print (-slope, obs_time[i])
                    # elif burst_type = 'micro':
                    
                    # else:
                    #     sys.exit()
                    #     break


            
        except:
            print ('Plot error: ' + str(SDATE))
        SDATE += relativedelta(months = move_ave)
    return obs_time, freq_drift_final, repeat_num, start_check_list, end_check_list, duration_list, obs_time_all, time_list

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
                    freq_drift_solar_minimum = file[1]
                elif burst == 'Ordinary type Ⅲ burst':
                    freq_drift_solar_maximum = file[2]
                    freq_drift_solar_minimum = file[3]
            
                if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
                    max_val = max(freq_drift_solar_minimum)
                else:
                    max_val = max(freq_drift_solar_maximum)
                
                if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
                    min_val = min(freq_drift_solar_minimum)
                else:
                    min_val = min(freq_drift_solar_maximum)
                bin_size = 8
                
                x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
                y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
                x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
                y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Around solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
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
            for period in ['Around solar maximum', 'Around solar minimum']:
                if file == each_freq_drift:
                    text = ''
                # elif file == freq_drift_day_list:
                    # text = ' observed more than ' +  str(average_threshold) + ' events a day'

                if period == 'Around solar maximum':
                    freq_drift_micro = file[0]
                    freq_drift_ordinary = file[2]
                    
                elif period == 'Around solar minimum':
                    freq_drift_micro = file[1]
                    freq_drift_ordinary = file[3]
            
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
                            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Ordinary type Ⅲ burst\n'+ str(len(freq_drift_ordinary)) + ' events')
                            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(freq_drift_micro)) + ' events')
                        else:
                            plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                            plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
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
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Around solar maximum\n'+ str(len(freq_start_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around solar minimum\n'+ str(len(freq_start_solar_minimum)) + ' events')
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
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Around solar maximum\n'+ str(len(freq_end_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around solar minimum\n'+ str(len(freq_end_solar_minimum)) + ' events')
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
            for period in ['Around solar maximum', 'Around solar minimum']:
                if file == each_start_frequency:
                    file1 = each_end_frequency
                    # file1 = end_frequency_each_active_list
                    # file2 = duration_each_active_list
                    text = ''
                else:
                    break
                if period == 'Around solar maximum':
                    freq_start_micro = file[0]
                    freq_start_ordinary = file[2]
                    freq_end_micro = file1[0]
                    freq_end_ordinary = file1[2]
                    # duration_micro = file2[0]
                    # duration_ordinary = file2[2]
                elif period == 'Around solar minimum':
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

    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/afjpgu_flare_associated_ordinary_dB.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time_ordinary, freq_drift_ordinary, repeat_num, freq_start_ordinary, freq_end_ordinary, duration_ordinary, obs_time_all_ordinary, time_list_ordinary = analysis_bursts(DATE, FDATE, csv_input_final, 'ordinary')
    except:
        print('DL error: ',DATE)


    start_day, end_day=date_in
    DATE=pd.to_datetime(start_day,format='%Y%m%d')
    FDATE = pd.to_datetime(end_day,format='%Y%m%d')

    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/afjpgu_micro_dB.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    
    # DATE=sdate
    # date=DATE.strftime(format='%Y%m%d')
    # print(date)
    try:
        obs_time_micro, freq_drift_micro, repeat_num, freq_start_micro, freq_end_micro, duration_micro, obs_time_all_micro, time_list_micro = analysis_bursts(DATE, FDATE, csv_input_final, 'micro')
    except:
        print('DL error: ',DATE)








burst_types = ["Micro-type Ⅲ burst", "Ordinary type Ⅲ burst"]
for burst_type in burst_types:
    if burst_type == "Micro-type Ⅲ burst":
        obs_time = obs_time_micro
        freq_drift = freq_drift_micro
        start_freq = freq_start_micro
        end_freq = freq_end_micro
    elif burst_type == "Ordinary type Ⅲ burst":
        obs_time = obs_time_ordinary
        freq_drift = freq_drift_ordinary
        start_freq = freq_start_ordinary
        end_freq = freq_end_ordinary
    else:
        break

    for period in analysis_period:
        each_freq_drift_list = []
        each_start_frequency_list = []
        each_end_frequency_list = []
        obs_time_list = []
        for i in range(len(obs_time)):
            if period[0] <= obs_time[i]:
                if period[1] >= obs_time[i]:
                    obs_time_list.append(obs_time[i])
                    each_freq_drift_list.append(freq_drift[i])
                    each_start_frequency_list.append(start_freq[i])
                    each_end_frequency_list.append(end_freq[i])
        each_freq_drift_1.append(each_freq_drift_list)
        each_start_frequency_1.append(each_start_frequency_list)
        each_end_frequency_1.append(each_end_frequency_list)
        each_obs_time_1.append(obs_time_list)
        # print (each_end_frequency_list)
for num in [0,1,4,5]:
    each_freq_drift_1[num].extend(each_freq_drift_1[num+2])
    each_start_frequency_1[num].extend(each_start_frequency_1[num+2])
    each_end_frequency_1[num].extend(each_end_frequency_1[num+2])
    each_obs_time_1[num].extend(each_obs_time_1[num+2])
    each_freq_drift.append(each_freq_drift_1[num])
    each_start_frequency.append(each_start_frequency_1[num])
    each_end_frequency.append(each_end_frequency_1[num])
    each_obs_time.append(each_obs_time_1[num])
#0: Micro-極大期
#1: Micro-極小期
#2: FAO 極大期
#3: FAO 極小期


frequency_hist_analysis_solar_cycle_dependence()
frequency_hist_analysis_micro_ordinary_solar_cycle_dependence()
start_end_duration_hist_analysis_solar_cycle_dependence()
start_end_duration_hist_analysis_micro_ordinary_solar_cycle_dependence()




print ('\nMicro-solar maximum: Event Num ' +str(len(each_freq_drift[0])) + ' \nMean DR '+str(np.mean(each_freq_drift[0]))+'  STD DR '+str(np.std(each_freq_drift[0]))+'\nDR Max ' + str(max(each_freq_drift[0]))+'  DR Min '+str(min(each_freq_drift[0])))
print ('\nMicro-solar minimum: Event Num ' +str(len(each_freq_drift[1])) + ' \nMean DR '+str(np.mean(each_freq_drift[1]))+'  STD DR '+str(np.std(each_freq_drift[1]))+'\nDR Max ' + str(max(each_freq_drift[1]))+'  DR Min '+str(min(each_freq_drift[1])))
print ('\nOrdinary-solar maximum: Event Num ' +str(len(each_freq_drift[2])) + ' \nMean DR '+str(np.mean(each_freq_drift[2]))+'  STD DR '+str(np.std(each_freq_drift[2]))+'\nDR Max ' + str(max(each_freq_drift[2]))+'  DR Min '+str(min(each_freq_drift[2])))
print ('\nOrdinary-solar minimum: Event Num ' +str(len(each_freq_drift[3])) + ' \nMean DR '+str(np.mean(each_freq_drift[3]))+'  STD DR '+str(np.std(each_freq_drift[3]))+'\nDR Max ' + str(max(each_freq_drift[3]))+'  DR Min '+str(min(each_freq_drift[3])))
# files_list = []
# for j in range(len(csv_input_final)):
#     start_check= str(csv_input_final[["freq_start"][0]][j])
#     end_check=str(csv_input_final[["freq_end"][0]][j])
#     event_end = str(csv_input_final[["event_end"][0]][j])
#     event_start = str(csv_input_final[["event_start"][0]][j])
#     event_date = str(csv_input_final['event_date'][j])
#     yyyy = str(csv_input_final['event_date'][j])[:4]
#     files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/ordinary/'+yyyy+'/'+event_date+'_*_'+event_start+'_'+event_end+'_'+start_check+'_'+end_check+'peak.png')
#     if len(files)==1:
#         files_list.append(files[0])
#     elif len(files)>=2:
#         print (files)
#     else:
#         sys.exit()
    # /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/ordinary/2020/20201128_100146_100826_8500_8900_360_370_63.55_29.95peak.png



freq_drift_min = np.min([np.min(each_freq_drift[0]), np.min(each_freq_drift[1]), np.min(each_freq_drift[2]), np.min(each_freq_drift[3])]) - 0.5
freq_drift_max = np.max([np.max(each_freq_drift[0]), np.max(each_freq_drift[1]), np.max(each_freq_drift[2]), np.max(each_freq_drift[3])]) + 0.5



# each_obs_time = np.array(each_obs_time)
for i in range(len(each_obs_time)):
    nd_obs_time = np.array(each_obs_time[i])
    if i == 0 or i == 1:
        burst_type = "Micro type Ⅲ burst"
        if i == 0:
            burst_period = "around the solar maximum"
        else:
            burst_period = "around the solar minimum"
    else:
        burst_type = "Ordinary type Ⅲ burst"
        if i == 2:
            burst_period = "around the solar maximum"
        else:
            burst_period = "around the solar minimum"
    for period in analysis_period:
        sdate,edate=period
        if len(nd_obs_time[(nd_obs_time <= edate) & (nd_obs_time >= sdate)]) > 0:
            DATE=sdate
            while DATE <= edate:
                if len(nd_obs_time[(nd_obs_time <= datetime.datetime.combine((DATE+relativedelta(months=move_ave_analysis)).date(), datetime.datetime.min.time())) & (nd_obs_time >= DATE)]) > 0:
                    index = np.where((nd_obs_time <= datetime.datetime.combine((DATE+relativedelta(months=move_ave_analysis)).date(), datetime.datetime.min.time())) & (nd_obs_time >= DATE))[0]
                    ax = plt.subplot()
                    ax.plot(nd_obs_time[index[0]:index[-1]+1], each_freq_drift[i][index[0]:index[-1]+1], '.')
                    
                    xfmt = mdates.DateFormatter("%m/%d")
                    xloc = mdates.DayLocator(interval=60)
                    ax.xaxis.set_major_locator(xloc)
                    ax.xaxis.set_major_formatter(xfmt)
                    
                    ax.set_ylim(freq_drift_min, freq_drift_max)
                    ax.set_title(burst_type + ' observed '+burst_period + '\n'+DATE.strftime(format='%Y%m%d')+' - ' + (datetime.datetime.combine((DATE+relativedelta(months=move_ave_analysis)).date(), datetime.datetime.min.time())-datetime.timedelta(days=1)).strftime(format='%Y%m%d'), fontsize = 14)
                    ax.set_ylabel("DRs[MHz/s]", fontsize = 12)
                    ax.set_xlabel("Time", fontsize = 12)
                    plt.show()
                    plt.close()

                




                DATE+=relativedelta(months = move_ave_analysis)
                DATE = datetime.datetime.combine(DATE.date(), datetime.datetime.min.time())





# plt.title('Time gap between 40MHz-37.5MHz: Micro')
# plt.hist(np.array(time_list_micro)[:,0])
# plt.xlabel('Time')
# plt.ylabel('Event number')
# plt.show()
# plt.close()
# plt.title('Time gap between 35MHz-32.5MHz: Micro')
# plt.hist(np.array(time_list_micro)[:,1])
# plt.xlabel('Time')
# plt.ylabel('Event number')
# plt.show()

# print('Max: '+str(np.max(np.array(time_list_micro)[:,0])) + ' Min: ' + str(np.min(np.max(np.array(time_list_micro)[:,0]))))
    
# plt.title('Time gap between 40MHz-37.5MHz: Ordinary')
# plt.hist(np.array(time_list_ordinary)[:,0])
# plt.xlabel('Time')
# plt.ylabel('Event number')
# plt.show()
# plt.close()
# plt.title('Time gap between 35MHz-32.5MHz: Ordinary')
# plt.hist(np.array(time_list_ordinary)[:,1])
# plt.xlabel('Time')
# plt.ylabel('Event number')
# plt.show()

# print('Max: '+str(np.max(np.array(time_list_ordinary)[:,0])) + ' Min: ' + str(np.min(np.max(np.array(time_list_ordinary)[:,0]))))
    
    
