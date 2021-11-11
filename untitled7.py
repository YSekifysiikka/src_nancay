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
freq_check = 35
move_ave = 12
move_plot = 4
#赤の線の変動を調べる
analysis_move_ave = 12
average_threshold = 5
error_threshold = 5
solar_maximum = [datetime.date(2012, 1, 1), datetime.date(2014, 12, 31)]
solar_minimum = [datetime.date(2017, 1, 1), datetime.date(2020, 12, 31)]
analysis_period = [solar_maximum, solar_minimum]

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
freq_drift_day_list = []
freq_drift_each_active_list = []
start_frequency_day_list = []
start_frequency_each_active_list = []
end_frequency_day_list = []
end_frequency_each_active_list = []
duration_day_list = []
duration_each_active_list = []



def analysis_bursts(DATE, FDATE, csv_input_final):
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
                    
                    slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
                    freq_drift.append(-slope)
                    freq_drift_final.append(-slope)


            
        except:
            print ('Plot error: ' + SDATE)
        SDATE+= relativedelta.relativedelta(months=move_ave)
    return obs_time, freq_drift_final, repeat_num, start_check_list, end_check_list, duration_list, obs_time_all

def frequency_hist_analysis_solar_cycle_dependence():
    for file in [freq_drift_each_active_list, freq_drift_day_list]:
        if len(file) == 4:
            for burst in ['Micro type Ⅲ burst', 'Ordinary type Ⅲ burst']:
                if file == freq_drift_each_active_list:
                    text = ' observed more than ' +  str(average_threshold) + ' events\nfrom same active region'
                elif file == freq_drift_day_list:
                    text = ' observed more than ' +  str(average_threshold) + ' events a day'
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
    for file in [freq_drift_each_active_list, freq_drift_day_list]:
        if len(file) == 4:
            for period in ['Around solar maximum', 'Around solar minimum']:
                if file == freq_drift_each_active_list:
                    text = ' observed more than ' +  str(average_threshold) + ' events\nfrom same active region'
                elif file == freq_drift_day_list:
                    text = ' observed more than ' +  str(average_threshold) + ' events a day'

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
    for file in [start_frequency_each_active_list, start_frequency_day_list]:
        if len(file) == 4:
            for burst in ['Micro type Ⅲ burst', 'Ordinary type Ⅲ burst']:
                if file == start_frequency_each_active_list:
                    file1 = end_frequency_each_active_list
                    file2 = duration_each_active_list
                    text = ' observed more than ' +  str(average_threshold) + ' events\nfrom same active region'
                elif file == start_frequency_day_list:
                    file1 = end_frequency_day_list
                    file2 = duration_day_list
                    text = ' observed more than ' +  str(average_threshold) + ' events a day'
                else:
                    break
                if burst == 'Micro type Ⅲ burst':
                    freq_start_solar_maximum = file[0]
                    freq_start_solar_minimum = file[1]
                    freq_end_solar_maximum = file1[0]
                    freq_end_solar_minimum = file1[1]
                    duration_solar_maximum = file2[0]
                    duration_solar_minimum = file2[1]
                elif burst == 'Ordinary type Ⅲ burst':
                    freq_start_solar_maximum = file[2]
                    freq_start_solar_minimum = file[3]
                    freq_end_solar_maximum = file1[2]
                    freq_end_solar_minimum = file1[3]
                    duration_solar_maximum = file2[2]
                    duration_solar_minimum = file2[3]
    
                ##############################################
                #開始周波数
                bin_size = 19
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
                
                
                # #終了周波数2
                # bin_size = 19
                
                
                # x_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.96, 79.825), density= None)[1])
                # y_hist = (plt.hist(freq_end_ordinary, bins = bin_size, range = (29.96, 79.825), density= None)[0]/len(freq_end_ordinary))
                # x_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.96, 79.825), density= None)[1])
                # y_hist_1 = (plt.hist(freq_end_micro, bins = bin_size, range = (29.96, 79.825), density= None)[0]/len(freq_end_micro))
                # plt.close()
                # width = x_hist[1]-x_hist[0]
                # for i in range(len(y_hist)):
                #     if i == 0:
                #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n' + str(len(freq_end_ordinary)) + 'events')
                #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n' + str(len(freq_end_micro)) + 'events')
                #     else:
                #         plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                #         plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                #     # plt.title('Start Frequency')
                # # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                #     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                #     plt.title(text, fontsize=15)
                #     plt.xlabel('End Frequency[MHz]',fontsize=15)
                #     plt.ylabel('Occurrence rate',fontsize=15)
                #     # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                #     plt.xticks(rotation = 20)
                # plt.show()
                # plt.close()
                
                
                
                ##############################################
                #継続時間
                if max(duration_solar_minimum) >= max(duration_solar_maximum):
                    max_val = max(duration_solar_minimum)
                else:
                    max_val = max(duration_solar_maximum)
                
                if min(duration_solar_minimum) <= min(duration_solar_maximum):
                    min_val = min(duration_solar_minimum)
                else:
                    min_val = min(duration_solar_maximum)
                bin_size = int((max_val-min_val)*2)
                
                
                x_hist = (plt.hist(duration_solar_maximum, bins = bin_size, range = (min_val,max_val), density= None)[1])
                y_hist = (plt.hist(duration_solar_maximum, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_solar_maximum))
                x_hist_1 = (plt.hist(duration_solar_minimum, bins = bin_size, range = (min_val,max_val), density= None)[1])
                y_hist_1 = (plt.hist(duration_solar_minimum, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_solar_minimum))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label = 'Around solar maximum\n'+ str(len(duration_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around solar minimum\n'+ str(len(duration_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                    # plt.title('Start Frequency')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                    plt.title(burst + text, fontsize=15)
                    plt.xlabel('Duration[sec]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
    return

def start_end_duration_hist_analysis_micro_ordinary_solar_cycle_dependence():
    for file in [start_frequency_each_active_list, start_frequency_day_list]:
        if len(file) == 4:
            for period in ['Around solar maximum', 'Around solar minimum']:
                if file == start_frequency_each_active_list:
                    file1 = end_frequency_each_active_list
                    file2 = duration_each_active_list
                    text = ' observed more than ' +  str(average_threshold) + ' events\nfrom same active region'
                elif file == start_frequency_day_list:
                    file1 = end_frequency_day_list
                    file2 = duration_day_list
                    text = ' observed more than ' +  str(average_threshold) + ' events a day'
                else:
                    break
                if period == 'Around solar maximum':
                    freq_start_micro = file[0]
                    freq_start_ordinary = file[2]
                    freq_end_micro = file1[0]
                    freq_end_ordinary = file1[2]
                    duration_micro = file2[0]
                    duration_ordinary = file2[2]
                elif period == 'Around solar minimum':
                    freq_start_micro = file[1]
                    freq_start_ordinary = file[3]
                    freq_end_ordinary = file1[1]
                    freq_end_ordinary = file1[3]
                    duration_micro = file2[1]
                    duration_ordinary = file2[3]
    
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
                
                
                
                ##############################################
                #継続時間
                if max(duration_ordinary) >= max(duration_micro):
                    max_val = max(duration_ordinary)
                else:
                    max_val = max(duration_micro)
                
                if min(duration_ordinary) <= min(duration_micro):
                    min_val = min(duration_ordinary)
                else:
                    min_val = min(duration_micro)
                bin_size = int((max_val-min_val)*2)
                
                
                x_hist = (plt.hist(duration_micro, bins = bin_size, range = (min_val,max_val), density= None)[1])
                y_hist = (plt.hist(duration_micro, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_micro))
                x_hist_1 = (plt.hist(duration_ordinary, bins = bin_size, range = (min_val,max_val), density= None)[1])
                y_hist_1 = (plt.hist(duration_ordinary, bins = bin_size, range = (min_val,max_val), density= None)[0]/len(duration_ordinary))
                plt.close()
                width = x_hist[1]-x_hist[0]
                for i in range(len(y_hist)):
                    if i == 0:
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3, label = 'Ordinary type Ⅲ burst\n'+ str(len(duration_ordinary)) + ' events')
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(duration_micro)) + ' events')
                        
                    else:
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'b', alpha = 0.3)
                        
                    # plt.title('Start Frequency')
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 11)
                    plt.title(period + text, fontsize=15)
                    plt.xlabel('Duration[sec]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
    return
        

    
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
        obs_time_ordinary, freq_drift_ordinary, repeat_num, freq_start_ordinary, freq_end_ordinary, duration_ordinary, obs_time_all_ordinary = analysis_bursts(DATE, FDATE, csv_input_final)
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
        obs_time_micro, freq_drift_micro, repeat_num, freq_start_micro, freq_end_micro, duration_micro, obs_time_all_micro = analysis_bursts(DATE, FDATE, csv_input_final)
    except:
        print('DL error: ',DATE)



burst_types = ["Micro-type Ⅲ burst", "Ordinary type Ⅲ burst"]
for burst_type in burst_types:
    if burst_type == "Micro-type Ⅲ burst":
        obs_time = obs_time_micro
        freq_drift = freq_drift_micro
    elif burst_type == "Ordinary type Ⅲ burst":
        obs_time = obs_time_ordinary
        freq_drift = freq_drift_ordinary
    else:
        break
        
##############################################################################################
#連続して発生したバーストを合わせて解析
    start_time = []
    end_time = []
    list_start_num = []
    list_end_num = []
    for i in range(len(obs_time)):
        if i == 0:
            start_time.append(obs_time[i])
            list_start_num.append(i)
            if not obs_time[0] == obs_time[1]:
                end_time.append(obs_time[i])
                list_end_num.append(i)

        elif i == len(obs_time) - 1:
            end_time.append(obs_time[i])
            list_end_num.append(i)
            if not obs_time[-1] == obs_time[-2]:
                start_time.append(obs_time[i])
                list_start_num.append(i)
        else:
            if not (obs_time[i] == obs_time[i-1] or obs_time[i] == obs_time[i-1] + relativedelta.relativedelta(days=1)) and not (obs_time[i] + relativedelta.relativedelta(days=1) == obs_time[i+1] or obs_time[i] == obs_time[i+1]):
                start_time.append(obs_time[i])
                list_start_num.append(i)
                end_time.append(obs_time[i])
                list_end_num.append(i)
            else:
                if not (obs_time[i] == obs_time[i-1] or obs_time[i] == obs_time[i-1] + relativedelta.relativedelta(days=1)):
                    # print (str(obs_time_micro[i]) + '-')
                    start_time.append(obs_time[i])
                    list_start_num.append(i)
                if not (obs_time[i] + relativedelta.relativedelta(days=1) == obs_time[i+1] or obs_time[i] == obs_time[i+1]):
                    # print ('-' + str(obs_time_micro[i]))
                    end_time.append(obs_time[i])
                    list_end_num.append(i)
                else:
                    pass
###############################################
    #Count_number
    x_list = []
    y_list = []
    for i in range (len(start_time)):
        start = start_time[i]
        end = end_time[i]
        while start <= end:
            x_list.append(start)
            y_list.append(list_end_num[i] - list_start_num[i])
            start += relativedelta.relativedelta(days=1)
    
    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.scatter(x_list, y_list)
    # plt.title(burst_type,fontsize=25)
    # plt.tick_params(labelsize=20)
    # plt.ylabel('Occurence Number',fontsize=25)
    # plt.show()
    # plt.close()


###############################################
    # 連続で観測されたデータの平均



    for period in analysis_period:
        freq_drift_mean_list = []
        freq_drift_std_list = []
        time_mean_list = []
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(len(start_time)):
###############################################
            #太陽極大期と極小期のセレクト
            if period[0] <= start_time[i] and period[1] >= start_time[i]:
###############################################
        #平均のデータの選定
                if len(freq_drift[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
        ###############################################
                    # print (i)
                    freq_drift_mean = np.mean(freq_drift[list_start_num[i]:list_end_num[i] + 1])
                    freq_drift_std = np.std(freq_drift[list_start_num[i]:list_end_num[i] + 1])
                    time_mean = start_time[i] + (end_time[i] - start_time[i])/2
                    freq_drift_mean_list.append(freq_drift_mean)
                    freq_drift_std_list.append(freq_drift_std)
                    time_mean_list.append(time_mean)
                    if len(freq_drift[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
                        ax.errorbar(time_mean, freq_drift_mean, yerr=freq_drift_std, color = 'b', marker = 's')

                    # else:
                        # ax.scatter(time_mean, freq_drift_mean, color = 'b')


    ###############################################
        #移動平均のプロット
        sobs_time = []
        eobs_time = []
        mobs_time = []
        month_frequency_mean = []
        month_frequency_std = []
        for i in range(60):
            month_freq_drift = []
            for j in range(len(time_mean_list)):
                if time_mean_list[j] >= date(time_mean_list[0].year, 1, 1)+relativedelta.relativedelta(months=analysis_move_ave*i) and time_mean_list[j] <= date(time_mean_list[0].year, 1, 1)+ relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1):
                    month_freq_drift.append(freq_drift_mean_list[j])
            if len(month_freq_drift)>0:
                sobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i))
                eobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1))
                month_frequency_mean.append(np.mean(month_freq_drift))
                month_frequency_std.append(np.std(month_freq_drift))
                if not analysis_move_ave == 1:
                    mobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(months=analysis_move_ave/2)))
                elif analysis_move_ave == 1:
                    mobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(days=15)))
    
        for i in range (len(sobs_time)):
            # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
            if i == 0:
                ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average \nObserved more than ' +  str(average_threshold) + ' events\nfrom same active region', linewidth = 5.0)
            else:
                ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', linewidth = 5.0)
            
        for i in range (len(sobs_time)):
            if eobs_time[i] + relativedelta.relativedelta(days=1) in sobs_time:
                ax.plot([eobs_time[i], sobs_time[i+1]], [month_frequency_mean[i], month_frequency_mean[i+1]], color = 'r', linewidth = 5.0)
    
        plt.title(burst_type + ' @' + str(freq_check) + '[MHz]',fontsize=25)
        plt.ylabel('Frequency drift rates[MHz/s]',fontsize=25)
        plt.xlim(period)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize = 24)
        # plt.show()
        plt.close()
        freq_drift_each_active_list.append(freq_drift_mean_list)








##############################################################################################
#連続して発生したバーストを1日ごとに解析
#周波数ドリフト率
    start_time = []
    end_time = []
    list_start_num = []
    list_end_num = []
    for i in range(len(obs_time)):
        if i == 0:
            start_time.append(obs_time[i])
            list_start_num.append(i)
            if not obs_time[0] == obs_time[1]:
                end_time.append(obs_time[i])
                list_end_num.append(i)
        elif i == len(obs_time) - 1:
            end_time.append(obs_time[i])
            list_end_num.append(i)
            if not obs_time[-1] == obs_time[-2]:
                start_time.append(obs_time[i])
                list_start_num.append(i)
        else:
            if not (obs_time[i] == obs_time[i-1]) and not (obs_time[i] == obs_time[i+1]):
                start_time.append(obs_time[i])
                list_start_num.append(i)
                end_time.append(obs_time[i])
                list_end_num.append(i)
            else:
                if not (obs_time[i] == obs_time[i-1]):
                    # print (str(obs_time_micro[i]) + '-')
                    start_time.append(obs_time[i])
                    list_start_num.append(i)
                if not (obs_time[i] == obs_time[i+1]):
                    # print ('-' + str(obs_time_micro[i]))
                    end_time.append(obs_time[i])
                    list_end_num.append(i)
                else:
                    pass

###############################################
    #Count_number
    x_list = []
    y_list = []
    for i in range (len(start_time)):
        start = start_time[i]
        end = end_time[i]
        while start <= end:
            x_list.append(start)
            y_list.append(list_end_num[i] - list_start_num[i])
            start += relativedelta.relativedelta(days=1)
    
    # fig, ax = plt.subplots(figsize=(14, 6))
    # ax.scatter(x_list, y_list)
    # plt.title(burst_type,fontsize=25)
    # plt.tick_params(labelsize=20)
    # plt.ylabel('Occurence Number',fontsize=25)
    # plt.show()
    # plt.close()



###############################################
    #連続で観測されたデータの平均

    for period in analysis_period:
        freq_drift_mean_list = []
        freq_drift_std_list = []
        time_mean_list = []
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(len(start_time)):
###############################################
            #太陽極大期と極小期のセレクト
            if period[0] <= start_time[i] and period[1] >= start_time[i]:
    ###############################################
        #平均のデータの選定
                if len(freq_drift[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
        ###############################################
                    # print (i)
                    freq_drift_mean = np.mean(freq_drift[list_start_num[i]:list_end_num[i] + 1])
                    freq_drift_std = np.std(freq_drift[list_start_num[i]:list_end_num[i] + 1])
                    time_mean = start_time[i] + (end_time[i] - start_time[i])/2
                    freq_drift_mean_list.append(freq_drift_mean)
                    freq_drift_std_list.append(freq_drift_std)
                    time_mean_list.append(time_mean)
                    if len(freq_drift[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
                        ax.errorbar(time_mean, freq_drift_mean, yerr=freq_drift_std, color = 'b', marker = 's')
                    # else:
                        # ax.scatter(time_mean, freq_drift_mean, color = 'b')

    ###############################################
        #移動平均のプロット
        sobs_time = []
        eobs_time = []
        mobs_time = []
        month_frequency_mean = []
        month_frequency_std = []
        for i in range(60):
            month_freq_drift = []
            for j in range(len(time_mean_list)):
                if time_mean_list[j] >= date(time_mean_list[0].year, 1, 1)+relativedelta.relativedelta(months=analysis_move_ave*i) and time_mean_list[j] <= date(time_mean_list[0].year, 1, 1)+ relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1):
                    month_freq_drift.append(freq_drift_mean_list[j])
            if len(month_freq_drift)>0:
                sobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i))
                eobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1))
                month_frequency_mean.append(np.mean(month_freq_drift))
                month_frequency_std.append(np.std(month_freq_drift))
                if not analysis_move_ave == 1:
                    mobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(months=analysis_move_ave/2)))
                elif analysis_move_ave == 1:
                    mobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(days=15)))
    
        for i in range (len(sobs_time)):
            # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
            if i == 0:
                ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average \nObserved more than ' +  str(average_threshold) + ' events a day', linewidth = 5.0)
            else:
                ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', linewidth = 5.0)
            
        for i in range (len(sobs_time)):
            if eobs_time[i] + relativedelta.relativedelta(days=1) in sobs_time:
                ax.plot([eobs_time[i], sobs_time[i+1]], [month_frequency_mean[i], month_frequency_mean[i+1]], color = 'r', linewidth = 5.0)
    ###############################################
        plt.title(burst_type + ' @' + str(freq_check) + '[MHz]',fontsize=25)
        plt.ylabel('Frequency drift rates[MHz/s]',fontsize=25)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize = 24)
        plt.xlim(period)
        # plt.show()
        plt.close()
        freq_drift_day_list.append(freq_drift_mean_list)



#############################################################################################################################################
frequency_hist_analysis_solar_cycle_dependence()
# frequency_hist_analysis_micro_ordinary_solar_cycle_dependence()
###############################################


burst_types = ["Micro-type Ⅲ burst", "Ordinary type Ⅲ burst"]
for burst_type in burst_types:
    if burst_type == "Micro-type Ⅲ burst":
        freq_start = freq_start_micro
        freq_end = freq_end_micro
        duration = duration_micro
        obs_time_all = obs_time_all_micro
    elif burst_type == "Ordinary type Ⅲ burst":
        freq_start = freq_start_ordinary
        freq_end = freq_end_ordinary
        duration = duration_ordinary
        obs_time_all = obs_time_all_ordinary
    else:
        break
        
##############################################################################################
#連続して発生したバーストを合わせて解析
    start_time = []
    end_time = []
    list_start_num = []
    list_end_num = []
    for i in range(len(obs_time_all)):
        if i == 0:
            start_time.append(obs_time_all[i])
            list_start_num.append(i)
            if not obs_time_all[0] == obs_time_all[1]:
                end_time.append(obs_time_all[i])
                list_end_num.append(i)
        elif i == len(obs_time_all) - 1:
            end_time.append(obs_time_all[i])
            list_end_num.append(i)
            if not obs_time_all[-1] == obs_time_all[-2]:
                start_time.append(obs_time_all[i])
                list_start_num.append(i)
        else:
            if not (obs_time_all[i] == obs_time_all[i-1] or obs_time_all[i] == obs_time_all[i-1] + relativedelta.relativedelta(days=1)) and not (obs_time_all[i] + relativedelta.relativedelta(days=1) == obs_time_all[i+1] or obs_time_all[i] == obs_time_all[i+1]):
                start_time.append(obs_time_all[i])
                list_start_num.append(i)
                end_time.append(obs_time_all[i])
                list_end_num.append(i)
            else:
                if not (obs_time_all[i] == obs_time_all[i-1] or obs_time_all[i] == obs_time_all[i-1] + relativedelta.relativedelta(days=1)):
                    # print (str(obs_time_micro[i]) + '-')
                    start_time.append(obs_time_all[i])
                    list_start_num.append(i)
                if not (obs_time_all[i] + relativedelta.relativedelta(days=1) == obs_time_all[i+1] or obs_time_all[i] == obs_time_all[i+1]):
                    # print ('-' + str(obs_time_micro[i]))
                    end_time.append(obs_time_all[i])
                    list_end_num.append(i)
                else:
                    pass
###############################################
    #連続で観測されたデータの平均
    for period in analysis_period:
        freq_start_mean_list = []
        freq_end_mean_list = []
        duration_mean_list = []
        time_mean_list = []
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(len(start_time)):
###############################################
            #太陽極大期と極小期のセレクト
            if period[0] <= start_time[i] and period[1] >= start_time[i]:
            ###############################################
            #平均のデータの選定
                    if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
            ###############################################
                        # print (i)
                        freq_start_mean = np.mean(freq_start[list_start_num[i]:list_end_num[i] + 1])
                        freq_start_mean_list.append(freq_start_mean)
                        freq_start_std = np.std(freq_start[list_start_num[i]:list_end_num[i] + 1])
                        freq_end_mean = np.mean(freq_end[list_start_num[i]:list_end_num[i] + 1])
                        freq_end_mean_list.append(freq_end_mean)
                        freq_end_std = np.std(freq_end[list_start_num[i]:list_end_num[i] + 1])
                        duration_mean = np.mean(duration[list_start_num[i]:list_end_num[i] + 1])
                        duration_mean_list.append(duration_mean)
                        duration_std = np.std(duration[list_start_num[i]:list_end_num[i] + 1])
                        time_mean = start_time[i] + (end_time[i] - start_time[i])/2
                        time_mean_list.append(time_mean)
                        if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
                            ax.errorbar(time_mean, freq_start_mean, yerr=freq_start_std, color = 'b', marker = 's')
        plt.title(burst_type + '\nObserved more than ' +  str(average_threshold) + ' events\nfrom same active region', fontsize=25)
        plt.ylabel('Start Frequency[MHz]',fontsize=25)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize = 24)
        # plt.show()
        plt.close()
        start_frequency_each_active_list.append(freq_start_mean_list)
        end_frequency_each_active_list.append(freq_end_mean_list)
        duration_each_active_list.append(duration_mean_list)



##############################################################################################
#連続して発生したバーストを1日ごとに解析
    start_time = []
    end_time = []
    list_start_num = []
    list_end_num = []
    for i in range(len(obs_time_all)):
        if i == 0:
            start_time.append(obs_time_all[i])
            list_start_num.append(i)
            if not obs_time_all[0] == obs_time_all[1]:
                end_time.append(obs_time_all[i])
                list_end_num.append(i)
        elif i == len(obs_time_all) - 1:
            end_time.append(obs_time_all[i])
            list_end_num.append(i)
            if not obs_time_all[-1] == obs_time_all[-2]:
                start_time.append(obs_time_all[i])
                list_start_num.append(i)
        else:
            if not (obs_time_all[i] == obs_time_all[i-1]) and not (obs_time_all[i] == obs_time_all[i+1]):
                start_time.append(obs_time_all[i])
                list_start_num.append(i)
                end_time.append(obs_time_all[i])
                list_end_num.append(i)
            else:
                if not (obs_time_all[i] == obs_time_all[i-1]):
                    # print (str(obs_time_micro[i]) + '-')
                    start_time.append(obs_time_all[i])
                    list_start_num.append(i)
                if not (obs_time_all[i] == obs_time_all[i+1]):
                    # print ('-' + str(obs_time_micro[i]))
                    end_time.append(obs_time_all[i])
                    list_end_num.append(i)
                else:
                    pass
###############################################
    #連続で観測されたデータの平均
    for period in analysis_period:
        freq_start_mean_list = []
        freq_end_mean_list = []
        duration_mean_list = []
        time_mean_list = []
        fig, ax = plt.subplots(figsize=(14, 6))
        for i in range(len(start_time)):
###############################################
            #太陽極大期と極小期のセレクト
            if period[0] <= start_time[i] and period[1] >= start_time[i]:
        ###############################################
        #平均のデータの選定
                if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
        ###############################################
                    # print (i)
                    freq_start_mean = np.mean(freq_start[list_start_num[i]:list_end_num[i] + 1])
                    freq_start_mean_list.append(freq_start_mean)
                    freq_start_std = np.std(freq_start[list_start_num[i]:list_end_num[i] + 1])
                    freq_end_mean = np.mean(freq_end[list_start_num[i]:list_end_num[i] + 1])
                    freq_end_mean_list.append(freq_end_mean)
                    freq_end_std = np.std(freq_end[list_start_num[i]:list_end_num[i] + 1])
                    duration_mean = np.mean(duration[list_start_num[i]:list_end_num[i] + 1])
                    duration_mean_list.append(duration_mean)
                    duration_std = np.std(duration[list_start_num[i]:list_end_num[i] + 1])
                    time_mean = start_time[i] + (end_time[i] - start_time[i])/2
                    time_mean_list.append(time_mean)
                    if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
                        ax.errorbar(time_mean, freq_start_mean, yerr=freq_start_std, color = 'b', marker = 's')
        plt.title(burst_type + '\nObserved more than ' +  str(average_threshold) + ' events a day', fontsize=25)
        plt.ylabel('Start Frequency[MHz]',fontsize=25)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize = 24)
        # plt.show()
        plt.close()
        start_frequency_day_list.append(freq_start_mean_list)
        end_frequency_day_list.append(freq_end_mean_list)
        duration_day_list.append(duration_mean_list)

###############################################
start_end_duration_hist_analysis_solar_cycle_dependence()
###############################################
# start_end_duration_hist_analysis_micro_ordinary_solar_cycle_dependence()