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
import scipy
from scipy import stats
import sys
freq_check = 40
move_ave = 12
move_plot = 4
#赤の線の変動を調べる
analysis_move_ave = 12
average_threshold = 1
error_threshold = 1
solar_maximum = [datetime.datetime(2012, 1, 1), datetime.datetime(2015, 1, 1)]
solar_minimum = [datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 1)]
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

import astropy.time
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import math

def solar_cos(obs_datetime):
    koko = EarthLocation(lat='47 22 24.00',lon='2 11 50.00', height = '150')
    obs_time = obs_datetime
    toki = astropy.time.Time(obs_time)
    taiyou = get_sun(toki).transform_to(AltAz(obstime=toki,location=koko))
    # print(taiyou)
    # print(taiyou.az) # 天球での方位角
    # print(taiyou.alt) # 天球での仰俯角
    # print(taiyou.distance) # 距離
    # print(taiyou.distance.au) # au単位での距離
    
    azimuth = float(str(taiyou.az).split('d')[0] + '.' + str(taiyou.az).split('d')[1].split('m')[0])
    altitude = float(str(taiyou.alt).split('d')[0] + '.' + str(taiyou.alt).split('d')[1].split('m')[0])
    
    solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
                            math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
                            math.sin(math.radians(altitude))])
    machine_place = np.array([-math.cos(math.radians(0)),
                              0,
                              math.sin(math.radians(0))])
    cos = np.dot(solar_place, machine_place)
    return cos


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
    dB_list = []
    power_list = []
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
                    obs_datetime = datetime.datetime(year,month,day,csv_input_final['event_hour'][j],csv_input_final['event_minite'][j])
                    obs_time_all.append(obs_datetime)
                    if csv_input_final[["freq_start"][0]][j] >= freq_check:
                        if csv_input_final[["freq_end"][0]][j] <= freq_check:
                            # print (csv_input_final['event_date'][j])
                            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
                            best_factor = csv_input_final["factor"][j]
                            for z in range(separate_num):
                                obs_time.append(obs_datetime)
                                freq.append(freq_check)
                                start_check.append(csv_input_final[["freq_start"][0]][j])
                                y_factor.append(best_factor)
                                y_velocity.append(float(velocity[best_factor-1]))
                                time_gap.append(csv_input_final[["event_end"][0]][j] - csv_input_final[["event_start"][0]][j] + 1)
                                peak_power = 10 ** (csv_input_final[["peak_dB_40MHz"][0]][j]/10)
                                BG_power = 10 ** (csv_input_final[["BG_decibel"][0]][j]/10)
                                cos = solar_cos(obs_datetime)
                                dB_from_BG = np.log10((peak_power - BG_power)/cos) * 10
                                dB_list.append(dB_from_BG)
                                power_list.append(peak_power - BG_power)
                                
                                
        
            
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
    return obs_time, freq_drift_final, repeat_num, start_check_list, end_check_list, duration_list, obs_time_all, dB_list, power_list



# for drift_rates in file:






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
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
                    else:
                        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
                        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
                    plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst + text)
                # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
                    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 13)
                    plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
                    plt.ylabel('Occurrence rate',fontsize=15)
                    plt.xlim(0.5,12)
                    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
                    plt.xticks(rotation = 20)
                plt.show()
                plt.close()
    return

def frequency_hist_analysis_micro_ordinary_solar_cycle_dependence():
    for file in [freq_drift_each_active_list, freq_drift_day_list]:
        if len(file) == 4:
            for period in ['Around the solar maximum', 'Around the solar minimum']:
                if file == freq_drift_each_active_list:
                    text = ' observed more than ' +  str(average_threshold) + ' events\nfrom same active region'
                elif file == freq_drift_day_list:
                    text = ' observed more than ' +  str(average_threshold) + ' events a day'

                if period == 'Around the solar maximum':
                    freq_drift_micro = file[0]
                    freq_drift_ordinary = file[2]
                    
                elif period == 'Around the solar minimum':
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
                        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 13)
                        plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
                        plt.ylabel('Occurrence rate',fontsize=15)
                        plt.xlim(0.5,12)
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


    
# bursts_types = ['ordinary', 'storm']
# analysis_methods = ['each', 'all']
# csv_files = ['ordinary_final_with_dB.csv', 'micro_final_with_dB.csv']
date_in=[20070101, 20201231]
file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

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
        obs_time_ordinary, freq_drift_ordinary, repeat_num, freq_start_ordinary, freq_end_ordinary, duration_ordinary, obs_time_all_ordinary, dB_list_ordinary, power_list_ordinary = analysis_bursts(DATE, FDATE, csv_input_final)
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
        obs_time_micro, freq_drift_micro, repeat_num, freq_start_micro, freq_end_micro, duration_micro, obs_time_all_micro, dB_list_micro, power_list_micro = analysis_bursts(DATE, FDATE, csv_input_final)
    except:
        print('DL error: ',DATE)

file = dB_list_ordinary
print ('Mean: '+str(np.mean(file)), 'STD: '+str(np.std(file)), 'Max: '+str(np.max(file)), 'Min: '+str(np.min(file)))

# db_all_list = []
# for i in range(len(dB_list_ordinary)):
#     db_all_list.append(dB_list_ordinary[i])
# for i in range(len(dB_list_micro)):
#     db_all_list.append(dB_list_micro[i])



from scipy import optimize


dB_list_maximum_ordinary = []
freq_drift_maximum_ordinary = []
dB_list_minimum_ordinary = []
freq_drift_minimum_ordinary = []
dB_list_maximum_micro = []
freq_drift_maximum_micro = []
dB_list_minimum_micro = []
freq_drift_minimum_micro = []

for i in range(len(obs_time_ordinary)):
    if obs_time_ordinary[i] >= datetime.datetime(2012,1,1):
        if obs_time_ordinary[i] <= datetime.datetime(2015,1,1):
            dB_list_maximum_ordinary.append(dB_list_ordinary[i])
            freq_drift_maximum_ordinary.append(freq_drift_ordinary[i])
    if obs_time_ordinary[i] >= datetime.datetime(2017,1,1):
        if obs_time_ordinary[i] <= datetime.datetime(2020,1,1):
            dB_list_minimum_ordinary.append(dB_list_ordinary[i])
            freq_drift_minimum_ordinary.append(freq_drift_ordinary[i])


for i in range(len(obs_time_micro)):
    if obs_time_micro[i] >= datetime.datetime(2012,1,1):
        if obs_time_micro[i] <= datetime.datetime(2015,1,1):
            dB_list_maximum_micro.append(dB_list_micro[i])
            freq_drift_maximum_micro.append(freq_drift_micro[i])
    if obs_time_micro[i] >= datetime.datetime(2017,1,1):
        if obs_time_micro[i] <= datetime.datetime(2020,1,1):
            dB_list_minimum_micro.append(dB_list_micro[i])
            freq_drift_minimum_micro.append(freq_drift_micro[i])

# 1次式の近似
def func_c1(x, a, b):
    return a*x + b
ordinary_line_maximum_a, ordinary_line_maximum_b = optimize.curve_fit(func_c1, dB_list_maximum_ordinary, freq_drift_maximum_ordinary)[0]
ordinary_line_minimum_a, ordinary_line_minimum_b = optimize.curve_fit(func_c1, dB_list_minimum_ordinary, freq_drift_minimum_ordinary)[0]
micro_line_maximum_a, micro_line_maximum_b = optimize.curve_fit(func_c1,dB_list_maximum_micro, freq_drift_maximum_micro )[0]
micro_line_minimum_a, micro_line_minimum_b = optimize.curve_fit(func_c1, dB_list_minimum_micro, freq_drift_minimum_micro)[0]
x = np.arange(20, 60, 5)

y_ordinary_maximum = x * ordinary_line_maximum_a + ordinary_line_maximum_b
y_ordinary_minimum = x * ordinary_line_minimum_a + ordinary_line_minimum_b
y_micro_maximum = x * micro_line_maximum_a + micro_line_maximum_b
y_micro_minimum = x * micro_line_minimum_a + micro_line_minimum_b


# plt.plot(dB_list_ordinary, freq_drift_ordinary, '.')
mean_ordinary_solar_maximum = np.mean(10 ** (np.array(dB_list_maximum_ordinary)/10))
mean_ordinary_solar_minimum = np.mean(10 ** (np.array(dB_list_minimum_ordinary)/10))
std_ordinary_solar_maximum = np.std(10 ** (np.array(dB_list_maximum_ordinary)/10))
std_ordinary_solar_minimum = np.std(10 ** (np.array(dB_list_minimum_ordinary)/10))
cor_ordinary_solar_maximum, p_value_ordinary_solar_maximum = stats.pearsonr(dB_list_maximum_ordinary, freq_drift_maximum_ordinary)
cor_ordinary_solar_minimum, p_value_ordinary_solar_minimum = stats.pearsonr(dB_list_minimum_ordinary, freq_drift_minimum_ordinary)
mean_micro_solar_maximum = np.mean(10 ** (np.array(dB_list_maximum_micro)/10))
mean_micro_solar_minimum = np.mean(10 ** (np.array(dB_list_minimum_micro)/10))
std_micro_solar_maximum = np.std(10 ** (np.array(dB_list_maximum_micro)/10))
std_micro_solar_minimum = np.std(10 ** (np.array(dB_list_minimum_micro)/10))
cor_micro_solar_maximum,p_value_micro_solar_maximum = stats.pearsonr(dB_list_maximum_micro, freq_drift_maximum_micro)
cor_micro_solar_minimum,p_value_micro_solar_minimum = stats.pearsonr(dB_list_minimum_micro, freq_drift_minimum_micro)

plt.plot(dB_list_maximum_ordinary, freq_drift_maximum_ordinary, '.', color = 'r', label = 'Around the solar maximum  '+str(len(dB_list_maximum_ordinary))+' events\nMean: ' + str(int(round(np.log10(mean_ordinary_solar_maximum) * 10,0)))+ '[dB] (' + str(int(round(mean_ordinary_solar_maximum,0)))+ ') '  + 'Std: (' + str(int(round(std_ordinary_solar_maximum,0))) + ')')
plt.plot(x, y_ordinary_maximum, color = 'r', label = 'r: '+ str(round(cor_ordinary_solar_maximum,2)) + '  p-value: '+ "3.6e-4")
plt.plot(dB_list_minimum_ordinary, freq_drift_minimum_ordinary, '.', color = 'b', label = 'Around the solar minimum '+str(len(dB_list_minimum_ordinary))+' events\nMean: ' + str(int(round(np.log10(mean_ordinary_solar_minimum) * 10,0)))+ '[dB] (' + str(int(round(mean_ordinary_solar_minimum,0)))+ ') '  + 'Std: (' + str(int(round(std_ordinary_solar_minimum,0))) + ')')
plt.plot(x, y_ordinary_minimum, color = 'b', label = 'r: '+ str(round(cor_ordinary_solar_minimum,2)) + '  p-value: '+ "0.81")

plt.title('Ordinary type Ⅲ burst', fontsize=12)
plt.ylabel('Frequency drift rates @40MHz[MHz/s]',fontsize=12)
plt.xlabel('from BG\n[Decibel]',fontsize=12)
plt.xlim(20, 55)
plt.ylim(2, 25)
plt.legend(fontsize=13)
plt.show()
plt.close()

# plt.plot(dB_list_micro, freq_drift_micro, '.')
std_micro_solar_maximum = np.std(10 ** (np.array(dB_list_maximum_micro)/10))
std_micro_solar_minimum = np.std(10 ** (np.array(dB_list_minimum_micro)/10))
plt.plot(dB_list_maximum_micro, freq_drift_maximum_micro, '.', color = 'r', label = 'Around the solar maximum '+str(len(dB_list_maximum_micro))+' events\nMean: ' + str(int(round(np.log10(mean_micro_solar_maximum) * 10,0)))+ '[dB] (' + str(int(round(mean_micro_solar_maximum,0)))+ ') '  + 'Std: (' + str(int(round(std_micro_solar_maximum,0))) + ')')
plt.plot(x, y_micro_maximum, color = 'r', label = 'r: '+ str(round(cor_micro_solar_maximum,2)) + '  p-value: '+ " 6.0e-13")
plt.plot(dB_list_minimum_micro, freq_drift_minimum_micro, '.', color = 'b', label = 'Around the solar minimum '+str(len(dB_list_minimum_micro))+' events\nMean: ' + str(int(round(np.log10(mean_micro_solar_minimum) * 10,0)))+ '[dB] (' + str(int(round(mean_micro_solar_minimum,0)))+ ') '  + 'Std: (' + str(int(round(std_micro_solar_minimum,0))) + ')')
plt.plot(x, y_micro_minimum, color = 'b', label = 'r: '+ str(round(cor_micro_solar_minimum,2)) + '  p-value: '+ "1.5e-5")
plt.title('Micro-type Ⅲ burst', fontsize=12)
plt.ylabel('Frequency drift rates @40MHz[MHz/s]',fontsize=12)
plt.xlabel('from BG\n[Decibel]',fontsize=12)
plt.xlim(20, 55)
plt.ylim(2, 25)
plt.legend(fontsize=13)
plt.show()
plt.close()


obs_daynight = []
for i in range(len(obs_time_micro)):
    hour_minute = obs_time_micro[i].strftime("%H:%M")
    obs_daynight.append(datetime.datetime(2013,1,1,int(hour_minute.split(':')[0]),int(hour_minute.split(':')[1])))

fig=plt.figure(1,figsize=(8,4))
ax1 = fig.add_subplot(111) 
ax1.plot(obs_daynight, dB_list_micro, '.')
ax1.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(date_format)
plt.title('Micro-type Ⅲ burst', fontsize=12)
plt.ylabel('from BG\n[Decibel]',fontsize=12)
plt.xlabel('Time',fontsize=12)
plt.ylim(20, 55)
plt.show()
    

obs_daynight = []
for i in range(len(obs_time_ordinary)):
    hour_minute = obs_time_ordinary[i].strftime("%H:%M")
    obs_daynight.append(datetime.datetime(2013,1,1,int(hour_minute.split(':')[0]),int(hour_minute.split(':')[1])))

fig=plt.figure(1,figsize=(8,4))
ax1 = fig.add_subplot(111) 
ax1.plot(obs_daynight, dB_list_ordinary, '.')
ax1.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(date_format)
plt.title('Ordinary type Ⅲ burst', fontsize=12)
plt.ylabel('from BG\n[Decibel]',fontsize=12)
plt.xlabel('Time',fontsize=12)
plt.ylim(20, 55)
plt.show()


# ordinary_line_a, ordinary_line_b = optimize.curve_fit(func_c1, power_list_ordinary, freq_drift_ordinary)[0]
# micro_line_a, micro_line_b = optimize.curve_fit(func_c1, power_list_micro, freq_drift_micro)[0]

# x = np.arange(20, 12000, 5)
# y_ordinary = x * ordinary_line_a + ordinary_line_b
# y_micro = x * micro_line_a + micro_line_b


# plt.plot(power_list_ordinary, freq_drift_ordinary, '.')
# plt.plot(x, y_ordinary)
# plt.title('Ordinary type Ⅲ burst', fontsize=12)
# plt.ylabel('Frequency drift rates @40MHz[MHz/s]',fontsize=12)
# plt.xlabel('from BG\n[Power]',fontsize=12)
# plt.xlim(0, 10000)
# plt.ylim(2, 12)
# plt.show()
# plt.close()

# plt.plot(power_list_micro, freq_drift_micro, '.')
# plt.plot(x, y_micro)
# plt.title('Micro type Ⅲ burst', fontsize=12)
# plt.ylabel('Frequency drift rates @40MHz[MHz/s]',fontsize=12)
# plt.xlabel('from BG\n[Decibel]',fontsize=12)
# plt.xlim(0, 10000)
# plt.ylim(2, 12)
# plt.show()
# plt.close()








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


    if burst_type == "Ordinary type Ⅲ burst":
        ordinary_solar_maximum_freqdrift_list = []
        ordinary_solar_maximum_freqdrift_list_all = []
        ordinary_solar_minimum_freqdrift_list = []
        ordinary_solar_minimum_freqdrift_list_all = []
        flare_index_list = []
        for obs_time_each in obs_time:
            flare_select_1 = flare_csv[flare_csv['start']<=(obs_time_each + pd.to_timedelta(5,unit='minute')).strftime("%Y/%m/%d %H:%M")]
            flare_select_2 = flare_select_1[flare_select_1['end']>=obs_time_each.strftime("%Y/%m/%d %H:%M")]
            if len(flare_select_2) == 1:
                flare_index = flare_select_2.index[0]
                flare_index_list.append(flare_index)
            else:
                print ('Error in flare list')
                sys.exit()
        #周波数ドリフト率のフレアごとの平均を求める
        freq_drift_mean_list = []
        obs_mean_list = []
        flare_index_final_list = list(set(flare_index_list))
        for period in analysis_period:
            fig, ax = plt.subplots(figsize=(14, 6))
            for flare_index_each in flare_index_final_list:
                flare_stime = datetime.datetime(int(flare_csv['start'][flare_index_each].split('/')[0]),int(flare_csv['start'][flare_index_each].split('/')[1]),int(flare_csv['start'][flare_index_each].split('/')[2].split(' ')[0]),int(flare_csv['start'][flare_index_each].split('/')[2].split(' ')[1].split(':')[0]),int(flare_csv['start'][flare_index_each].split('/')[2].split(' ')[1].split(':')[1]))
                flare_etime = datetime.datetime(int(flare_csv['end'][flare_index_each].split('/')[0]),int(flare_csv['end'][flare_index_each].split('/')[1]),int(flare_csv['end'][flare_index_each].split('/')[2].split(' ')[0]),int(flare_csv['end'][flare_index_each].split('/')[2].split(' ')[1].split(':')[0]),int(flare_csv['end'][flare_index_each].split('/')[2].split(' ')[1].split(':')[1]))
                obs_filter_1 = filter(lambda x: x >= flare_stime - pd.to_timedelta(5,unit='minute'), obs_time)
                obs_filter_2 = list(filter(lambda x: x <= flare_etime, obs_filter_1))
                freq_drift_mean = []
                obs_mean = []
                for i in range(len(obs_filter_2)):
                    freq_drift_mean.append(freq_drift[obs_time.index(obs_filter_2[i])])
                    obs_mean.append(obs_time[obs_time.index(obs_filter_2[i])])
                    if obs_time[obs_time.index(obs_filter_2[i])] >= solar_maximum[0]:
                        if obs_time[obs_time.index(obs_filter_2[i])] <= solar_maximum[1]:
                            ordinary_solar_maximum_freqdrift_list_all.append(freq_drift[obs_time.index(obs_filter_2[i])])

                    if obs_time[obs_time.index(obs_filter_2[i])] >= solar_minimum[0]:
                        if obs_time[obs_time.index(obs_filter_2[i])] <= solar_minimum[1]:
                            # print (obs_time[obs_time.index(obs_filter_2[i])] )
                            ordinary_solar_minimum_freqdrift_list_all.append(freq_drift[obs_time.index(obs_filter_2[i])])

                if period == solar_maximum:
                    freq_drift_mean_list.append(np.mean(freq_drift_mean))
                    obs_mean_list.append(sorted(obs_mean)[0] + (sorted(obs_mean)[0] - sorted(obs_mean)[-1])/2)






                if len(freq_drift_mean_list) >= error_threshold:
                    ax.errorbar(sorted(obs_mean)[0] + (sorted(obs_mean)[0] - sorted(obs_mean)[-1])/2, np.mean(freq_drift_mean), yerr=np.std(freq_drift_mean), color = 'b', marker = 's')
                    # if period == solar_minimum:
                    #     print(sorted(obs_mean)[0] + (sorted(obs_mean)[0] - sorted(obs_mean)[-1])/2, np.mean(freq_drift_mean))
                # else:
                    # ax.scatter(time_mean, freq_drift_mean, color = 'b')
            sobs_time_period = period[0]
            eobs_time_period = period[1]
            sobs_time = []
            eobs_time = []
            month_frequency_mean = []
            month_frequency_std = []
            time_mean_list = []
            mobs_time = []
            while eobs_time_period > sobs_time_period:
                month_freq_drift = []
                for j in range(len(obs_mean_list)):
                    if obs_mean_list[j] >= datetime.datetime(period[0].year, 1, 1)+relativedelta.relativedelta(months=analysis_move_ave*i) and obs_mean_list[j] <= datetime.datetime(period[0].year, 1, 1)+ relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1):
                        month_freq_drift.append(freq_drift_mean_list[j])
                        if period == solar_maximum:
                            ordinary_solar_maximum_freqdrift_list.append(freq_drift_mean_list[j])
                        else:
                            pass
                        if period == solar_minimum:
                            ordinary_solar_minimum_freqdrift_list.append(freq_drift_mean_list[j])
                        else:
                            pass
                            # print ('Error')
                            # sys.exit()

                if len(month_freq_drift)>0:
                    sobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i))
                    eobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1))
                    month_frequency_mean.append(np.mean(month_freq_drift))
                    month_frequency_std.append(np.std(month_freq_drift))
                    if not analysis_move_ave == 1:
                        mobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(months=analysis_move_ave/2)))
                    elif analysis_move_ave == 1:
                        mobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(days=15)))
                sobs_time_period += relativedelta.relativedelta(months=analysis_move_ave)
                i += 1
        
            for i in range (len(sobs_time)):
                # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
                if i == 0:
                    # ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average \nObserved more than ' +  str(average_threshold) + ' events a day', linewidth = 5.0)
                    ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average', linewidth = 5.0)
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
            plt.ylim(1.5,15)
            plt.show()
            plt.close()
            if period == solar_maximum:
                freq_drift_day_list.append(ordinary_solar_maximum_freqdrift_list)
                freq_drift_each_active_list.append(ordinary_solar_maximum_freqdrift_list)
            elif period == solar_minimum:
                freq_drift_day_list.append(ordinary_solar_minimum_freqdrift_list)
                freq_drift_each_active_list.append(ordinary_solar_minimum_freqdrift_list)
            else:
                print ('Error')
                sys.exit()
            
            
        ##############################################################################################
        #連続して発生したバーストを1日ごとに解析
        #周波数ドリフト率
    if burst_type == "Micro-type Ⅲ burst":
        micro_solar_maximum_freqdrift_list_all = []
        micro_solar_minimum_freqdrift_list_all = []
        start_time = []
        end_time = []
        list_start_num = []
        list_end_num = []
        for i in range(len(obs_time)):
            if i == 0:
                start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                list_start_num.append(i)
                if not obs_time[0].strftime("%Y%m%d") == obs_time[1].strftime("%Y%m%d"):
                    end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_end_num.append(i)
            elif i == len(obs_time) - 1:
                end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                list_end_num.append(i)
                if not obs_time[-1].strftime("%Y%m%d") == obs_time[-2].strftime("%Y%m%d"):
                    start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_start_num.append(i)
            else:
                if not (obs_time[i].strftime("%Y%m%d") == obs_time[i-1].strftime("%Y%m%d")) and not (obs_time[i].strftime("%Y%m%d") == obs_time[i+1].strftime("%Y%m%d")):
                    start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_start_num.append(i)
                    end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_end_num.append(i)
                else:
                    if not (obs_time[i].strftime("%Y%m%d") == obs_time[i-1].strftime("%Y%m%d")):
                        # print (str(obs_time_micro[i]) + '-')
                        start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                        list_start_num.append(i)
                    if not (obs_time[i].strftime("%Y%m%d") == obs_time[i+1].strftime("%Y%m%d")):
                        # print ('-' + str(obs_time_micro[i]))
                        end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
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
                        for micro_freq_drift_each in freq_drift[list_start_num[i]:list_end_num[i] + 1]:
                            if period == solar_maximum:
                                micro_solar_maximum_freqdrift_list_all.append(micro_freq_drift_each)
                            if period == solar_minimum:
                                micro_solar_minimum_freqdrift_list_all.append(micro_freq_drift_each)
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
                    if time_mean_list[j] >= datetime.datetime(period[0].year, 1, 1)+relativedelta.relativedelta(months=analysis_move_ave*i) and time_mean_list[j] <= datetime.datetime(period[0].year, 1, 1)+ relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1):
                        month_freq_drift.append(freq_drift_mean_list[j])
                if len(month_freq_drift)>0:
                    sobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i))
                    eobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1))
                    month_frequency_mean.append(np.mean(month_freq_drift))
                    month_frequency_std.append(np.std(month_freq_drift))
                    if not analysis_move_ave == 1:
                        mobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(months=analysis_move_ave/2)))
                    elif analysis_move_ave == 1:
                        mobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(days=15)))
        
            for i in range (len(sobs_time)):
                # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
                if i == 0:
                    # ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average \nObserved more than ' +  str(average_threshold) + ' event a day', linewidth = 5.0)
                    ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average', linewidth = 5.0)
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
            plt.ylim(1.5,15)
            plt.show()
            plt.close()
            freq_drift_day_list.append(freq_drift_mean_list)

    
    ##############################################################################################
    #連続して発生したバーストを合わせて解析

        start_time = []
        end_time = []
        list_start_num = []
        list_end_num = []
        for i in range(len(obs_time)):
            if i == 0:
                start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                list_start_num.append(i)
                if not obs_time[0].strftime("%Y%m%d") == obs_time[1].strftime("%Y%m%d"):
                    end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_end_num.append(i)
    
            elif i == len(obs_time) - 1:
                end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                list_end_num.append(i)
                if not obs_time[-1].strftime("%Y%m%d") == obs_time[-2].strftime("%Y%m%d"):
                    start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_start_num.append(i)
            else:
                if not (obs_time[i].strftime("%Y%m%d") == obs_time[i-1].strftime("%Y%m%d") or obs_time[i].strftime("%Y%m%d") == (obs_time[i-1] + relativedelta.relativedelta(days=1)).strftime("%Y%m%d")) and not ((obs_time[i] + relativedelta.relativedelta(days=1)).strftime("%Y%m%d") == obs_time[i+1].strftime("%Y%m%d") or obs_time[i].strftime("%Y%m%d") == obs_time[i+1].strftime("%Y%m%d")):
                    start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_start_num.append(i)
                    end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                    list_end_num.append(i)
                else:
                    if not (obs_time[i].strftime("%Y%m%d")  == obs_time[i-1].strftime("%Y%m%d") or obs_time[i].strftime("%Y%m%d")  == (obs_time[i-1] + relativedelta.relativedelta(days=1)).strftime("%Y%m%d")):
                        # print (str(obs_time_micro[i]) + '-')
                        start_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
                        list_start_num.append(i)
                    if not ((obs_time[i] + relativedelta.relativedelta(days=1)).strftime("%Y%m%d") == obs_time[i+1].strftime("%Y%m%d") or obs_time[i].strftime("%Y%m%d") == obs_time[i+1].strftime("%Y%m%d")):
                        # print ('-' + str(obs_time_micro[i]))
                        end_time.append(datetime.datetime(int(obs_time[i].strftime("%Y%m%d")[:4]), int(obs_time[i].strftime("%Y%m%d")[4:6]),int(obs_time[i].strftime("%Y%m%d")[6:8]), 12))
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
                    if time_mean_list[j] >= datetime.datetime(period[0].year, 1, 1)+relativedelta.relativedelta(months=analysis_move_ave*i) and time_mean_list[j] <= datetime.datetime(period[0].year, 1, 1)+ relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1):
                        month_freq_drift.append(freq_drift_mean_list[j])
                if len(month_freq_drift)>0:
                    sobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i))
                    eobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1))
                    month_frequency_mean.append(np.mean(month_freq_drift))
                    month_frequency_std.append(np.std(month_freq_drift))
                    if not analysis_move_ave == 1:
                        mobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(months=analysis_move_ave/2)))
                    elif analysis_move_ave == 1:
                        mobs_time.append(datetime.datetime(period[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(days=15)))

            for i in range (len(sobs_time)):
                # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
                if i == 0:
                    # ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average \nObserved more than ' +  str(average_threshold) + ' events\nfrom same active region', linewidth = 5.0)
                    ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average', linewidth = 5.0)
                else:
                    ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', linewidth = 5.0)
                
            for i in range (len(sobs_time)):
                if eobs_time[i] + relativedelta.relativedelta(days=1) in sobs_time:
                    ax.plot([eobs_time[i], sobs_time[i+1]], [month_frequency_mean[i], month_frequency_mean[i+1]], color = 'r', linewidth = 5.0)
        
            plt.title(burst_type + ' @' + str(freq_check) + '[MHz]',fontsize=25)
            plt.ylabel('Frequency drift rates[MHz/s]',fontsize=25)
            plt.xlim(period)
            plt.ylim(1.5,15)
            plt.tick_params(labelsize=20)
            plt.legend(fontsize = 24)
            plt.show()
            plt.close()
            freq_drift_each_active_list.append(freq_drift_mean_list)








# # ##############################################################################################
# # #連続して発生したバーストを1日ごとに解析
# # #周波数ドリフト率
# #     start_time = []
# #     end_time = []
# #     list_start_num = []
# #     list_end_num = []
# #     for i in range(len(obs_time)):
# #         if i == 0:
# #             start_time.append(obs_time[i])
# #             list_start_num.append(i)
# #             if not obs_time[0] == obs_time[1]:
# #                 end_time.append(obs_time[i])
# #                 list_end_num.append(i)
# #         elif i == len(obs_time) - 1:
# #             end_time.append(obs_time[i])
# #             list_end_num.append(i)
# #             if not obs_time[-1] == obs_time[-2]:
# #                 start_time.append(obs_time[i])
# #                 list_start_num.append(i)
# #         else:
# #             if not (obs_time[i] == obs_time[i-1]) and not (obs_time[i] == obs_time[i+1]):
# #                 start_time.append(obs_time[i])
# #                 list_start_num.append(i)
# #                 end_time.append(obs_time[i])
# #                 list_end_num.append(i)
# #             else:
# #                 if not (obs_time[i] == obs_time[i-1]):
# #                     # print (str(obs_time_micro[i]) + '-')
# #                     start_time.append(obs_time[i])
# #                     list_start_num.append(i)
# #                 if not (obs_time[i] == obs_time[i+1]):
# #                     # print ('-' + str(obs_time_micro[i]))
# #                     end_time.append(obs_time[i])
# #                     list_end_num.append(i)
# #                 else:
# #                     pass

# # ###############################################
# #     #Count_number
# #     x_list = []
# #     y_list = []
# #     for i in range (len(start_time)):
# #         start = start_time[i]
# #         end = end_time[i]
# #         while start <= end:
# #             x_list.append(start)
# #             y_list.append(list_end_num[i] - list_start_num[i])
# #             start += relativedelta.relativedelta(days=1)
    
# #     # fig, ax = plt.subplots(figsize=(14, 6))
# #     # ax.scatter(x_list, y_list)
# #     # plt.title(burst_type,fontsize=25)
# #     # plt.tick_params(labelsize=20)
# #     # plt.ylabel('Occurence Number',fontsize=25)
# #     # plt.show()
# #     # plt.close()



# # ###############################################
# #     #連続で観測されたデータの平均

# #     for period in analysis_period:
# #         freq_drift_mean_list = []
# #         freq_drift_std_list = []
# #         time_mean_list = []
# #         fig, ax = plt.subplots(figsize=(14, 6))
# #         for i in range(len(start_time)):
# # ###############################################
# #             #太陽極大期と極小期のセレクト
# #             if period[0] <= start_time[i] and period[1] >= start_time[i]:
# #     ###############################################
# #         #平均のデータの選定
# #                 if len(freq_drift[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
# #         ###############################################
# #                     # print (i)
# #                     freq_drift_mean = np.mean(freq_drift[list_start_num[i]:list_end_num[i] + 1])
# #                     freq_drift_std = np.std(freq_drift[list_start_num[i]:list_end_num[i] + 1])
# #                     time_mean = start_time[i] + (end_time[i] - start_time[i])/2
# #                     freq_drift_mean_list.append(freq_drift_mean)
# #                     freq_drift_std_list.append(freq_drift_std)
# #                     time_mean_list.append(time_mean)
# #                     if len(freq_drift[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
# #                         ax.errorbar(time_mean, freq_drift_mean, yerr=freq_drift_std, color = 'b', marker = 's')
# #                     # else:
# #                         # ax.scatter(time_mean, freq_drift_mean, color = 'b')

# #     ###############################################
# #         #移動平均のプロット
# #         sobs_time = []
# #         eobs_time = []
# #         mobs_time = []
# #         month_frequency_mean = []
# #         month_frequency_std = []
# #         for i in range(60):
# #             month_freq_drift = []
# #             for j in range(len(time_mean_list)):
# #                 if time_mean_list[j] >= date(time_mean_list[0].year, 1, 1)+relativedelta.relativedelta(months=analysis_move_ave*i) and time_mean_list[j] <= date(time_mean_list[0].year, 1, 1)+ relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1):
# #                     month_freq_drift.append(freq_drift_mean_list[j])
# #             if len(month_freq_drift)>0:
# #                 sobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i))
# #                 eobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*(i+1)) - relativedelta.relativedelta(days=1))
# #                 month_frequency_mean.append(np.mean(month_freq_drift))
# #                 month_frequency_std.append(np.std(month_freq_drift))
# #                 if not analysis_move_ave == 1:
# #                     mobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(months=analysis_move_ave/2)))
# #                 elif analysis_move_ave == 1:
# #                     mobs_time.append(date(time_mean_list[0].year, 1, 1) + relativedelta.relativedelta(months=analysis_move_ave*i) + (relativedelta.relativedelta(days=15)))
    
# #         for i in range (len(sobs_time)):
# #             # ax.errorbar(mobs_time[i], month_velocity_mean[i], yerr=month_frequency_std[i], capsize=3, color = 'r')
# #             if i == 0:
# #                 ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', label = str(analysis_move_ave) + ' months average \nObserved more than ' +  str(average_threshold) + ' events a day', linewidth = 5.0)
# #             else:
# #                 ax.plot([sobs_time[i], eobs_time[i]], [month_frequency_mean[i], month_frequency_mean[i]], color = 'r', linewidth = 5.0)
            
# #         for i in range (len(sobs_time)):
# #             if eobs_time[i] + relativedelta.relativedelta(days=1) in sobs_time:
# #                 ax.plot([eobs_time[i], sobs_time[i+1]], [month_frequency_mean[i], month_frequency_mean[i+1]], color = 'r', linewidth = 5.0)
# #     ###############################################
# #         plt.title(burst_type + ' @' + str(freq_check) + '[MHz]',fontsize=25)
# #         plt.ylabel('Frequency drift rates[MHz/s]',fontsize=25)
# #         plt.tick_params(labelsize=20)
# #         plt.legend(fontsize = 24)
# #         plt.xlim(period)
# #         # plt.show()
# #         plt.close()
# #         freq_drift_day_list.append(freq_drift_mean_list)



# # #############################################################################################################################################
# frequency_hist_analysis_solar_cycle_dependence()
# # # frequency_hist_analysis_micro_ordinary_solar_cycle_dependence()
# # ###############################################


# burst_types = ["Micro-type Ⅲ burst", "Ordinary type Ⅲ burst"]
# for burst_type in burst_types:
#     if burst_type == "Micro-type Ⅲ burst":
#         freq_start = freq_start_micro
#         freq_end = freq_end_micro
#         duration = duration_micro
#         obs_time_all = obs_time_all_micro
#     elif burst_type == "Ordinary type Ⅲ burst":
#         freq_start = freq_start_ordinary
#         freq_end = freq_end_ordinary
#         duration = duration_ordinary
#         obs_time_all = obs_time_all_ordinary
#     else:
#         break
        
# ##############################################################################################
# #連続して発生したバーストを合わせて解析
#     start_time = []
#     end_time = []
#     list_start_num = []
#     list_end_num = []
#     for i in range(len(obs_time_all)):
#         if i == 0:
#             start_time.append(obs_time_all[i])
#             list_start_num.append(i)
#             if not obs_time_all[0] == obs_time_all[1]:
#                 end_time.append(obs_time_all[i])
#                 list_end_num.append(i)
#         elif i == len(obs_time_all) - 1:
#             end_time.append(obs_time_all[i])
#             list_end_num.append(i)
#             if not obs_time_all[-1] == obs_time_all[-2]:
#                 start_time.append(obs_time_all[i])
#                 list_start_num.append(i)
#         else:
#             if not (obs_time_all[i] == obs_time_all[i-1] or obs_time_all[i] == obs_time_all[i-1] + relativedelta.relativedelta(days=1)) and not (obs_time_all[i] + relativedelta.relativedelta(days=1) == obs_time_all[i+1] or obs_time_all[i] == obs_time_all[i+1]):
#                 start_time.append(obs_time_all[i])
#                 list_start_num.append(i)
#                 end_time.append(obs_time_all[i])
#                 list_end_num.append(i)
#             else:
#                 if not (obs_time_all[i] == obs_time_all[i-1] or obs_time_all[i] == obs_time_all[i-1] + relativedelta.relativedelta(days=1)):
#                     # print (str(obs_time_micro[i]) + '-')
#                     start_time.append(obs_time_all[i])
#                     list_start_num.append(i)
#                 if not (obs_time_all[i] + relativedelta.relativedelta(days=1) == obs_time_all[i+1] or obs_time_all[i] == obs_time_all[i+1]):
#                     # print ('-' + str(obs_time_micro[i]))
#                     end_time.append(obs_time_all[i])
#                     list_end_num.append(i)
#                 else:
#                     pass
# ###############################################
#     #連続で観測されたデータの平均
#     for period in analysis_period:
#         freq_start_mean_list = []
#         freq_end_mean_list = []
#         duration_mean_list = []
#         time_mean_list = []
#         fig, ax = plt.subplots(figsize=(14, 6))
#         for i in range(len(start_time)):
# ###############################################
#             #太陽極大期と極小期のセレクト
#             if period[0] <= start_time[i] and period[1] >= start_time[i]:
#             ###############################################
#             #平均のデータの選定
#                     if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
#             ###############################################
#                         # print (i)
#                         freq_start_mean = np.mean(freq_start[list_start_num[i]:list_end_num[i] + 1])
#                         freq_start_mean_list.append(freq_start_mean)
#                         freq_start_std = np.std(freq_start[list_start_num[i]:list_end_num[i] + 1])
#                         freq_end_mean = np.mean(freq_end[list_start_num[i]:list_end_num[i] + 1])
#                         freq_end_mean_list.append(freq_end_mean)
#                         freq_end_std = np.std(freq_end[list_start_num[i]:list_end_num[i] + 1])
#                         duration_mean = np.mean(duration[list_start_num[i]:list_end_num[i] + 1])
#                         duration_mean_list.append(duration_mean)
#                         duration_std = np.std(duration[list_start_num[i]:list_end_num[i] + 1])
#                         time_mean = start_time[i] + (end_time[i] - start_time[i])/2
#                         time_mean_list.append(time_mean)
#                         if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
#                             ax.errorbar(time_mean, freq_start_mean, yerr=freq_start_std, color = 'b', marker = 's')
#         plt.title(burst_type + '\nObserved more than ' +  str(average_threshold) + ' events\nfrom same active region', fontsize=25)
#         plt.ylabel('Start Frequency[MHz]',fontsize=25)
#         plt.tick_params(labelsize=20)
#         plt.legend(fontsize = 24)
#         # plt.show()
#         plt.close()
#         start_frequency_each_active_list.append(freq_start_mean_list)
#         end_frequency_each_active_list.append(freq_end_mean_list)
#         duration_each_active_list.append(duration_mean_list)



# # ##############################################################################################
# # #連続して発生したバーストを1日ごとに解析
# #     start_time = []
# #     end_time = []
# #     list_start_num = []
# #     list_end_num = []
# #     for i in range(len(obs_time_all)):
# #         if i == 0:
# #             start_time.append(obs_time_all[i])
# #             list_start_num.append(i)
# #             if not obs_time_all[0] == obs_time_all[1]:
# #                 end_time.append(obs_time_all[i])
# #                 list_end_num.append(i)
# #         elif i == len(obs_time_all) - 1:
# #             end_time.append(obs_time_all[i])
# #             list_end_num.append(i)
# #             if not obs_time_all[-1] == obs_time_all[-2]:
# #                 start_time.append(obs_time_all[i])
# #                 list_start_num.append(i)
# #         else:
# #             if not (obs_time_all[i] == obs_time_all[i-1]) and not (obs_time_all[i] == obs_time_all[i+1]):
# #                 start_time.append(obs_time_all[i])
# #                 list_start_num.append(i)
# #                 end_time.append(obs_time_all[i])
# #                 list_end_num.append(i)
# #             else:
# #                 if not (obs_time_all[i] == obs_time_all[i-1]):
# #                     # print (str(obs_time_micro[i]) + '-')
# #                     start_time.append(obs_time_all[i])
# #                     list_start_num.append(i)
# #                 if not (obs_time_all[i] == obs_time_all[i+1]):
# #                     # print ('-' + str(obs_time_micro[i]))
# #                     end_time.append(obs_time_all[i])
# #                     list_end_num.append(i)
# #                 else:
# #                     pass
# # ###############################################
# #     #連続で観測されたデータの平均
# #     for period in analysis_period:
# #         freq_start_mean_list = []
# #         freq_end_mean_list = []
# #         duration_mean_list = []
# #         time_mean_list = []
# #         fig, ax = plt.subplots(figsize=(14, 6))
# #         for i in range(len(start_time)):
# # ###############################################
# #             #太陽極大期と極小期のセレクト
# #             if period[0] <= start_time[i] and period[1] >= start_time[i]:
# #         ###############################################
# #         #平均のデータの選定
# #                 if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= average_threshold:
# #         ###############################################
# #                     # print (i)
# #                     freq_start_mean = np.mean(freq_start[list_start_num[i]:list_end_num[i] + 1])
# #                     freq_start_mean_list.append(freq_start_mean)
# #                     freq_start_std = np.std(freq_start[list_start_num[i]:list_end_num[i] + 1])
# #                     freq_end_mean = np.mean(freq_end[list_start_num[i]:list_end_num[i] + 1])
# #                     freq_end_mean_list.append(freq_end_mean)
# #                     freq_end_std = np.std(freq_end[list_start_num[i]:list_end_num[i] + 1])
# #                     duration_mean = np.mean(duration[list_start_num[i]:list_end_num[i] + 1])
# #                     duration_mean_list.append(duration_mean)
# #                     duration_std = np.std(duration[list_start_num[i]:list_end_num[i] + 1])
# #                     time_mean = start_time[i] + (end_time[i] - start_time[i])/2
# #                     time_mean_list.append(time_mean)
# #                     if len(freq_start[list_start_num[i]:list_end_num[i] + 1]) >= error_threshold:
# #                         ax.errorbar(time_mean, freq_start_mean, yerr=freq_start_std, color = 'b', marker = 's')
# #         plt.title(burst_type + '\nObserved more than ' +  str(average_threshold) + ' events a day', fontsize=25)
# #         plt.ylabel('Start Frequency[MHz]',fontsize=25)
# #         plt.tick_params(labelsize=20)
# #         plt.legend(fontsize = 24)
# #         # plt.show()
# #         plt.close()
# #         start_frequency_day_list.append(freq_start_mean_list)
# #         end_frequency_day_list.append(freq_end_mean_list)
# #         duration_day_list.append(duration_mean_list)

# ###############################################
# start_end_duration_hist_analysis_solar_cycle_dependence()
# ###############################################
# # start_end_duration_hist_analysis_micro_ordinary_solar_cycle_dependence()




















  # for period in ['Around solar maximum', 'Around solar minimum']:
  #    text = ' observed events'

  #    if period == 'Around solar maximum':
  #        freq_drift_micro = micro_solar_maximum_freqdrift_list_all
  #        freq_drift_ordinary = ordinary_solar_maximum_freqdrift_list_all

  #    elif period == 'Around solar minimum':
  #        freq_drift_micro = micro_solar_minimum_freqdrift_list_all
  #        freq_drift_ordinary = ordinary_solar_minimum_freqdrift_list_all

  #    if len(freq_drift_micro) > 0 and len(freq_drift_ordinary) > 0:
  #        if max(freq_drift_micro) >= max(freq_drift_ordinary):
  #            max_val = max(freq_drift_micro)
  #        else:
  #            max_val = max(freq_drift_ordinary)

  #        if min(freq_drift_micro) <= min(freq_drift_ordinary):
  #            min_val = min(freq_drift_micro)
  #        else:
  #            min_val = min(freq_drift_ordinary)
  #        bin_size = 8

  #        x_hist = (plt.hist(freq_drift_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
  #        y_hist = (plt.hist(freq_drift_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_ordinary))
  #        x_hist_1 = (plt.hist(freq_drift_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
  #        y_hist_1 = (plt.hist(freq_drift_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_micro))
  #        plt.close()
  #        width = x_hist[1]-x_hist[0]
  #        for i in range(len(y_hist)):
  #            if i == 0:
  #                plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3, label =  'Ordinary type Ⅲ burst\n'+ str(len(freq_drift_ordinary)) + ' events')
  #                plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(freq_drift_micro)) + ' events')
  #            else:
  #                plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = 'r', alpha = 0.3)
  #                plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = 'b', alpha = 0.3)
  #            plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + period + text)
  #        # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
  #            plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
  #            plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
  #            plt.ylabel('Occurrence rate',fontsize=15)
  #            # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
  #            plt.xticks(rotation = 20)
  #        plt.show()
  #        plt.close()