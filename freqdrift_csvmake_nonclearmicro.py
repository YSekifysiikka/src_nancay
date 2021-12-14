#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:58:01 2021

@author: yuichiro
"""

import glob
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import csv
import sys

from pynverse import inversefunc
def allen_model(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5)))))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    
#        print ('aaa')
#        print(fitting)
    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start)


def residual_detection(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    for factor in factor_list:
        slide, time_rate5, residual, h_start = allen_model(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))
        
        cube_4 = (lambda h5_0: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)


    return residual_list, x_time, y_freq, time_rate_final

def allen_model2(factor_num, time_list, freq_list, velocity):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1


    time_rate4 = velocity
    cube_4 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values=i_value)
    s_1 = sum(invcube_4-time_list)/len(freq_list)
    residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
    fitting_new.append(residual_1)
    time_rate_result_new.append(time_rate4)
    slide_result_new.append(s_1)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start)


def residual_detection2(factor, freq_list, time_list, velocity):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    slide, time_rate5, residual, h_start = allen_model2(factor, time_list, freq_list, velocity)
    time_rate_final.append(time_rate5)
    residual_list.append(residual)
    slide_list.append(slide)
        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))
        
    cube_4 = (lambda h5_0: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0) * time_rate5 * 300000)/696000)**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values=freq_list)
    x_time.append(invcube_4 + slide)
    y_freq.append(freq_list)


    return residual_list, x_time, y_freq, time_rate_final


start_check_list = []
end_check_list = []
duration_list = []
factor_list = []
peak_time_list = []
peak_freq_list = []
resi_list = []
velocity_list = []
base_obs_time_list = []
driftrates_list = []
factor_list_5 = [1,2,3,4,5]
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

# with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/micro/solarmaxtotal.csv', 'w') as f:
#     w = csv.DictWriter(f, fieldnames=["obs_time", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list", "drift_rate_40MHz"])
#     w.writeheader()
    
#     files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/micro/solarmax/*compare*.csv')
#     for file_final in files:
#         csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")
        

#         for j in range(len(csv_input_final)):
#             obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
#             freq_start = csv_input_final["freq_start"][j]
#             freq_end = csv_input_final["freq_end"][j]
#             time_start = csv_input_final["event_start"][j]
#             time_end = csv_input_final["event_end"][j]
#             drift_rates = csv_input_final["drift_rate_40MHz"][j]
#             driftrates_list.append(drift_rates)
#             # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
#             # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
#             factor_list = csv_input_final["factor"][j]
#             peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].replace('\n', '').split(' ') if k != '']
#             peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j][1:-1].replace('\n', '').split(' ') if k != '']
#             resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
#             resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
#             velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]
#             w.writerow({'obs_time':obs_time,'velocity':velocity_list, 'residual':resi_list, 'event_start': time_start,'event_end':time_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':resi_idx+1, 'peak_time_list':peak_time_list, 'peak_freq_list':peak_freq_list, 'drift_rate_40MHz': drift_rates})


driftrates_list = []
obs_time_list = []
file_final = Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/micro/solarmaxtotal.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")

for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    obs_time_list.append(obs_time)
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    driftrates_list.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',') if k != '']
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j][1:-1].split(',') if k != '']
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')][resi_idx]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]


figure_=plt.figure(1,figsize=(6,8))
plt.plot(obs_time_list,driftrates_list , '.')
plt.xticks(rotation=20)
plt.show()

max_val = 15.606127

min_val = 1.303744

bin_size = 8

color_2 = "r"

x_hist_1 = (plt.hist(driftrates_list, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
y_hist_1 = (plt.hist(driftrates_list, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(driftrates_list))

plt.close()
width = x_hist_1[1]-x_hist_1[0]
for i in range(len(y_hist_1)):
    if i == 0:
        
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Micro-type Ⅲ burst\n'+ str(len(driftrates_list)) + ' events')
    else:
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
    plt.title('Frequency drift rates @ 40[MH/z]')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
    plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
    plt.xticks(rotation = 20)
plt.show()
plt.close()