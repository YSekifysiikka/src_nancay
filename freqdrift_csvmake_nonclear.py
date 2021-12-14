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



# start_check_list = []
# end_check_list = []
# duration_list = []
# factor_list = []
# peak_time_list = []
# peak_freq_list = []
# resi_list = []
# velocity_list = []
# base_obs_time_list = []
# factor_list_5 = [1,2,3,4,5]
# Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
# Parent_lab = len(Parent_directory.split('/')) - 1

# with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/ordinary/solarmintotal.csv', 'w') as f:
#     w = csv.DictWriter(f, fieldnames=["obs_time", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list"])
#     w.writeheader()
    
#     files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/ordinary/solarmin/*compare*.csv')
#     for file_final in files:
#         csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")
#         obs_date = file_final.split('/')[-1].split('_')[0]
#         obs_time = file_final.split('/')[-1].split('_')[1]
#         obs_time = datetime.datetime(int(obs_date[:4]), int(obs_date[4:6]), int(obs_date[6:8]), int(obs_time[:2]), int(obs_time[2:4]), int(obs_time[4:6]))
        
        

#         for j in range(len(csv_input_final)):
#             start_check_list.append(csv_input_final["freq_start"][j])
#             base_obs_time_list.append(obs_time)
#             end_check_list.append(csv_input_final["freq_end"][j])
#             duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
#             # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
#             factor_list.append(csv_input_final["factor"][j])
#             peak_time_list.append([int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')])
#             peak_freq_list.append([float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')])
#             resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
#             resi_list.append([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')][resi_idx])
#             velocity_list.append([float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')])
        
#     for i in range(len(peak_freq_list)):
#         freq_list_new = peak_freq_list[i]
#         time_list_new = peak_time_list[i]
#         residual_list, x_time, y_freq, time_rate_final = residual_detection(factor_list_5, freq_list_new , time_list_new)
#         resi_idx = np.argmin(residual_list)
#         # selected_event_plot(freq_list_new, time_list_new, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], sep_arr_sep_time_list, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, s_event_time, e_event_time, s_event_freq, e_event_freq, selected_Frequency, resi_idx, date_event_hour, date_event_minute)
#         time_gap_arr = x_time[resi_idx][np.where(np.array(y_freq[resi_idx]) == freq_list_new[0])[0][0]:np.where(np.array(y_freq[resi_idx]) == freq_list_new[-1])[0][0] + 1] - np.array(time_list_new)
#         delete_idx = np.where(np.abs(time_gap_arr) >= 2 * residual_list[resi_idx])[0]
#         selected_idx = np.where(np.abs(time_gap_arr) < 2 * residual_list[resi_idx])[0]
    
#         residual_list, x_time, y_freq, time_rate_final = residual_detection(factor_list_5, np.array(freq_list_new)[selected_idx], np.array(time_list_new)[selected_idx])
#         resi_idx = np.argmin(residual_list)
#         obs_time = base_obs_time_list[i] + datetime.timedelta(seconds = int(np.min(np.array(time_list_new)[selected_idx])))
#         if time_rate_final == velocity_list[i]:
#             print ('yes')
#             w.writerow({'obs_time':obs_time,'velocity':time_rate_final, 'residual':residual_list, 'event_start': np.min(np.array(time_list_new)[selected_idx]),'event_end': np.max(np.array(time_list_new)[selected_idx]),'freq_start': np.array(freq_list_new)[selected_idx][0],'freq_end':np.array(freq_list_new)[selected_idx][-1], 'factor':resi_idx+1, 'peak_time_list':np.array(time_list_new)[selected_idx], 'peak_freq_list':np.array(freq_list_new)[selected_idx]})
#         else:
#             print ('Error')
#             sys.exit()









start_check_list = []
end_check_list = []
duration_list = []
factor_list = []
peak_time_list = []
peak_freq_list = []
resi_list = []
velocity_list = []
base_obs_time_list = []
obs_time_list = []
driftrate_list = []
factor_list_5 = [1,2,3,4,5]
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
sun_to_earth = 150000000
sun_radius = 696000
light_v = 300000 #[km/s]
time_rate = 0.13

def numerical_diff_df_dn(ne):
    h = 1e-5
    f_1 = 9*np.sqrt(ne+h)/1e+3
    f_2 = 9*np.sqrt(ne-h)/1e+3
    return ((f_1 - f_2)/(2*h))

def numerical_diff_allen_dn_dr(factor, r):
    h = 1e-1
    ne_1 = factor * 10**8 * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))
    ne_2 = factor * 10**8 * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))
    return ((ne_1 - ne_2)/(2*h))

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/ordinary/solarmaxtotal2.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list", "drift_rate_40MHz"])
    w.writeheader()

    file_final = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis_nonclear/ordinary/solarmaxtotal.csv'
    csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")

    

    for j in range(len(csv_input_final)):
        obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
        obs_time_list.append(obs_time)
        start_check_list.append(csv_input_final["freq_start"][j])
        base_obs_time_list.append(obs_time)
        end_check_list.append(csv_input_final["freq_end"][j])
        duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
        # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
        factor_list.append(csv_input_final["factor"][j])
        peak_time_list.append([int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].replace('\n', '').split(' ') if k != ''])
        peak_freq_list.append([float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].replace('\n', '').split(' ') if k != ''])
        resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
        resi_list.append([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
        if [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')][resi_idx] <= 1.35:
            velocity = [float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]
            velocity_list.append(velocity)
            
            
            factor = csv_input_final["factor"][j]
            time_rate = velocity[resi_idx]
            cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
            r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
            r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
            ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
            # print ('\n'+str(factor)+'×B-A model' + 'emission fp')
            drift_rates = numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
            driftrate_list.append(drift_rates*(-1))
            # drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
            # print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_allen_dn_dr(factor, r)))
            w.writerow({'obs_time':obs_time,'velocity':velocity, 'residual':resi_list[j], 'event_start': csv_input_final["event_start"][j],'event_end': csv_input_final["event_end"][j],'freq_start': csv_input_final["freq_start"][j],'freq_end':csv_input_final["freq_end"][j], 'factor':resi_idx+1, 'peak_time_list':peak_time_list[j], 'peak_freq_list':peak_freq_list[j], 'drift_rate_40MHz': drift_rates*(-1)})
        else:
            print('No')