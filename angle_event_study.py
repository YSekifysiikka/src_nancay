#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 09:46:47 2021

@author: yuichiro
"""

import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np
import math
from pynverse import inversefunc

CR_list = [2216, 2020, 2125, 2126, 2134, 2147, 2200]
DR_ave_list = [7.0, 4.9, 7.2, 7.4, 8.4, 6.6, 4.8]
DR_std_list = [3.18, 0.94, 1.27, 3.12, 3.43, 1.94, 0]
x_list = np.arange(0,7,1)



velocity_fp = []
velocity_2fp = []
velocity_v2_fp = []
velocity_v2_2fp = []

total_velocity_fp = []
total_velocity_2fp = []
total_velocity_v2_fp = []
total_velocity_v2_2fp = []


Density_model_list = ['Wang min model', 'Wang max model', ['1×Newkirk model', '3×Newkirk model'], '1×Newkirk model']

burst_type_list = []
period_list = []


angle_fp_ave = []
angle_2fp_ave = []

angle_v2_fp_ave = []
angle_v2_2fp_ave = []

# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'

# files = glob.glob(Parent_directory  + '/solar_burst/magnet/analysis/*/*.csv')
# for file in files:
#     CR = file.split('/')[-1].split('.')[0]



for CR in CR_list:
    # if ((CR == 2020) | (CR == 2216)):
    if ((CR == 2200)):
        CR = str(CR)
        if (CR == '2216'):
            burst_type = 'Micro type III burst'
            period = 'Around the solar minimum'
            Density_model = Density_model_list[0]
        elif (CR == '2020'):
            burst_type = 'Micro type III burst'
            period = 'Around the solar maximum'
            Density_model = Density_model_list[1]
        elif ((CR == '2125') | (CR == '2134') | (CR == '2147') | (CR == '2126')):
            burst_type = 'Ordinary type III burst'
            period = 'Around the solar maximum'
            Density_model = Density_model_list[2]
        elif (CR == '2200'):
            burst_type = 'Ordinary type III burst'
            period = 'Around the solar minimum'
            Density_model = Density_model_list[3]
        burst_type_list.append(burst_type)
        period_list.append(period)
        file_final = Parent_directory  + '/solar_burst/magnet/analysis/'+CR+'/'+CR+'.csv'
        csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")
        
        
        angle_ave_list_all = []
        radius_list = []
        radius_start_list = []
        radius_end_list = []
        angle_ave_mean_list = []
        angle_ave_std_list = []
        
        # w = csv.DictWriter(f, fieldnames=["radius", "start", "end", "xmin", "xmax", "ymin", "ymax", "angle_ave_mean", "angle_ave_std", "angle_max_mean", "angle_max_std", "angle_ave_list", "angle_max_list"])
        for j in range(len(csv_input_final)):
            # obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
            radius = csv_input_final["radius"][j]
            angle_ave_list = [float(k) for k in csv_input_final["angle_ave_list"][j].split('[')[1].split(']')[0].split(',')]
            angle_ave_mean = csv_input_final["angle_ave_mean"][j]
            angle_ave_std = csv_input_final["angle_ave_std"][j]
            radius_start = csv_input_final["start"][j]
            radius_end = csv_input_final["end"][j]
            angle_ave_list_all.append(angle_ave_list)
            radius_list.append(radius)
            radius_start_list.append(radius_start)
            radius_end_list.append(radius_end)
            angle_ave_mean_list.append(angle_ave_mean)
            angle_ave_std_list.append(angle_ave_std)
        plt.title('CR:  '+ CR + ' ' + burst_type + '\n' + period)
        plt.plot(radius_list, angle_ave_mean_list, '.')
        plt.ylim(16,45)
        # plt.legend()
        plt.show()
        
        if Density_model =='Wang min model':
            s_radius_fp = 1.1
            e_radius_fp = 1.2
            s_radius_2fp = 1.4
            e_radius_2fp = 1.6
            radius_list = np.array(radius_list)
            angle_ave_mean_list = np.array(angle_ave_mean_list)
            radius_start_list = np.array(radius_start_list)
            radius_end_list = np.array(radius_end_list)


            idx = np.where((radius_start_list == s_radius_fp) & (radius_end_list == e_radius_fp))[0][0]
    
            CR_idx = CR_list.index(int(CR))
            # print (2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
    
            angle_fp_ave.append(angle_ave_mean_list[idx])
            angle_v2_fp_ave.append(angle_ave_mean_list[idx])
            freq = 40
            fp_cube = (lambda h: 9 * 1e-3 * np.sqrt((353766*((1+(h/69600000000))**(-1))+1.03359e+07*((1+(h/69600000000))**(-2))- 5.46541e+07*((1+(h/69600000000))**(-3))+ 8.24791e+07*((1+(h/69600000000))**(-4)))))
            s_freq = (freq + DR_ave_list[CR_idx] * 0.01)
            e_freq = (freq - DR_ave_list[CR_idx] * 0.01)
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_fp.append((e_radio-s_radio)/0.02/29979245800)
            velocity_v2_fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
            total_velocity_v2_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))


            angle_ave_list = np.array(angle_ave_list)
            plt.title('CR:  '+ CR + ' ' + burst_type + '\n' + period + ' f=fp R=1.15Rs')
            plt.hist(angle_ave_list_all[idx])
            plt.show()
            plt.close()

            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.2*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('fp Radial velocity')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
        
            idx = np.where((radius_start_list == s_radius_2fp) & (radius_end_list == e_radius_2fp))[0][0]
            angle_2fp_ave.append(angle_ave_mean_list[idx])
            angle_v2_2fp_ave.append(angle_ave_mean_list[idx])
            s_freq = (freq + DR_ave_list[CR_idx]*0.01)/2
            e_freq = (freq - DR_ave_list[CR_idx]*0.01)/2
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_2fp.append((e_radio-s_radio)/0.02/29979245800)
            velocity_v2_2fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
            total_velocity_v2_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))

            angle_ave_list = np.array(angle_ave_list)
            plt.title('CR:  '+ CR + ' ' + burst_type + '\n' + period + ' f=fp R=1.5Rs')
            plt.hist(angle_ave_list_all[idx])
            plt.show()
            plt.close()
    
            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.4*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('2fp Radial velocity')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
    
        elif Density_model =='Wang max model':
            s_radius_fp = 1.2
            e_radius_fp = 1.4
            s_radius_2fp = 1.6
            e_radius_2fp = 1.7
            
    
            radius_list = np.array(radius_list)
            angle_ave_mean_list = np.array(angle_ave_mean_list)
            radius_start_list = np.array(radius_start_list)
            radius_end_list = np.array(radius_end_list)          

            idx = np.where((radius_start_list == s_radius_fp) & (radius_end_list == e_radius_fp))[0][0]
    
            CR_idx = CR_list.index(int(CR))
            freq = 40
            fp_cube = (lambda h: 9 * 1e-3 * np.sqrt((-4.42158e+06*((1+(h/69600000000))**(-1))+5.41656e+07*((1+(h/69600000000))**(-2))- 1.86150e+08*((1+(h/69600000000))**(-3))+ 2.13102e+08*((1+(h/69600000000))**(-4)))))
            s_freq = (freq + DR_ave_list[CR_idx] * 0.01)
            e_freq = (freq - DR_ave_list[CR_idx] * 0.01)
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_fp.append((e_radio-s_radio)/0.02/29979245800)
            velocity_v2_fp.append((e_radio-s_radio)/0.02/29979245800)
            angle_fp_ave.append(angle_ave_mean_list[idx])
            angle_v2_fp_ave.append(angle_ave_mean_list[idx])
            total_velocity_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
            total_velocity_v2_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
        
            angle_ave_list = np.array(angle_ave_list)
            plt.title('CR:  '+ CR + ' ' + burst_type + '\n' + period + ' f=fp R=1.3Rs')
            plt.hist(angle_ave_list_all[idx])
            plt.show()
            plt.close()

            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.4*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('fp Radial velocity')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
    
    
            idx = np.where((radius_start_list == s_radius_2fp) & (radius_end_list == e_radius_2fp))[0][0]
            angle_2fp_ave.append(angle_ave_mean_list[idx])
            angle_v2_2fp_ave.append(angle_ave_mean_list[idx])
            s_freq = (freq + DR_ave_list[CR_idx]*0.01)/2
            e_freq = (freq - DR_ave_list[CR_idx]*0.01)/2
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_2fp.append((e_radio-s_radio)/0.02/29979245800)
            velocity_v2_2fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
            total_velocity_v2_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))

            angle_ave_list = np.array(angle_ave_list)
            plt.title('CR:  '+ CR + ' ' + burst_type + '\n' + period + ' f=2fp R=1.65Rs')
            plt.hist(angle_ave_list_all[idx])
            plt.show()
            plt.close()

            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.4*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('2fp Radial velocity')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()

    
        elif Density_model ==['1×Newkirk model', '3×Newkirk model']:
            s_radius_fp = 1.5
            e_radius_fp = 1.7
            s_radius_2fp = 1.9
            e_radius_2fp = 2.3
            s_radius3_fp = 1.9
            e_radius3_fp = 2.0
            radius_list = np.array(radius_list)
            angle_ave_mean_list = np.array(angle_ave_mean_list)
            radius_start_list = np.array(radius_start_list)
            radius_end_list = np.array(radius_end_list)
            idx = np.where((radius_start_list == s_radius_fp) & (radius_end_list == e_radius_fp))[0][0]
            angle_fp_ave.append(angle_ave_mean_list[idx])
            CR_idx = CR_list.index(int(CR))
            factor_velocity = 1
            freq = 40
            fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
            s_freq = (freq + DR_ave_list[CR_idx] * 0.01)
            e_freq = (freq - DR_ave_list[CR_idx] * 0.01)
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
    
            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.5*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('fp Radial velocity 1×Newkirk')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
    
            idx = np.where((radius_start_list == s_radius_2fp) & (radius_end_list == e_radius_2fp))[0][0]
            angle_2fp_ave.append(angle_ave_mean_list[idx])
            s_freq = (freq + DR_ave_list[CR_idx]*0.01)/2
            e_freq = (freq - DR_ave_list[CR_idx]*0.01)/2
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_2fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
    
            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.5*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('2fp Radial velocity 1×Newkirk')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()

    
            idx = np.where((radius_start_list == s_radius3_fp) & (radius_end_list == e_radius3_fp))[0][0]
            angle_v2_fp_ave.append(angle_ave_mean_list[idx])
            factor_velocity = 3
            freq = 40
            fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
            s_freq = (freq + DR_ave_list[CR_idx] * 0.01)
            e_freq = (freq - DR_ave_list[CR_idx] * 0.01)
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_v2_fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_v2_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))


            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.5*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('fp Radial velocity 3×Newkirk')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
    
            angle_v2_2fp_ave.append(0)
            s_freq = (freq + DR_ave_list[CR_idx]*0.01)/2
            e_freq = (freq - DR_ave_list[CR_idx]*0.01)/2
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_v2_2fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_v2_2fp.append((e_radio-s_radio)/0.02/29979245800)
            
            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.5*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('2fp Radial velocity 3×Newkirk')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()

        else:
            s_radius_fp = 1.5
            e_radius_fp = 1.7
            s_radius_2fp = 1.9
            e_radius_2fp = 2.3
            radius_list = np.array(radius_list)
            angle_ave_mean_list = np.array(angle_ave_mean_list)
            radius_start_list = np.array(radius_start_list)
            radius_end_list = np.array(radius_end_list)
            idx = np.where((radius_start_list == s_radius_fp) & (radius_end_list == e_radius_fp))[0][0]
            angle_fp_ave.append(angle_ave_mean_list[idx])
            angle_v2_fp_ave.append(angle_ave_mean_list[idx])
            CR_idx = CR_list.index(int(CR))
            factor_velocity = 1
            freq = 40
            fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
            s_freq = (freq + DR_ave_list[CR_idx] * 0.01)
            e_freq = (freq - DR_ave_list[CR_idx] * 0.01)
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_fp.append((e_radio-s_radio)/0.02/29979245800)
            velocity_v2_fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
            total_velocity_v2_fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))

            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.5*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('fp Radial velocity 1×Newkirk')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
    
            idx = np.where((radius_start_list == s_radius_2fp) & (radius_end_list == e_radius_2fp))[0][0]
            angle_2fp_ave.append(angle_ave_mean_list[idx])
            angle_v2_2fp_ave.append(angle_ave_mean_list[idx])
            s_freq = (freq + DR_ave_list[CR_idx]*0.01)/2
            e_freq = (freq - DR_ave_list[CR_idx]*0.01)/2
            s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
            e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
            velocity_2fp.append((e_radio-s_radio)/0.02/29979245800)
            velocity_v2_2fp.append((e_radio-s_radio)/0.02/29979245800)
            total_velocity_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
            total_velocity_v2_2fp.append(((e_radio-s_radio)/0.02/29979245800)/math.cos(math.radians(angle_ave_mean_list[idx])))
    
            velocity_hist_test = []
            for i in range(len(angle_ave_list_all[idx])):
                velocity_hist_test.append(0.5*math.cos(math.radians(angle_ave_list_all[idx][i])))
            plt.title('2fp Radial velocity 1×Newkirk')
            plt.hist(velocity_hist_test)
            plt.show()
            plt.close()
    

            