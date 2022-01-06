#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:23:38 2022

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
import glob

sun_to_earth = 150000000
sun_radius = 696000
light_v = 300000 #[km/s]

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

def numerical_diff_newkirk_dn_dr(factor, r):
    h = 1e-1
    ne_1 = factor * 4.2 * 10 ** (4+4.32/((r+h)/69600000000))
    ne_2 = factor * 4.2 * 10 ** (4+4.32/((r-h)/69600000000))
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_wangmin_dn_dr(factor, r):
    h = 1e-2
    ne_1 = factor * (353766/((r+h)/69600000000) + 1.03359e+07/((r+h)/69600000000)**2 - 5.46541e+07/((r+h)/69600000000)**3 + 8.24791e+07/((r+h)/69600000000)**4)
    ne_2 = factor * (353766/((r-h)/69600000000) + 1.03359e+07/((r-h)/69600000000)**2 - 5.46541e+07/((r-h)/69600000000)**3 + 8.24791e+07/((r-h)/69600000000)**4)
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_wangmax_dn_dr(factor, r):
    h = 1e-2
    ne_1 = factor * (-4.42158e+06/((r+h)/69600000000) + 5.41656e+07/((r+h)/69600000000)**2 - 1.86150e+08 /((r+h)/69600000000)**3 + 2.13102e+08/((r+h)/69600000000)**4)
    ne_2 = factor * (-4.42158e+06/((r-h)/69600000000) + 5.41656e+07/((r-h)/69600000000)**2 - 1.86150e+08 /((r-h)/69600000000)**3 + 2.13102e+08/((r-h)/69600000000)**4)
    return ((ne_1 - ne_2)/(2*h))



def driftrates_ci_plot(driftrates_ci_median, driftrates_ci_se):
    x = [driftrates_ci_median[0], driftrates_ci_median[1], driftrates_ci_median[2],driftrates_ci_median[3]] # 変数を初期化
    y = [8.5, 5.5, 3.5, 1.5]
    x_err = [driftrates_ci_se[0], driftrates_ci_se[1], driftrates_ci_se[2],driftrates_ci_se[3]]
    
    
    plt.figure(figsize=(4,7))
    plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=6, ecolor='r', markeredgecolor = "r", color='r')
    
    plt.plot([1, 17],[10, 10], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[9, 9], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[8, 8], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[7, 7], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[6, 6], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[5, 5], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[4, 4], color = "r", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[3, 3], color = "r", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[2, 2], color = "b", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[1, 1], color = "b", linewidth = 33.0, alpha = 0.1)
    
    
    
    plt.plot([1, 17],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([1, 17],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([1, 17],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)
    
    plt.plot([7.34, 7.34],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed", label = 'Zhang et al., 2018')
    plt.plot([5.29, 5.29],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed")
    
    plt.xlabel('Frequency drift rates [MHz/s]', fontsize=18)
    plt.ylabel('Type of bursts', fontsize=18)
    # plt.xlim(4, 9)

    plt.xlim(3, 11)
    plt.ylim(0.5, 12)
    plt.yticks([8.5,5.5,3.5,1.5], ['Ordinary type III burst\nAround the solar maximum', 'Ordinary type III burst\nAround the solar minimum', 'Micro type III burst\nAround the solar maximum', 'Micro type III burst\nAround the solar minimum'])
    plt.xticks(fontsize=16)
    plt.legend()
    plt.title('95% CI')
    ax = plt.gca()
    # ax.axes.yaxis.set_visible(False)
    # ax.axes.xaxis.set_visible(False)

    plt.show()

def driftrates_sd_plot(driftrates_ci_median, SD_list):
    x = [driftrates_ci_median[0], driftrates_ci_median[1], driftrates_ci_median[2],driftrates_ci_median[3]] # 変数を初期化
    y = [8.5, 5.5, 3.5, 1.5]
    x_err = [SD_list[0], SD_list[1], SD_list[2],SD_list[3]]
    
    
    plt.figure(figsize=(4,7))
    plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=6, ecolor='r', markeredgecolor = "r", color='r')
    
    plt.plot([1, 17],[10, 10], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[9, 9], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[8, 8], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[7, 7], color = "orange", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[6, 6], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[5, 5], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[4, 4], color = "r", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[3, 3], color = "r", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[2, 2], color = "b", linewidth = 33.0, alpha = 0.1)
    plt.plot([1, 17],[1, 1], color = "b", linewidth = 33.0, alpha = 0.1)
    
    
    
    plt.plot([1, 17],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([1, 17],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([1, 17],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)
    
    plt.plot([7.34, 7.34],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed", label = 'Zhang et al., 2018')
    plt.plot([5.29, 5.29],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed")
    
    plt.xlabel('Frequency drift rates [MHz/s]', fontsize=18)
    plt.ylabel('Type of bursts', fontsize=18)
    plt.xlim(3, 11)
    plt.ylim(0.5, 12)
    plt.yticks([8.5,5.5,3.5,1.5], ['Ordinary type III burst\nAround the solar maximum', 'Ordinary type III burst\nAround the solar minimum', 'Micro type III burst\nAround the solar maximum', 'Micro type III burst\nAround the solar minimum'])
    plt.xticks(fontsize=16)
    plt.legend()
    plt.title('Average ± 2σ')
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    plt.show()


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'


obs_time_micro = []
obs_time_ordinary = []
freq_drift_allen_micro = []
freq_drift_allen_ordinary = []
freq_drift_fp_micro = []
freq_drift_2fp_micro = []
freq_drift_fp_ordinary = []
freq_drift_2fp_ordinary = []


file_final = Parent_directory  + '/solar_burst/Nancay/af_sgepss_analysis_data/electrondensity_anaysis_shuron3.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")


for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    sunspots_num = csv_input_final["sunspots_num"][j]
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    burst_type = csv_input_final["burst_type"][j]
    if burst_type == 'ordinary':
        if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
            if sunspots_num >= 36:
                if ((freq_start >= 40) & (freq_end <= 40)):
                    allen_fp_dfdt_list = [float(k) for k in csv_input_final["allen_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    allen_fp_residual_list = [float(k) for k in csv_input_final["allen_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_allen = allen_fp_residual_list.index(min(allen_fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["allen_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_allen]

                    factor = idx_allen + 1
                    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
                    drift_rates_allen = numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    newkirk_fp_dfdt_list = [float(k) for k in csv_input_final["newkirk_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    newkirk_fp_residual_list = [float(k) for k in csv_input_final["newkirk_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_fp = newkirk_fp_residual_list.index(min(newkirk_fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["newkirk_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_fp]

                    factor = idx_fp + 1
                    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * 4.2 * 10 ** (4+4.32/r_1)
                    drift_rates_fp = numerical_diff_df_dn(ne) * numerical_diff_newkirk_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    newkirk_2fp_dfdt_list = [float(k) for k in csv_input_final["newkirk_2fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    newkirk_2fp_residual_list = [float(k) for k in csv_input_final["newkirk_2fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_2fp = newkirk_2fp_residual_list.index(min(newkirk_2fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["newkirk_2fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_2fp]

                    factor = idx_2fp + 1
                    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
                    r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
                    ne = factor * 4.2 * 10 ** (4+4.32/r_1)
                    drift_rates_2fp = 2 * numerical_diff_df_dn(ne) * numerical_diff_newkirk_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    obs_time_ordinary.append(obs_time)                    
                    freq_drift_allen_ordinary.append(drift_rates_allen*-1)
                    freq_drift_fp_ordinary.append(drift_rates_fp*-1)
                    freq_drift_2fp_ordinary.append(drift_rates_2fp*-1)

        if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23))) | ((obs_time >= datetime.datetime(1995,1,1)) & (obs_time <= datetime.datetime(1997,12,31,23)))):
            if sunspots_num <= 36:
                if ((freq_start >= 40) & (freq_end <= 40)):
                    allen_fp_dfdt_list = [float(k) for k in csv_input_final["allen_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    allen_fp_residual_list = [float(k) for k in csv_input_final["allen_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_allen = allen_fp_residual_list.index(min(allen_fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["allen_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_allen]

                    factor = idx_allen + 1
                    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
                    drift_rates_allen = numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    newkirk_fp_dfdt_list = [float(k) for k in csv_input_final["newkirk_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    newkirk_fp_residual_list = [float(k) for k in csv_input_final["newkirk_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_fp = newkirk_fp_residual_list.index(min(newkirk_fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["newkirk_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_fp]

                    factor = idx_fp + 1
                    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * 4.2 * 10 ** (4+4.32/r_1)
                    drift_rates_fp = numerical_diff_df_dn(ne) * numerical_diff_newkirk_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    newkirk_2fp_dfdt_list = [float(k) for k in csv_input_final["newkirk_2fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    newkirk_2fp_residual_list = [float(k) for k in csv_input_final["newkirk_2fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_2fp = newkirk_2fp_residual_list.index(min(newkirk_2fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["newkirk_2fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_2fp]

                    factor = idx_2fp + 1
                    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
                    r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
                    ne = factor * 4.2 * 10 ** (4+4.32/r_1)
                    drift_rates_2fp = 2 * numerical_diff_df_dn(ne) * numerical_diff_newkirk_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    obs_time_ordinary.append(obs_time)                    
                    freq_drift_allen_ordinary.append(drift_rates_allen*-1)
                    freq_drift_fp_ordinary.append(drift_rates_fp*-1)
                    freq_drift_2fp_ordinary.append(drift_rates_2fp*-1)

    else:
        if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
            if sunspots_num >= 36:
                if ((freq_start >= 40) & (freq_end <= 40)):
                    allen_fp_dfdt_list = [float(k) for k in csv_input_final["allen_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    allen_fp_residual_list = [float(k) for k in csv_input_final["allen_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_allen = allen_fp_residual_list.index(min(allen_fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["allen_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_allen]

                    factor = idx_allen + 1
                    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
                    drift_rates_allen = numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    wangmax_fp_dfdt_list = [float(k) for k in csv_input_final["wangmax_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    wangmax_fp_residual_list = [float(k) for k in csv_input_final["wangmax_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    # idx_fp = wangmax_fp_residual_list.index(min(wangmax_fp_residual_list))
                    idx_fp = 0
                    time_rate = [float(k) for k in csv_input_final["wangmax_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_fp]

                    factor = idx_fp + 1
                    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(-4.42158e+06/(1+(h/696000)) + 5.41656e+07/(1+(h/696000))**2 - 1.86150e+08 /(1+(h/696000))**3 + 2.13102e+08/(1+(h/696000))**4))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * (-4.42158e+06/(r_1) + 5.41656e+07/(r_1)**2 - 1.86150e+08 /(r_1)**3 + 2.13102e+08/(r_1)**4)
                    drift_rates_fp = numerical_diff_df_dn(ne) * numerical_diff_wangmax_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    wangmax_2fp_dfdt_list = [float(k) for k in csv_input_final["wangmax_2fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    wangmax_2fp_residual_list = [float(k) for k in csv_input_final["wangmax_2fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    # idx_2fp = wangmax_2fp_residual_list.index(min(wangmax_2fp_residual_list))
                    idx_2fp = 0
                    time_rate = [float(k) for k in csv_input_final["wangmax_2fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_2fp]

                    factor = idx_2fp + 1
                    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(-4.42158e+06/(1+(h/696000)) + 5.41656e+07/(1+(h/696000))**2 - 1.86150e+08 /(1+(h/696000))**3 + 2.13102e+08/(1+(h/696000))**4))
                    r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
                    ne = factor * (-4.42158e+06/(r_1) + 5.41656e+07/(r_1)**2 - 1.86150e+08 /(r_1)**3 + 2.13102e+08/(r_1)**4)
                    drift_rates_2fp = 2 * numerical_diff_df_dn(ne) * numerical_diff_wangmax_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    obs_time_micro.append(obs_time)
                    freq_drift_allen_micro.append(drift_rates_allen*-1)
                    freq_drift_fp_micro.append(drift_rates_fp*-1)
                    freq_drift_2fp_micro.append(drift_rates_2fp*-1)
                    
        if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23)))):
            if sunspots_num <= 36:
                if ((freq_start >= 40) & (freq_end <= 40)):
                    allen_fp_dfdt_list = [float(k) for k in csv_input_final["allen_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    allen_fp_residual_list = [float(k) for k in csv_input_final["allen_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_allen = allen_fp_residual_list.index(min(allen_fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["allen_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_allen]

                    factor = idx_allen + 1
                    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
                    drift_rates_allen = numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    wangmin_fp_dfdt_list = [float(k) for k in csv_input_final["wangmin_fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    wangmin_fp_residual_list = [float(k) for k in csv_input_final["wangmin_fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    # idx_fp = wangmin_fp_residual_list.index(min(wangmin_fp_residual_list))
                    idx_fp = 0
                    time_rate = [float(k) for k in csv_input_final["wangmin_fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_fp]

                    factor = idx_fp + 1
                    cube_4 =  (lambda h: 9 * 1e-3 * np.sqrt(353766/(1+(h/696000)) + 1.03359e+07/(1+(h/696000))**2 - 5.46541e+07/(1+(h/696000))**3 + 8.24791e+07/(1+(h/696000))**4))
                    r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
                    ne = factor * (353766/(r_1) + 1.03359e+07/(r_1)**2 - 5.46541e+07/(r_1)**3 + 8.24791e+07/(r_1)**4)
                    drift_rates_fp = numerical_diff_df_dn(ne) * numerical_diff_wangmin_dn_dr(factor, r) * time_rate * light_v * 1e+5


                    wangmin_2fp_dfdt_list = [float(k) for k in csv_input_final["wangmin_2fp_dfdt_list"][j].split('[')[1].split(']')[0].split(',')]
                    wangmin_2fp_residual_list = [float(k) for k in csv_input_final["wangmin_2fp_residual"][j].split('[')[1].split(']')[0].split(',')]
                    idx_2fp = 0
                    # idx_2fp = wangmin_2fp_residual_list.index(min(wangmin_2fp_residual_list))
                    time_rate = [float(k) for k in csv_input_final["wangmin_2fp_velocity"][j].split('[')[1].split(']')[0].split(',')][idx_2fp]

                    factor = idx_2fp + 1
                    cube_4 =(lambda h: 9 * 1e-3 * np.sqrt(353766/(1+(h/696000)) + 1.03359e+07/(1+(h/696000))**2 - 5.46541e+07/(1+(h/696000))**3 + 8.24791e+07/(1+(h/696000))**4))
                    r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
                    r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
                    ne = factor * (353766/(r_1) + 1.03359e+07/(r_1)**2 - 5.46541e+07/(r_1)**3 + 8.24791e+07/(r_1)**4)
                    drift_rates_2fp = 2 * numerical_diff_df_dn(ne) * numerical_diff_wangmin_dn_dr(factor, r) * time_rate * light_v * 1e+5



                    obs_time_micro.append(obs_time)
                    freq_drift_allen_micro.append(drift_rates_allen*-1)
                    freq_drift_fp_micro.append(drift_rates_fp*-1)
                    freq_drift_2fp_micro.append(drift_rates_2fp*-1)

freq_drift_fp_micro = np.array(freq_drift_fp_micro)
freq_drift_2fp_micro = np.array(freq_drift_2fp_micro)
freq_drift_allen_micro = np.array(freq_drift_allen_micro)
obs_time_micro = np.array(obs_time_micro)

freq_drift_fp_ordinary = np.array(freq_drift_fp_ordinary)
freq_drift_2fp_ordinary = np.array(freq_drift_2fp_ordinary)
freq_drift_allen_ordinary = np.array(freq_drift_allen_ordinary)
obs_time_ordinary = np.array(obs_time_ordinary)


micro_solar_max_idx = np.where((obs_time_micro >= datetime.datetime(2012,1,1)) & (obs_time_micro <= datetime.datetime(2014,12,31,23)))[0]
micro_solar_min_idx = np.where(((obs_time_micro >= datetime.datetime(2007,1,1)) & (obs_time_micro <= datetime.datetime(2009,12,31,23))) | ((obs_time_micro >= datetime.datetime(2017,1,1)) & (obs_time_micro <= datetime.datetime(2020,12,31,23))))[0]

ordinary_solar_max_idx = np.where((obs_time_ordinary >= datetime.datetime(2012,1,1)) & (obs_time_ordinary <= datetime.datetime(2014,12,31,23)))[0]
ordinary_solar_min_idx = np.where(((obs_time_ordinary >= datetime.datetime(2007,1,1)) & (obs_time_ordinary <= datetime.datetime(2009,12,31,23))) | ((obs_time_ordinary >= datetime.datetime(2017,1,1)) & (obs_time_ordinary <= datetime.datetime(2020,12,31,23))) | ((obs_time_ordinary >= datetime.datetime(1995,1,1)) & (obs_time_ordinary <= datetime.datetime(1997,12,31,23))))[0]

freq_check = 40


for i in range(3):
    if i == 0:
        freq_drift_ordinary = freq_drift_allen_ordinary
        freq_drift_micro = freq_drift_allen_micro
    elif i == 1:
        freq_drift_ordinary = freq_drift_fp_ordinary
        freq_drift_micro = freq_drift_fp_micro
    elif i == 2:
        freq_drift_ordinary = freq_drift_2fp_ordinary
        freq_drift_micro = freq_drift_2fp_micro
        
    driftrates_ci_median = []
    driftrates_ci_se = []

    
    
    
    for data in [freq_drift_ordinary[ordinary_solar_max_idx], freq_drift_ordinary[ordinary_solar_min_idx], freq_drift_micro[micro_solar_max_idx], freq_drift_micro[micro_solar_min_idx]]:
        if (len(data) >= 50 & len(data) <= 90):
            coef = 2
        elif (len(data) > 90 & len(data) <= 150):
            coef = 1.98
        elif (len(data) > 150):
            coef = 1.96
        else:
            print ('Event num error')
            sys.exit()
        driftrates_ci_median.append(np.mean(data))
        driftrates_ci_se.append(np.std(data)/np.sqrt(len(data))*coef)

    
    SD_list = [np.nanstd(freq_drift_ordinary[ordinary_solar_max_idx]), np.nanstd(freq_drift_ordinary[ordinary_solar_min_idx]), np.nanstd(freq_drift_micro[micro_solar_max_idx]), np.nanstd(freq_drift_micro[micro_solar_min_idx])]
    driftrates_ci_plot(driftrates_ci_median, driftrates_ci_se)
    driftrates_sd_plot(driftrates_ci_median, SD_list)
# velocity_ci_plot(velocity_ci_median, velocity_ci_se)


