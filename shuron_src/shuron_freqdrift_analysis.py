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
import glob

def numerical_diff_newkirk_velocity_fp(factor, r):
    h = 1e-2
    
    ne_1 = np.log(factor * 4.2 * 10 ** (4+4.32/((r+h)/69600000000)))
    ne_2 = np.log(factor * 4.2 * 10 ** (4+4.32/((r-h)/69600000000)))
    return ((ne_1 - ne_2)/(2*h))

def csv_input_analysis_micro_solarmax():
    obs_time_micro.append(obs_time)
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    freq_drift_micro.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')]
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')]
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]
    freq = 40
    fp_cube = (lambda h: 9 * 1e-3 * np.sqrt((-4.42158e+06*((1+(h/69600000000))**(-1))+5.41656e+07*((1+(h/69600000000))**(-2))- 1.86150e+08*((1+(h/69600000000))**(-3))+ 2.13102e+08*((1+(h/69600000000))**(-4)))))
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_fp_list_micro.append((e_radio-s_radio)/0.02/29979245800)
    # print (2/numerical_diff_allen_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
    
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_2fp_list_micro.append((e_radio-s_radio)/0.02/29979245800)
    return

def csv_input_analysis_micro_solarmin():
    obs_time_micro.append(obs_time)
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    freq_drift_micro.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')]
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')]
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]
    
    freq = 40
    fp_cube = (lambda h: 9 * 1e-3 * np.sqrt((353766*((1+(h/69600000000))**(-1))+1.03359e+07*((1+(h/69600000000))**(-2))- 5.46541e+07*((1+(h/69600000000))**(-3))+ 8.24791e+07*((1+(h/69600000000))**(-4)))))
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_fp_list_micro.append((e_radio-s_radio)/0.02/29979245800)
    
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_2fp_list_micro.append((e_radio-s_radio)/0.02/29979245800)
    return

def csv_input_analysis_ordinary_solarmax():
    obs_time_ordinary.append(obs_time)
    obs_time_ordinary_maximum.append(obs_time)
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    freq_drift_ordinary.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')]
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')]
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]
    factor_velocity = 1
    freq = 40
    fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
    h_radio = inversefunc(fp_cube, y_values = freq) + 69600000000
    velocity_fp_list_factor1_ordinary.append(2/numerical_diff_newkirk_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
    
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_2fp_list_factor1_ordinary.append((e_radio-s_radio)/0.02/29979245800)

    factor_velocity = 3
    freq = 40
    fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
    h_radio = inversefunc(fp_cube, y_values = freq) + 69600000000
    velocity_fp_list_factor3_ordinary.append(2/numerical_diff_newkirk_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
    
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_2fp_list_factor3_ordinary.append((e_radio-s_radio)/0.02/29979245800)
    return

def csv_input_analysis_ordinary_solarmin():
    obs_time_ordinary.append(obs_time)
    time_start = csv_input_final["event_start"][j]
    time_end = csv_input_final["event_end"][j]
    drift_rates = csv_input_final["drift_rate_40MHz"][j]
    freq_drift_ordinary.append(drift_rates)
    # duration_list.append(csv_input_final["event_end"][j]-csv_input_final["event_start"][j]+1)
    # freq_drift_final.append(csv_input_final["drift_rate_40MHz"][j])
    factor_list = csv_input_final["factor"][j]
    peak_time_list = [int(k) for k in csv_input_final["peak_time_list"][j].split('[')[1].split(']')[0].split(',')]
    peak_freq_list = [float(k) for k in csv_input_final["peak_freq_list"][j].split('[')[1].split(']')[0].split(',')]
    resi_idx = np.argmin([float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')])
    resi_list= [float(k) for k in csv_input_final["residual"][j].split('[')[1].split(']')[0].split(',')]
    velocity_list=[float(k) for k in csv_input_final["velocity"][j].split('[')[1].split(']')[0].split(',')]
    
    factor_velocity = 1
    freq = 40
    fp_cube = (lambda h: 9 * 1e-3 * np.sqrt(factor_velocity * 4.2 * 10 ** (4+4.32/(1+(h/69600000000)))))
    h_radio = inversefunc(fp_cube, y_values = freq) + 69600000000
    velocity_fp_list_factor1_ordinary.append(2/numerical_diff_newkirk_velocity_fp(factor_velocity, h_radio)/freq*csv_input_final["drift_rate_40MHz"][j]/29979245800 * -1)
    
    s_freq = (freq + csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    e_freq = (freq - csv_input_final["drift_rate_40MHz"][j]*0.01)/2
    s_radio = inversefunc(fp_cube, y_values = s_freq) + 69600000000
    e_radio = inversefunc(fp_cube, y_values = e_freq) + 69600000000
    velocity_2fp_list_factor1_ordinary.append((e_radio-s_radio)/0.02/29979245800)
    return

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
    
    plt.plot([5.3, 5.3],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed", label = 'P. J. Zhang et al., 2018')
    plt.plot([7.3, 7.3],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed")
    
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
    
    # plt.plot([6.3, 6.3],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed", label = 'P. J. Zhang et al., 2018')
    
    plt.xlabel('Frequency drift rates [MHz/s]', fontsize=18)
    plt.ylabel('Type of bursts', fontsize=18)
    plt.xlim(3, 11)
    plt.ylim(0.5, 12)
    plt.yticks([8.5,5.5,3.5,1.5], ['Ordinary type III burst\nAround the solar maximum', 'Ordinary type III burst\nAround the solar minimum', 'Micro type III burst\nAround the solar maximum', 'Micro type III burst\nAround the solar minimum'])
    plt.xticks(fontsize=16)
    # plt.legend()
    plt.title('Average ± 1σ')
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    plt.show()


def velocity_ci_plot(velocity_ci_median, velocity_ci_se):
    x = [velocity_ci_median[0], velocity_ci_median[2], velocity_ci_median[4],velocity_ci_median[6],velocity_ci_median[8]] # 変数を初期化
    y = [10,8,6,4,2]
    x_err = [velocity_ci_se[0], velocity_ci_se[2], velocity_ci_se[4],velocity_ci_se[6],velocity_ci_se[8]]
    
    
    
    # x_1 = [0.33, 0.28, 0.54, 0.33, 0.18, 0.22] # 変数を初期化
    x_1 = [velocity_ci_median[1],velocity_ci_median[3],velocity_ci_median[5],velocity_ci_median[7],velocity_ci_median[9]]
    y_1 = [9,7,5,3,1]
    x_err_1 = [velocity_ci_se[1],velocity_ci_se[3],velocity_ci_se[5],velocity_ci_se[7],velocity_ci_se[9]]
    
    plt.figure(figsize=(6,7))
    
    plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=5, ecolor='r', markeredgecolor = "r", color='r')
    plt.errorbar(x_1, y_1, xerr = x_err_1, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
    
    # plt.scatter(0.13,4, color = "k", marker="*", s = 300) 
    
    # plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
    # plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
    
    plt.plot([0, 1],[10, 10], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[8, 8], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[6, 6], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)
    
    plt.plot([0, 1],[9, 9], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[7, 7], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[5, 5], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[4, 4], color = "r", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[3, 3], color = "r", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[2, 2], color = "b", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[1, 1], color = "b", linewidth = 35.0, alpha = 0.1)
    
    
    plt.plot([0, 1],[9.5, 9.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[8.5, 8.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[7.5, 7.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[5.5, 5.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[3.5, 3.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[1.5, 1.5], color = "k", linewidth = 1.0, alpha = 1)
    # plt.plot([0, 1],[0.5, 0.5], color = "k", linewidth = 1.0, alpha = 1)
    
    plt.xlabel('Radial velocity [c]', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.title('95% CI')
    plt.xlim(0.0, 1)
    plt.ylim(0.5, 12)
    plt.xticks(fontsize=16)
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()


def velocity_sd_plot():
    x = [velocity_ci_median[0], velocity_ci_median[2], velocity_ci_median[4],velocity_ci_median[6],velocity_ci_median[8]] # 変数を初期化
    y = [10,8,6,4,2]
    x_err = [np.std(velocity_fp_list_factor1_ordinary[ordinary_solar_max_idx]), np.std(velocity_fp_list_factor3_ordinary), np.std(velocity_fp_list_factor1_ordinary[ordinary_solar_min_idx]),np.std(velocity_fp_list_micro[micro_solar_max_idx]),np.std(velocity_fp_list_micro[micro_solar_min_idx])]



    # x_1 = [0.33, 0.28, 0.54, 0.33, 0.18, 0.22] # 変数を初期化
    x_1 = [velocity_ci_median[1],velocity_ci_median[3],velocity_ci_median[5],velocity_ci_median[7],velocity_ci_median[9]]
    y_1 = [9,7,5,3,1]
    x_err_1 = [np.std(velocity_2fp_list_factor1_ordinary[ordinary_solar_max_idx]),np.std(velocity_2fp_list_factor3_ordinary),np.std(velocity_2fp_list_factor1_ordinary[ordinary_solar_min_idx]),np.std(velocity_2fp_list_micro[micro_solar_max_idx]),np.std(velocity_2fp_list_micro[micro_solar_min_idx])]
    
    plt.figure(figsize=(6,7))
    
    plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=5, ecolor='r', markeredgecolor = "r", color='r')
    plt.errorbar(x_1, y_1, xerr = x_err_1, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')
    
    # plt.scatter(0.13,4, color = "k", marker="*", s = 300) 
    
    # plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
    # plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
    
    plt.plot([0, 1],[10, 10], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[8, 8], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[6, 6], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)
    
    plt.plot([0, 1],[9, 9], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[7, 7], color = "orange", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[5, 5], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[4, 4], color = "r", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[3, 3], color = "r", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[2, 2], color = "b", linewidth = 35.0, alpha = 0.1)
    plt.plot([0, 1],[1, 1], color = "b", linewidth = 35.0, alpha = 0.1)
    
    
    plt.plot([0, 1],[9.5, 9.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[8.5, 8.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[7.5, 7.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[5.5, 5.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[3.5, 3.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)
    plt.plot([0, 1],[1.5, 1.5], color = "k", linewidth = 1.0, alpha = 1)
    # plt.plot([0, 1],[0.5, 0.5], color = "k", linewidth = 1.0, alpha = 1)
    
    plt.xlabel('Radial velocity [c]', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.title('Average ± 1σ')
    plt.xlim(0.0, 1)
    plt.ylim(0.5, 12)
    plt.xticks(fontsize=16)
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.show()







def box_plot():
    fig, ax = plt.subplots()
    
    results = (freq_drift_micro[micro_solar_min_idx], freq_drift_micro[micro_solar_max_idx], freq_drift_ordinary[ordinary_solar_min_idx], freq_drift_ordinary[ordinary_solar_max_idx])
    bp = ax.boxplot(results)
    
    ax.set_xticklabels(['Micro type III burst\nAround the solar minimum', 'Micro type III burst\nAround the solar maximum'
                        , 'Ordinary type III burst\nAround the solar minimum', 'Ordinary type III burst\nAround the solar maximum'])
    
    plt.title('Box plot')
    plt.xlabel('Burst types')
    plt.ylabel('Frequency drift rates[MHz/s]')
    # Y軸のメモリのrange
    plt.ylim([0,25])
    plt.grid()
    plt.xticks(rotation=20)
    # 描画
    plt.show()
    return

def printresults():
    print ('\nOrdinary type III burst ~極大期~')
    print ('イベント数:' + str(len(freq_drift_ordinary[ordinary_solar_max_idx])))
    print ('DR 平均値:' + str(round(np.nanmean(freq_drift_ordinary[ordinary_solar_max_idx]),2)))
    print ('DR 標準偏差(SD):' + str(round(np.nanstd(freq_drift_ordinary[ordinary_solar_max_idx]),2)))
    print ('DR 標準偏差(SE):' + str(round(driftrates_ci_se[0],2)))
    print ('RV(fp) F = 1 平均値:' + str(round(velocity_ci_median[0],2)))
    print ('RV(fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_fp_list_factor1_ordinary[ordinary_solar_max_idx]),3)))
    print ('RV(fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[0],2)))
    print ('RV(2fp) F = 1 平均値:' + str(round(velocity_ci_median[1],2)))
    print ('RV(2fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_2fp_list_factor1_ordinary[ordinary_solar_max_idx]),3)))
    print ('RV(2fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[1],2)))
    print ('RV(fp) F = 3 平均値:' + str(round(velocity_ci_median[2],2)))
    print ('RV(fp) F = 3 標準偏差(SD):' + str(round(np.std(velocity_fp_list_factor3_ordinary),3)))
    print ('RV(fp) F = 3 標準偏差(SE):' + str(round(velocity_ci_se[2],2)))
    print ('RV(2fp) F = 3 平均値:' + str(round(velocity_ci_median[3],2)))
    print ('RV(2fp) F = 3 標準偏差(SD):' + str(round(np.std(velocity_2fp_list_factor3_ordinary),3)))
    print ('RV(2fp) F = 3 標準偏差(SE):' + str(round(velocity_ci_se[3],2)))
    print ('最小値:' + str(round(np.nanmin(freq_drift_ordinary[ordinary_solar_max_idx]),1)))
    print ('最大値:' + str(round(np.nanmax(freq_drift_ordinary[ordinary_solar_max_idx]),1)))    

    print ('\nOrdinary type III burst ~極小期~')
    print ('イベント数:' + str(len(freq_drift_ordinary[ordinary_solar_min_idx])))
    print ('平均値:' + str(round(np.nanmean(freq_drift_ordinary[ordinary_solar_min_idx]),2)))
    print ('DR 標準偏差(SD):' + str(round(np.nanstd(freq_drift_ordinary[ordinary_solar_min_idx]),2)))
    print ('DR 標準偏差(SE):' + str(round(driftrates_ci_se[1],2)))
    print ('RV(fp) F = 1 平均値:' + str(round(velocity_ci_median[4],2)))
    print ('RV(fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_fp_list_factor1_ordinary[ordinary_solar_min_idx]),3)))
    print ('RV(fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[4],2)))
    print ('RV(2fp) F = 1 平均値:' + str(round(velocity_ci_median[5],2)))
    print ('RV(2fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_2fp_list_factor1_ordinary[ordinary_solar_min_idx]),3)))
    print ('RV(2fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[5],2)))
    print ('最小値:' + str(round(np.nanmin(freq_drift_ordinary[ordinary_solar_min_idx]),1)))
    print ('最大値:' + str(round(np.nanmax(freq_drift_ordinary[ordinary_solar_min_idx]),1)))
    
    print ('\nMicro type III burst ~極大期~')
    print ('イベント数:' + str(len(freq_drift_micro[micro_solar_max_idx])))
    print ('平均値:' + str(round(np.nanmean(freq_drift_micro[micro_solar_max_idx]),2)))
    print ('DR 標準偏差(SD):' + str(round(np.nanstd(freq_drift_micro[micro_solar_max_idx]),2)))
    print ('DR 標準偏差(SE):' + str(round(driftrates_ci_se[2],2)))
    print ('RV(fp) F = 1 平均値:' + str(round(velocity_ci_median[6],2)))
    print ('RV(fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_fp_list_micro[micro_solar_max_idx]),3)))
    print ('RV(fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[6],2)))
    print ('RV(2fp) F = 1 平均値:' + str(round(velocity_ci_median[7],2)))
    print ('RV(2fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_2fp_list_micro[micro_solar_max_idx]),3)))
    print ('RV(2fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[7],2)))
    print ('最小値:' + str(round(np.nanmin(freq_drift_micro[micro_solar_max_idx]),1)))
    print ('最大値:' + str(round(np.nanmax(freq_drift_micro[micro_solar_max_idx]),1)))
    
    print ('\nMicro type III burst ~極小期~')
    print ('イベント数:' + str(len(freq_drift_micro[micro_solar_min_idx])))
    print ('平均値:' + str(round(np.nanmean(freq_drift_micro[micro_solar_min_idx]),2)))
    print ('DR 標準偏差(SD):' + str(round(np.nanstd(freq_drift_micro[micro_solar_min_idx]),2)))
    print ('DR 標準偏差(SE):' + str(round(driftrates_ci_se[3],2)))
    print ('RV(fp) F = 1 平均値:' + str(round(velocity_ci_median[8],2)))
    print ('RV(fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_fp_list_micro[micro_solar_min_idx]),3)))
    print ('RV(fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[8],2)))
    print ('RV(2fp) F = 1 平均値:' + str(round(velocity_ci_median[9],2)))
    print ('RV(2fp) F = 1 標準偏差(SD):' + str(round(np.std(velocity_2fp_list_micro[micro_solar_min_idx]),3)))
    print ('RV(2fp) F = 1 標準偏差(SE):' + str(round(velocity_ci_se[9],2)))
    print ('最小値:' + str(round(np.nanmin(freq_drift_micro[micro_solar_min_idx]),1)))
    print ('最大値:' + str(round(np.nanmax(freq_drift_micro[micro_solar_min_idx]),1)))
    return







freq_drift_micro = []
obs_time_micro = []
freq_drift_ordinary = []
obs_time_ordinary = []
obs_time_ordinary_maximum = []
velocity_fp_list_factor1_ordinary = []
velocity_2fp_list_factor1_ordinary = []
velocity_fp_list_factor3_ordinary = []
velocity_2fp_list_factor3_ordinary = []
velocity_fp_list_micro = []
velocity_2fp_list_micro = []

Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'


file_final = Parent_directory  + '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_micro_LL_RR.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")



for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    sunspots_num = csv_input_final["sunspots_num"][j]
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
        if sunspots_num >= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                csv_input_analysis_micro_solarmax()
    if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23)))):
        if sunspots_num <= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                csv_input_analysis_micro_solarmin()
file_final = Parent_directory  + '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_ordinary_withnonclear.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")
for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    sunspots_num = csv_input_final["sunspots_num"][j]
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
        if sunspots_num >= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                csv_input_analysis_ordinary_solarmax()
    if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23))) | ((obs_time >= datetime.datetime(1995,1,1)) & (obs_time <= datetime.datetime(1997,12,31,23)))):
        if sunspots_num <= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                csv_input_analysis_ordinary_solarmin()


freq_drift_micro = np.array(freq_drift_micro)
obs_time_micro = np.array(obs_time_micro)
freq_drift_ordinary = np.array(freq_drift_ordinary)
obs_time_ordinary = np.array(obs_time_ordinary)

micro_solar_max_idx = np.where((obs_time_micro >= datetime.datetime(2012,1,1)) & (obs_time_micro <= datetime.datetime(2014,12,31,23)))[0]
micro_solar_min_idx = np.where(((obs_time_micro >= datetime.datetime(2007,1,1)) & (obs_time_micro <= datetime.datetime(2009,12,31,23))) | ((obs_time_micro >= datetime.datetime(2017,1,1)) & (obs_time_micro <= datetime.datetime(2020,12,31,23))))[0]
obs_time_ordinary_maximum = np.array(obs_time_ordinary_maximum)
velocity_fp_list_factor1_ordinary = np.array(velocity_fp_list_factor1_ordinary)
velocity_2fp_list_factor1_ordinary = np.array(velocity_2fp_list_factor1_ordinary)
velocity_fp_list_factor3_ordinary = np.array(velocity_fp_list_factor3_ordinary)
velocity_2fp_list_factor3_ordinary = np.array(velocity_2fp_list_factor3_ordinary)
velocity_fp_list_micro = np.array(velocity_fp_list_micro)
velocity_2fp_list_micro = np.array(velocity_2fp_list_micro)




ordinary_solar_max_idx = np.where((obs_time_ordinary >= datetime.datetime(2012,1,1)) & (obs_time_ordinary <= datetime.datetime(2014,12,31,23)))[0]
ordinary_solar_min_idx = np.where(((obs_time_ordinary >= datetime.datetime(2007,1,1)) & (obs_time_ordinary <= datetime.datetime(2009,12,31,23))) | ((obs_time_ordinary >= datetime.datetime(2017,1,1)) & (obs_time_ordinary <= datetime.datetime(2020,12,31,23))) | ((obs_time_ordinary >= datetime.datetime(1995,1,1)) & (obs_time_ordinary <= datetime.datetime(1997,12,31,23))))[0]

freq_check = 40

#statistical_study
bursts = ['Ordinary type Ⅲ burst', 'Micro type Ⅲ burst']
for burst in bursts:
    if burst == 'Micro type Ⅲ burst':
    
        text = ''
        freq_drift_solar_maximum = freq_drift_micro[micro_solar_max_idx]
        color_1 = "r"
        freq_drift_solar_minimum = freq_drift_micro[micro_solar_min_idx]
        color_2 = "b"
        
        
        # if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
        #     max_val = max(freq_drift_solar_minimum)
        # else:
        #     max_val = max(freq_drift_solar_maximum)
        
        # if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
        #     min_val = min(freq_drift_solar_minimum)
        # else:
        #     min_val = min(freq_drift_solar_maximum)

        max_val = 26
        min_val = 1.1
        bin_size = 20
        
        x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
        y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
        x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
        y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
        plt.close()
        width = x_hist[1]-x_hist[0]
        for i in range(len(y_hist)):
            if i == 0:
                plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
                # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
                plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
                # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
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
        plt.ylim(0,0.35)
        plt.show()
        plt.close()
        # print (burst + ' ~極小期~')
        # print ('イベント数:' + str(len(freq_drift_solar_minimum)))
        # print ('平均値:' + str(round(np.nanmean(freq_drift_solar_minimum),1)))
        # print ('標準偏差:' + str(round(np.nanstd(freq_drift_solar_minimum),2)))
        # print ('最小値:' + str(round(np.nanmin(freq_drift_solar_minimum),1)))
        # print ('最大値:' + str(round(np.nanmax(freq_drift_solar_minimum),1)))
        
        # print (burst + ' ~極大期~')
        # print ('イベント数:' + str(len(freq_drift_solar_maximum)))
        # print ('平均値:' + str(round(np.nanmean(freq_drift_solar_maximum),1)))
        # print ('標準偏差:' + str(round(np.nanstd(freq_drift_solar_maximum),2)))
        # print ('最小値:' + str(round(np.nanmin(freq_drift_solar_maximum),1)))
        # print ('最大値:' + str(round(np.nanmax(freq_drift_solar_maximum),1)))
    
    if burst == 'Ordinary type Ⅲ burst':
    

        freq_drift_solar_maximum = freq_drift_ordinary[ordinary_solar_max_idx]
        color_1 = "orange"
        freq_drift_solar_minimum = freq_drift_ordinary[ordinary_solar_min_idx]
        color_2 = "deepskyblue"
        
        # if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
        #     max_val = max(freq_drift_solar_minimum)
        # else:
        #     max_val = max(freq_drift_solar_maximum)
        
        # if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
        #     min_val = min(freq_drift_solar_minimum)
        # else:
        #     min_val = min(freq_drift_solar_maximum)
        max_val = 26
        min_val = 1.1
        bin_size = 20
        
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
        plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst)
    # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
        plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
        plt.ylabel('Occurrence rate',fontsize=15)
            # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
        plt.xticks(rotation = 20)
        plt.ylim(0,0.35)
        plt.show()
        plt.close()

period_list = ['Around the solar maximum', 'Around the solar minimum']
for period in period_list:
    if period == 'Around the solar maximum':
    
        text = ''
        freq_drift_solar_maximum_ordinary = freq_drift_ordinary[ordinary_solar_max_idx]
        color_1 = "orange"
        freq_drift_solar_maximum_micro = freq_drift_micro[micro_solar_max_idx]
        color_2 = "r"
        
        
        # if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
        #     max_val = max(freq_drift_solar_minimum)
        # else:
        #     max_val = max(freq_drift_solar_maximum)
        
        # if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
        #     min_val = min(freq_drift_solar_minimum)
        # else:
        #     min_val = min(freq_drift_solar_maximum)

        max_val = 26
        min_val = 1.1
        bin_size = 20
        
        x_hist = (plt.hist(freq_drift_solar_maximum_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
        y_hist = (plt.hist(freq_drift_solar_maximum_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum_ordinary))
        x_hist_1 = (plt.hist(freq_drift_solar_maximum_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
        y_hist_1 = (plt.hist(freq_drift_solar_maximum_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum_micro))
        plt.close()
        width = x_hist[1]-x_hist[0]
        for i in range(len(y_hist)):
            if i == 0:
                plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Ordinary type Ⅲ burst\n'+ str(len(freq_drift_solar_maximum_ordinary)) + ' events')
                # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
                plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Micro type Ⅲ burst\n'+ str(len(freq_drift_solar_maximum_micro)) + ' events')
                # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
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
        plt.ylim(0,0.35)
        plt.show()
        plt.close()

    if period == 'Around the solar minimum':
    
        text = ''
        freq_drift_solar_minimum_ordinary = freq_drift_ordinary[ordinary_solar_min_idx]
        color_1 = "deepskyblue"
        freq_drift_solar_minimum_micro = freq_drift_micro[micro_solar_min_idx]
        color_2 = "b"
        
        
        # if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
        #     max_val = max(freq_drift_solar_minimum)
        # else:
        #     max_val = max(freq_drift_solar_maximum)
        
        # if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
        #     min_val = min(freq_drift_solar_minimum)
        # else:
        #     min_val = min(freq_drift_solar_maximum)

        max_val = 26
        min_val = 1.1
        bin_size = 20
        
        x_hist = (plt.hist(freq_drift_solar_minimum_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
        y_hist = (plt.hist(freq_drift_solar_minimum_ordinary, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum_ordinary))
        x_hist_1 = (plt.hist(freq_drift_solar_minimum_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
        y_hist_1 = (plt.hist(freq_drift_solar_minimum_micro, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum_micro))
        plt.close()
        width = x_hist[1]-x_hist[0]
        for i in range(len(y_hist)):
            if i == 0:
                plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Ordinary type Ⅲ burst\n'+ str(len(freq_drift_solar_minimum_ordinary)) + ' events')
                # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
                plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Micro type Ⅲ burst\n'+ str(len(freq_drift_solar_minimum_micro)) + ' events')

                # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
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
        plt.ylim(0,0.35)
        plt.show()
        plt.close()





box_plot()

driftrates_ci_median = []
driftrates_ci_se = []

velocity_ci_median = []
velocity_ci_se = []

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

for data in [velocity_fp_list_factor1_ordinary[ordinary_solar_max_idx], velocity_2fp_list_factor1_ordinary[ordinary_solar_max_idx], velocity_fp_list_factor3_ordinary, velocity_2fp_list_factor3_ordinary,
             velocity_fp_list_factor1_ordinary[ordinary_solar_min_idx], velocity_2fp_list_factor1_ordinary[ordinary_solar_min_idx], velocity_fp_list_micro[micro_solar_max_idx], velocity_2fp_list_micro[micro_solar_max_idx], velocity_fp_list_micro[micro_solar_min_idx], velocity_2fp_list_micro[micro_solar_min_idx]]:
    if (len(data) >= 50 & len(data) <= 90):
        coef = 2
    elif (len(data) > 90 & len(data) <= 150):
        coef = 1.98
    elif (len(data) > 150):
        coef = 1.96
    else:
        print ('Event num error')
        sys.exit()
    velocity_ci_median.append(np.mean(data))
    velocity_ci_se.append(np.std(data)/np.sqrt(len(data))*coef)

SD_list = [np.nanstd(freq_drift_ordinary[ordinary_solar_max_idx]), np.nanstd(freq_drift_ordinary[ordinary_solar_min_idx]), np.nanstd(freq_drift_micro[micro_solar_max_idx]), np.nanstd(freq_drift_micro[micro_solar_min_idx])]
driftrates_ci_plot(driftrates_ci_median, driftrates_ci_se)
driftrates_sd_plot(driftrates_ci_median, SD_list)
velocity_ci_plot(velocity_ci_median, velocity_ci_se)
velocity_sd_plot()

printresults()

# file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'

# sunspot_obs_times = []
# sunspot_obs_num_list = []
# print (file_gain)
# csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# # print(csv_input['Time_list'])
# for i in range(len(csv_input)):
#     BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
#     # if (BG_obs_time_event >= datetime.datetime(2008, 12, 1)) & (BG_obs_time_event <= datetime.datetime(2019, 12, 31)):
#     if (BG_obs_time_event >= datetime.datetime(1995, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2021, 1, 1)):
#         sunspot_num = csv_input['sunspot_number'][i]
#         if not sunspot_num == -1:
#             sunspot_obs_times.append(BG_obs_time_event)
#             # Frequency_list = csv_input['Frequency'][i]
#             sunspot_obs_num_list.append(sunspot_num)
#         else:
#             print (BG_obs_time_event)

# sunspot_obs_times = np.array(sunspot_obs_times)
# sunspot_obs_num_list = np.array(sunspot_obs_num_list)
#ordinaryplot
# xlims_list = [[datetime.datetime(1995,1,1), datetime.datetime(1997,12,31,23)], [datetime.datetime(2007,1,1), datetime.datetime(2009,12,31,23)], [datetime.datetime(2012,1,1), datetime.datetime(2014,12,31,23)], [datetime.datetime(2017,1,1), datetime.datetime(2020,12,31,23)]]
# for xlims in xlims_list:
#     ordinary_solar_idx = np.where(((obs_time_ordinary >= xlims[0]) & (obs_time_ordinary <= xlims[1])))[0]
#     sunspots_selected_idx = np.where(((sunspot_obs_times >= xlims[0]) & (sunspot_obs_times <= xlims[1])))[0]
    
#     deleted_time_list = []
#     if ((xlims[0] == datetime.datetime(1995,1,1)) | (xlims[0] == datetime.datetime(2007,1,1)) | (xlims[0] == datetime.datetime(2017,1,1))):
#         for i in range(len(sunspot_obs_times[sunspots_selected_idx])):
#             if sunspot_obs_num_list[sunspots_selected_idx][i] > 36:
#                 deleted_time_list.append(sunspot_obs_times[sunspots_selected_idx][i])
#     elif xlims[0] == datetime.datetime(2012,1,1):
#         for i in range(len(sunspot_obs_times[sunspots_selected_idx])):
#             if sunspot_obs_num_list[sunspots_selected_idx][i] < 36:
#                 deleted_time_list.append(sunspot_obs_times[sunspots_selected_idx][i])
#     plt.figure(figsize=(15,6))
#     for deleted_time in deleted_time_list:
#         plt.plot([deleted_time, deleted_time+datetime.timedelta(days=1)-datetime.timedelta(seconds=1)],[1, 17], color = "k", alpha = 0.1)
#     target_driftrates = freq_drift_ordinary[ordinary_solar_idx]
#     target_obstime = obs_time_ordinary[ordinary_solar_idx]
#     plt.plot(target_obstime, target_driftrates, '.')
#     plt.xlim(xlims)
#     plt.ylim(1,17)
#     plt.title('Ordinary type III burst')
#     plt.xlabel('Time', fontsize = 15)
#     plt.ylabel('Frequency drift rates[MHz/s]', fontsize = 15)
#     plt.show()
xlims_list = [[datetime.datetime(1995,1,1), datetime.datetime(1997,12,31,23)], [datetime.datetime(2007,1,1), datetime.datetime(2009,12,31,23)], [datetime.datetime(2012,1,1), datetime.datetime(2014,12,31,23)], [datetime.datetime(2017,1,1), datetime.datetime(2020,12,31,23)]]
for xlims in xlims_list:
    ordinary_solar_idx = np.where(((obs_time_ordinary >= xlims[0]) & (obs_time_ordinary <= xlims[1])))[0]

    plt.figure(figsize=(8,6))
    target_driftrates = freq_drift_ordinary[ordinary_solar_idx]
    target_obstime = obs_time_ordinary[ordinary_solar_idx]
    plt.plot(target_obstime, target_driftrates, '.')
    plt.xlim(xlims)
    plt.ylim(1,26)
    plt.title('Ordinary type III burst')
    plt.xlabel('Time', fontsize = 15)
    plt.ylabel('Frequency drift rates[MHz/s]', fontsize = 15)
    plt.show()

xlims_list = [[datetime.datetime(2007,1,1), datetime.datetime(2009,12,31,23)], [datetime.datetime(2012,1,1), datetime.datetime(2014,12,31,23)], [datetime.datetime(2017,1,1), datetime.datetime(2020,12,31,23)]]
for xlims in xlims_list:
    micro_solar_idx = np.where(((obs_time_micro >= xlims[0]) & (obs_time_micro <= xlims[1])))[0]
    
    target_driftrates = freq_drift_micro[micro_solar_idx]
    target_obstime = obs_time_micro[micro_solar_idx]
    plt.figure(figsize=(8,6))
    plt.plot(target_obstime, target_driftrates, '.')
    plt.xlim(xlims)
    plt.ylim(1,26)
    plt.title('Micro type III burst')
    plt.xlabel('Time', fontsize = 15)
    plt.ylabel('Frequency drift rates[MHz/s]', fontsize = 15)
    plt.show()








# #statistical_study2異常値を除外
# bursts = ['Ordinary type Ⅲ burst', 'Micro type Ⅲ burst']
# for burst in bursts:
#     if burst == 'Micro type Ⅲ burst':
    
#         text = ''
#         freq_drift_solar_maximum = freq_drift_micro[micro_solar_max_idx]
#         color_1 = "r"
#         freq_drift_solar_minimum = freq_drift_micro[micro_solar_min_idx]
#         color_2 = "b"
#         selected_micro_solar_max_idx = np.where((freq_drift_solar_maximum < np.mean(freq_drift_solar_maximum) + 2 *np.std(freq_drift_solar_maximum)) & (freq_drift_solar_maximum > np.mean(freq_drift_solar_maximum) - 2 *np.std(freq_drift_solar_maximum)))[0]
#         freq_drift_solar_maximum = freq_drift_solar_maximum[selected_micro_solar_max_idx]

#         selected_micro_solar_min_idx = np.where((freq_drift_solar_minimum < np.mean(freq_drift_solar_minimum) + 2 *np.std(freq_drift_solar_minimum)) & (freq_drift_solar_minimum > np.mean(freq_drift_solar_minimum) - 2 *np.std(freq_drift_solar_minimum)))[0]
#         freq_drift_solar_minimum = freq_drift_solar_minimum[selected_micro_solar_min_idx]
#         max_val = 26
#         min_val = 1.1
#         bin_size = 20
        
#         x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
#         y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
#         x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
#         y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
#         plt.close()
#         width = x_hist[1]-x_hist[0]
#         for i in range(len(y_hist)):
#             if i == 0:
#                 plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
#                 # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
#                 plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
#                 # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
#             else:
#                 plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3)
#                 plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
#         plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst + text)
#     # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
#         plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
#         plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
#         plt.ylabel('Occurrence rate',fontsize=15)
#         # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
#         plt.xticks(rotation = 20)
#         plt.ylim(0,0.35)
#         plt.show()
#         plt.close()
#         print (burst + ' ~極小期~')
#         print ('イベント数:' + str(len(freq_drift_solar_minimum)))
#         print ('平均値:' + str(round(np.nanmean(freq_drift_solar_minimum),1)))
#         print ('標準偏差:' + str(round(np.nanstd(freq_drift_solar_minimum),2)))
#         print ('最小値:' + str(round(np.nanmin(freq_drift_solar_minimum),1)))
#         print ('最大値:' + str(round(np.nanmax(freq_drift_solar_minimum),1)))
        
#         print (burst + ' ~極大期~')
#         print ('イベント数:' + str(len(freq_drift_solar_maximum)))
#         print ('平均値:' + str(round(np.nanmean(freq_drift_solar_maximum),1)))
#         print ('標準偏差:' + str(round(np.nanstd(freq_drift_solar_maximum),2)))
#         print ('最小値:' + str(round(np.nanmin(freq_drift_solar_maximum),1)))
#         print ('最大値:' + str(round(np.nanmax(freq_drift_solar_maximum),1)))
#     if burst == 'Ordinary type Ⅲ burst':

        
    
#         freq_drift_solar_maximum = freq_drift_ordinary[ordinary_solar_max_idx]
#         color_1 = "orange"
#         freq_drift_solar_minimum = freq_drift_ordinary[ordinary_solar_min_idx]
#         color_2 = "deepskyblue"

#         selected_ordinary_solar_max_idx = np.where((freq_drift_solar_maximum < np.mean(freq_drift_solar_maximum) + 2 *np.std(freq_drift_solar_maximum)) & (freq_drift_solar_maximum > np.mean(freq_drift_solar_maximum) - 2 *np.std(freq_drift_solar_maximum)))[0]
#         freq_drift_solar_maximum = freq_drift_solar_maximum[selected_ordinary_solar_max_idx]

#         selected_ordinary_solar_min_idx = np.where((freq_drift_solar_minimum < np.mean(freq_drift_solar_minimum) + 2 *np.std(freq_drift_solar_minimum)) & (freq_drift_solar_minimum > np.mean(freq_drift_solar_minimum) - 2 *np.std(freq_drift_solar_minimum)))[0]
#         freq_drift_solar_minimum = freq_drift_solar_minimum[selected_ordinary_solar_min_idx]

#         max_val = 26
#         min_val = 1.1
#         bin_size = 20
        
#         x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
#         y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_maximum))
#         x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
#         y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0]/len(freq_drift_solar_minimum))
        
    
#         plt.close()
#         width = x_hist[1]-x_hist[0]
#         for i in range(len(y_hist)):
#             if i == 0:
#                 plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label =  'Around the solar maximum\n'+ str(len(freq_drift_solar_maximum)) + ' events')
#                 plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_solar_minimum)) + ' events')
#             else:
#                 plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3)
#                 plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3)
#         plt.title('Frequency drift rates @ ' + str(freq_check) + '[MH/z]' + '\n' + burst)
#     # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
#         plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 10)
#         plt.xlabel('Frequency drift rates[MHz/s]',fontsize=15)
#         plt.ylabel('Occurrence rate',fontsize=15)
#             # plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
#         plt.xticks(rotation = 20)
#         plt.ylim(0,0.35)
#         plt.show()
#         plt.close()

#         print (burst + ' ~極小期~')
#         print ('イベント数:' + str(len(freq_drift_solar_minimum)))
#         print ('平均値:' + str(round(np.nanmean(freq_drift_solar_minimum),1)))
#         print ('標準偏差:' + str(round(np.nanstd(freq_drift_solar_minimum),2)))
#         print ('最小値:' + str(round(np.nanmin(freq_drift_solar_minimum),1)))
#         print ('最大値:' + str(round(np.nanmax(freq_drift_solar_minimum),1)))
        
#         print (burst + ' ~極大期~')
#         print ('イベント数:' + str(len(freq_drift_solar_maximum)))
#         print ('平均値:' + str(round(np.nanmean(freq_drift_solar_maximum),1)))
#         print ('標準偏差:' + str(round(np.nanstd(freq_drift_solar_maximum),2)))
#         print ('最小値:' + str(round(np.nanmin(freq_drift_solar_maximum),1)))
#         print ('最大値:' + str(round(np.nanmax(freq_drift_solar_maximum),1)))



