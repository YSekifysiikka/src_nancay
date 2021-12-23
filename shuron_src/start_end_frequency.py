#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 19:07:06 2021

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
    
    plt.plot([6.3, 6.3],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed")
    
    plt.xlabel('Frequency drift rates [MHz/s]', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.xlim(4, 9)
    plt.ylim(0.5, 12)
    plt.xticks(fontsize=16)
    plt.title('95% CI')
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
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
    
    plt.scatter(0.13,4, color = "k", marker="*", s = 300) 
    
    plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
    plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
    
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
    plt.xlim(0.0, 0.65)
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
    print ('DR 平均値:' + str(round(np.nanmean(freq_drift_ordinary[ordinary_solar_max_idx]),1)))
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
    print ('平均値:' + str(round(np.nanmean(freq_drift_ordinary[ordinary_solar_min_idx]),1)))
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
    print ('平均値:' + str(round(np.nanmean(freq_drift_micro[micro_solar_max_idx]),1)))
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
    print ('平均値:' + str(round(np.nanmean(freq_drift_micro[micro_solar_min_idx]),1)))
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

freq_start_list = []
freq_end_list = []

velocity_fp_list_factor1_ordinary = []
velocity_2fp_list_factor1_ordinary = []
velocity_fp_list_factor3_ordinary = []
velocity_2fp_list_factor3_ordinary = []
velocity_fp_list_micro = []
velocity_2fp_list_micro = []

Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'


file_final = Parent_directory  + '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_micro_LL_RR.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")


count = 0
count_1 = 0
count_2 = 0
count_3 = 0
for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    sunspots_num = csv_input_final["sunspots_num"][j]
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    # if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
    #     if sunspots_num >= 36:
    #         if ((freq_start >= 40) & (freq_end <= 40)):
    #             csv_input_analysis_micro_solarmax()
    # if (((obs_time >= datetime.datetime(2007,1,1)) & (obs_time <= datetime.datetime(2009,12,31,23))) | ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23)))):
    #     if sunspots_num <= 36:
    #         if ((freq_start >= 40) & (freq_end <= 40)):
    #             csv_input_analysis_micro_solarmin()
    if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
        if sunspots_num >= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                freq_start_list.append(freq_start)
                freq_end_list.append(freq_end)
                count += 1
            else:
                print (freq_start, freq_end)
            # csv_input_analysis_ordinary_solarmax()
    if ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23))):
        if sunspots_num <= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                freq_start_list.append(freq_start)
                freq_end_list.append(freq_end)
                count_1 += 1
            else:
                print (freq_start, freq_end)
            # csv_input_analysis_ordinary_solarmin()

file_final = Parent_directory  + '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_ordinary_LL_RR.csv'
csv_input_final = pd.read_csv(filepath_or_buffer= file_final, sep=",")
for j in range(len(csv_input_final)):
    obs_time = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    sunspots_num = csv_input_final["sunspots_num"][j]
    freq_start = csv_input_final["freq_start"][j]
    freq_end = csv_input_final["freq_end"][j]
    if ((obs_time >= datetime.datetime(2012,1,1)) & (obs_time <= datetime.datetime(2014,12,31,23))):
        if sunspots_num >= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                freq_start_list.append(freq_start)
                freq_end_list.append(freq_end)
                count_2 += 1
            else:
                print (freq_start, freq_end)
            # freq_start_list.append(freq_start)
            # freq_end_list.append(freq_end)
            # if ((freq_start >= 40) & (freq_end <= 40)):
            # csv_input_analysis_ordinary_solarmax()
    if ((obs_time >= datetime.datetime(2017,1,1)) & (obs_time <= datetime.datetime(2020,12,31,23))):
        if sunspots_num <= 36:
            if ((freq_start >= 40) & (freq_end <= 40)):
                freq_start_list.append(freq_start)
                freq_end_list.append(freq_end)
                count_3 += 1
            else:
                print (freq_start, freq_end)
            # freq_start_list.append(freq_start)
            # freq_end_list.append(freq_end)
            # if ((freq_start >= 40) & (freq_end <= 40)):
            # csv_input_analysis_ordinary_solarmin()




text = ''
freq_drift_solar_maximum = freq_start_list
color_1 = "r"
freq_drift_solar_minimum =freq_end_list
color_2 = "b"


# if max(freq_drift_solar_minimum) >= max(freq_drift_solar_maximum):
#     max_val = max(freq_drift_solar_minimum)
# else:
#     max_val = max(freq_drift_solar_maximum)

# if min(freq_drift_solar_minimum) <= min(freq_drift_solar_maximum):
#     min_val = min(freq_drift_solar_minimum)
# else:
#     min_val = min(freq_drift_solar_maximum)

max_val = 80
min_val = 30
bin_size = 15

x_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
y_hist = (plt.hist(freq_drift_solar_maximum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0])
x_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[1])
y_hist_1 = (plt.hist(freq_drift_solar_minimum, bins = bin_size, range = (min_val-1,max_val+1), density= None)[0])
plt.close()
width = x_hist[1]-x_hist[0]
for i in range(len(y_hist)):
    if i == 0:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_2, alpha = 0.3, label =  '$f_{start}$  '+ str(len(freq_drift_solar_maximum)) + ' events')
        # plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_1, alpha = 0.3, label = 'Around the solar maximum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_1, alpha = 0.3, label = '$f_{end}$  '+ str(len(freq_drift_solar_minimum)) + ' events')
        # plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_2, alpha = 0.3, label = 'Around the solar minimum\n'+ str(len(freq_drift_micro[micro_idx2])) + ' events')
    else:
        plt.bar(x_hist[i] + (x_hist[1]-x_hist[0])/2, y_hist[i], width= width, color = color_2, alpha = 0.3)
        plt.bar(x_hist_1[i] + (x_hist_1[1]-x_hist_1[0])/2, y_hist_1[i], width= width, color = color_1, alpha = 0.3)
# plt.title('' + str(freq_check) + '[MH/z]' + '\n' + burst + text)
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1, fontsize = 15)
plt.xlabel('Frequency (MHz)',fontsize=15)
plt.ylabel('Number of events',fontsize=15)
# plt.xticks([0,5,10,15], ['0 - 1','5 - 6','10 - 11', '15 - 16']) 
plt.xticks(rotation = 20)
# plt.ylim(0,0.35)
plt.show()
plt.close()