#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:16:56 2021

@author: yuichiro
"""


import pandas as pd
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import glob
import shutil
import os
import cdflib
import math
import datetime as dt
import scipy.io as sio
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec
from scipy import signal
import itertools
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]




Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_all_freq_final.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

obs_time = []
decibel_list = []
for i in range(len(antenna1_csv)):
    obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    # decibel_list.append(antenna1_csv['decibel'][i])
    l = antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')
    decibel_list.append([float(s) for s in l if s != ''])
    # for j in range(len(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' '))):
    #     if not antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j] == '':
    #         decibel_list.append(float(antenna1_csv['decibel'][i].replace('\n', '')[1:-1].split(' ')[j]))

obs_time = np.array(obs_time)
decibel_list = np.array(decibel_list)

def check_BG_1(event_date, event_check_days, file_name):
    event_date = str(event_date)
    yyyy = event_date[:4]
    MM = event_date[4:6]
    dd = event_date[6:8]
    select_date = datetime.datetime(int(yyyy),int(MM),int(dd))
    check_decibel = []
    check_obs_time = []
    if event_check_days == 1:
        start_date = select_date
        for i in range(1):
            obs_indexes = np.where((obs_time >= select_date) & (obs_time <= select_date + datetime.timedelta(days=1)))[0]
            for obs_index in obs_indexes:
            # if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])
    else:
        pass
        # start_date = select_date - datetime.timedelta(days=event_check_days/2)
        # for i in range(event_check_days+1):
        #     check_date = start_date + datetime.timedelta(days=i)
        #     obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
        #     if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
        #         check_decibel.append(decibel_list[obs_index])
        #         check_obs_time.append(obs_time[obs_index])
    
    check_decibel = np.array(check_decibel)
    check_obs_time = np.array(check_obs_time)

    return np.median(check_decibel, axis=0)

def check_BG(event_date, event_hour, event_minite, event_check_days, file_name):
    event_date = str(event_date)
    yyyy = event_date[:4]
    MM = event_date[4:6]
    dd = event_date[6:8]
    hh = event_hour
    mm = event_minite
    select_date = datetime.datetime(int(yyyy),int(MM),int(dd),int(hh),int(mm))
    check_decibel = []
    check_obs_time = []
    if event_check_days == 1:
        start_date = select_date
        for i in range(1):
            check_date = start_date
            obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
            if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])
    else:
        start_date = select_date - datetime.timedelta(days=event_check_days/2)
        for i in range(event_check_days+1):
            check_date = start_date + datetime.timedelta(days=i)
            obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
            if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
                check_decibel.append(decibel_list[obs_index])
                check_obs_time.append(obs_time[obs_index])

    check_decibel = np.array(check_decibel)
    check_obs_time = np.array(check_obs_time)

    return np.median(check_decibel, axis=0)


class read_waves:
    
    def __init__(self, **kwargs):
        self.date_in  = kwargs['date_in']
        self.HH = kwargs['HH']
        self.MM = kwargs['MM']
        self.SS = kwargs['SS']
        self.duration = kwargs['duration']
        self.directry = kwargs['directry']
        
        self.str_date = str(self.date_in)
        self.time_axis = pd.date_range(start=self.str_date, periods=1440, freq='T')
        self.yyyy = self.str_date[0:4]
        self.mm   = self.str_date[4:6]
        self.dd   = self.str_date[6:8]
        
        start = pd.to_datetime(self.str_date + self.HH + self.MM + self.SS,
                               format='%Y%m%d%H%M%S')
        
        end = start + pd.to_timedelta(self.duration, unit='min')

        self.time_range = [start, end]
        
        return
    
    def tlimit(self, df):
        if type(df) != pd.core.frame.DataFrame:
            print('tlimit \n Type error: data type must be DataFrame')
            sys.exit()
        
        

        tl_df =    df[   df.index >= self.time_range[0]]
        tl_df = tl_df[tl_df.index  <= self.time_range[1]]
        
        return tl_df
    
    
    def read_rad(self, receiver):
        if type(receiver) != str:
            print('read_rad \n Keyword error: reciever must be a string')
            sys.exit()
        
        if receiver == 'rad1':
            extension = '.R1'
        elif receiver == 'rad2':
            extension = '.R2'
        else:
            print('read_rad \n Name error: receiver name')
            sys.exit()
        file_path = self.directry + self.str_date + extension
        sav_data = sio.readsav(file_path)
        data = sav_data['arrayb'][:, 0:1440]
        BG   = sav_data['arrayb'][:, 1440]
        
        rad_data = np.where(data==0, np.nan, data)
        rad_data = rad_data.T
        rad = pd.DataFrame(rad_data)
        
        rad.index = self.time_axis
        rad = self.tlimit(rad)
        return rad, BG
    
    def read_waves(self):
        rad1 = self.read_rad('rad1')
        rad2 = self.read_rad('rad2')
        waves = pd.concat([rad1, rad2], axis=1)
        
        return waves



def waves_peak_finder(data):
    if type(data) != pd.core.frame.DataFrame:
        print('waves_peak_finder \n Type error: data type must be DataFrame')
        sys.exit()
    data = data.reset_index(drop=True)
    peak = data.max(axis=0)
    idx  = data.idxmax(axis=0)
    result = pd.concat([idx, peak], axis=1)
    result.columns = ['index', 'peak']
    return result

def freq_setting(receiver):
    if receiver == 'rad1':
        freq = 0.02 + 0.004*np.arange(256)
    elif receiver == 'rad2':
        freq = 1.075 + 0.05*np.arange(256)
    elif receiver == 'waves':
        freq1 = 0.02 + 0.004*np.arange(256)
        freq2 = 1.075 + 0.05*np.arange(256)
        freq  = np.hstack([freq1, freq2])
    else:
        print('freq_setting \n Name error: receiver name')
    return freq

def linear_fit(data, receiver='rad1', freq_band=[0.02, 1.04],
               p0=[0,0], bounds = ([-np.inf, -np.inf], [np.inf, np.inf])):
    
    def linear_func(x, a, b):
        return a*x + b
    
    peak_data = waves_peak_finder(data)
    index = peak_data['index']
    peak  = peak_data['peak']
    
    freq = freq_setting(receiver)
    freq = pd.DataFrame(freq)
    
    cat_data = pd.concat([peak_data,freq], axis=1)
    cat_data.columns = ['index', 'peak', 'freq']
    
    flimit_df =  cat_data[ cat_data['freq'] >= freq_band[0]]
    flimit_df = flimit_df[flimit_df['freq'] <= freq_band[1]]
    
    l_index = flimit_df['index'].values
    l_peak  = flimit_df['peak'].values
    l_freq = flimit_df['freq'].values
    
    if len(l_index) == 0:
        print('linear_fit \n Value error: freq_band range are illegal values for fitting')
        sys.exit()
    
    x = []
    y = []
    
    for i in range(len(l_index)):
        if np.isnan(l_peak[i]) != True:
            x.append(l_index[i])
            y.append(l_freq[i])
    
    popt, pcov = curve_fit(linear_func,x, y, p0=p0, bounds=bounds)
    error = np.sqrt(np.diag(pcov))
    
    plt.figure()
    plt.plot(cat_data['index'], cat_data['freq'], 'ro')
    plt.plot(index, linear_func(index, popt[0], popt[1]))
    plt.axhline(freq_band[0], xmin=0, xmax=1, color='blue', linestyle='dashed')
    plt.axhline(freq_band[1], xmin=0, xmax=1, color='blue', linestyle='dashed')
    plt.ylim(freq.iloc[0][0], freq.iloc[-1][0])
    plt.xlabel('Time [min]')
    plt.ylabel('Frequency [MHz]')
    return popt, error
def radio_plot(data_1, receiver_1, BG_1, data_2, receiver_2, BG_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs):
    # freq_list = [Frequency_start, Frequency_end]
    NDA_freq = Frequency_start/Frequency_end
    p_data_2 = 20*np.log10(data_2)
    vmin_2 = 0
    vmax_2 = 2.5
    
    freq_axis = freq_setting(receiver_2)
    y_lim_2 = [freq_axis[0], freq_axis[-1]]
    rad_2_freq = freq_axis[-1]/freq_axis[0]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_2.index
    x_lim_2 = [time_axis[0], time_axis[-1]]
    x_lim_2 = mdates.date2num(x_lim_2)

    p_data_1 = 20*np.log10(data_1)
    vmin_1 = 0
    vmax_1 = 8
    
    freq_axis = freq_setting(receiver_1)
    y_lim_1 = [freq_axis[0], freq_axis[-1]]
    rad_1_freq = freq_axis[-1]/freq_axis[1]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_1.index
    x_lim_1 = [time_axis[0], time_axis[-1]]
    x_lim_1 = mdates.date2num(x_lim_1)


    plt.close()
    year=date_OBs[0:4]
    month=date_OBs[4:6]
    day=date_OBs[6:8]
    if type(data_1) != pd.core.frame.DataFrame:
        print('radio_plot \n Type error: data type must be DataFrame')
        sys.exit()

    NDA_gs = int(round((NDA_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_2_gs = int(round((rad_2_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_1_gs = int(round((rad_1_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    fig = plt.figure(figsize=(12.0, 12.0))
    gs = gridspec.GridSpec(131, 1)
    ax = plt.subplot(gs[0:NDA_gs, :])
    ax1 = plt.subplot(gs[NDA_gs+1:NDA_gs+rad_2_gs+1, :])
    ax2 = plt.subplot(gs[NDA_gs+rad_2_gs+1:NDA_gs+rad_2_gs+rad_1_gs+1, :])
    ax3 = plt.subplot(gs[108:118, :])
    ax4 = plt.subplot(gs[118:128, :])
    # fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12.0, 12.0))
    plt.subplots_adjust(hspace=0.001)

    y_lims = [Frequency_end, Frequency_start]
    ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=5,vmax=30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(labelsize=10)
    # ax.set_xlabel('Time [UT]')
    # ax.set_ylabel('Frequency [MHz]')
    ax.set_yscale('log')

    

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')
    

    ax2.imshow(p_data_1.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_1[0],x_lim_1[1],y_lim_1[0],y_lim_1[1]],
                vmin=vmin_1, vmax=vmax_1)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(labelsize=10)
    ax2.set_xlabel('Time [UT]')
    # ax2.set_ylabel('Frequency [MHz]')
    ax2.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False) 
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False) 
    ax2.xaxis.tick_bottom()
    ax2.set_yscale('log')

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')

    ax3.plot(np.arange(0,len(p_data_2.T.values.tolist()[247]),1),p_data_2.T.values.tolist()[247], '-', label = '13.425MHz')
    maxid = signal.argrelmax(np.array(p_data_2.T.values.tolist()[247]), order=width)
    ax3.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], np.array(p_data_2.T.values.tolist()[247])[maxid[0]])
    ax3.set_xlim(0,len(p_data_2.T.values.tolist()[247])-1)
    ax3.legend()
    ax3.spines['bottom'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop=False) 
    ax4.plot(np.arange(0,len(p_data_1.T.values.tolist()[224]),1),p_data_1.T.values.tolist()[224], '-', label = '916kHz')
    maxid = signal.argrelmax(np.array(p_data_1.T.values.tolist()[224]), order=width)
    ax4.scatter(np.arange(0,time_band+time_co,1)[maxid[0]], np.array(p_data_1.T.values.tolist()[224])[maxid[0]])
    ax4.set_xlim(0,len(p_data_1.T.values.tolist()[224])-1)
    ax4.legend()
    # ax4.spines['bottom'].set_visible(False)
    ax4.xaxis.tick_top()
    ax4.tick_params(labeltop=False) 


    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancaywind_4/'+year + '/' + month):
        # os.makedirs(Parent_directory + '/solar_burst/Nancaywind_4/'+year + '/' + month)
    # filename = Parent_directory + '/solar_burst/Nancaywind_4/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    # plt.savefig(filename)
    plt.show()
    return


def radio_plot_1(data_1, receiver_1, BG_1, data_2, receiver_2, BG_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs):
    # freq_list = [Frequency_start, Frequency_end]
    NDA_freq = Frequency_start/Frequency_end
    p_data_2 = 20*np.log10(data_2)
    vmin_2 = 0
    # vmax_2 = 2.5
    vmax_2 = 7

    freq_axis = freq_setting(receiver_2)
    y_lim_2 = [freq_axis[0], freq_axis[-1]]
    rad_2_freq = freq_axis[-1]/freq_axis[0]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_2.index
    x_lim_2 = [time_axis[0], time_axis[-1]]
    x_lim_2 = mdates.date2num(x_lim_2)

    p_data_1 = 20*np.log10(data_1)
    vmin_1 = 0
    # vmax_1 = 8
    vmax_1 = 20
    
    freq_axis = freq_setting(receiver_1)
    y_lim_1 = [freq_axis[0], freq_axis[-1]]
    rad_1_freq = freq_axis[-1]/freq_axis[1]
    # freq_list.append(freq_axis[0])
    # freq_list.append(freq_axis[-1])
    
    time_axis = data_1.index
    x_lim_1 = [time_axis[0], time_axis[-1]]
    x_lim_1 = mdates.date2num(x_lim_1)


    plt.close()
    year=date_OBs[0:4]
    month=date_OBs[4:6]
    day=date_OBs[6:8]
    if type(data_1) != pd.core.frame.DataFrame:
        print('radio_plot \n Type error: data type must be DataFrame')
        sys.exit()

    NDA_gs = int(round((NDA_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_2_gs = int(round((rad_2_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    rad_1_gs = int(round((rad_1_freq/(NDA_freq+rad_2_freq+rad_1_freq))*100))
    fig = plt.figure(figsize=(18.0, 11.0))
    gs = gridspec.GridSpec(151, 1)
    ax = plt.subplot(gs[0:NDA_gs, :])
    ax1 = plt.subplot(gs[NDA_gs+1:NDA_gs+rad_2_gs+1, :])
    ax2 = plt.subplot(gs[NDA_gs+rad_2_gs+1:NDA_gs+rad_2_gs+rad_1_gs+1, :])
    ax3 = plt.subplot(gs[108:128, :])
    ax4 = plt.subplot(gs[128:148, :])
    # fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12.0, 12.0))
    plt.subplots_adjust(hspace=0.001)

    y_lims = [Frequency_end, Frequency_start]
    ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=5,vmax=30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(labelsize=10)
    # ax.set_xlabel('Time [UT]')
    # ax.set_ylabel('Frequency [MHz]')
    ax.set_yscale('log')

    

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')
    

    ax2.imshow(p_data_1.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_1[0],x_lim_1[1],y_lim_1[0],y_lim_1[1]],
                vmin=vmin_1, vmax=vmax_1)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax2.tick_params(labelsize=10)
    ax2.set_xlabel('Time [UT]')
    # ax2.set_ylabel('Frequency [MHz]')
    ax2.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop=False) 
    ax1.spines['bottom'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False) 
    ax2.xaxis.tick_bottom()
    ax2.set_yscale('log')

    num=10
    b=np.ones(num)/num
    y_2=np.convolve(p_data_2.T.values.tolist()[247], b, mode='valid')
    ax3.plot(np.arange(0,len(y_2),1),y_2, '-', label = '13.425MHz')
    ax3.set_ylim(min(y_2),np.percentile([x for x in y_2 if str(x) != 'nan'], 90))
    y_1=np.convolve(p_data_1.T.values.tolist()[224], b, mode='valid')
    ax4.plot(np.arange(0,len(y_1),1),y_1, '-', label = '916kHz')
    ax4.set_ylim(min(y_1),np.percentile([x for x in y_1 if str(x) != 'nan'], 90))

    ax3.legend()
    ax3.spines['bottom'].set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop=False) 
    ax4.legend()
    # ax4.spines['bottom'].set_visible(False)
    ax4.xaxis.tick_top()
    ax4.tick_params(labeltop=False) 

    if not os.path.isdir(Parent_directory + '/solar_burst/Nancaywind_all/'+year + '/' + month):
        os.makedirs(Parent_directory + '/solar_burst/Nancaywind_all/'+year + '/' + month)
    filename = Parent_directory + '/solar_burst/Nancaywind_all/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    plt.savefig(filename)
    plt.show()
    return

def radio_plot_rad2(data, receiver='rad2'):
    plt.close()
    if type(data) != pd.core.frame.DataFrame:
        print('radio_plot \n Type error: data type must be DataFrame')
        sys.exit()
    
    p_data = 20*np.log10(data)
    vmin = 0
    vmax = 4
    
    freq_axis = freq_setting(receiver)
    y_lim = [freq_axis[0], freq_axis[-1]]
    
    time_axis = data.index
    x_lim = [time_axis[0], time_axis[-1]]
    x_lim = mdates.date2num(x_lim)
    
    fig = plt.figure(figsize=[8,6])
    
    axes = fig.add_subplot(111)
    axes.imshow(p_data.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim[0],x_lim[1],y_lim[0],y_lim[1]],
                vmin=vmin, vmax=vmax)
    axes.xaxis_date()
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes.tick_params(labelsize=10)
    axes.set_xlabel('Time [UT]')
    axes.set_ylabel('Frequency [MHz]')
    plt.show()
    return

# import os


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

def file_generator(file):
    with open(file, encoding="utf-8") as f:
        for line in f:
            yield line

def final_txt_make(Parent_directory, Parent_lab, year, start, end):
    start = format(start, '04') 
    end = format(end, '04') 
    year = str(year)
    start = int(year + start)
    end = int(year + end)
    path = Parent_directory + '/solar_burst/Nancay/data/' + year + '/*/*'+'.cdf'
    File = glob.glob(path, recursive=True)
    File = sorted(File)
    i = open(Parent_directory + '/solar_burst/Nancay/final.txt', 'w')
    for cstr in File:
        a = cstr.split('/')
        line = a[Parent_lab + 6]+'\n'
        a1 = line.split('_')
        if (int(a1[5][:8])) >= start:
          if (int(a1[5][:8])) <= end:
            i.write(line)
    i.close()
    return start, end



def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]

def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res


def read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end):
    file_name_separate =file_name.split('_')
    Date_start = file_name_separate[5]
    date_OBs=str(Date_start)
    year=date_OBs[0:4]
    month=date_OBs[4:6]

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
    Status = cdf_file['Status']
    Frequency = cdf_file['Frequency']
    #for_return_frequency_range
    Frequency_start = round(float(getNearestValue(Frequency, Freq_start)), 5)
    Frequency_end = round(float(getNearestValue(Frequency, Freq_end)), 5)
    resolution = round(float(Frequency[1] - Frequency[0]), 5)

    freq_start_idx = np.where(np.flipud(Frequency) == getNearestValue(Frequency, Freq_start))[0][0]
    freq_end_idx = np.where(np.flipud(Frequency) == getNearestValue(Frequency, Freq_end))[0][0]
    Frequency = np.flipud(Frequency)[freq_start_idx:freq_end_idx + 1]


    
    LL = cdf_file['LL'] 
    RR = cdf_file['RR'] 
    
    data_r_0 = RR
    data_l_0 = LL
    diff_r_last =(data_r_0).T
    diff_r_last = np.flipud(diff_r_last)[freq_start_idx:freq_end_idx + 1] * 0.3125
    diff_l_last =(data_l_0).T
    diff_l_last = np.flipud(diff_l_last)[freq_start_idx:freq_end_idx + 1] * 0.3125

    diff_P = (((10 ** ((diff_r_last[:,:])/10)) + (10 ** ((diff_l_last[:,:])/10)))/2)
    diff_db = np.log10(diff_P) * 10
    ####################
    y_power = []
    num = int(move_ave)
    b = np.ones(num)/num
    for i in range (diff_db.shape[0]):
        y_power.append(np.convolve(diff_P[i], b, mode='valid'))
    y_power = np.array(y_power)
    diff_move_db = np.log10(y_power) * 10
    ####################
    # min_power = np.amin(y_power, axis=1)
    # min_db_LL = np.amin(y_power, axis=1)
    # min_db = np.log10(min_power) * 10
    # min_db_LL = np.amin(diff_r_last, axis=1)
    # min_db_RR = np.amin(diff_l_last, axis=1)
    # diff_db_min_med = (diff_move_db.T - min_db).T
    # LL_min = (diff_l_last.T - min_db_LL).T
    # RR_min = (diff_r_last.T - min_db_RR).T
    return diff_l_last, diff_r_last, diff_move_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)

def separated_data_LL_RR(LL, RR, diff_move_db, epoch, time_co, time_band, t):
    if t == math.floor((diff_move_db.shape[1]-time_co)/time_band):
        t = (diff_move_db.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    time = round(time_band*t)
    print (time)
    if time < 0:
        time = 0
    #+1 is due to move_average
    start = epoch[time + 1]
    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time = round(time_band*(t+1) + time_co)
    if time > len(epoch):
        time = len(epoch) - 1
    end = epoch[time + 1]
    end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
    # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)
    start = start.timestamp()
    end = end.timestamp()
    
    # print (start, end)

    x_lims = []
    x_lims.append(dt.datetime.fromtimestamp(start))
    x_lims.append(dt.datetime.fromtimestamp(end))

    x_lims = mdates.date2num(x_lims)



    diff_db_plot_sep = diff_move_db[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    LL_sep = LL[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    RR_sep = RR[freq_start_idx:freq_end_idx + 1, time - time_band - time_co:time]
    return LL_sep, RR_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t


def separated_data_LL_RR_1(LL, RR, diff_move_db, epoch, time_co, time_band, t):
    if t == math.floor((diff_move_db.shape[1]-time_co)/time_band):
        t = (diff_move_db.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    time = round(time_band*t)
    print (time)
    if time < 0:
        time = 0
    #+1 is due to move_average
    start = epoch[time + 1]
    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time = len(epoch) - 1
    # if time > len(epoch):
    #     time = len(epoch) - 1
    end = epoch[time - 1]
    end = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
    # end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)
    start = start.timestamp()
    end = end.timestamp()
    
    # print (start, end)

    x_lims = []
    x_lims.append(dt.datetime.fromtimestamp(start))
    x_lims.append(dt.datetime.fromtimestamp(end))

    x_lims = mdates.date2num(x_lims)



    diff_db_plot_sep = diff_move_db[freq_start_idx:freq_end_idx + 1]
    LL_sep = LL[freq_start_idx:freq_end_idx + 1]
    RR_sep = RR[freq_start_idx:freq_end_idx + 1]
    return LL_sep, RR_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t
    

#dB設定に問題あり
# def quick_look(date_OBs, x_lims, LL_sep, RR_sep, diff_db_min_med, Frequency_start, Frequency_end, Time_start, Time_end):
#     year = date_OBs[0:4]
#     month = date_OBs[4:6]
#     day = date_OBs[6:8]

#     # Set some generic y-limits.
#     y_lims = [Frequency_end, Frequency_start]

#     plt.close(1)
#     fig = plt.figure(1,figsize=(16,9))


#     ax0, ax1, ax2, ax3, ax4 = fig.subplots(5, 1, sharey=True,sharex=True)

#     fig.suptitle('NANCAY DECAMETER ARRAY: '+year+
#                      '-'+month+'-'+day + '  ' + Time_start[:5] + ' - ' + Time_end[:5],fontsize=23)
#     plt.ylabel('Frequency [MHz]',fontsize=18)
#     plt.xlabel('Time (UT)',fontsize=20)

#     # gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1]) 
#     # ax0 = plt.subplot(gs[0])


#     ax_0 = ax0.imshow(diff_db_min_med, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin=0,vmax=10)
#     ax0.xaxis_date()
#     date_format = mdates.DateFormatter('%H:%M:%S')
#     ax0.xaxis.set_major_formatter(date_format)

#     cbar = fig.colorbar(ax_0, ax=ax0)
#     cbar.set_label("from \nBackground [dB]")

#     # plt.colorbar(ax0,label='from Background [dB]')
    
#     ax_1 = ax1.imshow(LL_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin=0,vmax=10)
#     cbar = fig.colorbar(ax_1, ax=ax1)
#     cbar.set_label("from \nBackground [dB]")
#     ax_2 = ax2.imshow(RR_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin=0,vmax=10)
#     cbar = fig.colorbar(ax_2, ax=ax2)
#     cbar.set_label("from \nBackground [dB]")
#     ax_3 = ax3.imshow((LL_sep - RR_sep)/(LL_sep + RR_sep), extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin=-1,vmax=1)
#     cbar = fig.colorbar(ax_3, ax=ax3)
#     cbar.set_label("Ratio [dB]")
#     ax_4 = ax4.imshow((LL_sep - RR_sep)/(LL_sep + RR_sep), extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin=-0.25,vmax=0.25)
#     cbar = fig.colorbar(ax_4, ax=ax4)
#     cbar.set_label("Ratio [dB]")

# #R-L/R+L



    
#     plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
#     fig.autofmt_xdate()
#     if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/QL/'+year + '/' + month):
#         os.makedirs(Parent_directory + '/solar_burst/Nancay/QL/'+year + '/' + month)
#     filename = Parent_directory + '/solar_burst/Nancay/QL/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
#     # plt.savefig(filename)
#     plt.show()
#     return Time_start


sigma_value = 2
after_plot = str('af_sgepss')
time_band = 1740
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
freq_check_range = 20
threshold_frequency_final = 10.5
cnn_plot_time = 50
file_path = Parent_directory + '/solar_burst/Nancay/final.txt'
save_place = 'cnn_af_sgepss'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 1.35
db_setting = 40
Freq_start = 80
Freq_end = 10
event_check_days = 1
width = 5
BG1_list = []
BG2_list = []
data1_list = []
data1_real = []
data2_list = []
data2_real = []
OBs_date = []
OBs_real_data = []

date_in=[20170715,20170720]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')

DATE=sdate
while DATE <= edate:

    date=DATE.strftime(format='%Y%m%d')
    print(date)
    year = date[:4]
    mm = date[4:6]
    try:
        file_name = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/'+year+'/'+mm+'/*'+ date +'*cdf')[0].split('/')[-1]
        # print (file_name)
# start_date, end_date = final_txt_make(Parent_directory, Parent_lab, int(year), 101,1231)
# gen = file_generator(file_path)
# for file in gen:
#     file_name = file[:-1]
        LL, RR, diff_move_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end)
    #     for t in range (math.floor(((diff_move_db.shape[1]-time_co)/time_band) + 1)):
    #         LL_db_sep, RR_db_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t_1 = separated_data_LL_RR(LL, RR, diff_move_db, epoch, time_co, time_band, t)
    #         HH = (dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=15)).strftime(format='%H')
    #         MM = ((dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=15)).strftime(format='%M'))
    #         BG_decibel = check_BG(date, HH, MM, event_check_days, file_name)
    #         diff_db_plot_sep_power = ((10 ** ((diff_db_plot_sep[:,:])/10)).T - (10 ** ((BG_decibel)/10))).T
    #         diff_db_plot_sep = np.where(diff_db_plot_sep_power <= 1, 0, np.log10(diff_db_plot_sep_power) * 10)
            

    #         # quick_look(date_OBs, x_lims, LL_db_sep, RR_db_sep, diff_db_plot_sep, Frequency_start, Frequency_end, Time_start, Time_end)
    #         if int(Time_start.split(':')[2]) >= 30:
    #             wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=1)
    #         else:
    #             wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]))
    #         waves_setting_1 = {'receiver'   : 'rad1',
    #                           'date_in'    : int(date_OBs[:8]),#yyyymmdd
    #                           'HH'         : str(wind_Time.hour).zfill(2),#hour
    #                           'MM'         : str(wind_Time.minute).zfill(2),#minute
    #                           'SS'         : '00',#second
    #                           'duration'   :  30,#min
    #                           'freq_band'  :  [0.3, 0.7],
    #                           'init_param' : [0, 0],
    #                           'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
    #                           'directry'   : '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R1/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
    #                           }
    #         waves_setting_2 = {'receiver'   : 'rad2',
    #                           'date_in'    : int(date_OBs[:8]),#yyyymmdd
    #                           'HH'         : str(wind_Time.hour).zfill(2),#hour
    #                           'MM'         : str(wind_Time.minute).zfill(2),#minute
    #                           'SS'         : '00',#second
    #                           'duration'   :  30,#min
    #                           'freq_band'  :  [0.3, 0.7],
    #                           'init_param' : [0, 0],
    #                           'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
    #                           'directry'   : '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R2/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
    #                           }
    #         try:
    #             rw_1 = read_waves(**waves_setting_1)
    #             rw_2 = read_waves(**waves_setting_2)
                
    #             receiver_1 = waves_setting_1['receiver']
    #             receiver_2 = waves_setting_2['receiver']
    #             data_1, BG_1 = rw_1.read_rad(receiver_1)
    #             data_2, BG_2 = rw_2.read_rad(receiver_2)
    #             radio_plot(data_1, receiver_1, BG_1, data_2, receiver_2, BG_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs[:8])
    #         except:
    #             print ('No data: wind' + date_OBs[:8])
    # # except:
    # #     print('Plot error: ',date)





        for t in range (1):
            LL_db_sep, RR_db_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t_1 = separated_data_LL_RR_1(LL, RR, diff_move_db, epoch, time_co, time_band, t)
            # HH = (dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=15)).strftime(format='%H')
            # MM = ((dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=15)).strftime(format='%M'))
            BG_decibel = check_BG_1(date, event_check_days, file_name)
            diff_db_plot_sep_power = ((10 ** ((diff_db_plot_sep[:,:])/10)).T - (10 ** ((BG_decibel)/10))).T
            diff_db_plot_sep = np.where(diff_db_plot_sep_power <= 1, 0, np.log10(diff_db_plot_sep_power) * 10)
            

            # quick_look(date_OBs, x_lims, LL_db_sep, RR_db_sep, diff_db_plot_sep, Frequency_start, Frequency_end, Time_start, Time_end)
            if int(Time_start.split(':')[2]) >= 30:
                wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=1)
            else:
                wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]))
            # sys.exit()
            waves_setting_1 = {'receiver'   : 'rad1',
                             'date_in'    : int(date_OBs[:8]),#yyyymmdd
                             'HH'         : str(wind_Time.hour).zfill(2),#hour
                             'MM'         : str(wind_Time.minute).zfill(2),#minute
                             'SS'         : '00',#second
                             'duration'   :  round(diff_db_plot_sep.shape[1]/60),#min
                             'freq_band'  :  [0.3, 0.7],
                             'init_param' : [0, 0],
                             'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
                             'directry'   : '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R1/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
                             }
            waves_setting_2 = {'receiver'   : 'rad2',
                             'date_in'    : int(date_OBs[:8]),#yyyymmdd
                             'HH'         : str(wind_Time.hour).zfill(2),#hour
                             'MM'         : str(wind_Time.minute).zfill(2),#minute
                             'SS'         : '00',#second
                             'duration'   :  round(diff_db_plot_sep.shape[1]/60),#min
                             'freq_band'  :  [0.3, 0.7],
                             'init_param' : [0, 0],
                             'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
                             'directry'   : '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R2/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
                             }
            try:
                rw_1 = read_waves(**waves_setting_1)
                rw_2 = read_waves(**waves_setting_2)
                
                receiver_1 = waves_setting_1['receiver']
                receiver_2 = waves_setting_2['receiver']
                data_1, BG_1 = rw_1.read_rad(receiver_1)
                data_2, BG_2 = rw_2.read_rad(receiver_2)
                # radio_plot_1(data_1, receiver_1, BG_1, data_2, receiver_2, BG_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs[:8])
                # sys.exit()
                BG1_list.append(BG_1[224])
                BG2_list.append(BG_2[247])
                num=9
                b=np.ones(num)/num
                OBs_real_data.append(data_1.index.values[int((num-1)/2):int(-(num-1)/2)])
                data1_list.append(np.convolve(data_1[224].T.values.tolist(), b, mode='valid'))
                data2_list.append(np.convolve(data_2[247].T.values.tolist(), b, mode='valid'))
                data1_real.append(np.array(np.convolve(data_1[224].T.values.tolist(), b, mode='valid')*BG_1[224]))
                data2_real.append(np.array(np.convolve(data_2[247].T.values.tolist(), b, mode='valid'))*BG_2[247])

                OBs_date.append(data_1.index.values[int(len(data_1)/2)])
            except:
                print ('No data: wind' + date_OBs[:8])
    except:
        print('Plot error: ',date)
    DATE+=pd.to_timedelta(1,unit='day')

fig = plt.figure(figsize=(35, 14))
ax1 = fig.add_subplot(3,2,1) 
ax2 = fig.add_subplot(3,2,2) 
ax3 = fig.add_subplot(3,2,3) 
ax4 = fig.add_subplot(3,2,4) 
ax5 = fig.add_subplot(3,2,5) 
ax6 = fig.add_subplot(3,2,6) 

ax1.plot(OBs_date, BG1_list, '.-')
ax1.set_title("BG: 916kHz", fontsize="20")
# ax1.set_xlim(OBs_real_data[0][0], OBs_real_data[-1][-1])
# ax1.xticks(rotation=70)


ax2.plot(OBs_date, BG2_list, '.-')
ax2.set_title('BG: 13.425MHz', fontsize="20")
ax2.set_xlim(OBs_real_data[0][0], OBs_real_data[-1][-1])
# ax1..xticks(rotation=70)

for i in range(len(OBs_real_data)):
# xrange_1 = np.arange(0, len(data1_list), 1)
    ax3.plot(OBs_real_data[i], data1_list[i], '.-', color = 'b')
    
    # 

# plt.xticks(rotation=70)

# xrange_2 = np.arange(0, len(data2_list), 1)
    ax4.plot(OBs_real_data[i], data2_list[i], '.-', color = 'b')
    
# ax4.set_ylim(np.percentile([x for x in data2_list if str(x) != 'nan'], 1),np.percentile([x for x in data2_list if str(x) != 'nan'], 95))

# plt.xticks(rotation=70)


# xrange_3 = np.arange(0, len(data1_real), 1)
    ax5.plot(OBs_real_data[i], data1_real[i], '.-', color = 'b')
    
# ax5.set_ylim(np.percentile([x for x in data1_real if str(x) != 'nan'], 1),np.percentile([x for x in data1_real if str(x) != 'nan'], 95))

# plt.xticks(rotation=70)

# xrange_4 = np.arange(0, len(data2_real), 1)
    ax6.plot(OBs_real_data[i], data2_real[i], '.-', color = 'b')
ax3.set_title('Relative value: 916kHz', fontsize="20")
ax4.set_title('Relative value: 13.425MHz', fontsize="20")
ax5.set_title('Real value: 916kHz', fontsize="20")
ax6.set_title('Real value: 13.425MHz', fontsize="20")
# ax6.set_ylim(np.percentile([x for x in data2_real if str(x) != 'nan'], 1),np.percentile([x for x in data2_real if str(x) != 'nan'], 95))

# plt.xticks(rotation=70)

ax3.set_ylim(np.percentile([x for x in list(itertools.chain.from_iterable(data1_list)) if str(x) != 'nan'], 1),np.percentile([x for x in list(itertools.chain.from_iterable(data1_list)) if str(x) != 'nan'], 95))
ax3.set_xlim(OBs_real_data[0][0], OBs_real_data[-1][-1])
ax4.set_ylim(np.percentile([x for x in list(itertools.chain.from_iterable(data2_list)) if str(x) != 'nan'], 1),np.percentile([x for x in list(itertools.chain.from_iterable(data2_list)) if str(x) != 'nan'], 95))
ax5.set_xlim(OBs_real_data[0][0], OBs_real_data[-1][-1])
ax5.set_ylim(np.percentile([x for x in list(itertools.chain.from_iterable(data1_real)) if str(x) != 'nan'], 1),np.percentile([x for x in list(itertools.chain.from_iterable(data1_real)) if str(x) != 'nan'], 95))
ax6.set_xlim(OBs_real_data[0][0], OBs_real_data[-1][-1])
ax6.set_ylim(np.percentile([x for x in list(itertools.chain.from_iterable(data2_real)) if str(x) != 'nan'], 1),np.percentile([x for x in list(itertools.chain.from_iterable(data2_real)) if str(x) != 'nan'], 95))
plt.show()
plt.close()


# date_in=[20091231,20091231]
# start_day,end_day=date_in
# sdate=pd.to_datetime(start_day,format='%Y%m%d')
# edate=pd.to_datetime(end_day,format='%Y%m%d')

# burst_type = 'fao'


# date_list = []
# if burst_type =='micro':
#     files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/storm/*/*.png')
# elif burst_type =='fao':
#     files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_associated_ordinary/*/*.png')
# # elif burst_type =='all':
#     # files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_associated_ordinary/*/*.png')
# else:
#     pass
# for file in files:
#     date_list.append(file.split('/')[-1].split('_')[0])
# date_list = list(set(date_list))



# rad1 = []
# rad2 = []
# time_range = np.arange(0,1440,1)
# for date_in in date_list:
    
# # date_in=[20090707,20090707]
#     start_day=date_in
#     sdate=pd.to_datetime(start_day,format='%Y%m%d')
# # edate=pd.to_datetime(end_day,format='%Y%m%d')

#     DATE=sdate
# # while DATE <= edate:
#     date=DATE.strftime(format='%Y%m%d')
#     # print(date)
#     year = date[:4]
#     mm = date[4:6]
#     try:
# # while DATE <= edate:
#         date=DATE.strftime(format='%Y%m%d')
#         # print (date)
#         sav_data_rad1 = sio.readsav('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R1/'+date[:4]+'/'+date[4:6]+'/'+date+'.R1')
#         rad1.append((np.where(sav_data_rad1['arrayb'][:, 1440] > 0)[0]).tolist())
#         # print (np.where(sav_data_rad1['arrayb'][:, 1440] > 0)[0])
#         sav_data_rad2 = sio.readsav('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/data/R2/'+date[:4]+'/'+date[4:6]+'/'+date+'.R2')
#         rad2.append((np.where(sav_data_rad2['arrayb'][:, 1440] > 0)[0]).tolist())
#         plt.plot(time_range, sav_data_rad2['arrayb'][247, 0:1440])
#         plt.title('rad2 13.425MHz')
#         plt.show()
#         plt.close()
#         plt.title('rad1 916kHz')
#         plt.plot(time_range, sav_data_rad1['arrayb'][224, 0:1440])
#         plt.show()
#         plt.close()
#         DATE+=pd.to_timedelta(1,unit='day')
#     except:
#         print ('Error')
#         sys.exit()
        
        
        
        
        
        
        
        
        
# import itertools
# import collections
# print (len(rad1))
# rad1_flatten = list(itertools.chain.from_iterable(rad1))
# c_rad1 = collections.Counter(rad1_flatten)
# rad2_flatten = list(itertools.chain.from_iterable(rad2))
# c_rad2 = collections.Counter(rad2_flatten)
# print ('Rad1')
# print(c_rad1.most_common())
# print ('Rad2')
# print(c_rad2.most_common())
    
    
    
# files_list_1 = []
# for file in files_1:
#     if len(file.split('/')[-1])==8:
#         files_list_1.append(file.split('/')[-1])

    
# {'20120703',
#  '20120704',
#  '20120716',
#  '20120801',
#  '20120817',
#  '20121220',
#  '20130424'}

