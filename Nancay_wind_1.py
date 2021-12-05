#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:34:19 2021

@author: yuichiro
"""


import glob
import numpy as np
import cdflib
import math
import datetime as dt
import matplotlib.dates as mdates
import datetime
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.io as sio
from scipy.optimize import curve_fit
import sys
import matplotlib.gridspec as gridspec


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
        return rad
    
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


def radio_plot(data_1, receiver_1, data_2, receiver_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs):
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
    gs = gridspec.GridSpec(101, 1)
    ax = plt.subplot(gs[0:NDA_gs, :])
    ax1 = plt.subplot(gs[NDA_gs+1:NDA_gs+rad_2_gs+1, :])
    ax2 = plt.subplot(gs[NDA_gs+rad_2_gs+1:NDA_gs+rad_2_gs+rad_1_gs+1, :])
    # fig, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(12.0, 12.0))
    plt.subplots_adjust(hspace=0.001)

    y_lims = [Frequency_end, Frequency_start]
    ax.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=10,vmax=40)
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

    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancaywind_2/'+year + '/' + month):
    #     os.makedirs(Parent_directory + '/solar_burst/Nancaywind_2/'+year + '/' + month)
    # filename = Parent_directory + '/solar_burst/Nancaywind_2/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    # plt.savefig(filename)
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



date_in=[20170711,20170711]
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
        print (file_name)
# start_date, end_date = final_txt_make(Parent_directory, Parent_lab, int(year), 101,1231)
# gen = file_generator(file_path)
# for file in gen:
#     file_name = file[:-1]
        LL, RR, diff_move_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data_LL_RR(Parent_directory, file_name, move_ave, Freq_start, Freq_end)
    
        for t in range (math.floor(((diff_move_db.shape[1]-time_co)/time_band) + 1)):
            LL_db_sep, RR_db_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t_1 = separated_data_LL_RR(LL, RR, diff_move_db, epoch, time_co, time_band, t)
            # quick_look(date_OBs, x_lims, LL_db_sep, RR_db_sep, diff_db_plot_sep, Frequency_start, Frequency_end, Time_start, Time_end)
            if int(Time_start.split(':')[2]) >= 30:
                wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1])) + datetime.timedelta(minutes=1)
            else:
                wind_Time = dt.datetime(int(year), int(date_OBs[4:6]), int(date_OBs[6:8]), int(Time_start.split(':')[0]), int(Time_start.split(':')[1]))
            waves_setting_1 = {'receiver'   : 'rad1',
                             'date_in'    : int(date_OBs[:8]),#yyyymmdd
                             'HH'         : str(wind_Time.hour).zfill(2),#hour
                             'MM'         : str(wind_Time.minute).zfill(2),#minute
                             'SS'         : '00',#second
                             'duration'   :  29,#min
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
                             'duration'   :  29,#min
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
                data_1 = rw_1.read_rad(receiver_1)
                data_2 = rw_2.read_rad(receiver_2)
                radio_plot(data_1, receiver_1, data_2, receiver_2, diff_db_plot_sep, x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs[:8])
            except:
                print ('No data: wind' + date_OBs[:8])
    except:
        print('Plot error: ',date)
    DATE+=pd.to_timedelta(1,unit='day')



