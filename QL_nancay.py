#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 15:39:15 2020

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
from matplotlib import gridspec
import os
# import os


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
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
    min_power = np.amin(y_power, axis=1)
    min_db_LL = np.amin(y_power, axis=1)
    min_db = np.log10(min_power) * 10
    min_db_LL = np.amin(diff_r_last, axis=1)
    min_db_RR = np.amin(diff_l_last, axis=1)
    diff_db_min_med = (diff_move_db.T - min_db).T
    LL_min = (diff_l_last.T - min_db_LL).T
    RR_min = (diff_r_last.T - min_db_RR).T
    return LL_min, RR_min, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs, diff_db

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)




def separated_data_LL_RR(LL, RR, diff_db_min_med, epoch, time_co, time_band, t,  diff_db):
    if t == math.floor((diff_db_min_med.shape[1]-time_co)/time_band):
        t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    # if t == 1:
    #     t = (5340+1980)/time_band
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



    diff_db_plot_sep = diff_db_min_med[:, time - time_band - time_co:time]
    diff_db_sep =  diff_db[:, time - time_band - time_co:time]
    LL_sep = LL[:, time - time_band - time_co:time]
    RR_sep = RR[:, time - time_band - time_co:time]
    return LL_sep, RR_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t, diff_db_sep
    


def quick_look(date_OBs, x_lims, LL_sep, RR_sep, diff_db_min_med, Frequency_start, Frequency_end, Time_start, Time_end, diff_db_sep):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]

    # Set some generic y-limits.
    y_lims = [Frequency_end, Frequency_start]

    plt.close(1)
    fig = plt.figure(1,figsize=(16,9))


    ax0, ax1, ax2, ax3, ax4 = fig.subplots(5, 1, sharey=True,sharex=True)
    # ax0 = fig.subplots(1, 1, sharey=True,sharex=True)
    fig.suptitle('NANCAY DECAMETER ARRAY: '+year+
                     '-'+month+'-'+day + '  ' + Time_start[:5] + ' - ' + Time_end[:5],fontsize=25)
    # plt.ylabel('Frequency [MHz]',fontsize=18)
    # plt.xlabel('Time (UT)',fontsize=20)


    # gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1]) 
    # ax0 = plt.subplot(gs[0])

    # plt.ylabel('Frequency [MHz]',fontsize=15)
    # plt.xlabel('Time (UT)',fontsize=15)
    # ax_0 = ax0.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
    #           aspect='auto',cmap='jet',vmin=17,vmax=30)
    # ax0.xaxis_date()
    # date_format = mdates.DateFormatter('%H:%M:%S')
    # ax0.xaxis.set_major_formatter(date_format)

    # cbar = fig.colorbar(ax_0, ax=ax0)
    # cbar.set_label("Decibel [dB]", fontsize = '20')
    # cbar.ax.tick_params(labelsize=15)
    # plt.tick_params(labelsize = 15)

    

    ax_0 = ax0.imshow(diff_db_min_med, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=0,vmax=10)
    ax0.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    ax0.xaxis.set_major_formatter(date_format)

    cbar = fig.colorbar(ax_0, ax=ax0)
    cbar.set_label("from \nBackground [dB]")

    # plt.colorbar(ax0,label='from Background [dB]')
    
    ax_1 = ax1.imshow(LL_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=0,vmax=10)
    cbar = fig.colorbar(ax_1, ax=ax1)
    cbar.set_label("from \nBackground [dB]")
    ax_2 = ax2.imshow(RR_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=0,vmax=10)
    cbar = fig.colorbar(ax_2, ax=ax2)
    cbar.set_label("from \nBackground [dB]")
    ax_3 = ax3.imshow((LL_sep - RR_sep)/(LL_sep + RR_sep), extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=-1,vmax=1)
    cbar = fig.colorbar(ax_3, ax=ax3)
    cbar.set_label("Ratio [dB]")
    ax_4 = ax4.imshow((LL_sep - RR_sep)/(LL_sep + RR_sep), extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=-0.25,vmax=0.25)
    cbar = fig.colorbar(ax_4, ax=ax4)
    cbar.set_label("Ratio [dB]")

#R-L/R+L



    
    plt.subplots_adjust(bottom=0.08,right=1,top=0.8,hspace=0.5)
    fig.autofmt_xdate()
    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/QL/'+year + '/' + month):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/QL/'+year + '/' + month)
    filename = Parent_directory + '/solar_burst/Nancay/QL/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    plt.savefig(filename)
    plt.show()
    return
    # ax0.set_ylim([0, 250])
    # ax0.set_xlabel('Year')
    # ax0.set_ylabel('M class flare',fontsize=15)
    
    # ax1 = plt.subplot(gs[1], sharex = ax0)
    # line1, = ax1.plot(year,x_flare, color='b')
    # ax1.set_ylim([0, 26])
    # ax1.set_ylabel('X class flare',fontsize=15)
    # plt.setp(ax0.get_xticklabels(), visible=False)
    # yticks = ax1.yaxis.get_major_ticks()
    # yticks[-1].label1.set_visible(False)
    # # ax0.legend((line0, line1), ('M class flare', 'X class flare'), loc='lower left')
    # plt.xlabel('Year',fontsize=15)
    # plt.subplots_adjust(hspace=.0)
    # plt.show()

    # plt.close(1)



    # figure_=plt.figure(1,figsize=(16,4))
    # figure_.suptitle('NANCAY DECAMETER ARRAY: '+year+
    #                  '-'+month+'-'+day + '  ' + Time_start[:5] + ' - ' + Time_end[:5],fontsize=23)
    # axes_2=figure_.add_subplot(111)
    # ax2 = axes_2.imshow(diff_db_min_med, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
    #           aspect='auto',cmap='jet',vmin=0,vmax=10)
    # axes_2.xaxis_date()
    # date_format = mdates.DateFormatter('%H:%M:%S')
    # axes_2.xaxis.set_major_formatter(date_format)
    # axes_2.tick_params(labelsize=ticksize)
    # # plt.title('Left Handed Polarization',fontsize=15)
    # plt.xlabel('Time (UT)',fontsize=20)
    # plt.ylabel('Frequency [MHz]',fontsize=18)
    # plt.colorbar(ax2,label='from Background [dB]')
    
    # plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    # figure_.autofmt_xdate()
    # plt.show()



    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
    #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
    # plt.savefig(filename)
    # plt.show()
    # plt.close()


sigma_value = 2
after_plot = str('af_sgepss')
time_band = 1740
# time_band = 5340
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
Freq_end = 30


# import csv
# year = str(2009)
# with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/residual_test_' + year + '_.csv', 'w') as f:
#     w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list"])
#     w.writeheader()

    # start_date, end_date = final_txt_make(Parent_directory, Parent_lab, int(year), 101, 631)
#     cnn_model = load_model_flare(Parent_directory, file_name = '/solar_burst/Nancay/data/keras/pkl_file_new/keras_param_128_0.9945.hdf5', 
#                 color_setting = 1, image_size = 128, fw = 3, strides = 1, fn_conv2d = 16, output_size = 2)




import pandas as pd

date_in=[20040908,20040912]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')

DATE=sdate
while DATE <= edate:
    date=DATE.strftime(format='%Y%m%d')
    print(date)
    try:
        yyyy = date[:4]
        mm = date[4:6]
        file_names = glob.glob(Parent_directory + '/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ date +'*cdf')
        for file_name in file_names:
            file_name = file_name.split('/')[10]
            if int(yyyy) <= 1997:
                LL_min, RR_min, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs,  diff_db= read_data_LL_RR(Parent_directory, file_name, 3, 70, 30)
            else:
                LL_min, RR_min, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs,  diff_db= read_data_LL_RR(Parent_directory, file_name, 3, 80, 30)
        
            for t in range (math.floor(((diff_db_min_med.shape[1]-time_co)/time_band) + 1)):
                LL_plot_sep, RR_plot_sep, diff_db_plot_sep, x_lims, time, Time_start, Time_end, t_1, diff_db_sep = separated_data_LL_RR(LL_min, RR_min, diff_db_min_med, epoch, time_co, time_band, t,  diff_db)
                quick_look(date_OBs, x_lims, LL_plot_sep, RR_plot_sep, diff_db_plot_sep, Frequency_start, Frequency_end, Time_start, Time_end, diff_db_sep)
                # plot_data(diff_db_plot_sep, diff_db_sep, freq_list[i], time_list[i], arr_5_list[i], x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start_list[i], event_end_list[i], freq_start_list[i], freq_end_list[i], event_time_gap_list[i], freq_gap_list[i], vmin_1_list[i], vmax_1_list[i], arr_sep_time_list[i], quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot)
    except:
        print('Plot error: ',date)
    DATE+=pd.to_timedelta(1,unit='day')