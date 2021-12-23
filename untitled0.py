#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 14:46:17 2021

@author: yuichiro
"""
import glob
# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
# Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'



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

# start_date, end_date = final_txt_make(Parent_directory, Parent_lab, 2013, 401, 401)


from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def load_model_flare(Parent_directory, file_name, color_setting, image_size, fw, strides, fn_conv2d, output_size):
    color_setting = color_setting  #ここを変更。データセット画像のカラー：「1」はモノクロ・グレースケール。「3」はカラー。
    image_size = 128 # ここを変更。必要に応じて変更してください。「28」を指定した場合、縦28横28ピクセルの画像に変換します。
    fw = 3
    strides = 1
    fn_conv2d = 16
    output_size = 2
    model = Sequential()
    model.add(Conv2D(fn_conv2d, (fw, fw), padding='same', strides=strides,
              input_shape=(image_size, image_size, color_setting), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))               
    model.add(Conv2D(128, (fw, fw), padding='same', strides=strides, activation='relu'))
    model.add(Conv2D(256, (fw, fw), padding='same', strides=strides, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2))                
    model.add(Dropout(0.2))                                   
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))                                 
    model.add(Dense(output_size, activation='softmax'))
    model.load_weights(Parent_directory + file_name)
    return model


import numpy as np
import cdflib

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
              aspect='auto',cmap='jet',vmin=15,vmax=30)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.tick_params(labelsize=10)
    # ax.set_xlabel('Time [UT]')
    # ax.set_ylabel('Frequency [MHz]')
    ax.set_yscale('log')
    # plt.yticks(fontsize= 15)

    

    ax1.imshow(p_data_2.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_2[0],x_lim_2[1],y_lim_2[0],y_lim_2[1]],
                vmin=vmin_2, vmax=vmax_2)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.tick_params(labelsize=10)
    ax1.set_xlabel('Time [UT]')
    # ax1.set_ylabel('Frequency [MHz]')
    ax1.set_yscale('log')
    # plt.yticks(fontsize= 15)
    # ax1.set_xticklabels(xvalues, fontsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax.tick_params(axis='y', labelsize=30)
    

    ax2.imshow(p_data_1.T, origin='lower', aspect='auto', cmap='jet',
                extent=[x_lim_1[0],x_lim_1[1],y_lim_1[0],y_lim_1[1]],
                vmin=vmin_1, vmax=vmax_1)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    # ax2.tick_params(labelsize=10)
    # ax2.set_xlabel('Time [UT]')
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
    plt.xticks(fontsize= 15)
    plt.yticks(fontsize= 15)
    plt.xlabel('Time [UT]', fontsize = 20)
    if not os.path.isdir(Parent_directory + '/solar_burst/Nancaywind2/'+year + '/' + month):
        os.makedirs(Parent_directory + '/solar_burst/Nancaywind2/'+year + '/' + month)
    filename = Parent_directory + '/solar_burst/Nancaywind2/'+year + '/' + month + '/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '.png'
    plt.savefig(filename)
    plt.show()
    return

def read_data(Parent_directory, file_name, move_ave, Freq_start, Freq_end):
    file_name_separate =file_name.split('_')
    Date_start = file_name_separate[5]
    date_OBs=str(Date_start)
    year=date_OBs[0:4]
    month=date_OBs[4:6]

    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    epoch = cdf_file['Epoch'] 
    epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
    epoch_list = []
    for i in range(len(epoch)):
        epoch_list.append(datetime.datetime(epoch[i][0], epoch[i][1], epoch[i][2], epoch[i][3], epoch[i][4], epoch[i][5], epoch[i][6]))
    epoch = np.array(epoch_list)
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
    min_db = np.log10(min_power) * 10
    diff_db_min_med = (diff_move_db.T - min_db).T
    return diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs

# diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency= read_data(Parent_directory, 'srn_nda_routine_sun_edr_201401010755_201401011553_V13.cdf', 3, 80, 30)


import math
import datetime as dt
import matplotlib.dates as mdates
import datetime

def separated_data(diff_db, diff_db_min_med, epoch, time_co, time_band, t):
    if t == math.floor((diff_db_min_med.shape[1]-time_co)/time_band):
        t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
    # if t >  36:
    #     sys.exit()
    time = round(time_band*t)
    print (time)
    #+1 is due to move_average
    start = epoch[time + 1]
    start = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
    # start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
    time = round(time_band*(t+1) + time_co)
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
    diff_db_sep = diff_db[:, time - time_band - time_co:time]
    return diff_db_plot_sep, diff_db_sep, x_lims, time, Time_start, Time_end, t
    
def threshold_array(diff_db_plot_sep, freq_start_idx, freq_end_idx, sigma_value, Frequency, threshold_frequency, duration):
    quartile_db_l = []
    mean_l_list = []
    stdev_sub = []
    quartile_power = []



    for i in range(diff_db_plot_sep.shape[0]):
    #    for i in range(0, 357, 1):
        quartile_db_25_l = np.percentile(diff_db_plot_sep[i], 25)
        quartile_db_each_l = []
        for k in range(diff_db_plot_sep[i].shape[0]):
            if diff_db_plot_sep[i][k] <= quartile_db_25_l:
                diff_power_quartile_l = (10 ** ((diff_db_plot_sep[i][k])/10))
                quartile_db_each_l.append(diff_power_quartile_l)
    #                quartile_db_each.append(math.log10(diff_power) * 10)
        m_l = np.mean(quartile_db_each_l)
        stdev_l = np.std(quartile_db_each_l)
        sigma_l = m_l + sigma_value * stdev_l
        sigma_db_l = (math.log10(sigma_l) * 10)
        quartile_power.append(sigma_l)
        quartile_db_l.append(sigma_db_l)
        mean_l_list.append(math.log10(m_l) * 10)
    quartile_power = np.array(quartile_power)
    quartile_db_l = np.array(quartile_db_l)
    mean_l_list = np.array(mean_l_list)
    stdev_sub = quartile_db_l - mean_l_list
    diff_power_last_l = ((10 ** ((diff_db_plot_sep)/10)).T - quartile_power).T
    
    arr_threshold_1 = np.where(diff_power_last_l > 1, diff_power_last_l, 1)
    arr_threshold = np.log10(arr_threshold_1) * 10
    return arr_threshold, mean_l_list, quartile_db_l, quartile_power, diff_power_last_l, stdev_sub


def plot_array_threshold(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    figure_=plt.figure(1,figsize=(12,8))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    ax2 = axes_2.imshow(arr_threshold, extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= 0 ,vmax = 1)
    # ax2 = axes_2.imshow(arr_threshold ,
    #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day, fontsize = 20)
    plt.xlabel('Time',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Decibel [dB]', size=20)
    axes_2.tick_params(labelsize=18)

    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()

    plt.ylim(Frequency[-1], Frequency[0])
    plt.show()
    plt.close()
    return

def plot_array_threshold_2(arr_threshold, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, min_db):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    figure_=plt.figure(1,figsize=(20,8))
    gs = gridspec.GridSpec(6, 4)
    axes_2 = figure_.add_subplot(gs[:, :])
    ax2 = axes_2.imshow(arr_threshold, extent = [x_lims[0], x_lims[1],  Frequency[-1], Frequency[0]], 
              aspect='auto',cmap='jet',vmin= 15,vmax = 30)
    # ax2 = axes_2.imshow(arr_threshold ,
    #           aspect='auto',cmap='jet',vmin= 2 ,vmax = 12)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day, fontsize = 20)
    plt.xlabel('Time [UT]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=18)
    cbar.set_label('Decibel [dB]', size=20)
    axes_2.tick_params(labelsize=18)

    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()

    plt.ylim(Frequency[-1], Frequency[0])
    plt.show()
    plt.close()
    return



def sep_array(arr_threshold, diff_db_plot_sep, Frequency, time_band, time_co,  quartile_power, time, duration, resolution, Status, cnn_plot_time, t_1):
    y_over_analysis_time_l = []
    y_over_analysis_data_l = []
    TYPE3 = []
    TYPE3_group = []
    y_over_l = []
    arr_5_list = []
    event_start_list = []
    event_end_list = []
    freq_start_list = []
    freq_end_list = []
    event_time_gap_list = []
    freq_gap_list = []
    vmin_1_list = []
    vmax_1_list = []
    freq_list_1 = []
    time_list_1 = []
    arr_sep_time_list = []
    for j in range(0, time_band + time_co, 1):
        y_le = arr_threshold[:, j]
        y_over_l = []
        for i in range(y_le.shape[0]):
            if y_le[i] > 0:
                y_over_l.append(i)
            else:
                pass
        y_over_final_l = []
        if len(y_over_l) > 0:
            y_over_group_l = groupSequence(y_over_l)
            for i in range(len(y_over_group_l)):
                if len(y_over_group_l[i]) >= (round(threshold_frequency/resolution,1)):
                    y_over_final_l.append(y_over_group_l[i])
                    y_over_analysis_time_l.append(round(time_band*t_1 + j))
                    y_over_analysis_data_l.append(y_over_group_l[i])

        if len(y_over_final_l) > 0:
            TYPE3.append(round(time_band * t_1 + j))


    TYPE3_final = np.unique(TYPE3)
    if len(TYPE3_final) > 0:
        TYPE3_group.append(groupSequence(TYPE3_final))


    if len(TYPE3_group)>0:
        arr_shuron = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
        arr_shuron_1 = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
        for m in range (len(TYPE3_group[0])):
            if len(TYPE3_group[0][m]) >= duration:
                if TYPE3_group[0][m][0] >= 0 and TYPE3_group[0][m][-1] <= diff_db_min_med.shape[1]:  
                    arr = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                    for i in range(len(TYPE3_group[0][m])):
    
                        check_start_time_l = ([y for y, Frequency in enumerate(y_over_analysis_time_l) if Frequency == TYPE3_group[0][m][i]])
                        for q in range(len(check_start_time_l)):
                            for p in range(len(y_over_analysis_data_l[check_start_time_l[q]])):
                                arr[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = diff_db_plot_sep[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)]
                                arr_shuron_1[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = 100
                            # print (TYPE3_group[0][m][i] - (time - time_band - time_co))
                    arr_freq_time = np.array(arr)
                    # plot_array_threshold(arr_shuron_1, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)
                    



                    frequency_sequence = []
                    frequency_separate = []
                    for i in range(arr_freq_time.shape[0]):
                        for j in range(arr_freq_time.shape[1]):
                            if arr_freq_time[i][j] > -10:
                                frequency_sequence.append(i)
                    frequency_sequence_final = np.unique(frequency_sequence)
                    if len(frequency_sequence_final) > 0:
                        frequency_separate.append(groupSequence(frequency_sequence_final))
                        for i in range(len(frequency_separate[0])):
                            arr_1 = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                            for j in range(len(frequency_separate[0][i])):
                                for k in range(arr_freq_time.shape[1]):
                                    if arr_freq_time[frequency_separate[0][i][j]][k] > -10:
                                        arr_1[frequency_separate[0][i][j]][k] = arr_freq_time[frequency_separate[0][i][j]][k]
                                        arr_shuron[frequency_separate[0][i][j]][k] = 100
                            arr_sep_freq = np.array(arr_1)
                            # plot_array_threshold(arr_shuron, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)
                            # plot_array_threshold(arr_sep_freq, x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx)

                            time_sequence = []
                            time_sequence_final_group = []
                            for j in range(arr_sep_freq.shape[1]):
                                for k in range(arr_sep_freq.shape[0]):
                                    if not arr_sep_freq[k][j] == -10:
                                        time_sequence.append(j)
                            time_sequence_final = np.unique(time_sequence)
                            if len(time_sequence) > 0:
                                time_sequence_final_group.append(groupSequence(time_sequence_final))
                                for j in range(len(time_sequence_final_group[0])):
                                    if len(time_sequence_final_group[0][j]) >= duration:
                                        arr_2 = [[-10 for i in range(time_band + time_co)] for j in range(Frequency.shape[0])]
                                        for k in range(len(time_sequence_final_group[0][j])):
                                            for l in range(arr_freq_time.shape[0]):
                                                if arr_sep_freq[l][time_sequence_final_group[0][j][k]] > -10:
                                                    arr_2[l][time_sequence_final_group[0][j][k]] = arr_sep_freq[l][time_sequence_final_group[0][j][k]]
                                        arr_sep_time = np.array(arr_2)
            
                                        drift_freq = []
                                        drift_time_end = []
                                        drift_time_start = []
                                        for k in range(arr_freq_time.shape[0]):
                                            drift_time = []
                                            for l in range(arr_freq_time.shape[1]):
                                                if arr_sep_time[k][l] > -10:
                                                    drift_freq.append(k)
                                                    drift_time.append(l)
                                            if len(drift_time) > 0:
                                                drift_time_end.append(drift_time[-1])
                                                drift_time_start.append(drift_time[0])
                                        if len(drift_time_end) >= (round(threshold_frequency_final/resolution,1)):
                                            freq_each_max = np.max(arr_sep_time, axis=1)
                                            arr3 = []
                                            for l in range(len(freq_each_max)):
                                                if freq_each_max[l] > -10:
                                                    arr3.append(freq_each_max[l])
                                            arr4 = []
                                            for l in range(len(arr_sep_time)):
                                                for k in range(len(arr_sep_time[l])):
                                                    if arr_sep_time[l][k] > -10:
                                                        arr4.append(arr_sep_time[l][k])
                                                    
                                            freq_list = []
                                            time_list = []
                                            for k in range(arr_freq_time.shape[0]):
                                                if max(arr_sep_time[k]) > -10:
                                                    if (len([l for l in arr_sep_time[k] if l == max(arr_sep_time[k])])) == 1:
                                                        freq_list.append(Frequency[k])
                                                        time_list.append(np.argmax(arr_sep_time[k]))
            
                                            p_1 = np.polyfit(freq_list, time_list, 1) #determine coefficients
            
            
                                            if 1/p_1[0] < 0:
                                                #drift_allen_model_*1_80_69.5
                                                if 1/p_1[0] > -107.22096882538592:
                                                    if np.count_nonzero(Status[time - time_band - time_co + 1:time + 1] == 0) == 0:
                                                        if np.count_nonzero(Status[time - time_band - time_co + 1:time + 1] == 17) == 0:
                                                            middle = round((min(drift_time_start)+max(drift_time_end))/2) - 25
                                                            arr_5 = [[-10 for l in range(cnn_plot_time)] for n in range(Frequency.shape[0])]
                                                            for k in range(cnn_plot_time):
                                                                middle_k = middle + k
                                                                if middle_k >= 0:
                                                                    if middle_k <= 399:
                                                                        for l in range(arr_sep_time.shape[0]):
                                                                            if arr_2[l][middle_k] > -10:
                                                                                arr_5[l][k] = arr_2[l][middle_k]
                                                            arr_5_list.append(arr_5)      
                                                            event_start_list.append(str(min(drift_time_start)))
                                                            event_end_list.append(str(max(drift_time_end)))
                                                            freq_start_list.append(str(Frequency[min(drift_freq)]))
                                                            freq_end_list.append(str(Frequency[max(drift_freq)]))
                                                            event_time_gap_list.append(str(max(drift_time_end) - min(drift_time_start) + 1))
                                                            freq_gap_list.append(str(round(Frequency[min(drift_freq)] - Frequency[max(drift_freq)] + resolution, 3)))
                                                            vmin_1_list.append(min(arr4))
                                                            vmax_1_list.append(np.percentile(arr3, 75))
                                                            freq_list_1.append(freq_list)
                                                            time_list_1.append(time_list)
                                                            arr_sep_time_list.append(arr_sep_time)
    # print(event_start_list, event_end_list)

    return arr_5_list, event_start_list, event_end_list, freq_start_list, freq_end_list, event_time_gap_list, freq_gap_list, vmin_1_list, vmax_1_list, freq_list_1, time_list_1, arr_sep_time_list

import matplotlib.pyplot as plt
import os
import random
from keras.preprocessing.image import load_img, img_to_array
import shutil
def cnn_detection(arr_5, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, date_OBs, Time_start, Time_end, color_setting, image_size, img_rows, img_cols, model, save_place, Frequency, x_lims):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    y_lims = [Frequency[-1], Frequency[0]]
    plt.close(1)

    figure_=plt.figure(1,figsize=(10,10))
    axes_2 = figure_.add_subplot(1, 1, 1)
    axes_2.imshow(arr_5, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1-2 ,vmax = vmax_1)
    plt.axis('off')
    
    extent = axes_2.get_window_extent().transformed(figure_.dpi_scale_trans.inverted())
    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all'):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all')
    filename = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
    if os.path.isfile(filename):
      os.remove(filename)
    plt.savefig(filename, bbox_inches=extent)
    # plt.show()
    plt.close()

    files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png') #ここを変更。png形式のファイルを利用する場合のサンプルです。
    print('--- 読み込んだデータセットは', Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', 'です。')

    for i, file in enumerate(random.sample(files,len(files))):  
      if color_setting == 1:
        img = load_img(file, color_mode = 'grayscale' ,target_size=(image_size, image_size))  
      elif color_setting == 3:
        img = load_img(file, color_mode = 'rgb' ,target_size=(image_size, image_size))
      array = img_to_array(img)
      x_test = np.array(array)

    if K.image_data_format() == 'channels_first':
        x_test = x_test.reshape(1, 1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(1, img_rows, img_cols, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    predict = model.predict(x_test)
    print ('flare: ' + str(predict[0][0]) + ' , others: '+ str(predict[0][1]))
    if predict[0][0] >= 0.5:
      save_directory = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/flare'
      filename2 = save_directory + '/simple/' +year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
      if os.path.isfile(filename2):
        os.remove(filename2)
      shutil.move(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory + '/simple/')
    else:
      save_directory = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/others'
      filename2 = save_directory + '/simple/' +year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
      if os.path.isfile(filename2):
        os.remove(filename2)
      shutil.move(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory + '/simple/')

    return save_directory


def cnn_detection2(arr_5, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, date_OBs, Time_start, Time_end, color_setting, image_size, img_rows, img_cols, model, save_place, Frequency, x_lims, date_event_hour, date_event_minute):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    y_lims = [Frequency[-1], Frequency[0]]
    plt.close(1)

    figure_=plt.figure(1,figsize=(10,10))
    axes_2 = figure_.add_subplot(1, 1, 1)
    axes_2.imshow(arr_5, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1-2 ,vmax = vmax_1)
    # plt.axis('off')

    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ '+date_event_hour+':'+date_event_minute,fontsize=24)
    plt.xlabel('Time (UT)',fontsize=18)
    plt.ylabel('Frequency [MHz]',fontsize=18)
    axes_2.tick_params(labelsize=15)
    figure_.autofmt_xdate()
    # extent = axes_2.get_window_extent().transformed(figure_.dpi_scale_trans.inverted())
    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all'):
        # os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all')
    # filename = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/all/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
    # if os.path.isfile(filename):
      # os.remove(filename)
    # plt.savefig(filename, bbox_inches=extent)
    plt.show()
    plt.close()
    return

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


def residual_detection(Parent_directory, save_directory, factor_list, freq_list, time_list, save_place, residual_threshold, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, Frequency):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
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
        invcube_4 = inversefunc(cube_4, y_values=Frequency)
        x_time.append(invcube_4 + slide)
        y_freq.append(Frequency)
    # if min(residual_list) <= residual_threshold:
    #     save_directory_1 = Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/flare_clear'
    #     filename2 = save_directory_1 + '/simple/' +year + month + day +'_' + Time_start[0:2] + Time_start[3:5] + Time_start[6:8] + '_' + Time_end[0:2] + Time_end[3:5] + Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
    #     if os.path.isfile(filename2):
    #       os.remove(filename2)
    #     shutil.move(Parent_directory + '/solar_burst/Nancay/plot/'+ save_place +'/flare/simple/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]  + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png', save_directory_1 + '/simple/')
    # else:
    save_directory_1 = save_directory
    return residual_list, save_directory_1, x_time, y_freq, time_rate_final


import matplotlib.gridspec as gridspec
def plot_data(diff_db_plot_sep, diff_db_sep, freq_list, time_list, arr_5, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(140,50))
    gs = gridspec.GridSpec(140, 50)


    axes_1 = figure_.add_subplot(gs[:55, :20])
#                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
    ax1 = axes_1.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 5)
    axes_1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_1.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
    plt.xlabel('Time (UT)',fontsize=fontsize )
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('Decibel [dB]', size=fontsize)
    axes_1.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()


    axes_1 = figure_.add_subplot(gs[85:, :20])
    ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
    axes_1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_1.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
    plt.xlabel('Time (UT)',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('from Background [dB]', size=fontsize)
    axes_1.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()


    axes_2 = figure_.add_subplot(gs[:, 21:34])
    ax2 = axes_2.imshow(arr_5, extent = [0, 50, 30, y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1 -2 ,vmax = vmax_1)
    plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('from Background [dB]', size=fontsize)
    axes_2.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()
    

    axes_2 = figure_.add_subplot(gs[:,38:51])
    
    ######################################################################################


    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=30)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    # cycle = 0
    # for factor in factor_list:
    #     ###########################################
    #     axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 30.0)
    #     cycle += 1
#                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
    plt.xlim(min(time_list) - 10, max(time_list) + 10)
    plt.ylim(min(freq_list), max(freq_list))
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize  + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.legend(fontsize=ticksize - 40)
    figure_.autofmt_xdate()

    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
    #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
    # plt.savefig(filename)
    plt.show()
    plt.close()





#     gs = gridspec.GridSpec(140, 50)


#     axes_1 = figure_.add_subplot(gs[:55, :20])
# #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
#     ax1 = axes_1.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 5)
#     axes_1.xaxis_date()
#     date_format = mdates.DateFormatter('%H:%M:%S')
#     axes_1.xaxis.set_major_formatter(date_format)
#     plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
#     plt.xlabel('Time (UT)',fontsize=fontsize )
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     cbar = plt.colorbar(ax1)
#     cbar.ax.tick_params(labelsize=ticksize)
#     cbar.set_label('Decibel [dB]', size=fontsize)
#     axes_1.tick_params(labelsize=ticksize)
#     figure_.autofmt_xdate()


#     axes_1 = figure_.add_subplot(gs[85:, :20])
#     ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#               aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
#     axes_1.xaxis_date()
#     date_format = mdates.DateFormatter('%H:%M:%S')
#     axes_1.xaxis.set_major_formatter(date_format)
#     plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
#     plt.xlabel('Time (UT)',fontsize=fontsize)
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     cbar = plt.colorbar(ax1)
#     cbar.ax.tick_params(labelsize=ticksize)
#     cbar.set_label('from Background [dB]', size=fontsize)
#     axes_1.tick_params(labelsize=ticksize)
#     figure_.autofmt_xdate()


#     axes_2 = figure_.add_subplot(gs[:, 21:34])
#     ax2 = axes_2.imshow(arr_5, extent = [0, 50, 30, y_lims[1]], 
#               aspect='auto',cmap='jet',vmin= vmin_1 -2 ,vmax = vmax_1)
#     plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
#     plt.xlabel('Time[sec]',fontsize=fontsize)
#     plt.ylabel('Frequency [MHz]',fontsize=fontsize)
#     cbar = plt.colorbar(ax2)
#     cbar.ax.tick_params(labelsize=ticksize)
#     cbar.set_label('from Background [dB]', size=fontsize)
#     axes_2.tick_params(labelsize=ticksize)
#     figure_.autofmt_xdate()
    


    ######################################################################################


    # axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    # x_cmap = Frequency
    # y_cmap = np.arange(0, time_band + time_co, 1)
    # cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    # cs.cmap.set_over('red')
    # cs.cmap.set_under('blue')
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        ###########################################
        # year = date_OBs[0:4]
        # month = date_OBs[4:6]
        # day = date_OBs[6:8]
        # fontsize = 110
        # ticksize = 100
        # y_lims = [Frequency[-1], Frequency[0]]
        # db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
        # plt.close()
        # figure_=plt.figure(1,figsize=(8,8))
        # axes_2 = figure_.add_subplot(gs[:,:])
        # axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
        # x_cmap = Frequency
        # y_cmap = np.arange(0, time_band + time_co, 1)
        # cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
        # cs.cmap.set_over('red')
        # cs.cmap.set_under('blue')
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == 2:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
            # plt.xlim(min(time_list) - 10, max(time_list) + 10)
            # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
            plt.xlim(np.median(time_list)-10, np.median(time_list)+40)
            plt.ylim(min(freq_list), max(freq_list))
            plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ 12:00',fontsize=20)
            plt.xlabel('Time[sec]',fontsize=20)
            plt.ylabel('Frequency [MHz]',fontsize=20)
            plt.tick_params(labelsize=18)
            plt.legend(fontsize=18)
            # figure_.autofmt_xdate()
        
            values =np.arange(0,50,5)
            x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
            plt.xticks(x,values)
        cycle += 1
    
    
        # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
        #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
        # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
        # plt.savefig(filename)
    plt.show()
    plt.close()

    return

def plot_data_non_clear(diff_db_plot_sep, diff_db_sep, freq_list, time_list, arr_5, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, date_event_hour, date_event_minute, best_factor):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(140,50))
    gs = gridspec.GridSpec(140, 50)


    axes_1 = figure_.add_subplot(gs[:55, :20])
#                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
    ax1 = axes_1.imshow(diff_db_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= -5 + min_db[db_standard],vmax = quartile_db_l[db_standard] + min_db[db_standard] + 5)
    axes_1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_1.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
    plt.xlabel('Time (UT)',fontsize=fontsize )
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('Decibel [dB]', size=fontsize)
    axes_1.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()


    axes_1 = figure_.add_subplot(gs[85:, :20])
    ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
    axes_1.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_1.xaxis.set_major_formatter(date_format)
    plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
    plt.xlabel('Time (UT)',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax1)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('from Background [dB]', size=fontsize)
    axes_1.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()


    axes_2 = figure_.add_subplot(gs[:, 21:34])
    ax2 = axes_2.imshow(arr_5, extent = [0, 50, 30, y_lims[1]], 
              aspect='auto',cmap='jet',vmin= vmin_1 -2 ,vmax = vmax_1)
    plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    cbar = plt.colorbar(ax2)
    cbar.ax.tick_params(labelsize=ticksize)
    cbar.set_label('from Background [dB]', size=fontsize)
    axes_2.tick_params(labelsize=ticksize)
    figure_.autofmt_xdate()
    

    axes_2 = figure_.add_subplot(gs[:,38:51])
    
    ######################################################################################


    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=30)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    # cycle = 0
    # for factor in factor_list:
    #     ###########################################
    #     axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 30.0)
    #     cycle += 1
#                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
    plt.xlim(min(time_list) - 10, max(time_list) + 10)
    plt.ylim(min(freq_list), max(freq_list))
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=fontsize  + 10)
    plt.xlabel('Time[sec]',fontsize=fontsize)
    plt.ylabel('Frequency [MHz]',fontsize=fontsize)
    plt.tick_params(labelsize=ticksize)
    plt.legend(fontsize=ticksize - 40)
    figure_.autofmt_xdate()

    # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
    #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ 'peak.png'
    # plt.savefig(filename)
    plt.show()
    plt.close()


    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    fontsize = 110
    ticksize = 100
    y_lims = [Frequency[-1], Frequency[0]]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    axes_2 = figure_.add_subplot(gs[:,:])
    # axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    x_cmap = Frequency
    y_cmap = np.arange(0, time_band + time_co, 1)
    cs = axes_2.contourf(y_cmap, x_cmap, arr_sep_time, levels= 30, extend='both', vmin= 5,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = 'k'
        elif factor == 4:
            color_setting = '#f781bf'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        # if factor == best_factor:
            # axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
        cycle += 1
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
        # plt.xlim(min(time_list) - 10, max(time_list) + 10)
        # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
        # plt.xlim(np.median(time_list)-10, np.median(time_list)+40)

        # figure_.autofmt_xdate()
    
        # values =np.arange(0,50,5)
        # x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
            
        # plt.xticks(x,values)
    plt.ylim(min(Frequency), max(Frequency))
    # plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ 12:00',fontsize=20)
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ '+date_event_hour+':'+date_event_minute,fontsize=20)
    plt.xlabel('Time[sec]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.tick_params(labelsize=18)
    # plt.legend(fontsize=18)

    xrange= np.arange(np.min(np.where(arr_sep_time > -10)[1]), np.min(np.where(arr_sep_time > -10)[1])+51, 10)
    plt.xlim(xrange[0], xrange[-1])
    
    values = ['0', '10', '20', '30','40','50'] 
    plt.xticks(xrange,values)
    plt.show()
    plt.close()

    return


def selected_event_plot(freq_list, time_list, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, s_event_time, e_event_time, s_event_freq, e_event_freq, selected_Frequency, resi_idx, date_event_hour, date_event_minute):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    gs = gridspec.GridSpec(140, 50)
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(time_list, freq_list, "wo", label = 'Peak data', markersize=4)
    y_cmap = selected_Frequency
    x_cmap = np.arange(s_event_time, e_event_time + 1, 1)
    cs = axes_2.contourf(x_cmap, y_cmap, arr_sep_time, levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[db_standard] + 10)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == resi_idx+1:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
            # plt.xlim(min(time_list) - 10, max(time_list) + 10)
            # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
            # plt.xlim(np.median(time_list)-10, np.median(time_list)+40)
            # figure_.autofmt_xdate()
        
            # values =np.arange(0,50,5)
            # x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
                
            # plt.xticks(x,values)
        cycle += 1
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ '+date_event_hour+':'+date_event_minute,fontsize=20)
    plt.xlabel('Time[sec]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.ylim(selected_Frequency[-1], selected_Frequency[0])
    plt.xlim(s_event_time, e_event_time)
    plt.show()
    plt.close()

    return

def selected_event_plot_2(freq_list, time_list, x_time, y_freq, time_rate_final, save_place, date_OBs, Time_start, Time_end, event_start, event_end, freq_start, freq_end, event_time_gap, freq_gap, vmin_1, vmax_1, arr_sep_time, quartile_db_l, min_db, Frequency, freq_start_idx, freq_end_idx, db_setting, after_plot, s_event_time, e_event_time, s_event_freq, e_event_freq, selected_Frequency, resi_idx, delete_idx, selected_idx, date_event_hour, date_event_minute):
    year = date_OBs[0:4]
    month = date_OBs[4:6]
    day = date_OBs[6:8]
    db_standard = np.where(Frequency == getNearestValue(Frequency,db_setting))
    plt.close()
    figure_=plt.figure(1,figsize=(8,8))
    gs = gridspec.GridSpec(140, 50)
    axes_2 = figure_.add_subplot(gs[:,:])
    axes_2.plot(np.array(time_list)[selected_idx], np.array(freq_list)[selected_idx], "wo", label = 'Peak data(selected)', markersize=4)
    axes_2.plot(np.array(time_list)[delete_idx], np.array(freq_list)[delete_idx], "ko", label = 'Peak data(deleted)', markersize=4)
    y_cmap = selected_Frequency
    x_cmap = np.arange(s_event_time, e_event_time + 1, 1)
    cs = axes_2.contourf(x_cmap, y_cmap, arr_sep_time, levels= 30, extend='both', vmin= 15,vmax = quartile_db_l[db_standard] + 25)
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cycle = 0
    for factor in factor_list:
        if factor == 1:
            color_setting = '#1f77b4'
        elif factor == 2:
            color_setting = '#ff7f0e'
        elif factor == 3:
            color_setting = '#2ca02c'
        elif factor == 4:
            color_setting = '#d62728'
        elif factor == 5:
            color_setting = '#9467bd'
        else:
            pass
        
        if factor == resi_idx+1:
            axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 6.0, color = color_setting)
            
    #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
            # plt.xlim(min(time_list) - 10, max(time_list) + 10)
            # plt.xlim(np.median(time_list)-20, np.median(time_list)+30)
            # plt.xlim(np.median(time_list)-10, np.median(time_list)+40)

            # figure_.autofmt_xdate()
        
            # values =np.arange(0,50,5)
            # x = np.arange(np.median(time_list)-10, np.median(time_list)+40, 5)
                
            # plt.xticks(x,values)
        cycle += 1
    plt.title('Nancay: '+year+'-'+month+'-'+day+ ' @ '+date_event_hour+':'+date_event_minute,fontsize=20)
    plt.xlabel('Time[sec]',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.ylim(selected_Frequency[-1], selected_Frequency[0])
    plt.xlim(s_event_time, e_event_time)
    if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year):
        os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year)
    filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+year+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8]+ '_' + str(time - time_band - time_co) + '_' + str(time) +'_' + str(np.min(np.array(time_list)[selected_idx]))+'_'+str(np.max(np.array(time_list)[selected_idx]))+'_'+str(np.max(np.array(freq_list)[selected_idx]))+'_'+str(np.min(np.array(freq_list)[selected_idx]))+ 'peak.png'
    plt.savefig(filename)
    plt.show()
    plt.close()

    return


sigma_value = 2
after_plot = str('nonclearevent_analysis')
time_band = 340
time_co = 60
move_ave = 3
duration = 7
threshold_frequency = 3.5
threshold_frequency_final = 10.5
cnn_plot_time = 50
save_place = 'cnn_used_data/shuron_plot_random'
color_setting, image_size = 1, 128
img_rows, img_cols = image_size, image_size
factor_list = [1,2,3,4,5]
residual_threshold = 100
db_setting = 40



import csv
import pandas as pd
import scipy.io as sio

# /Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/flare_clear/simple/20120424_132426_133106_20060_20460_152_159_79.825_29.95compare.png
#/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/flare_clear/simple/20120314_082927_083607_1700_2100_101_134_65.475_29.95compare.png
Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
# selecteddata =  '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_associated_ordinary/2014/20140411_112826_113506_12920_13320_85_109_58.825_31.525peak.png'
selecteddata =  '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_associated_ordinary/2014/20140411_112826_113506_12920_13320_85_109_58.825_31.525peak.png'
if len(selecteddata.split('/')) > 1:
    selecteddata = selecteddata.split('/')[-1]
selecteddata_stime = int(selecteddata.split('_')[5])
selecteddata_etime = int(selecteddata.split('_')[6])

selected_time=datetime.datetime.strptime(selecteddata.split('_')[0]+selecteddata.split('_')[1], '%Y%m%d%H%M%S')+datetime.timedelta(seconds =int(selecteddata.split('_')[5]))
selected_stime=datetime.datetime.strptime(selecteddata.split('_')[0]+selecteddata.split('_')[1], '%Y%m%d%H%M%S')+datetime.timedelta(seconds =int(selecteddata.split('_')[5])) - datetime.timedelta(minutes =45)
selected_etime=datetime.datetime.strptime(selecteddata.split('_')[0]+selecteddata.split('_')[1], '%Y%m%d%H%M%S')+datetime.timedelta(seconds =int(selecteddata.split('_')[5])) + datetime.timedelta(minutes =45)
# csvfile = selecteddata.split('.p')[0] + '.csv'




date_in=[int(selecteddata.split('_')[0]),int(selecteddata.split('_')[0])]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')


DATE=sdate



date=DATE.strftime(format='%Y%m%d')
print(date)
yyyy = date[:4]
mm = date[4:6]
file_names = glob.glob(Parent_directory + '/solar_burst/Nancay/data/'+yyyy+'/'+mm+'/*'+ date +'*cdf')
for file_name in file_names:
    file_name = file_name.split('/')[10]
    if int(yyyy) <= 2010:
        cnn_model = load_model_flare(Parent_directory, file_name = '/solar_burst/Nancay/data/keras/pkl_file_af_jpgu_70/af_jpgu_keras_param.hdf5', 
                    color_setting = 1, image_size = 128, fw = 3, strides = 1, fn_conv2d = 16, output_size = 2)
    elif int(yyyy) >= 2010:
        cnn_model = load_model_flare(Parent_directory, file_name = '/solar_burst/Nancay/data/keras/pkl_file_new/keras_param_128_0.9945.hdf5', 
                    color_setting = 1, image_size = 128, fw = 3, strides = 1, fn_conv2d = 16, output_size = 2)


    if int(yyyy) <= 1997:
        diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data(Parent_directory, file_name, 3, 70, 30)
    else:
        diff_db, diff_db_min_med, min_db, Frequency_start, Frequency_end, resolution, epoch, freq_start_idx, freq_end_idx, Frequency, Status, date_OBs= read_data(Parent_directory, file_name, 3, 80, 30)
    print (Frequency_start, Frequency_end)
    idx = np.where((epoch >= selected_stime) & (epoch <= selected_etime))[0]

    start = epoch[idx][0]
    end = epoch[idx][-1]
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)
    start = start.timestamp()
    end = end.timestamp()
    x_lims = []
    x_lims.append(dt.datetime.fromtimestamp(start))
    x_lims.append(dt.datetime.fromtimestamp(end))

    x_lims = mdates.date2num(x_lims)

    plot_array_threshold_2(diff_db[:,idx], x_lims, Frequency, date_OBs, freq_start_idx, freq_end_idx, min_db)


    if epoch[idx][0].minute >= 30:
        wind_Time = epoch[idx][0] + datetime.timedelta(minutes=1)
    else:
        wind_Time = epoch[idx][0]
    waves_setting_1 = {'receiver'   : 'rad1',
                     'date_in'    : int(date_OBs[:8]),#yyyymmdd
                     'HH'         : str(wind_Time.hour).zfill(2),#hour
                     'MM'         : str(wind_Time.minute).zfill(2),#minute
                     'SS'         : '00',#second
                     'duration'   :  90,#min
                     'freq_band'  :  [0.3, 0.7],
                     'init_param' : [0, 0],
                     'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
                     'directry'   :  Parent_directory + '/solar_burst/wind/data/R1/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
                     }
    waves_setting_2 = {'receiver'   : 'rad2',
                     'date_in'    : int(date_OBs[:8]),#yyyymmdd
                     'HH'         : str(wind_Time.hour).zfill(2),#hour
                     'MM'         : str(wind_Time.minute).zfill(2),#minute
                     'SS'         : '00',#second
                     'duration'   :  90,#min
                     'freq_band'  :  [0.3, 0.7],
                     'init_param' : [0, 0],
                     'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
                     'directry'   :  Parent_directory + '/solar_burst/wind/data/R2/'+date_OBs[:4]+'/'+date_OBs[4:6]+'/'
                     }
    try:
        rw_1 = read_waves(**waves_setting_1)
        rw_2 = read_waves(**waves_setting_2)
        
        receiver_1 = waves_setting_1['receiver']
        receiver_2 = waves_setting_2['receiver']
        data_1 = rw_1.read_rad(receiver_1)
        data_2 = rw_2.read_rad(receiver_2)
        radio_plot(data_1, receiver_1, data_2, receiver_2,diff_db[:,idx], x_lims, Frequency_start, Frequency_end, Time_start, Time_end, date_OBs[:8])
    except:
        print ('No data: wind' + date_OBs[:8])

    
    