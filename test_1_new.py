#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:08:15 2020

@author: yuichiro
"""
import csv
with open('velocity_factor_check_why.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "freq_drift", "factor"])
    w.writeheader()
    import sys
    sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
    import cdflib
    import scipy
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import fftpack
    from scipy import signal
    import pandas as pd
    import matplotlib.dates as mdates
    import datetime as dt
    import datetime
    from astropy.time import TimeDelta, Time
    from astropy import units as u
    import os
    import math
    import glob
    import matplotlib.gridspec as gridspec
    from pynverse import inversefunc
    
    sigma_range = 1
    sigma_start = 2
    sigma_mul = 1
    #when you only check one threshold
    sigma_value = 2
    after_plot = str('drift_check')
    after_after_check = str('bursts')
    after_after_plot = str('intensity')
    #x_start_range = 10
    #x_end_range = 79.825
    #x_whole_range = 400
    x_start_range = 29.95
    x_end_range = 79.825
    x_whole_range = 286
    #check_frequency_range = 400
    check_frequency_range = 286
    time_band = 340
    time_co = 60
    move_ave = 3
    median_size = 1
    duration = 7
    threshold_frequency = 20
    freq_check_range = 20
    intensity_range = 10
    threshold_frequency_final = 3 * freq_check_range
    Parent_directory = '/Volumes/HDPH-UT/lab'

    
    def groupSequence(lst): 
        res = [[lst[0]]] 
      
        for i in range(1, len(lst)): 
            if lst[i-1]+1 == lst[i]: 
                res[-1].append(lst[i]) 
            else: 
                res.append([lst[i]]) 
        return res
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
        print (h_start)
#        print ('aaa')
#        print (fitting_new[-1])
        return (-slide_result_new[-1], time_rate_result_new[-1], fitting_new[-1], h_start)
    #####################
    #plot_date_select
    Obs_date = []
#    path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' +str(sigma_value)+ '/*/*/*'
    path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/*/*'
    File = glob.glob(path, recursive=True)
    #print(File)
    File1=len(File)
    print (File1)
    for cstr in File:
        a=cstr.split('/')
    #            line=a[9]+'\n'
#        line = a[12]
        line = a[10]
    #    print(line)
        file_name_separate = line.split('_')
        if not file_name_separate[4] == 'sigma':
            Obs_date.append(file_name_separate[0])
    Obs_date_final = np.unique(Obs_date)
    #####################
    #plot_date_time_select
    for x in range(len(Obs_date_final)):
        Obs_time_start = []
        Obs_burst_start = []
        Obs_burst_end = []
        year = Obs_date_final[x][:4]
        month = Obs_date_final[x][4:6]
        day = Obs_date_final[x][6:8]
#        path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' +str(sigma_value)+ '/' + year + '/'+ month +'/'+ Obs_date_final[x] +'*'
        path = Parent_directory + '/solar_burst/Nancay/plot/'+ after_plot +'/'+ after_after_check +'/' + year + '/'+ Obs_date_final[x] +'*'
        File = glob.glob(path, recursive=True)
        File1=len(File)
        print (File1)
        for cstr in File:
            a = cstr.split('/')
#            line = a[12]
            line = a[10]
        #    print(line)
            file_name_separate = line.split('_')
            if not file_name_separate[4] == 'sigma':
                Obs_time_start.append(int(file_name_separate[3]))
                Obs_burst_start.append(int(file_name_separate[5]))
                Obs_burst_end.append(int(file_name_separate[6]))
    #####################
    #data_import_and_editation
        path = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/*' + year + month + day + '*_' + year + month + day + '*' + '.cdf'
        File = glob.glob(path, recursive=True)
        print (File)
        for cstr in File:
            a=cstr.split('/')
            line=a[9]
            print (line)
            file_name = line
            file_name_separate =file_name.split('_')
            Date_start = file_name_separate[5]
            date_OBs=str(Date_start)
            year = date_OBs[0:4]
            month = date_OBs[4:6]
            day = date_OBs[6:8]
            start_h = date_OBs[8:10]
            start_m = date_OBs[10:12]
            Date_stop = file_name_separate[6]
            end_h = Date_stop[8:10]
            end_m = Date_stop[10:12]
            file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
            cdf_file = cdflib.CDF(file)
            epoch = cdf_file['Epoch'] 
    
            file = Parent_directory + '/solar_burst/Nancay/csv/min_med/' + year + '/' + month + '/' + file_name[:-4] + '.csv'
            diff_db_min_med_0 = pd.read_csv(file, header=None)
            diff_db_min_med = diff_db_min_med_0.values
            file = Parent_directory + '/solar_burst/Nancay/csv/min/' + year + '/' + month + '/' + file_name[:-4] + '.csv'
            min_db_0 = pd.read_csv(file, header=None)
            min_db_1 = min_db_0.values
            min_db = min_db_1[0]
            file = Parent_directory + '/solar_burst/Nancay/csv/db/' + year + '/' + month + '/' + file_name[:-4] + '.csv'
            diff_db_0 = pd.read_csv(file, header=None)
            diff_db = diff_db_0.values
            diff_power_simple = 10 ** ((diff_db[:,:])/10)
            y_power = []
            num = int(move_ave)
            b = np.ones(num)/num
            for i in range (diff_db.shape[0]):
                y_power.append(np.convolve(diff_power_simple[i], b, mode='valid'))
            y_power = np.array(y_power)

            diff_move_db = np.log10(y_power) * 10
            ####################
            for z in range(len(Obs_time_start)):
                if int(Obs_time_start[z]) % time_band == 0:
                    t = round(int(Obs_time_start[z])/time_band)
                else:
                    t = (diff_db_min_med.shape[1] + (-1*(time_band+time_co)))/time_band
                time = round(time_band*t)
                print (time)
                #+1 is due to move_average
                start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
                time = round(time_band*(t+1) + time_co)
                end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time + 1]/1000000000)
                Time_start = start.strftime('%H:%M:%S')
                Time_end = end.strftime('%H:%M:%S')
                print (start)
                print(Time_start+'-'+Time_end)
                start_1 = start.timestamp()
                end_1 = end.timestamp()
        
                x_lims_1 = list(map(dt.datetime.fromtimestamp, [start_1, end_1]))
                x_lims = mdates.date2num(x_lims_1)
        
                # Set some generic y-limits.
                y_lims = [10, 80]
        
        
                diff_db_plot_l = diff_db_min_med[:, time - time_band - time_co:time]
                diff_db_l = diff_db[:, time - time_band - time_co:time]
                diff_move_db_new = diff_move_db[:, time - time_band - time_co:time]

        
                y_over_analysis_time_l = []
                y_over_analysis_data_l = []
        
                TYPE3 = []
                TYPE3_group = [] 
                quartile_db_l = []
                mean_l_list = []
                stdev_sub = []
                quartile_power = []
        
        
        
                for i in range(0, check_frequency_range, 1):
                #    for i in range(0, 357, 1):
                    quartile_db_25_l = np.percentile(diff_db_plot_l[i], 25)
                    quartile_db_each_l = []
                    for k in range(diff_db_plot_l[i].shape[0]):
                        if diff_db_plot_l[i][k] <= quartile_db_25_l:
                            diff_power_quartile_l = (10 ** ((diff_db_plot_l[i][k])/10))
                            quartile_db_each_l.append(diff_power_quartile_l)
                #                quartile_db_each.append(math.log10(diff_power) * 10)
                    m_l = np.mean(quartile_db_each_l)
                    stdev_l = np.std(quartile_db_each_l)
                    sigma_l = m_l + sigma_value*stdev_l
                    sigma_db_l = (math.log10(sigma_l) * 10)
                    quartile_power.append(sigma_l)
                    quartile_db_l.append(sigma_db_l)
                    mean_l_list.append(math.log10(m_l) * 10)
                for i in range(len(quartile_db_l)):
                    stdev_sub.append(quartile_db_l[i] - mean_l_list[i])
        
        
                #        y3_db = diff_db_plot[0:400, j+2]
                x = np.flip(np.linspace(x_start_range, x_end_range, x_whole_range))
                x2 = np.flip(np.linspace(10, x_end_range, 400))
                x1 = x
                for j in range(0, time_band + time_co, 1):
                    y1_db_l = diff_db_plot_l[0:check_frequency_range, j]
    
                    y_le = []
                    for i in range(len(y1_db_l)):
                        diff_power_last_l = (10 ** ((y1_db_l[i])/10)) - quartile_power[i]
                        if diff_power_last_l > 0:
                            y_le.append(math.log10(diff_power_last_l) * 10)
                        else:
                            y_le.append(0)
                    y_over_l = []
                    for i in range(len(y_le)):
                        if y_le[i] > 0:
                            y_over_l.append(i)
                        else:
                            pass
                    y_over_final_l = []
                    if len(y_over_l) > 0:
                        y_over_group_l = groupSequence(y_over_l)
                        for i in range(len(y_over_group_l)):
                            if len(y_over_group_l[i]) >= threshold_frequency:
                                y_over_final_l.append(y_over_group_l[i])
                                y_over_analysis_time_l.append(round(time_band*t + j))
                                y_over_analysis_data_l.append(y_over_group_l[i])
        
                    if len(y_over_final_l) > 0:
                        TYPE3.append(round(time_band*t + j))
        
        
                TYPE3_final = np.unique(TYPE3)
                if len(TYPE3_final) > 0:
                    TYPE3_group.append(groupSequence(TYPE3_final))
        
        
                if len(TYPE3_group)>0:
                    for m in range (len(TYPE3_group[0])):
                        if len(TYPE3_group[0][m]) >= duration:
                            if TYPE3_group[0][m][0] >= 0 and TYPE3_group[0][m][-1] <= diff_db_min_med.shape[1]:
                                arr = [[-10 for i in range(400)] for j in range(400)]
                                for i in range(len(TYPE3_group[0][m])):
                                    check_start_time_l = ([y for y, x in enumerate(y_over_analysis_time_l) if x == TYPE3_group[0][m][i]])
                                    for q in range(len(check_start_time_l)):
                                        for p in range(len(y_over_analysis_data_l[check_start_time_l[q]])):
                                            arr[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)] = diff_db_plot_l[y_over_analysis_data_l[check_start_time_l[q]][p]][TYPE3_group[0][m][i] - (time - time_band - time_co)]
        #frequency_sequence
                                frequency_sequence = []
                                frequency_separate = []
                                for i in range(check_frequency_range):
                                    for j in range(400):
                                        if arr[i][j] > -10:
                                            frequency_sequence.append(i)
                                frequency_sequence_final = np.unique(frequency_sequence)
                                if len(frequency_sequence_final) > 0:
                                    frequency_separate.append(groupSequence(frequency_sequence_final))
                                    for i in range(len(frequency_separate[0])):
                                        arr1 = [[-10 for l in range(400)] for n in range(400)]
                                        for j in range(len(frequency_separate[0][i])):
                                            for k in range(400):
                                                if arr[frequency_separate[0][i][j]][k] > -10:
                                                    arr1[frequency_separate[0][i][j]][k] = arr[frequency_separate[0][i][j]][k]
        #check_time_sequence
                                        time_sequence = []
                                        time_sequence_final_group = []
                                        for j in range(400):
                                            for k in range(check_frequency_range):
                                                if not arr1[k][j] == -10:
                                                    time_sequence.append(j)
                                        time_sequence_final = np.unique(time_sequence)
                                        if len(time_sequence) > 0:
                                            time_sequence_final_group.append(groupSequence(time_sequence_final))
                                            for j in range(len(time_sequence_final_group[0])):
                                                if len(time_sequence_final_group[0][j]) >= duration:
                                                    arr2 = [[-10 for l in range(400)] for n in range(400)]
                                                    for k in range(len(time_sequence_final_group[0][j])):
                                                        for l in range(check_frequency_range):
                                                            if arr1[l][time_sequence_final_group[0][j][k]] > -10:
                                                                arr2[l][time_sequence_final_group[0][j][k]] = arr1[l][time_sequence_final_group[0][j][k]]
        #check_drift
                                                    drift_freq = []
                                                    drift_time_end = []
                                                    drift_time_start = []
                                                    for k in range(check_frequency_range):
                                                        drift_time = []
                                                        for l in range(400):
                                                            if arr2[k][l] > -10:
                                                                drift_freq.append(k)
                                                                drift_time.append(l)
                                                        if len(drift_time) > 0:
                                                            drift_time_end.append(drift_time[-1])
                                                            drift_time_start.append(drift_time[0])
                                                    if len(drift_time_end) >= threshold_frequency_final:
        #                                                if drift_time_end[0] < drift_time_end[-1]:
        #                                                if not max(drift_freq) == 285:
        #                                                    if not min(drift_freq) ==0:
        #make_arr3_for_dB_setting
                                                        freq_each_max = np.max(arr2, axis=1)
                                                        arr3 = []
                                                        for l in range(len(freq_each_max)):
                                                            if freq_each_max[l] > -10:
                                                                arr3.append(freq_each_max[l])
                                                        arr4 = []
                                                        for l in range(len(arr2)):
                                                            for k in range(len(arr2[l])):
                                                                if arr2[l][k] > -10:
                                                                    arr4.append(arr2[l][k])
                                                                
                                                        freq_list = []
                                                        time_list = []
                                                        for k in range(check_frequency_range):
                                                            if max(arr2[k]) > -10:
                                                                if (len([l for l in arr2[k] if l == max(arr2[k])])) == 1:
                                                                    freq_list.append(cdf_file['Frequency'][399 - k])
                                                                    time_list.append(np.argmax(arr2[k]))
    
                                                        p_1 = np.polyfit(freq_list, time_list, 1) #determine coefficients
    
    
                                                        if 1/p_1[0] < 0:
                                                            #drift_allen_model_*1_80_69.5
                                                            if 1/p_1[0] > -107.22096882538592:
                                                                if np.count_nonzero(cdf_file['Status'][time - time_band - time_co + 1:time + 1] == 0) == 0:
                                                                    if np.count_nonzero(cdf_file['Status'][time - time_band - time_co + 1:time + 1] == 17) == 0:
                                                                        middle = round((min(drift_time_start)+max(drift_time_end))/2) - 25
                                                                        arr5 = [[-10 for l in range(50)] for n in range(400)]
                                                                        try_list = [[-10 for l in range(50)] for n in range(400)]
                                                                        sss = 0
                                                                        for k in range(50):
                                                                            middle_k = middle + k
                                                                            if middle_k >= 0:
                                                                                if middle_k <= 399:
                                                                                    start_number = 0
                                                                                    for l in range(check_frequency_range):
                                                                                        if arr2[l][middle_k] > -10:
                                                                                            arr5[l][k] = arr2[l][middle_k]
                                                                                            try_list[l][k] = diff_move_db_new[l][min(drift_time_start) + sss]
                                                                                            start_number += 1
                                                                                    if start_number > 0:
                                                                                        sss += 1

                                                                        try_list = np.array(try_list)
                                                                        event_start = str(min(drift_time_start))
                                                                        event_end = str(max(drift_time_end))
        ######################################################################################
                                                                        if int(event_start) == int(Obs_burst_start[z]):
                                                                            if int(event_end) == int(Obs_burst_end[z]):
                                                                                ###########################################
                                                                                freq_start = str(cdf_file['Frequency'][399 - min(drift_freq)])
                                                                                freq_end = str(cdf_file['Frequency'][399 - max(drift_freq)])
                                                                                event_time_gap = str(max(drift_time_end) - min(drift_time_start) + 1)
                                                                                freq_gap = str(round(cdf_file['Frequency'][399 - min(drift_freq)] - cdf_file['Frequency'][399 - max(drift_freq)] + 0.175, 3))
                                                                                y_lims = [10, 80]
                                                                                plt.close(1)
                                                                                #                   
                                                                                figure_=plt.figure(1,figsize=(20,10))
                                                                                gs = gridspec.GridSpec(20, 10)
                                                                                axes_2 = figure_.add_subplot(gs[:, :])
                                                                                #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                                ax2 = axes_2.imshow(arr5[0:286], extent = [0, 50, 30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                                # ax2.set_label('from Background [dB]',size=32)
                                                                                # axes_2.xaxis_date()
                                                                                # date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                # axes_2.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=32)
                                                                                plt.xlabel('Time',fontsize=32)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=32)
                                                                                cbar = plt.colorbar(ax2)
                                                                                cbar.ax.tick_params(labelsize=32)
                                                                                cbar.set_label('from Background [dB]', size=32)
                                                                                axes_2.tick_params(labelsize=25)
                                                                                
                                                                                plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                                figure_.autofmt_xdate()
                                                                                if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                                    os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                                filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8] + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
                                                                                plt.savefig(filename)
                                                                                plt.show()
                                                                                plt.close()


                                                                                figure_=plt.figure(1,figsize=(18,5))
                                                                                gs = gridspec.GridSpec(21, 12)

                                                                                axes_1 = figure_.add_subplot(gs[:8, :5])
                                                                        #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                                ax1 = axes_1.imshow(diff_db_l[0:286], extent = [x_lims[0], x_lims[1],  30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= -5 + min_db[228],vmax = quartile_db_l[228] + min_db[228] + 5)
                                                                                axes_1.xaxis_date()
                                                                                date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                axes_1.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=15)
                                                                                plt.xlabel('Time (UT)',fontsize=10)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                                plt.colorbar(ax1,label='from Background [dB]')
                                                                                figure_.autofmt_xdate()
                                                                        #                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)

                                                                                axes_1 = figure_.add_subplot(gs[:8, 6:11])
                                                                                ax1 = axes_1.imshow(diff_db_plot_l[0:286], extent = [x_lims[0], x_lims[1],  30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[228] + 10)
                                                                                axes_1.xaxis_date()
                                                                                date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                axes_1.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=15)
                                                                                plt.xlabel('Time (UT)',fontsize=10)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                                plt.colorbar(ax1,label='from Background [dB]')
                                                                                figure_.autofmt_xdate()
                                                                                

                                                                                # figure_=plt.figure(1,figsize=(18,5))
                                                                                # gs = gridspec.GridSpec(21, 12)
                                                                                short_start = time - time_band - time_co + int(event_start) - intensity_range - 6
                                                                                short_end = time - time_band - time_co + int(event_end) + intensity_range + 7
                                                                                start_new = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[short_start + 1]/1000000000)
                                                                                end_new = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[short_end + 1]/1000000000)
                                                                                Time_start_new = start_new.strftime('%H:%M:%S')
                                                                                Time_end_new = end_new.strftime('%H:%M:%S')
                                                                                start_2 = start_new.timestamp()
                                                                                end_2 = end_new.timestamp()
                                                                                
                                                                                x_lims_2 = list(map(dt.datetime.fromtimestamp, [start_2, end_2]))
                                                                                x_lims_new = mdates.date2num(x_lims_2)
                                                                                

                                                                                short_data_1 = np.copy(diff_move_db[:, short_start:short_end])
                                                                                short_data_1[:, intensity_range:short_end - short_start - intensity_range] = 0
                                                                                axes_1 = figure_.add_subplot(gs[14:, 0:5])
                                                                                ax1 = axes_1.imshow(short_data_1[0:286], extent = [x_lims_new[0], x_lims_new[1],  30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= -5 + min_db[228],vmax = quartile_db_l[228] + min_db[228] + 5)
                                                                                axes_1.xaxis_date()
                                                                                date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                axes_1.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start_new)+'-'+str(Time_end_new),fontsize=15)
                                                                                plt.xlabel('Time (UT)',fontsize=10)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                                plt.colorbar(ax1,label='from Background [dB]')
                                                                                figure_.autofmt_xdate()

                                                                                # figure_=plt.figure(1,figsize=(18,5))
                                                                                # gs = gridspec.GridSpec(21, 12)
                                                                                short_data = diff_db_min_med[:, short_start:short_end]
                                                                                axes_1 = figure_.add_subplot(gs[14:, 6:11])
                                                                                ax1 = axes_1.imshow(short_data[0:286], extent = [x_lims_new[0], x_lims_new[1],  30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= min(arr4)-2 ,vmax = np.percentile(arr3, 75))
                                                                                axes_1.xaxis_date()
                                                                                date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                axes_1.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start_new)+'-'+str(Time_end_new),fontsize=15)
                                                                                plt.xlabel('Time (UT)',fontsize=10)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                                plt.colorbar(ax1,label='from Background [dB]')
                                                                                figure_.autofmt_xdate()



                                                                                
                                                                                plt.show()

                                                                                figure_=plt.figure(1,figsize=(18,5))
                                                                                gs = gridspec.GridSpec(21, 12)
                                                                                short_data_2 = np.hstack((short_data_1[:,:intensity_range],short_data_1[:,short_end - short_start - intensity_range:]))
                                                                                axes_1 = figure_.add_subplot(gs[14:, 6:11])
                                                                                ax1 = axes_1.imshow(short_data_2[0:286], extent = [0, intensity_range*2 ,  30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= -5 + min_db[228],vmax = quartile_db_l[228] + min_db[228] + 5)
                                                                                # axes_1.xaxis_date()
                                                                                # date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                # axes_1.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start_new)+'-'+str(Time_end_new),fontsize=15)
                                                                                plt.xlabel('Time [sec]',fontsize=10)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=10)
                                                                                # plt.colorbar(ax1,label='from Background [dB]')
                                                                                cbar = plt.colorbar(ax1)
                                                                                cbar.ax.tick_params(labelsize=10)
                                                                                cbar.set_label('from Background [dB]', size=10)
                                                                                figure_.autofmt_xdate()
                                                                                plt.show()
                                                                                
                                                                    

                                                                                diff_power = (10 ** ((short_data_2[:,:])/10))
                                                                                power_mean = np.mean(diff_power, axis = 1)
                                                                                arr6 = np.where(try_list > -10, (10 ** (try_list/10)), try_list)
                                                                                arr6 = arr6.T - power_mean
                                                                                arr6 = arr6.T

                                                                                arr6 = np.where(arr6 < 1, 1, arr6)
                                                                                arr7 = np.where(arr6 > 1, np.log10(arr6) * 10, -10)
                                                                                over_0 = round(np.count_nonzero(arr7 > 0) * 0.1)
                                                                                percent = (1 - over_0/20000) * 100


                                                                                x = np.arange(29.95, 80, 0.175)
                                                                                x = x[::-1]
                                                                                
                                                                                
                                                                                fig, ax = plt.subplots()
                                                                                mean_list = power_mean[0:286]
                                                                                std_list = np.std(diff_power[0:286], axis = 1)
                                                                                ax.errorbar(x, mean_list, yerr = std_list, capsize=5, fmt='.', markersize=10, ecolor='black', markeredgecolor = "black", color='w')
                                                                                ax.axvline(float(freq_start), ls = "--", color = "navy", label = 'Freq_start: ' + str(freq_start))
                                                                                ax.axvline(float(freq_end), ls = "--", color = "navy", label = 'Freq_end: ' + str(freq_end))
                                                                                ax.set_yscale("log", subsy=[2, 5])
                                                                                plt.xlim(30,80)
                                                                                # plt.ylim( -15 + min_db[228],quartile_db_l[228] + min_db[228] + 20)
                                                                                plt.tick_params(labelsize=12)
                                                                                plt.xlabel('Frequency[MHz]', fontsize = 20)
                                                                                plt.ylabel('Power[W]', fontsize = 20)
                                                                                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1) 
                                                                                plt.tick_params(labelsize=14)
                                                                                plt.show()
            

                                                                                plt.close(1)
                                                                                #                   
                                                                                figure_=plt.figure(1,figsize=(20,10))
                                                                                gs = gridspec.GridSpec(20, 10)
                                                                                axes_2 = figure_.add_subplot(gs[:, :])
                                                                                #                                    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
                                                                                ax2 = axes_2.imshow(arr7[0:286], extent = [0, 50, 30, y_lims[1]], 
                                                                                          aspect='auto',cmap='jet',vmin= np.min(np.where(arr7 < 0, 1000, arr7)) ,vmax = np.percentile(arr7, percent))
                                                                                # ax2.set_label('from Background [dB]',size=32)
                                                                                # axes_2.xaxis_date()
                                                                                # date_format = mdates.DateFormatter('%H:%M:%S')
                                                                                # axes_2.xaxis.set_major_formatter(date_format)
                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day + ' Start:'+ event_start +' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=32)
                                                                                plt.xlabel('Time[sec]',fontsize=32)
                                                                                plt.ylabel('Frequency [MHz]',fontsize=32)
                                                                                cbar = plt.colorbar(ax2)
                                                                                cbar.ax.tick_params(labelsize=32)
                                                                                cbar.set_label('from Background [dB]', size=32)
                                                                                axes_2.tick_params(labelsize=25)
                                                                                plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
                                                                                figure_.autofmt_xdate()
                                                                                # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
                                                                                #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
                                                                                # filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8] + '_' + str(time - time_band - time_co) + '_' + str(time) + '_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+'_' + 'intensity_' +'compare.png'
                                                                                # plt.savefig(filename)
                                                                                plt.show()
                                                                                plt.close()
                                                                                sys.exit()
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                
                                                                                #        
#                                                                                figure_=plt.figure(1,figsize=(20,10))
#                                                                                gs = gridspec.GridSpec(20, 10)   
#                                                                                ax2 = figure_.add_subplot(gs[:, :])
#                                                                                ax2.plot(time_list, freq_list, "bo", label = 'Peak data') #Interpolation
#                                                                                
#                                                                                ######################################################################################
#                                                                                time_rate_final = []
#                                                                                residual_list = []
#                                                                                slide_list = []
#                                                                                x_time = []
#                                                                                y_freq = []
#                                                                                factor_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
##                                                                                factor_list = [1,7,10,20]
#                                                                                for factor in factor_list:
#                                                                                    slide, time_rate5, residual, h_start = allen_model(factor, time_list, freq_list)
#                                                                                    time_rate_final.append(time_rate5)
#                                                                                    residual_list.append(residual)
#                                                                                    slide_list.append(slide)
#                                                                                    h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
#                                                                                    h5_0 = h5_0/1000
#                                                                                    x_time.append(h5_0)
#                                                                                    y_freq.append(9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))
#                                                                                cycle = 0
#                                                                                for factor in factor_list:
#                                                                                    ax2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) +'×B-A model/v=' + str(time_rate_final[cycle]) + 'c', linewidth = 3.0)
#                                                                                    cycle += 1
#                                                                                    ######################################################################################
#                                                                                xx_2 = np.linspace(min(freq_list) - 5, max(freq_list) + 5, 100)
#                                                                                yy_1 = np.polyval(p_1, xx_2)
##                                                                                ax2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
#                                                                                plt.xlabel('Time[sec]',fontsize=20)
#                                                                                plt.ylabel('Frequency[MHz]',fontsize=20)
#                                                                                plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1,fontsize=15)
#                                                                                plt.tick_params(labelsize=18)
#                                                                                plt.xlim(min(time_list), max(time_list) )
#                                                                                plt.ylim(min(freq_list), max(freq_list))
#                                                                                if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
#                                                                                    os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
#                                                                                filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8] +'_' +event_start+'_'+event_end+'_'+freq_start+'_'+freq_end +'compare.png'
#                                                                                plt.savefig(filename)
#                                                                                plt.show()
#                                                                                plt.close()
#
#                                                                                cycle = 0
#                                                                                figure_=plt.figure(1,figsize=(20,10))
#                                                                                gs = gridspec.GridSpec(20, 10)
#                                                                                axes_2 = figure_.add_subplot(gs[:, :])
##                                                                                figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
#                                                                                x_cmap = np.arange(10, 80, 0.175)
#                                                                                y_cmap = np.arange(0, 400, 1)
#                                                                                cs = axes_2.contourf(y_cmap, x_cmap, arr2[::-1], levels= 30, extend='both', vmin= 0,vmax = quartile_db_l[228] + 10)
#                                                                                cs.cmap.set_over('red')
#                                                                                cs.cmap.set_under('blue')
#                                                                                for factor in factor_list:
#                                                                                    ###########################################
#                                                                                    axes_2.plot(x_time[cycle], y_freq[cycle], '-', label = str(factor) + '×B-A model', linewidth = 3.0)
#                                                                                    cycle += 1
##                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
#                                                                                plt.xlim(min(time_list) - 10, max(time_list) + 10)
#                                                                                plt.ylim(min(freq_list), max(freq_list))
#                                                                                plt.title('Nancay: '+year+'-'+month+'-'+day+' T:'+event_time_gap + ' F:'+ freq_gap,fontsize=20)
#                                                                                plt.xlabel('Time[sec]',fontsize=20)
#                                                                                plt.ylabel('Frequency [MHz]',fontsize=20)
#                                                                                plt.tick_params(labelsize=18)
#                                                                                plt.legend(fontsize=12)
#                                                                                plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
#                                                                                figure_.autofmt_xdate()
#                                                                                if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month):
#                                                                                    os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month)
#                                                                                filename = Parent_directory + '/solar_burst/Nancay/plot/'+after_plot+'/'+after_after_plot+'/'+str(sigma_value)+'/'+year+'/'+month+'/'+year+month+day+'_'+Time_start[0:2]+Time_start[3:5]+Time_start[6:8]+'_'+Time_end[0:2]+Time_end[3:5]+Time_end[6:8] +'_'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+ '.png'
#                                                                                plt.savefig(filename)
#                                                                                plt.show()
#                                                                                plt.close()
#                                                                                best_factor = np.argmin(residual_list) + 1
#                                                                                time_event = dt.timedelta(seconds=(max(drift_time_end) + min(drift_time_start))/2) + dt.datetime(int(year), int(month), int(day),int(Time_start[0:2]), int(Time_start[3:5]), int(Time_start[6:8]))
#                                                                                date_event = str(time_event.date())[0:4] + str(time_event.date())[5:7] + str(time_event.date())[8:10]
#                                                                                date_event_hour = str(time_event.hour)
#                                                                                date_event_minute = str(time_event.minute)
#                                                                                print (time_rate_final)
#                                                                                print (best_factor)
#                                                                                w.writerow({'event_date':date_event, 'event_hour':date_event_hour, 'event_minite':date_event_minute,'velocity':time_rate_final, 'residual':residual_list, 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'freq_drift':-1/p_1[0], 'factor':best_factor})
#import pandas as pd
#csv_input = pd.read_csv(filepath_or_buffer="velocity_factor1.csv", sep=",")
## インプットの項目数（行数 * カラム数）を返却します。
#print(csv_input.size)
## 指定したカラムだけ抽出したDataFrameオブジェクトを返却します。
#print(csv_input[["velocity"][0]])