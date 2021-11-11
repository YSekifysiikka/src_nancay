#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:16:56 2021

@author: yuichiro
"""


import pandas as pd
import glob
import csv
import sys
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/afjpgu_flare_associated_ordinary_dB.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
flare_csv_1 = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


# /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/ordinary/2012/20120102_132446_133126_19720_20120_67_75_48.5_35.55peak.png

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/afjpgu_flare_associated_ordinary_dB_with_flare.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor","AR_location","X-ray_class", "peak_time_list", "peak_freq_list", "peak_dB_40MHz"])
    # w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor"])
    w.writeheader()
    for i in range(len(csv_input_final)):
        yyyy = str(csv_input_final['event_date'][i])[:4]
        mm = str(csv_input_final['event_date'][i])[4:6]
        dd = str(csv_input_final['event_date'][i])[6:8]
        hour = str(csv_input_final['event_hour'][i]).zfill(2)
        minite = str(csv_input_final['event_minite'][i]).zfill(2)
        event_start = str(csv_input_final['event_start'][i])
        event_end = str(csv_input_final['event_end'][i])
        freq_start = str(csv_input_final['freq_start'][i])
        freq_end = str(csv_input_final['freq_end'][i])
        peak_time_list = str(csv_input_final[["peak_time_list"][0]][i])
        peak_freq_list = str(csv_input_final[["peak_freq_list"][0]][i])
        peak_dB_40MHz = str(csv_input_final[['peak_dB_40MHz'][0]][i])
        files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_associated_ordinary/'+ yyyy +'/'+yyyy+mm+dd+'_*'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+'peak.png')
        if len(files)>1:
            print('break')
            break
        elif len(files)==1:
            # w.writerow({'event_date':csv_input_final['event_date'][i], 'event_hour':hour, 'event_minite':minite,'velocity':csv_input_final['velocity'][i], 'residual':csv_input_final['residual'][i], 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':csv_input_final['factor'][i]})
            files_list = files[0]
            count = 0
            flare_csv_2 = flare_csv_1[flare_csv_1['peak']>=(pd.to_datetime(files_list.split('/')[-1].split('_')[0])- pd.to_timedelta(1,unit='day')).strftime('%Y/%m/%d')]
            flare_csv = flare_csv_2[flare_csv_2['peak']<=(pd.to_datetime(files_list.split('/')[-1].split('_')[0])+ pd.to_timedelta(1,unit='day')).strftime('%Y/%m/%d')]
            for z in range (len(flare_csv['peak'])):
                j = z + flare_csv.index[0]
                yyyy = flare_csv['peak'][j].split('/')[0]
                mm = flare_csv['peak'][j].split('/')[1]
                dd = flare_csv['peak'][j].split('/')[2].split(' ')[0]
                str_date = yyyy + mm + dd
                HH = flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[0]
                MM = flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[1]
                pd_peak_time = pd.to_datetime(flare_csv['peak'][j].split('/')[0] + flare_csv['peak'][j].split('/')[1] + flare_csv['peak'][j].split('/')[2].split(' ')[0] + flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                pd_start_time = pd.to_datetime(flare_csv['start'][j].split('/')[0] + flare_csv['start'][j].split('/')[1] + flare_csv['start'][j].split('/')[2].split(' ')[0] + flare_csv['start'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                pd_end_time = pd.to_datetime(flare_csv['end'][j].split('/')[0] + flare_csv['end'][j].split('/')[1] + flare_csv['end'][j].split('/')[2].split(' ')[0] + flare_csv['end'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                if (pd_start_time >= pd.to_datetime('20070101') and pd_start_time <= pd.to_datetime('20100101')) or (pd_start_time >= pd.to_datetime('20120101') and pd_start_time <= pd.to_datetime('20150101')) or pd_end_time >= pd.to_datetime('20170101'):
                    if pd_peak_time + pd.to_timedelta(10,unit='minute') >= pd.to_datetime(files_list.split('/')[-1].split('_')[0] + files_list.split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list.split('/')[-1].split('_')[5]),unit='second'):
                        if pd_peak_time - pd.to_timedelta(10,unit='minute') <= pd.to_datetime(files_list.split('/')[-1].split('_')[0] + files_list.split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list.split('/')[-1].split('_')[5]),unit='second'):
                    # if pd_end_time >= pd.to_datetime(str(csv_input_final['event_date'][i]) + str(csv_input_final['event_hour'][i])+ str(csv_input_final['event_minite'][i]),format='%Y%m%d%H%M'):
                    #     # print (pd_peak_time, pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
                    #     if pd_start_time <= pd.to_datetime(str(csv_input_final['event_date'][i]) + str(csv_input_final['event_hour'][i])+ str(csv_input_final['event_minite'][i]),format='%Y%m%d%H%M') + pd.to_timedelta(5,unit='minute'):
                            # print (pd_peak_time)
                            ar_location = flare_csv['AR location'][j]
                            flare_class = flare_csv['X-ray class'][j]
                            count += 1

                            
            if count == 1:
                w.writerow({'event_date':csv_input_final['event_date'][i], 'event_hour':hour, 'event_minite':minite,'velocity':csv_input_final['velocity'][i], 'residual':csv_input_final['residual'][i], 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':csv_input_final['factor'][i], 'AR_location':ar_location, 'X-ray_class':flare_class, 'peak_time_list':peak_time_list, 'peak_freq_list':peak_freq_list, 'peak_dB_40MHz':peak_dB_40MHz})
                print (csv_input_final['event_date'][i])
            else:
                ar_location = 'None'
                flare_class = 'None'
                w.writerow({'event_date':csv_input_final['event_date'][i], 'event_hour':hour, 'event_minite':minite,'velocity':csv_input_final['velocity'][i], 'residual':csv_input_final['residual'][i], 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':csv_input_final['factor'][i], 'AR_location':ar_location, 'X-ray_class':flare_class, 'peak_time_list':peak_time_list, 'peak_freq_list':peak_freq_list, 'peak_dB_40MHz':peak_dB_40MHz})
                print (count)
                print (csv_input_final['event_date'][i])
                # print ('Error')
                # sys.exit()
        else:
            pass
        
        

#################################################################

import csv
import pandas as pd
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dateutil.relativedelta import relativedelta
import glob
import shutil
import os
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

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/antenna_40MHz_final.csv"
antenna1_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

obs_time = []
decibel_list = []
for i in range(len(antenna1_csv)):
    obs_time.append(datetime.datetime(int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[0]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[1]),int(antenna1_csv['obs_time'][i].split(' ')[0].split('-')[2]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[0]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[1]),int(antenna1_csv['obs_time'][i].split(' ')[1].split(':')[2][:2])))
    decibel_list.append(antenna1_csv['decibel'][i])

obs_time = np.array(obs_time)
decibel_list = np.array(decibel_list)

def check_BG(event_date, event_hour, event_minite, event_check_days, file_name, file_dir):
    event_date = str(event_date)
    yyyy = event_date[:4]
    MM = event_date[4:6]
    dd = event_date[6:8]
    hh = event_hour
    mm = event_minite
    select_date = datetime.datetime(int(yyyy),int(MM),int(dd),int(hh),int(mm))
    check_decibel = []
    check_obs_time = []
    
    start_date = select_date - datetime.timedelta(days=event_check_days/2)
    for i in range(event_check_days+1):
        check_date = start_date + datetime.timedelta(days=i)
        obs_index = np.where(obs_time == getNearestValue(obs_time,check_date))[0][0]
        if abs(obs_time[obs_index] - check_date) <= datetime.timedelta(seconds=60*90):
            check_decibel.append(decibel_list[obs_index])
            check_obs_time.append(obs_time[obs_index])
    
    check_decibel = np.array(check_decibel)
    check_obs_time = np.array(check_obs_time)
    
    plot_range = 2
    min_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date - datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
    max_obs_index = np.where(obs_time == getNearestValue(obs_time,select_date + datetime.timedelta(days=event_check_days/2 + plot_range)))[0][0]
    dB_max = max(decibel_list[min_obs_index:max_obs_index + 1])
    dB_min = min(decibel_list[min_obs_index:max_obs_index + 1])
    plt.close()
    fig=plt.figure(1,figsize=(8,4))
    ax1 = fig.add_subplot(311) 
    ax1.plot(obs_time, decibel_list,'.')
    
    ax1.xaxis_date()
    date_format = mdates.DateFormatter('%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    ax1.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
    ax1.set_ylim([dB_min, dB_max])
    ax1.set_ylabel('Decibel [dB]',fontsize=10)
    ax1.set_title('Time varidation of BG around ' + select_date.strftime("%Y/%m/%d"))
    
    
    obs_index_final = np.where(check_decibel == getNearestValue(check_decibel,np.median(check_decibel)))[0][0]
    ax2 = fig.add_subplot(313)
    ax2.plot(check_obs_time,check_decibel,'.')
    ax2.axhline(np.median(check_decibel), ls = "--", color = "magenta", label = check_obs_time[obs_index_final].strftime("%Y/%m/%d %H:%M") + ' :' + str(np.median(check_decibel))[:4] + '[dB]')
    ax2.xaxis_date()
    date_format = mdates.DateFormatter('%m-%d')
    ax2.xaxis.set_major_formatter(date_format)
    ax2.set_xlim([select_date - datetime.timedelta(days=event_check_days/2 + plot_range), select_date + datetime.timedelta(days=event_check_days/2 + plot_range)])
    ax2.set_ylim([dB_min, dB_max])
    ax2.set_ylabel('Decibel [dB]',fontsize=10)
    ax2.set_title('Time varidation of BG around ' + select_date.strftime("%H:%M"))
    ax2.legend(fontsize = 8, loc = 'upper right')
    plt.savefig(file_dir + '/' + file_name.split('p')[0] + 'peak_1.png')
    # plt.show()
    plt.close()
    return np.median(check_decibel)




file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/afjpgu_flare_associated_ordinary_dB_with_flare.csv"
ordinary_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/afjpgu_micro_dB.csv"
micro_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

# file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/ordinary_jpgu_with_dB.csv"
# ordinary_csv_with_dB = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/micro_jpgu_with_dB.csv"
micro_csv_with_dB = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

burst_types = ['flare_associated_ordinary']
csv_files = [ordinary_csv]
# csv_files_with_dB = [ordinary_csv_with_dB]
csv_files_names = ['afjpgu_flare_associated_ordinary_dB_new']

event_check_days = 30

for i in range(len(burst_types)):
    with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/'+csv_files_names[i], 'w') as f:
        w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", 
                                          "event_end", "freq_start", "freq_end", "factor", "AR_location", "X-ray_class", "peak_time_list", "peak_freq_list", "peak_dB_40MHz", "BG_decibel"])
        w.writeheader()
        for j in range(len(csv_files[i])):
            event_date = csv_files[i][["event_date"][0]][j]
            event_hour = csv_files[i][["event_hour"][0]][j]
            event_minite = csv_files[i][["event_minite"][0]][j]
            velocity = csv_files[i][["velocity"][0]][j]
            residual = csv_files[i][["residual"][0]][j]
            event_start = csv_files[i][["event_start"][0]][j]
            event_end = csv_files[i][["event_end"][0]][j]
            freq_start = csv_files[i][["freq_start"][0]][j]
            freq_end = csv_files[i][["freq_end"][0]][j]
            peak_time_list = csv_files[i][["peak_time_list"][0]][j]
            peak_freq_list = csv_files[i][["peak_freq_list"][0]][j]
            factor = csv_files[i][["factor"][0]][j]
            AR_location = csv_files[i][["AR_location"][0]][j]
            Xray_class = csv_files[i][["X-ray_class"][0]][j]
            # same_date_csv = csv_files_with_dB[i][csv_files_with_dB[i]['event_date']==event_date]
            # same_freq_start_csv = same_date_csv[same_date_csv['freq_start']==freq_start]
            # same_freq_end_csv = same_freq_start_csv[same_freq_start_csv['freq_end']==freq_end]
            # same_time_start_csv = same_freq_end_csv[same_freq_end_csv['event_start']==event_start]
            # same_time_end_csv = same_time_start_csv[same_time_start_csv['event_end']==event_end]
            # if len(same_time_end_csv)==0:
            #     print ('No data: '+event_date,event_hour,event_minite)
            #     sys.exit()
            # elif len(same_time_end_csv)>1:
            #     print ('More than 1 data: '+event_date,event_hour,event_minite)
            #     sys.exit()
            # else:
            peak_dB_40MHz = csv_files[i][['peak_dB_40MHz'][0]][j]
            # same_time_end_csv['peak_dB_40MHz'][same_time_end_csv.index[0]]
            file_place = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/'+burst_types[i]+'/' + str(event_date)[:4] + '/' + str(event_date) + '_*_' + str(event_start) + '_' + str(event_end) + '_' + str(freq_start) + '_' + str(freq_end) + '*.png')[0]
            file_name = file_place.split('/')[11]
            # /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/flare_related_ordinary/
            file_dir = Parent_directory + '/solar_burst/Nancay/plot/afjpgu_dB/'+burst_types[i]+'/'+str(event_date)[:4]
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)
            shutil.copy(file_place, file_dir)
            BG_decibel = check_BG(event_date, event_hour, event_minite, event_check_days, file_name, file_dir)
            print (event_date)
            w.writerow({'event_date':event_date, 'event_hour':event_hour, 'event_minite':event_minite,'velocity':velocity, 'residual':residual, 'event_start': event_start,'event_end': event_end
                        ,'freq_start': freq_start,'freq_end':freq_end, 'factor':factor, 'AR_location': AR_location,'X-ray_class':Xray_class, 'peak_time_list': peak_time_list, 'peak_freq_list': peak_freq_list, 'peak_dB_40MHz': peak_dB_40MHz, 'BG_decibel': BG_decibel})

# burst_types = ['storm']
# csv_files = [micro_csv]
# csv_files_with_dB = [micro_csv_with_dB]
# csv_files_names = ['afjpgu_micro_dB_new']

# for i in range(len(burst_types)):
#     with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/'+csv_files_names[i], 'w') as f:
#         w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", 
#                                           "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list", "peak_dB_40MHz", "BG_decibel"])
#         w.writeheader()
#         for j in range(len(csv_files[i])):
#             event_date = csv_files[i][["event_date"][0]][j]
#             event_hour = csv_files[i][["event_hour"][0]][j]
#             event_minite = csv_files[i][["event_minite"][0]][j]
#             velocity = csv_files[i][["velocity"][0]][j]
#             residual = csv_files[i][["residual"][0]][j]
#             event_start = csv_files[i][["event_start"][0]][j]
#             event_end = csv_files[i][["event_end"][0]][j]
#             freq_start = csv_files[i][["freq_start"][0]][j]
#             freq_end = csv_files[i][["freq_end"][0]][j]
#             peak_time_list = csv_files[i][["peak_time_list"][0]][j]
#             peak_freq_list = csv_files[i][["peak_freq_list"][0]][j]
#             factor = csv_files[i][["factor"][0]][j]
#             # AR_location = csv_files[i][["AR_location"][0]][j]
#             # Xray_class = csv_files[i][["X-ray_class"][0]][j]
#             # event_number = csv_files[i][["event_number"][0]][j]
#             # same_date_csv = csv_files_with_dB[i][csv_files_with_dB[i]['event_date']==event_date]
#             # same_freq_start_csv = same_date_csv[same_date_csv['freq_start']==freq_start]
#             # same_freq_end_csv = same_freq_start_csv[same_freq_start_csv['freq_end']==freq_end]
#             # same_time_start_csv = same_freq_end_csv[same_freq_end_csv['event_start']==event_start]
#             # same_time_end_csv = same_time_start_csv[same_time_start_csv['event_end']==event_end]
#             # if len(same_time_end_csv)==0:
#             #     print ('No data: '+event_date,event_hour,event_minite)
#             #     sys.exit()
#             # elif len(same_time_end_csv)>1:
#             #     print ('More than 1 data: '+event_date,event_hour,event_minite)
#             #     sys.exit()
#             # else:
#             peak_dB_40MHz = csv_files[i][['peak_dB_40MHz'][0]][j]
#             # peak_dB_40MHz = same_time_end_csv['peak_dB_40MHz'][same_time_end_csv.index[0]]
#             file_place = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/'+burst_types[i]+'/' + str(event_date)[:4] + '/' + str(event_date) + '_*_' + str(event_start) + '_' + str(event_end) + '_' + str(freq_start) + '_' + str(freq_end) + '*.png')[0]
#             file_name = file_place.split('/')[11]
#             file_dir = Parent_directory + '/solar_burst/Nancay/plot/afjpgu_dB/'+burst_types[i]+'/'+str(event_date)[:4]
#             if not os.path.isdir(file_dir):
#                 os.makedirs(file_dir)
#             shutil.copy(file_place, file_dir)
#             BG_decibel = check_BG(event_date, event_hour, event_minite, event_check_days, file_name, file_dir)
#             print (event_date)
#             w.writerow({'event_date':event_date, 'event_hour':event_hour, 'event_minite':event_minite,'velocity':velocity, 'residual':residual, 'event_start': event_start,'event_end': event_end
#                         ,'freq_start': freq_start,'freq_end':freq_end, 'factor':factor, 'peak_time_list': peak_time_list, 'peak_freq_list': peak_freq_list, 'peak_dB_40MHz': peak_dB_40MHz, 'BG_decibel': BG_decibel})
