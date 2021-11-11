#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:01:12 2021

@author: yuichiro
"""

import pandas as pd
import glob
import csv
import sys
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1

file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

flare_csv = flare_csv[flare_csv['peak']>'2007/01/01']
# /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/ordinary/2012/20120102_132446_133126_19720_20120_67_75_48.5_35.55peak.png

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/ordinary_burst_jpgu_final.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor","AR_location","X-ray_class"])
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
        files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cleareventfinaljpgu/ordinary/'+ yyyy +'/'+yyyy+mm+dd+'_*'+event_start+'_'+event_end+'_'+freq_start+'_'+freq_end+'peak.png')
        if len(files)>1:
            print('break')
            break
        elif len(files)==1:
            # w.writerow({'event_date':csv_input_final['event_date'][i], 'event_hour':hour, 'event_minite':minite,'velocity':csv_input_final['velocity'][i], 'residual':csv_input_final['residual'][i], 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':csv_input_final['factor'][i]})
            
            count = 0
            for j in range (len(flare_csv['peak'])):
                yyyy = flare_csv['peak'][j].split('/')[0]
                mm = flare_csv['peak'][j].split('/')[1]
                dd = flare_csv['peak'][j].split('/')[2].split(' ')[0]
                str_date = yyyy + mm + dd
                HH = flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[0]
                MM = flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[1]
                pd_peak_time = pd.to_datetime(flare_csv['peak'][j].split('/')[0] + flare_csv['peak'][j].split('/')[1] + flare_csv['peak'][j].split('/')[2].split(' ')[0] + flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                pd_start_time = pd.to_datetime(flare_csv['start'][j].split('/')[0] + flare_csv['start'][j].split('/')[1] + flare_csv['start'][j].split('/')[2].split(' ')[0] + flare_csv['start'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                pd_end_time = pd.to_datetime(flare_csv['end'][j].split('/')[0] + flare_csv['end'][j].split('/')[1] + flare_csv['end'][j].split('/')[2].split(' ')[0] + flare_csv['end'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
                if pd_start_time <= pd.to_datetime('20141231') or pd_end_time >= pd.to_datetime('20170101'):
                    if pd_end_time >= pd.to_datetime(str(csv_input_final['event_date'][i]) + str(csv_input_final['event_hour'][i])+ str(csv_input_final['event_minite'][i]),format='%Y%m%d%H%M'):
                        # print (pd_peak_time, pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
                        if pd_start_time <= pd.to_datetime(str(csv_input_final['event_date'][i]) + str(csv_input_final['event_hour'][i])+ str(csv_input_final['event_minite'][i]),format='%Y%m%d%H%M') + pd.to_timedelta(5,unit='minute'):

                            ar_location = flare_csv['AR location'][j]
                            flare_class = flare_csv['X-ray class'][j]
                            count += 1

                            
            if count == 1:
                w.writerow({'event_date':csv_input_final['event_date'][i], 'event_hour':hour, 'event_minite':minite,'velocity':csv_input_final['velocity'][i], 'residual':csv_input_final['residual'][i], 'event_start': event_start,'event_end': event_end,'freq_start': freq_start,'freq_end':freq_end, 'factor':csv_input_final['factor'][i], 'AR_location':ar_location, 'X-ray_class':flare_class})
                print (csv_input_final['event_date'][i])
            else:
                print (count)
                print (csv_input_final['event_date'][i])
                print ('Error')
                sys.exit()
        else:
            pass
    # print (len(files))
    # if len(files)>1:
    #     print (len(files))