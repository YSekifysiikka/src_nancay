#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 09:47:01 2021

@author: yuichiro
"""
import pandas as pd
import sys
import os
import glob
import shutil
import csv

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

burst_type = 'ordinary'
# flare_event.csv
if burst_type == 'ordinary':
    # file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/flare_event.csv"
if burst_type == 'storm':
    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/storm_burst_cycle24.csv"

type_3_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/afjpgu_'+burst_type+'1.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["event_date", "event_hour", "event_minite", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor"])
    w.writeheader()
    for j in range(len(type_3_csv)):
        # print (pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
        yyyy = str(type_3_csv['event_date'][j])[:4]
        str_date = str(type_3_csv['event_date'][j])
    
        file = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgu2021/'+burst_type+'/'+yyyy +'/'+str_date+'_*_' + str(type_3_csv['event_start'][j]) + '_' + str(type_3_csv['event_end'][j]) + '_' + str(type_3_csv['freq_start'][j]) + '_' + str(type_3_csv['freq_end'][j]) + 'peak.png')
        if len(file) > 1:
            print ('Too much data: ' + pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
            sys.exit()
        elif len(file) == 1:
            # file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/'+burst_type+'/' + yyyy
            w.writerow({'event_date':type_3_csv['event_date'][j], 'event_hour':type_3_csv['event_hour'][j], 'event_minite':type_3_csv['event_minite'][j],'velocity':type_3_csv['velocity'][j], 'residual':type_3_csv['residual'][j], 'event_start': type_3_csv['event_start'][j],'event_end': type_3_csv['event_end'][j],'freq_start': type_3_csv['freq_start'][j],'freq_end':type_3_csv['freq_end'][j], 'factor':type_3_csv['factor'][j]})
        else:
            pass
            # print ('Something wrong: ' + str(pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M')))
            # sys.exit()