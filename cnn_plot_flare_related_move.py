#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 13:07:18 2021

@author: yuichiro
"""

# #flare_event_selection

import pandas as pd
import sys
import os
import glob
import shutil

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/hinode_catalog/Hinode Flare Catalogue new.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_final_clear_event/*compare.png')
sdate = '19950101'
edate = '19971231'
files_list = []
for i in range(len(files)):
    if int(files[i].split('/')[-1].split('_')[0]) >= int(sdate) and int(files[i].split('/')[-1].split('_')[0]) <= int(edate):
        files_list.append(files[i])


event_list = []
for i in range (len(flare_csv['peak'])):
    yyyy = flare_csv['peak'][i].split('/')[0]
    mm = flare_csv['peak'][i].split('/')[1]
    dd = flare_csv['peak'][i].split('/')[2].split(' ')[0]
    str_date = yyyy + mm + dd
    HH = flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0]
    MM = flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1]
    pd_peak_time = pd.to_datetime(flare_csv['peak'][i].split('/')[0] + flare_csv['peak'][i].split('/')[1] + flare_csv['peak'][i].split('/')[2].split(' ')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
    pd_start_time = pd.to_datetime(flare_csv['start'][i].split('/')[0] + flare_csv['start'][i].split('/')[1] + flare_csv['start'][i].split('/')[2].split(' ')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
    pd_end_time = pd.to_datetime(flare_csv['end'][i].split('/')[0] + flare_csv['end'][i].split('/')[1] + flare_csv['end'][i].split('/')[2].split(' ')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
    # if pd_end_time < pd.to_datetime('20170101'):
    #     sys.exit()




# files_list[j].split('/')[-1].split('_')[0]
    if pd_start_time >= pd.to_datetime(sdate) and pd_end_time <= pd.to_datetime(edate):
        # print (pd_peak_time)
        for j in range(len(files_list)):
            # print ('aa')
            if str(files_list[j].split('/')[-1].split('_')[0]) == str_date:
                if pd_peak_time + pd.to_timedelta(10,unit='minute') >= pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'):
                    if pd_peak_time - pd.to_timedelta(10,unit='minute') <= pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'):
                        print (pd_peak_time)
                        print (pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'))
                        event_list.append(pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'))

                        file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_final_flare_related/'+ yyyy
                        if not os.path.isdir(file_dir):
                            os.makedirs(file_dir)
                        if not os.path.isfile(file_dir+'/'+files_list[j].split('/')[-1]):
                            shutil.copy(files_list[j], file_dir)
                        
                        