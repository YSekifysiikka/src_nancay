#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:25:42 2021

@author: yuichiro
"""

import glob
import shutil
import datetime
import numpy as np

file_dir_2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_af_sgepss/flare_clear/simple/*.png'
file_dir_1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_af_jpgu/flare_clear/simple/*.png'

files_1 = glob.glob(file_dir_1)
files_2 = glob.glob(file_dir_2)

for file_1 in files_1:
    date = file_1.split('/')[-1].split('_')[0]
    time = file_1.split('/')[-1].split('_')[1]
    start_sec = file_1.split('/')[-1].split('_')[5]
    file_final = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_final_clear_event/'+date+'_'+time+'_*_'+start_sec+'_*.png')
    if len(file_final) == 0:
        files_each = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_af_sgepss/flare_clear/simple/'+date+'_'+time+'_*_'+start_sec+'_*.png')
        files_1_each = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_af_jpgu/flare_clear/simple/'+date+'_'+time+'_*_'+start_sec+'_*.png')
        files_each.extend(files_1_each)
        if len(files_each) > 1:
            freq_max = []
            dir_place = []
            for file_each in files_each:
                freq_max.append(float(file_each.split('/')[-1].split('_')[7]))
                dir_place.append(file_each.split('/')[-4])
            idx = freq_max.index(np.max(freq_max))
            select_file = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/'+dir_place[idx]+'/flare_clear/simple/'+date+'_'+time+'_*_'+start_sec+'_*_'+str(freq_max[idx])+'*.png')[0]
            shutil.copy(select_file, '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_final_clear_event')
        else:
            shutil.copy(files_each[0], '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cnn_used_data/cnn_final_clear_event')
    else:
        pass