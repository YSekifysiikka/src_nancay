#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:34:51 2021

@author: yuichiro
"""
import glob
import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import sys
import matplotlib.pyplot as plt
# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'

cnn_all_files = [Parent_directory + '/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/flare/simple/*.png',
                 Parent_directory + '/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/flare_clear/simple/*.png',
                 Parent_directory + '/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/others/simple/*.png']

date_in=[20180101, 20181231] 

flare_files = []
flare_clear_files = []
others_files = []

for cnn_files_dir in cnn_all_files:
    burst_type = cnn_files_dir.split('/')[-3]
    cnn_files = glob.glob(cnn_files_dir)
    for cnn_file in cnn_files:
        if burst_type == 'flare':
            if (int(cnn_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(cnn_file.split('/')[-1].split('_')[0]) <= date_in[1]):
                flare_files.append(cnn_file.split('/')[-1])
        elif burst_type == 'flare_clear':
            if (int(cnn_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(cnn_file.split('/')[-1].split('_')[0]) <= date_in[1]):
                flare_clear_files.append(cnn_file.split('/')[-1])
        else:
            if (int(cnn_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(cnn_file.split('/')[-1].split('_')[0]) <= date_in[1]):
                others_files.append(cnn_file.split('/')[-1])

final_selected_files = []
selected_files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/afjpgusimpleselect/*/*/*.png')
for selected_file in selected_files:
    if (int(selected_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(selected_file.split('/')[-1].split('_')[0]) <= date_in[1]):
        files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/flare_clear/simple/' + selected_file.split('/')[-1].split('p')[0] + 'compare.png')
        if len(files) == 1:
            # print ('Yes')
            final_selected_files.append(selected_file.split('/')[-1])
        else:
            print ('Error')
            sys.exit()
            
# date_in=[20120101, 20141231] 
# for cnn_files_dir in cnn_all_files:
#     burst_type = cnn_files_dir.split('/')[-3]
#     cnn_files = glob.glob(cnn_files_dir)
#     for cnn_file in cnn_files:
#         if burst_type == 'flare':
#             if (int(cnn_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(cnn_file.split('/')[-1].split('_')[0]) <= date_in[1]):
#                 flare_files.append(cnn_file.split('/')[-1])
#         elif burst_type == 'flare_clear':
#             if (int(cnn_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(cnn_file.split('/')[-1].split('_')[0]) <= date_in[1]):
#                 flare_clear_files.append(cnn_file.split('/')[-1])
#         else:
#             if (int(cnn_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(cnn_file.split('/')[-1].split('_')[0]) <= date_in[1]):
#                 others_files.append(cnn_file.split('/')[-1])

# final_selected_files = []
# selected_files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/afjpgusimpleselect/*/*/*.png')
# for selected_file in selected_files:
#     if (int(selected_file.split('/')[-1].split('_')[0]) >= date_in[0]) and (int(selected_file.split('/')[-1].split('_')[0]) <= date_in[1]):
#         files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/cnn_used_data/cnn_shuron/flare_clear/simple/' + selected_file.split('/')[-1].split('p')[0] + 'compare.png')
#         if len(files) == 1:
#             # print ('Yes')
#             final_selected_files.append(selected_file.split('/')[-1])
#         else:
#             print ('Error')
#             sys.exit()

print ('Threshold ' + str(len(others_files)+len(flare_clear_files)+len(flare_files)))
print ('   ratio ' + str(((len(flare_clear_files)+len(flare_files))/(len(others_files)+len(flare_clear_files)+len(flare_files)))))
print ('CNN ' + str(len(flare_clear_files)+len(flare_files)))
print ('   ratio ' + str((len(flare_clear_files))/(len(flare_clear_files)+len(flare_files))))
print ('Curve fitting ' + str(len(flare_clear_files)))
print ('   ratio ' + str((len(final_selected_files)/len(flare_clear_files))))
print ('Visual check ' + str(len(final_selected_files)))



