#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 12:22:35 2021

@author: yuichiro
"""
# import pandas as pd
# import sys
# import os
# import glob
# import shutil

# Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
# Parent_lab = len(Parent_directory.split('/')) - 1
# file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
# flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

# burst_type = 'ordinary'
# # flare_event.csv
# if burst_type == 'ordinary':
#     file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
# if burst_type == 'storm':
#     file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/storm_burst_cycle24.csv"

# # /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv
# type_3_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")



# event_list = []

# for i in range (len(flare_csv['peak'])):
#     yyyy = flare_csv['peak'][i].split('/')[0]
#     mm = flare_csv['peak'][i].split('/')[1]
#     dd = flare_csv['peak'][i].split('/')[2].split(' ')[0]
#     str_date = yyyy + mm + dd
#     HH = flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0]
#     MM = flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1]
#     pd_peak_time = pd.to_datetime(flare_csv['peak'][i].split('/')[0] + flare_csv['peak'][i].split('/')[1] + flare_csv['peak'][i].split('/')[2].split(' ')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['peak'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
#     pd_start_time = pd.to_datetime(flare_csv['start'][i].split('/')[0] + flare_csv['start'][i].split('/')[1] + flare_csv['start'][i].split('/')[2].split(' ')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
#     pd_end_time = pd.to_datetime(flare_csv['end'][i].split('/')[0] + flare_csv['end'][i].split('/')[1] + flare_csv['end'][i].split('/')[2].split(' ')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][i].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
#     if pd_end_time < pd.to_datetime('20120101'):
#         sys.exit()

#     if pd_start_time <= pd.to_datetime('20141231') or pd_end_time >= pd.to_datetime('20170101'):
#         # print (pd_peak_time)
#         for j in range(len(type_3_csv)):
#             # print ('aa')
#             if str(type_3_csv['event_date'][j]) == str_date:
#                 # if pd_peak_time + pd.to_timedelta(10,unit='minute') >= pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'):
#                 if pd_end_time >= pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'):
#                     # print (pd_peak_time, pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
#                     # if pd_peak_time - pd.to_timedelta(10,unit='minute') <= pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'):
#                     if pd_start_time <= pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M') + pd.to_timedelta(5,unit='minute'):
#                         # print ('aa')
#                         print (pd_peak_time)
#                         print (pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
#                         event_list.append(pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
#                         file = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/'+burst_type+'/'+yyyy +'/'+str_date+'_*_' + str(type_3_csv['event_start'][j]) + '_' + str(type_3_csv['event_end'][j]) + '_' + str(type_3_csv['freq_start'][j]) + '_' + str(type_3_csv['freq_end'][j]) + 'peak.png')
#                         if len(file) > 1:
#                             print ('Too much data: ' + pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
#                             sys.exit()
#                         elif len(file) == 1:
#                             file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/'+burst_type+'/' + yyyy
#                             if not os.path.isdir(file_dir):
#                                 os.makedirs(file_dir)
#                             shutil.copy(file[0], file_dir)
#                         else:
#                             print ('Something wrong: ' + str(pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M')))
#                             sys.exit()


# #flare_event_selection

import pandas as pd
import sys
import os
import glob
import shutil

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

# burst_type = 'storm'
# if burst_type == 'ordinary':

#     file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
# if burst_type == 'storm':
#     file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/storm_burst_cycle24.csv"

# /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv
# type_3_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
files = glob.glob(Parent_directory + '/solar_burst/Nancay/plot/afjpgu_simple_select/marginal/*/*peak.png')
sdate = '20120101'
edate = '20141231'
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
    if pd_end_time < pd.to_datetime('20070101'):
        sys.exit()




# files_list[j].split('/')[-1].split('_')[0]
    if pd_start_time >= pd.to_datetime(sdate) and pd_end_time <= pd.to_datetime(edate):
        # print (pd_peak_time)
        for j in range(len(files_list)):
            # print ('aa')
            if str(files_list[j].split('/')[-1].split('_')[0]) == str_date:
                if pd_peak_time + pd.to_timedelta(10,unit='minute') >= pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'):
                # if pd_end_time >= pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'):
                    # print (pd_peak_time, pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
                    if pd_peak_time - pd.to_timedelta(10,unit='minute') <= pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'):
                    # if pd_start_time <= pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M') + pd.to_timedelta(5,unit='minute'):
                        # print ('aa')
                        print (pd_peak_time)
                        print (pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'))
                        event_list.append(pd.to_datetime(files_list[j].split('/')[-1].split('_')[0] + files_list[j].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+ pd.to_timedelta(int(files_list[j].split('/')[-1].split('_')[5]),unit='second'))
                        # file = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cleareventfinaljpgu/'+burst_type+'/'+yyyy +'/'+str_date+'_*_' + str(type_3_csv['event_start'][j]) + '_' + str(type_3_csv['event_end'][j]) + '_' + str(type_3_csv['freq_start'][j]) + '_' + str(type_3_csv['freq_end'][j]) + 'peak.png')
                        # if len(file) > 1:
                        #     print ('Too much data: ' + pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M'))
                        #     sys.exit()
                        # elif len(file) == 1:
                        file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgu_simple_select/maybe_ordinary/'+ yyyy
                        if not os.path.isdir(file_dir):
                            os.makedirs(file_dir)
                        shutil.move(files_list[j], file_dir)

                            # print ('Something wrong: ' + str(pd.to_datetime(str(type_3_csv['event_date'][j]) + str(type_3_csv['event_hour'][j])+ str(type_3_csv['event_minite'][j]),format='%Y%m%d%H%M')))
                            # sys.exit()

# import glob
# import os
# import shutil
# files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/ordinary/*/*.png')
# for file in files:
#     if os.path.exists('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cleareventfinaljpgu/ordinary/' + file.split('/')[10] + '/' + file.split('/')[11]) == True:
#         file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/解析リスト/' + file.split('/')[10]
#         if not os.path.isdir(file_dir):
#             os.makedirs(file_dir)
#         shutil.copy(file, file_dir)
#     else:
#         if os.path.exists('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/ordinary/' + file.split('/')[10] + '/' + file.split('/')[11]) == True:
#             file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/解析検討リスト/' + file.split('/')[10]
#             if not os.path.isdir(file_dir):
#                 os.makedirs(file_dir)
#             shutil.copy(file, file_dir)
#         else:
#             file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/非解析検討リスト/' + file.split('/')[10]
#             if not os.path.isdir(file_dir):
#                 os.makedirs(file_dir)
#             shutil.copy(file, file_dir)

# import glob
# import os
# import shutil
# files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/cleareventfinaljpgu/storm/*/*.png')
# for file in files:
#     if os.path.exists('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgu2021_1/storm_1/' + file.split('/')[10] + '/' + file.split('/')[11]) == True:
#         pass
#         # file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/clearevent_test/解析リスト/' + file.split('/')[10]
#         # if not os.path.isdir(file_dir):
#         #     os.makedirs(file_dir)
#         # shutil.copy(file, file_dir)
#     else:
#         file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgu2021/storm/' + file.split('/')[10]
#         if not os.path.isdir(file_dir):
#             os.makedirs(file_dir)
#         shutil.copy(file, file_dir)





