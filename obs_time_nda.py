#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:53:59 2020

@author: yuichiro
"""
import glob
import datetime
import numpy as np
import cdflib
import csv
import pandas as pd

Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'

Parent_lab = len(Parent_directory.split('/')) - 1


date_in=[20180101, 20191231]


start_day, end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')

DATE=sdate


with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/nda_obs_'+str(date_in[0])+'_'+str(date_in[-1])+'.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_start", "obs_end", "freq_start", "freq_end"])
    w.writeheader()
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        yyyy = date[:4]
        mm = date[4:6]
        path=Parent_directory + '/solar_burst/Nancay/data/' + yyyy + '/'+mm+'/*'+date+'*'+'.cdf'
        Files=sorted(glob.glob(path, recursive=True))
        #print(File)
        # File1=len(File)
        #        print (File1)
        for file in Files:

            file_name =file.split('/')[-1]
            print (file_name)
            Date_start = file_name.split('_')[5]
            date_OBs=str(Date_start)
            year=date_OBs[0:4]
            month=date_OBs[4:6]
        
            file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
            cdf_file = cdflib.CDF(file)
            epoch = cdf_file['Epoch'] 
            epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
            start = epoch[0]
            obs_start = datetime.datetime(start[0], start[1], start[2], start[3], start[4], start[5], start[6])
            end = epoch[-1]
            obs_end = datetime.datetime(end[0], end[1], end[2], end[3], end[4], end[5], end[6])
            Frequency = cdf_file['Frequency']
            freq_start = Frequency[-1]
            freq_end = Frequency[0]
            w.writerow({'obs_start':obs_start, 'obs_end':obs_end, 'freq_start':freq_start,'freq_end':freq_end})
        DATE+=pd.to_timedelta(1,unit='day')
        
        
        
        
        
        
        
        
        
    # #################################################
    #     Date_start = file_name_separate[5]
    #     year_0 = int(Date_start[:4])
    #     if int(Date_start[4:5]) == 0:
    #         month_0 = int(Date_start[5:6])
    #     else:
    #         month_0 = int(Date_start[4:6])
    #     if int(Date_start[6:7]) == 0:
    #         day_0 = int(Date_start[7:8])
    #     else:
    #         day_0 = int(Date_start[6:8])
    #     if int(Date_start[8:9]) == 0:
    #         hour_0 = int(Date_start[9:10])
    #         start_hour.append(hour_0)
    #     else:
    #         hour_0 = int(Date_start[8:10])
    #         start_hour.append(hour_0)
    #     if int(Date_start[10:11]) == 0:
    #         minites_0 = int(Date_start[11:12])
    #         start_minite.append(minites_0)
    #     else:
    #         minites_0 = int(Date_start[10:12])
    #         start_minite.append(minites_0)
    # #################################################
    #     Date_end = file_name_separate[6]
    #     year_1 = int(Date_end[:4])
    #     if int(Date_end[4:5]) == 0:
    #         month_1 = int(Date_end[5:6])
    #     else:
    #         month_1 = int(Date_end[4:6])
    #     if int(Date_end[6:7]) == 0:
    #         day_1 = int(Date_end[7:8])
    #     else:
    #         day_1 = int(Date_end[6:8])
    #     if int(Date_end[8:9]) == 0:
    #         hour_1 = int(Date_end[9:10])
    #         end_hour.append(hour_1)
    #     else:
    #         hour_1 = int(Date_end[8:10])
    #         end_hour.append(hour_1)
    #     if int(Date_end[10:11]) == 0:
    #         minites_1 = int(Date_end[11:12])
    #         end_minite.append(minites_1)
    #     else:
    #         minites_1 = int(Date_end[10:12])
    #         end_minite.append(minites_1)
    #     obs_start = datetime.datetime(year_0, month_0, day_0, hour_0, minites_0, tzinfo=datetime.timezone.utc).timestamp()
    #     obs_end =  datetime.datetime(year_1, month_1, day_1, hour_1, minites_1, tzinfo=datetime.timezone.utc).timestamp()
    #     total_time.append((obs_end - obs_start)/3600)
    # #    sys.exit()
    # print(sum(total_time))
    
    # obs_time = np.array([float(0)]*24)
    # for i in range(len(start_hour)):
    #     obs_time_oneday = []
    #     for j in range(24):
    #         if start_hour[i] == j:
    #             obs_time_oneday.append(1 - start_minite[i]/60)
    #         elif start_hour[i] > j:
    #             obs_time_oneday.append(0)
    #         else:
    #             if end_hour[i] > j:
    #                 obs_time_oneday.append(1)
    #             elif end_hour[i] == j:
    #                 obs_time_oneday.append(end_minite[i]/60)
    #             else:
    #                 obs_time_oneday.append(0)
    #     obs_time += np.array(obs_time_oneday)
    # print (sum(obs_time))