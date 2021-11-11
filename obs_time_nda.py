#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:53:59 2020

@author: yuichiro
"""
import glob
import datetime
import numpy as np


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


years = ['2012','2013','2014','2017','2018','2019']
for year in years:
    total_time = []
    start_hour = []
    start_minite = []
    end_hour = []
    end_minite = []
    path=Parent_directory + '/solar_burst/Nancay/data/' + year + '/*/*'+'.cdf'
    File=glob.glob(path, recursive=True)
    #print(File)
    File1=len(File)
    #        print (File1)
    for cstr in File:
        a=cstr.split('/')
        line = a[Parent_lab + 6]
    #    print(line)
        file_name_separate = line.split('_')
    #################################################
        Date_start = file_name_separate[5]
        year_0 = int(Date_start[:4])
        if int(Date_start[4:5]) == 0:
            month_0 = int(Date_start[5:6])
        else:
            month_0 = int(Date_start[4:6])
        if int(Date_start[6:7]) == 0:
            day_0 = int(Date_start[7:8])
        else:
            day_0 = int(Date_start[6:8])
        if int(Date_start[8:9]) == 0:
            hour_0 = int(Date_start[9:10])
            start_hour.append(hour_0)
        else:
            hour_0 = int(Date_start[8:10])
            start_hour.append(hour_0)
        if int(Date_start[10:11]) == 0:
            minites_0 = int(Date_start[11:12])
            start_minite.append(minites_0)
        else:
            minites_0 = int(Date_start[10:12])
            start_minite.append(minites_0)
    #################################################
        Date_end = file_name_separate[6]
        year_1 = int(Date_end[:4])
        if int(Date_end[4:5]) == 0:
            month_1 = int(Date_end[5:6])
        else:
            month_1 = int(Date_end[4:6])
        if int(Date_end[6:7]) == 0:
            day_1 = int(Date_end[7:8])
        else:
            day_1 = int(Date_end[6:8])
        if int(Date_end[8:9]) == 0:
            hour_1 = int(Date_end[9:10])
            end_hour.append(hour_1)
        else:
            hour_1 = int(Date_end[8:10])
            end_hour.append(hour_1)
        if int(Date_end[10:11]) == 0:
            minites_1 = int(Date_end[11:12])
            end_minite.append(minites_1)
        else:
            minites_1 = int(Date_end[10:12])
            end_minite.append(minites_1)
        obs_start = datetime.datetime(year_0, month_0, day_0, hour_0, minites_0, tzinfo=datetime.timezone.utc).timestamp()
        obs_end =  datetime.datetime(year_1, month_1, day_1, hour_1, minites_1, tzinfo=datetime.timezone.utc).timestamp()
        total_time.append((obs_end - obs_start)/3600)
    #    sys.exit()
    print(sum(total_time))
    
    obs_time = np.array([float(0)]*24)
    for i in range(len(start_hour)):
        obs_time_oneday = []
        for j in range(24):
            if start_hour[i] == j:
                obs_time_oneday.append(1 - start_minite[i]/60)
            elif start_hour[i] > j:
                obs_time_oneday.append(0)
            else:
                if end_hour[i] > j:
                    obs_time_oneday.append(1)
                elif end_hour[i] == j:
                    obs_time_oneday.append(end_minite[i]/60)
                else:
                    obs_time_oneday.append(0)
        obs_time += np.array(obs_time_oneday)
    print (sum(obs_time))