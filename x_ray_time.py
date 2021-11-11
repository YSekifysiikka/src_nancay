#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 13:16:09 2019

@author: yuichiro
"""

import pandas as pd
import glob
import datetime
df = pd.read_csv('/Volumes/HDPH-UT/lab/Hinode Flare Catalogue.csv')
year = str(2018)

#df['Event number']
#df['start']
#df['peak']
#df['end']
#df['AR location']
#df['X-ray class']
#df['FG']
#df['SP']
#df['XRT']
#df['EIS']
#df['DARTS']
#df['RHESSI']
#df['Suzaku/WAM']
#df['NoRH']
#line1 = []
total_time = []
for i in range(df.shape[0]):
    start = df['start'][i][:4] + df['start'][i][5:7] + df['start'][i][8:10] + df['start'][i][11:13] + df['start'][i][14:16]
    peak = df['peak'][i][:4] + df['peak'][i][5:7] + df['peak'][i][8:10] + df['peak'][i][11:13] + df['peak'][i][14:16]
    end = df['end'][i][:4] + df['end'][i][5:7] + df['end'][i][8:10] + df['end'][i][11:13] + df['end'][i][14:16]
    if start[:4] == str(year):
        list = []
        list.append(start)
        list.append(end)
        year_0 = int(str(list[0])[:4])
        if int(str(list[0])[4:5]) == 0:
            month_0 = int(str(list[0])[5:6])
        else:
            month_0 = int(str(list[0])[4:6])
        if int(str(list[0])[6:7]) == 0:
            day_0 = int(str(list[0])[7:8])
        else:
            day_0 = int(str(list[0])[6:8])
        if int(str(list[0])[8:9]) == 0:
            hour_0 = int(str(list[0])[9:10])
        else:
            hour_0 = int(str(list[0])[8:10])
        if int(str(list[0])[10:11]) == 0:
            minites_0 = int(str(list[0])[11:12])
        else:
            minites_0 = int(str(list[0])[10:12])

        year_1 = int(str(list[1])[:4])
        if int(str(list[1])[4:5]) == 0:
            month_1 = int(str(list[1])[5:6])
        else:
            month_1 = int(str(list[1])[4:6])
        if int(str(list[1])[6:7]) == 0:
            day_1 = int(str(list[1])[7:8])
        else:
            day_1 = int(str(list[1])[6:8])
        if int(str(list[1])[8:9]) == 0:
            hour_1 = int(str(list[1])[9:10])
        else:
            hour_1 = int(str(list[1])[8:10])
        if int(str(list[1])[10:11]) == 0:
            minites_1 = int(str(list[1])[11:12])
        else:
            minites_1 = int(str(list[1])[10:12])
        flare_start = datetime.datetime(year_0, month_0, day_0, hour_0, minites_0, tzinfo=datetime.timezone.utc).timestamp()
        flare_end =  datetime.datetime(year_1, month_1, day_1, hour_1, minites_1, tzinfo=datetime.timezone.utc).timestamp()
#        line1.append(start+','+end)
#    line1.append(peak)
#        print("{0[-1]}".format(line1), file=i)  

#check_start_and_end
#with open('/Volumes/HDPH-UT/lab/flare_history_all.txt', 'w') as i:
#    for cstr in line1:
#        print (cstr)
#        i.write(cstr+'\n')
#    i.close()
        path='/Volumes/HDPH-UT/lab/solar_burst/Nancay/data/'+ year +'/*/*'+'.cdf'
        File=glob.glob(path, recursive=True)
        #print(File)
        File1=len(File)
#        print (File1)
        for cstr in File:
            a=cstr.split('/')
#            line=a[9]+'\n'
            line = a[9]
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
            else:
                hour_0 = int(Date_start[8:10])
            if int(Date_start[10:11]) == 0:
                minites_0 = int(Date_start[11:12])
            else:
                minites_0 = int(Date_start[10:12])
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
            else:
                hour_1 = int(Date_end[8:10])
            if int(Date_end[10:11]) == 0:
                minites_1 = int(Date_end[11:12])
            else:
                minites_1 = int(Date_end[10:12])
            obs_start = datetime.datetime(year_0, month_0, day_0, hour_0, minites_0, tzinfo=datetime.timezone.utc).timestamp()
            obs_end =  datetime.datetime(year_1, month_1, day_1, hour_1, minites_1, tzinfo=datetime.timezone.utc).timestamp()
            if obs_start <= flare_start < flare_end <= obs_end:
                total_time.append((flare_end - flare_start)/3600)
            elif flare_start < obs_start < flare_end <= obs_end:
                total_time.append((flare_end - obs_start)/3600)
            elif obs_start <= flare_start < obs_end < flare_end:
                total_time.append((obs_end - flare_start)/3600)
            elif flare_start < obs_start < obs_end < flare_end:
                total_time.append((obs_end - obs_start)/3600)
print(sum(total_time))