#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:00:47 2021

@author: yuichiro
"""


import urllib.request
import os
import pandas as pd
import csv

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'


def download_sdo_mv(date):
    str_date=str(date)
    yyyy=str_date[0:4]
    mm=str_date[4:6]
    dd=str_date[6:8]
    
    url1='https://sdo.gsfc.nasa.gov/assets/img/dailymov/'+yyyy+ '/'+mm+'/'+dd+'/'
    
    url2=str_date+'_1024_HMIB.mp4'
    
    directry='/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo/' + yyyy + '/' + mm + '/'
    if not os.path.isdir(directry):
        os.makedirs(directry)
    title=directry+url2

    if not os.path.isfile(directry + url2) == True:
        urllib.request.urlretrieve(url1+url2,"{0}".format(title))
    return
    
##############################################
# def re_download_sdo_mv(csv_file):
#     file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
#     csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
#     for j in range(len(csv_input_final)):
#         if csv_input_final['event_date'][j] 
#     return
    


# date_in=[20170101,20181231]
date_in=[20190101,20201231]


if __name__=='__main__':
    with open(Parent_directory+ '/solar_pic/sdo_download_list/' +str(date_in[0]) + '_' + str(date_in[1]) + '.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=["event_date"])
        w.writeheader()
        start_day,end_day=date_in
        sdate=pd.to_datetime(start_day,format='%Y%m%d')
        edate=pd.to_datetime(end_day,format='%Y%m%d')
        
        DATE=sdate
        while DATE <= edate:
            date=DATE.strftime(format='%Y%m%d')
            print(date)
            try:
                download_sdo_mv(date)
            except:
                print('DL error: ',date)
                w.writerow({'event_date':date})
            DATE+=pd.to_timedelta(1,unit='day')


