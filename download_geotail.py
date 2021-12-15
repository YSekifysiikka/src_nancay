#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 18:45:29 2021

@author: yuichiro
"""




import urllib.request
import pandas as pd
import os

def download_geotail_data(date):
    str_date=str(date)
    yyyy=str_date[0:4]
    mm=str_date[4:6]
    dd=str_date[6:8]
    url1='http://space.rish.kyoto-u.ac.jp/gtlpwi/24hour/'+yyyy[2:]+mm+dd+'/'
    url2=yyyy[2:]+mm+dd+'00.gif'

    directry='/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/geotail/plot_QL/' +yyyy
    if not os.path.isdir(directry):
        os.makedirs(directry)
    title=directry+'/'+url2
    if not os.path.isfile(directry+'/'+url2):
        urllib.request.urlretrieve(url1+url2,"{0}".format(title))


date_in=[20040903,20040912]


if __name__=='__main__':
    start_day,end_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    
    DATE=sdate
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        try:
            download_geotail_data(date)
        except:
            print('DL error: ',date)
        DATE+=pd.to_timedelta(1,unit='day')
    