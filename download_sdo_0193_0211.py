#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 14:37:53 2021

@author: yuichiro
"""


from bs4 import BeautifulSoup
import urllib.request as req
import time
import numpy as np
import os
import urllib.request
import pandas as pd

def getNearestValue(list, num):

    # 昇順に挿入する際のインデックスを取得
    sortIdx = np.searchsorted(list, num, side='left')
    return list[sortIdx]

def download_sdo_0193_0211(date):
    str_date=str(date)
    yyyy=str_date[0:4]
    mm=str_date[4:6]
    dd=str_date[6:8]
    url = 'https://sdo.gsfc.nasa.gov/assets/img/browse/' + yyyy + '/' + mm + '/' + dd + '/'
    res = req.urlopen(url)
    soup = BeautifulSoup(res, "html.parser")
    HMIB_pic_list = []
    url_list = soup.find_all("a")
    for url_ in url_list:
        href=  url_.attrs['href']
        try:
            pic_type = href.split('_')[-2] + '_' + href.split('_')[-1]
            if pic_type == '512_0193.jpg':
                url2 = href
                directry='/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_0193/'+yyyy+'/'+mm
                if not os.path.isdir(directry):
                    os.makedirs(directry)
                title= directry+'/'+url2
                if not os.path.isfile(title):
                    urllib.request.urlretrieve(url+url2,"{0}".format(title))
                
            if pic_type == '512_0211.jpg':
                url2 = href
                directry='/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_0211/'+yyyy+'/'+mm
                if not os.path.isdir(directry):
                    os.makedirs(directry)
                title= directry+'/'+url2
                if not os.path.isfile(title):
                    urllib.request.urlretrieve(url+url2,"{0}".format(title))
        except:
            pass    

    time.sleep(2)
    return



# /html/body/div[3]/div/div[2]/pre/a[8]
# <a href="20210411_000000_1024_HMIIC.jpg">20210411_000000_1024_HMIIC.jpg</a>
# /html/body/div[3]/div/div[2]/pre/a[8]
# body > div.container > div > div.col-md-9 > pre > a:nth-child(9)
# body > div.container > div > div.col-md-9 > pre > a:nth-child(10)


date_in=[20070101,20091231]


if __name__=='__main__':
    start_day,end_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    
    DATE=sdate
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        try:
            download_sdo_0193_0211(date)
        except:
            print('DL error: ',date)
        DATE+=pd.to_timedelta(1,unit='day')