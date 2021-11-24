#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:16:56 2021

@author: yuichiro
"""


import pandas as pd
import glob
import csv
import sys
import csv
import pandas as pd
import sys
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from dateutil.relativedelta import relativedelta
import glob
import shutil
import datetime
import os
from dateutil import relativedelta
def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
flare_csv = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


# /Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/ordinary/2012/20120102_132446_133126_19720_20120_67_75_48.5_35.55peak.png


x_obs_date = []
m_obs_date = []


for j in range (len(flare_csv['peak'])):
    # j = z + flare_csv.index[0]
    yyyy = flare_csv['peak'][j].split('/')[0]
    mm = flare_csv['peak'][j].split('/')[1]
    dd = flare_csv['peak'][j].split('/')[2].split(' ')[0]
    str_date = yyyy + mm + dd
    HH = flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[0]
    MM = flare_csv['peak'][j].split('/')[2].split(' ')[1].split(':')[1]
    pd_peak_time = datetime.datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM))
    # pd_start_time = pd.to_datetime(flare_csv['start'][j].split('/')[0] + flare_csv['start'][j].split('/')[1] + flare_csv['start'][j].split('/')[2].split(' ')[0] + flare_csv['start'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['start'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
    # pd_end_time = pd.to_datetime(flare_csv['end'][j].split('/')[0] + flare_csv['end'][j].split('/')[1] + flare_csv['end'][j].split('/')[2].split(' ')[0] + flare_csv['end'][j].split('/')[2].split(' ')[1].split(':')[0] + flare_csv['end'][j].split('/')[2].split(' ')[1].split(':')[1],format='%Y%m%d%H%M')
    # ar_location = flare_csv['AR location'][j]
    flare_class = flare_csv['X-ray class'][j]
    if flare_class[0]=='M':
        m_obs_date.append(pd_peak_time)
    elif flare_class[0]=='X':
        x_obs_date.append(pd_peak_time)
        print (pd_peak_time)
    else:
        pass

x_obs_date = np.array(x_obs_date)
m_obs_date = np.array(m_obs_date)

date_in=[20070101,20201231]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')

m_class_num = []
x_class_num = []
year_list = []

DATE=sdate
while DATE <= edate:
    date=DATE.strftime(format='%Y%m%d')
    print(date)
    selected_x_obs = np.where((datetime.datetime(DATE.year, DATE.month, DATE.day, DATE.hour)<= x_obs_date) & (datetime.datetime(DATE.year + 1, 1, 1, DATE.minute)>= x_obs_date))[0]
    selected_m_obs = np.where((datetime.datetime(DATE.year, DATE.month, DATE.day, DATE.hour)<= m_obs_date) & (datetime.datetime(DATE.year + 1, 1, 1, DATE.minute)>= m_obs_date))[0]
    m_class_num.append(len(selected_m_obs))
    x_class_num.append(len(selected_x_obs))
    year_list.append(DATE.year)

    DATE+=relativedelta.relativedelta(years=1)


figure_=plt.figure(1,figsize=(6,6))
plt.bar(year_list, x_class_num, align="center", color = 'brown')
plt.ylabel('X class flare', fontsize = 20)
plt.show()
plt.close()
figure_=plt.figure(1,figsize=(6,6))
plt.bar(year_list, m_class_num, align="center")
plt.ylabel('M class flare', fontsize = 20)
plt.tick_params(labelsize=12)
plt.show()
plt.close()

