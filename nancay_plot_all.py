#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:24:46 2019

@author: yuichiro
"""
import sys
sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
import cdflib
#from spacepy import pycdf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import datetime
from astropy.time import TimeDelta, Time
from astropy import units as u

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1


file_name = 'srn_nda_routine_sun_edr_201304220750_201304221548_V13.cdf'
file_name_separate =file_name.split('_')
Date_start = file_name_separate[5]
date_OBs=str(Date_start)
year=date_OBs[0:4]
month=date_OBs[4:6]
day=date_OBs[6:8]
start_h = date_OBs[8:10]
start_m = date_OBs[10:12]

Date_stop = file_name_separate[6]
end_h = Date_stop[8:10]
end_m = Date_stop[10:12]


file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
#cdf_file = pycdf.CDF(file)
cdf_file = cdflib.CDF(file)
epoch = cdf_file['Epoch'] 
LL = cdf_file['LL'] 
RR = cdf_file['RR'] 

data_r = RR
data_l = LL



#when you want to change time range
#start_time = 1152
#end_time = 1305
#start_h = str(start_time)[0:2]
#start_m = str(start_time)[2:4]
#end_h = str(end_time)[0:2]
#end_m = str(end_time)[2:4]

#start = dt.datetime.strptime(year+'/'+month+'/'+day+' '+start_h+':'+start_m+':00', '%Y/%m/%d %H:%M:%S')
#end = dt.datetime.strptime(year+'/'+month+'/'+day+' '+end_h+':'+end_m+':00', '%Y/%m/%d %H:%M:%S')


time = 0
while time < epoch.shape[0]-300:
    start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time]/1000000000)
    time += 300
    end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time]/1000000000)
    Time_start = start.strftime('%H%M%S')
    Time_end = end.strftime('%H%M%S')
    print(start.strftime('%H:%M:%S')+'-'+end.strftime('%H:%M:%S'))

    #(Time(2000, format='jyear') + TimeDelta(epoch[-1]/1000000000*u.s)).iso    
    
    start = start.timestamp()
    end = end.timestamp()
    #
    #
    data_r = RR[time-300:time]
    data_l = LL[time-300:time]
    mean_r=np.mean(data_r,axis=0)
    mean_l=np.mean(data_l,axis=0)
    diff_r=(data_r-mean_r).T
    diff_l=(data_l-mean_l).T
    diff_r = np.flipud(diff_r)
    diff_l = np.flipud(diff_l)
    x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
    x_lims = mdates.date2num(x_lims)
    
    # Set some generic y-limits.
    y_lims = [10, 80]
    
    plt.close(1)
    figure_=plt.figure(1,figsize=(16,10))
    figure_.suptitle('NANCAY DECAMETER ARRAY: '+year+
                     '-'+month+'-'+day,fontsize=30)
    
#    axes_1=figure_.add_subplot(211)
#    ax1 = axes_1.imshow(diff_r, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#              aspect='auto',cmap='jet',vmin=0,vmax=30)
#    axes_1.xaxis_date()
#    date_format = mdates.DateFormatter('%H:%M:%S')
#    axes_1.xaxis.set_major_formatter(date_format)
#    plt.title('Right Handed Polarization',fontsize=15)
#    plt.xlabel('Time (UT)',fontsize=20)
#    plt.ylabel('Frequency [MHz]',fontsize=20)
#    plt.colorbar(ax1,label='from Background [dB]')
    
#    axes_2=figure_.add_subplot(212)    
    axes_2=figure_.add_subplot(111)
    ax2 = axes_2.imshow(diff_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=0,vmax=30)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Left Handed Polarization',fontsize=15)
    plt.xlabel('Time (UT)',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.colorbar(ax2,label='from Background [dB]')
    
    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()
#    if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/notFFT/'+year+'/'+month+'/'+year+month+day):
#        os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/notFFT/'+year+'/'+month+'/'+year+month+day)
#    filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/notFFT/'+year+'/'+month+'/'+year+month+day+'/'+Time_start+'-'+Time_end+'.jpg'
#    plt.savefig(filename)
    plt.show()

    
    
#next plot    
    time -= 60

else:
    time = -301
    start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time]/1000000000)
    time += 300
    end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time]/1000000000)
    print(start.strftime('%H:%M:%S')+'-'+end.strftime('%H:%M:%S'))
    
    start = start.timestamp()
    end = end.timestamp()
    #
    #
    #
#    data_r = RR[time-300:time]
    data_l = LL[time-300:time]
#    mean_r=np.mean(data_r,axis=0)
    mean_l=np.mean(data_l,axis=0)
#    diff_r=(data_r-mean_r).T
    diff_l=(data_l-mean_l).T
#    diff_r = np.flipud(diff_r)
    diff_l = np.flipud(diff_l)
    
    
    
    x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
    x_lims = mdates.date2num(x_lims)
    
    # Set some generic y-limits.
    y_lims = [10, 80]
    
    plt.close(1)
    figure_=plt.figure(1,figsize=(16,10))
#    figure_.suptitle('NANCAY DECAMETER ARRAY: '+year+
#                     '-'+month+'-'+day,fontsize=30)
#    
#    axes_1=figure_.add_subplot(211)
#    ax1 = axes_1.imshow(diff_r, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#              aspect='auto',cmap='jet',vmin=0,vmax=30)
#    axes_1.xaxis_date()
#    date_format = mdates.DateFormatter('%H:%M:%S')
#    axes_1.xaxis.set_major_formatter(date_format)
#    plt.title('Right Handed Polarization',fontsize=15)
#    plt.xlabel('Time (UT)',fontsize=20)
#    plt.ylabel('Frequency [MHz]',fontsize=20)
#    plt.colorbar(ax1,label='from Background [dB]')
    
    
#    axes_2=figure_.add_subplot(212)
    axes_2=figure_.add_subplot(111)
    ax2 = axes_2.imshow(diff_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin=0,vmax=30)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Left Handed Polarization',fontsize=15)
    plt.xlabel('Time (UT)',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.colorbar(ax2,label='from Background [dB]')
    
    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()
#    if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/notFFT/'+year+'/'+month+'/'+year+month+day):
#        os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/notFFT/'+year+'/'+month+'/'+year+month+day)
#    filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/notFFT/'+year+'/'+month+'/'+year+month+day+'/'+Time_start+'-'+Time_end+'.jpg'
#    plt.savefig(filename)
    plt.show()
    
    
