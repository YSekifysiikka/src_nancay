#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 17:17:01 2019

@author: yuichiro
"""

import sys
sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
import cdflib
import scipy
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
from scipy import signal
import csv
import pandas as pd
import matplotlib.dates as mdates
import datetime as dt
import datetime
from astropy.time import TimeDelta, Time
from astropy import units as u
import os
import math
import glob

def groupSequence(lst): 
    res = [[lst[0]]] 
  
    for i in range(1, len(lst)): 
        if lst[i-1]+1 == lst[i]: 
            res[-1].append(lst[i]) 
        else: 
            res.append([lst[i]]) 
    return res



            
###############
data_want = str(20120411)

path='/Volumes/HDPH-UT/lab/solar_burst/Nancay/data/*/*/*'+data_want+'*.cdf'
File = glob.glob(path, recursive=True)
for cstr in File:
    a = cstr.split('/')
    file_name = a[9]
###############
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

time_band = 240
time_co = 60


file = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
cdf_file = cdflib.CDF(file)
epoch = cdf_file['Epoch'] 
LL = cdf_file['LL'] 
RR = cdf_file['RR'] 

data_r_0 = RR
data_l_0 = LL
diff_r_last =(data_r_0).T
diff_r_last = np.flipud(diff_r_last)
diff_l_last =(data_l_0).T
diff_l_last = np.flipud(diff_l_last)
r_f = []
for change in range(diff_r_last.shape[1]):
    r_f.append(scipy.signal.medfilt(diff_r_last[:,change], kernel_size = 5))  
diff_r_last_1 = np.array(r_f).T

l_f = []
for change in range(diff_l_last.shape[1]):
    l_f.append(scipy.signal.medfilt(diff_l_last[:,change], kernel_size = 5))  
diff_l_last_1 = np.array(l_f).T

#diff_power = []
#for i in range(diff_l_last_1.shape[0])
#    r = diff_r_last_1[i]
#    l = diff_l_last_1[i]
#    for k in range(diff_l_last_1.shape[1])
#        r_l = (((10 ** ((r[k])/10)) + (10 ** ((l[k])/10)))/2)
#        diff_power.append(r_l)
#    print (sigma)
#    Time_result.append(np.where(x > sigma)[0])

####################
y_r = []
num = int(3.0)
b = np.ones(num)/num
for i in range (diff_r_last_1.shape[0]):
    y_r.append(np.convolve(diff_r_last_1[i], b, mode='valid'))
#y_noise = np.convolve(diff_l_last_1[0], b, mode='same')
####################
y_l = []
num = int(3.0)
b = np.ones(num)/num
for i in range (diff_l_last_1.shape[0]):
    y_l.append(np.convolve(diff_l_last_1[i], b, mode='valid'))

min_r = np.amin(y_r, axis=1)
diff_r_min = (diff_r_last_1.T - min_r).T
min_l = np.amin(y_l, axis=1)
diff_l_min = (diff_l_last_1.T - min_l).T


TYPE3 = []
TYPE3_group = []
#for t in range(0, 1, 1):
for t in range (math.floor((data_l_0.shape[0]-time_co)/time_band)):
#for t in range (15, 16, 1):
#for t in range (math.floor(data_l_0.shape[0]/time_band/2) - 18, (math.floor(data_l_0.shape[0]/time_band/2)) + 18, 1):
#for t in range (36):
#for t in range (-1, -36, -1):
#for t in range (-1, -5, -1):
    time = time_band*t
    print (time)
    start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time]/1000000000)
    time = time_band*(t+1) + time_co
    end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time]/1000000000)
    Time_start = start.strftime('%H:%M:%S')
    Time_end = end.strftime('%H:%M:%S')
    print (start)
    print(Time_start+'-'+Time_end)

    start = start.timestamp()
    end = end.timestamp()
    diff_r = diff_r_min[:, time - time_band - time_co:time]
    diff_l = diff_l_min[:, time - time_band - time_co:time]

    x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
    x_lims = mdates.date2num(x_lims)

    # Set some generic y-limits.
    y_lims = [10, 80]
    
    plt.close(1)
    figure_=plt.figure(1,figsize=(16,10))
    axes_2=figure_.add_subplot(111)
    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
    ax2 = axes_2.imshow(diff_r, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= 0,vmax = 60)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Right Handed Polarization',fontsize=15)
    plt.xlabel('Time (UT)',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.colorbar(ax2,label='from Background [dB]')
    
    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()
    #if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day):
    #    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day)
    #filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day+'/'+Time_start+'-'+Time_end+'.jpg'
    #plt.savefig(filename)
    plt.show()
    plt.close()

    figure_=plt.figure(1,figsize=(16,10))
    axes_2=figure_.add_subplot(111)
    figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
    ax2 = axes_2.imshow(diff_l, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
              aspect='auto',cmap='jet',vmin= 0,vmax = 60)
    axes_2.xaxis_date()
    date_format = mdates.DateFormatter('%H:%M:%S')
    axes_2.xaxis.set_major_formatter(date_format)
    plt.title('Left Handed Polarization',fontsize=15)
    plt.xlabel('Time (UT)',fontsize=20)
    plt.ylabel('Frequency [MHz]',fontsize=20)
    plt.colorbar(ax2,label='from Background [dB]')
    
    plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
    figure_.autofmt_xdate()
    #if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day):
    #    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day)
    #filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day+'/'+Time_start+'-'+Time_end+'.jpg'
    #plt.savefig(filename)
    plt.show()
    plt.close()
    quartile_r = []
    for i in range(0, 400, 1):
        quartile_r.append(np.percentile(diff_r[i], 25))
    quartile_l = []
    for i in range(0, 400, 1):
        quartile_l.append(np.percentile(diff_l[i], 25))
    quartile_1 = []
    for i in range (len(quartile_l)):
        quartile_1.append((quartile_l[i]+quartile_r[i])/2)

    x = np.flip(np.linspace(10, 79.825, 400))
    x1 = x
    for j in range(0, time_band + time_co - 2, 1):
#        print(j)
        y1_r = diff_r[0:400, j]
        y2_r = diff_r[0:400, j+1]
        y3_r = diff_r[0:400, j+2]
        y1_l = diff_l[0:400, j]
        y2_l = diff_l[0:400, j+1]
        y3_l = diff_l[0:400, j+2]
#        plt.scatter(x1, y1)
#        plt.scatter(x1, quartile_1)
        y1 = []
        for i in range(len(y1_r)):
            if (y1_r[i] > 0) & (y1_l[i]>0):
                y1.append(np.sqrt(y1_r[i]*y1_l[i]))
            else:
                y1.append(0)
        y2 = []
        for i in range(len(y2_r)):
            if (y2_r[i] > 0) & (y2_l[i]>0):
                y2.append(np.sqrt(y2_r[i]*y2_l[i]))
            else:
                y2.append(0)
        y3 = []
        for i in range(len(y3_r)):
            if (y3_r[i] > 0) & (y3_l[i]>0):
                y3.append(np.sqrt(y3_r[i]*y3_l[i]))
            else:
                y3.append(0)
#        plt.plot(x1, np.sqrt(y1_r*y1_l) + np.sqrt(y2_r*y2_l) - quartile_1)
        y = []
        for i in range(len(y1)):
            y.append(((y1[i] + y2[i] + y3[i])/3) - quartile_1[i])
#        y_sub = []
#        for i in range(len(y1)):
#            y_sub.append(-y1[i]+y2[i])
        y_over = []
        for i in range(len(y)):
            if y[i] > 10:
                y_over.append(i)
            else:
                pass
        
        if len(y_over) > 0:
            y_over_group = groupSequence(y_over)
        
            y_over_final = []
            for i in range(len(y_over_group)):
                if len(y_over_group[i]) > 20:
                    y_over_final.append(y_over_group[i])
        
#        print (y_over_final)
        
                
        plt.plot(x1, y)
        plt.ylim(0, 50)
        plt.show()
        plt.close()


#        plt.plot(x1, y_sub)
#        plt.ylim(0, 40)
#        plt.show()
#        plt.close()
        
#        if len(y_over_final) > 0:
#            TYPE3.append(time_band*t + j+1)
#TYPE3_final = np.unique(TYPE3)
#if len(TYPE3_final) > 0:
#    TYPE3_group.append(groupSequence(TYPE3_final))
#    
#if len(TYPE3_group)>0:
#    for t in range (len(TYPE3_group[0])):
#        if len(TYPE3_group[0][t]) > 2:
#        #for t in range (15, 16, 1):
#        #for t in range (math.floor(data_l_0.shape[0]/time_band/2) - 18, (math.floor(data_l_0.shape[0]/time_band/2)) + 18, 1):
#        #for t in range (36):
#        #for t in range (-1, -36, -1):
#        #for t in range (-1, -5, -1):
#            print (TYPE3_group[0][t][0])
#            print (TYPE3_group[0][t][-1])
#            time_start = TYPE3_group[0][t][0] - 10
#        #    print (time_start)
#            start = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time_start]/1000000000)
#            time_end = TYPE3_group[0][t][-1] + 10
#            end = dt.datetime(2000, 1, 1, 11, 58, 55, 816) + datetime.timedelta(seconds=epoch[time_end]/1000000000)
#            Time_start = start.strftime('%H:%M:%S')
#            Time_end = end.strftime('%H:%M:%S')
#            print (start)
#            print(Time_start+'-'+Time_end)
#        
#            start = start.timestamp()
#            end = end.timestamp()
#            diff_r_type3 = diff_r_min[:, time_start - 10:time_end + 10]
#            diff_l_type3 = diff_l_min[:, time_start - 10:time_end + 10]
#        
#            x_lims = list(map(dt.datetime.fromtimestamp, [start, end]))
#            x_lims = mdates.date2num(x_lims)
#        
#            # Set some generic y-limits.
#            y_lims = [10, 80]
#        
#            plt.close(1)
#            figure_=plt.figure(1,figsize=(16,10))
#            axes_2=figure_.add_subplot(111)
#            figure_.suptitle('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=20)
#            ax2 = axes_2.imshow(diff_r_type3, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
#                      aspect='auto',cmap='jet',vmin= 0,vmax = 60)
#            axes_2.xaxis_date()
#            date_format = mdates.DateFormatter('%H:%M:%S')
#            axes_2.xaxis.set_major_formatter(date_format)
#            plt.title('Right Handed Polarization',fontsize=15)
#            plt.xlabel('Time (UT)',fontsize=20)
#            plt.ylabel('Frequency [MHz]',fontsize=20)
#            plt.colorbar(ax2,label='from Background [dB]')
#        
#            plt.subplots_adjust(bottom=0.08,right=1,top=0.9,hspace=0.5)
#            figure_.autofmt_xdate()
#            #if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day):
#            #    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day)
#            #filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/'+year+'/'+month+'/'+year+month+day+'/'+Time_start+'-'+Time_end+'.jpg'
#            #plt.savefig(filename)
#            plt.show()
#            plt.close()
#        
#        
#        
