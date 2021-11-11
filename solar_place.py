#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 13:50:14 2020

@author: yuichiro
"""
import datetime
import astropy.time
import astropy.units as u
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

cos_list = []
time_list = []
for i in range(10):
    koko = EarthLocation(lat='47 22 24.00',lon='2 11 50.00', height = '150')
    obs_time = datetime.datetime(2013,1,16,7,0,0) + datetime.timedelta(hours = i)
    toki = astropy.time.Time(obs_time)
    taiyou = get_sun(toki).transform_to(AltAz(obstime=toki,location=koko))
    # print(taiyou)
    # print(taiyou.az) # 天球での方位角
    print(taiyou.alt) # 天球での仰俯角
    # print(taiyou.distance) # 距離
    # print(taiyou.distance.au) # au単位での距離
    
    azimuth = float(str(taiyou.az).split('d')[0] + '.' + str(taiyou.az).split('d')[1].split('m')[0])
    altitude = float(str(taiyou.alt).split('d')[0] + '.' + str(taiyou.alt).split('d')[1].split('m')[0])
    
    solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
                            math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
                            math.sin(math.radians(altitude))])
    machine_place = np.array([-math.cos(math.radians(0)),
                              0,
                              math.sin(math.radians(0))])
    cos = np.dot(solar_place, machine_place)
    cos_list.append(cos)
    time_list.append(obs_time)
    print(cos)
    
    
fig=plt.figure(1,figsize=(8,4))
ax1 = fig.add_subplot(311) 
ax1.plot(time_list, cos_list)
ax1.xaxis_date()
date_format = mdates.DateFormatter('%H:%M')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_ylabel('Decibel [dB]',fontsize=10)
plt.show()
