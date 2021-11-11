#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 21:57:01 2021

@author: yuichiro
"""

import glob

obs_dates = []
files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/afjpgusimpleselect/*/*/*.png')
for file in files:
    obs_time = int(file.split('/')[-1].split('_')[0])
    if ((obs_time >= 20120101) & (obs_time <= 20171231)):
        obs_dates.append(obs_time)

import collections
c = collections.Counter(obs_dates)
print(c.most_common(5))