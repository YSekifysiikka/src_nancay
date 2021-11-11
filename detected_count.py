#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 15:19:29 2020

@author: yuichiro
"""

import glob
#select_year_for_check
check_year = [2013, 2018]
for i in range(len(check_year)):
    year = str(check_year[i])
#year = str(2018)
    path = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/drift_check/pra4/2/'+ year +'/*/*'
    File = glob.glob(path, recursive=True)
    #print(File)
    File1 = len(File)
    print (year + '-all:' + str(File1))

    path_0 ='/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/drift_check/pra4/2/'+ year +'/*/*'+'__sigma_l_r.png'
    File_0 = glob.glob(path_0, recursive=True)
    #print(File)
    File1_0 = len(File_0)
    print (year + '-sigma:' + str(File1_0))
    print (year + '-detected:' + str(File1-File1_0))