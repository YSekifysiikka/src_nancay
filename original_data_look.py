#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:32:42 2019

@author: yuichiro
"""

import glob
path='/Volumes/HDPH-UT/lab/solar_burst/Nancay/data/*/*/*'
File=glob.glob(path, recursive=True)
#print(File)
File1=len(File)
print (File1)


i=open('/Volumes/HDPH-UT/lab/solar_burst/Nancay/date/out2.txt', 'w')
for cstr in File:
    a=cstr.split('/')
    line=a[9]+'\n'
    i.write(line)
    
    
i.close()