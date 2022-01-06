#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 03:17:24 2021

@author: yuichiro
"""

import numpy as np
m = 9.10938356e-31
keV = 1.602176565e-16
c = 299792458
#[m/s]

num = 12
v2 = 2*num * keV / m
v = np.sqrt(v2)/c
print (v)