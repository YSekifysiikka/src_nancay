##!/usr/bin/env python3
## -*- coding: utf-8 -*-
#"""
#Created on Wed Jan  8 23:07:20 2020
#
#@author: yuichiro
#"""
#
#import sympy
#import math
import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc



def numerical_diff_df_dn(ne):
    h = 1e-5
    f_1 = 9*np.sqrt(ne+h)/1e+3
    f_2 = 9*np.sqrt(ne-h)/1e+3
    return ((f_1 - f_2)/(2*h))

def numerical_diff_allen_dn_dr(factor, r):
    h = 1e-1
    ne_1 = factor * 10**8 * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))
    ne_2 = factor * 10**8 * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_newkirk_dn_dr(factor, r):
    h = 1e-1
    ne_1 = factor * 4.2 * 10 ** (4+4.32/((r+h)/69600000000))
    ne_2 = factor * 4.2 * 10 ** (4+4.32/((r-h)/69600000000))
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_wangmin_dn_dr(factor, r):
    h = 1e-2
    ne_1 = factor * (353766/((r+h)/69600000000) + 1.03359e+07/((r+h)/69600000000)**2 - 5.46541e+07/((r+h)/69600000000)**3 + 8.24791e+07/((r+h)/69600000000)**4)
    ne_2 = factor * (353766/((r-h)/69600000000) + 1.03359e+07/((r-h)/69600000000)**2 - 5.46541e+07/((r-h)/69600000000)**3 + 8.24791e+07/((r-h)/69600000000)**4)
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_wangmax_dn_dr(factor, r):
    h = 1e-2
    ne_1 = factor * (-4.42158e+06/((r+h)/69600000000) + 5.41656e+07/((r+h)/69600000000)**2 - 1.86150e+08 /((r+h)/69600000000)**3 + 2.13102e+08/((r+h)/69600000000)**4)
    ne_2 = factor * (-4.42158e+06/((r-h)/69600000000) + 5.41656e+07/((r-h)/69600000000)**2 - 1.86150e+08 /((r-h)/69600000000)**3 + 2.13102e+08/((r-h)/69600000000)**4)
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_gibson_dn_dr(factor, r):
    h = 1e-2
    ne_1 = factor * (3.60/((r+h)/69600000000) ** (15.3) + 0.990/((r+h)/69600000000) ** (7.34) + 0.365/((r+h)/69600000000) ** (4.31)) * 1e+8
    ne_2 = factor * (3.60/((r-h)/69600000000) ** (15.3) + 0.990/((r-h)/69600000000) ** (7.34) + 0.365/((r-h)/69600000000) ** (4.31)) * 1e+8
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_leblanc_dn_dr(factor, r):
    h = 1e-2
    ne_1 = factor * (3.3e+05/((r+h)/69600000000)**2 + 4.1e+06 /((r+h)/69600000000)**4 + 8.0e+07/((r+h)/69600000000)**6)
    ne_2 = factor * (3.3e+05/((r-h)/69600000000)**2 + 4.1e+06 /((r-h)/69600000000)**4 + 8.0e+07/((r-h)/69600000000)**6)   
    return ((ne_1 - ne_2)/(2*h))



#km
sun_to_earth = 150000000
sun_radius = 696000
light_v = 300000 #[km/s]
time_rate = 0.15
# time_rate2 = 0.42

print ('velocity= ' + str(time_rate) +  'c')

########
#allen fp
factor = 1
cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
print ('\n'+str(factor)+'×B-A model' + 'emission fp')
drift_rates = numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_allen_dn_dr(factor, r)))

########
#allen 2fp
factor = 1
cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
ne = factor * 10**8 * (2.99*(r_1)**(-16)+1.55*(r_1)**(-6)+0.036*(r_1)**(-1.5))
print ('\n'+str(factor)+'×B-A model' + 'emission 2fp')
drift_rates = 2 * numerical_diff_df_dn(ne) * numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 20/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\n2*df/dn: ' + str(2*numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_allen_dn_dr(factor, r)))


########
#newkirk fp
factor = 1
cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
ne = factor * 4.2 * 10 ** (4+4.32/r_1)
print ('\n'+str(factor)+'×newkirk model' + 'emission fp')
drift_rates = numerical_diff_df_dn(ne) * numerical_diff_newkirk_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_newkirk_dn_dr(factor, r)))

########
#newkirk 2fp
factor = 3
cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
ne = factor * 4.2 * 10 ** (4+4.32/r_1)
print ('\n'+str(factor)+'×newkirk model' + 'emission 2fp')
drift_rates = 2 * numerical_diff_df_dn(ne) * numerical_diff_newkirk_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\n2*df/dn: ' + str(2*numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_newkirk_dn_dr(factor, r)))



factor = 1
########
#wang min fp

cube_4 =  (lambda h: 9 * 1e-3 * np.sqrt(353766/(1+(h/696000)) + 1.03359e+07/(1+(h/696000))**2 - 5.46541e+07/(1+(h/696000))**3 + 8.24791e+07/(1+(h/696000))**4))
r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
ne = factor * (353766/(r_1) + 1.03359e+07/(r_1)**2 - 5.46541e+07/(r_1)**3 + 8.24791e+07/(r_1)**4)
print ('\nWang model (Solar minimum)' + ' emission fp')
drift_rates = numerical_diff_df_dn(ne) * numerical_diff_wangmin_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_wangmin_dn_dr(factor, r)))

########
#wang min 2fp

cube_4 =(lambda h: 9 * 1e-3 * np.sqrt(353766/(1+(h/696000)) + 1.03359e+07/(1+(h/696000))**2 - 5.46541e+07/(1+(h/696000))**3 + 8.24791e+07/(1+(h/696000))**4))
r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
ne = factor * (353766/(r_1) + 1.03359e+07/(r_1)**2 - 5.46541e+07/(r_1)**3 + 8.24791e+07/(r_1)**4)
print ('\nWang model (Solar minimum)' + ' emission 2fp')
drift_rates = 2 * numerical_diff_df_dn(ne) * numerical_diff_wangmin_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 20/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\n2*df/dn: ' + str(2*numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_wangmin_dn_dr(factor, r)))



########
#wang max fp

cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(-4.42158e+06/(1+(h/696000)) + 5.41656e+07/(1+(h/696000))**2 - 1.86150e+08 /(1+(h/696000))**3 + 2.13102e+08/(1+(h/696000))**4))
r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
ne = factor * (-4.42158e+06/(r_1) + 5.41656e+07/(r_1)**2 - 1.86150e+08 /(r_1)**3 + 2.13102e+08/(r_1)**4)
print ('\nWang model (Solar maximum)' + ' emission fp')
drift_rates = numerical_diff_df_dn(ne) * numerical_diff_wangmax_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_wangmax_dn_dr(factor, r)))

########
#wang max 2fp
cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(-4.42158e+06/(1+(h/696000)) + 5.41656e+07/(1+(h/696000))**2 - 1.86150e+08 /(1+(h/696000))**3 + 2.13102e+08/(1+(h/696000))**4))
r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
ne = factor * (-4.42158e+06/(r_1) + 5.41656e+07/(r_1)**2 - 1.86150e+08 /(r_1)**3 + 2.13102e+08/(r_1)**4)
print ('\nWang model (Solar maximum)' + ' emission 2fp')
drift_rates = 2 * numerical_diff_df_dn(ne) * numerical_diff_wangmax_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\n2*df/dn: ' + str(2*numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_wangmax_dn_dr(factor, r)))




factor = 1
cube_4 = (lambda h: 9 * 1e-3 * np.sqrt((3.60/(1+(h/696000)) ** (15.3) + 0.990/(1+(h/696000)) ** (7.34) + 0.365/(1+(h/696000)) ** (4.31)) * 1e+8))
r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
ne = factor * (3.60/(r_1) ** (15.3) + 0.990/(r_1) ** (7.34) + 0.365/(r_1) ** (4.31)) * 1e+8
print ('\nGibson model' + ' emission fp')
drift_rates = numerical_diff_df_dn(ne) * numerical_diff_gibson_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_gibson_dn_dr(factor, r)))

factor = 1
cube_4 = (lambda h: 9 * 1e-3 * np.sqrt((3.60/(1+(h/696000)) ** (15.3) + 0.990/(1+(h/696000)) ** (7.34) + 0.365/(1+(h/696000)) ** (4.31)) * 1e+8))
r = (inversefunc(cube_4, y_values = 20) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 20) + 696000)/696000
ne = factor * (3.60/(r_1) ** (15.3) + 0.990/(r_1) ** (7.34) + 0.365/(r_1) ** (4.31)) * 1e+8
print ('\nGibson model' + ' emission 2fp')
drift_rates = 2 * numerical_diff_df_dn(ne) * numerical_diff_gibson_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\n2*df/dn: ' + str(2 * numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_gibson_dn_dr(factor, r)))


factor = 1
cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(3.3e+05/(1+(h/696000))**2 + 4.1e+06 /(1+(h/696000))**4 + 8.0e+07/(1+(h/696000))**6))
r = (inversefunc(cube_4, y_values = 40) + 696000)*1e+5
r_1 = (inversefunc(cube_4, y_values = 40) + 696000)/696000
ne = factor * (3.3e+05/(r_1)**2 + 4.1e+06 /(r_1)**4 + 8.0e+07/(r_1)**6)
print ('\nLeblanc model' + ' emission fp')
drift_rates = numerical_diff_df_dn(ne) * numerical_diff_leblanc_dn_dr(factor, r) * time_rate * light_v * 1e+5
# drift_rates_1 = 40/2/ne* numerical_diff_allen_dn_dr(factor, r) * time_rate * light_v * 1e+5
print ('df/dt: ' + str(drift_rates*(-1)) + '\ndf/dn: ' + str(numerical_diff_df_dn(ne)) + '\ndn/dr: '+str(numerical_diff_leblanc_dn_dr(factor, r)))



