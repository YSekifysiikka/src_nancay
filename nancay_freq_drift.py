#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:03:02 2020

@author: yuichiro
"""
import sys
sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
import numpy as np
import pandas as pd
import datetime as dt
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import glob
import os
from pynverse import inversefunc
import scipy




def numerical_diff_allen(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))

y_factor = []
y_velocity = []
freq_start = []
freq_end = []
freq_mean = []
freq_drift = []
freq = []
start_check = []
start_h = []
time_gap = []

#file = "velocity_factor_jpgu_test.csv"
file = "velocity_factor_jpgu_pparc1111.csv"
#file = "velocity_factor_jpgu_1.csv"
separate_num = 3
csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
for i in range(len(csv_input)):
    for j in range(separate_num):
        velocity = csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')
        best_factor = csv_input[["factor"][0]][i]
#        if i < 627:
#            best_factor = 10
#        else:
#            best_factor = 1
        freq.append((csv_input[["freq_start"][0]][i] - csv_input[["freq_end"][0]][i])*((j + 1)/(separate_num+1)) + csv_input[["freq_end"][0]][i])
        start_check.append(csv_input[["freq_start"][0]][i])
        y_factor.append(best_factor)
        y_velocity.append(float(velocity[best_factor-1]))
        time_gap.append(csv_input[["event_end"][0]][i] - csv_input[["event_start"][0]][i] + 1)

t = np.arange(0, 2000, 1)
t = (t+1)/100
for i in range (separate_num*len(csv_input)):
    plt.close()
    factor = y_factor[i]
    velocity = y_velocity[i]
    freq_max = start_check[i]
    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/69600000000 + 1
    y = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5))))
    cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
    invcube_3 = inversefunc(cube_3, y_values=freq[i])
    
    slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
    y_slope = slope * (t - invcube_3) + freq[i]
    freq_drift.append(-slope)
    start_h.append(h_start)
#    print (slope)
#    print (invcube_3)
#    #    print (freq_mean[i])
#    plt.ylim(30, 80)
#    plt.xlim(0, 30)
#    plt.plot(t, y, '.')
#    plt.plot(t, y_slope)
#    plt.plot()
#    plt.show()
#    plt.close()
    

xdata_2013 = np.array(freq[0:separate_num*627])
ydata_2013 = np.array(freq_drift[0:separate_num*627])
#xdata_2013 = np.array(freq_mean[0:3749])
#ydata_2013 = np.array(freq_drift[0:3749])
# initial guess for the parameters
parameter_initial = np.array([0.067, 1.23])
# function to fit
def func(f, a, b):
    return a * (f ** b)

paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2013, ydata_2013, p0=parameter_initial)

error = np.sqrt(np.diag(covariance))
#print ("paramater =", paramater_optimal)
final_xdata_2013 = np.arange(min(xdata_2013) - 8, max(xdata_2013) + 15, 0.05)
y_2013 = func(final_xdata_2013,paramater_optimal[0],paramater_optimal[1])
y_2013_up = func(final_xdata_2013,paramater_optimal[0] + error[0],paramater_optimal[1] + error[1])
y_2013_down = func(final_xdata_2013,paramater_optimal[0] - error[0],paramater_optimal[1] - error[1])
print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
plt.plot(xdata_2013, ydata_2013, '.', color = '#ff7f00', markersize=2)


#plt.plot(final_xdata_2013, y, '-', color = '#ff7f00')
#plt.ylim(10, 80)    


xdata_2017 = np.array(freq[separate_num*627:separate_num*782])
ydata_2017 = np.array(freq_drift[separate_num*627:separate_num*782])
#xdata_2017 = np.array(freq_mean[3749:5383])
#ydata_2017 = np.array(freq_drift[3749:5383])
# initial guess for the parameters
parameter_initial = np.array([0.067, 1.23])
# function to fit
paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2017, ydata_2017, p0=parameter_initial)
error = np.sqrt(np.diag(covariance))
#print ("paramater =", paramater_optimal)
final_xdata_2017 = np.arange(min(xdata_2017) - 8, max(xdata_2017) + 15, 0.05)
y_2017 = func(final_xdata_2017,paramater_optimal[0],paramater_optimal[1])
y_2017_up = func(final_xdata_2017,paramater_optimal[0] + error[0],paramater_optimal[1] + error[1])
y_2017_down = func(final_xdata_2017,paramater_optimal[0] - error[0],paramater_optimal[1] - error[1])
print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
plt.plot(xdata_2017, ydata_2017, '.', color = '#377eb8', markersize=2)

# xdata_low_12 = []
# ydata_low_12 = []
# xdata_high_12 = []
# ydata_high_12 = []
# for i in range(separate_num*782):
#     if time_gap[i] <= 12:
#         xdata_low_12.append(freq[i])
#         ydata_low_12.append(freq_drift[i])
#     else:
#         xdata_high_12.append(freq[i])
#         ydata_high_12.append(freq_drift[i])

# xdata_low_12 = np.array(xdata_low_12)
# ydata_low_12 = np.array(ydata_low_12)
# parameter_initial = np.array([0.067, 1.23])
# # function to fit
# paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_low_12, ydata_low_12, p0=parameter_initial)
# #print ("paramater =", paramater_optimal)
# final_xdata_low_12 = np.arange(min(xdata_2017) - 8, max(xdata_2017) + 15, 0.05)
# y_low_12 = func(final_xdata_low_12,paramater_optimal[0],paramater_optimal[1])
# print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
# plt.plot(final_xdata_low_12, y_low_12, '-', color = 'b', label = 'low_12', linewidth = 3.0)

# xdata_high_12 = np.array(xdata_high_12)
# ydata_high_12 = np.array(ydata_high_12)
# parameter_initial = np.array([0.067, 1.23])
# # function to fit
# paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_high_12, ydata_high_12, p0=parameter_initial)
# #print ("paramater =", paramater_optimal)
# final_xdata_high_12 = np.arange(min(xdata_2017) - 8, max(xdata_2017) + 15, 0.05)
# y_high_12 = func(final_xdata_high_12,paramater_optimal[0],paramater_optimal[1])
# print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
# plt.plot(final_xdata_high_12, y_high_12, '-', color = 'r', label = 'high_12', linewidth = 3.0)



plt.plot(final_xdata_2013, y_2013, '-', color = 'r', label = '2013', linewidth = 3.0)
# plt.plot(final_xdata_2013, y_2013_up, '--', color = 'r', linewidth = 1.0)
# plt.plot(final_xdata_2013, y_2013_down, '--', color = 'r', linewidth = 1.0)


plt.plot(final_xdata_2017, y_2017, '-', color = 'b', label = '2017', linewidth = 3.0)
# plt.plot(final_xdata_2017, y_2017_up, '--', color = 'b', linewidth = 3.0)
# plt.plot(final_xdata_2017, y_2017_down, '--', color = 'b', linewidth = 3.0)


y = func(final_xdata_2013,0.0672,1.23)
plt.plot(final_xdata_2013, y, '-', color = 'k', label = 'P. J. Zhang et al., 2018', linewidth = 3.0)

y = func(final_xdata_2013,0.073,1.25)
plt.plot(final_xdata_2013, y, '--', color = 'k', linewidth = 1.0)

y = func(final_xdata_2013,0.061,1.21)
plt.plot(final_xdata_2013, y, '--', color = 'k', linewidth = 1.0)

y = func(final_xdata_2013,0.0068,1.82)
plt.plot(final_xdata_2013, y, '-', color = 'g', label = 'Morosan et al., 2015', linewidth = 3.0)
#y = func(final_xdata_2013,0.01,1.84)
#plt.plot(final_xdata_2013, y, '-', color = 'y', label = 'Alvarez&Haddock(1973)', linewidth = 3.0)
plt.ylim(2, 20)
# plt.xlim(30, 80)
plt.xlim(31, 63)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#plt.title('Frequency drift',fontsize=20)

#plt.tick_params(axis='x', rotation=300, labelsize=labelsize)
plt.tick_params(labelsize=15)
plt.xlabel('Frequency [MHz]',fontsize=15)
plt.ylabel('Frequency drift rate[MHz/s]',fontsize=15)
#ax4.tick_params(axis='y', labelsize=labelsize)
plt.show()
plt.close()






#y_factor = []
#y_velocity = []
#freq_start = []
#freq_end = []
#freq_mean = []
#freq_drift = []
#freq = []
##file = "velocity_factor_jpgu_test.csv"
#file = "velocity_factor1.csv"
#separate_num = 3
#csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
#for i in range(len(csv_input)):
#    for j in range(separate_num):
#        velocity = csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')
#        best_factor = csv_input[["factor"][0]][i]
#        freq.append((csv_input[["freq_start"][0]][i] - csv_input[["freq_end"][0]][i])*((j + 1)/(separate_num+1)) + csv_input[["freq_end"][0]][i])
#        y_factor.append(best_factor)
#        y_velocity.append(float(velocity[best_factor-1]))
#
#t = np.arange(0, 2000, 1)
#t = (t+1)/100
#for i in range (separate_num*len(csv_input)):
#    plt.close()
#    factor = y_factor[i]
#    velocity = y_velocity[i]
#    allen_model = 10 * np.sqrt(factor * (2.99*(1+(t/696000)*(velocity * 300000))**(-16)+1.55*(1+(t/696000)*(velocity * 300000))**(-6)+0.036*(1+(t/696000)*(velocity * 300000))**(-1.5)))
#    y = 9 * allen_model
#    cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99*(1+(t/696000)*(velocity * 300000))**(-16)+1.55*(1+(t/696000)*(velocity * 300000))**(-6)+0.036*(1+(t/696000)*(velocity * 300000))**(-1.5))))
#    invcube_3 = inversefunc(cube_3, y_values = freq[i])
#    slope = numerical_diff_allen(factor, velocity, invcube_3)
#    y_slope = slope * (t - invcube_3) + freq[i]
#    freq_drift.append(-slope)
##    print (slope)
##    print (invcube_3)
##    print (freq_mean[i])
##    plt.ylim(30, 80)
##    plt.xlim(0, 30)
##    plt.plot(t, y, '.')
##    plt.plot(t, y_slope)
##    plt.plot()
##    plt.show()
##    plt.close()
#    
#
#xdata_2013 = np.array(freq[0:separate_num*627])
#ydata_2013 = np.array(freq_drift[0:separate_num*627])
##xdata_2013 = np.array(freq_mean[0:3749])
##ydata_2013 = np.array(freq_drift[0:3749])
## initial guess for the parameters
#parameter_initial = np.array([0.067, 1.23])
## function to fit
#def func(f, a, b):
#    return a * (f ** b)
#
#paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2013, ydata_2013, p0=parameter_initial)
##print ("paramater =", paramater_optimal)
#final_xdata_2013 = np.arange(min(xdata_2013) - 8, max(xdata_2013) + 15, 0.05)
#y_2013 = func(final_xdata_2013,paramater_optimal[0],paramater_optimal[1])
#print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
#plt.plot(xdata_2013, ydata_2013, '.', color = '#ff7f00', markersize=2)
##plt.plot(final_xdata_2013, y, '-', color = '#ff7f00')
##plt.ylim(10, 80)
#
#xdata_2017 = np.array(freq[separate_num*627:separate_num*782])
#ydata_2017 = np.array(freq_drift[separate_num*627:separate_num*782])
##xdata_2017 = np.array(freq_mean[3749:5383])
##ydata_2017 = np.array(freq_drift[3749:5383])
## initial guess for the parameters
#parameter_initial = np.array([0.067, 1.23])
## function to fit
#paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2017, ydata_2017, p0=parameter_initial)
##print ("paramater =", paramater_optimal)
#final_xdata_2017 = np.arange(min(xdata_2017) - 8, max(xdata_2017) + 15, 0.05)
#y_2017 = func(final_xdata_2017,paramater_optimal[0],paramater_optimal[1])
#print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
#plt.plot(xdata_2017, ydata_2017, '.', color = '#377eb8', markersize=2)
#plt.plot(final_xdata_2013, y_2013, '-', color = 'r', label = '2013', linewidth = 3.0)
#plt.plot(final_xdata_2017, y_2017, '-', color = 'b', label = '2017', linewidth = 3.0)
#
#y = func(final_xdata_2013,0.0672,1.23)
#plt.plot(final_xdata_2013, y, '-', color = 'k', label = 'P. J. Zhang et al., 2018', linewidth = 3.0)
#
#y = func(final_xdata_2013,0.073,1.25)
#plt.plot(final_xdata_2013, y, '--', color = 'k', linewidth = 1.0)
#
#y = func(final_xdata_2013,0.061,1.21)
#plt.plot(final_xdata_2013, y, '--', color = 'k', linewidth = 1.0)
#y = func(final_xdata_2013,0.01,1.84)
#plt.plot(final_xdata_2013, y, '-', color = 'y', label = 'Alvarez&Haddock', linewidth = 3.0)
#plt.ylim(2, 40)
#plt.xlim(30, 80)
#plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
##plt.title('Frequency drift',fontsize=20)
#
##plt.tick_params(axis='x', rotation=300, labelsize=labelsize)
#plt.tick_params(labelsize=15)
#plt.xlabel('Frequency [MHz]',fontsize=15)
#plt.ylabel('Frequency drift [MHz/s]',fontsize=15)
##ax4.tick_params(axis='y', labelsize=labelsize)
#plt.show()
#plt.close()
#
#
#
#
#
#
#import sys
#sys.path.append('/Users/yuichiro/.pyenv/versions/3.7.4/lib/python3.7/site-packages')
#import numpy as np
#import pandas as pd
#import datetime as dt
#import datetime
#import matplotlib.pyplot as plt
#from matplotlib.dates import DateFormatter
#import os
#from pynverse import inversefunc
#import scipy
#
#
#def numerical_diff_allen(factor, velocity, t):
#    h = 1e-4
#    f_1= 9 * 10 * np.sqrt(factor * (2.99*(1+((t + h)/696000)*(velocity * 300000))**(-16)+1.55*(1+((t+h)/696000)*(velocity * 300000))**(-6)+0.036*(1+((t+h)/696000)*(velocity * 300000))**(-1.5)))
#    f_2 = 9 * 10 * np.sqrt(factor * (2.99*(1+((t - h)/696000)*(velocity * 300000))**(-16)+1.55*(1+((t-h)/696000)*(velocity * 300000))**(-6)+0.036*(1+((t-h)/696000)*(velocity * 300000))**(-1.5)))
#    return ((f_1 - f_2)/(2*h))
#
#y_factor = []
#y_velocity = []
#freq_start = []
#freq_end = []
#freq_mean = []
#freq_drift = []
#freq = []
#file = "velocity_factor_jpgu_test.csv"
##file = "velocity_factor1.csv"
#csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
#for i in range(len(csv_input)):
#    for j in range(separate_num):
#        velocity = csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')
#        best_factor = csv_input[["factor"][0]][i]
#        freq.append((csv_input[["freq_start"][0]][i] - csv_input[["freq_end"][0]][i])*((j + 1)/(separate_num+1)) + csv_input[["freq_end"][0]][i])
#        y_factor.append(best_factor)
#        y_velocity.append(float(velocity[best_factor-1]))
#
#t = np.arange(0, 2000, 1)
#t = (t+1)/100
#for i in range (separate_num*len(csv_input)):
#    plt.close()
#    factor = y_factor[i]
#    velocity = y_velocity[i]
#    allen_model = 10 * np.sqrt(factor * (2.99*(1+(t/696000)*(velocity * 300000))**(-16)+1.55*(1+(t/696000)*(velocity * 300000))**(-6)+0.036*(1+(t/696000)*(velocity * 300000))**(-1.5)))
#    y = 9 * allen_model
#    cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99*(1+(t/696000)*(velocity * 300000))**(-16)+1.55*(1+(t/696000)*(velocity * 300000))**(-6)+0.036*(1+(t/696000)*(velocity * 300000))**(-1.5))))
#    invcube_3 = inversefunc(cube_3, y_values = freq[i])
#    slope = numerical_diff_allen(factor, velocity, invcube_3)
#    y_slope = slope * (t - invcube_3) + freq[i]
#    freq_drift.append(-slope)
#
#xdata_2013 = np.array(freq[0:separate_num*3749])
#ydata_2013 = np.array(freq_drift[0:separate_num*3749])
## initial guess for the parameters
#parameter_initial = np.array([0.067, 1.23])
## function to fit
#def func(f, a, b):
#    return a * (f ** b)
#
#paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2013, ydata_2013, p0=parameter_initial)
##print ("paramater =", paramater_optimal)
#final_xdata_2013 = np.arange(min(xdata_2013) - 8, max(xdata_2013) + 15, 0.05)
#y_2013 = func(final_xdata_2013,paramater_optimal[0],paramater_optimal[1])
#print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
#plt.plot(xdata_2013, ydata_2013, '.', color = '#ff7f00', markersize=2)
##plt.plot(final_xdata_2013, y, '-', color = '#ff7f00')
##plt.ylim(10, 80)
#
##xdata_2017 = np.array(freq_mean[627:782])
##ydata_2017 = np.array(freq_drift[627:782])
#xdata_2017 = np.array(freq[separate_num*3749:separate_num*5383])
#ydata_2017 = np.array(freq_drift[separate_num*3749:separate_num*5383])
## initial guess for the parameters
#parameter_initial = np.array([0.067, 1.23])
## function to fit
#paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2017, ydata_2017, p0=parameter_initial)
##print ("paramater =", paramater_optimal)
#final_xdata_2017 = np.arange(min(xdata_2017) - 8, max(xdata_2017) + 15, 0.05)
#y_2017 = func(final_xdata_2017,paramater_optimal[0],paramater_optimal[1])
#print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
#plt.plot(xdata_2017, ydata_2017, '.', color = '#377eb8', markersize=2)
#plt.plot(final_xdata_2013, y_2013, '-', color = 'r', label = '2013', linewidth = 3.0)
#plt.plot(final_xdata_2017, y_2017, '-', color = 'b', label = '2017', linewidth = 3.0)
#
#y = func(final_xdata_2013,0.0672,1.23)
#plt.plot(final_xdata_2013, y, '-', color = 'k', label = 'P. J. Zhang et al., 2018', linewidth = 3.0)
#
#y = func(final_xdata_2013,0.073,1.25)
#plt.plot(final_xdata_2013, y, '--', color = 'k', linewidth = 1.0)
#
#y = func(final_xdata_2013,0.061,1.21)
#plt.plot(final_xdata_2013, y, '--', color = 'k', linewidth = 1.0)
#y = func(final_xdata_2013,0.01,1.84)
#plt.plot(final_xdata_2013, y, '-', color = 'y', label = 'Alvarez&Haddock', linewidth = 3.0)
#
#y = func(final_xdata_2013,0.0068,1.82)
#plt.plot(final_xdata_2013, y, '-', color = 'g', label = 'Morosan et al., 2015', linewidth = 3.0)
##
##y = func(final_xdata_2013,0.007,1.76)
##plt.plot(final_xdata_2013, y, '-', color = 'k', label = 'Mann', linewidth = 3.0)
##y = func(final_xdata,0.0084,1.57)
##plt.plot(final_xdata, y, '-', color = 'k', label = 'Clarke et al 2019')
##y = func(final_xdata,0.0074,1.65)
##plt.plot(final_xdata, y, '-', color = 'k', label = 'PeiJin Zhang et al 2020')
#plt.ylim(2, 40)
#plt.xlim(30, 80)
#plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
##plt.title('Frequency drift',fontsize=20)
#
##plt.tick_params(axis='x', rotation=300, labelsize=labelsize)
#plt.tick_params(labelsize=15)
#plt.xlabel('Frequency [MHz]',fontsize=15)
#plt.ylabel('Frequency drift [MHz/s]',fontsize=15)
##ax4.tick_params(axis='y', labelsize=labelsize)