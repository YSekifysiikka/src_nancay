#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 12:28:18 2020

@author: yuichiro
"""
from PIL import Image
import numpy as np
import pandas as pd
import datetime as dt
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import glob
import os
import sys
import shutil
import csv
from pynverse import inversefunc
import scipy


labelsize = 18
fontsize = 20
color_list = ['#ff7f00', '#377eb8']
color_list_1 = ['r', 'b']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/solar_burst/Nancay/analysis_data/sgepss_1.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

# for i in range(len(csv_input_final)):
#     residual = csv_input_final["residual"]
#     residual_list_1 = np.array([float(s) for s in residual])
#     sigma = np.sqrt(min(residual_list_1))

#     velocity = csv_input_final[["velocity"][0]][i].lstrip("['")[:-1].split(',')
#     velocity_list = [float(s) for s in velocity]
#     time_rate_1 = velocity_list[0]

#     date_event = csv_input_final['event_date'][i]
#     date_event_hour = csv_input_final['event_hour'][i]
#     date_event_minute = csv_input_final['event_minite'][i]
#     event_start = csv_input_final['event_start'][i]
#     event_end = csv_input_final['event_end'][i]
#     freq_start = csv_input_final['freq_start'][i]
#     freq_end = csv_input_final['freq_end'][i]
#     best_factor = csv_input_final[["factor"][0]][i]
    

year_list_1 = [20120101, 20170101]
year_list_2 = [20141231, 20191231]

year_num = []
for i in range(len(year_list_1)):
    year_num.append(np.count_nonzero((csv_input_final['event_date'] >= year_list_1[i]) & (csv_input_final['event_date'] <= year_list_2[i])))
print(year_num)

font_dict = dict(style="italic",size=16)

x_range = []
y_range = []
mean_val = []
std_val = []
for i in range(len(year_list_1)):
    velocity_fac_1 = []
    for j in range(len(csv_input_final)):
        if csv_input_final['event_date'][j] >= year_list_1[i] and csv_input_final['event_date'][j] <= year_list_2[i]:
            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
            velocity_list = [float(s) for s in velocity]
            if i == 0:
                velocity_fac_1.append(velocity_list[1])
            else:
                velocity_fac_1.append(velocity_list[1])
        else:
            pass
    plt.close()
    x_range.append(plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)[1])
    y_range.append(plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)[0]/len(velocity_fac_1))
    mean_val.append(round(np.mean(velocity_fac_1), 3))
    std_val.append(round(np.std(velocity_fac_1), 4))
    plt.close()
    # plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)


label = []
for i in range(x_range[0].shape[0] - 1):
    label.append(str('{:.02f}'.format(round(x_range[0][i], 3))) + '-' + str('{:.02f}'.format(round(x_range[0][i+1], 3))))


width = 0.022
x_range = np.array(x_range) - (width/2)
for i in range(len(y_range)):
    plt.bar(x_range[i][:20] + i * width, y_range[i], width= width, label = str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' ARV: ' + str('{:.03f}'.format(mean_val[i])) + 'c' + ' SD: ' + str('{:.04f}'.format(std_val[i])) + 'c', color = color_list[i])
    # plt.title(str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' velocity')
# plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
    plt.xlabel('Velocity[c]',fontsize=15)
    plt.ylabel('Occurrence rate',fontsize=15)
    plt.xticks([0,0.2,0.4,0.6, 0.8], ['0.00 - 0.05','0.20 - 0.25','0.40 - 0.45', '0.60 - 0.65', '0.80 - 0.85']) 
    plt.xticks(rotation = 20)
    plt.show()




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


separate_num = 3
for i in range(len(csv_input_final)):
    for j in range(separate_num):
        velocity = csv_input_final[["velocity"][0]][i].lstrip("['")[:-1].split(',')
        best_factor = 2
        # best_factor = csv_input_final[["factor"][0]][i]
        # if (csv_input_final[["freq_start"][0]][i] - csv_input_final[["freq_end"][0]][i])*((j + 1)/(separate_num+1)) + csv_input_final[["freq_end"][0]][i] >= 50:
        if float(velocity[best_factor-1]) < 0.50:
            freq.append((csv_input_final[["freq_start"][0]][i] - csv_input_final[["freq_end"][0]][i])*((j + 1)/(separate_num+1)) + csv_input_final[["freq_end"][0]][i])
            start_check.append(csv_input_final[["freq_start"][0]][i])
            y_factor.append(best_factor)
            y_velocity.append(float(velocity[best_factor-1]))
            time_gap.append(csv_input_final[["event_end"][0]][i] - csv_input_final[["event_start"][0]][i] + 1)

t = np.arange(0, 2000, 1)
t = (t+1)/100

for i in range (len(y_factor)):
    plt.close()
    factor = y_factor[i]
    velocity = y_velocity[i]
    freq_max = start_check[i]
    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/69600000000 + 1
    # print (h_start)
    y = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5))))
    cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
    invcube_3 = inversefunc(cube_3, y_values=freq[i])
    
    slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
    y_slope = slope * (t - invcube_3) + freq[i]
    freq_drift.append(-slope)
    start_h.append(h_start)

parameter_initial = np.array([0.067, 1.23])
# function to fit
def func(f, a, b):
    return a * (f ** b)


# xdata_new = []
# ydata_new = []
for i in range(len(year_num)):
    xdata = np.array(freq[separate_num * sum(year_num[:i]):separate_num * sum(year_num[:i + 1])])
    ydata = np.array(freq_drift[separate_num * sum(year_num[:i]):separate_num * sum(year_num[:i + 1])])
    # xdata = np.array(freq)
    # ydata = np.array(freq_drift)


    print (year_num[i])
    print (separate_num * year_num[i])
    paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata, ydata, p0=parameter_initial)
    error = np.sqrt(np.diag(covariance))
    #print ("paramater =", paramater_optimal)
    final_xdata = np.arange(min(xdata) - 8, max(xdata) + 15, 0.05)
    final_xdata = np.arange(30, 80, 0.05)
    y = func(final_xdata,paramater_optimal[0],paramater_optimal[1])
    y_up = func(final_xdata,paramater_optimal[0] + 2 * error[0],paramater_optimal[1] + 2 * error[1])
    y_down = func(final_xdata,paramater_optimal[0] - 2 * error[0],paramater_optimal[1] - 2 * error[1])
    print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
    # plt.plot(xdata, ydata, '.', color = '#ff7f00', markersize=2)
    plt.plot(xdata, ydata, '.', markersize=2, color = color_list[i])
    plt.plot(final_xdata, y, '-', color = color_list_1[i], label = str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' (This work)', linewidth = 3.0)
    plt.plot(final_xdata, y_up, '--', color = color_list_1[i], linewidth = 1.0)
    plt.plot(final_xdata, y_down, '--', color = color_list_1[i], linewidth = 1.0)

y = func(final_xdata,0.0672,1.23)
plt.plot(final_xdata, y, '-', color = 'k', label = 'P. J. Zhang et al., 2018', linewidth = 3.0)

y = func(final_xdata,0.0068,1.82)
plt.plot(final_xdata, y, '-', color = 'g', label = 'Morosan et al., 2015', linewidth = 3.0)

# y = func(final_xdata,0.073,1.25)
# plt.plot(final_xdata, y, '--', color = 'k', linewidth = 1.0)

# y = func(final_xdata,0.061,1.21)
# plt.plot(final_xdata, y, '--', color = 'k', linewidth = 1.0)

plt.ylim(0,60)
plt.xlim(30,80)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1)
plt.tick_params(labelsize=15)
plt.xlabel('Frequency [MHz]',fontsize=15)
plt.ylabel('Frequency drift rate[MHz/s]',fontsize=15)
plt.show()
plt.close()



