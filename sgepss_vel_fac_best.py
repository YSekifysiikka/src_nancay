#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:52:10 2020

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

def numerical_diff_allen_velocity(factor, r):
    h = 1e-2
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))
def numerical_diff_allen(factor, velocity, t, h_start):
    h = 1e-3
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))


labelsize = 18
fontsize = 20
factor_velocity = 2
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm', 'g']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/storm_burst_cycle24.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


# year_list_1 = [20010101, 20070101, 20120101, 20170101]
# year_list_2 = [20011231, 20091231, 20141231, 20191231]
year_list_1 = [20120101,20130101,20140101, 20170101,20190101, 20200101]
year_list_2 = [20121231,20131231,20141231, 20171231,20191231, 20201231]
# year_list_1 = [20120101,20130101,20190101]
# year_list_2 = [20121231,20131231,20201231]
# year_list_1 = [20010101, 20070101]
# year_list_2 = [20011231, 20091231]

# year_list_1 = [20170101, 20190101]
# year_list_2 = [20171231, 20191231]

year_num = []
for i in range(len(year_list_1)):
    year_num.append(np.count_nonzero((csv_input_final['event_date'] >= year_list_1[i]) & (csv_input_final['event_date']<= year_list_2[i])))
print(year_num)

font_dict = dict(style="italic",size=16)



# x_range = []
# y_range = []
# mean_val = []
# std_val = []
# for i in range(len(year_list_1)):
#     velocity_fac_1 = []
#     for j in range(len(csv_input_final)):
#         if csv_input_final['event_date'][j] >= year_list_1[i] and csv_input_final['event_date'][j] <= year_list_2[i]:
#             # print(csv_input_final['event_date'][j])
#             velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
#             velocity_list = [float(s) for s in velocity]
#             if i == 0:
#                 velocity_fac_1.append(velocity_list[1])
#             else:
#                 velocity_fac_1.append(velocity_list[1])
#         else:
#             pass
#     plt.close()
#     x_range.append(plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)[1])
#     y_range.append(plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)[0]/len(velocity_fac_1))
#     mean_val.append(round(np.mean(velocity_fac_1), 3))
#     std_val.append(round(np.std(velocity_fac_1), 4))
#     plt.close()
#     # plt.hist(velocity_fac_1, bins = 20, range = (0,1), density= None)


# label = []
# for i in range(x_range[0].shape[0] - 1):
#     label.append(str('{:.02f}'.format(round(x_range[0][i], 3))) + '-' + str('{:.02f}'.format(round(x_range[0][i+1], 3))))


# width = 0.022
# x_range = np.array(x_range) - (width/2)
# for i in range(len(y_range)):
#     plt.bar(x_range[i][:20] + i * width, y_range[i], width= width, label = str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' ARV: ' + str('{:.03f}'.format(mean_val[i])) + 'c' + ' SD: ' + str('{:.04f}'.format(std_val[i])) + 'c', color = color_list[i])
#     # plt.title(str(year_list_1[i])[:4] + '-' + str(year_list_2[i])[:4] + ' velocity')
# # plt.text(0.8, 0.1, "Armadillo", fontdict = font_dict)
#     plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#     plt.xlabel('Velocity[c]',fontsize=15)
#     plt.ylabel('Occurrence rate',fontsize=15)
#     plt.xticks([0,0.2,0.4,0.6, 0.8], ['0.00 - 0.05','0.20 - 0.25','0.40 - 0.45', '0.60 - 0.65', '0.80 - 0.85']) 
#     plt.xticks(rotation = 20)
# plt.show()
# plt.close()

x_range = []
y_range = []
mean_val = []
std_val = []

for x in range(len(year_list_1)):
    y_factor = []
    y_velocity = []
    freq_start = []
    freq_end = []
    freq_mean = []
    freq_drift = []
    freq = []
    start_check = []
    time_gap = []
    velocity_real = []
    separate_num = 1

    for j in range(len(csv_input_final)):
        if csv_input_final['event_date'][j] >= year_list_1[x] and csv_input_final['event_date'][j] <= year_list_2[x]:
            # print (csv_input_final['event_date'][j])
            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
            best_factor = csv_input_final["factor"][j]
            for z in range(separate_num):
                freq.append((csv_input_final[["freq_start"][0]][j] - csv_input_final[["freq_end"][0]][j])*((z + 1)/(separate_num+1)) + csv_input_final[["freq_end"][0]][j])
                start_check.append(csv_input_final[["freq_start"][0]][j])
                y_factor.append(best_factor)
                y_velocity.append(float(velocity[best_factor-1]))
                time_gap.append(csv_input_final[["event_end"][0]][j] - csv_input_final[["event_start"][0]][j] + 1)

    

    t = np.arange(0, 2000, 1)
    t = (t+1)/100
    
    for i in range (len(y_factor)):
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

        cube_5 = (lambda h: 9 * 10 * np.sqrt(factor_velocity * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
        invcube_5 = inversefunc(cube_5, y_values = freq[i])
        h_radio = invcube_5 + 69600000000
        velocity_real.append(2/numerical_diff_allen_velocity(factor_velocity, h_radio)/freq[i]*slope/29979245800)



# for i in range(len(year_list_1)):
#     velocity_fac_1 = []
#     for j in range(len(csv_input_final)):
#         if csv_input_final['event_date'][j] >= year_list_1[i] and csv_input_final['event_date'][j] <= year_list_2[i]:
#             # print(csv_input_final['event_date'][j])
#             velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
#             velocity_list = [float(s) for s in velocity]
#             if i == 0:
#                 velocity_fac_1.append(velocity_list[1])
#             else:
#                 velocity_fac_1.append(velocity_list[1])
#         else:
#             pass
    plt.close()
    x_range.append(plt.hist(velocity_real, bins = 20, range = (0,1), density= None)[1])
    y_range.append(plt.hist(velocity_real, bins = 20, range = (0,1), density= None)[0]/len(velocity_real))
    mean_val.append(round(np.mean(velocity_real), 3))
    std_val.append(round(np.std(velocity_real), 4))
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
plt.close()




parameter_initial = np.array([0.067, 1.23])
# function to fit
def func(f, a, b):
    return a * (f ** b)



for x in range(len(year_list_1)):
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
    separate_num = 27
    for j in range(len(csv_input_final)):
        if csv_input_final['event_date'][j] >= year_list_1[x] and csv_input_final['event_date'][j] <= year_list_2[x]:
            # print (csv_input_final['event_date'][j])
            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
            best_factor = csv_input_final["factor"][j]
            for z in range(separate_num):
                freq.append((csv_input_final[["freq_start"][0]][j] - csv_input_final[["freq_end"][0]][j])*((z + 1)/(separate_num+1)) + csv_input_final[["freq_end"][0]][j])
                start_check.append(csv_input_final[["freq_start"][0]][j])
                y_factor.append(best_factor)
                y_velocity.append(float(velocity[best_factor-1]))
                time_gap.append(csv_input_final[["event_end"][0]][j] - csv_input_final[["event_start"][0]][j] + 1)

    

    t = np.arange(0, 2000, 1)
    t = (t+1)/100
    
    for i in range (len(y_factor)):
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
    
    

    xdata = np.array(freq)
    ydata = np.array(freq_drift)
    freq_range = np.arange(30, 84, 4)
    xdata_list_final = []
    ydata_list_final = []
    xdata_error = []
    ydata_error = []
    for i in range(len(freq_range) - 1):
        xdata_list = []
        ydata_list = []
        for j in range(len(xdata)):
            if xdata[j] >= freq_range[i]:
                if xdata[j] < freq_range[i + 1]:
                    xdata_list.append(xdata[j])
                    ydata_list.append(ydata[j])
        if len(xdata_list) >= 100:
            xdata_list_final.append(np.mean(xdata_list))
            ydata_list_final.append(np.mean(ydata_list))
            ydata_error.append(np.std(ydata_list))
            xdata_error.append(np.std(xdata_list))


    print (year_num[x])
    print (separate_num * year_num[x])
    paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata, ydata, p0=parameter_initial)
    error = 2 * np.sqrt(np.diag(covariance))
    #print ("paramater =", paramater_optimal)
    # final_xdata = np.arange(min(xdata) - 8, max(xdata) + 15, 0.05)
    final_xdata = np.arange(10, 80, 0.05)
    y = func(final_xdata,paramater_optimal[0],paramater_optimal[1])
    y_up = func(final_xdata,paramater_optimal[0] + error[0],paramater_optimal[1] +  error[1])
    y_down = func(final_xdata,paramater_optimal[0] - error[0],paramater_optimal[1] -  error[1])
    print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
    # plt.plot(xdata, ydata, '.', color = '#ff7f00', markersize=2)
    # plt.plot(xdata, ydata, '.', markersize=2, color = color_list[x])

    plt.errorbar(xdata_list_final, ydata_list_final, yerr = ydata_error, fmt=color_list_1[x] + 'o', ecolor=color_list_1[x], markeredgecolor = color_list_1[x], capsize=4)
    plt.plot(final_xdata, y, '-', color = color_list_1[x], label = str(year_list_1[x])[:4] + '-' + str(year_list_2[x])[:4] + ' (This work)', linewidth = 3.0)
    # plt.plot(final_xdata, y_up, '--', color = color_list_1[x], linewidth = 1.0)
    # plt.plot(final_xdata, y_down, '--', color = color_list_1[x], linewidth = 1.0)
    
# y = func(final_xdata,0.0672,1.23)
# plt.plot(final_xdata, y, '-', color = 'k', label = 'P. J. Zhang et al., 2018', linewidth = 3.0)
    
plt.ylim(0,22)
plt.xlim(30,60)
# plt.ylim(0,25)
# plt.xlim(30,62)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1)
plt.tick_params(labelsize=15)
plt.xlabel('Frequency [MHz]',fontsize=15)
plt.ylabel('Frequency drift rate[MHz/s]',fontsize=15)
plt.show()










fig = plt.figure()
ax = fig.add_subplot(111)
for x in range(len(year_list_1)):
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
    event_date_l = []
    for j in range(len(csv_input_final)):
        if csv_input_final['event_date'][j] >= year_list_1[x] and csv_input_final['event_date'][j] <= year_list_2[x]:
            # print (csv_input_final['event_date'][j])
            velocity = csv_input_final[["velocity"][0]][j].lstrip("['")[:-1].split(',')
            best_factor = csv_input_final["factor"][j]
            separate_num = 11
            for z in range(separate_num):
                event_date_l.append(csv_input_final['event_date'][j])
                freq.append((csv_input_final[["freq_start"][0]][j] - csv_input_final[["freq_end"][0]][j])*((z + 1)/(separate_num+1)) + csv_input_final[["freq_end"][0]][j])
                start_check.append(csv_input_final[["freq_start"][0]][j])
                y_factor.append(best_factor)
                y_velocity.append(float(velocity[best_factor-1]))
                time_gap.append(csv_input_final[["event_end"][0]][j] - csv_input_final[["event_start"][0]][j] + 1)

    

    t = np.arange(0, 2000, 1)
    t = (t+1)/100
    
    for i in range (len(y_factor)):
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
    
    

    xdata = np.array(freq)
    ydata = np.array(freq_drift)
    event_date = np.array(event_date_l)
    freq_range = np.arange(30, 84, 4)
    xdata_list_final = []
    ydata_list_final = []
    xdata_error = []
    ydata_error = []
    for i in range(len(freq_range) - 1):
        xdata_list = []
        ydata_list = []
        for j in range(len(xdata)):
            if xdata[j] >= freq_range[i]:
                if xdata[j] < freq_range[i + 1]:
                    xdata_list.append(xdata[j])
                    ydata_list.append(ydata[j])
        if len(xdata_list) >= 100:
            xdata_list_final.append(np.mean(xdata_list))
            ydata_list_final.append(np.mean(ydata_list))
            ydata_error.append(np.std(ydata_list))
            xdata_error.append(np.std(xdata_list))


    print (year_num[x])
    print (separate_num * year_num[x])
    paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata, ydata, p0=parameter_initial)
    error = 2 * np.sqrt(np.diag(covariance))
    #print ("paramater =", paramater_optimal)
    # final_xdata = np.arange(min(xdata) - 8, max(xdata) + 15, 0.05)
    final_xdata = np.arange(10, 80, 0.05)
    y = func(final_xdata,paramater_optimal[0],paramater_optimal[1])
    y_up = func(final_xdata,paramater_optimal[0] + error[0],paramater_optimal[1] +  error[1])
    y_down = func(final_xdata,paramater_optimal[0] - error[0],paramater_optimal[1] -  error[1])
    print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
    # plt.plot(xdata, ydata, '.', color = '#ff7f00', markersize=2)
    im = ax.scatter(xdata, ydata, c=event_date/10000, cmap = 'RdBu', s = 2, vmin = min(year_list_1)/10000, vmax = max(year_list_2)/10000)
    # plt.errorbar(xdata_list_final, ydata_list_final, yerr = ydata_error, fmt=color_list_1[x] + 'o', ecolor=color_list_1[x], markeredgecolor = color_list_1[x], capsize=4)
    ax.plot(final_xdata, y, '-', color = color_list_1[x], label = str(year_list_1[x])[:4] + '-' + str(year_list_2[x])[:4] + ' (This work)', linewidth = 3.0)
    # plt.plot(final_xdata, y_up, '--', color = color_list_1[x], linewidth = 1.0)
    # plt.plot(final_xdata, y_down, '--', color = color_list_1[x], linewidth = 1.0)


# y = func(final_xdata,0.0672,1.23)
# plt.plot(final_xdata, y, '-', color = 'k', label = 'P. J. Zhang et al., 2018', linewidth = 3.0)

y = func(final_xdata,0.0068,1.82)
ax.plot(final_xdata, y, '-', color = 'w', label = 'Morosan et al., 2015', linewidth = 3.0)


# y = func(final_xdata,0.073,1.25)
# plt.plot(final_xdata, y, '--', color = 'k', linewidth = 1.0)

# y = func(final_xdata,0.061,1.21)
# plt.plot(final_xdata, y, '--', color = 'k', linewidth = 1.0)
# cbar = fig.colorbar(im, ticks = [min(year_list_1)/10000, (int(str(min(year_list_1))[:4])+int(str(max(year_list_2))[:4]))/2, int(str(max(year_list_2)/10000)[:4])])
cbar = fig.colorbar(im, ticks =np.arange(min(year_list_1)/10000,int(str(max(year_list_2)/10000 + 1)[:4]),1))
cbar.ax.set_ylim(min(year_list_1)/10000,max(year_list_2)/10000)
tick_list = []
for i in np.arange(min(year_list_1)/10000,int(str(max(year_list_2)/10000 + 1)[:4]),1):
    tick_list.append(str(i)[:4])
cbar.ax.set_yticklabels(tick_list)
# plt.colorbar(ticks=[min(year_list_1)/10000, (int(str(min(year_list_1))[:4])+int(str(max(year_list_2))[:4]))/2, max(event_date)/10000])
# cbar = plt.colorbar(ticks=[min(year_list_1)/10000, (int(str(min(year_list_1))[:4])+int(str(max(year_list_2))[:4]))/2, max(year_list_2)/10000])
# cbar.set_ticklabels([str(min(year_list_1)/10000)[:4], str((int(str(min(year_list_1))[:4])+int(str(max(year_list_2))[:4]))/2), str(max(year_list_2)/10000)[:4]])


# ticks = cbar.get_ticks() # [-4. -2.  0.  2.  4.]
# ticklabels = [ticklabel.get_text() for ticklabel in cbar.ax.get_xticklabels()]
# # ['−4', '−2', '0', '2', '4']
# ticklabels[-1] += ' [year]'
# cbar.set_ticks(ticks)
# cbar.set_ticklabels(ticklabels)


plt.ylim(0,22)
plt.xlim(30,60)
# plt.ylim(0,25)
# plt.xlim(30,62)
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=1)
plt.tick_params(labelsize=15)
plt.xlabel('Frequency [MHz]',fontsize=15)
plt.ylabel('Frequency drift rate[MHz/s]',fontsize=15)
plt.show()
plt.close()
    
    # number_count = 0
    # for j in range(len(freq)):
    #     h = freq[j]
    #     up_value = (paramater_optimal[0] + error[0]) * (h ** (paramater_optimal[1] + error[1]))
    #     down_value = (paramater_optimal[0] - error[0]) * (h ** (paramater_optimal[1] - error[1]))
    #     if freq_drift[j] >= down_value:
    #         if freq_drift[j] <= up_value:
    #             number_count += 1
    # print (number_count/len(freq))



