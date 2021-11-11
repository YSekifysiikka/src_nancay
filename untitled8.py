#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 04:04:20 2021

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
    h = 1e-4
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))
def numerical_diff_allen(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))



# r = 69600000000*2
# f = 45
# dfdt = 5
# v = 2/numerical_diff_allen_velocity(factor, r)/f*-dfdt/29979245800


labelsize = 18
fontsize = 20
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/original_burst_cycle24.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


# year_list_1 = [20010101, 20070101, 20120101, 20170101]
# year_list_2 = [20011231, 20091231, 20141231, 20191231]
year_list_1 = [20170101, 20190101]
year_list_2 = [20171231, 20191231]
# year_list_1 = [20010101, 20070101]
# year_list_2 = [20011231, 20091231]

# year_list_1 = [20170101, 20190101]
# year_list_2 = [20171231, 20191231]

year_num = []
for i in range(len(year_list_1)):
    year_num.append(np.count_nonzero((csv_input_final['event_date'] >= year_list_1[i]) & (csv_input_final['event_date']<= year_list_2[i])))
print(year_num)

font_dict = dict(style="italic",size=16)

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
            best_factor = 2
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

        cube_5 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
        invcube_5 = inversefunc(cube_5, y_values = freq[i])
        h_radio = invcube_5 + 69600000000
        velocity_real.append(2/numerical_diff_allen_velocity(factor, h_radio)/freq[i]*slope/29979245800)



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


