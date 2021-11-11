#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:04:33 2020

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
import numpy as np

x_list = []
y_freq_drift = []
factor_list = []
y_velocity_factor_2 = []


file = "velocity_factor_jpgu_1.csv"
csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
x = np.arange(0,24,1)
obs_time_2013 = np.array([0.,0.,0.,0.,0.,0.,0.,47.45,334.98333333,353.05,358.08333333,359.,358.81666667, 358.73333333, 355., 289.15,3.15,0.,0.,0.,0.,0.,0.,0.])
obs_time_2017 = np.array([0.,0.,0.,0.,0.,0.,0.,35.26666667,254.,257.8,254.75,231.33333333,213.06666667,214.48333333,208.61666667,133.61666667,0.,0.,0.,0.,0.,0.,0.,0.])
obs_time_all = obs_time_2013 + obs_time_2017
for i in range(len(csv_input)):
    x_list.append(dt.datetime(int(str(csv_input[["event_date"][0]][i])[0:4]), int(str(csv_input[["event_date"][0]][i])[4:6]),int(str(csv_input[["event_date"][0]][i])[6:8]), int(csv_input[["event_hour"][0]][i]), int(csv_input[["event_minite"][0]][i])))
    y_freq_drift.append(csv_input[["freq_drift"][0]][i])
    factor_list.append(csv_input[["factor"][0]][i])
    y_velocity_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))

#print(csv_input[["event_date"][0]])
#print(csv_input[["event_hour"][0]])
#print(csv_input[["event_minite"][0]])
#print(csv_input[["velocity"][0]])
#print(csv_input[["residual"][0]])
#print(csv_input[["freq_start"][0]])
#print(csv_input[["freq_end"][0]])
#print(csv_input[["time_gap"][0]])
#print(-1/csv_input[["freq_drift"][0]])
#print(csv_input[["factor"][0]])
#csv_input[["velocity"][0]][0].lstrip("['")[:-1]

obs_event_number = np.array([0]*24)
for i in range(len(csv_input)):
    obs_one_event = []
    for j in range(24):
        if csv_input[["event_hour"][0]][i] == j:
            obs_one_event.append(1)
        else:
            obs_one_event.append(0)
    obs_event_number += np.array(obs_one_event)

occ_rate_list = []
x_new = []
for i in range(len(x)):
    if obs_time_all[i] == 0:
        pass
    else:
        occ_rate_list.append(obs_event_number[i]/obs_time_all[i])
        x_new.append(i)

#日変化の解析
x_list_hour = []
for i in range(len(x_list)):
    x_list_new = datetime.datetime(2016, 7, 1, x_list[i].hour, x_list[i].minute)
    x_list_hour.append(x_list_new)
x_list_mean_plot = []
for i in range(min(x_new), max(x_new) + 2,1):
    x_list_mean_plot.append(datetime.datetime(2016, 7, 1, i, 0))

y_freq_hour = []
y_factor_hour = []
y_velocity_f2_hour = []
for j in x_new:
    y_freq_one_hour = []
    y_factor_one_hour = []
    y_velocity_f2_one_hour = []
    for i in range(len(x_list_hour)):
        if x_list_hour[i].hour == j:
            y_freq_one_hour.append(y_freq_drift[i])
            y_factor_one_hour.append(factor_list[i])
            y_velocity_f2_one_hour.append(y_velocity_factor_2[i])
    if len(y_freq_one_hour) > 0:
        y_freq_hour.append(np.mean(y_freq_one_hour))
        y_factor_hour.append(np.mean(y_factor_one_hour))
        y_velocity_f2_hour.append(np.mean(y_velocity_f2_one_hour))
    else:
        y_freq_hour.append(0.)
        y_factor_hour.append(0.)
        y_velocity_f2_hour.append(0.)


        

plt.close(1)
fig = plt.figure(figsize=(10,14),dpi=80)
ax1 = fig.add_subplot(5, 2, 1, title='All events')
ax2 = fig.add_subplot(5, 2, 2, title='All events')
ax3 = fig.add_subplot(5, 2, 5, title='All events')
ax4 = fig.add_subplot(5, 2, 6, title='Hourly varidation of the Frequency drift', xlim = (x_list_mean_plot[0],x_list_mean_plot[-1]), ylim = (min(y_freq_drift), max(y_freq_drift)))
ax5 = fig.add_subplot(5, 2, 9, title='Hourly varidation of the Factor', xlim = (x_list_mean_plot[0],x_list_mean_plot[-1]), ylim = (min(factor_list), max(factor_list)))
ax6 = fig.add_subplot(5, 2, 10, title='Hourly varidation of the Velocity/F:2', xlim = (x_list_mean_plot[0],x_list_mean_plot[-1]), ylim = (min(y_velocity_factor_2), max(y_velocity_factor_2)))
labelsize = 12
ax1.bar(x_new, obs_event_number[x_new[0]:x_new[-1] + 1])
ax1.set_xlabel(xlabel = 'UT(Hour)',fontsize=labelsize)
ax1.set_ylabel(ylabel = 'Number of events', fontsize=labelsize)
ax2.bar(x_new, occ_rate_list)
ax2.set_xlabel(xlabel = 'UT(Hour)',fontsize=labelsize)
ax2.set_ylabel(ylabel = 'Events/Hour', fontsize=labelsize)
ax3.bar(x_new, obs_time_all[x_new[0]:x_new[-1] + 1])
ax3.set_xlabel(xlabel = 'UT(Hour)',fontsize=labelsize)
ax3.set_ylabel(ylabel = 'Obs_time[Hour]', fontsize=labelsize)

ax4.plot(x_list_hour, y_freq_drift, '.', color = '0.8')
for i in range(len(x_list_mean_plot)-1):
    ax4.plot(x_list_mean_plot[i:i+2],  [y_freq_hour[i]] * 2, color = 'r')
    if not x_list_mean_plot[i + 1].hour == x_new[-1] + 1:
        ax4.plot([x_list_mean_plot[i + 1]]*2, [y_freq_hour[i], y_freq_hour[i+1]], color = 'r')
ax4.plot(x_list_mean_plot[i:i+2],  [y_freq_hour[i]] * 2, color = 'r', label = 'Hourly average')
xaxis_ = ax4.xaxis
xaxis_.set_major_formatter(DateFormatter('%H:%M'))
ax4.tick_params(axis='x', rotation=300, labelsize=labelsize)
ax4.tick_params(axis='y', labelsize=labelsize)
ax4.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
ax4.set_xlabel(xlabel = 'UT(Hour)',fontsize=labelsize)
ax4.set_ylabel(ylabel = 'Frequency drift[MHz/s]', fontsize=labelsize)

ax5.plot(x_list_hour, factor_list, '.', color = '0.8')
for i in range(len(x_list_mean_plot)-1):
    ax5.plot(x_list_mean_plot[i:i+2],  [y_factor_hour[i]] * 2, color = 'r')
    if not x_list_mean_plot[i + 1].hour == x_new[-1] + 1:
        ax5.plot([x_list_mean_plot[i + 1]]*2, [y_factor_hour[i], y_factor_hour[i+1]], color = 'r')
ax5.plot(x_list_mean_plot[i:i+2],  [y_factor_hour[i]] * 2, color = 'r', label = 'Hourly average')
xaxis_ = ax5.xaxis
xaxis_.set_major_formatter(DateFormatter('%H:%M'))
ax5.tick_params(axis='x', rotation=300, labelsize=labelsize)
ax5.tick_params(axis='y', labelsize=labelsize)
ax5.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
ax5.set_xlabel(xlabel = 'UT(Hour)',fontsize=labelsize)
ax5.set_ylabel(ylabel = 'Factor', fontsize=labelsize)


ax6.plot(x_list_hour, y_velocity_factor_2, '.', color = '0.8')
for i in range(len(x_list_mean_plot)-1):
    ax6.plot(x_list_mean_plot[i:i+2],  [y_velocity_f2_hour[i]] * 2, color = 'r')
    if not x_list_mean_plot[i + 1].hour == x_new[-1] + 1:
        ax6.plot([x_list_mean_plot[i + 1]]*2, [y_velocity_f2_hour[i], y_velocity_f2_hour[i+1]], color = 'r')
ax6.plot(x_list_mean_plot[i:i+2],  [y_velocity_f2_hour[i]] * 2, color = 'r', label = 'Hourly average')
xaxis_ = ax6.xaxis
xaxis_.set_major_formatter(DateFormatter('%H:%M'))
ax6.tick_params(axis='x', rotation=300, labelsize=labelsize)
ax6.tick_params(axis='y', labelsize=labelsize)
ax6.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
ax6.set_xlabel(xlabel = 'UT(Hour)',fontsize=labelsize)
ax6.set_ylabel(ylabel = 'Velocity/F:2(c)', fontsize=labelsize)
plt.show()