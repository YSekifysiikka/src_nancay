#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 08:17:50 2020

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

factor_list = []

index_2013 = []
index_2017 = []
index_2018 = []
index_2013.append(0)
index_2017.append(0)
index_2018.append(0)
x_list = []
y_factor = []
y_residual = []
y_list_2013 = []
y_list_2017 = []
y_list_2018 = []
residual_list_2013 = []
residual_list_2017 = []
residual_list_2018 = []
velocity_2013 = []
velocity_2017 = []
velocity_2018 = []
velocity_2013_factor_2 = []
velocity_2017_factor_2 = []
velocity_2018_factor_2 = []
freq_drift_2013 = []
freq_drift_2017 = []
freq_drift_2018 = []
y_freq_drift = []
y_velocity = []
y_velocity_factor_2 = []
check_factor = []
#file = "velocity_factor_jpgu_test.csv"
file = 'velocity_factor_jpgu_pparc1111.csv'
#file = "velocity_factor_jpgu_1.csv"
csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")


# インプットの項目数（行数 * カラム数）を返却します。
#print(csv_input.size)
## 指定したカラムだけ抽出したDataFrameオブジェクトを返却します。
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

#ファクターの年変化の解析
for i in range(len(csv_input)):
    velocity = csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')
    residual = csv_input[["residual"][0]][i].lstrip("['")[:-1].split(',')
    velocity_list = [float(s) for s in velocity]
    residual_list = [float(s) for s in residual]
    best_factor = csv_input[["factor"][0]][i]
    factor_list.append(best_factor)
    x_list.append(dt.datetime(int(str(csv_input[["event_date"][0]][i])[0:4]), int(str(csv_input[["event_date"][0]][i])[4:6]),int(str(csv_input[["event_date"][0]][i])[6:8]), int(csv_input[["event_hour"][0]][i]), int(csv_input[["event_minite"][0]][i])))
    y_factor.append(best_factor)
    y_freq_drift.append(csv_input[["freq_drift"][0]][i])
    y_velocity.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[best_factor-1]))
    y_velocity_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))
    y_residual.append(min(residual_list))
#thresh_freq_drift = np.mean(y_freq_drift) + 2*np.std(y_freq_drift)
#for i in range(len(csv_input)):
#    if y_freq_drift[i] <= thresh_freq_drift:
    if int(str(csv_input[["event_date"][0]][i])[0:4]) == 2013:
        y_list_2013.append(best_factor)
        residual_list_2013.append(min(residual_list))
        velocity_2013.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[best_factor-1]))
        velocity_2013_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))
        freq_drift_2013.append(csv_input[["freq_drift"][0]][i])
        index_2013.append(i + 1)
    elif int(str(csv_input[["event_date"][0]][i])[0:4]) == 2017:
        y_list_2017.append(best_factor)
        residual_list_2017.append(min(residual_list))
        velocity_2017.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[best_factor-1]))
        velocity_2017_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))
        freq_drift_2017.append(csv_input[["freq_drift"][0]][i])
        index_2017.append(i + 1)
    elif int(str(csv_input[["event_date"][0]][i])[0:4]) == 2018:
        y_list_2018.append(best_factor)
        residual_list_2018.append(min(residual_list))
        velocity_2018.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[best_factor-1]))
        velocity_2018_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))
        freq_drift_2018.append(csv_input[["freq_drift"][0]][i])
        index_2018.append(i + 1)
#    if best_factor > 19:
#        check_factor.append(best_factor)
#
#        path = "/Volumes/HDPH-UT/lab/solar_burst/Nancay/plot/drift_check/test_jpgu/2/" + (str(csv_input[['event_date'][0]][i])[0:4]) + "/" + (str(csv_input[['event_date'][0]][i])[4:6]) + "/" + (str(csv_input[['event_date'][0]][i])) + "_*_*_" +(str(csv_input[['event_start'][0]][i]))+"_"+(str(csv_input[['event_end'][0]][i]))+"_"+(str(csv_input[['freq_start'][0]][i]))+"_"+(str(csv_input[['freq_end'][0]][i]))+"*.png"
#        File = glob.glob(path, recursive=True)
#        if len(File) == 2:
#            print(str(csv_input[['event_date'][0]][i]))
#            for j in range(2):
#                im = Image.open(File[j])
#                im_list = np.asarray(im)
#                plt.figure(figsize=(4, 4), dpi=250)
#                plt.tick_params(bottom=False,left=False,right=False,top=False)
#                plt.tick_params(labelbottom=False,labelleft=False,labelright=False,labeltop=False)
#                plt.imshow(im_list)
#                plt.show()
        
df_2013 = pd.read_csv(filepath_or_buffer=file, sep=",", skiprows=lambda x: x not in np.array(index_2013))
df_2017 = pd.read_csv(filepath_or_buffer=file, sep=",", skiprows=lambda x: x not in np.array(index_2017))
df_2018 = pd.read_csv(filepath_or_buffer=file, sep=",", skiprows=lambda x: x not in np.array(index_2018))
df_2013['appropriate_velocity'] = velocity_2013
df_2017['appropriate_velocity'] = velocity_2017
df_2013['factor2_velocity'] = velocity_2013_factor_2
df_2017['factor2_velocity'] = velocity_2017_factor_2
df_2018['appropriate_velocity'] = velocity_2018
df_2018['factor2_velocity'] = velocity_2018_factor_2
#        check_factor.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))



y_2013 = []
y_2017 = []
y_2018 = []
x = np.arange(min(factor_list), max(factor_list) + 1, 1)
for i in x:
    y_2013.append(y_list_2013.count(i))
    y_2017.append(y_list_2017.count(i))
    y_2018.append(y_list_2018.count(i))
y_2013 = np.array(y_2013)
y_2017 = np.array(y_2017)
y_2018 = np.array(y_2018)

min_year = 2013
check_year = [2013, 2017, 2018]
x_month_list = []
number = []
number.append(0)
for y in check_year:
    for x_month in range(12):
        x_minus = [i for i in np.array(x_list) - datetime.datetime(y, x_month + 1, 1, 0, 0) if i < datetime.timedelta(days=0, seconds=0)]
        if len(x_minus) > 0:
            number.append(np.argmax(x_minus)+1)
        x_month_list.append(datetime.datetime(y, x_month + 1, 1, 0, 0))
number.append(len(x_list))
x_month_list.append(datetime.datetime(2019, 1, 1, 0, 0))

plt.close(1)
fig = plt.figure(figsize=(10,14),dpi=80)
ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the factor', ylabel = 'Factor', xlim = (datetime.datetime(2013, 1, 1, 0, 0),datetime.datetime(2013, 12, 30, 14, 32)), ylim = (min(x)-1, max(x) + 1))
ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the factor', ylabel = 'Factor', xlim = (datetime.datetime(2017, 1, 1, 0, 0),datetime.datetime(2017, 12, 30, 14, 32)), ylim = (min(x)-1, max(x) + 1))
ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Appropriate Factor of B-A model',ylabel = 'The number of events')
ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Appropriate Factor of B-A model', ylabel = 'The number of events')
ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the factor',xlabel = 'Factor', ylabel = 'Occurrence rate')
labelsize = 12
ax1.plot(x_list, y_factor, '.', color = '#ff7f00')
for i in range(len(number)-1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax1.plot(x_month_list[i:i+2], [np.mean(y_factor[number[i]:number[i + 1]])] * 2, color = 'r')
        ax1.plot([x_month_list[i + 1]]*2, [np.mean(y_factor[number[i]:number[i + 1]]), np.mean(y_factor[number[i + 1]:number[i + 2]])], color = 'r')
ax1.plot(x_month_list[i:i+2], [np.mean(y_factor[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax1.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

ax2.plot(x_list, y_factor, '.', color ="b")
for i in range(len(number)-1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax2.plot(x_month_list[i:i+2], [np.mean(y_factor[number[i]:number[i + 1]])] * 2, color = 'r')
        ax2.plot([x_month_list[i + 1]]*2, [np.mean(y_factor[number[i]:number[i + 1]]), np.mean(y_factor[number[i + 1]:number[i + 2]])], color = 'r')
ax2.plot(x_month_list[i:i+2], [np.mean(y_factor[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_2 = ax2.xaxis
xaxis_2.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

ax3.bar(x, y_2013, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = '#ff7f00')
ax3.tick_params(labelsize=labelsize)

ax4.bar(x, y_2017, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color ="b")
ax4.tick_params(labelsize=labelsize)

ax5.bar(x + 0.4, y_2017/len(y_list_2017), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label='2017', color ="b")
ax5.bar(x, y_2013/len(y_list_2013), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label='2013', color = '#ff7f00')
ax5.legend()
ax5.set_xticks(x + 0.2)
ax5.set_xticklabels(x)
ax5.tick_params(labelsize=labelsize - 2)
if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/factor.png'
plt.savefig(filename)
plt.show()


#周波数の年変化の解析
plt.close(1)
fig = plt.figure(figsize=(10,14),dpi=80)
ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the frequency drift rate', ylabel = 'Frequency drift[MHz/s]', xlim = (datetime.datetime(2013, 1, 1, 0, 0),datetime.datetime(2013, 12, 30, 14, 32)))
ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the frequency drift rate', ylabel = 'Frequency drift[MHz/s]', xlim = (datetime.datetime(2017, 1, 1, 0, 0),datetime.datetime(2017, 12, 30, 14, 32)))
ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Frequency drift[MHz/s]',ylabel = 'The number of events')
ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Frequency drift[MHz/s]', ylabel = 'The number of events')
ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the frequency drift rate',xlabel = 'Frequency drift[MHz/s]', ylabel = 'Occurrence rate')
#ax6 = fig.add_subplot(5, 2, 10, title='Occurrence rate of each factor',xlabel = 'Factor', ylabel = 'Occurrence rate')


labelsize = 12
ax1.plot(x_list, y_freq_drift, '.', color = '#ff7f00')
for i in range(len(number) -1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax1.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r')
        ax1.plot([x_month_list[i + 1]]*2, [np.mean(y_freq_drift[number[i]:number[i + 1]]), np.mean(y_freq_drift[number[i + 1]:number[i + 2]])], color = 'r')
ax1.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax1.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

ax2.plot(x_list, y_freq_drift, '.', color = 'b')
for i in range(len(number) - 1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax2.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r')
        ax2.plot([x_month_list[i + 1]]*2, [np.mean(y_freq_drift[number[i]:number[i + 1]]), np.mean(y_freq_drift[number[i + 1]:number[i + 2]])], color = 'r')
ax2.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax2.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)


data_inverval = 1
min_value = min(y_freq_drift).astype(int)
max_value = max(y_freq_drift).round(decimals=0) + data_inverval
test_2013 = df_2013.groupby(pd.cut(df_2013[["freq_drift"][0]], np.arange(min_value, max_value, data_inverval))).size()
ax3.bar([x for x in range(0, len(test_2013.index))], test_2013.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = '#ff7f00')
#for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
#    ax3.text(i, ypos + data_inverval, ypos, horizontalalignment='center', color='black', fontweight='bold')
xticks = ['{:.1f} - {:.1f}'.format(i, i + data_inverval) for i in np.arange(min_value - 5, max_value - 5, 5 * data_inverval)]
ax3.set_xticklabels(xticks, rotation=45, size='small')
ax3.tick_params(labelsize=labelsize-2)



test_2017 = df_2017.groupby(pd.cut(df_2017[["freq_drift"][0]], np.arange(min_value, max_value, data_inverval))).size()
ax4.bar([x for x in range(0, len(test_2017.index))], test_2017.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = 'b')
#for i, ypos in enumerate(test_2017.values):
#    ax4.text(i, ypos + data_inverval, ypos, horizontalalignment='center', color='black', fontweight='bold')
xticks = ['{:.1f} - {:.1f}'.format(i, i + data_inverval) for i in np.arange(min_value - 5, max_value - 5, 5 * data_inverval)]
ax4.set_xticklabels(xticks, rotation=45)
ax4.tick_params(labelsize=labelsize-2)




ax5.bar(np.array([x for x in range(0, len(test_2013.index))]) - 0.2, test_2013.values/len(y_list_2013), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2013', color = '#ff7f00')
ax5.bar(np.array([x for x in range(0, len(test_2017.index))]) + 0.2, test_2017.values/len(y_list_2017), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2017', color = 'b')
ax5.legend()
xticks = ['{:.1f} - {:.1f}'.format(i, i + data_inverval) for i in np.arange(min_value - 5, max_value - 5, 5 * data_inverval)]
ax5.set_xticklabels(xticks, rotation=45)
ax5.tick_params(labelsize=labelsize)
if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/drift_rate.png'
plt.savefig(filename)
plt.show()




#速度の年変化の解析with best factor
plt.close(1)
fig = plt.figure(figsize=(10,14),dpi=80)
ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the velocity/app',ylim = (0,1), ylabel = 'Velocity(c)/app', xlim = (datetime.datetime(2013, 1, 1, 0, 0),datetime.datetime(2013, 12, 30, 14, 32)))
ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the velocity/app',ylim = (0,1), ylabel = 'Velocity(c)/app', xlim = (datetime.datetime(2017, 1, 1, 0, 0),datetime.datetime(2017, 12, 30, 14, 32)))
ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Velocity(c)/app',ylabel = 'The number of events')
ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Velocity(c)/app', ylabel = 'The number of events')
ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the velocity/app',xlabel = 'Velocity(c)/app', ylabel = 'Occurrence rate')

labelsize = 12
ax1.plot(x_list, y_velocity, '.', color = '#ff7f00')
for i in range(len(number) - 1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax1.plot(x_month_list[i:i+2], [np.mean(y_velocity[number[i]:number[i + 1]])] * 2, color = 'r')
        ax1.plot([x_month_list[i + 1]]*2, [np.mean(y_velocity[number[i]:number[i + 1]]), np.mean(y_velocity[number[i + 1]:number[i + 2]])], color = 'r')
ax1.plot(x_month_list[i:i+2], [np.mean(y_velocity[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax1.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

ax2.plot(x_list, y_velocity, '.', color = 'b')
for i in range(len(number) - 1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax2.plot(x_month_list[i:i+2], [np.mean(y_velocity[number[i]:number[i + 1]])] * 2, color = 'r')
        ax2.plot([x_month_list[i + 1]]*2, [np.mean(y_velocity[number[i]:number[i + 1]]), np.mean(y_velocity[number[i + 1]:number[i + 2]])], color = 'r')
ax2.plot(x_month_list[i:i+2], [np.mean(y_velocity[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax2.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)


data_interval = 0.05
min_value = 0
max_value = 1 + data_interval

test_2013 = df_2013.groupby(pd.cut(df_2013[["appropriate_velocity"][0]], np.arange(min_value, max_value, data_interval))).size()
ax3.bar([x for x in range(0, len(test_2013.index))], test_2013.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = '#ff7f00')
#for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
#    ax3.text(i, ypos + data_interval, ypos, horizontalalignment='center', color='black', fontweight='bold')
xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value-0.25 , max_value -0.25, data_interval*5)]
ax3.set_xticklabels(xticks, rotation=45, size='small')
ax3.tick_params(labelsize=labelsize-2)



test_2017 = df_2017.groupby(pd.cut(df_2017[["appropriate_velocity"][0]], np.arange(min_value, max_value, data_interval))).size()
ax4.bar([x for x in range(0, len(test_2017.index))], test_2017.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = 'b')
#for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
#    ax3.text(i, ypos + data_interval, ypos, horizontalalignment='center', color='black', fontweight='bold')
xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value-0.25 , max_value -0.25, data_interval*5)]
ax4.set_xticklabels(xticks, rotation=45, size='small')
ax4.tick_params(labelsize=labelsize-2)




ax5.bar(np.array([x for x in range(0, len(test_2013.index))]) - 0.2, test_2013.values/len(y_list_2013), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2013', color = '#ff7f00')
ax5.bar(np.array([x for x in range(0, len(test_2017.index))]) + 0.2, test_2017.values/len(y_list_2017), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2017', color = 'b')
ax5.legend()
ax5.set_xticks([0,5,10,15])
xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value , max_value -0.25, data_interval*5)]
ax5.set_xticklabels(xticks, rotation=30)
ax5.tick_params(labelsize=labelsize)
if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/velocity_app.png'
plt.savefig(filename)
plt.show()


#xticks = ['{:.1f} - {:.1f}'.format(i, i + data_interval) for i in np.arange(min_value-0.2 , max_value -0.2, data_interval*2)]




#速度の年変化の解析with factor=2
plt.close(1)
fig = plt.figure(figsize=(10,14),dpi=80)
ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the velocity/F:2', ylabel = 'Velocity(c)/F:2',ylim = (0,1), xlim = (datetime.datetime(2013, 1, 1, 0, 0),datetime.datetime(2013, 12, 30, 14, 32)))
ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the velocity/F:2', ylabel = 'Velocity(c)/F:2',ylim = (0,1), xlim = (datetime.datetime(2017, 1, 1, 0, 0),datetime.datetime(2017, 12, 30, 14, 32)))
ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Velocity(c)/F:2',ylabel = 'The number of events')
ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Velocity(c)/F:2', ylabel = 'The number of events')
ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the velocity/F:2',xlabel = 'Velocity(c)/F:2', ylabel = 'Occurrence rate')

labelsize = 12
ax1.plot(x_list, y_velocity_factor_2, '.', color = '#ff7f00')
for i in range(len(number) - 1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax1.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r')
        ax1.plot([x_month_list[i + 1]]*2, [np.mean(y_velocity_factor_2[number[i]:number[i + 1]]), np.mean(y_velocity_factor_2[number[i + 1]:number[i + 2]])], color = 'r')
ax1.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax1.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax1.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax1.tick_params(axis='y', labelsize=labelsize)
ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

ax2.plot(x_list, y_velocity_factor_2, '.', color = 'b')
for i in range(len(number) - 1):
    if len(x_list[number[i]:number[i + 1]]) > 0:
        ax2.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r')
        ax2.plot([x_month_list[i + 1]]*2, [np.mean(y_velocity_factor_2[number[i]:number[i + 1]]), np.mean(y_velocity_factor_2[number[i + 1]:number[i + 2]])], color = 'r')
ax2.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
xaxis_ = ax2.xaxis
xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax2.tick_params(axis='x', rotation=270, labelsize=labelsize)
ax2.tick_params(axis='y', labelsize=labelsize)
ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)


data_interval = 0.05
min_value = 0
max_value = 1 + data_interval

test_2013 = df_2013.groupby(pd.cut(df_2013[["factor2_velocity"][0]], np.arange(min_value, max_value, data_interval))).size()
ax3.bar([x for x in range(0, len(test_2013.index))], test_2013.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = '#ff7f00')
#for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
#    ax3.text(i, ypos + data_interval, ypos, horizontalalignment='center', color='black', fontweight='bold')
xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value-0.25 , max_value -0.25, data_interval*5)]
ax3.set_xticklabels(xticks, rotation=45, size='small')
ax3.tick_params(labelsize=labelsize-2)



test_2017 = df_2017.groupby(pd.cut(df_2017[["factor2_velocity"][0]], np.arange(min_value, max_value, data_interval))).size()
ax4.bar([x for x in range(0, len(test_2017.index))], test_2017.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = 'b')
#for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
#    ax3.text(i, ypos + data_interval, ypos, horizontalalignment='center', color='black', fontweight='bold')
xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value-0.25 , max_value -0.25, data_interval*5)]
ax4.set_xticklabels(xticks, rotation=45, size='small')
ax4.tick_params(labelsize=labelsize-2)


ax5.bar(np.array([x for x in range(0, len(test_2013.index))]) - 0.2, test_2013.values/len(y_list_2013), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2013', color = '#ff7f00')
ax5.bar(np.array([x for x in range(0, len(test_2017.index))]) + 0.2, test_2017.values/len(y_list_2017), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2017', color = 'b')
ax5.legend()
ax5.set_xticks([0,5,10,15])
xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value , max_value -0.25, data_interval*5)]
ax5.set_xticklabels(xticks, rotation=45)
ax5.tick_params(labelsize=labelsize)
if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/velocity_f2.png'
plt.savefig(filename)
plt.show()











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
#    for j in range(3):
#        velocity = csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')
#        best_factor = csv_input[["factor"][0]][i]
#        freq.append((csv_input[["freq_start"][0]][i] - csv_input[["freq_end"][0]][i])*((j + 1)/4) + csv_input[["freq_end"][0]][i])
#        y_factor.append(best_factor)
#        y_velocity.append(float(velocity[best_factor-1]))
#
#t = np.arange(0, 2000, 1)
#t = (t+1)/100
#for i in range (3*len(csv_input)):
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
#xdata_2013 = np.array(freq[0:3*3749])
#ydata_2013 = np.array(freq_drift[0:3*3749])
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
#plt.plot(xdata_2013, ydata_2013, '.', color = '#ff7f00', markersize=3)
##plt.plot(final_xdata_2013, y, '-', color = '#ff7f00')
##plt.ylim(10, 80)
#
##xdata_2017 = np.array(freq_mean[627:782])
##ydata_2017 = np.array(freq_drift[627:782])
#xdata_2017 = np.array(freq[3*3749:3*5383])
#ydata_2017 = np.array(freq_drift[3*3749:3*5383])
## initial guess for the parameters
#parameter_initial = np.array([0.067, 1.23])
## function to fit
#paramater_optimal, covariance = scipy.optimize.curve_fit(func, xdata_2017, ydata_2017, p0=parameter_initial)
##print ("paramater =", paramater_optimal)
#final_xdata_2017 = np.arange(min(xdata_2017) - 8, max(xdata_2017) + 15, 0.05)
#y_2017 = func(final_xdata_2017,paramater_optimal[0],paramater_optimal[1])
#print (str(paramater_optimal[0]) + 'f^' + str(paramater_optimal[1]))
#plt.plot(xdata_2017, ydata_2017, '.', color = '#377eb8', markersize=3)
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
#plt.plot(final_xdata_2013, y, '-', color = 'k', label = 'Alvarez&Haddock', linewidth = 3.0)
##
#y = func(final_xdata_2013,0.007,1.76)
#plt.plot(final_xdata_2013, y, '-', color = 'k', label = 'Mann', linewidth = 3.0)
##y = func(final_xdata,0.0084,1.57)
##plt.plot(final_xdata, y, '-', color = 'k', label = 'Clarke et al 2019')
##y = func(final_xdata,0.0074,1.65)
##plt.plot(final_xdata, y, '-', color = 'k', label = 'PeiJin Zhang et al 2020')
#plt.ylim(2, 40)
#plt.xlim(min(freq), max(freq)+5)
#plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
##plt.title('Frequency drift',fontsize=20)
#
##plt.tick_params(axis='x', rotation=300, labelsize=labelsize)
#plt.tick_params(labelsize=15)
#plt.xlabel('Frequency [MHz]',fontsize=15)
#plt.ylabel('Frequency drift [MHz/s]',fontsize=15)
##ax4.tick_params(axis='y', labelsize=labelsize)