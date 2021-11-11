#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 02:35:06 2020

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

factor_list = []

index_2013 = []
index_2017 = []
index_2013.append(0)
index_2017.append(0)
x_list = []
y_factor = []
y_residual = []
y_list_2013 = []
y_list_2017 = []
residual_list_2013 = []
residual_list_2017 = []
velocity_2013 = []
velocity_2017 = []
velocity_2013_factor_2 = []
velocity_2017_factor_2 = []
freq_drift_2013 = []
freq_drift_2017 = []
y_freq_drift = []
y_velocity = []
y_velocity_factor_2 = []
check_factor = []
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
file = "/solar_burst/Nancay/analysis_data/velocity_factor_jpgu_pparc11.csv"
# file = "velocity_factor_jpgu_1.csv"
#file = "velocity_factor1.csv"
csv_input = pd.read_csv(filepath_or_buffer= Parent_directory + file, sep=",")


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
    if csv_input[["freq_start"][0]][i] - csv_input[["freq_end"][0]][i] > 40:
        if csv_input[["factor"][0]][i] < 6:
        #     if min([float(s) for s in csv_input[["residual"][0]][i].lstrip("['")[:-1].split(',')]) < 3.51:
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
                if int(str(csv_input[["event_date"][0]][i])[0:4]) == 2013:
                    y_list_2013.append(best_factor)
                    residual_list_2013.append(min(residual_list))
                    velocity_2013.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[best_factor-1]))
                    velocity_2013_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[3]))
                    freq_drift_2013.append(csv_input[["freq_drift"][0]][i])
                    index_2013.append(i + 1)
                elif int(str(csv_input[["event_date"][0]][i])[0:4]) == 2017:
                    y_list_2017.append(best_factor)
                    residual_list_2017.append(min(residual_list))
                    velocity_2017.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[best_factor-1]))
                    velocity_2017_factor_2.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[0]))
                    freq_drift_2017.append(csv_input[["freq_drift"][0]][i])
                    index_2017.append(i + 1)
                elif int(str(csv_input[["event_date"][0]][i])[0:4]) == 2018:
                    print('yes')
#    if float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[0]) > 0.50:
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


df_2013 = pd.read_csv(filepath_or_buffer=Parent_directory + file, sep=",", skiprows=lambda x: x not in np.array(index_2013))
df_2017 = pd.read_csv(filepath_or_buffer=Parent_directory + file, sep=",", skiprows=lambda x: x not in np.array(index_2017))
df_2013['appropriate_velocity'] = velocity_2013
df_2017['appropriate_velocity'] = velocity_2017
df_2013['factor2_velocity'] = velocity_2013_factor_2
df_2017['factor2_velocity'] = velocity_2017_factor_2
#        check_factor.append(float(csv_input[["velocity"][0]][i].lstrip("['")[:-1].split(',')[1]))



y_2013 = []
y_2017 = []
x = np.arange(min(factor_list), max(factor_list) + 1, 1)
for i in x:
    y_2013.append(y_list_2013.count(i))
    y_2017.append(y_list_2017.count(i))
y_2013 = np.array(y_2013)
y_2017 = np.array(y_2017)

min_year = 2013
check_year = [2013, 2017]
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
ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the factor', ylabel = 'Factor', xlim = (datetime.datetime(2013, 1, 2, 10, 59),datetime.datetime(2013, 12, 30, 14, 32)), ylim = (min(x)-1, max(x) + 1))
ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the factor', ylabel = 'Factor', xlim = (datetime.datetime(2017, 1, 2, 10, 59),datetime.datetime(2017, 12, 30, 14, 32)), ylim = (min(x)-1, max(x) + 1))
ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Appropriate Factor of B-A model',ylabel = 'The number of events')
ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Appropriate Factor of B-A model', ylabel = 'The number of events')
ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the factor')
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
ax5.set_xlabel('Factor',fontsize=15)
ax5.set_ylabel('Occurrence rate',fontsize=15)
if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/factor.png'
plt.savefig(filename)
plt.show()

# x = np.arange(min(factor_list), max(factor_list) + 1, 1)
# trytrytry = []
# for i in x:
#     trytry = []
#     for j in range(len(csv_input)):
#         if j < 627:
#             best_factor = csv_input[["factor"][0]][j]
#             if best_factor == i:
#                 trytry.append(float(csv_input[["velocity"][0]][j].lstrip("['")[:-1].split(',')[best_factor-1]))
#     trytrytry.append(np.mean(trytry))
# x = x - 0.5
# plt.close(1)
# fig = plt.figure(figsize=(10,14),dpi=80)
# ax1 = fig.add_subplot(1, 1, 1, title='factor vs velocity 2013', ylabel = 'Factor', xlim = (0,21), ylim = (0,1))
# ax1.plot(y_factor[:627], y_velocity[:627], '.', color = '#ff7f00')
# for i in range(19):
#     ax1.plot(x[i:i+2], [trytrytry[i]] * 2, color = 'r')
#     ax1.plot([x[i + 1]]*2, [trytrytry[i], trytrytry[i+1]], color = 'r')
# i += 1
# ax1.plot([19.5,20.5], [trytrytry[i]] * 2, color = 'r', label = 'Average')
# ax1.set_xlabel('Factor',fontsize=15)
# ax1.set_ylabel('velocity(cc)',fontsize=15)
# plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
# plt.show()

##周波数の年変化の解析
#plt.close(1)
#fig = plt.figure(figsize=(10,14),dpi=80)
#ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the frequency drift rate', ylabel = 'Frequency drift[MHz/s]', xlim = (datetime.datetime(2013, 1, 1, 0, 0),datetime.datetime(2013, 12, 30, 14, 32)))
#ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the frequency drift rate', ylabel = 'Frequency drift[MHz/s]', xlim = (datetime.datetime(2017, 1, 1, 0, 0),datetime.datetime(2017, 12, 30, 14, 32)))
#ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Frequency drift[MHz/s]',ylabel = 'The number of events')
#ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Frequency drift[MHz/s]', ylabel = 'The number of events')
#ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the frequency drift rate',xlabel = 'Frequency drift[MHz/s]', ylabel = 'Occurrence rate')
##ax6 = fig.add_subplot(5, 2, 10, title='Occurrence rate of each factor',xlabel = 'Factor', ylabel = 'Occurrence rate')
#
#
#labelsize = 12
#ax1.plot(x_list, y_freq_drift, '.', color = '#ff7f00')
#for i in range(len(number) -1):
#    if len(x_list[number[i]:number[i + 1]]) > 0:
#        ax1.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r')
#        ax1.plot([x_month_list[i + 1]]*2, [np.mean(y_freq_drift[number[i]:number[i + 1]]), np.mean(y_freq_drift[number[i + 1]:number[i + 2]])], color = 'r')
#ax1.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
#xaxis_ = ax1.xaxis
#xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
#ax1.tick_params(axis='x', rotation=270, labelsize=labelsize)
#ax1.tick_params(axis='y', labelsize=labelsize)
#ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#
#ax2.plot(x_list, y_freq_drift, '.', color = 'b')
#for i in range(len(number) - 1):
#    if len(x_list[number[i]:number[i + 1]]) > 0:
#        ax2.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r')
#        ax2.plot([x_month_list[i + 1]]*2, [np.mean(y_freq_drift[number[i]:number[i + 1]]), np.mean(y_freq_drift[number[i + 1]:number[i + 2]])], color = 'r')
#ax2.plot(x_month_list[i:i+2], [np.mean(y_freq_drift[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
#xaxis_ = ax2.xaxis
#xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
#ax2.tick_params(axis='x', rotation=270, labelsize=labelsize)
#ax2.tick_params(axis='y', labelsize=labelsize)
#ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#
#
#data_inverval = 1
#min_value = min(y_freq_drift).astype(int)
#max_value = max(y_freq_drift).round(decimals=0) + data_inverval
#test_2013 = df_2013.groupby(pd.cut(df_2013[["freq_drift"][0]], np.arange(min_value, max_value, data_inverval))).size()
#ax3.bar([x for x in range(0, len(test_2013.index))], test_2013.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = '#ff7f00')
##for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
##    ax3.text(i, ypos + data_inverval, ypos, horizontalalignment='center', color='black', fontweight='bold')
#xticks = ['{:.1f} - {:.1f}'.format(i, i + data_inverval) for i in np.arange(min_value - 5, max_value - 5, 5 * data_inverval)]
#ax3.set_xticklabels(xticks, rotation=45, size='small')
#ax3.tick_params(labelsize=labelsize-2)
#
#
#
#test_2017 = df_2017.groupby(pd.cut(df_2017[["freq_drift"][0]], np.arange(min_value, max_value, data_inverval))).size()
#ax4.bar([x for x in range(0, len(test_2017.index))], test_2017.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = 'b')
##for i, ypos in enumerate(test_2017.values):
##    ax4.text(i, ypos + data_inverval, ypos, horizontalalignment='center', color='black', fontweight='bold')
#xticks = ['{:.1f} - {:.1f}'.format(i, i + data_inverval) for i in np.arange(min_value - 5, max_value - 5, 5 * data_inverval)]
#ax4.set_xticklabels(xticks, rotation=45)
#ax4.tick_params(labelsize=labelsize-2)
#
#
#
#
#ax5.bar(np.array([x for x in range(0, len(test_2013.index))]) - 0.2, test_2013.values/len(y_list_2013), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2013', color = '#ff7f00')
#ax5.bar(np.array([x for x in range(0, len(test_2017.index))]) + 0.2, test_2017.values/len(y_list_2017), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2017', color = 'b')
#ax5.legend()
#xticks = ['{:.1f} - {:.1f}'.format(i, i + data_inverval) for i in np.arange(min_value - 5, max_value - 5, 5 * data_inverval)]
#ax5.set_xticklabels(xticks, rotation=45)
#ax5.tick_params(labelsize=labelsize)
#if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
#    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
#filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/drift_rate.png'
#plt.savefig(filename)
#plt.show()




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
#ax5.xlabel()
#ax5.ylabel(,fontsize=15)
ax5.set_xlabel('Velocity(c)/app',fontsize=15)
ax5.set_ylabel('Occurrence rate',fontsize=15)
if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/velocity_app.png'
plt.savefig(filename)
plt.show()


#xticks = ['{:.1f} - {:.1f}'.format(i, i + data_interval) for i in np.arange(min_value-0.2 , max_value -0.2, data_interval*2)]




# #速度の年変化の解析with factor=2
# plt.close(1)
# fig = plt.figure(figsize=(10,14),dpi=80)
# ax1 = fig.add_subplot(5, 2, 1, title='2013 : Temporal varidation of the velocity/F:2', ylabel = 'Velocity(c)/F:2',ylim = (0,1), xlim = (datetime.datetime(2013, 1, 1, 0, 0),datetime.datetime(2013, 12, 30, 14, 32)))
# ax2 = fig.add_subplot(5, 2, 2, title='2017 : Temporal varidation of the velocity/F:2', ylabel = 'Velocity(c)/F:2',ylim = (0,1), xlim = (datetime.datetime(2017, 1, 1, 0, 0),datetime.datetime(2017, 12, 30, 14, 32)))
# ax3 = fig.add_subplot(7, 2, 5, title='2013 events',xlabel = 'Velocity(c)/F:2',ylabel = 'The number of events')
# ax4 = fig.add_subplot(7, 2, 6, title='2017 events',xlabel = 'Velocity(c)/F:2', ylabel = 'The number of events')
# ax5 = fig.add_subplot(4, 1, 3, title='Occurrence rate of the velocity/F:2')

# labelsize = 12
# ax1.plot(x_list, y_velocity_factor_2, '.', color = '#ff7f00')
# for i in range(len(number) - 1):
#     if len(x_list[number[i]:number[i + 1]]) > 0:
#         ax1.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r')
#         ax1.plot([x_month_list[i + 1]]*2, [np.mean(y_velocity_factor_2[number[i]:number[i + 1]]), np.mean(y_velocity_factor_2[number[i + 1]:number[i + 2]])], color = 'r')
# ax1.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
# xaxis_ = ax1.xaxis
# xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
# ax1.tick_params(axis='x', rotation=270, labelsize=labelsize)
# ax1.tick_params(axis='y', labelsize=labelsize)
# ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)

# ax2.plot(x_list, y_velocity_factor_2, '.', color = 'b')
# for i in range(len(number) - 1):
#     if len(x_list[number[i]:number[i + 1]]) > 0:
#         ax2.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r')
#         ax2.plot([x_month_list[i + 1]]*2, [np.mean(y_velocity_factor_2[number[i]:number[i + 1]]), np.mean(y_velocity_factor_2[number[i + 1]:number[i + 2]])], color = 'r')
# ax2.plot(x_month_list[i:i+2], [np.mean(y_velocity_factor_2[number[i]:number[i + 1]])] * 2, color = 'r', label = 'Monthly average')
# xaxis_ = ax2.xaxis
# xaxis_.set_major_formatter(DateFormatter('%Y-%m-%d'))
# ax2.tick_params(axis='x', rotation=270, labelsize=labelsize)
# ax2.tick_params(axis='y', labelsize=labelsize)
# ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)


# data_interval = 0.05
# min_value = 0
# max_value = 1 + data_interval

# test_2013 = df_2013.groupby(pd.cut(df_2013[["factor2_velocity"][0]], np.arange(min_value, max_value, data_interval))).size()
# ax3.bar([x for x in range(0, len(test_2013.index))], test_2013.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = '#ff7f00')
# #for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
# #    ax3.text(i, ypos + data_interval, ypos, horizontalalignment='center', color='black', fontweight='bold')
# xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value-0.25 , max_value -0.25, data_interval*5)]
# ax3.set_xticklabels(xticks, rotation=45, size='small')
# ax3.tick_params(labelsize=labelsize-2)



# test_2017 = df_2017.groupby(pd.cut(df_2017[["factor2_velocity"][0]], np.arange(min_value, max_value, data_interval))).size()
# ax4.bar([x for x in range(0, len(test_2017.index))], test_2017.values, width=1.0, alpha=0.8, edgecolor='black', linewidth=1.0, color = 'b')
# #for i, ypos in enumerate(test_2013.values/len(y_freq_drift)):
# #    ax3.text(i, ypos + data_interval, ypos, horizontalalignment='center', color='black', fontweight='bold')
# xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value-0.25 , max_value -0.25, data_interval*5)]
# ax4.set_xticklabels(xticks, rotation=45, size='small')
# ax4.tick_params(labelsize=labelsize-2)


# ax5.bar(np.array([x for x in range(0, len(test_2013.index))]) - 0.2, test_2013.values/len(y_list_2013), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2013', color = '#ff7f00')
# ax5.bar(np.array([x for x in range(0, len(test_2017.index))]) + 0.2, test_2017.values/len(y_list_2017), width=0.4, alpha=0.8, edgecolor='black', linewidth=1.0, label = '2017', color = 'b')
# ax5.legend()
# ax5.set_xticks([0,5,10,15])
# xticks = ['{:.2f} - {:.2f}'.format(i, i + data_interval) for i in np.arange(min_value , max_value -0.25, data_interval*5)]
# ax5.set_xticklabels(xticks, rotation=45)
# ax5.tick_params(labelsize=labelsize)
# ax5.set_xlabel('Velocity(c)/F:2',fontsize=15)
# ax5.set_ylabel('Occurrence rate',fontsize=15)
# if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
#     os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
# filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/velocity_f2.png'
# plt.savefig(filename)
# plt.show()




##月変化の解析
#x_list_month = []
#for i in range(len(x_list)):
#    if not x_list[i].year == 2016:
#        x_list_new = datetime.datetime(2016, x_list[i].month, x_list[i].day, x_list[i].hour, x_list[i].minute)
#        x_list_month.append(x_list_new)
#    else:
#        x_list_month.append(x_list[i])
#
#
#x_month_list_new = []
#for i in range(len(x_month_list)):
#    if not x_month_list[i].year == 2016:
#        x_list_new = datetime.datetime(2016, x_month_list[i].month, x_month_list[i].day, x_month_list[i].hour, x_month_list[i].minute)
#        x_month_list_new.append(x_list_new)
#    else:
#        x_month_list_new.append(x_month_list[i])
#

#plt.close(1)
#fig = plt.figure(figsize=(10,14),dpi=80)
#ax1 = fig.add_subplot(5, 2, 1, title='Temporal varidation of the velocity/F:2', ylabel = 'Velocity(c)/F:2',ylim = (0,1), xlim = (datetime.datetime(2016, 1, 1, 0, 0),datetime.datetime(2016, 12, 30, 14, 32)))
#ax2 = fig.add_subplot(5, 2, 2, title='Temporal varidation of the velocity/app', ylabel = 'Velocity(c)/app',ylim = (0,1), xlim = (datetime.datetime(2016, 1, 1, 0, 0),datetime.datetime(2016, 12, 30, 14, 32)))
#ax3 = fig.add_subplot(5, 2, 5, title='Temporal varidation of the Frequency drift', ylabel = 'Frequency drift[MHz/s]', xlim = (datetime.datetime(2016, 1, 1, 0, 0),datetime.datetime(2016, 12, 30, 14, 32)))
#ax4 = fig.add_subplot(5, 2, 6, title='Temporal varidation of the Factor', ylabel = 'Factor', xlim = (datetime.datetime(2016, 1, 1, 0, 0),datetime.datetime(2016, 12, 30, 14, 32)))
##ax5 = fig.add_subplot(5, 1, 3, title='Occurrence rate of the velocity/F:2',xlabel = 'Velocity(c)/F:2', ylabel = 'Occurrence rate')

#############################################
##一年分のみ対応
#labelsize = 12
#ax1.plot(x_list_month, y_velocity_factor_2, '.', color = '0.8')
#for i in range(12):
#    if len(x_list_month[number[i]:number[i + 1]]) + len(x_list_month[number[i + 12]:number[i + 13]]) > 0:
#        if not i == 11:
#            ax1.plot(x_month_list_new[i:i+2], [(np.sum(y_velocity_factor_2[number[i]:number[i + 1]]) + np.sum(y_velocity_factor_2[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#            ax1.plot([x_month_list_new[i + 1]]*2, [(np.sum(y_velocity_factor_2[number[i]:number[i + 1]]) + np.sum(y_velocity_factor_2[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12]), (np.sum(y_velocity_factor_2[number[i+ 1]:number[i + 2]]) + np.sum(y_velocity_factor_2[number[i + 13]:number[i + 14]]))/(number[i + 2]-number[i + 1] + number[i + 14]-number[i + 13])], color = 'r')
#        else:
#            ax1.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_velocity_factor_2[number[i]:number[i + 1]]) + np.sum(y_velocity_factor_2[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#ax1.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_velocity_factor_2[number[i]:number[i + 1]]) + np.sum(y_velocity_factor_2[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r', label = 'Monthly average')
#xaxis_ = ax1.xaxis
#xaxis_.set_major_formatter(DateFormatter('%m-%d'))
#ax1.tick_params(axis='x', rotation=300, labelsize=labelsize)
#ax1.tick_params(axis='y', labelsize=labelsize)
#ax1.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#
#
#ax2.plot(x_list_month, y_velocity, '.', color = '0.8')
#for i in range(12):
#    if len(x_list_month[number[i]:number[i + 1]]) + len(x_list_month[number[i + 12]:number[i + 13]]) > 0:
#        if not i == 11:
#            ax2.plot(x_month_list_new[i:i+2], [(np.sum(y_velocity[number[i]:number[i + 1]]) + np.sum(y_velocity[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#            ax2.plot([x_month_list_new[i + 1]]*2, [(np.sum(y_velocity[number[i]:number[i + 1]]) + np.sum(y_velocity[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12]), (np.sum(y_velocity[number[i+ 1]:number[i + 2]]) + np.sum(y_velocity[number[i + 13]:number[i + 14]]))/(number[i + 2]-number[i + 1] + number[i + 14]-number[i + 13])], color = 'r')
#        else:
#            ax2.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_velocity[number[i]:number[i + 1]]) + np.sum(y_velocity[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#ax2.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_velocity[number[i]:number[i + 1]]) + np.sum(y_velocity[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r', label = 'Monthly average')
#xaxis_ = ax2.xaxis
#xaxis_.set_major_formatter(DateFormatter('%m-%d'))
#ax2.tick_params(axis='x', rotation=300, labelsize=labelsize)
#ax2.tick_params(axis='y', labelsize=labelsize)
#ax2.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#
#
#
#ax3.plot(x_list_month, y_freq_drift, '.', color = '0.8')
#for i in range(12):
#    if len(x_list_month[number[i]:number[i + 1]]) + len(x_list_month[number[i + 12]:number[i + 13]]) > 0:
#        if not i == 11:
#            ax3.plot(x_month_list_new[i:i+2], [(np.sum(y_freq_drift[number[i]:number[i + 1]]) + np.sum(y_freq_drift[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#            ax3.plot([x_month_list_new[i + 1]]*2, [(np.sum(y_freq_drift[number[i]:number[i + 1]]) + np.sum(y_freq_drift[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12]), (np.sum(y_freq_drift[number[i+ 1]:number[i + 2]]) + np.sum(y_freq_drift[number[i + 13]:number[i + 14]]))/(number[i + 2]-number[i + 1] + number[i + 14]-number[i + 13])], color = 'r')
#        else:
#            ax3.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_freq_drift[number[i]:number[i + 1]]) + np.sum(y_freq_drift[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#ax3.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_freq_drift[number[i]:number[i + 1]]) + np.sum(y_freq_drift[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r', label = 'Monthly average')
#xaxis_ = ax3.xaxis
#xaxis_.set_major_formatter(DateFormatter('%m-%d'))
#ax3.tick_params(axis='x', rotation=300, labelsize=labelsize)
#ax3.tick_params(axis='y', labelsize=labelsize)
#ax3.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#
#
#
#ax4.plot(x_list_month, y_factor, '.', color = '0.8')
#for i in range(12):
#    if len(x_list_month[number[i]:number[i + 1]]) + len(x_list_month[number[i + 12]:number[i + 13]]) > 0:
#        if not i == 11:
#            ax4.plot(x_month_list_new[i:i+2], [(np.sum(y_factor[number[i]:number[i + 1]]) + np.sum(y_factor[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#            ax4.plot([x_month_list_new[i + 1]]*2, [(np.sum(y_factor[number[i]:number[i + 1]]) + np.sum(y_factor[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12]), (np.sum(y_factor[number[i+ 1]:number[i + 2]]) + np.sum(y_factor[number[i + 13]:number[i + 14]]))/(number[i + 2]-number[i + 1] + number[i + 14]-number[i + 13])], color = 'r')
#        else:
#            ax4.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_factor[number[i]:number[i + 1]]) + np.sum(y_factor[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r')
#ax4.plot([datetime.datetime(2016, 12, 1, 0, 0), datetime.datetime(2017, 1, 1, 0, 0)], [(np.sum(y_factor[number[i]:number[i + 1]]) + np.sum(y_factor[number[i + 12]:number[i + 13]]))/(number[i + 1]-number[i] + number[i + 13]-number[i + 12])] * 2, color = 'r', label = 'Monthly average')
#xaxis_ = ax4.xaxis
#xaxis_.set_major_formatter(DateFormatter('%m-%d'))
#ax4.tick_params(axis='x', rotation=300, labelsize=labelsize)
#ax4.tick_params(axis='y', labelsize=labelsize)
#ax4.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
#if not os.path.isdir('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data'):
#    os.makedirs('/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data')
#filename = '/Volumes/HDPH-UT/lab/solar_burst/Nancay/jpgu_data/month.png'
#plt.savefig(filename)
#plt.show()