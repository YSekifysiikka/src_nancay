#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:52:25 2021

@author: yuichiro
"""
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

labelsize = 18
fontsize = 20
factor_velocity = 1
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm']
# Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1




file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/electrondensity_anaysis.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


# print (SDATE)
obs_time_all = []
dfdt_all = []
factor_all = []
residual_all = []
velocity_all = []
    
Burst_type = 'micro'

for j in range(len(csv_input_final)):
    obs_date = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
    burst_type = csv_input_final['burst_type'][j]
    if burst_type == Burst_type:
        dfdt_list = np.array([float(k) for k in csv_input_final["dfdt_list"][j][1:-1].replace(' ', '').split(',') if k != ''])
        factor_list = np.array([float(k) for k in csv_input_final["factor_list"][j][1:-1].replace(' ', '').split(',') if k != ''])
        residual_list = np.array([float(k) for k in csv_input_final["residual_list"][j][1:-1].replace(' ', '').split(',') if k != ''])
        velocity_list = np.array([float(k) for k in csv_input_final["velocity_list"][j][1:-1].replace(' ', '').split(',') if k != ''])
        if np.max(velocity_list) == 0.99 or np.min(velocity_list) == 0.01:
            pass
        else:

            obs_time_all.append(obs_date)
            dfdt_all.append(dfdt_list)
            factor_all.append(factor_list)
            residual_all.append(residual_list)
            velocity_all.append(velocity_list)

obs_time_all = np.array(obs_time_all)
dfdt_all = np.array(dfdt_all)
factor_all = np.array(factor_all)
residual_all = np.array(residual_all)
velocity_all = np.array(velocity_all)





x = np.array([0, 1, 2, 3, 4])
values = ['B-A(2fp)', 'Newkirk(fp)', 'Newkirk(2fp)', 'Wang_max(fp)','Wang_max(2fp)']

y = []
yerr = []
for i in range(5):
    y.append(np.mean(dfdt_all[:,0] - dfdt_all[:,i+1]))
    yerr.append(np.std(dfdt_all[:,0] - dfdt_all[:,i+1]))


fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=yerr, capsize=4, fmt='o', ecolor='red', color='black')
# ax.set_xlabel('x')
ax.set_ylabel('y')
plt.xticks(x,values, rotation = 10)
plt.title(Burst_type)
plt.show()
plt.close()


x = np.array([0, 1, 2, 3, 4, 5])
values = ['B-A(fp)', 'B-A(2fp)', 'Newkirk(fp)', 'Newkirk(2fp)', 'Wang_max(fp)','Wang_max(2fp)']
y = []
yerr = []
for i in range(6):
    y.append(np.mean(dfdt_all[:,i]))
    yerr.append(np.std(dfdt_all[:,i]))


fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=yerr, capsize=4, fmt='o', ecolor='red', color='black')
# ax.set_xlabel('x')
ax.set_ylabel('y')
plt.xticks(x,values, rotation = 10)
plt.title(Burst_type)
plt.show()
plt.close()

# density_model = ['B-A(2fp)', 'Newkirk(fp)', 'Newkirk(2fp)', 'Wang_max(fp)','Wang_max(2fp)']
# for i in range(5):
#     plt.title(Burst_type+' B-A(fp)-'+density_model[i])
#     plt.hist(dfdt_all[:,0] - dfdt_all[:,i+1])
#     plt.show()
#     plt.close()

density_model = ['B-A(2fp)', 'Newkirk(fp)', 'Newkirk(2fp)', 'Wang_max(fp)','Wang_max(2fp)']
for i in range(5):
    plt.title(Burst_type + ' B-A(fp)-'+density_model[i])
    plt.hist(residual_all[:,0] - residual_all[:,i+1])
    plt.show()
    plt.close()

