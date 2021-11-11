#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 15:15:24 2020

@author: yuichiro
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from pynverse import inversefunc

# factor = 2
# velocity = 0.3
# freq_max = 79.975
# t = np.arange(0, 200, 1)
# t = (t+1)/100
# freq = np.arange(29.95, 79.975, 0.175*4)
# cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
# invcube_4 = inversefunc(cube_4, y_values = freq_max)
# h_start = invcube_4/69600000000 + 1
# # print (h_start)
# y = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5))))
# cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
# invcube_3 = inversefunc(cube_3, y_values=freq)
# plt.plot(invcube_3, freq, '.')
# plt.xlim(0,10)
# plt.show()

# def allen_model_factor(factor_num, time_list, freq_list):
#     i_value = np.array(freq_list)
#     factor = factor_num
#     fitting = []
#     time_rate_result = []
#     fitting_new = []
#     time_rate_result_new = []
#     slide_result_new = []
#     freq_max = max(freq_list)
#     cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
#     invcube_4 = inversefunc(cube_4, y_values = freq_max)
#     h_start = invcube_4/696000 + 1
    
#     fitting.append(100)
#     time_rate_result.append(100)
    
#     for i in range(10, 100, 10):
#         time_rate3 = i/100
#         cube_3 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5)))))
#         invcube_3 = inversefunc(cube_3, y_values=i_value)
#         s_0 = sum(invcube_3-time_list)/len(freq_list)
#         residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
#         if min(fitting) > residual_0:
#             fitting.append(residual_0)
#             time_rate_result.append(time_rate3)
    
# #        print ('aaa')
# #        print(fitting)
#     fitting_new.append(100)
#     time_rate_result_new.append(100)
#     slide_result_new.append(100)
#     if int(time_rate_result[-1]*100) == 10:
#         time_new = 11
#     else:
#         time_new = int(time_rate_result[-1]*100)
#     for i in range(time_new -10, time_new + 10, 1):
#     #    print(i)
#         time_rate4 = i/100
#         cube_4 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
#         invcube_4 = inversefunc(cube_4, y_values=i_value)
#         s_1 = sum(invcube_4-time_list)/len(freq_list)
#         residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
#         if min(fitting_new) > residual_1:
#             fitting_new.append(residual_1)
#             time_rate_result_new.append(time_rate4)
#             slide_result_new.append(s_1)
#     print (h_start)
#     return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start)


import numpy as np
import cdflib
import glob

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_path = Parent_directory + '/solar_burst/Nancay/final_1.txt'
def file_generator(file):
    with open(file, encoding="utf-8") as f:
        for line in f:
            yield line

def final_txt_make(Parent_directory, Parent_lab, year, start, end):
    start = format(start, '04') 
    end = format(end, '04') 
    year = str(year)
    start = int(year + start)
    end = int(year + end)
    path = Parent_directory + '/solar_burst/Nancay/data/' + year + '/*/*'+'.cdf'
    File = glob.glob(path, recursive=True)
    File = sorted(File)
    i = open(Parent_directory + '/solar_burst/Nancay/final_1.txt', 'w')
    for cstr in File:
        a = cstr.split('/')
        line = a[Parent_lab + 6]+'\n'
        a1 = line.split('_')
        if (int(a1[5][:8])) >= start:
          if (int(a1[5][:8])) <= end:
            i.write(line)
    i.close()
    return start, end
start_date, end_date = final_txt_make(Parent_directory, Parent_lab, int(2010), 101, 1231)

gen = file_generator(Parent_directory + '/solar_burst/Nancay/final_1.txt')
for file in gen:
    file_name = file[:-1]
    file_name_separate =file_name.split('_')
    Date_start = file_name_separate[5]
    date_OBs=str(Date_start)
    year=date_OBs[0:4]
    month=date_OBs[4:6]
    file = Parent_directory + '/solar_burst/Nancay/data/'+year+'/'+month+'/'+file_name
    cdf_file = cdflib.CDF(file)
    Frequency = cdf_file['Frequency']
    print(Frequency[0], Frequency[-1])

