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

def numerical_diff_allen(factor, velocity, t, h_start):
    h = 1e-3
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))

#km
sun_to_earth = 150000000
sun_radius = 696000
light_v = 300000 #[km/s]
time_rate1 = 1.00
time_rate2 = 0.42

#allen_model = 10 * np.sqrt(factor * (2.99*(1+(h5/696000)*(time_rate3 * 300000))**(-16)+1.55*(1+(h5/696000)*(time_rate3 * 300000))**(-6)+0.036*(1+(h5/696000)*(time_rate3 * 300000))**(-1.5)))



factor_list = np.array([1, 3])
fig = plt.figure()
ax1 = fig.add_subplot(111)


for factor in factor_list:

    h2_1 = np.arange(1, 4, 0.1)
    newkirk = factor * 4.2 * 10 ** (4+4.32/h2_1)
    x2_1 = newkirk
    if factor == 3:
        ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×Newkirk model',color = 'r')
    elif factor == 1:
        ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×Newkirk model', color = 'b')
    plt.grid(which='both')
    freq=9*np.sqrt(newkirk)/10**3
    ax2=ax1.twinx()

    if factor == 3:
        ln1 = ax2.plot(h2_1,freq, color = 'r')
    elif factor == 1:
        ln1 = ax2.plot(h2_1,freq, color = 'b')

    allen_model = factor * 10**8 * (2.99*(h2_1)**(-16)+1.55*(h2_1)**(-6)+0.036*(h2_1)**(-1.5))
    x2_1 = allen_model
    if factor == 3:
        ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×B-A model',linestyle =  '--', color = 'r')
    elif factor == 1:
        ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×B-A model',linestyle =  '--', color = 'b')
    ax2.set_yscale('log')
    
    # h2_1 = np.arange(1, 4.2, 0.1)

    # plt.grid(which='both')
    # freq=9*np.sqrt(allen_model)/10**3
    # ax2=ax1.twinx()
    # if factor == 5:
    #     ln3=ax2.plot(h2_1,freq, color = 'r')
    # elif factor == 1:
    #     ln3=ax2.plot(h2_1,freq, color = 'b')
    # if factor == 5:
    #     ax1.axhline(19753086.419753086, ls = "--", color = "darkred", label = '40MHz', linewidth=5)
    #     # ax1.axhline(4938271.604938271, ls = "--", color = "tan", label = '20MHz', linewidth=5)
    ax2.set_yscale('log')
    ax1.set_yscale('log')
    ax1.set_ylim(2301606.3846679423, 2516800000.0)
    ax2.set_ylim(13.653941451394296, 451.5094683392586)
    ax1.set_xlim(1, 3)
    ax2.set_xlim(1, 3)






    # h2_1 = np.arange(1, 4, 0.1)
    # wang_min = 353766/h2_1 + 1.03359e+07/(h2_1)**2 - 5.46541e+07/(h2_1)**3 + 8.24791e+07/(h2_1)**4
    # x2_1 = wang_min 
    # ln1 = ax1.plot(h2_1, x2_1, label = 'Wang model(solar min)')

    # h2_1 = np.arange(1, 4, 0.1)
    # wang_max = -4.42158e+06/h2_1 + 5.41656e+07/(h2_1)**2 - 1.86150e+08 /(h2_1)**3 + 2.13102e+08/(h2_1)**4
    # x2_1 = wang_max
    # ln1 = ax1.plot(h2_1, x2_1, label = 'Wang model(solar max)')


    # ax1.set_yscale('log')
    # ax1.set_ylim(np.min(wang_min)*0.9 ,np.max(newkirk)*1.1)
    # ax2.set_ylim(9*np.sqrt(np.min(wang_min)*0.9)/10**3, 9*np.sqrt(np.max(newkirk)*1.1)/10**3)
    
plt.title("Coronal electron density model",fontsize=16)
ax1.legend(loc='upper right')
# ax2.legend(loc='upper right')
ax1.set_xlabel("Heliocentric distance [Rs]",fontsize=16)
ax2.set_ylabel("Frequency [MHz]",fontsize=16)
ax1.set_ylabel("Density [$/cc$]",fontsize=16)
plt.grid(which='both')
# plt.xlim(1,4)
plt.show()
plt.close()




factor_list = np.array([1])
fig = plt.figure()
ax1 = fig.add_subplot(111)


for factor in factor_list:

    
    factor = 3
    h2_1 = np.arange(1, 4, 0.1)
    newkirk = factor * 4.2 * 10 ** (4+4.32/h2_1)
    x2_1 = newkirk
    ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×Newkirk model', color = "orange")

    factor = 1
    h2_1 = np.arange(1, 4, 0.1)
    newkirk = factor * 4.2 * 10 ** (4+4.32/h2_1)
    x2_1 = newkirk
    ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×Newkirk model', color = "deepskyblue")
    # plt.grid(which='both')
    freq=9*np.sqrt(newkirk)/10**3
    ax2=ax1.twinx()
    ln3=ax2.plot(h2_1,freq, color = "deepskyblue")
    ax2.set_yscale('log')







    h2_1 = np.arange(1.5, 4, 0.1)
    wang_min = 353766/h2_1 + 1.03359e+07/(h2_1)**2 - 5.46541e+07/(h2_1)**3 + 8.24791e+07/(h2_1)**4
    x2_1 = wang_min 
    ln1 = ax1.plot(h2_1, x2_1, label = 'Wang model\nAround the solar minimum', color = "b")

    h2_1 = np.arange(1.5, 4, 0.1)
    h2_1 = 1.2
    wang_max = -4.42158e+06/h2_1 + 5.41656e+07/(h2_1)**2 - 1.86150e+08 /(h2_1)**3 + 2.13102e+08/(h2_1)**4
    x2_1 = wang_max
    ln1 = ax1.plot(h2_1, x2_1, label = 'Wang model\nAround the solar maximum', color = "r")

    h2_1 = np.arange(1, 1.6, 0.05)
    wang_min_1 = 353766/h2_1 + 1.03359e+07/(h2_1)**2 - 5.46541e+07/(h2_1)**3 + 8.24791e+07/(h2_1)**4
    x2_1 = wang_min_1
    ln1 = ax1.plot(h2_1, x2_1, ls = '--', color = "b")

    h2_1 = np.arange(1, 1.6, 0.05)
    wang_max_1 = -4.42158e+06/h2_1 + 5.41656e+07/(h2_1)**2 - 1.86150e+08 /(h2_1)**3 + 2.13102e+08/(h2_1)**4
    x2_1 = wang_max_1
    ln1 = ax1.plot(h2_1, x2_1, ls = '--', color = "r")
    
    h2_1 = np.arange(1, 4, 0.1)
    Leblanc = 3.3e+05/(h2_1)**2 + 4.1e+06 /(h2_1)**4 + 8.0e+07/(h2_1)**6
    x2_1 = Leblanc
    ln1 = ax1.plot(h2_1, x2_1, label = 'Leblanc', color = "k")


    ax1.set_yscale('log')
    ax1.set_ylim(np.min(wang_min)*0.8 ,np.max(newkirk)*1.2)
    ax2.set_ylim(9*np.sqrt(np.min(wang_min)*0.8)/10**3, 9*np.sqrt(np.max(newkirk)*1.2)/10**3)
    ax1.axhline(19753086.419753086, ls = "--", color = "darkred", label = '40MHz', linewidth=5, alpha = 0.4)
    ax1.axhline(4938271.604938271, ls = "--", color = "tan", label = '20MHz', linewidth=5, alpha = 0.7)
    
plt.title("Coronal electron density model",fontsize=16)
ax1.legend(loc='upper right', fontsize = 8)
# ax2.legend(loc='upper right')
ax1.set_xlabel("Heliocentric distance [Rs]",fontsize=16)
ax2.set_ylabel("Frequency [MHz]",fontsize=16)
ax1.set_ylabel("Density [$/cc$]",fontsize=16)
plt.grid(which='both')
plt.xlim(1,3.7)
plt.show()
plt.close()





# # for factor in factor_list:
# #     fig = plt.figure()
# #     ax1 = fig.add_subplot(111)
# #     t = h2_1 = np.arange(0, 1000000, 100)
# #     (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
# #     allen_model = factor * 10**8 * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))
# #     y1 = allen_model
# #     ln1=ax1.plot(t, y1,'C0',label=r'$y=sin(2\pi fst)$')
    
# #     ax2 = ax1.twinx()
# #     y2 = 9 * np.sqrt(allen_model)/1000
# #     ln2=ax2.plot(t,y2,'C1',label=r'$y=at+b$')
    
# #     h1, l1 = ax1.get_legend_handles_labels()
# #     h2, l2 = ax2.get_legend_handles_labels()
# #     ax1.legend(h1+h2, l1+l2, loc='lower right')
# #     ax1.set_yscale('log')
# #     ax1.set_xlabel('t')
# #     ax1.set_ylabel(r'$y=sin(2\pi fst)$')
# #     ax1.grid(True)
# #     ax2.set_ylabel(r'$y=at+b$')
# #     plt.xlim(0,1000000)
# # plt.show()
# # plt.close()



# # fig = plt.figure()
# # ax1 = fig.add_subplot(111)
# # for factor in factor_list:
    
# #     h2_1 = np.arange(0, 1000000, 100)
# #     allen_model = factor * 10**8 * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))
# #     x2_1 = allen_model
# #     ln1 = ax1.plot(h2_1, x2_1, label = str(factor) + '×B-A model')
# #     ax1.set_yscale('log')
# #     plt.title("Coronal electron density model",fontsize=16)
# # ax2 = ax1.twinx()
# # for factor in factor_list:
# #     h2_1 = np.arange(0, 1000000, 100)
# #     allen_model = factor * 10**8 * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))
# #     # x2_1 = 9 * np.sqrt(allen_model)/1000
# #     x2_1 = allen_model
# #     ln2 = ax2.plot(h2_1, x2_1)
# #     ax2.set_yscale('log')

# # h1, l1 = ax1.get_legend_handles_labels()
# # h2, l2 = ax2.get_legend_handles_labels()
# # ax1.legend(h1+h2, l1+l2, loc='upper right')

# # ax1.set_xlabel("Distance from sun surface[Rs]",fontsize=16)
# # ax1.set_ylabel("Density[$cm^{{-}3}$]",fontsize=16)
# # ax1.grid(True)
# # ax2.set_ylabel("Frequency[MHz]",fontsize=16)
# # ax2.set_yticks([1111111111, 493827160, 123456790,30864197, 11111111])
# # ax2.set_yticklabels(['300','200','100', '50', '30']) 

# # # plt.yscale('log')
# # # plt.axvline(696340, ls = "--", color = "navy", label = 'Solar radius')
# # # plt.xlim(0, 1000000)
# # # plt.ylim(0, 2500000000)
# # # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1) 
# # plt.xticks([0,348170,696340,1044510], ['0','0.5','1', '1.5']) 
# # plt.tick_params(labelsize=14)
# # plt.show()
# # plt.close()


# # for factor in factor_list:
    
# #     h2_1 = np.arange(0, 1000000, 100)
# #     allen_model = factor * 10**8 * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))
# #     x2_1 = allen_model
# #     plt.plot(h2_1, x2_1, label = str(factor) + '×B-A model')
# #     plt.title("Coronal electron density model",fontsize=16)
# #     plt.xlabel("Distance from sun surface[km]",fontsize=16)
# #     plt.ylabel("Density[$cm^{{-}3}$]",fontsize=16)
# # #plt.show()
# # #plt.close()
# # plt.yscale('log')
# # # plt.axvline(696340, ls = "--", color = "navy", label = 'Solar radius')
# # plt.xlim(0, 1000000)
# # plt.ylim(0, 2500000000)
# # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1) 
# # plt.tick_params(labelsize=14)
# # plt.show()
# # plt.close()


# # for factor in factor_list:
# #     h2_1 = np.arange(0, 1000000, 100)
# #     allen_model = factor * 10**8 * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))
# #     x2_1 = 9 * np.sqrt(allen_model)/1000
# #     plt.plot(h2_1, x2_1, label = str(factor) + '×B-A model')
# #     plt.title("Coronal electron density model",fontsize=16)
# #     plt.xlabel("Distance from sun surface[km]",fontsize=16)
# #     plt.ylabel("Frequency[MHz]",fontsize=16)
# # plt.axvline(696340, ls = "--", color = "navy", label = 'Solar radius')
# # plt.xlim(0, 1000000)
# # plt.ylim(0, 200)
# # #plt.legend() 
# # plt.tick_params(labelsize=14)
# # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
# # plt.show()
# # plt.close()


# # #4_times
# for f in range(3):
#     factor = 2 * f + 1
#     h2_1 = np.arange(0, 83520000, 100)
#     allen_model = 10 * np.sqrt(factor * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5)))
#     x2_1 = 9 * allen_model
#     plt.plot(h2_1, x2_1, label = str(factor) + '×B-A model')
#     plt.title("Coronal electron density model",fontsize=16)
#     plt.xlabel("Distance from sun surface[km]",fontsize=16)
#     plt.ylabel("Frequency[MHz]",fontsize=16)
# plt.xlim(0, 1000000)
# plt.ylim(0, 500)
# plt.legend()
# plt.show()
# plt.close()

# # # plt.axvline(696340, ls = "--", color = "navy", label = 'Solar radius')
# # # plt.xlim(0, 1000000)
# # # plt.ylim(0, 500)
# # # plt.legend()
# # # plt.show()
# # # plt.close()

# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.dates as mdates
# # import pandas as pd
# # import scipy.io as sio
# # from scipy.optimize import curve_fit
# # import sys
# # from pynverse import inversefunc

# # # def allen_model(factor_num, time_list, freq_list):
# # #     i_value = np.array(freq_list)
# # #     factor = factor_num
# # #     fitting = []
# # #     time_rate_result = []
# # #     fitting_new = []
# # #     time_rate_result_new = []
# # #     slide_result_new = []
# # #     freq_max = max(freq_list)
# # #     cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
# # #     invcube_4 = inversefunc(cube_4, y_values = freq_max)
# # #     h_start = invcube_4/696000 + 1
    
# # #     fitting.append(100000000000000)
# # #     time_rate_result.append(100)
    
# # #     for i in range(10, 100, 10):
# # #         time_rate3 = i/100
# # #         cube_3 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5)))))
# # #         invcube_3 = inversefunc(cube_3, y_values=i_value)
# # #         s_0 = sum(invcube_3-time_list)/len(freq_list)
# # #         residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
# # #         if min(fitting) > residual_0:
# # #             fitting.append(residual_0)
# # #             time_rate_result.append(time_rate3)
    
# # # #        print ('aaa')
# # # #        print(fitting)
# # #     fitting_new.append(100000000000000)
# # #     time_rate_result_new.append(100)
# # #     slide_result_new.append(100)
# # #     if int(time_rate_result[-1]*100) == 10:
# # #         time_new = 11
# # #     else:
# # #         time_new = int(time_rate_result[-1]*100)
# # #     for i in range(time_new -10, time_new + 10, 1):
# # #     #    print(i)
# # #         time_rate4 = i/100
# # #         cube_4 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
# # #         invcube_4 = inversefunc(cube_4, y_values=i_value)
# # #         s_1 = sum(invcube_4-time_list)/len(freq_list)
# # #         residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
# # #         if min(fitting_new) > residual_1:
# # #             fitting_new.append(residual_1)
# # #             time_rate_result_new.append(time_rate4)
# # #             slide_result_new.append(s_1)
# # #     print (h_start)
# # # #        print ('aaa')
# # # #        print (fitting_new[-1])
# # #     return (-slide_result_new[-1], time_rate_result_new[-1], fitting_new[-1], h_start)





# # # class read_waves:
    
# # #     def __init__(self, **kwargs):
# # #         self.date_in  = kwargs['date_in']
# # #         self.HH = kwargs['HH']
# # #         self.MM = kwargs['MM']
# # #         self.SS = kwargs['SS']
# # #         self.duration = kwargs['duration']
# # #         self.directry = kwargs['directry']
        
# # #         self.str_date = str(self.date_in)
# # #         self.time_axis = pd.date_range(start=self.str_date, periods=1440, freq='T')
# # #         self.yyyy = self.str_date[0:4]
# # #         self.mm   = self.str_date[4:6]
# # #         self.dd   = self.str_date[6:8]
        
# # #         start = pd.to_datetime(self.str_date + self.HH + self.MM + self.SS,
# # #                                format='%Y%m%d%H%M%S')
# # #         end = start + pd.to_timedelta(self.duration, unit='min')
# # #         self.time_range = [start, end]
        
# # #         return
    
# # #     def tlimit(self, df):
# # #         if type(df) != pd.core.frame.DataFrame:
# # #             print('tlimit \n Type error: data type must be DataFrame')
# # #             sys.exit()
        
        
        
# # #         tl_df =    df[   df.index >= self.time_range[0]]
# # #         tl_df = tl_df[tl_df.index  < self.time_range[1]]
        
# # #         return tl_df
    
    
# # #     def read_rad(self, receiver):
# # #         if type(receiver) != str:
# # #             print('read_rad \n Keyword error: reciever must be a string')
# # #             sys.exit()
        
# # #         if receiver == 'rad1':
# # #             extension = '.R1'
# # #         elif receiver == 'rad2':
# # #             extension = '.R2'
# # #         else:
# # #             print('read_rad \n Name error: receiver name')
# # #             sys.exit()
# # #         file_path = self.directry + self.str_date + extension
# # #         sav_data = sio.readsav(file_path)
# # #         data = sav_data['arrayb'][:, 0:1440]
# # #         BG   = sav_data['arrayb'][:, 1440]
        
# # #         rad_data = np.where(data==0, np.nan, data)
# # #         rad_data = rad_data.T
# # #         rad = pd.DataFrame(rad_data)
        
# # #         rad.index = self.time_axis
# # #         rad = self.tlimit(rad)
# # #         return rad
    
# # #     def read_waves(self):
# # #         rad1 = self.read_rad('rad1')
# # #         rad2 = self.read_rad('rad2')
# # #         waves = pd.concat([rad1, rad2], axis=1)
        
# # #         return waves

# # # def waves_peak_finder(data):
# # #     if type(data) != pd.core.frame.DataFrame:
# # #         print('waves_peak_finder \n Type error: data type must be DataFrame')
# # #         sys.exit()
# # #     data = data.reset_index(drop=True)
# # #     peak = data.max(axis=0)
# # #     idx  = data.idxmax(axis=0)
# # #     result = pd.concat([idx, peak], axis=1)
# # #     result.columns = ['index', 'peak']
# # #     return result

# # # def freq_setting(receiver):
# # #     if receiver == 'rad1':
# # #         freq = 0.02 + 0.004*np.arange(256)
# # #     elif receiver == 'rad2':
# # #         freq = 1.075 + 0.05*np.arange(256)
# # #     elif receiver == 'waves':
# # #         freq1 = 0.02 + 0.004*np.arange(256)
# # #         freq2 = 1.075 + 0.05*np.arange(256)
# # #         freq  = np.hstack([freq1, freq2])
# # #     else:
# # #         print('freq_setting \n Name error: receiver name')
# # #     return freq

# # # def linear_fit(data, receiver='rad1', freq_band=[0.02, 1.04],
# # #                p0=[0,0], bounds = ([-np.inf, -np.inf], [np.inf, np.inf])):
    
# # #     def linear_func(x, a, b):
# # #         return a*x + b
    
# # #     peak_data = waves_peak_finder(data)
# # #     index = peak_data['index']
# # #     peak  = peak_data['peak']
    
# # #     freq = freq_setting(receiver)
# # #     freq = pd.DataFrame(freq)
    
# # #     cat_data = pd.concat([peak_data,freq], axis=1)
# # #     cat_data.columns = ['index', 'peak', 'freq']
    
# # #     flimit_df =  cat_data[ cat_data['freq'] >= freq_band[0]]
# # #     flimit_df = flimit_df[flimit_df['freq'] <= freq_band[1]]
    
# # #     l_index = flimit_df['index'].values
# # #     l_peak  = flimit_df['peak'].values
# # #     l_freq = flimit_df['freq'].values
    
# # #     if len(l_index) == 0:
# # #         print('linear_fit \n Value error: freq_band range are illegal values for fitting')
# # #         sys.exit()
    
# # #     x = []
# # #     y = []
    
# # #     for i in range(len(l_index)):
# # #         if np.isnan(l_peak[i]) != True:
# # #             x.append(l_index[i])
# # #             y.append(l_freq[i])
    
# # #     popt, pcov = curve_fit(linear_func,x, y, p0=p0, bounds=bounds)
# # #     error = np.sqrt(np.diag(pcov))


# # #     plt.figure()




# # #     plt.plot(cat_data['index'][:128]*60 - 1400, cat_data['freq'][:128], 'r.', markersize=1)
# # #     print ('aaa' + str(min(cat_data['index']*60 - 1300)))
# # #     print ('bbb' + str(max(cat_data['freq'])))
# # #     plt.plot(index * 60, linear_func(index * 60, popt[0], popt[1]))
# # #     plt.axhline(freq_band[0], xmin=0, xmax=1, color='blue', linestyle='dashed')
# # #     plt.axhline(freq_band[1], xmin=0, xmax=1, color='blue', linestyle='dashed')
# # #     plt.ylim(freq.iloc[0][0], freq.iloc[-1][0])
# # #     plt.xlabel('Time [sec]')
# # #     plt.ylabel('Frequency [MHz]')
# # #     # plt.xlim(-2000, 3000)
# # #     return popt, error


# # # def radio_plot(data, receiver='rad1'):
# # #     if type(data) != pd.core.frame.DataFrame:
# # #         print('radio_plot \n Type error: data type must be DataFrame')
# # #         sys.exit()
    
# # #     p_data = 20*np.log10(data)
# # #     vmin = 0
# # #     vmax = 20
    
# # #     freq_axis = freq_setting(receiver)
# # #     y_lim = [freq_axis[0], freq_axis[-1]]
    
# # #     time_axis = data.index
# # #     x_lim = [time_axis[0], time_axis[-1]]
# # #     x_lim = mdates.date2num(x_lim)
    
# # #     fig = plt.figure(figsize=[8,6])
    
# # #     axes = fig.add_subplot(111)
# # #     axes.imshow(p_data.T, origin='lower', aspect='auto',
# # #                 extent=[x_lim[0],x_lim[1],y_lim[0],y_lim[1]],
# # #                 vmin=vmin, vmax=vmax)
# # #     axes.xaxis_date()
# # #     axes.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# # #     axes.tick_params(labelsize=10)
# # #     axes.set_xlabel('Time [UT]')
# # #     axes.set_ylabel('Frequency [MHz]')
    
# # #     return axes



# # # if __name__ == '__main__':
# # #     plt.close('all')
    
# # #     waves_setting = {'receiver'   : 'rad1',
# # #                      'date_in'    : 20170228,#yyyymmdd
# # #                      'HH'         : '13',#hour
# # #                      'MM'         : '30',#minute
# # #                      'SS'         : '00',#second
# # #                      'duration'   :   60,#min
# # #                      'freq_band'  :  [0.3, 0.7],
# # #                      'init_param' : [0, 0],
# # #                      'bounds'     : ([-np.inf, -np.inf], [np.inf, np.inf]),
# # #                      'directry'   : r'/Users/yuichiro/Downloads/',
# # #                      }
    
# # #     rw = read_waves(**waves_setting)
    
# # #     receiver = waves_setting['receiver']
# # #     data = rw.read_rad(receiver)
# # #     axes = radio_plot(data, receiver=receiver)
    
    
# # #     freq_band = waves_setting['freq_band']
# # #     p0        = waves_setting['init_param']
# # #     bounds    = waves_setting['bounds']
# # #     popt, error = linear_fit(data, receiver=receiver, freq_band=freq_band, 
# # #                              p0=p0, bounds=bounds)
# # #     print('Drift rate: %f [MHz/min]' %popt[0])




# # freq_list = np.arange(1, 104,0.175)
# # freq_list = freq_list/100
# # freq = []
# # start_check = []
# # y_factor = []
# # y_velocity = []
# # t = np.arange(0, 2000000, 1)
# # t = (t+1)/100-5
# # separate_num = 3
# # for j in range(separate_num):
# #     freq.append((max(freq_list) - min(freq_list))*((j + 1)/(separate_num+1)) + min(freq_list))
# #     start_check.append(max(freq_list))
# #     y_factor.append(0.01)
# #     y_velocity.append(float(0.21))
# # for i in range(3):
# #     factor = y_factor[i]
# #     velocity = y_velocity[i]
# #     freq_max = start_check[i]
# #     cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
# #     invcube_4 = inversefunc(cube_4, y_values = freq_max)
# #     h_start = invcube_4/69600000000 + 1
# #     y = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 30000000000)/69600000000)**(-16)) + 1.55 * ((h_start + (t * velocity * 30000000000)/69600000000)**(-6)) + 0.036 * ((h_start + (t * velocity * 30000000000)/69600000000)**(-1.5))))
# #     cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 30000000000)/69600000000)**(-16)) + 1.55 * ((h_start + (t * velocity * 30000000000)/69600000000)**(-6)) + 0.036 * ((h_start + (t * velocity * 30000000000)/69600000000)**(-1.5)))))
# #     invcube_3 = inversefunc(cube_3, y_values=freq[i])

# #     slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
# #     y_slope = slope * (t - invcube_3) + freq[i]

# #     print (slope)
# #     print (invcube_3)
# # #    print (freq_mean[i])
# #     # plt.ylim(min(freq_list), max(freq_list))
# #     plt.xlim(-2, 3000)
# #     # plt.plot(t, y_slope, label = str(i+1) + '/4  ' + str(round(freq[i]*10)/10) + '[MHz]  ' + 'Df=' + str(round(slope*10)/10) + '[MHz/s]')
# # plt.plot(t, y, '-', label = 'type Ⅲ burst', linewidth = 2.0)
# # plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
# # plt.tick_params(labelsize=15)
# # plt.xlabel('Time [sec]',fontsize=15)
# # plt.ylabel('Frequency drift rate[MHz/s]',fontsize=15)
# # plt.show()
# # plt.close()



# #freq_max =80
# #time_rate3 = 0.20
# # time_rate3_list = [0.40]
# # for factor in factor_list:
# #     for time_rate3 in time_rate3_list:
# #         cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
# #         invcube_4 = inversefunc(cube_4, y_values = freq_max)
# #         h_start = invcube_4/69600000000 + 1
# #         h5 = np.arange(0, 100000, 1)
# #         h5 = (h5+1)/1000
# #         x5 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5))))
# #         plt.plot(h5, x5, '-', label = str(factor) + '×B-A model/v=' + str(time_rate3) + 'c')
# # #plt.axvline(7.733333333333333, ls = "--", color = "navy", label = 'Solar radius')
# # plt.legend(fontsize=12)
# # plt.tick_params(labelsize=14)
# # plt.ylim(30, 80)
# # plt.xlim(-1, 15)
# # plt.title("Simulations of coronal type III solar radio bursts",fontsize=15)
# # plt.xlabel("Time[sec]",fontsize=16)
# # plt.ylabel("Frequency[MHz]",fontsize=16)
# # plt.show()
# # plt.close()


# fig = plt.figure(dpi=100, figsize=(7,3))
# factor = 5
# for velo in range(2):
#     velocity = (2 * velo + 2)/10
#     h5 = np.arange(0, 10000, 1)
#     h5 = (h5+1)/100
#     allen_model = 10 * np.sqrt(factor * (2.99*(1+(h5/696000)*(velocity * 300000))**(-16)+1.55*(1+(h5/696000)*(velocity * 300000))**(-6)+0.036*(1+(h5/696000)*(velocity * 300000))**(-1.5)))
#     x5 = 9 * allen_model
#     h5 = h5
#     x5 = x5
#     # plt.plot(h5, x5, '-', label = 'v=' + str(velocity) + 'c')
#     if velocity == 0.2:
#         slope = 'Small'
#     else:
#         slope = 'Large'
#     plt.plot(h5, x5, '-', label = slope + ' frequency drift rates')
#     plt.legend(fontsize=12)
# # plt.axhline(38.706433335383885, ls = "--", color = "navy", label = 'Solar radius')
# plt.legend(fontsize=14)
# plt.tick_params(labelsize=14)
# plt.ylim(30, 80)
# plt.xlim(0, 25)
# # plt.title("Simulations of coronal type III solar radio bursts:"+ str(factor) +"×B-A model",fontsize=15)
# plt.title("Simulations of coronal type III solar radio bursts")
# plt.xlabel("Time[sec]",fontsize=16)
# plt.ylabel("Frequency[MHz]",fontsize=16)
# plt.show()
# plt.close()


# # # # #
# # # # factor = 3
# # # # h2_1 = np.arange(0, 83520000, 100)
# # # # x2_1 = factor * 9 * 10 * np.sqrt((2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5)))
# # # # plt.plot(h2_1, x2_1, label = str(factor) + '×B-A model')
# # # # plt.title("Coronal electron density model",fontsize=16)
# # # # plt.xlabel("Distance from sun surface[km]",fontsize=16)
# # # # plt.ylabel("Frequency[MHz]",fontsize=16)
# # # # #plt.show()
# # # # #plt.close()

# # # # plt.axvline(696340, ls = "--", color = "navy", label = 'Solar radius')
# # # # plt.xlim(0, 1000000)
# # # # plt.ylim(0, 900)
# # # # plt.legend() 
# # # # plt.show()
# # # # plt.close()




        




# # # ###############################
# # # # distance = []
# # # # i = 30
# # # # cube_1 = (lambda h2:9 * 10 * np.sqrt((2.99*(1+(h2/696000))**(-16)+1.55*(1+(h2/696000))**(-6)+0.036*(1+(h2/696000))**(-1.5))))
# # # # invcube_1 = inversefunc(cube_1, y_values=i)
# # # # print (str(i) + ':' + str(invcube_1))
# # # # distance.append(invcube_1)

# # # # cube_2 = (lambda h2_1:4 * 9 * 10 * np.sqrt((2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))))
# # # # invcube_2 = inversefunc(cube_2, y_values=i)
# # # # print (str(i) + ':' + str(invcube_2))
# # # # distance.append(invcube_2)
# # # # #for i in range((len(distance)-1)):
# # # # #    print ((distance[i] - distance[i+1])/(300000*0.16))
# # # # i = 80
# # # # cube_3 = (lambda h5: 1 * 9 * 10 * np.sqrt((2.99*(1+(h5/696000)*(time_rate3 * 300000))**(-16)+1.55*(1+(h5/696000)*(time_rate3 * 300000))**(-6)+0.036*(1+(h5/696000)*(time_rate3 * 300000))**(-1.5))))
# # # # invcube_3 = inversefunc(cube_3, y_values=i)
# # # # print (str(i) + ':' + str(invcube_3))

# # # # j = 30
# # # # cube_4 = (lambda h5: 1 * 9 * 10 * np.sqrt((2.99*(1+(h5/696000)*(time_rate3 * 300000))**(-16)+1.55*(1+(h5/696000)*(time_rate3 * 300000))**(-6)+0.036*(1+(h5/696000)*(time_rate3 * 300000))**(-1.5))))
# # # # invcube_4 = inversefunc(cube_4, y_values=j)
# # # # print (str(j) + ':' + str(invcube_4))
# # # # print ((i-j)/(invcube_4-invcube_3))