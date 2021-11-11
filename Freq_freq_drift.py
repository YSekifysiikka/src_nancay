#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 15:00:47 2021

@author: yuichiro
"""

#周波数ドリフト率と速度の関係解析Try結果（放射時間のズレの考慮が難しく途中リタイア）

import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc
import matplotlib.gridspec as gridspec

fontsize= 12
c = 299792.458 #[km/s]
Rs = 696000 #[km]
factor_list = np.array([1, 2, 3, 4, 5])
velocity_list = np.arange(0.05, 0.55, 0.05)
Freq_list = np.arange(30, 80, 0.175)
# check_frequency = 30


def numerical_diff_allen_velocity(factor, r):
    h = 1e-2
    ne_1 = factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8
    ne_2 = factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8
    return ((ne_1 - ne_2)/(2*h))

def numerical_diff_allen_velocity_log(factor, r):
    # print (factor)
    h = 1e-2
    ne_1 = np.log(factor * (2.99*((r+h)/69600000000)**(-16)+1.55*((r+h)/69600000000)**(-6)+0.036*((r+h)/69600000000)**(-1.5))*1e8)
    ne_2 = np.log(factor * (2.99*((r-h)/69600000000)**(-16)+1.55*((r-h)/69600000000)**(-6)+0.036*((r-h)/69600000000)**(-1.5))*1e8)
    return ((ne_1 - ne_2)/(2*h))

def allen_model(factor, r):
    ne = factor * (2.99*((r/69600000000)**(-16))+1.55*((r/69600000000)**(-6))+0.036*((r/69600000000)**(-1.5)))*1e8
    return ne
def func(f, a, b):
    return a * (f ** b)


for factor in factor_list:
    for r_velocity in velocity_list:
        df_dt_list = []
        for check_frequency in Freq_list:
            cube_1 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/Rs))**(-16))+1.55*((1+(h/Rs))**(-6))+0.036*((1+(h/Rs))**(-1.5)))))
            invcube_1 = inversefunc(cube_1, y_values = check_frequency)
            h_start = (invcube_1 + Rs)* 100000
            # print (h_start)
            dne_dr = numerical_diff_allen_velocity(factor, h_start)
            sqrtne = np.sqrt(allen_model(factor, h_start))
            df_dt = 9/2/sqrtne * dne_dr * r_velocity*c*100 #[MHz/s]
            # print(-df_dt)
            df_dt_list.append(-df_dt)
        plt.plot(Freq_list, df_dt_list, '-', label = 'V: ' + str(round(r_velocity, 2)) + 'c')
    y = func(Freq_list,0.00464,1.929)
    plt.plot(Freq_list, y, '--', color = 'r', label = '2012', linewidth = 3.0)
    y = func(Freq_list,0.02024,1.455)
    plt.plot(Freq_list, y, '--', color = 'k', label = '2019', linewidth = 3.0)
    y = func(Freq_list,0.006186,1.80984)
    plt.plot(Freq_list, y, '--', color = 'b', label = '2020', linewidth = 3.0)
    plt.legend(fontsize=6)
    plt.xlim(30, 60)
    plt.ylim(0,40)
    plt.title("Frequency drift rate vs Frequency drift Factor: " + str(factor),fontsize=15)
    plt.xlabel('Velocity (c)',fontsize=fontsize)
    plt.ylabel('Frequency drift rate [MHz]',fontsize=fontsize)
    plt.show()
    plt.close()
    





# for factor in factor_list:
#     print (factor)
#     df_dt_list = []
#     for r_velocity in velocity_list:
#         cube_1 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/Rs))**(-16))+1.55*((1+(h/Rs))**(-6))+0.036*((1+(h/Rs))**(-1.5)))))
#         invcube_1 = inversefunc(cube_1, y_values = check_frequency)
#         h_start = (invcube_1 + Rs)* 100000
#         # print (h_start)
#         # dne_dr = numerical_diff_allen_velocity(factor, h_start)
#         # sqrtne = np.sqrt(allen_model(factor, h_start))
#         df_dt = check_frequency/2*numerical_diff_allen_velocity_log(factor, h_start)* r_velocity*c*100000#[MHz/s]
#         print(-df_dt)
#         df_dt_list.append(-df_dt)
#     plt.plot(velocity_list, df_dt_list, '-', label = 'Factor:' + str(factor))
# plt.legend(fontsize=12)
# plt.axvline(x=0.3)
# plt.ylim(0,40)
# plt.title("Frequency drift rate vs Frequency drift",fontsize=15)
# plt.xlabel('Velocity (c)',fontsize=fontsize)
# plt.ylabel('Frequency drift rate [MHz] @' + str(check_frequency) + 'MHz',fontsize=fontsize)
# plt.show()
# plt.close()


# def numerical_diff_allen(factor, velocity, t, h_start):
#     h = 1e-4
#     f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * c)/Rs)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * c)/Rs)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * c)/Rs)**(-1.5))))
#     f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * c)/Rs)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity *c)/Rs)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * c)/Rs)**(-1.5))))
#     return ((f_1 - f_2)/(2*h))




# factor_list = np.array([1, 2, 3, 4, 5])
# velocity_list = np.arange(0.05, 1.0, 0.1)
# freq_range = np.arange(30,30.0175,0.175)
# # for factor in factor_list:
# #     for r_velocity in velocity_list:
# #         h2_1 = np.arange(0, 1000000, 100)
# #         allen_model = factor * 10**8 * (2.99*(1+(h2_1/696000))**(-16)+1.55*(1+(h2_1/696000))**(-6)+0.036*(1+(h2_1/696000))**(-1.5))
# #         plt.plot(h2_1, allen_model)
# #         plt.show()
# #         plt.close()
# for r_velocity in velocity_list:
#     figure_=plt.figure(1,figsize=(15,10))
#     gs = gridspec.GridSpec(15, 10)
#     ax_1 = figure_.add_subplot(gs[:7, :])
#     ax_2 = figure_.add_subplot(gs[8:, :])
#     for factor in factor_list:
#         freq_drift = []
#         h5 = np.arange(0, 10000, 1)
#         h5 = (h5+1)/100
#         # (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
#         allen_model = 10 * np.sqrt(factor * (2.99*(1+(h5/696000)*(r_velocity*c))**(-16)+1.55*(1+(h5/696000)*(r_velocity*c))**(-6)+0.036*(1+(h5/696000)*(r_velocity*c))**(-1.5)))
#         x5 = 9 * allen_model
#         ax_1.plot(h5, x5, '-', label = 'Factor=' + str(factor))
#         ax_1.legend(fontsize=12)
#         ax_1.set_ylim(30, 80)
#         ax_1.set_xlim(0, 40)
#         cube_1 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/Rs))**(-16))+1.55*((1+(h/Rs))**(-6))+0.036*((1+(h/Rs))**(-1.5)))))
#         invcube_1 = inversefunc(cube_1, y_values = 30)
#         h_start = invcube_1/Rs + 1


#         cube_2 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * r_velocity * c)/Rs)**(-16)) + 1.55 * ((h_start + (t * r_velocity * c)/Rs)**(-6)) + 0.036 * ((h_start + (t * r_velocity * c)/Rs)**(-1.5)))))
#         for freq in freq_range:
#             invcube_2 = inversefunc(cube_2, y_values=freq)
#             slope = numerical_diff_allen(factor, r_velocity, invcube_2, h_start)
#             freq_drift.append(-slope)
#             print (-slope)
#         ax_2.plot(factor, freq_drift, '.')
#         ax_2.set_ylim(0, 10)
#         ax_2.set_xlim(0, 6)

            
#     # plt.axhline(38.706433335383885, ls = "--", color = "navy", label = 'Solar radius')
#     # plt.legend(fontsize=12)
#     plt.tick_params(labelsize=14)
#     # plt.ylim(30, 80)
#     # plt.xlim(0, 30)
#     plt.title("Simulations of coronal type III solar radio bursts:"+ str(r_velocity) +"×c",fontsize=15)
#     plt.xlabel("Time[sec]",fontsize=16)
#     plt.ylabel("Frequency[MHz]",fontsize=16)
#     plt.show()
#     plt.close()




# #     axes_1 = figure_.add_subplot(gs[:8, :])

# #     plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
# #     plt.xlabel('Time (UT)',fontsize=fontsize )
# #     plt.ylabel('Frequency [MHz]',fontsize=fontsize)



# #     axes_1 = figure_.add_subplot(gs[10:, :])
# #     ax1 = axes_1.imshow(diff_db_plot_sep, extent = [x_lims[0], x_lims[1],  y_lims[0], y_lims[1]], 
# #               aspect='auto',cmap='jet',vmin= 0,vmax = quartile_db_l[db_standard] + 5)
# #     plt.title('Nancay: '+year+'-'+month+'-'+day+' '+str(Time_start)+'-'+str(Time_end),fontsize=fontsize)
# #     plt.xlabel('Time (UT)',fontsize=fontsize)
# #     plt.ylabel('Frequency [MHz]',fontsize=fontsize)

# # #                                                                                axes_2.plot(yy_1, xx_2, 'k', label = 'freq_drift(linear)')
# #     plt.xlim(min(time_list) - 10, max(time_list) + 10)
# #     plt.ylim(min(freq_list), max(freq_list))

# #     plt.tick_params(labelsize=ticksize)
# #     plt.legend(fontsize=ticksize - 40)

# #     plt.show()
# #     plt.close()
    
    
    
