#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 14:12:01 2021

@author: yuichiro
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pynverse import inversefunc
from dateutil import relativedelta
from matplotlib import dates as mdates
from datetime import date
import scipy
from scipy import stats
import sys
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from dateutil import relativedelta
freq_check = 40
move_ave = 12
move_plot = 4
#赤の線の変動を調べる
analysis_move_ave = 12
average_threshold = 1
error_threshold = 1
# solar_maximum = [datetime.datetime(2012, 1, 1), datetime.datetime(2015, 1, 1)]
# solar_minimum = [datetime.datetime(2017, 1, 1), datetime.datetime(2021, 1, 1)]
# analysis_period = [solar_maximum, solar_minimum]

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

def func(x, a, b):
    return a * x + b

labelsize = 18
fontsize = 20
factor_velocity = 2
color_list = ['#ff7f00', '#377eb8','#ff7f00', '#377eb8', '#377eb8']
color_list_1 = ['r', 'b','k', 'y', 'm']
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1




obs_burst = 'Ordinary type Ⅲ bursts'
obs_burst_1 = 'Micro-type Ⅲ bursts'
freq_drift_day_list = []
freq_drift_each_active_list = []
start_frequency_day_list = []
start_frequency_each_active_list = []
end_frequency_day_list = []
end_frequency_each_active_list = []
duration_day_list = []
duration_each_active_list = []

import astropy.time
from astropy.coordinates import get_sun
from astropy.coordinates import AltAz
from astropy.coordinates import EarthLocation
import math

def solar_cos(obs_datetime):
    koko = EarthLocation(lat='47 22 24.00',lon='2 11 50.00', height = '150')
    obs_time = obs_datetime
    toki = astropy.time.Time(obs_time)
    taiyou = get_sun(toki).transform_to(AltAz(obstime=toki,location=koko))
    # print(taiyou)
    # print(taiyou.az) # 天球での方位角
    # print(taiyou.alt) # 天球での仰俯角
    # print(taiyou.distance) # 距離
    # print(taiyou.distance.au) # au単位での距離


    azimuth = float(str(taiyou.az).split('d')[0] + '.' + str(taiyou.az).split('d')[1].split('m')[0])
    altitude = float(str(taiyou.alt).split('d')[0] + '.' + str(taiyou.alt).split('d')[1].split('m')[0])
    # print (azimuth, altitude)
    
    solar_place = np.array([math.cos(math.radians(altitude)) * math.cos(math.radians(azimuth)),
                            math.cos(math.radians(altitude)) * math.sin(math.radians(azimuth)),
                            math.sin(math.radians(altitude))])
    machine_place = np.array([math.cos(math.radians(70)) * math.cos(math.radians(180)),
                            math.cos(math.radians(70)) * math.sin(math.radians(180)),
                            math.sin(math.radians(70))])
    cos = np.dot(solar_place, machine_place)
    return cos

def getNearestValue(list, num):
    """
    概要: リストからある値に最も近い値を返却する関数
    @param list: データ配列
    @param num: 対象値
    @return 対象値に最も近い値
    """

    # リスト要素と対象値の差分を計算し最小値のインデックスを取得
    idx = np.abs(np.asarray(list) - num).argmin()
    return list[idx]


datetime_start_end = [datetime.datetime(2007, 1, 1, 0, 0), datetime.datetime(2021, 5, 8, 23, 59)]
# datetime_start_end = [datetime.datetime(2012, 5, 1, 0, 0), datetime.datetime(2019, 6, 30, 23, 59)]


bursts_obs_times_od = []
bursts_times_only_od = []
peak_LL_od = []
peak_RR_od = []
cos_od = []
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/afjpgu_LL_RR_flare_associated_ordinary_dB.csv"
csv_input_ordinary = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
for i in range(len(csv_input_ordinary)):
    obs_time = datetime.datetime(int(str(csv_input_ordinary['event_date'][i])[:4]), int(str(csv_input_ordinary['event_date'][i])[4:6]), int(str(csv_input_ordinary['event_date'][i])[6:8]), csv_input_ordinary['event_hour'][i], csv_input_ordinary['event_minite'][i])

    if obs_time <= datetime.datetime(2010,1,1):
        freq_40 = 40
    else:
        freq_40 = 39.925
    if (csv_input_ordinary['freq_start'][i] >= freq_40) & (csv_input_ordinary['freq_end'][i] <= freq_40):
        if (obs_time <= datetime_start_end[1]) & (obs_time >= datetime_start_end[0]):
            bursts_obs_times_od.append(obs_time)
            bursts_times_only_od.append(datetime.datetime(2013, 1, 1, csv_input_ordinary['event_hour'][i], csv_input_ordinary['event_minite'][i]))
            peak_LL_od.append(csv_input_ordinary['peak_LL_40MHz'][i])
            peak_RR_od.append(csv_input_ordinary['peak_RR_40MHz'][i])
            cos_od.append(solar_cos(obs_time))

bursts_obs_times_od = np.array(bursts_obs_times_od)
peak_LL_od = np.array(peak_LL_od)
peak_RR_od = np.array(peak_RR_od)
bursts_times_only_od = np.array(bursts_times_only_od)
cos_micro = np.array(cos_od)

bursts_obs_times_micro = []
bursts_times_only_micro = []
peak_LL_micro = []
peak_RR_micro = []
cos_micro = []
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/afjpgu_LL_RR_micro_dB.csv"
csv_input_micro = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
for i in range(len(csv_input_micro)):
    obs_time = datetime.datetime(int(str(csv_input_micro['event_date'][i])[:4]), int(str(csv_input_micro['event_date'][i])[4:6]), int(str(csv_input_micro['event_date'][i])[6:8]), csv_input_micro['event_hour'][i], csv_input_micro['event_minite'][i])

    if obs_time <= datetime.datetime(2010,1,1):
        freq_micro = 40
    else:
        freq_micro = 39.925
    if (csv_input_micro['freq_start'][i] >= freq_40) & (csv_input_micro['freq_end'][i] <= freq_40):
        if (obs_time <= datetime_start_end[1]) & (obs_time >= datetime_start_end[0]):
            bursts_obs_times_micro.append(obs_time)
            bursts_times_only_micro.append(datetime.datetime(2013, 1, 1, csv_input_micro['event_hour'][i], csv_input_micro['event_minite'][i]))
            peak_LL_micro.append(csv_input_micro['peak_LL_40MHz'][i])
            peak_RR_micro.append(csv_input_micro['peak_RR_40MHz'][i])
            cos_micro.append(solar_cos(obs_time))

bursts_obs_times_micro = np.array(bursts_obs_times_micro)
peak_LL_micro = np.array(peak_LL_micro)
peak_RR_micro = np.array(peak_RR_micro)
bursts_times_only_micro = np.array(bursts_times_only_micro)
cos_micro = np.array(cos_micro)


BG_obs_times = []
gain_RR = []
gain_LL = []
Trx_RR = []
Trx_LL = []
cali_move_decibel_list_r = []
cali_move_decibel_list_l = []
Used_db_list = []

# file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/RR_LL_gain_movecaliBG_40MHz_analysis_calibrated_1.csv'
# file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/RR_LL_gain_movecaliBG_40MHz_analysis_calibrated_median.csv'

file_gain = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/RR_LL_gain_movecaliBG/40MHz/under25_RR_LL_gain_movecaliBG_40MHz_analysis_calibrated_median.csv'
print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=",")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(int(csv_input['obs_time'][i].split('-')[0]), int(csv_input['obs_time'][i].split('-')[1]), int(csv_input['obs_time'][i].split(' ')[0][-2:]), int(csv_input['obs_time'][i].split(' ')[1][:2]), int(csv_input['obs_time'][i].split(':')[1]), int(csv_input['obs_time'][i].split(':')[2][:2]))
    BG_obs_times.append(BG_obs_time_event)
    # Frequency_list = csv_input['Frequency'][i]
    cali_move_RR = csv_input['Right-BG_move_Calibrated'][i]
    cali_move_decibel_list_r.append(cali_move_RR)
    cali_move_LL = csv_input['Left-BG_move_Calibrated'][i]
    cali_move_decibel_list_l.append(cali_move_LL)
    Used_db = csv_input['Used_dB_median'][i]
    Used_db_list.append(Used_db)
    RR_gain = csv_input['Right-gain'][i]
    gain_RR.append(RR_gain)
    LL_gain = csv_input['Left-gain'][i]
    gain_LL.append(LL_gain)
    RR_Trx = csv_input['Right-Trx'][i]
    Trx_RR.append(RR_Trx)
    LL_Trx = csv_input['Left-Trx'][i]
    Trx_LL.append(LL_Trx)


BG_obs_times = np.array(BG_obs_times)
gain_RR = np.array(gain_RR)
gain_LL = np.array(gain_LL)
Trx_RR = np.array(Trx_RR)
Trx_LL = np.array(Trx_LL)
cali_move_BG_r = np.array(cali_move_decibel_list_r)
cali_move_BG_l = np.array(cali_move_decibel_list_l)
Used_db_list = np.array(Used_db_list)

hist_intensity = []

fontsize_title = 24
fontsize_labelsize = 20

burst_types = ["Micro type III burst", "Ordinary type III burst"]
fig = plt.figure(figsize=(36.0, 36.0))
gs = gridspec.GridSpec(315, 4)
for burst_type in burst_types:

    obs_time_list = []
    time_only_list = []
    intensity_db_r_list = []
    intensity_db_l_list = []
    intensity_db_list = []
    intensity_db_cos_list = []
    cos_list = []
    pol_list = []
    Gain_RR_list = []
    Gain_LL_list = []
    Trx_RR_list = []
    Trx_LL_list = []
    cali_move_bg_r_list = []
    cali_move_bg_l_list = []
    peak_l_list = []
    peak_r_list = []
    Trx_cali_list = []

    if burst_type == "Micro type III burst":
        obs_times = bursts_obs_times_micro
        time_only = bursts_times_only_micro
        peak_LL = peak_LL_micro
        peak_RR = peak_RR_micro
        cos_val = cos_micro
        ax = plt.subplot(gs[0:40, :1])
        ax1 = plt.subplot(gs[55:95, :1])
        ax2 = plt.subplot(gs[110:150, :1])
        ax3 = plt.subplot(gs[165:205, :1])
        ax4 = plt.subplot(gs[220:260, :1])
        ax5 = plt.subplot(gs[0:40, 2:3])
        ax6 = plt.subplot(gs[55:95, 2:3])
        ax7 = plt.subplot(gs[110:150, 2:3])
        ax8 = plt.subplot(gs[165:205, 2:3])
        ax13 = plt.subplot(gs[220:260, 2:3])
        ax9 = plt.subplot(gs[0:40, 3:4])
        ax10 = plt.subplot(gs[55:95, 3:4])
        ax11 = plt.subplot(gs[110:150, 3:4])
        ax12 = plt.subplot(gs[165:205, 3:4])
        # ax14 = plt.subplot(gs[220:260, 3:4])
        # ax15 = plt.subplot(gs[275:315, 3:4])

        color_setting = 'b'
    elif burst_type == "Ordinary type III burst":
        obs_times = bursts_obs_times_od
        time_only = bursts_times_only_od
        peak_LL = peak_LL_od
        peak_RR = peak_RR_od
        cos_val = cos_od
        ax = plt.subplot(gs[0:40, 1:2])
        ax1 = plt.subplot(gs[55:95, 1:2])
        ax2 = plt.subplot(gs[110:150, 1:2])
        ax3 = plt.subplot(gs[165:205, 1:2])
        ax4 = plt.subplot(gs[220:260, 1:2])

        color_setting = 'b'
        # ax = plt.subplot(gs[0:40, :1])
        # ax1 = plt.subplot(gs[55:95, :1])
        # ax2 = plt.subplot(gs[110:150, :1])
        # ax3 = plt.subplot(gs[165:205, :1])
        # ax4 = plt.subplot(gs[220:260, :1])
        # color_setting = 'r'
    for i in range(len(obs_times)):

        if np.abs(getNearestValue(BG_obs_times, obs_times[i])-obs_times[i]) <= datetime.timedelta(seconds=60*90):
            BG_idx = np.where(BG_obs_times == getNearestValue(BG_obs_times, obs_times[i]))
            obs_time_list.append(obs_times[i])
            print (obs_times[i])
            time_only_list.append(time_only[i])
            cos = cos_val[i]
            gain_l = gain_LL[BG_idx]
            Gain_LL_list.append(gain_l)
            Trx_LL_list.append(Trx_LL[BG_idx])
            cali_move_bg_l = cali_move_BG_l[BG_idx]
            power_cali_move_bg_l = 10 ** (cali_move_bg_l/10)
            Used_db = Used_db_list[BG_idx]
            peak_l = peak_LL[i]
            peak_cali_l = peak_l + (np.log10(Used_db/gain_l) * 10)
            Trx_cali_l = Trx_LL[BG_idx] + (np.log10(Used_db/gain_l) * 10)
            power_peak_cali_l = (10 ** (peak_cali_l/10))/cos
            intensity_power_l = power_peak_cali_l - power_cali_move_bg_l
            intensity_db_l = np.log10(intensity_power_l) * 10
            intensity_db_cos_l = np.log10((intensity_power_l)) * 10

            gain_r = gain_RR[BG_idx]
            Gain_RR_list.append(gain_r)
            Trx_RR_list.append(Trx_RR[BG_idx])
            cali_move_bg_r = cali_move_BG_r[BG_idx]
            power_cali_move_bg_r = 10 ** (cali_move_bg_r/10)
            peak_r = peak_RR[i]
            peak_cali_r = peak_r + (np.log10(Used_db/gain_r) * 10)
            Trx_cali_r = Trx_RR[BG_idx] + (np.log10(Used_db/gain_r) * 10)
            power_peak_cali_r = (10 ** (peak_cali_r/10))/cos
            intensity_power_r = power_peak_cali_r - power_cali_move_bg_r
            intensity_db_r = np.log10(intensity_power_r) * 10
            intensity_db_cos_r = np.log10((intensity_power_r)) * 10
            Trx_cali = (Trx_cali_l + Trx_cali_r)
            
            pol = ((intensity_power_r - intensity_power_l)/(intensity_power_r + intensity_power_l))[0]

            intensity_power_cos = (intensity_power_l + intensity_power_r)/2
            intensity_power = ((10 ** (peak_cali_l/10)) + (10 ** (peak_cali_r/10)))/2
            intensity_db = np.log10(intensity_power) * 10
            intensity_db_cos = np.log10(intensity_power_cos) * 10

            intensity_db_list.append(intensity_db)
            intensity_db_cos_list.append(intensity_db_cos)
            # intensity_db_r_list.append(intensity_db_r)
            # intensity_db_l_list.append(intensity_db_l)
            cos_list.append(cos)
            pol_list.append(pol)
            cali_move_bg_r_list.append(cali_move_bg_r)
            cali_move_bg_l_list.append(cali_move_bg_l)
            peak_r_list.append(peak_r)
            peak_l_list.append(peak_l)
            Trx_cali_list.append(Trx_cali)

    hist_intensity.append(intensity_db_cos_list)
    fmt = mdates.DateFormatter('%H:%M') 
    # ax.plot(time_only_list, intensity_db_list, color = 'b', marker='o', linestyle='dashdot')
    ax.scatter(time_only_list, intensity_db_list, c=color_setting, s=7)
    ax.set_title(burst_type + ': intensity', fontsize = fontsize_title)

    ax.xaxis.set_major_formatter(fmt)
    ax.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
    ax.tick_params(axis='y', labelsize=fontsize_labelsize)
    ax.set_xlabel('Time', fontsize = fontsize_title-4)
    ax.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
    ax.set_ylim(23, 50)


    ax1.scatter(time_only_list, intensity_db_cos_list, c=color_setting, s=7)
    ax1.set_title(burst_type + ': intensity/cos', fontsize = fontsize_title)
    ax1.xaxis.set_major_formatter(fmt)
    ax1.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
    ax1.tick_params(axis='y', labelsize=fontsize_labelsize)
    ax1.set_xlabel('Time', fontsize = fontsize_title-4)
    ax1.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
    ax1.set_ylim(23, 50)

    ax2.scatter(cos_list, intensity_db_list, c=color_setting, s=7)
    ax2.set_title(burst_type + ': intensity', fontsize = fontsize_title)
    ax2.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
    ax2.tick_params(axis='y', labelsize=fontsize_labelsize)
    ax2.set_xlabel('cosθ', fontsize = fontsize_title-4)
    ax2.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
    ax2.set_xlim(0.7,1)
    ax2.set_ylim(23, 50)

    ax3.scatter(cos_list, intensity_db_cos_list, c=color_setting, s=7)
    ax3.set_title(burst_type + ': intensity/cos', fontsize = fontsize_title)
    ax3.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
    ax3.tick_params(axis='y', labelsize=fontsize_labelsize)
    ax3.set_xlabel('cosθ', fontsize = fontsize_title-4)
    ax3.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
    ax3.set_xlim(0.7,1)
    ax3.set_ylim(23, 50)

    ax4.scatter(intensity_db_cos_list, pol_list, c=color_setting, s=7)
    ax4.set_title(burst_type + ': Polarization', fontsize = fontsize_title)

    ax4.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
    ax4.tick_params(axis='y', labelsize=fontsize_labelsize)
    ax4.set_xlabel('from background[dB]', fontsize = fontsize_title-4)
    ax4.set_ylabel('polarization ', fontsize = fontsize_title-2)
    ax4.set_xlim(23, 50)
    ax4.set_ylim(-1, 1)


    if burst_type == "Micro type III burst":
        print ('plot micro')
        # ax.plot(time_only_list, intensity_db_list, color = 'b', marker='o', linestyle='dashdot')
        ax5.scatter(time_only_list, Gain_RR_list, c=color_setting, s=7)
        ax5.set_title(burst_type + ': Gain_RR', fontsize = fontsize_title)
        ax5.xaxis.set_major_formatter(fmt)
        ax5.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax5.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax5.set_xlabel('Time', fontsize = fontsize_title-4)
        ax5.set_ylabel('Gain', fontsize = fontsize_title-2)
        ax5.set_ylim(np.min([np.min(Gain_RR_list), np.min(Gain_LL_list)]), np.max([np.max(Gain_RR_list), np.max(Gain_LL_list)]))

        ax6.scatter(time_only_list, Gain_LL_list, c=color_setting, s=7)
        ax6.set_title(burst_type + ': Gain_LL', fontsize = fontsize_title)
        ax6.xaxis.set_major_formatter(fmt)
        ax6.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax6.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax6.set_xlabel('Time', fontsize = fontsize_title-4)
        ax6.set_ylabel('Gain', fontsize = fontsize_title-2)
        ax6.set_ylim(np.min([np.min(Gain_RR_list), np.min(Gain_LL_list)]), np.max([np.max(Gain_RR_list), np.max(Gain_LL_list)]))
        # ax6.set_ylim(23, 50)

        ax7.scatter(time_only_list, Trx_RR_list, c=color_setting, s=7)
        ax7.set_title(burst_type + ': Trx_RR', fontsize = fontsize_title)
        ax7.xaxis.set_major_formatter(fmt)
        ax7.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax7.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax7.set_xlabel('Time', fontsize = fontsize_title-4)
        ax7.set_ylabel('Trx[k]', fontsize = fontsize_title-2)
        ax7.set_ylim(np.min([np.min(Trx_RR_list), np.min(Trx_LL_list)]), np.max([np.max(Trx_RR_list), np.max(Trx_LL_list)]))


        ax8.scatter(time_only_list, Trx_LL_list, c=color_setting, s=7)
        ax8.set_title(burst_type + ': Trx_LL', fontsize = fontsize_title)
        ax8.xaxis.set_major_formatter(fmt)
        ax8.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax8.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax8.set_xlabel('Time', fontsize = fontsize_title-4)
        ax8.set_ylabel('Trx[k]', fontsize = fontsize_title-2)
        ax8.set_ylim(np.min([np.min(Trx_RR_list), np.min(Trx_LL_list)]), np.max([np.max(Trx_RR_list), np.max(Trx_LL_list)]))

        ax9.scatter(time_only_list, cali_move_bg_r_list, c=color_setting, s=7)
        ax9.set_title(burst_type + ': BG_RR (calibrated)', fontsize = fontsize_title)
        ax9.xaxis.set_major_formatter(fmt)
        ax9.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax9.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax9.set_xlabel('Time', fontsize = fontsize_title-4)
        ax9.set_ylabel('BG[dB]', fontsize = fontsize_title-2)
        ax9.set_ylim(np.min([np.min(cali_move_bg_r_list), np.min(cali_move_bg_l_list)]), np.max([np.max(cali_move_bg_r_list), np.max(cali_move_bg_l_list)]))


        ax10.scatter(time_only_list, cali_move_bg_l_list, c=color_setting, s=7)
        ax10.set_title(burst_type + ': BG_LL (calibrated)', fontsize = fontsize_title)
        ax10.xaxis.set_major_formatter(fmt)
        ax10.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax10.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax10.set_xlabel('Time', fontsize = fontsize_title-4)
        ax10.set_ylabel('BG[dB]', fontsize = fontsize_title-2)
        ax10.set_ylim(np.min([np.min(cali_move_bg_r_list), np.min(cali_move_bg_l_list)]), np.max([np.max(cali_move_bg_r_list), np.max(cali_move_bg_l_list)]))

        ax11.scatter(time_only_list, peak_r_list, c=color_setting, s=7)
        ax11.set_title(burst_type + ': Peak_RR', fontsize = fontsize_title)
        ax11.xaxis.set_major_formatter(fmt)
        ax11.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax11.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax11.set_xlabel('Time', fontsize = fontsize_title-4)
        ax11.set_ylabel('Intensity[dB]', fontsize = fontsize_title-2)
        ax11.set_ylim(np.min([np.min(peak_r_list), np.min(peak_l_list)]), np.max([np.max(peak_r_list), np.max(peak_l_list)]))

        ax12.scatter(time_only_list, peak_l_list, c=color_setting, s=7)
        ax12.set_title(burst_type + ': Peak_LL', fontsize = fontsize_title)
        ax12.xaxis.set_major_formatter(fmt)
        ax12.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax12.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax12.set_xlabel('Time', fontsize = fontsize_title-4)
        ax12.set_ylabel('Intensity[dB]', fontsize = fontsize_title-2)
        ax12.set_ylim(np.min([np.min(peak_r_list), np.min(peak_l_list)]), np.max([np.max(peak_r_list), np.max(peak_l_list)]))
        
        ax13.scatter(time_only_list, Trx_cali_list, c=color_setting, s=7)
        ax13.set_title(burst_type + ': Total Trx', fontsize = fontsize_title)
        ax13.xaxis.set_major_formatter(fmt)
        ax13.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        ax13.tick_params(axis='y', labelsize=fontsize_labelsize)
        ax13.set_xlabel('Time', fontsize = fontsize_title-4)
        ax13.set_ylabel('Temperature[dB]', fontsize = fontsize_title-2)
        ax13.set_ylim(np.min(Trx_cali_list), np.max(Trx_cali_list))


        
        # ax14.scatter(time_only_list, intensity_db_r_list, c=color_setting, s=7)
        # ax14.set_title(burst_type + ': Intensity RR', fontsize = fontsize_title)
        # ax14.xaxis.set_major_formatter(fmt)
        # ax14.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        # ax14.tick_params(axis='y', labelsize=fontsize_labelsize)
        # ax14.set_xlabel('Time', fontsize = fontsize_title-4)
        # ax14.set_ylabel('From background[dB]', fontsize = fontsize_title-2)
        # ax14.set_ylim(np.min(intensity_db_r_list), np.max(intensity_db_r_list))

        # ax15.scatter(time_only_list, intensity_db_l_list, c=color_setting, s=7)
        # ax15.set_title(burst_type + ': Intensity LL', fontsize = fontsize_title)
        # ax15.xaxis.set_major_formatter(fmt)
        # ax15.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
        # ax15.tick_params(axis='y', labelsize=fontsize_labelsize)
        # ax15.set_xlabel('Time', fontsize = fontsize_title-4)
        # ax15.set_ylabel('From background[dB]', fontsize = fontsize_title-2)
        # ax15.set_ylim(np.min(intensity_db_l_list), np.max(intensity_db_l_list))

    obs_time_list = np.array(obs_time_list)
    time_only_list = np.array(time_only_list)
    intensity_db_list = np.array(intensity_db_list)
    intensity_db_cos_list = np.array(intensity_db_cos_list)
    cos_list = np.array(cos_list)
    pol_list = np.array(pol_list)

    Gain_RR_list = np.array(Gain_RR_list)
    Gain_LL_list = np.array(Gain_LL_list)
    Trx_LL_list = np.array(Trx_LL_list)
    Trx_RR_list = np.array(Trx_RR_list)
    cali_move_bg_l_list = np.array(cali_move_bg_l_list)
    cali_move_bg_r_list = np.array(cali_move_bg_r_list)
    peak_l_list = np.array(peak_l_list)
    peak_r_list = np.array(peak_r_list)
    Trx_cali_list = np.array(Trx_cali_list)
    # intensity_db_r_list = np.array(intensity_db_r_list)
    # intensity_db_l_list = np.array(intensity_db_l_list)


    sep_time_start = np.min(time_only_list)
    sep_time_final = np.max(time_only_list)
    while sep_time_start < sep_time_final:
        sep_time_end = sep_time_start + datetime.timedelta(minutes = 30)
        sep_idx = np.where((time_only_list >= sep_time_start) & (time_only_list < sep_time_end))[0]
        if len(sep_idx) > 0:
            if len(sep_idx) > 5:
                ax.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),fmt='o')
                ax1.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),fmt='o')
                if burst_type == "Micro type III burst":
                    ax5.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Gain_RR_list[sep_idx]), color = 'r', yerr=np.std(Gain_RR_list[sep_idx]),fmt='o')
                    print ('Done ax5')
                    ax6.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Gain_LL_list[sep_idx]), color = 'r', yerr=np.std(Gain_LL_list[sep_idx]),fmt='o')
                    print ('Done ax6')
                    ax7.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_RR_list[sep_idx]), color = 'r', yerr=np.std(Trx_RR_list[sep_idx]),fmt='o')
                    print ('Done ax7')
                    ax8.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_LL_list[sep_idx]), color = 'r', yerr=np.std(Trx_LL_list[sep_idx]),fmt='o')
                    print ('Done ax8')
                    ax9.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(cali_move_bg_r_list[sep_idx]), color = 'r', yerr=np.std(cali_move_bg_r_list[sep_idx]),fmt='o')
                    print ('Done ax9')
                    ax10.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(cali_move_bg_l_list[sep_idx]), color = 'r', yerr=np.std(cali_move_bg_l_list[sep_idx]),fmt='o')
                    print ('Done ax10')
                    ax11.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(peak_r_list[sep_idx]), color = 'r', yerr=np.std(peak_r_list[sep_idx]),fmt='o')
                    print ('Done ax11')
                    ax12.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(peak_l_list[sep_idx]), color = 'r', yerr=np.std(peak_l_list[sep_idx]),fmt='o')
                    print ('Done ax12')
                    ax13.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_cali_list[sep_idx]), color = 'r', yerr=np.std(Trx_cali_list[sep_idx]),fmt='o')
                    print ('Done ax13')
                    # ax14.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_r_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_r_list[sep_idx]),fmt='o')
                    # ax15.errorbar(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_l_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_l_list[sep_idx]),fmt='o')
                # ax.axvline(sep_time_start, ls = "--", color = "navy")
                # ax1.axvline(sep_time_start, ls = "--", color = "navy")
            else:
                ax.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_list[sep_idx]), color = 'r',s = 15)
                ax1.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_cos_list[sep_idx]), color = 'r',s = 15)
                if burst_type == "Micro type III burst":
                    ax5.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Gain_RR_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax5')
                    ax6.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Gain_LL_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax6')
                    ax7.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_RR_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax7')
                    ax8.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_LL_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax8')
                    ax9.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(cali_move_bg_r_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax9')
                    ax10.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(cali_move_bg_l_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax10')
                    ax11.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(peak_r_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax11')
                    # ax12.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(peak_l_list[sep_idx]), color = 'r',s = 15)
                    # ax13.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_cali_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax12')
                    ax12.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(peak_l_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax12')
                    ax13.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(Trx_cali_list[sep_idx]), color = 'r',s = 15)
                    print ('Done ax13')
                    # ax12.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_r_list[sep_idx]), color = 'r',s = 15)
                    # ax13.scatter(sep_time_start + datetime.timedelta(minutes = 15), np.mean(intensity_db_l_list[sep_idx]), color = 'r',s = 15)
                # ax.axvline(sep_time_start, ls = "--", color = "navy")
                # ax1.axvline(sep_time_start, ls = "--", color = "navy")
        sep_time_start += datetime.timedelta(minutes = 30)



    sep_cos_start = np.min(cos_list)
    sep_cos_final = np.max(cos_list)
    while sep_cos_start < sep_cos_final:
        sep_cos_end = sep_cos_start + 0.05
        sep_idx = np.where((cos_list >= sep_cos_start) & (cos_list < sep_cos_end))[0]
        if len(sep_idx) > 0:
            if len(sep_idx) > 5:
                ax2.errorbar(sep_cos_start + 0.025, np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),fmt='o')
                print ('Done cos')
                ax3.errorbar(sep_cos_start + 0.025, np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),fmt='o')
                print ('Done cos1')
                # ax2.axvline(sep_cos_start, ls = "--", color = "navy")
                # ax3.axvline(sep_cos_start, ls = "--", color = "navy")
            else:
                ax2.scatter(sep_cos_start + 0.025, np.mean(intensity_db_list[sep_idx]), color = 'r',s = 15)
                print ('Done cos')
                ax3.scatter(sep_cos_start + 0.025, np.mean(intensity_db_cos_list[sep_idx]), color = 'r',s = 15)
                print ('Done cos1')
                # ax2.errorbar(sep_cos_start + 0.125, np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),'o')
                # ax3.errorbar(sep_cos_start + 0.125, np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),'o')
                # ax2.axvline(sep_cos_start, ls = "--", color = "navy")
                # ax3.axvline(sep_cos_start, ls = "--", color = "navy")
        sep_cos_start += 0.05

    sep_pol_start = np.min(intensity_db_cos_list)
    sep_pol_final = np.max(intensity_db_cos_list)
    while sep_pol_start < sep_pol_final:
        sep_pol_end = sep_pol_start + 5
        sep_idx = np.where((intensity_db_cos_list >= sep_pol_start) & (intensity_db_cos_list < sep_pol_end))[0]
        if len(sep_idx) > 0:
            if len(sep_idx) > 5:
                ax4.errorbar(sep_pol_start + 2.5, np.mean(pol_list[sep_idx]), color = 'r', yerr=np.std(pol_list[sep_idx]),fmt='o')
                print ('Done pol')
                # ax3.errorbar(sep_pol_start + 2.5, np.mean(pol_list[sep_idx]), color = 'r', yerr=np.std(pol_list[sep_idx]),fmt='o')
                # print ('Done pol1')
                # ax2.axvline(sep_cos_start, ls = "--", color = "navy")
                # ax3.axvline(sep_cos_start, ls = "--", color = "navy")
            else:
                ax2.scatter(sep_pol_start + 2.5, np.mean(pol_list[sep_idx]), color = 'r',s = 15)
                print ('Done pol')
                # ax3.scatter(sep_pol_start + 2.5, np.mean(intensity_db_cos_list[sep_idx]), color = 'r',s = 15)
                # print ('Done pol1')
                # ax2.errorbar(sep_cos_start + 0.125, np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),'o')
                # ax3.errorbar(sep_cos_start + 0.125, np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),'o')
                # ax2.axvline(sep_cos_start, ls = "--", color = "navy")
                # ax3.axvline(sep_cos_start, ls = "--", color = "navy")
        sep_pol_start += 5
        

plt.show()
plt.close()

intensity_hist = []
for i in range(len(hist_intensity[0])):
    intensity_hist.append(hist_intensity[0][i][0])

plt.hist(intensity_hist, bins = 10)
plt.show()

intensity_hist_1 = []
for i in range(len(hist_intensity[1])):
    intensity_hist_1.append(hist_intensity[1][i][0])

plt.hist(intensity_hist_1, bins = 10)
plt.show()


height = plt.hist(intensity_hist)[0]/len(intensity_hist)
width = plt.hist(intensity_hist)[1][:-1]
height_1 = plt.hist(intensity_hist_1)[0]/len(intensity_hist_1)
width_1 = plt.hist(intensity_hist_1)[1][:-1]
plt.close()
plt.bar(width, height)


plt.bar(width_1, height_1)

plt.show()


total_intensity_hist = []
total_intensity_hist.extend(intensity_hist)
total_intensity_hist.extend(intensity_hist_1)
plt.hist(total_intensity_hist, bins = 10)
plt.yscale('log')
plt.xlabel('From background[dB]')
plt.show()
#     fmt = mdates.DateFormatter('%Y-%m') 
#     ax.scatter(obs_time_list, intensity_db_list, c=color_setting, s=7)
#     ax.set_title(burst_type + ': intensity', fontsize = fontsize_title)

#     ax.xaxis.set_major_formatter(fmt)
#     ax.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
#     ax.tick_params(axis='y', labelsize=fontsize_labelsize)
#     ax.set_xlabel('Time', fontsize = fontsize_title-4)
#     ax.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
#     ax.set_ylim(23, 50)


#     ax1.scatter(obs_time_list, intensity_db_cos_list, c=color_setting, s=7)
#     ax1.set_title(burst_type + ': intensity/cos', fontsize = fontsize_title)
#     ax1.xaxis.set_major_formatter(fmt)
#     ax1.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
#     ax1.tick_params(axis='y', labelsize=fontsize_labelsize)
#     ax1.set_xlabel('Time', fontsize = fontsize_title-4)
#     ax1.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
#     ax1.set_ylim(23, 50)

#     ax2.scatter(cos_list, intensity_db_list, c=color_setting, s=7)
#     ax2.set_title(burst_type + ': intensity', fontsize = fontsize_title)
#     ax2.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
#     ax2.tick_params(axis='y', labelsize=fontsize_labelsize)
#     ax2.set_xlabel('cosθ', fontsize = fontsize_title-4)
#     ax2.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
#     ax2.set_xlim(0.7,1)

#     ax3.scatter(cos_list, intensity_db_cos_list, c=color_setting, s=7)
#     ax3.set_title(burst_type + ': intensity/cos', fontsize = fontsize_title)
#     ax3.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
#     ax3.tick_params(axis='y', labelsize=fontsize_labelsize)
#     ax3.set_xlabel('cosθ', fontsize = fontsize_title-4)
#     ax3.set_ylabel('from background[dB]', fontsize = fontsize_title-2)
#     ax3.set_xlim(0.7,1)

#     ax4.scatter(intensity_db_cos_list, pol_list, c=color_setting, s=7)
#     ax4.set_title(burst_type + ': Polarization', fontsize = fontsize_title)
#     # ax4.xaxis.set_major_formatter(fmt)
#     ax4.tick_params(axis='x', labelrotation=20, labelsize=fontsize_labelsize)
#     ax4.tick_params(axis='y', labelsize=fontsize_labelsize)
#     ax4.set_xlabel('from background[dB]', fontsize = fontsize_title-4)
#     ax4.set_ylabel('polarization ', fontsize = fontsize_title-2)
#     ax4.set_xlim(23, 50)
#     ax4.set_ylim(-0.01, 0.045)

#     obs_time_list = np.array(obs_time_list)
#     time_only_list = np.array(time_only_list)
#     intensity_db_list = np.array(intensity_db_list)
#     intensity_db_cos_list = np.array(intensity_db_cos_list)
#     cos_list = np.array(cos_list)
#     pol_list  = np.array(pol_list)

#     sep_time_start = np.min(obs_time_list)
#     sep_time_final = np.max(obs_time_list)
#     while sep_time_start < sep_time_final:
#         sep_time_end = sep_time_start + relativedelta.relativedelta(years=1)
#         sep_idx = np.where((obs_time_list >= sep_time_start) & (obs_time_list < sep_time_end))[0]
#         if len(sep_idx) > 0:
#             if len(sep_idx) > 5:
#                 ax.errorbar(sep_time_start + datetime.timedelta(days = 182), np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),fmt='o')
#                 ax1.errorbar(sep_time_start + datetime.timedelta(days = 182), np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),fmt='o')
#                 # ax.axvline(sep_time_start, ls = "--", color = "navy")
#                 # ax1.axvline(sep_time_start, ls = "--", color = "navy")
#             else:
#                 ax.scatter(sep_time_start + datetime.timedelta(days = 182), np.mean(intensity_db_list[sep_idx]), color = 'r',s = 15)
#                 ax1.scatter(sep_time_start + datetime.timedelta(days = 182), np.mean(intensity_db_cos_list[sep_idx]), color = 'r',s = 15)
#                 # ax.axvline(sep_time_start, ls = "--", color = "navy")
#                 # ax1.axvline(sep_time_start, ls = "--", color = "navy")
#         sep_time_start += relativedelta.relativedelta(years=1)

#     # sep_cos_start = np.min(cos_list)
#     # sep_cos_final = np.max(cos_list)
#     # while sep_cos_start < sep_cos_final:
#     #     sep_cos_end = sep_cos_start + 0.05
#     #     sep_idx = np.where((cos_list >= sep_cos_start) & (cos_list < sep_cos_end))[0]
#     #     if len(sep_idx) > 0:
#     #         if len(sep_idx) > 5:
#     #             ax2.errorbar(sep_cos_start + 0.025, np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),fmt='o')
#     #             ax3.errorbar(sep_cos_start + 0.025, np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),fmt='o')
#     #             # ax2.axvline(sep_cos_start, ls = "--", color = "navy")
#     #             # ax3.axvline(sep_cos_start, ls = "--", color = "navy")
#     #         else:
#     #             ax2.scatter(sep_cos_start + 0.025, np.mean(intensity_db_list[sep_idx]), color = 'r',s = 15)
#     #             ax3.scatter(sep_cos_start + 0.025, np.mean(intensity_db_cos_list[sep_idx]), color = 'r',s = 15)
#     #             # ax2.errorbar(sep_cos_start + 0.125, np.mean(intensity_db_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_list[sep_idx]),'o')
#     #             # ax3.errorbar(sep_cos_start + 0.125, np.mean(intensity_db_cos_list[sep_idx]), color = 'r', yerr=np.std(intensity_db_cos_list[sep_idx]),'o')
#     #             # ax2.axvline(sep_cos_start, ls = "--", color = "navy")
#     #             # ax3.axvline(sep_cos_start, ls = "--", color = "navy")
#     #     sep_cos_start += 0.05
        

# plt.show()
# plt.close()