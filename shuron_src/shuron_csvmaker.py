#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 14:14:39 2021

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
import csv
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
velocity_list_od = []
residual_list_od = []
event_start_od = []
event_end_od = []
freq_start_od = []
freq_end_od = []
peak_time_list_od = []
peak_freq_list_od = []
best_factor_od = []
freq_check_od  = []
freq_drift_od  = []
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/afjpgu_LL_RR_flare_associated_ordinary_dB.csv"
csv_input_ordinary = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
for i in range(len(csv_input_ordinary)):
    obs_time = datetime.datetime(int(str(csv_input_ordinary['event_date'][i])[:4]), int(str(csv_input_ordinary['event_date'][i])[4:6]), int(str(csv_input_ordinary['event_date'][i])[6:8]), csv_input_ordinary['event_hour'][i], csv_input_ordinary['event_minite'][i])

    freq_40 = 40
    # if (csv_input_ordinary['freq_start'][i] >= freq_40) & (csv_input_ordinary['freq_end'][i] <= freq_40):
    if (obs_time <= datetime_start_end[1]) & (obs_time >= datetime_start_end[0]):
        bursts_obs_times_od.append(obs_time)
        bursts_times_only_od.append(datetime.datetime(2013, 1, 1, csv_input_ordinary['event_hour'][i], csv_input_ordinary['event_minite'][i]))
        peak_LL_od.append(csv_input_ordinary['peak_LL_40MHz'][i])
        peak_RR_od.append(csv_input_ordinary['peak_RR_40MHz'][i])
        cos_od.append(solar_cos(obs_time))
        velocity_list_od.append([float(k) for k in csv_input_ordinary['velocity'][i].split('[')[1].split(']')[0].split(',')])
        residual_list_od.append([float(k) for k in csv_input_ordinary['residual'][i].split('[')[1].split(']')[0].split(',')])
        peak_time_list_od.append([int(k) for k in csv_input_ordinary['peak_time_list'][i].split('[')[1].split(']')[0].split(',')])
        peak_freq_list_od.append([float(k) for k in csv_input_ordinary['peak_freq_list'][i].split('[')[1].split(']')[0].split(',')])
        event_start_od.append(csv_input_ordinary['event_start'][i])
        event_end_od.append(csv_input_ordinary['event_end'][i])
        freq_start_od.append(csv_input_ordinary['freq_start'][i])
        freq_end_od.append(csv_input_ordinary['freq_end'][i])
        best_factor_od.append(csv_input_ordinary['factor'][i])
        freq_check_od.append(freq_40)
t = np.arange(0, 2000, 1)
t = (t+1)/100
if len(best_factor_od) > 0:
    for i in range (len(best_factor_od)):
        factor = best_factor_od[i]
        velocity = velocity_list_od[i][factor-1]
        freq_max = freq_start_od[i]
        cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values = freq_max)
        h_start = invcube_4/69600000000 + 1
        # print (h_start)
        cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
        invcube_3 = inversefunc(cube_3, y_values=freq_check_od[i])
        # plt.plot(freq[i], invcube_3)
        # plt.show()
        # plt.close()
        
        slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
        freq_drift_od.append(-slope)
        


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
velocity_list_micro = []
residual_list_micro = []
event_start_micro = []
event_end_micro = []
freq_start_micro = []
freq_end_micro = []
peak_time_list_micro = []
peak_freq_list_micro = []
best_factor_micro = []
freq_check_micro = []
freq_drift_micro = []
file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/afjpgu_LL_RR_micro_dB.csv"
csv_input_micro = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
for i in range(len(csv_input_micro)):
    obs_time = datetime.datetime(int(str(csv_input_micro['event_date'][i])[:4]), int(str(csv_input_micro['event_date'][i])[4:6]), int(str(csv_input_micro['event_date'][i])[6:8]), csv_input_micro['event_hour'][i], csv_input_micro['event_minite'][i])

    freq_micro = 40

    if (obs_time <= datetime_start_end[1]) & (obs_time >= datetime_start_end[0]):
        bursts_obs_times_micro.append(obs_time)
        bursts_times_only_micro.append(datetime.datetime(2013, 1, 1, csv_input_micro['event_hour'][i], csv_input_micro['event_minite'][i]))
        peak_LL_micro.append(csv_input_micro['peak_LL_40MHz'][i])
        peak_RR_micro.append(csv_input_micro['peak_RR_40MHz'][i])
        cos_micro.append(solar_cos(obs_time))
        velocity_list_micro.append([float(k) for k in csv_input_micro['velocity'][i].split('[')[1].split(']')[0].split(',')])
        residual_list_micro.append([float(k) for k in csv_input_micro['residual'][i].split('[')[1].split(']')[0].split(',')])
        peak_time_list_micro.append([int(k) for k in csv_input_micro['peak_time_list'][i].split('[')[1].split(']')[0].split(',')])
        peak_freq_list_micro.append([float(k) for k in csv_input_micro['peak_freq_list'][i].split('[')[1].split(']')[0].split(',')])
        event_start_micro.append(csv_input_micro['event_start'][i])
        event_end_micro.append(csv_input_micro['event_end'][i])
        freq_start_micro.append(csv_input_micro['freq_start'][i])
        freq_end_micro.append(csv_input_micro['freq_end'][i])
        best_factor_micro.append(csv_input_micro['factor'][i])
        freq_check_micro.append(freq_micro)

t = np.arange(0, 2000, 1)
t = (t+1)/100
if len(best_factor_micro) > 0:
    for i in range (len(best_factor_micro)):
        factor = best_factor_micro[i]
        velocity = velocity_list_micro[i][factor-1]
        freq_max = freq_start_micro[i]
        cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/69600000000))**(-16))+1.55*((1+(h/69600000000))**(-6))+0.036*((1+(h/69600000000))**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values = freq_max)
        h_start = invcube_4/69600000000 + 1
        # print (h_start)
        cube_3 = (lambda t: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (t * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + (t * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + (t * velocity * 300000)/696000)**(-1.5)))))
        invcube_3 = inversefunc(cube_3, y_values=freq_check_micro[i])
        # plt.plot(freq[i], invcube_3)
        # plt.show()
        # plt.close()
        
        slope = numerical_diff_allen(factor, velocity, invcube_3, h_start)
        freq_drift_micro.append(-slope)

        # time_list.append([inversefunc(cube_3, y_values=37.5)-inversefunc(cube_3, y_values=40), inversefunc(cube_3, y_values=32.5)-inversefunc(cube_3, y_values=35)])
        
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

file_gain = '/Users/yuichiro/Downloads/SN_d_tot_V2.0.csv'



sunspot_obs_times = []
sunspot_num_list = []
print (file_gain)
csv_input = pd.read_csv(filepath_or_buffer= file_gain, sep=";")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    BG_obs_time_event = datetime.datetime(csv_input['Year'][i], csv_input['Month'][i], csv_input['Day'][i])
    if (BG_obs_time_event >= datetime.datetime(2007, 1, 1)) & (BG_obs_time_event <= datetime.datetime(2021, 1, 1)):
        sunspot_num = csv_input['sunspot_number'][i]
        if not sunspot_num == -1:
            sunspot_obs_times.append(BG_obs_time_event)
            # Frequency_list = csv_input['Frequency'][i]
            sunspot_num_list.append(sunspot_num)
        else:
            print (BG_obs_time_event)
sunspot_obs_times = np.array(sunspot_obs_times)
sunspot_num_list = np.array(sunspot_num_list)


fontsize_title = 24
fontsize_labelsize = 20

# burst_types = ["Micro type III burst", "Ordinary type III burst"]


with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_micro_LL_RR.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "velocity", "residual", "event_start", "event_end", "freq_start", "freq_end", "factor", "peak_time_list", "peak_freq_list", "peak_RR_40MHz", "peak_LL_40MHz", "peak_intensity_calibrated", "BG_RR_40MHz", "BG_LL_40MHz", "sunspots_num", "Gain_RR_40MHz", "Gain_LL_40MHz", "Trx_RR_40MHz", "Trx_LL_40MHz", "Polarization", "cos", "Fixed_gain", "40MHz", "drift_rate_40MHz"])
    w.writeheader()
    burst_types = ["Micro type III burst"]
    # fig = plt.figure(figsize=(36.0, 36.0))
    # gs = gridspec.GridSpec(315, 4)
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
            velocity_list = velocity_list_micro
            residual_list = residual_list_micro
            event_start = event_start_micro
            event_end = event_end_micro
            freq_start = freq_start_micro
            freq_end = freq_end_micro
            peak_time = peak_time_list_micro
            peak_freq = peak_freq_list_micro
            best_factor = best_factor_micro
            freq_check_40  = freq_check_micro
            freq_drift_40  = freq_drift_micro
    
        elif burst_type == "Ordinary type III burst":
            obs_times = bursts_obs_times_od
            time_only = bursts_times_only_od
            peak_LL = peak_LL_od
            peak_RR = peak_RR_od
            cos_val = cos_od
            velocity_list = velocity_list_od
            residual_list = residual_list_od
            event_start = event_start_od
            event_end = event_end_od
            freq_start = freq_start_od
            freq_end = freq_end_od
            peak_time = peak_time_list_od
            peak_freq = peak_freq_list_od
            best_factor = best_factor_od
            freq_check_40  = freq_check_od
            freq_drift_40  = freq_drift_od
    
        for i in range(len(obs_times)):
    
            if np.abs(getNearestValue(BG_obs_times, obs_times[i])-obs_times[i]) <= datetime.timedelta(seconds=60*90):
                freq_40 = 40
                if (freq_start[i] >= freq_40) & (freq_end[i] <= freq_40):
                    BG_idx = np.where(BG_obs_times == getNearestValue(BG_obs_times, obs_times[i]))
                    obs_time_list.append(obs_times[i])
                    print (obs_times[i])
                    sunidx = np.where(sunspot_obs_times == datetime.datetime(obs_times[i].year, obs_times[i].month, obs_times[i].day))[0][0]
                    sunspots_num = sunspot_num_list[sunidx]
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
                    w.writerow({'obs_time':obs_times[i], 'velocity': velocity_list[i], 'residual':residual_list[i], 'event_start':event_start[i], 'event_end':event_end[i], 'freq_start':freq_start[i], 'freq_end':freq_end[i], 'factor':best_factor[i], 'peak_time_list':peak_time[i], 'peak_freq_list':peak_freq[i], 'peak_RR_40MHz':peak_RR[i], 'peak_LL_40MHz':peak_LL[i], 'peak_intensity_calibrated':intensity_db_cos[0], 'BG_RR_40MHz':cali_move_bg_r[0], 'BG_LL_40MHz':cali_move_bg_l[0], 'sunspots_num':sunspots_num, 'Gain_RR_40MHz':gain_r[0], 'Gain_LL_40MHz':gain_l[0], 'Trx_RR_40MHz':Trx_RR[BG_idx][0], 'Trx_LL_40MHz':Trx_LL[BG_idx][0], 'Polarization':pol, 'cos':cos, 'Fixed_gain':Used_db[0], '40MHz':freq_check_40[i], 'drift_rate_40MHz':freq_drift_40[i]})
                else:
                    sunidx = np.where(sunspot_obs_times == datetime.datetime(obs_times[i].year, obs_times[i].month, obs_times[i].day))[0][0]
                    sunspots_num = sunspot_num_list[sunidx]
                    w.writerow({'obs_time':obs_times[i], 'velocity': velocity_list[i], 'residual':residual_list[i], 'event_start':event_start[i], 'event_end':event_end[i], 'freq_start':freq_start[i], 'freq_end':freq_end[i], 'factor':best_factor[i], 'peak_time_list':peak_time[i], 'peak_freq_list':peak_freq[i], 'peak_RR_40MHz':peak_RR[i], 'peak_LL_40MHz':peak_LL[i], 'peak_intensity_calibrated':np.nan, 'BG_RR_40MHz':np.nan, 'BG_LL_40MHz':np.nan, 'sunspots_num':sunspots_num, 'Gain_RR_40MHz':np.nan, 'Gain_LL_40MHz':np.nan, 'Trx_RR_40MHz':np.nan, 'Trx_LL_40MHz':np.nan, 'Polarization':np.nan, 'cos':cos_val[i], 'Fixed_gain':np.nan, '40MHz':freq_check_40[i], 'drift_rate_40MHz':np.nan})
            else:
                sunidx = np.where(sunspot_obs_times == datetime.datetime(obs_times[i].year, obs_times[i].month, obs_times[i].day))[0][0]
                sunspots_num = sunspot_num_list[sunidx]
                freq_40 = 40
                if (freq_start[i] >= freq_40) & (freq_end[i] <= freq_40):
                    w.writerow({'obs_time':obs_times[i], 'velocity': velocity_list[i], 'residual':residual_list[i], 'event_start':event_start[i], 'event_end':event_end[i], 'freq_start':freq_start[i], 'freq_end':freq_end[i], 'factor':best_factor[i], 'peak_time_list':peak_time[i], 'peak_freq_list':peak_freq[i], 'peak_RR_40MHz':peak_RR[i], 'peak_LL_40MHz':peak_LL[i], 'peak_intensity_calibrated':np.nan, 'BG_RR_40MHz':np.nan, 'BG_LL_40MHz':np.nan, 'sunspots_num':sunspots_num, 'Gain_RR_40MHz':np.nan, 'Gain_LL_40MHz':np.nan, 'Trx_RR_40MHz':np.nan, 'Trx_LL_40MHz':np.nan, 'Polarization':np.nan, 'cos':cos_val[i], 'Fixed_gain':np.nan, '40MHz':freq_check_40[i], 'drift_rate_40MHz':freq_drift_40[i]})
                else:
                    w.writerow({'obs_time':obs_times[i], 'velocity': velocity_list[i], 'residual':residual_list[i], 'event_start':event_start[i], 'event_end':event_end[i], 'freq_start':freq_start[i], 'freq_end':freq_end[i], 'factor':best_factor[i], 'peak_time_list':peak_time[i], 'peak_freq_list':peak_freq[i], 'peak_RR_40MHz':peak_RR[i], 'peak_LL_40MHz':peak_LL[i], 'peak_intensity_calibrated':np.nan, 'BG_RR_40MHz':np.nan, 'BG_LL_40MHz':np.nan, 'sunspots_num':sunspots_num, 'Gain_RR_40MHz':np.nan, 'Gain_LL_40MHz':np.nan, 'Trx_RR_40MHz':np.nan, 'Trx_LL_40MHz':np.nan, 'Polarization':np.nan, 'cos':cos_val[i], 'Fixed_gain':np.nan, '40MHz':freq_check_40[i], 'drift_rate_40MHz':np.nan})
                # w.writerow({'obs_time': obs_time_each, '30MHz[dB]': around_30, '32.5MHz[dB]':around_32_5, '35MHz[dB]': around_35, '37.5MHz[dB]':around_37_5, '40MHz[dB]': around_40})
    
    
        # obs_time_list = np.array(obs_time_list)
        # time_only_list = np.array(time_only_list)
        # intensity_db_list = np.array(intensity_db_list)
        # intensity_db_cos_list = np.array(intensity_db_cos_list)
        # cos_list = np.array(cos_list)
        # pol_list = np.array(pol_list)
    
        # Gain_RR_list = np.array(Gain_RR_list)
        # Gain_LL_list = np.array(Gain_LL_list)
        # Trx_LL_list = np.array(Trx_LL_list)
        # Trx_RR_list = np.array(Trx_RR_list)
        # cali_move_bg_l_list = np.array(cali_move_bg_l_list)
        # cali_move_bg_r_list = np.array(cali_move_bg_r_list)
        # peak_l_list = np.array(peak_l_list)
        # peak_r_list = np.array(peak_r_list)
        # Trx_cali_list = np.array(Trx_cali_list)
        # # intensity_db_r_list = np.array(intensity_db_r_list)
        # # intensity_db_l_list = np.array(intensity_db_l_list)