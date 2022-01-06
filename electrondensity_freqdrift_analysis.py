#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 09:35:03 2021

@author: yuichiro
"""


def analysis_bursts_peak(csv_input_final, burst_type):
    # print (SDATE)
    obs_time = []
    peak_time_list = []
    peak_freq_list = []
    event_start_list = []
    event_end_list = []
    freq_start_list = []
    freq_end_list = []
    sunspots_num_list = []


        
    for j in range(len(csv_input_final)):

        peak_time_list.append(np.array([float(k) for k in csv_input_final["peak_time_list"][j][1:-1].replace(' ', '').split(',') if k != '']))
        peak_freq_list.append(np.array([float(k) for k in csv_input_final["peak_freq_list"][j][1:-1].replace(' ', '').split(',') if k != '']))
        obs_date = datetime.datetime(int(csv_input_final['obs_time'][j].split('-')[0]), int(csv_input_final['obs_time'][j].split('-')[1]), int(csv_input_final['obs_time'][j].split(' ')[0][-2:]), int(csv_input_final['obs_time'][j].split(' ')[1][:2]), int(csv_input_final['obs_time'][j].split(':')[1]), int(csv_input_final['obs_time'][j].split(':')[2][:2]))
        obs_time.append(obs_date)
        event_start_list.append(int(csv_input_final['event_start'][j]))
        event_end_list.append(int(csv_input_final['event_end'][j]))
        freq_start_list.append(float(csv_input_final['freq_start'][j]))
        freq_end_list.append(float(csv_input_final['freq_end'][j]))
        sunspots_num_list.append(int(csv_input_final['sunspots_num'][j]))

    return obs_time, peak_time_list, peak_freq_list, event_start_list, event_end_list, freq_start_list, freq_end_list, sunspots_num_list

def numerical_diff_allen_fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_allen_2fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 2 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t+h) * velocity * 300000)/696000)**(-1.5))))
    f_2 = 9 * 2 * 10 * np.sqrt(factor * (2.99 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-16)) + 1.55 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-6)) + 0.036 * ((h_start + ((t-h) * velocity * 300000)/696000)**(-1.5))))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_newkirk_fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + ((t+h) * velocity * 300000)/696000)))
    f_2 = 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + ((t-h) * velocity * 300000)/696000)))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_newkirk_2fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + ((t+h) * velocity * 300000)/696000)))
    f_2 = 9 * 2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + ((t-h) * velocity * 300000)/696000)))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_wang_max_fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + ((t+h) * velocity * 300000)/696000) + 5.41656e+07/(h_start + ((t+h) * velocity * 300000)/696000)**2 - 1.86150e+08 /(h_start + ((t+h) * velocity * 300000)/696000)**3 + 2.13102e+08/(h_start + ((t+h) * velocity * 300000)/696000)**4))
    f_2 = 9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + ((t-h) * velocity * 300000)/696000) + 5.41656e+07/(h_start + ((t-h) * velocity * 300000)/696000)**2 - 1.86150e+08 /(h_start + ((t-h) * velocity * 300000)/696000)**3 + 2.13102e+08/(h_start + ((t-h) * velocity * 300000)/696000)**4))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_wang_max_2fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + ((t+h) * velocity * 300000)/696000) + 5.41656e+07/(h_start + ((t+h) * velocity * 300000)/696000)**2 - 1.86150e+08 /(h_start + ((t+h) * velocity * 300000)/696000)**3 + 2.13102e+08/(h_start + ((t+h) * velocity * 300000)/696000)**4))
    f_2= 9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + ((t-h) * velocity * 300000)/696000) + 5.41656e+07/(h_start + ((t-h) * velocity * 300000)/696000)**2 - 1.86150e+08 /(h_start + ((t-h) * velocity * 300000)/696000)**3 + 2.13102e+08/(h_start + ((t-h) * velocity * 300000)/696000)**4))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_wang_min_fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 1e-3 * np.sqrt(factor * (353766/(h_start + ((t+h) * velocity * 300000)/696000) + 1.03359e+07/(h_start + ((t+h) * velocity * 300000)/696000)**2 - 5.46541e+07 /(h_start + ((t+h) * velocity * 300000)/696000)**3 + 8.24791e+07/(h_start + ((t+h) * velocity * 300000)/696000)**4))
    f_2 = 9 * 1e-3 * np.sqrt(factor * (353766/(h_start + ((t-h) * velocity * 300000)/696000) + 1.03359e+07/(h_start + ((t-h) * velocity * 300000)/696000)**2 - 5.46541e+07 /(h_start + ((t-h) * velocity * 300000)/696000)**3 + 8.24791e+07/(h_start + ((t-h) * velocity * 300000)/696000)**4))
    return ((f_1 - f_2)/(2*h))

def numerical_diff_wang_min_2fp(factor, velocity, t, h_start):
    h = 1e-4
    f_1= 9 * 2 * 1e-3 * np.sqrt(factor * (353766/(h_start + ((t+h) * velocity * 300000)/696000) + 1.03359e+07/(h_start + ((t+h) * velocity * 300000)/696000)**2 - 5.46541e+07 /(h_start + ((t+h) * velocity * 300000)/696000)**3 + 8.24791e+07/(h_start + ((t+h) * velocity * 300000)/696000)**4))
    f_2= 9 * 2 * 1e-3 * np.sqrt(factor * (353766/(h_start + ((t-h) * velocity * 300000)/696000) + 1.03359e+07/(h_start + ((t-h) * velocity * 300000)/696000)**2 - 5.46541e+07 /(h_start + ((t-h) * velocity * 300000)/696000)**3 + 8.24791e+07/(h_start + ((t-h) * velocity * 300000)/696000)**4))
    return ((f_1 - f_2)/(2*h))

def allen_model_fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5)))))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_allen_fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)

def allen_model_2fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 2 * 10 * np.sqrt(factor * (2.99*((1+(h/696000))**(-16))+1.55*((1+(h/696000))**(-6))+0.036*((1+(h/696000))**(-1.5)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 2 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate3 * 300000)/696000)**(-1.5)))))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 2 * 10 * np.sqrt(factor * (2.99 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-16)) + 1.55 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-6)) + 0.036 * ((h_start + (h5 * time_rate4 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_allen_2fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)


def newkirk_model_fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + (h5 * time_rate3 * 300000)/696000))))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + (h5 * time_rate4 * 300000)/696000))))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_newkirk_fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)

def newkirk_model_2fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 *2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(1+(h/696000)))))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + (h5 * time_rate3 * 300000)/696000))))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(h_start + (h5 * time_rate4 * 300000)/696000))))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_newkirk_2fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)



def wang_max_model_fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(1+(h/696000)) + 5.41656e+07/(1+(h/696000))**2 - 1.86150e+08 /(1+(h/696000))**3 + 2.13102e+08/(1+(h/696000))**4)))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + (h5 * time_rate3 * 300000)/696000) + 5.41656e+07/(h_start + (h5 * time_rate3 * 300000)/696000)**2 - 1.86150e+08 /(h_start + (h5 * time_rate3 * 300000)/696000)**3 + 2.13102e+08/(h_start + (h5 * time_rate3 * 300000)/696000)**4)))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + (h5 * time_rate4 * 300000)/696000) + 5.41656e+07/(h_start + (h5 * time_rate4 * 300000)/696000)**2 - 1.86150e+08 /(h_start + (h5 * time_rate4 * 300000)/696000)**3 + 2.13102e+08/(h_start + (h5 * time_rate4 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_wang_max_fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)


def wang_max_model_2fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(1+(h/696000)) + 5.41656e+07/(1+(h/696000))**2 - 1.86150e+08 /(1+(h/696000))**3 + 2.13102e+08/(1+(h/696000))**4)))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + (h5 * time_rate3 * 300000)/696000) + 5.41656e+07/(h_start + (h5 * time_rate3 * 300000)/696000)**2 - 1.86150e+08 /(h_start + (h5 * time_rate3 * 300000)/696000)**3 + 2.13102e+08/(h_start + (h5 * time_rate3 * 300000)/696000)**4)))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(h_start + (h5 * time_rate4 * 300000)/696000) + 5.41656e+07/(h_start + (h5 * time_rate4 * 300000)/696000)**2 - 1.86150e+08 /(h_start + (h5 * time_rate4 * 300000)/696000)**3 + 2.13102e+08/(h_start + (h5 * time_rate4 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_wang_max_2fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)


def wang_min_model_fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 1e-3 * np.sqrt(factor * (353766/(1+(h/696000)) + 1.03359e+07/(1+(h/696000))**2 - 5.46541e+07 /(1+(h/696000))**3 + 8.24791e+07/(1+(h/696000))**4)))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 1e-3 * np.sqrt(factor * (353766/(h_start + (h5 * time_rate3 * 300000)/696000) + 1.03359e+07/(h_start + (h5 * time_rate3 * 300000)/696000)**2 - 5.46541e+07 /(h_start + (h5 * time_rate3 * 300000)/696000)**3 + 8.24791e+07/(h_start + (h5 * time_rate3 * 300000)/696000)**4)))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 1e-3 * np.sqrt(factor * (353766/(h_start + (h5 * time_rate4 * 300000)/696000) + 1.03359e+07/(h_start + (h5 * time_rate4 * 300000)/696000)**2 -  5.46541e+07 /(h_start + (h5 * time_rate4 * 300000)/696000)**3 + 8.24791e+07/(h_start + (h5 * time_rate4 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_wang_min_fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)

def wang_min_model_2fp(factor_num, time_list, freq_list):
    i_value = np.array(freq_list)
    factor = factor_num
    fitting = []
    time_rate_result = []
    fitting_new = []
    time_rate_result_new = []
    slide_result_new = []
    freq_max = max(freq_list)
    cube_4 = (lambda h: 9 * 2 * 1e-3 * np.sqrt(factor * (353766/(1+(h/696000)) + 1.03359e+07/(1+(h/696000))**2 - 5.46541e+07 /(1+(h/696000))**3 + 8.24791e+07/(1+(h/696000))**4)))
    invcube_4 = inversefunc(cube_4, y_values = freq_max)
    h_start = invcube_4/696000 + 1
    
    fitting.append(100)
    time_rate_result.append(100)
    
    for i in range(10, 100, 10):
        time_rate3 = i/100
        cube_3 = (lambda h5: 9 * 2 * 1e-3 * np.sqrt(factor * (353766/(h_start + (h5 * time_rate3 * 300000)/696000) + 1.03359e+07/(h_start + (h5 * time_rate3 * 300000)/696000)**2 - 5.46541e+07 /(h_start + (h5 * time_rate3 * 300000)/696000)**3 + 8.24791e+07/(h_start + (h5 * time_rate3 * 300000)/696000)**4)))
        invcube_3 = inversefunc(cube_3, y_values=i_value)
        s_0 = sum(invcube_3-time_list)/len(freq_list)
        residual_0 = -s_0**2 + sum((invcube_3 - time_list)**2)/len(freq_list)
        if min(fitting) > residual_0:
            fitting.append(residual_0)
            time_rate_result.append(time_rate3)
    

    fitting_new.append(100)
    time_rate_result_new.append(100)
    slide_result_new.append(100)
    if int(time_rate_result[-1]*100) == 10:
        time_new = 11
    else:
        time_new = int(time_rate_result[-1]*100)
    for i in range(time_new -10, time_new + 10, 1):
    #    print(i)
        time_rate4 = i/100
        cube_4 = (lambda h5: 9 * 2 * 1e-3 * np.sqrt(factor * (353766/(h_start + (h5 * time_rate4 * 300000)/696000) + 1.03359e+07/(h_start + (h5 * time_rate4 * 300000)/696000)**2 -  5.46541e+07 /(h_start + (h5 * time_rate4 * 300000)/696000)**3 + 8.24791e+07/(h_start + (h5 * time_rate4 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=i_value)
        s_1 = sum(invcube_4-time_list)/len(freq_list)
        residual_1 = -s_1**2 + sum((invcube_4-time_list)**2)/len(freq_list)
        if min(fitting_new) > residual_1:
            fitting_new.append(residual_1)
            time_rate_result_new.append(time_rate4)
            slide_result_new.append(s_1)
    t_40MHz = inversefunc(cube_4, y_values=40)
    slope = numerical_diff_wang_min_2fp(factor, time_rate_result_new[-1], t_40MHz, h_start)
    # print (h_start)
    return (-slide_result_new[-1], time_rate_result_new[-1], np.sqrt(fitting_new[-1]), h_start, slope, t_40MHz)



def residual_detection_allen_fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz = allen_model_fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        dfdt.append(-1*slope)
        h_start.append(start)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 10 * np.sqrt(factor * (2.99 * ((start + ((h5_0) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((start + ((h5_0) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((start + ((h5_0) * time_rate5 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)

        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 20)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 10 * np.sqrt(factor * (2.99 * ((start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))

    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_allen_2fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = allen_model_2fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        dfdt.append(-1*slope)
        h_start.append(start)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 2 * 10 * np.sqrt(factor * (2.99 * ((start + ((h5_0) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((start + ((h5_0) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((start + ((h5_0) * time_rate5 * 300000)/696000)**(-1.5)))))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)

        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 2 * 10 * np.sqrt(factor * (2.99 * ((start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-16)) + 1.55 * ((start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-6)) + 0.036 * ((start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**(-1.5)))))

    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_newkirk_fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = newkirk_model_fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        dfdt.append(-1*slope)
        h_start.append(start)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(start + ((h5_0) * time_rate5 * 300000)/696000))))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)

        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000))))
        

    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_newkirk_2fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = newkirk_model_2fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        dfdt.append(-1*slope)
        h_start.append(start)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(start + ((h5_0) * time_rate5 * 300000)/696000))))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)

        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 2 * 1e-3 * np.sqrt(factor * 4.2 * 10 ** (4+4.32/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000))))

    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_wang_max_fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = wang_max_model_fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        dfdt.append(-1*slope)
        h_start.append(start)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(start + ((h5_0) * time_rate5 * 300000)/696000) + 5.41656e+07/(start + ((h5_0) * time_rate5 * 300000)/696000)**2 - 1.86150e+08 /(start + ((h5_0) * time_rate5 * 300000)/696000)**3 + 2.13102e+08/(start + ((h5_0) * time_rate5 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)
        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000) + 5.41656e+07/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**2 - 1.86150e+08 /(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**3 + 2.13102e+08/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**4)))

    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_wang_max_2fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = wang_max_model_2fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        h_start.append(start)
        dfdt.append(-1*slope)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(start + ((h5_0) * time_rate5 * 300000)/696000) + 5.41656e+07/(start + ((h5_0) * time_rate5 * 300000)/696000)**2 - 1.86150e+08 /(start + ((h5_0) * time_rate5 * 300000)/696000)**3 + 2.13102e+08/(start + ((h5_0) * time_rate5 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)
        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 2 * 1e-3 * np.sqrt(factor * (-4.42158e+06/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000) + 5.41656e+07/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**2 - 1.86150e+08 /(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**3 + 2.13102e+08/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**4)))

    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_wang_min_fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = wang_min_model_fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        h_start.append(start)
        dfdt.append(-1*slope)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 1e-3 * np.sqrt(factor * (353766/(start + ((h5_0) * time_rate5 * 300000)/696000) + 1.03359e+07/(start + ((h5_0) * time_rate5 * 300000)/696000)**2 - 5.46541e+07 /(start + ((h5_0) * time_rate5 * 300000)/696000)**3 + 8.24791e+07/(start + ((h5_0) * time_rate5 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)
        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 1e-3 * np.sqrt(factor * (353766/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000) + 1.03359e+07/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**2 - 5.46541e+07 /(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**3 + 8.24791e+07/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**4)))


    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list

def residual_detection_wang_min_2fp(factor_list, freq_list, time_list):
    time_rate_final = []
    residual_list = []
    slide_list = []
    x_time = []
    y_freq = []
    dfdt = []
    h_start = []
    t_40MHz_list = []
    for factor in factor_list:
        slide, time_rate5, residual, start, slope, t_40MHz  = wang_min_model_2fp(factor, time_list, freq_list)
        time_rate_final.append(time_rate5)
        residual_list.append(residual)
        slide_list.append(slide)
        h_start.append(start)
        dfdt.append(-1*slope)
        t_40MHz_list.append(t_40MHz)

        cube_4 = (lambda h5_0: 9 * 2 * 1e-3 * np.sqrt(factor * (353766/(start + ((h5_0) * time_rate5 * 300000)/696000) + 1.03359e+07/(start + ((h5_0) * time_rate5 * 300000)/696000)**2 - 5.46541e+07 /(start + ((h5_0) * time_rate5 * 300000)/696000)**3 + 8.24791e+07/(start + ((h5_0) * time_rate5 * 300000)/696000)**4)))
        invcube_4 = inversefunc(cube_4, y_values=freq_list)
        x_time.append(invcube_4 + slide)
        y_freq.append(freq_list)

        # h5_0 = np.arange(np.ceil(slide*1000)- 3000, (max(time_list) + 5) * 1000, 10)
        # h5_0 = h5_0/1000
        # x_time.append(h5_0)
        # y_freq.append(9 * 2 * 1e-3 * np.sqrt(factor * (353766/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000) + 1.03359e+07/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**2 - 5.46541e+07 /(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**3 + 8.24791e+07/(start + ((h5_0 - slide) * time_rate5 * 300000)/696000)**4)))


    return residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list


import pandas as pd
import numpy as np
import datetime
from pynverse import inversefunc
import csv
import matplotlib.pyplot as plt
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
factor_list = [1,2,3,4,5]



if __name__=='__main__':
    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_ordinary_withnonclear.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


    try:
        obs_time_ordinary, peak_time_list_ordinary, peak_freq_list_ordinary, event_start_list_ordinary, event_end_list_ordinary, freq_start_list_ordinary, freq_end_list_ordinary, sunspots_num_list_ordinary = analysis_bursts_peak(csv_input_final, 'ordinary')
    except:
        print('DL error')



    file_final = "/solar_burst/Nancay/af_sgepss_analysis_data/burst_analysis/shuron_data/shuron_micro_LL_RR.csv"
    csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")
    

    try:
        obs_time_micro, peak_time_list_micro, peak_freq_list_micro, event_start_list_micro, event_end_list_micro, freq_start_list_micro, freq_end_list_micro, sunspots_num_list_micro = analysis_bursts_peak(csv_input_final, 'micro')
    except:
        print('DL error')

with open(Parent_directory+ '/solar_burst/Nancay/af_sgepss_analysis_data/electrondensity_anaysis_shuron3.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["obs_time", "burst_type", "event_start", "event_end", "freq_start", "freq_end", "peak_time_list", "peak_freq_list", "sunspots_num", "allen_fp_dfdt_list", "allen_fp_residual", "allen_fp_velocity", "allen_2fp_dfdt_list", "allen_2fp_residual", "allen_2fp_velocity", "newkirk_fp_dfdt_list", "newkirk_fp_residual", "newkirk_fp_velocity", "newkirk_2fp_dfdt_list", "newkirk_2fp_residual", "newkirk_2fp_velocity", "wangmax_fp_dfdt_list", "wangmax_fp_residual", "wangmax_fp_velocity", "wangmax_2fp_dfdt_list", "wangmax_2fp_residual", "wangmax_2fp_velocity", "wangmin_fp_dfdt_list", "wangmin_fp_residual", "wangmin_fp_velocity", "wangmin_2fp_dfdt_list", "wangmin_2fp_residual", "wangmin_2fp_velocity"])
    w.writeheader()

    factor_all = []
    residual_all = []
    for i in range(len(obs_time_ordinary)):
        burst_type = 'ordinary'
        # factor_list_all = []
        # residual_list_all = []
        # dfdt_list = []
        # velocity_list = []
        
        freq_list = peak_freq_list_ordinary[i]
        time_list = peak_time_list_ordinary[i]
        plt.plot(time_list, freq_list, '.')
        obs_time =  obs_time_ordinary[i]
        event_start = event_start_list_ordinary[i]
        event_end = event_end_list_ordinary[i]
        freq_start = freq_start_list_ordinary[i]
        freq_end = freq_end_list_ordinary[i]
        sunspots_num = sunspots_num_list_ordinary[i]
    
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_allen_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('B-A fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        allen_fp_residual = residual_list
        allen_fp_dfdt_list = dfdt
        allen_fp_velocity = time_rate_final
        allen_fp_factor = factor
        
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× B-A(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_allen_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('B-A 2fp: h_start=' + str(h_start[idx]) + '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        allen_2fp_residual = residual_list
        allen_2fp_dfdt_list = dfdt
        allen_2fp_velocity = time_rate_final
        allen_2fp_factor = factor

        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× B-A(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_newkirk_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Newkirk fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1

        newkirk_fp_residual = residual_list
        newkirk_fp_dfdt_list = dfdt
        newkirk_fp_velocity = time_rate_final
        newkirk_fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Newkirk(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
    
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_newkirk_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Newkirk 2fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        newkirk_2fp_residual = residual_list
        newkirk_2fp_dfdt_list = dfdt
        newkirk_2fp_velocity = time_rate_final
        newkirk_2fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Newkirk(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_max_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang max fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmax_fp_residual = residual_list
        wangmax_fp_dfdt_list = dfdt
        wangmax_fp_velocity = time_rate_final
        wangmax_fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
    
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_max_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang max 2fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmax_2fp_residual = residual_list
        wangmax_2fp_dfdt_list = dfdt
        wangmax_2fp_velocity = time_rate_final
        wangmax_2fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))


        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_min_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang min fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmin_fp_residual = residual_list
        wangmin_fp_dfdt_list = dfdt
        wangmin_fp_velocity = time_rate_final
        wangmin_fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
    

        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_min_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang min 2fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmin_2fp_residual = residual_list
        wangmin_2fp_dfdt_list = dfdt
        wangmin_2fp_velocity = time_rate_final
        wangmin_2fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        

        plt.ylim(30, 80)
        plt.legend()
        filename = Parent_directory + '/solar_burst/Nancay/plot/density_analysis/'+burst_type+'/'+obs_time.strftime("%Y%m%d%H%M")+ '.png'
        plt.savefig(filename)
        # plt.show()
        plt.close()
        # factor_all.append(factor_list_all)
        # residual_all.append(residual_list)
        w.writerow({'obs_time':obs_time, 'burst_type':burst_type, 'event_start':event_start,'event_end':event_end, 'freq_start':freq_start, 'freq_end':freq_end,'peak_time_list':time_list,'peak_freq_list':freq_list, 'sunspots_num':sunspots_num, 'allen_fp_dfdt_list':allen_fp_dfdt_list,'allen_fp_residual':allen_fp_residual, 'allen_fp_velocity':allen_fp_velocity, 'allen_2fp_dfdt_list':allen_2fp_dfdt_list, 'allen_2fp_residual':allen_2fp_residual, 'allen_2fp_velocity':allen_2fp_velocity,'newkirk_fp_dfdt_list':newkirk_fp_dfdt_list, 'newkirk_fp_residual':newkirk_fp_residual, 'newkirk_fp_velocity':newkirk_fp_velocity,'newkirk_2fp_dfdt_list':newkirk_2fp_dfdt_list, 'newkirk_2fp_residual':newkirk_2fp_residual, 'newkirk_2fp_velocity':newkirk_2fp_velocity,'wangmax_fp_dfdt_list':wangmax_fp_dfdt_list, 'wangmax_fp_residual':wangmax_fp_residual, 'wangmax_fp_velocity':wangmax_fp_velocity, 'wangmax_2fp_dfdt_list':wangmax_2fp_dfdt_list,'wangmax_2fp_residual':wangmax_2fp_residual, 'wangmax_2fp_velocity':wangmax_2fp_velocity, 'wangmin_fp_dfdt_list':wangmin_fp_dfdt_list, 'wangmin_fp_residual':wangmin_fp_residual, 'wangmin_fp_velocity':wangmin_fp_velocity,'wangmin_2fp_dfdt_list':wangmin_2fp_dfdt_list, 'wangmin_2fp_residual':wangmin_2fp_residual, 'wangmin_2fp_velocity':wangmin_2fp_velocity})

    # factor_all = []
    # residual_all = []
    for i in range(len(obs_time_micro)):
        burst_type = 'micro'
        # factor_list_all = []
        # residual_list_all = []
        # dfdt_list = []
        # velocity_list = []
        
        freq_list = peak_freq_list_micro[i]
        time_list = peak_time_list_micro[i]
        plt.plot(time_list, freq_list, '.')
        obs_time =  obs_time_micro[i]
        event_start = event_start_list_micro[i]
        event_end = event_end_list_micro[i]
        freq_start = freq_start_list_micro[i]
        freq_end = freq_end_list_micro[i]
        sunspots_num = sunspots_num_list_micro[i]
    
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_allen_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('B-A fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        allen_fp_residual = residual_list
        allen_fp_dfdt_list = dfdt
        allen_fp_velocity = time_rate_final
        allen_fp_factor = factor
        
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× B-A(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_allen_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('B-A 2fp: h_start=' + str(h_start[idx]) + '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        allen_2fp_residual = residual_list
        allen_2fp_dfdt_list = dfdt
        allen_2fp_velocity = time_rate_final
        allen_2fp_factor = factor

        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× B-A(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_newkirk_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Newkirk fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1

        newkirk_fp_residual = residual_list
        newkirk_fp_dfdt_list = dfdt
        newkirk_fp_velocity = time_rate_final
        newkirk_fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Newkirk(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
    
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_newkirk_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Newkirk 2fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        newkirk_2fp_residual = residual_list
        newkirk_2fp_dfdt_list = dfdt
        newkirk_2fp_velocity = time_rate_final
        newkirk_2fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Newkirk(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_max_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang max fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmax_fp_residual = residual_list
        wangmax_fp_dfdt_list = dfdt
        wangmax_fp_velocity = time_rate_final
        wangmax_fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
    
        
        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_max_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang max 2fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmax_2fp_residual = residual_list
        wangmax_2fp_dfdt_list = dfdt
        wangmax_2fp_velocity = time_rate_final
        wangmax_2fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))


        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_min_fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang min fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmin_fp_residual = residual_list
        wangmin_fp_dfdt_list = dfdt
        wangmin_fp_velocity = time_rate_final
        wangmin_fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
    

        residual_list, x_time, y_freq, time_rate_final, dfdt, h_start, t_40MHz_list = residual_detection_wang_min_2fp(factor_list, freq_list, time_list)
        idx = np.where(np.array(residual_list)==np.min(residual_list))[0][0]
        print ('Wang min 2fp: h_start=' + str(h_start[idx])+ '  t40MHz=' + str(t_40MHz_list[idx]))
        factor = idx+1
        wangmin_2fp_residual = residual_list
        wangmin_2fp_dfdt_list = dfdt
        wangmin_2fp_velocity = time_rate_final
        wangmin_2fp_factor = factor
        # residual_list_all.append(residual_list[idx])
        # dfdt_list.append(dfdt[idx])
        # velocity_list.append(time_rate_final[idx])
        # factor_list_all.append(factor)
        plt.plot(x_time[idx], y_freq[idx], label = str(factor)+'× Wang_max(2fp): vr=' + str(time_rate_final[idx]) + ' df/dt='+str(round(dfdt[idx],1)))
        

        plt.ylim(30, 80)
        plt.legend()
        filename = Parent_directory + '/solar_burst/Nancay/plot/density_analysis/'+burst_type+'/'+obs_time.strftime("%Y%m%d%H%M")+ '.png'
        plt.savefig(filename)
        # plt.show()
        plt.close()
        # factor_all.append(factor_list_all)
        # residual_all.append(residual_list)
        w.writerow({'obs_time':obs_time, 'burst_type':burst_type, 'event_start':event_start,'event_end':event_end, 'freq_start':freq_start, 'freq_end':freq_end,'peak_time_list':time_list,'peak_freq_list':freq_list, 'sunspots_num':sunspots_num, 'allen_fp_dfdt_list':allen_fp_dfdt_list,'allen_fp_residual':allen_fp_residual, 'allen_fp_velocity':allen_fp_velocity, 'allen_2fp_dfdt_list':allen_2fp_dfdt_list, 'allen_2fp_residual':allen_2fp_residual, 'allen_2fp_velocity':allen_2fp_velocity,'newkirk_fp_dfdt_list':newkirk_fp_dfdt_list, 'newkirk_fp_residual':newkirk_fp_residual, 'newkirk_fp_velocity':newkirk_fp_velocity,'newkirk_2fp_dfdt_list':newkirk_2fp_dfdt_list, 'newkirk_2fp_residual':newkirk_2fp_residual, 'newkirk_2fp_velocity':newkirk_2fp_velocity,'wangmax_fp_dfdt_list':wangmax_fp_dfdt_list, 'wangmax_fp_residual':wangmax_fp_residual, 'wangmax_fp_velocity':wangmax_fp_velocity, 'wangmax_2fp_dfdt_list':wangmax_2fp_dfdt_list,'wangmax_2fp_residual':wangmax_2fp_residual, 'wangmax_2fp_velocity':wangmax_2fp_velocity, 'wangmin_fp_dfdt_list':wangmin_fp_dfdt_list, 'wangmin_fp_residual':wangmin_fp_residual, 'wangmin_fp_velocity':wangmin_fp_velocity,'wangmin_2fp_dfdt_list':wangmin_2fp_dfdt_list, 'wangmin_2fp_residual':wangmin_2fp_residual, 'wangmin_2fp_velocity':wangmin_2fp_velocity})