#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 09:57:27 2021

@author: yuichiro
"""
import cv2
import glob
import os
import shutil
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.dates import drange
from matplotlib.dates import date2num


WINDOW_NAME_0 = "wind"
WINDOW_NAME_1 = "geotail"
WINDOW_NAME_2 = "detected type 3 burst  "
WINDOW_NAME_3 = "Nancay QL"
WINDOW_NAME_4 = "Flare"
# WINDOW_NAME_5 = 'SDO_HMIB'
WINDOW_NAME_6 = 'Nancay Wind'
# WINDOW_NAME_7  = 'SDO_0193'
# WINDOW_NAME_8 = 'SDO_0211'

Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")


def ordinary_or_storm():
    i = 0
    while i < 1:
        choice = input("Please respond with 'micro' or 'storm'[o/s]: ").lower()
        if choice in ['s']:
            i += 1
            return 'micro'
        elif choice in ['o']:
            i += 1
            return 'fao'


def wind_geotail_flare(yyyy, mm, dd, WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4):        
    if os.path.isfile("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/plot_QL/" + yyyy + "/wav_summary_" + yyyy + mm + dd + ".png") == True:
        img_wind = cv2.imread("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/plot_QL/" + yyyy + "/wav_summary_" + yyyy + mm + dd + ".png", cv2.IMREAD_COLOR)
        img_wind_1 = img_wind[50:380, 20:790]
        image_wind = cv2.resize(img_wind_1, dsize=None, fx=1.24*factor, fy=1.24*factor)
        cv2.namedWindow(WINDOW_NAME_0, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME_0, image_wind)
        cv2.moveWindow(WINDOW_NAME_0, 1450, 0)
    else:
        print('No data: wind')
    if os.path.isfile('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/geotail/plot_QL/' + yyyy + '/' +yyyy[2:4] + mm + dd +'00.gif') == True:
        gif_geotail = cv2.VideoCapture('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/geotail/plot_QL/' + yyyy + '/' +yyyy[2:4] + mm + dd +'00.gif')
        is_success, img_geotail = gif_geotail.read()
        img_geotail_1 = img_geotail[0:400, 40:1105]
        image_geotail = cv2.resize(img_geotail_1, dsize=None, fx=0.9*factor, fy=0.9*factor)
        cv2.namedWindow(WINDOW_NAME_1, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME_1, image_geotail)
        cv2.moveWindow(WINDOW_NAME_1, 1450, 250)
    else:
        print('No data: Geotail')
    flare_check(yyyy, dd, mm)
    if len(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/flare_plot/' + yyyy+mm+dd +'.png')) == 1:
        img_flare = cv2.imread('/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/flare_plot/' + yyyy+mm+dd +'.png', cv2.IMREAD_COLOR)
        # img_flare_1 = img_flare[30:280, 30:700]
        # image_flare = cv2.resize(img_flare_1, dsize=None, fx=0.95, fy=0.95)
        # cv2.namedWindow(WINDOW_NAME_4, cv2.WINDOW_AUTOSIZE)
        # cv2.imshow(WINDOW_NAME_4, image_flare)
        # cv2.moveWindow(WINDOW_NAME_4, 0, 503)
        img_flare_1 = img_flare[30:280, 40:700]
        image_flare = cv2.resize(img_flare_1, dsize=None, fx=1.455*factor, fy=1.455*factor)
        cv2.namedWindow(WINDOW_NAME_4, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(WINDOW_NAME_4, image_flare)
        cv2.moveWindow(WINDOW_NAME_4, 1450, 451)
        # cv2.moveWindow(WINDOW_NAME_4, 1450, 825*factor)
    else:
        print('No data: Flare')
    cv2.waitKey(1)
    return


def choice():
    i = 0
    while i < 1:
        choice = input("Please respond with 'ordinary', 'storm', 'marginal', 'pass', '1', '2', '3', '4', '5', or '6' [o/s/m/p/1/2/3/4/5/6]: ").lower()
        if choice in ['s']:
            i += 1
            return 'storm'
        elif choice in ['m']:
            i += 1
            return 'marginal'
        elif choice in ['o']:
            i += 1
            return 'ordinary'
        elif choice in ['p']:
            i += 1
            return 'pass'
        elif choice in ['1']:
            i += 1
            return '1'
        elif choice in ['2']:
            i += 1
            return '2'
        elif choice in ['3']:
            i += 1
            return '3'
        elif choice in ['4']:
            i += 1
            return '4'
        elif choice in ['5']:
            i += 1
            return '5'
        elif choice in ['6']:
            i += 1
            return '6'
    return

def window_close(file, files, WINDOW_NAME_2, WINDOW_NAME_3, WINDOW_NAME_5):
    if file == files:
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        cv2.waitKey(1)
        cv2.destroyWindow(WINDOW_NAME_2)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.destroyWindow(WINDOW_NAME_3)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.destroyWindow(WINDOW_NAME_5)
        cv2.waitKey(1)
    return
def window_close_1(WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4):
    cv2.waitKey(1)
    cv2.destroyWindow(WINDOW_NAME_0)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.destroyWindow(WINDOW_NAME_1)
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.destroyWindow(WINDOW_NAME_4)
    cv2.waitKey(1)

    return

def window_close_2(file, files, WINDOW_NAME_2, WINDOW_NAME_3):
    if file == files[-1]:
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    else:
        cv2.waitKey(1)
        cv2.destroyWindow(WINDOW_NAME_2)
        cv2.waitKey(1)
        cv2.waitKey(1)
        cv2.destroyWindow(WINDOW_NAME_3)
        cv2.waitKey(1)
    return

def flare_check(yyyy, dd, mm):
    if csv_input_final['peak'].str.contains(yyyy + '/'+mm+'/'+dd).sum() > 0:
        select_flarelist = csv_input_final[csv_input_final['peak'].str.contains(yyyy + '/'+mm+'/'+dd)]
        
        str_date_flare = select_flarelist['peak'][select_flarelist['peak'].index[0]].split(' ')[0]
        f_yyyy = str_date_flare[:4]
        f_mm = str_date_flare[5:7]
        f_dd = str_date_flare[8:10]
        date1 = datetime.datetime(int(f_yyyy), int(f_mm), int(f_dd), 0)
        date2 = date1 + datetime.timedelta(days=1)
        
        m_size = 0.95
        plt.close(1)
        figure_ = plt.figure(1, figsize=(10,4))
        axes_ = figure_.add_subplot(111)
        
        # plot
        for i in reversed(select_flarelist['start'].index):
            print (select_flarelist['start'][i].split('/')[1] +'/'+ select_flarelist['start'][i].split('/')[2] +' - '+ select_flarelist['end'][i].split('/')[1] +'/'+ select_flarelist['end'][i].split('/')[2] + ' (peak at '+select_flarelist['peak'][i].split('/')[1] +'/'+ select_flarelist['peak'][i].split('/')[2]+')' + '  X-ray class: ' + select_flarelist['X-ray class'][i])
            if select_flarelist['start'][i].split(' ')[0][8:] == f_dd:
                sflare_time = (int(select_flarelist['start'][i].split(' ')[1][:2])*60+int(select_flarelist['start'][i].split(' ')[1][3:]))/(60*24)
                sflare_time_date = date1 + datetime.timedelta(hours = int(select_flarelist['start'][i].split(' ')[1][:2]))+ datetime.timedelta(minutes = int(select_flarelist['start'][i].split(' ')[1][3:]))
            else:
                sflare_time = 0
                sflare_time_date = date1
            if select_flarelist['end'][i].split(' ')[0][8:] == f_dd:
                eflare_time = (int(select_flarelist['end'][i].split(' ')[1][:2])*60+int(select_flarelist['end'][i].split(' ')[1][3:]))/(60*24)
                eflare_time_date = date1 + datetime.timedelta(hours = int(select_flarelist['end'][i].split(' ')[1][:2])) + datetime.timedelta(minutes = int(select_flarelist['end'][i].split(' ')[1][3:]))
            else:
                eflare_time = 24*60
                eflare_time_date = date2
            if select_flarelist['X-ray class'][i][:1][0] == 'A':
                color = 'gray'
            elif select_flarelist['X-ray class'][i][:1][0] == 'B':
                color = 'sienna'
            elif select_flarelist['X-ray class'][i][:1][0] == 'C':
                color = 'tomato'
            elif select_flarelist['X-ray class'][i][:1][0] == 'M':
                color = 'darkred'
            elif select_flarelist['X-ray class'][i][:1][0] == 'X':
                color = 'red'
            axes_.text(date1 + datetime.timedelta(hours = int(select_flarelist['end'][i].split(' ')[1][:2])) + datetime.timedelta(minutes = int(select_flarelist['end'][i].split(' ')[1][3:]) + 30), m_size, select_flarelist['AR location'][i], size = 12, verticalalignment="center")

            axes_.plot([sflare_time_date, eflare_time_date], [m_size, m_size], ls = "-", color = color, linewidth = 10.0)
            axes_.set_xlim(date2num([date1,date2]))
            axes_.set_ylim(0,1)

            m_size -= 0.05
        axes_.axvline(x = date1 + datetime.timedelta(hours=4), ls = "--", color = "k", linewidth = 1.0)
        axes_.axvline(x = date1 + datetime.timedelta(hours=8), ls = "--", color = "k", linewidth = 1.0)
        axes_.axvline(x = date1 + datetime.timedelta(hours=12), ls = "--", color = "k", linewidth = 1.0)
        axes_.axvline(x = date1 + datetime.timedelta(hours=16), ls = "--", color = "k", linewidth = 1.0)
        axes_.axvline(x = date1 + datetime.timedelta(hours=20), ls = "--", color = "k", linewidth = 1.0)
        axes_.axvline(x = date1 + datetime.timedelta(hours=24), ls = "--", color = "k", linewidth = 1.0)
    
        axes_.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        axes_.text(date1 + datetime.timedelta(hours=1), 0.5, "A class", backgroundcolor='gray', size = 10, color="white", fontweight="bold")
        axes_.text(date1 + datetime.timedelta(hours=1), 0.4, "B class", backgroundcolor='sienna', size = 10, color="white", fontweight="bold")
        axes_.text(date1 + datetime.timedelta(hours=1), 0.3, "C class", backgroundcolor='tomato', size = 10, color="white", fontweight="bold")
        axes_.text(date1 + datetime.timedelta(hours=1), 0.2, "M class", backgroundcolor='darkred', size = 10, color="white", fontweight="bold")
        axes_.text(date1 + datetime.timedelta(hours=1), 0.1, "X class", backgroundcolor='red', size = 10, color="white", fontweight="bold")
        
        plt.title('Flare Occurrence Time  ' + date1.strftime("%Y/%m/%d"), fontsize=18)
        plt.tick_params(labelsize=14)
        plt.yticks(color="None")
        plt.xticks(drange(date1, date2, datetime.timedelta(hours=2)), fontsize = 13)
    
        # plt.show()
        #     if select_flarelist['start'][i].split(' ')[0][8:] == f_dd:
        #         sflare_time = (int(select_flarelist['start'][i].split(' ')[1][:2])*60+int(select_flarelist['start'][i].split(' ')[1][3:]))/(60*24)
        #     else:
        #         sflare_time = 0
        #     if select_flarelist['end'][i].split(' ')[0][8:] == f_dd:
        #         eflare_time = (int(select_flarelist['end'][i].split(' ')[1][:2])*60+int(select_flarelist['end'][i].split(' ')[1][3:]))/(60*24)
        #     else:
        #         eflare_time = 24*60
        #     if select_flarelist['X-ray class'][i][:1][0] == 'A':
        #         color = 'gray'
        #     elif select_flarelist['X-ray class'][i][:1][0] == 'B':
        #         color = 'sienna'
        #     elif select_flarelist['X-ray class'][i][:1][0] == 'C':
        #         color = 'tomato'
        #     elif select_flarelist['X-ray class'][i][:1][0] == 'M':
        #         color = 'darkred'
        #     elif select_flarelist['X-ray class'][i][:1][0] == 'X':
        #         color = 'red'
        #     axes_.axhline(m_size, xmin=sflare_time, xmax=eflare_time, ls = "-", color = color, linewidth = 8.0)
        #     axes_.axvline(x = date1 + datetime.timedelta(hours=4), ls = "--", color = "k", linewidth = 1.0)
        #     axes_.axvline(x = date1 + datetime.timedelta(hours=8), ls = "--", color = "k", linewidth = 1.0)
        #     axes_.axvline(x = date1 + datetime.timedelta(hours=12), ls = "--", color = "k", linewidth = 1.0)
        #     axes_.axvline(x = date1 + datetime.timedelta(hours=16), ls = "--", color = "k", linewidth = 1.0)
        #     axes_.axvline(x = date1 + datetime.timedelta(hours=20), ls = "--", color = "k", linewidth = 1.0)
        #     axes_.axvline(x = date1 + datetime.timedelta(hours=24), ls = "--", color = "k", linewidth = 1.0)
        #     m_size -= 0.05
        
        # axes_.set_xlim(date2num([date1,date2]))
        # axes_.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        
        # axes_.text(date1 + datetime.timedelta(hours=1), 0.5, "A class", backgroundcolor='gray')
        # axes_.text(date1 + datetime.timedelta(hours=1), 0.4, "B class", backgroundcolor='sienna')
        # axes_.text(date1 + datetime.timedelta(hours=1), 0.3, "C class", backgroundcolor='tomato')
        # axes_.text(date1 + datetime.timedelta(hours=1), 0.2, "M class", backgroundcolor='darkred')
        # axes_.text(date1 + datetime.timedelta(hours=1), 0.1, "X class", backgroundcolor='red')
        
        # plt.title('Flare Occurrence Time  ' + date1.strftime("%Y/%m/%d"), fontsize=18)
        # plt.tick_params(labelsize=14)
        # plt.yticks(color="None")
        # plt.xticks(drange(date1, date2, datetime.timedelta(hours=2)))
        plt.savefig('/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/flare_plot/' + date1.strftime("%Y%m%d"))
        plt.close()
    return


def wind_geotail_nancay_plot_classification(date, DATE, count, check_dir):
    str_date=str(date)
    yyyy=str_date[0:4]
    mm=str_date[4:6]
    dd=str_date[6:8]
    WINDOW_NAME_0 = "wind"
    WINDOW_NAME_1 = "geotail"
    WINDOW_NAME_2 = "detected type 3 burst  "
    WINDOW_NAME_3 = "Nancay QL"
    WINDOW_NAME_4 = "Flare"
    # WINDOW_NAME_5 = 'SDO_HMIB'
    WINDOW_NAME_6 = 'Nancay Wind'


    files = sorted(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+check_dir_original+'/' + check_dir + '/'+yyyy+'/'+mm+'/'+yyyy+mm+dd+'/'+yyyy+mm+dd+'_*.png'))
    if len(files) > 0:
        cv2.startWindowThread()
        wind_geotail_flare(yyyy, mm, dd, WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4)
        burst_colar_pic = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/dust/burst_color.png'
        img_burst_colar_pic = cv2.imread(burst_colar_pic, cv2.IMREAD_COLOR)
        img_burst_colar_pic_1 = img_burst_colar_pic[35:250, :]
        image_burst_colar_pic = cv2.resize(img_burst_colar_pic_1, dsize=None, fx=1*factor, fy=1*factor)
        cv2.namedWindow('Burst classification', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Burst classification', image_burst_colar_pic)
        cv2.moveWindow('Burst classification', 2418, 512)
        cv2.waitKey(1)


        for i in range(int(len(files)/2)):
            nancay_wind_QL_list = sorted(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancaywind_3/' + yyyy + '/' + mm + '/' + str_date + '*'))
            nancay_QL_list = sorted(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/QL/' + yyyy + '/' + mm + '/' + str_date + '*'))
            start_time = pd.to_datetime(date + files[2*i+1].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')
            start_time_1 = pd.to_datetime(date + files[2*i].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+pd.to_timedelta(200,unit='sec')
            start_time_2 = pd.to_datetime(date + files[2*i+1].split('/')[-1].split('_')[1],format='%Y%m%d%H%M%S')+pd.to_timedelta(200,unit='sec')
            print (start_time)
            int_start_time = int(start_time.strftime("%H%M%S"))
            str_start_time = start_time.strftime("%H:%M")
            str_start_time_1 = start_time_1.strftime("%H:%M")
            str_start_time_2 = start_time_2.strftime("%H:%M") 


            img_nancay = cv2.imread(files[2*i], cv2.IMREAD_COLOR)
            img_nancay_1 = img_nancay[80:800, 70:800]
            image_nancay = cv2.resize(img_nancay_1, dsize=None, fx=1.3*factor, fy=1.3*factor)
            cv2.namedWindow(WINDOW_NAME_2 + str_start_time_1, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_2 + str_start_time_1, image_nancay)
            cv2.moveWindow(WINDOW_NAME_2 + str_start_time_1, 2418, 0)
            cv2.waitKey(1)

            img_nancay = cv2.imread(files[2*i+1], cv2.IMREAD_COLOR)
            img_nancay_1 = img_nancay[80:800, 70:800]
            image_nancay = cv2.resize(img_nancay_1, dsize=None, fx=1.3*factor, fy=1.3*factor)
            cv2.namedWindow(WINDOW_NAME_2 + str_start_time_2, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_2 + str_start_time_2, image_nancay)
            cv2.moveWindow(WINDOW_NAME_2 + str_start_time_2, 2890, 0)
            cv2.waitKey(1)

            figure_ = plt.figure(1, figsize=(10,3))
            axes_ = figure_.add_subplot(111)
            axes_.text(0.5, 0.5, yyyy + '/' +mm+'/'+dd+' '+ str_start_time, size = 60, color="black", fontweight="bold", horizontalalignment="center", verticalalignment="center")
            axes_.axis("off")
            time_file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/dust/aaa.png'
            plt.savefig(time_file)
            plt.close()
            cv2.waitKey(1)
            cv2.destroyWindow('Obs_time')
            cv2.waitKey(1)
            time_data = cv2.imread(time_file, cv2.IMREAD_COLOR)
            image_time = cv2.resize(time_data, dsize=None, fx=1*factor, fy=1*factor)
            cv2.namedWindow('Obs_time', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('Obs_time', image_time)
            cv2.moveWindow('Obs_time', 1931, 500)
            cv2.waitKey(1)



            NQL = []
            for nancay_QL in nancay_QL_list:
                if int(nancay_QL.split('.')[0].split('/')[-1].split('_')[1]) <= int_start_time:
                    if int(nancay_QL.split('.')[0].split('/')[-1].split('_')[2]) >= int_start_time:
                        NQL.append(nancay_QL)
            if len(NQL) == 0:
                print('Find error: No Nancay QL is found   filename_' + files[2*i])
            elif len(NQL) == 1:
                plot_nancay_QL = NQL[0]
            else:
                stime_list = []
                for QL in NQL:
                    stime_list.append(int(QL.split('.')[0].split('_')[2]))
                plot_nancay_QL = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/QL/' + yyyy + '/' + mm + '/' + str_date + '_*' + str(min(stime_list)) +'_*')[0]
            img_nancay_QL = cv2.imread(plot_nancay_QL, cv2.IMREAD_COLOR)
            img_nancay_QL_1 = img_nancay_QL[0:600, 120:1000]
            image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.7, fy=0.7)
            # img_nancay_QL_1 = img_nancay_QL[0:400, 120:1800]
            # image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.72, fy=0.72)
            cv2.namedWindow(WINDOW_NAME_3, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_3, image_nancay_QL)
            cv2.moveWindow(WINDOW_NAME_3, 2418, 640)
            cv2.waitKey(1)


            Nancay_wind_QL = []
            for nancay_wind_QL in nancay_wind_QL_list:
                if int(nancay_wind_QL.split('.')[0].split('/')[-1].split('_')[1]) <= int_start_time:
                    if int(nancay_wind_QL.split('.')[0].split('/')[-1].split('_')[2]) >= int_start_time:
                        Nancay_wind_QL.append(nancay_wind_QL)
            if len(Nancay_wind_QL) == 0:
                print('Find error: No Nancay and Wind QL is found   filename_' + files[2*i])
            else:
                if len(Nancay_wind_QL) == 1:
                    plot_nancay_wind_QL = Nancay_wind_QL[0]
                else:
                    stime_list = []
                    for NWQL in Nancay_wind_QL:
                        stime_list.append(int(NWQL.split('.')[0].split('/')[-1].split('_')[1]))
                    plot_nancay_wind_QL = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancaywind_3/' + yyyy + '/' + mm + '/' + str_date +  '_*' + str(min(stime_list)) +'_*')[0]
                img_nancay_wind_QL = cv2.imread(plot_nancay_wind_QL, cv2.IMREAD_COLOR)
                img_nancay_wind_QL_1 = img_nancay_wind_QL[100:800, 50:800]
                image_nancay_wind_QL = cv2.resize(img_nancay_wind_QL_1, dsize=None, fx=1.3*factor, fy=1.3*factor)
                # img_nancay_QL_1 = img_nancay_QL[0:400, 120:1800]
                # image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.72, fy=0.72)
                cv2.namedWindow(WINDOW_NAME_6, cv2.WINDOW_AUTOSIZE)
                cv2.imshow(WINDOW_NAME_6, image_nancay_wind_QL)
                cv2.moveWindow(WINDOW_NAME_6, 1928, 0)
                cv2.waitKey(1)


            cv2.waitKey(1000)
            count -= 1
            print ('Last :' + str(count))
            check_num = 0
            day_move = 0
            while check_num == 0:
                choice_result = choice()
                if choice_result == '1':
                    cv2.waitKey(1)
                    cv2.destroyWindow(WINDOW_NAME_3)
                    cv2.waitKey(1)
                    if nancay_QL_list.index(plot_nancay_QL) - 1 >= 0:
                        plot_nancay_QL = nancay_QL_list[nancay_QL_list.index(plot_nancay_QL) - 1]
                        img_nancay_QL = cv2.imread(plot_nancay_QL, cv2.IMREAD_COLOR)
                        img_nancay_QL_1 = img_nancay_QL[0:600, 120:1000]
                        image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.7, fy=0.7)
                        cv2.namedWindow(WINDOW_NAME_3, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(WINDOW_NAME_3, image_nancay_QL)
                        cv2.moveWindow(WINDOW_NAME_3, 2418, 640)
                        cv2.waitKey(1)
                    else:
                        print ('No data found: Nancay QL')
                    cv2.waitKey(1)
                    cv2.destroyWindow(WINDOW_NAME_6)
                    cv2.waitKey(1)
                    if nancay_wind_QL_list.index(plot_nancay_wind_QL) - 1 >= 0:
                        plot_nancay_wind_QL = nancay_wind_QL_list[nancay_wind_QL_list.index(plot_nancay_wind_QL) - 1]
                        img_nancay_wind_QL = cv2.imread(plot_nancay_wind_QL, cv2.IMREAD_COLOR)
                        img_nancay_wind_QL_1 = img_nancay_wind_QL[100:800, 50:800]
                        image_nancay_wind_QL = cv2.resize(img_nancay_wind_QL_1, dsize=None, fx=1.3*factor, fy=1.3*factor)
                        # img_nancay_QL_1 = img_nancay_QL[0:400, 120:1800]
                        # image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.72, fy=0.72)
                        cv2.namedWindow(WINDOW_NAME_6, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(WINDOW_NAME_6, image_nancay_wind_QL)
                        cv2.moveWindow(WINDOW_NAME_6, 1928, 0)
                        cv2.waitKey(1)
                    else:
                        print ('No data found: Nancay Wind QL')

                elif choice_result == '2':
                    cv2.waitKey(1)
                    cv2.destroyWindow(WINDOW_NAME_3)
                    cv2.waitKey(1)
                    if len(nancay_QL_list) > nancay_QL_list.index(plot_nancay_QL) + 1:
                        plot_nancay_QL = nancay_QL_list[nancay_QL_list.index(plot_nancay_QL) + 1]
                        img_nancay_QL = cv2.imread(plot_nancay_QL, cv2.IMREAD_COLOR)
                        img_nancay_QL_1 = img_nancay_QL[0:600, 120:1000]
                        image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.7, fy=0.7)
                        cv2.namedWindow(WINDOW_NAME_3, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(WINDOW_NAME_3, image_nancay_QL)
                        cv2.moveWindow(WINDOW_NAME_3, 2418, 640)
                        cv2.waitKey(1)
                    else:
                        print ('No data found: QL')
                    cv2.waitKey(1)
                    cv2.destroyWindow(WINDOW_NAME_6)
                    cv2.waitKey(1)
                    if len(nancay_wind_QL_list) > nancay_wind_QL_list.index(plot_nancay_wind_QL) + 1:
                        plot_nancay_wind_QL = nancay_wind_QL_list[nancay_wind_QL_list.index(plot_nancay_wind_QL) + 1]
                        img_nancay_wind_QL = cv2.imread(plot_nancay_wind_QL, cv2.IMREAD_COLOR)
                        img_nancay_wind_QL_1 = img_nancay_wind_QL[100:800, 50:800]
                        image_nancay_wind_QL = cv2.resize(img_nancay_wind_QL_1, dsize=None, fx=1.3*factor, fy=1.3*factor)
                        # img_nancay_QL_1 = img_nancay_QL[0:400, 120:1800]
                        # image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.72, fy=0.72)
                        cv2.namedWindow(WINDOW_NAME_6, cv2.WINDOW_AUTOSIZE)
                        cv2.imshow(WINDOW_NAME_6, image_nancay_wind_QL)
                        cv2.moveWindow(WINDOW_NAME_6, 1928, 0)

                        cv2.waitKey(1)
                    else:
                        print ('No data found: QL')

                elif choice_result == '3':
                    window_close_1(WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4)
                    day_move -= 1
                    DATE_choice = DATE + pd.to_timedelta(day_move,unit='day')
                    date_choice = DATE_choice.strftime(format='%Y%m%d')
                    yyyy_choice = date_choice[0:4]
                    mm_choice = date_choice[4:6]
                    dd_choice = date_choice[6:8]
                    wind_geotail_flare(yyyy_choice, mm_choice, dd_choice, WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4)
                    # cv2.waitKey(1000)
                elif choice_result == '4':
                    window_close_1(WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4)
                    day_move += 1
                    DATE_choice = DATE + pd.to_timedelta(day_move,unit='day')
                    date_choice = DATE_choice.strftime(format='%Y%m%d')
                    yyyy_choice = date_choice[0:4]
                    mm_choice = date_choice[4:6]
                    dd_choice = date_choice[6:8]
                    wind_geotail_flare(yyyy_choice, mm_choice, dd_choice, WINDOW_NAME_0, WINDOW_NAME_1, WINDOW_NAME_4)


                else:
                    if choice_result == check_dir:
                        print ('choose other choice')
                        pass
                    else:
                        if choice_result == 'ordinary':

                            check_num += 1
                        elif choice_result == 'storm':

                            check_num += 1
                        elif choice_result == 'marginal':

                            check_num += 1
                        elif choice_result == 'pass':

                            check_num += 1
                            pass
                        # print (WINDOW_NAME_8+sdo_time_0211)
                        if len(files)%2==0:
                            window_close(files[2*i+1], files[-1], WINDOW_NAME_2 + str_start_time_1, WINDOW_NAME_3, WINDOW_NAME_2 + str_start_time_2)
                        else:
                            window_close(files[2*i+1], files[-2], WINDOW_NAME_2 + str_start_time_1, WINDOW_NAME_3, WINDOW_NAME_2 + str_start_time_2)

    return count





date_range=[20120602,20211231]
check_dir_original = 'freq_check_around'
factor = 0.5
plot_time = 400
# date_in=[20170101,20200101]
# date_in[0]
# glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/af_sgepss/'+yyyy+'/'+yyyy+mm+dd+'_*.png')

if __name__=='__main__':
    date_list = []
    count = 0
    check_dir = ordinary_or_storm()
    check_plots = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+check_dir_original+'/' + check_dir + '/*/*/*/*.png')
    for check_plot in check_plots:
        if int(check_plot.split('/')[-1].split('_')[0]) >= date_range[0]:
            if int(check_plot.split('/')[-1].split('_')[0]) <= date_range[1]:
                date_list.append(int(check_plot.split('/')[-1].split('_')[0]))
                count += 1
    date_list = sorted(list(set(date_list)))
    for date_in in date_list:
        
    # date_in=[20090707,20090707]
        start_day=date_in
        DATE=pd.to_datetime(start_day,format='%Y%m%d')
    # edate=pd.to_datetime(end_day,format='%Y%m%d')
    
    # while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        year = date[:4]
        mm = date[4:6]
    

        try:
            count = wind_geotail_nancay_plot_classification(date, DATE, count, check_dir)
        except:
            print('Plot error: ',date)
            break
        DATE+=pd.to_timedelta(1,unit='day')

