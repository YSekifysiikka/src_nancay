#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 16:51:04 2021

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
import sys
import threading
import time


Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
Parent_lab = len(Parent_directory.split('/')) - 1
file_final = "/hinode_catalog/Hinode Flare Catalogue.csv"
csv_input_final = pd.read_csv(filepath_or_buffer= Parent_directory + file_final, sep=",")

def ordinary_or_storm():
    i = 0
    while i < 1:
        choice = input("Please respond with 'ordinary', 'storm', 'marginal', or 'pass' [o/s/m/p]: ").lower()
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
def window_close(file, files, WINDOW_NAME_2, WINDOW_NAME_3):
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
                sflare_time_date = date1 + datetime.timedelta(hours = int(select_flarelist['start'][i].split(' ')[1][:2]))+ datetime.timedelta(minutes = int(select_flarelist['start'][i].split(' ')[1][3:]))
            else:
                sflare_time_date = date1
            if select_flarelist['end'][i].split(' ')[0][8:] == f_dd:
                eflare_time_date = date1 + datetime.timedelta(hours = int(select_flarelist['end'][i].split(' ')[1][:2])) + datetime.timedelta(minutes = int(select_flarelist['end'][i].split(' ')[1][3:]))
            else:
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

def sdo_mp4(date):
    str_date=str(date)
    yyyy=str_date[0:4]
    mm=str_date[4:6]
    # dd=str_date[6:8]
    WINDOW_NAME_5 = 'SDO_HMI'
    if os.path.isfile('/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo/' + yyyy + '/' + mm + '/' + str_date +'_1024_HMIB.mp4') == True:
        file_path_sdo = '/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo/' + yyyy + '/' + mm + '/' + str_date +'_1024_HMIB.mp4'
        delay = 25
        
        cap = cv2.VideoCapture(file_path_sdo)
        
        if not cap.isOpened():
            sys.exit()
        print ('a')
        
        while True:
            ret, frame = cap.read()
            # time.sleep(10)
            if ret:
                print ('b')
                sdo_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                sdo_gray_1 = cv2.resize(sdo_gray, dsize=None, fx=0.22, fy=0.22)
                cv2.imshow(WINDOW_NAME_5, sdo_gray_1)
                cv2.moveWindow(WINDOW_NAME_5, 640, 600)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
            else:
                cap.release()
                # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    else:
        print ('No data: SDO HMI')


def wind_geotail_nancay_plot_classification(date, count):
    str_date=str(date)
    yyyy=str_date[0:4]
    mm=str_date[4:6]
    dd=str_date[6:8]
    WINDOW_NAME_0 = "wind"
    WINDOW_NAME_1 = "geotail"
    WINDOW_NAME_2 = "detected type 3 burst  "
    WINDOW_NAME_3 = "Nancay QL"
    WINDOW_NAME_4 = "Flare"
    WINDOW_NAME_5 = 'SDO_HMIB'
    
    files = sorted(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+check_dir +yyyy+'/'+yyyy+mm+dd+'_*.png'))
    if len(files) > 0:
        cv2.startWindowThread()

            
        if os.path.isfile("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/plot_QL/" + yyyy + "/wav_summary_" + yyyy + mm + dd + ".png") == True:
            img_wind = cv2.imread("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/wind/plot_QL/" + yyyy + "/wav_summary_" + yyyy + mm + dd + ".png", cv2.IMREAD_COLOR)
            img_wind_1 = img_wind[50:380, 20:790]
            image_wind = cv2.resize(img_wind_1, dsize=None, fx=0.83, fy=0.83)
            cv2.namedWindow(WINDOW_NAME_0, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_0, image_wind)
            cv2.moveWindow(WINDOW_NAME_0, 0, -100)
        else:
            print('No data: wind')
        if os.path.isfile('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/geotail/plot_QL/' + yyyy + '/' +yyyy[2:4] + mm + dd +'00.gif') == True:
            gif_geotail = cv2.VideoCapture('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/geotail/plot_QL/' + yyyy + '/' +yyyy[2:4] + mm + dd +'00.gif')
            is_success, img_geotail = gif_geotail.read()
            img_geotail_1 = img_geotail[0:400, 40:1105]
            # img_geotail_1 = img_geotail
            image_geotail = cv2.resize(img_geotail_1, dsize=None, fx=0.6, fy=0.6)
            cv2.namedWindow(WINDOW_NAME_1, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_1, image_geotail)
            cv2.moveWindow(WINDOW_NAME_1, 0, 240)
        else:
            print('No data: Geotail')
        flare_check(yyyy, dd, mm)
        if len(glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/flare_plot/' + yyyy+mm+dd +'.png')) == 1:
            img_flare = cv2.imread('/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/flare_plot/' + yyyy+mm+dd +'.png', cv2.IMREAD_COLOR)
            img_flare_1 = img_flare[30:280, 40:700]
            image_flare = cv2.resize(img_flare_1, dsize=None, fx=0.97, fy=0.97)
            cv2.namedWindow(WINDOW_NAME_4, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_4, image_flare)
            cv2.moveWindow(WINDOW_NAME_4, 0, 503)
        else:
            print('No data: Flare')
        if len(glob.glob("/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_HMIB/" + yyyy + "/" + mm+ "/" + yyyy + mm + dd + "*.jpg")) == 1:
            img_HMIB = cv2.imread(glob.glob("/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_HMIB/" + yyyy + "/" + mm+ "/" + yyyy + mm + dd + "*.jpg")[0], cv2.IMREAD_COLOR)
            image_HMIB = cv2.resize(img_HMIB, dsize=None, fx=0.21, fy=0.21)
            cv2.namedWindow(WINDOW_NAME_5, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_5, image_HMIB)
            cv2.moveWindow(WINDOW_NAME_5, 640, 530)
        else:
            print('No data: HMIB')



        for file in files:
            nancay_QL_list = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/QL/' + yyyy + '/' + mm + '/' + str_date + '*')
            start_time = pd.to_datetime(date + file.split('_')[3],format='%Y%m%d%H%M%S') + pd.to_timedelta(int(file.split('_')[7]),unit='sec')
            print (start_time)
            int_start_time = int(start_time.strftime("%H%M%S"))
            str_start_time = start_time.strftime("%H:%M")

            img_nancay = cv2.imread(file, cv2.IMREAD_COLOR)
            img_nancay_1 = img_nancay[40:3480, 800:9500]
            image_nancay = cv2.resize(img_nancay_1, dsize=None, fx=0.07, fy=0.07)
            cv2.namedWindow(WINDOW_NAME_2 + str_start_time, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_2 + str_start_time, image_nancay)
            cv2.moveWindow(WINDOW_NAME_2 + str_start_time, 640, -100)



            NQL = []
            for nancay_QL in nancay_QL_list:
                if int(nancay_QL.split('.')[0].split('_')[2]) <= int_start_time:
                    if int(nancay_QL.split('.')[0].split('_')[3]) >= int_start_time:
                        NQL.append(nancay_QL)
            if len(NQL) == 0:
                print('Find error: No Nancay QL is found   filename_' + file)
            elif len(NQL) == 1:
                plot_nancay_QL = NQL[0]
            else:
                stime_list = []
                for QL in NQL:
                    stime_list.append(int(QL.split('.')[0].split('_')[2]))
                plot_nancay_QL = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/QL/' + yyyy + '/' + mm + '/' + str_date + '_*' + str(min(stime_list)) +'_*')[0]
            img_nancay_QL = cv2.imread(plot_nancay_QL, cv2.IMREAD_COLOR)
            img_nancay_QL_1 = img_nancay_QL[0:600, 120:1000]
            image_nancay_QL = cv2.resize(img_nancay_QL_1, dsize=None, fx=0.65, fy=0.5)
            cv2.namedWindow(WINDOW_NAME_3, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(WINDOW_NAME_3, image_nancay_QL)
            cv2.moveWindow(WINDOW_NAME_3, 640, 210)





            

            cv2.waitKey(1000)
            count -= 1
            print ('Last :' + str(count))
            o_or_s_or_m = ordinary_or_storm()
            # time.sleep(2)
            if o_or_s_or_m == 'ordinary':
                file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+save_dir+'/ordinary/'+yyyy
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                shutil.copy(file, file_dir)
            elif o_or_s_or_m == 'storm':
                file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+save_dir+'/storm/'+yyyy
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                shutil.copy(file, file_dir)
            elif o_or_s_or_m == 'marginal':
                file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+save_dir+'/marginal/'+yyyy
                if not os.path.isdir(file_dir):
                    os.makedirs(file_dir)
                shutil.copy(file, file_dir)
            elif o_or_s_or_m == 'pass':
                # file_dir = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+save_dir+'/marginal/'+yyyy
                # if not os.path.isdir(file_dir):
                    # os.makedirs(file_dir)
                # shutil.copy(file, file_dir)
                pass
            window_close(file, files, WINDOW_NAME_2 + str_start_time, WINDOW_NAME_3)
    return count


#20130203綺麗

date_in=[20170101,20191231]
check_dir = 'clearevent_test/解析検討リスト/'
save_dir = 'afjpgu2021'
# date_in[0]
# glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/af_sgepss/'+yyyy+'/'+yyyy+mm+dd+'_*.png')

if __name__=='__main__':
    start_day,end_day=date_in
    sdate=pd.to_datetime(start_day,format='%Y%m%d')
    edate=pd.to_datetime(end_day,format='%Y%m%d')
    syear = int(sdate.strftime("%Y"))
    year_list = []
    while syear <= int(edate.strftime("%Y")):
        year_list.append(str(syear))
        syear += 1
    count = 0
    for year in year_list:
        check_plots = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/'+check_dir +year+'/*.png')
        for check_plot in check_plots:
            if int(check_plot.split('/')[11].split('_')[0]) >= date_in[0]:
                if int(check_plot.split('/')[11].split('_')[0]) <= date_in[1]:
                    count += 1
            

    DATE=sdate
    while DATE <= edate:
        date=DATE.strftime(format='%Y%m%d')
        print(date)
        # sdo_mp4(date)
        try:
            count = wind_geotail_nancay_plot_classification(date, count)
            # th2 = sdo_mp4(date)
            # th1 = threading.Thread(target=wind_geotail_nancay_plot_classification(date, count))
            # th2.start()
            # th1.start()
            
        except:
            print('Plot error: ',date)
            break
        DATE+=pd.to_timedelta(1,unit='day')

        