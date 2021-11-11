#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:50:51 2020

@author: yuichiro
"""

#Newversion
# file1 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20120101_20120117.csv'
file2 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/30dB_40dB_gain_analysis.csv'
# file3 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20120101_20141231.csv'
# file4 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/test/new_gain_analysis_20170101_20201231.csv'
# file5 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20140901_20141101.csv'
# file6 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20141129_20141231.csv'
# file7 = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/af_sgepss_analysis_data/morioka_reproduce_20170101_20171231.csv'

import numpy as np
import pandas as pd
import datetime
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

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

def plot_arange_day(ax_list):
    for ax in ax_list:
        if plot_days == 1:
            ax.set_xlim(obs_time[full_idxes][0] - datetime.timedelta(minutes=5), obs_time[full_idxes][-1]+ datetime.timedelta(minutes=5))
            ax.set_ylim(ymin*0.95, ymax*1.05)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                           labelright=False,
                           labeltop=False)
        else:
            ax.set_xlim(obs_time[full_idxes][0] - datetime.timedelta(minutes=45), obs_time[full_idxes][-1]+ datetime.timedelta(minutes=45))
            ax.set_ylim(ymin*0.95, ymax*1.05)
            ax.legend(fontsize = 12)
            ax.tick_params(labelbottom=False,
                           labelright=False,
                           labeltop=False)
    return

obs_time = []
gain_40_list = []
gain_37_5_list = []
gain_35_list = []
gain_32_5_list = []
gain_30_list = []





file_list = [file2]

for file in file_list:
    print (file)

    csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
    # print(csv_input['Time_list'])
    for i in range(len(csv_input)):
        obs_time_event = datetime.datetime(int(csv_input['obs_time'][i].split('-')[0]), int(csv_input['obs_time'][i].split('-')[1]), int(csv_input['obs_time'][i].split(' ')[0][-2:]), int(csv_input['obs_time'][i].split(' ')[1][:2]), int(csv_input['obs_time'][i].split(':')[1]), int(csv_input['obs_time'][i].split(':')[2][:2]))
        # Frequency_list = csv_input['Frequency'][i]
        Frequency = np.array([float(k) for k in csv_input['Frequency'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        gain = np.array([float(k) for k in csv_input['Right-gain'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        Trx = np.array([float(k) for k in csv_input['Right-Trx'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        hot_dB = np.array([float(k) for k in csv_input['Right-hot_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        cold_dB = np.array([float(k) for k in csv_input['Right-cold_dB'][i][1:-1].replace('\n', '').split(' ') if k != ''])
        # Frequency = np.array(Frequency)
        # gain = []
        # for j in range(len(csv_input['gain'][i][1:-1].replace('\n', '').split(' '))):
            # if not csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j] == '':
                # gain.append(float(csv_input['gain'][i][1:-1].replace('\n', '').split(' ')[j]))
        # Gain_list = csv_input['gain'][i][1:-1].replace('\n', '').split(' ')
        # gain = csv_input['Right-gain'][i]

        # Trx_list = csv_input['Right-Trx'][i]
        # hot_dB_list = csv_input['Right-hot_dB'][i]
        # cold_dB_list = csv_input['Right-cold_dB'][i]

        
        # print ('a')
        # if not len(Gain_list) == len(Frequency):
        #     print ('c')
        #     sys.exit()
        # print ('b')
        obs_time.append(obs_time_event)
        gain_40_list.append(gain[0])
        gain_37_5_list.append(gain[1])
        gain_35_list.append(gain[2])
        gain_32_5_list.append(gain[3])
        gain_30_list.append(gain[4])


print ('Done')

obs_time = np.array(obs_time)
gain_40_list = np.array(gain_40_list)
gain_37_5_list = np.array(gain_37_5_list)
gain_35_list = np.array(gain_35_list)
gain_32_5_list = np.array(gain_32_5_list)
gain_30_list = np.array(gain_30_list)


# plt.title('Total')
# plt.hist(gain_list, bins = 20)
# # plt.yscale('log')
# # plt.xscale('log')
# plt.xlabel('Gain[dB]')
# plt.ylabel('Occurence Number')
# # plt.xticks(rotation=45)
# plt.show()
# plt.close()


#日変化

# gain_list = np.array(gain_list)
check_data = 7
plot_days = 10


date_in = [20070302,20070302]
start_day,end_day=date_in
sdate=pd.to_datetime(start_day,format='%Y%m%d')
edate=pd.to_datetime(end_day,format='%Y%m%d')

    
DATE=sdate
while DATE <= edate:
    date=DATE.strftime(format='%Y%m%d')
    print(date)
    yyyy = date[:4]
    mm = date[4:6]
    dd = date[6:8]
    try:
        # epoch = cdf_file['Epoch'] 
        # epoch = cdflib.epochs.CDFepoch.breakdown_tt2000(epoch)
        # Status = cdf_file['Status']
        # obs_time_list_each = []
        # dB_40_list_each = []
    
    
        if len(obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]) > 0:
            # for obs_time_each in obs_time[(obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days))]:
            # idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)))[0]
            fig = plt.figure(figsize=(18.0, 12.0))
            gs = gridspec.GridSpec(20, 1)
            # ax1 = plt.subplot(gs[:, :])#30MHz
            fig = plt.figure(figsize=(18.0, 12.0))
            gs = gridspec.GridSpec(121, 1)
            ax1 = plt.subplot(gs[0:20, :])#40MHz
            ax2 = plt.subplot(gs[25:45, :])#37.5MHz
            ax3 = plt.subplot(gs[50:70, :])#35MHz
            ax4 = plt.subplot(gs[75:95, :])#32.5MHz
            ax5 = plt.subplot(gs[100:120, :])#30MHz
            for i in range(plot_days):
                idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd))+ datetime.timedelta(days=i))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=i + 1)))[0]
                if i == 0:
                    ax1.plot(obs_time[idxes], gain_40_list[idxes], '.--', label = '40MHz')
                    ax2.plot(obs_time[idxes], gain_37_5_list[idxes], '.--', label = '37.5MHz')
                    ax3.plot(obs_time[idxes], gain_35_list[idxes], '.--', label = '35MHz')
                    ax4.plot(obs_time[idxes], gain_32_5_list[idxes], '.--', label = '32.5MHz')
                    ax5.plot(obs_time[idxes], gain_30_list[idxes], '.--', label = '30MHz')

                else:
                    ax1.plot(obs_time[idxes], gain_40_list[idxes], '.--')
                    ax2.plot(obs_time[idxes], gain_37_5_list[idxes], '.--')
                    ax3.plot(obs_time[idxes], gain_35_list[idxes], '.--')
                    ax4.plot(obs_time[idxes], gain_32_5_list[idxes], '.--')
                    ax5.plot(obs_time[idxes], gain_30_list[idxes], '.--')


            full_idxes = np.where((obs_time>datetime.datetime(int(yyyy),int(mm),int(dd)))&(obs_time<datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)))[0]
            ymax = np.max([np.max(gain_40_list[full_idxes]), np.max(gain_37_5_list[full_idxes]), np.max(gain_35_list[full_idxes]), np.max(gain_32_5_list[full_idxes]), np.max(gain_30_list[full_idxes])])
            ymin = np.min([np.max(gain_40_list[full_idxes]), np.min(gain_37_5_list[full_idxes]), np.min(gain_35_list[full_idxes]), np.min(gain_32_5_list[full_idxes]), np.min(gain_30_list[full_idxes])])
            plot_arange_day([ax1, ax2, ax3, ax4])
            if plot_days == 1:
                Minute_fmt = mdates.DateFormatter('%H:%M')  
                ax5.xaxis.set_major_formatter(Minute_fmt)
                ax5.set_xlim(obs_time[full_idxes][0] - datetime.timedelta(minutes=5), obs_time[full_idxes][-1]+ datetime.timedelta(minutes=5))
            else:
                fmt = mdates.DateFormatter('%m/%d %H') 
                ax5.xaxis.set_major_formatter(fmt)
                ax5.set_xlim(obs_time[full_idxes][0] - datetime.timedelta(minutes=45), obs_time[full_idxes][-1]+ datetime.timedelta(minutes=45))
                
            ax5.set_ylim(ymin*0.95, ymax*1.05)
            ax5.legend(fontsize = 12)
            # for ax in [ax1, ax2, ax3, ax4, ax5]:
            #     ax.legend()
            #     ax.set_xlim(obs_time_list_each[0], obs_time_list_each[-1])
            # ax5.xlim(obs_time_list_each[0], obs_time_list_each[-1])
            plt.xlabel('Time', fontsize = 20)
            if plot_days == 1:
                ax1.set_title('Antenna analysis: ' + date, fontsize = 25)
            else:
                ax1.set_title('Antenna analysis: ' + date + ' - ' + (datetime.datetime(int(yyyy),int(mm),int(dd)) + datetime.timedelta(days=plot_days)).strftime(format='%Y%m%d'), fontsize = 25)
            ax3.set_ylabel('Gain[dB]', fontsize = 20)
            # ax.set_yscale('log')
            plt.tick_params(axis='x', which='major', labelsize=15)
            # if not os.path.isdir(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm):
            #     os.makedirs(Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm)
            # filename = Parent_directory + '/solar_burst/Nancay/plot/antenna_test_5days_nonmove/'+yyyy + '/' + mm + '/'+date+'.png'
            # plt.savefig(filename)
            plt.show()
            plt.close()
    except:
        print ('Error: '+str(date))
    # plt.plot()
    # print ('a')
    DATE+=pd.to_timedelta(15,unit='day')