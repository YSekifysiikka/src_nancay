#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 15:51:30 2021

@author: yuichiro
"""
import matplotlib.pyplot as plt
import numpy as np
 

from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import glob
import csv

CR = '2020'
xlim = [80, 110]
ylim = [-20, 0]





Parent_directory = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab'


# check_radius = [1.15,1.5]
# check_radius = [1.3,1.65]
# check_radius = [1.3,1.5,1.95]
# check_radius = [1.5]
file_name = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/new_result/'+CR+'/bss_'+CR+'.fits')[0]
bss_image_file = get_pkg_data_filename(file_name)
bss_image_data = fits.getdata(bss_image_file, ext=0)
bss_image_data = np.flipud(bss_image_data)
plt.figure(figsize=(8,5))
plt.imshow(bss_image_data,cmap='jet',aspect='auto')
plt.colorbar()
bss_neutral_line = np.where((bss_image_data <= 0.0012) & (bss_image_data >= -0.0012))
plt.scatter(bss_neutral_line[1], bss_neutral_line[0], color = 'k', marker= '.')

plt.xlabel('Carrington Longitude')
plt.ylabel('Latitude')
y = [0, 30, 60, 90, 120, 150, 179]
plt.yticks(y, ['90', '60', '30', '0','-30', '-60', '-90',])
save_name = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/plot_new/'+CR+ '/' + file_name.split('/')[-1].split('.')[0] + '.png'
plt.savefig(save_name)
plt.show()



Radius = []
start_list = []
end_list = []
angle_ave = []
angle_std = []


# with open(Parent_directory+ '/solar_burst/magnet/analysis/'+CR+'/'+CR+'.csv', 'w') as f:
#     w = csv.DictWriter(f, fieldnames=["radius", "start", "end", "xmin", "xmax", "ymin", "ymax", "angle_ave_mean", "angle_ave_std", "angle_max_mean", "angle_max_std", "angle_ave_list", "angle_max_list"])
#     w.writeheader()


#磁力線のradialからの平均ずれ角
file_names = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/new_result/'+CR+'/amn_'+CR+'*.fits')
for file_name in file_names:
    start = int(file_name.split('.fits')[0].split('_')[-1].split('-')[0])
    end = int(file_name.split('.fits')[0].split('_')[-1].split('-')[1])
    print (start, end)
    # if (((start == 11) & (end == 12)) | ((start == 14) & (end == 16))):
    if (((start == 12) & (end == 14)) | ((start == 16) & (end == 17))):
        # if (((start+end)/20 == check_radius[0]) | ((start+end)/20 == check_radius[1]) | ((start+end)/20 == check_radius[2])):
        # if (((start+end)/20 == check_radius[0]) | ((start+end)/20 == check_radius[1])):
        # if (((start+end)/20 == check_radius[0])):
        print ((start+end)/20, start, end)
        file_name = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/new_result/'+CR+'/amn_'+CR+'_' + str(start) + '-' + str(end) + '.fits')[0]
        amn_image_file = get_pkg_data_filename(file_name)
        amn_image_data = fits.getdata(amn_image_file, ext=0)
        amn_image_data = np.flipud(amn_image_data)
        plt.figure(figsize=(8,5))
        plt.imshow(amn_image_data,cmap='jet',aspect='auto')
        c = plt.colorbar()
        plt.clim(0, 45) 
        # amn_neutral_line = np.where((amn_image_data <= 0.0012) & (amn_image_data >= -0.0012))
        # plt.scatter(amn_neutral_line[1], amn_neutral_line[0], color = 'k', marker= '.')
        
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
        y = [0, 30, 60, 90, 120, 150, 179]
        plt.yticks(y, ['90', '60', '30', '0','-30', '-60', '-90',])
        save_name = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/plot_new/'+CR+ '/' + file_name.split('/')[-1].split('.')[0] + '.png'
        plt.savefig(save_name)
        plt.show()
        
        
        #磁力線のradialからの最大ずれ角
        file_name = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/new_result/'+CR+'/amx_'+CR+'_' + str(start) + '-' + str(end) + '.fits')[0]
        amx_image_file = get_pkg_data_filename(file_name)
        amx_image_data = fits.getdata(amx_image_file, ext=0)
        amx_image_data = np.flipud(amx_image_data)
        plt.figure(figsize=(8,5))
        plt.imshow(amx_image_data,cmap='jet',aspect='auto')
        c = plt.colorbar()
        # plt.clim(0, 50) 
        # amn_neutral_line = np.where((amn_image_data <= 0.0012) & (amn_image_data >= -0.0012))
        # plt.scatter(amn_neutral_line[1], amn_neutral_line[0], color = 'k', marker= '.')
        
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
        y = [0, 30, 60, 90, 120, 150, 179]
        plt.yticks(y, ['90', '60', '30', '0','-30', '-60', '-90',])
        save_name = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/plot_new/'+CR+ '/' + file_name.split('/')[-1].split('.')[0] + '.png'
        plt.savefig(save_name)
        plt.show()
        
        #ソース面の座標[i,j]に繋がる光球面上の経度
        file_name = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/new_result/'+CR+'/xph_'+CR+'.fits')[0]
        xph_image_file = get_pkg_data_filename(file_name)
        xph_image_data = fits.getdata(xph_image_file, ext=0)
        
        #ソース面の座標[i,j]に繋がる光球面上の緯度
        file_name = glob.glob('/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/new_result/'+CR+'/yth_'+CR+'.fits')[0]
        yth_image_file = get_pkg_data_filename(file_name)
        yth_image_data = fits.getdata(yth_image_file, ext=0)
        
        # xph,ythはちょっとややこしいですが、例えばソース面でθ=10、Φ=160の点が光球
        # 面に落ちる点の座標は( xph[10+89,160], yth[10+89,160] )で与えられます。
        
        latitude_list_ave = []
        longitude_list_ave = []
        power_list_ave = []
        
        for i in range(amn_image_data.shape[0]):
            for j in range(amn_image_data.shape[1]):
                x_param = i
                y_param = j
                latitude_list_ave.append(yth_image_data[i,j])
                longitude_list_ave.append(xph_image_data[i,j])
                power_list_ave.append(amn_image_data[i][j])
        figure_=plt.figure(1,figsize=(8,5))
        SC=plt.scatter(longitude_list_ave, latitude_list_ave, c=power_list_ave, cmap='jet', s = 5)
        # plt.scatter(x, y, s=100, c=value, cmap='Blues') 
        c = plt.colorbar(SC)  
        c.set_label('angle[deg]', size=15)
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
        # plt.clim(0, 50)              
        plt.xlim(0,360)
        plt.ylim(-90,90)
        save_name = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/plot_new/'+CR+ '/' + file_name.split('/')[-1].split('.')[0] + '.png'
        plt.savefig(save_name)
        plt.show()
        
        
        latitude_list_max = []
        longitude_list_max = []
        power_list_max = []
        
        for i in range(amx_image_data.shape[0]):
            for j in range(amx_image_data.shape[1]):
                x_param = i
                y_param = j
                latitude_list_max.append(yth_image_data[i,j])
                longitude_list_max.append(xph_image_data[i,j])
                power_list_max.append(amx_image_data[i][j])
        figure_=plt.figure(1,figsize=(8,5))
        SC=plt.scatter(longitude_list_max, latitude_list_max, c=power_list_max, cmap='jet', s = 5)
        # plt.scatter(x, y, s=100, c=value, cmap='Blues') 
        c = plt.colorbar(SC)  
        c.set_label('angle[deg]', size=15)
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
        plt.clim(0, 50)              
        plt.xlim(0,360)
        plt.ylim(-90,90)
        save_name = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/plot_new/'+CR+ '/' + file_name.split('/')[-1].split('.')[0] + '.png'
        plt.savefig(save_name)
        plt.show()
        
        
        
        
    
        
        
        latitude_list_ave = []
        longitude_list_ave = []
        power_list_ave = []
        power_list_max = []
        
        for i in range(amn_image_data.shape[0]):
            for j in range(amn_image_data.shape[1]):
                x_param = i
                y_param = j
                if ((yth_image_data[i,j] >= ylim[0]) & (yth_image_data[i,j] <= ylim[1])):
                    if ((xph_image_data[i,j] >= xlim[0]) & (xph_image_data[i,j] <= xlim[1])):
                        latitude_list_ave.append(yth_image_data[i,j])
                        longitude_list_ave.append(xph_image_data[i,j])
                        power_list_ave.append(amn_image_data[i][j])
                        power_list_max.append(amx_image_data[i][j])
        figure_=plt.figure(1,figsize=(8,5))
        SC=plt.scatter(longitude_list_ave, latitude_list_ave, c=power_list_ave, cmap='jet', s = 5)
        # plt.scatter(x, y, s=100, c=value, cmap='Blues') 
        c = plt.colorbar(SC)  
        c.set_label('angle[deg]', size=15)
        plt.xlabel('Carrington Longitude')
        plt.ylabel('Latitude')
        # plt.clim(0, 50)              
        plt.xlim(0,360)
        plt.ylim(-90,90)
        # save_name = '/Volumes/GoogleDrive-110582226816677617731/マイドライブ/lab/solar_burst/magnet/plot/'+CR+ '/' + file_name.split('/')[-1].split('.')[0] + '.png'
        # plt.savefig(save_name)
        plt.show()
        # if (((start+end)/20 == 1.15) or ((start+end)/20 == 1.5)):
            
    
        Radius.append((start+end)/20)
        start_list.append(start/10)
        end_list.append(end/10)
        angle_ave.append(np.mean(power_list_ave))
        angle_std.append(np.std(power_list_ave))
        plt.title('CR: ' + CR+'  ' + str(start/10) +'-'+ str(end/10))
        plt.hist(power_list_ave, label = 'AVE: ' + str(np.mean(power_list_ave)) + '\n' + 'SD: ' + str(np.std(power_list_ave)))
        plt.legend()
        plt.show()
        # w.writerow({'radius':(start+end)/20, 'start':start/10, 'end':end/10,'xmin':xlim[0], 'xmax':xlim[1], 'ymin': ylim[0],'ymax':ylim[1],'angle_ave_mean': np.mean(power_list_ave),'angle_ave_std': np.std(power_list_ave),'angle_max_mean':np.mean(power_list_max),'angle_max_std':np.std(power_list_max), 'angle_ave_list':power_list_ave, 'angle_max_list':power_list_max})
    # print ()
        # print (np.mean(power_list_ave))
