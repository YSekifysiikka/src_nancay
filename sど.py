#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 16:04:00 2021

@author: yuichiro
"""
import imageio
import os, sys
import numpy as np
import glob

class TargetFormat(object):
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"

def convertFile(inputpath, targetFormat):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
    file_dir =  "/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_pic/" + os.path.splitext(inputpath)[0].split('/')[7] + "/" + os.path.splitext(inputpath)[0].split('/')[8]
    outputpath = file_dir + "/" + os.path.splitext(inputpath)[0].split('/')[9] + targetFormat
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    # print(outputpath)

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    num_list = []
    for i,im in enumerate(reader):
        num_list.append(i)
    for i,im in enumerate(reader):
        if i == int(round(np.mean(num_list),0)):
            # sys.stdout.write("\rframe {0}".format(i))
            sys.stdout.flush()
            writer.append_data(im)
    # print("\r\nFinalizing...")
    writer.close()
    # print("Done.")

date_in = [20120101, 20131231]
    
files = glob.glob("/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo/*/*/*_1024_HMIB.mp4")
for file in files:
    if int(file.split('/')[9].split('_')[0]) >= date_in[0] & int(file.split('/')[9].split('_')[0]) <= date_in[1]:
        if not os.path.isfile("/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_pic/" + file.split('/')[7] + "/" + file.split('/')[8] + "/" + os.path.splitext(file)[0].split('/')[9] + ".gif"):
            try:
                convertFile(file, TargetFormat.GIF)
            except:
                print ('Plot error: ' + file.split('/')[9].split('_')[0])


