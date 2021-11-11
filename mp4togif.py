#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:38:52 2021

@author: yuichiro
"""
import imageio
import os, sys
import glob

class TargetFormat(object):
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"
def convertFile(inputpath, targetFormat):
    """Reference: http://imageio.readthedocs.io/en/latest/examples.html#convert-a-movie"""
    outputpath = '/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_pic/'+os.path.splitext(inputpath)[0].split('/')[7]+'/'+ os.path.splitext(inputpath)[0].split('/')[8] +'/'+ os.path.splitext(inputpath)[0].split('/')[9] +targetFormat
    if  not os.path.isdir('/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_pic/'+os.path.splitext(inputpath)[0].split('/')[7]+'/'+ os.path.splitext(inputpath)[0].split('/')[8]):
        os.makedirs('/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo_pic/'+os.path.splitext(inputpath)[0].split('/')[7]+'/'+ os.path.splitext(inputpath)[0].split('/')[8])
    # print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))
    print (inputpath)
    try:
        reader = imageio.get_reader(inputpath)
        fps = reader.get_meta_data()['fps']
    
        writer = imageio.get_writer(outputpath, fps=fps)
        for i,im in enumerate(reader):
            # print (i)
            if i == 55:
                # sys.stdout.write("\rframe {0}".format(i))
                sys.stdout.flush()
                writer.append_data(im)
        # print("\r\nFinalizing...")
        writer.close()
        # print("Done.")
    except:
        print ("Error")


files = glob.glob('/Volumes/GoogleDrive/マイドライブ/lab/solar_pic/sdo/2013/*/*')
for file in files:
    convertFile(file, TargetFormat.GIF)