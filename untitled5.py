#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 16:38:32 2021

@author: yuichiro
"""
#overall
import glob
dir_names = ['marginal', 'ordinary', 'storm']
count_num = []
for dir_name in dir_names:
    files = glob.glob("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/"+dir_name+"/*/*")
    # print (dir_name + ': ' + str(len(files)))
    count_num.append(len(files))
print ('total')
print ('marginal num:' +str(count_num[0])+ ' rate: ' + str(round(count_num[0]/sum(count_num),2)))
print ('ordinary num:'+ str(count_num[1])+ ' rate: ' + str(round(count_num[1]/sum(count_num),2)))
print ('storm num:'+ str(count_num[2])+ ' rate: ' + str(round(count_num[2]/sum(count_num),2)))


#maximum&minimum
import glob
dir_names = ['marginal', 'ordinary', 'storm']
dir_maximums = ['2014']
dir_minimums = ['2018']

print ("____________________\n\nsolar maximum")
count_num_maximums = []
for dir_name in dir_names:
    for dir_maximum in dir_maximums:
        files = glob.glob("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/"+dir_name+"/" + dir_maximum + "/*")
    # print (dir_name + ': ' + str(len(files)))
        count_num_maximums.append(len(files))
print ('marginal num: ' + str(sum(count_num_maximums[0:len(dir_maximums)]))+ ' rate: '+ str(round(sum(count_num_maximums[0:len(dir_maximums)])/sum(count_num_maximums),2)))
print ('ordinary num: ' + str(sum(count_num_maximums[len(dir_maximums):len(dir_maximums)*2]))+ ' rate: '+ str(round(sum(count_num_maximums[len(dir_maximums):len(dir_maximums)*2])/sum(count_num_maximums),2)))
print ('storm: num: ' + str(sum(count_num_maximums[len(dir_maximums)*2:len(dir_maximums)*3])) + ' rate: ' + str(round(sum(count_num_maximums[len(dir_maximums)*2:len(dir_maximums)*3])/sum(count_num_maximums),2)))
print ("____________________\n\nsolar minimum")

count_num_minimums = []
for dir_name in dir_names:
    for dir_minimum in dir_minimums:
        files = glob.glob("/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/plot/stormororiginal/"+dir_name+"/" + dir_minimum + "/*")
    # print (dir_name + ': ' + str(len(files)))
        count_num_minimums.append(len(files))
print ('marginal num: ' + str(sum(count_num_minimums[0:len(dir_maximums)]))+ ' rate: ' + str(round(sum(count_num_minimums[0:len(dir_maximums)])/sum(count_num_minimums),2)))
print ('ordinary num: ' + str(sum(count_num_minimums[len(dir_maximums)*1:len(dir_maximums)*2]))+ ' rate: '+ str(round(sum(count_num_minimums[len(dir_maximums)*1:len(dir_maximums)*2])/sum(count_num_minimums),2)))
print ('storm num: ' + str(sum(count_num_minimums[len(dir_maximums)*2:len(dir_maximums)*3]))+ ' rate: '+ str(round(sum(count_num_minimums[len(dir_maximums)*2:len(dir_maximums)*3])/sum(count_num_minimums),2)))


#solar_marginal = [366, 260, 241, 113, 0, 94, 4]
