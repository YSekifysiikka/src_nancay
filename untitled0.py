#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 10:14:43 2021

@author: yuichiro
"""

import pandas as pd

file = '/Volumes/GoogleDrive/マイドライブ/lab/solar_burst/Nancay/data/keras/pkl_file_af_jpgu_70/af_jpgu_inputdata_name128.csv'

print (file)

count = 0
csv_input = pd.read_csv(filepath_or_buffer= file, sep=",")
# print(csv_input['Time_list'])
for i in range(len(csv_input)):
    file_name = csv_input['file_name'][i]
    test_or_train = csv_input['train_or_test'][i]
    if (file_name == 'noise') and (test_or_train == 'train'):
        count+= 1
print (count)
    