#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 13:27:41 2021

@author: yuichiro
"""

import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
eventnum = -1
files = ["/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/goes-xrs-report_1997.txt", "/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/goes-xrs-report_1996.txt", "/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/goes-xrs-report_1995.txt"]
with open(Parent_directory+ '/hinode_catalog/goes_xrs_1995_1997.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["Event number", "start", "peak", "end", "X-ray class"])

    w.writeheader()
    for file in files:
        f = open(file, 'r')
        datalists = f.readlines()
        for datalist in datalists:
            datalist_event = [a for a in datalist.split(' ') if a != '']
            datalist_day = '19'+datalist_event[0][5:7]+'/'+ datalist_event[0][7:9] +'/'+datalist_event[0][9:11]
            if ((len(datalist_event[1][0:2]) == 2) & (len(datalist_event[2][0:2]) == 2) & (len(datalist_event[3][0:2]) == 2) & (len(datalist_event[1][2:4]) == 2) & (len(datalist_event[2][2:4]) == 2) & (len(datalist_event[3][2:4]) == 2)):
                start = datalist_day+' '+datalist_event[1][0:2] + ':' + datalist_event[1][2:4]
                peak = datalist_day+' '+datalist_event[3][0:2] + ':' + datalist_event[3][2:4]
                end = datalist_day+' '+datalist_event[2][0:2] + ':' + datalist_event[2][2:4]
                if len(datalist_event[4]) == 1:
                    X_ray_class = datalist_event[4] + datalist_event[5].zfill(2)[:1] + '.' + datalist_event[5].zfill(2)[1:2]
                    eventnum += 1
                else:
                    if len(datalist_event[5]) == 1:
                        X_ray_class = datalist_event[5] + datalist_event[6].zfill(2)[:1] + '.' + datalist_event[6].zfill(2)[1:2]
                        eventnum += 1
                    else:
                        pass
                w.writerow({'Event number':eventnum, 'start':start, 'peak':peak, 'end':end,'X-ray class':X_ray_class})

import csv
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
eventnum = -1
files = ["/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/goes-xrs-report_2001.txt"]
with open(Parent_directory+ '/hinode_catalog/goes_xrs_2001.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["Event number", "start", "peak", "end", "X-ray class"])

    w.writeheader()
    for file in files:
        f = open(file, 'r')
        datalists = f.readlines()
        for datalist in datalists:
            datalist_event = [a for a in datalist.split(' ') if a != '']
            datalist_day = '20'+datalist_event[0][5:7]+'/'+ datalist_event[0][7:9] +'/'+datalist_event[0][9:11]
            if ((len(datalist_event[1][0:2]) == 2) & (len(datalist_event[2][0:2]) == 2) & (len(datalist_event[3][0:2]) == 2) & (len(datalist_event[1][2:4]) == 2) & (len(datalist_event[2][2:4]) == 2) & (len(datalist_event[3][2:4]) == 2)):
                start = datalist_day+' '+datalist_event[1][0:2] + ':' + datalist_event[1][2:4]
                peak = datalist_day+' '+datalist_event[3][0:2] + ':' + datalist_event[3][2:4]
                end = datalist_day+' '+datalist_event[2][0:2] + ':' + datalist_event[2][2:4]
                if len(datalist_event[4]) == 1:
                    X_ray_class = datalist_event[4] + datalist_event[5].zfill(2)[:1] + '.' + datalist_event[5].zfill(2)[1:2]
                    eventnum += 1
                else:
                    if len(datalist_event[5]) == 1:
                        X_ray_class = datalist_event[5] + datalist_event[6].zfill(2)[:1] + '.' + datalist_event[6].zfill(2)[1:2]
                        eventnum += 1
                    else:
                        pass
                w.writerow({'Event number':eventnum, 'start':start, 'peak':peak, 'end':end,'X-ray class':X_ray_class})

import csv
import pandas as pd
Parent_directory = '/Volumes/GoogleDrive/マイドライブ/lab'
eventnum = -1
files = ['/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/Hinode Flare Catalogue.csv', Parent_directory+ '/hinode_catalog/goes_xrs_2001.csv', Parent_directory+ '/hinode_catalog/goes_xrs_1995_1997.csv']
with open(Parent_directory+ '/hinode_catalog/Hinode Flare Catalogue new.csv', 'w') as f:
    w = csv.DictWriter(f, fieldnames=["Event number", "start", "peak", "end", "AR location", "X-ray class"])
    w.writeheader()
    for file in files:
        csv_input_final = pd.read_csv(filepath_or_buffer= file, sep=",")
        for i in range(len(csv_input_final)):
            start = csv_input_final["start"][i]
            peak = csv_input_final["peak"][i]
            end = csv_input_final["end"][i]
            X_ray_class = csv_input_final["X-ray class"][i]
            eventnum += 1
            if file == '/Volumes/GoogleDrive/マイドライブ/lab/hinode_catalog/Hinode Flare Catalogue.csv':
                AR = csv_input_final["AR location"][i]
                w.writerow({'Event number':eventnum, 'start':start, 'peak':peak, 'end':end,'AR location':AR,'X-ray class':X_ray_class})
            else:
                w.writerow({'Event number':eventnum, 'start':start, 'peak':peak, 'end':end,'X-ray class':X_ray_class})
            





            