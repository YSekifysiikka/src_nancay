#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:56:58 2021

@author: yuichiro
"""


# coding: utf-8
# 自分の得意な言語で
# Let's チャレンジ！！
# coding: utf-8
# 自分の得意な言語で
# Let's チャレンジ！！


# coding: utf-8
# 自分の得意な言語で
# Let's チャレンジ！！
input_line = input()
supecial_guest_num = int(input_line.split(' ')[0])
normal_guest_num = int(input_line.split(' ')[1])
import numpy as np

if supecial_guest_num > 0:
    supecial_guest_datelist = []
    for i in range(supecial_guest_num):
        input_line = input()
        event_list = int(input_line.split(' ')[0])
        for j in range(event_list):
            supecial_guest_datelist.append(input_line.split(' ')[j + 1])
    supecial_guest_datelist = np.array(supecial_guest_datelist)
    
    datelist_cans = list(set(supecial_guest_datelist))
    for datelist_can in datelist_cans:
        if len(np.where(supecial_guest_datelist == datelist_can)[0]) == supecial_guest_num:
            pass
        else:
            datelist_cans.remove(datelist_can)
    datelist_cans = np.array(sorted(datelist_cans))
    
    participants = np.zeros(len(datelist_cans))
    if normal_guest_num > 0:
        for i in range(normal_guest_num):
            input_line = input()
            event_list = int(input_line.split(' ')[0])
            for j in range(event_list):
                normal_can = input_line.split(' ')[j + 1]
                date_idx = np.where(datelist_cans == normal_can)[0]
                if len(date_idx) > 0:
                    participants[date_idx] += 1
    final_date_idx = np.where(participants == np.max(participants))[0][0]
    print (datelist_cans[final_date_idx] + ' ' + str(int(np.max(participants)) +supecial_guest_num))

else:
    normal_guest_datelist = []
    for i in range(normal_guest_num):
        input_line = input()
        event_list = int(input_line.split(' ')[0])
        for j in range(event_list):
            normal_guest_datelist.append(input_line.split(' ')[j + 1])
    normal_guest_datelist = np.array(sorted(normal_guest_datelist))
    
    datelist_cans = sorted(list(set(normal_guest_datelist)))
    count_days = np.zeros(len(normal_guest_datelist))
    for i in range (len(datelist_cans)):
        count = len(np.where(normal_guest_datelist == datelist_cans[i])[0])
        count_days[i] = count
    final_date_idx = np.where(count_days == np.max(count_days))[0][0]
    print (datelist_cans[final_date_idx] + ' ' + str(int(np.max(count_days))))
            

# input_line = input()
# present_num = int(input_line.split(' ')[0])
# ribbon_length = int(input_line.split(' ')[1])

# ribbon_length_mid = ribbon_length
# ribbon_num = 1
# for i in range(present_num):
#     present_length = int(input())
#     if ribbon_length_mid >= present_length:
#         ribbon_length_mid -= present_length
#     else:
#         ribbon_num += 1
#         ribbon_length_mid = ribbon_length
#         ribbon_length_mid -= present_length
# print (ribbon_num)








# input_line = input()
# gomoku_number = int(input_line.split(' ')[0])
# your_turn = int(input_line.split(' ')[1])
# import numpy as np

# gomoku_field = []
# for i in range(gomoku_number):
#     gomoku_field_each = []
#     s = input().rstrip().split(' ')
#     for j in range(len(s)):
#         gomoku_field_each.append(int(s[j]))
#     gomoku_field.append(gomoku_field_each)
# gomoku_field = np.array(gomoku_field)
# #黒0
# #白1
# #置かれていない-1
# #先手黒
# #後手白

# if your_turn == 0:
#     your_target = 0
# else:
#     your_target = 1

# win_idx = []
# #縦横確認
# for i in range(gomoku_number):
#     count = 0
#     for j in range(len(gomoku_field[i,:])):
#         if gomoku_field[i,j] == your_target:
#             count += 1
#             if count == 4:
#                 if (j != 3) & (j != gomoku_number - 1):
#                     if gomoku_field[i,j+1] == -1:
#                         win_idx.append([i,j+1])
#                     if gomoku_field[i,j-4] == -1:
#                         win_idx.append([i,j-4])
#                 elif j == 3:
#                     if gomoku_field[i,j+1] == -1:
#                         win_idx.append([i,j+1])
#                 else:
#                     if gomoku_field[i,j-4] == -1:
#                         win_idx.append([i,j-4])
#                 count = 0
#         else:
#             count = 0


# for i in range(gomoku_number):
#     count = 0
#     for j in range(len(gomoku_field[:,i])):
#         if gomoku_field[j,i] == your_target:
#             count += 1
#             if count == 4:
#                 if (j != 3) & (j != gomoku_number - 1):
#                     if gomoku_field[j+1,i] == -1:
#                         win_idx.append([j+1,i])
#                     if gomoku_field[j-4,i] == -1:
#                         win_idx.append([i,j-4])
#                 elif j == 3:
#                     if gomoku_field[j+1,i] == -1:
#                         win_idx.append([j+1,i])
#                 else:
#                     if gomoku_field[j-4,i] == -1:
#                         win_idx.append([j-4,i])
#                 count = 0
#         else:
#             count = 0

# #右肩あがり
# check_place = gomoku_number - 4
# for i in range(check_place):
#     check_length = gomoku_number - i
#     count = 0
#     for j in range(check_length):
#         if gomoku_field[i+j,j] == your_target:
#             count += 1
#             if count == 4:
#                 if (i + j != gomoku_number - 1):
#                     if gomoku_field[i+j+1,j+1] == -1:
#                         win_idx.append([i+j+1,j+1])
#                 count = 0
#         else:
#             count = 0
# for i in range(check_place):
#     check_length = gomoku_number - i
#     count = 0
#     for j in range(check_length):
#         if gomoku_field[j,j+i] == your_target:
#             count += 1
#             if count == 4:
#                 if (j != 0) & (i + j != gomoku_number - 1):
#                     if gomoku_field[j+1,i+j+1] == -1:
#                         win_idx.append([j+1,i+j+1])
#                 count = 0
#         else:
#             count = 0
# #右肩さがり
# check_place = gomoku_number - 4
# for i in range(check_place):
#     check_length = gomoku_number - i
#     count = 0
#     for j in range(check_length):
#         if gomoku_field[gomoku_number-1-i-j,j] == your_target:
#             count += 1
#             if count == 4:
#                 if (gomoku_number-1-i-j != 0):
#                     if gomoku_field[gomoku_number-1-i-j-1,j+1] == -1:
#                         win_idx.append([gomoku_number-1-i-j-1,j+1])
#                 count = 0
#         else:
#             count == 0

# for i in range(check_place):
#     check_length = gomoku_number - i
#     for j in range(check_length):
#         if gomoku_field[j,gomoku_number-1-i-j] == your_target:
#             count += 1
#             if count == 4:
#                 if (gomoku_number-1-i-j != 0):
#                     if gomoku_field[j+1,gomoku_number-1-i-j-1] == -1:
#                         win_idx.append([j+1,gomoku_number-1-i-j-1])
#                 count = 0
#         else:
#             count == 0

# if len(win_idx)>0:
#     print (str(win_idx[0][1]+1) +' '+ str(win_idx[0][0]+1))

# else:
#     if your_turn == 0:
#         your_target = 1
#     else:
#         your_target = 0
    
#     lose_idx = []
#     #縦横確認
#     for i in range(gomoku_number):
#         count = 0
#         for j in range(len(gomoku_field[i,:])):
#             if gomoku_field[i,j] == your_target:
#                 count += 1
#                 if count == 4:
#                     if (j != 3) & (j != gomoku_number - 1):
#                         if gomoku_field[i,j+1] == -1:
#                             lose_idx.append([i,j+1])
#                         if gomoku_field[i,j-4] == -1:
#                             lose_idx.append([i,j-4])
#                     elif j == 3:
#                         if gomoku_field[i,j+1] == -1:
#                             lose_idx.append([i,j+1])
#                     else:
#                         if gomoku_field[i,j-4] == -1:
#                             lose_idx.append([i,j-4])
#                     count = 0
#             else:
#                 count = 0
    
    
#     for i in range(gomoku_number):
#         count = 0
#         for j in range(len(gomoku_field[:,i])):
#             if gomoku_field[j,i] == your_target:
#                 count += 1
#                 if count == 4:
#                     if (j != 3) & (j != gomoku_number - 1):
#                         if gomoku_field[j+1,i] == -1:
#                             lose_idx.append([j+1,i])
#                         if gomoku_field[j-4,i] == -1:
#                             lose_idx.append([i,j-4])
#                     elif j == 3:
#                         if gomoku_field[j+1,i] == -1:
#                             lose_idx.append([j+1,i])
#                     else:
#                         if gomoku_field[j-4,i] == -1:
#                             lose_idx.append([j-4,i])
#                     count = 0
#             else:
#                 count = 0
    
#     #右肩あがり
#     check_place = gomoku_number - 4
#     for i in range(check_place):
#         check_length = gomoku_number - i
#         count = 0
#         for j in range(check_length):
#             if gomoku_field[i+j,j] == your_target:
#                 count += 1
#                 if count == 4:
#                     # print (i+j+1,j+1)
#                     if (i + j != gomoku_number - 1):
#                         if gomoku_field[i+j+1,j+1] == -1:
#                             lose_idx.append([i+j+1,j+1])
#                     if j != 3:
#                         if gomoku_field[i+j-4,j-4] == -1:
#                             lose_idx.append([i+j-4,j-4])
#                     count = 0
#             else:
#                 count = 0
#     for i in range(check_place):
#         check_length = gomoku_number - i
#         count = 0
#         for j in range(check_length):
#             if gomoku_field[j,j+i] == your_target:
#                 count += 1
#                 if count == 4:
#                     # print (i+j+1,j+1)
#                     if (i + j!= gomoku_number - 1):
#                         if gomoku_field[j+1,i+j+1] == -1:
#                             lose_idx.append([j+1,i+j+1])
#                     if j != 3:
#                         if gomoku_field[j-4,i+j-4] == -1:
#                             lose_idx.append([j-4,i+j-4])
#                     count = 0
#             else:
#                 count = 0
#     #右肩さがり
#     check_place = gomoku_number - 4
#     for i in range(check_place):
#         check_length = gomoku_number - i
#         count = 0
#         for j in range(check_length):
#             if gomoku_field[gomoku_number-1-i-j,j] == your_target:
#                 count += 1
#                 if count == 4:
#                     if (gomoku_number-1-i-j != 0):
#                         if gomoku_field[gomoku_number-1-i-j-1,j+1] == -1:
#                             lose_idx.append([gomoku_number-1-i-j-1,j+1])
#                     if j != 3:
#                         if gomoku_field[gomoku_number-1-i-j+4,j-4] == -1:
#                             lose_idx.append([gomoku_number-1-i-j+4,j-4])
#                     count = 0
#             else:
#                 count == 0
    
#     for i in range(check_place):
#         check_length = gomoku_number - i
#         for j in range(check_length):
#             if gomoku_field[j,gomoku_number-1-i-j] == your_target:
#                 count += 1
#                 if count == 4:
#                     if (gomoku_number-1-i-j != 0):
#                         if gomoku_field[j+1,gomoku_number-1-i-j-1] == -1:
#                             lose_idx.append([j+1,gomoku_number-1-i-j-1])
#                     if j != 3:
#                         if gomoku_field[gomoku_number-1-i-j+4,j-4] == -1:
#                             lose_idx.append([j-4,gomoku_number-1-i-j+4])
#                     count = 0
#             else:
#                 count == 0
#     lose_list_count = 0
#     if len(lose_idx) >= 2:
#         for i in range(len(lose_idx)-1):
#             for j in range(len(lose_idx)-i-1):
#                 if (lose_idx[i][0] == lose_idx[i+j+1][0]) & (lose_idx[i][0] == lose_idx[i+j+1][0]):
#                     lose_list_count += 1
#     final_count = len(lose_idx) - lose_list_count
#     if final_count==1:
#         print (str(lose_idx[0][1]+1) +' '+ str(lose_idx[0][0]+1))
#     elif final_count>1:
#         print ('LOSE')
#     else:
#         ramdom_idx = np.where(gomoku_field == -1)
#         print (str(ramdom_idx[0][1]+1) +' '+ str(ramdom_idx[0][0]+1))