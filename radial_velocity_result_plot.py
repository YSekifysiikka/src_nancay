#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 12:35:28 2021

@author: yuichiro
"""
import matplotlib.pyplot as plt
import numpy as np
from pynverse import inversefunc


from matplotlib import pyplot as plt
import random
import numpy as np


# ## テストデータの作成
# #SD使用
# x = [0.19, 0.33, 0.28, 0.54, 0.20, 0.33, 0.15, 0.18, 0.17, 0.22] # 変数を初期化
# y = np.flipud(np.arange(1,11,1))
# x_err = [0.06, 0.10, 0.09, 0.18, 0.08, 0.14, 0.05, 0.058, 0.071, 0.088] # 誤差範囲を乱数で生成

# plt.figure(figsize=(6,7))
# plt.plot([0.09, 0.33],[10, 10], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.16, 0.55],[9, 9], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.15, 0.49],[8, 8], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.26, 0.94],[7, 7], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([0.11, 0.36],[6, 6], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.18, 0.60],[5, 5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([0.03, 0.41],[4, 4], color = "k", linewidth = 5.0, alpha = 0.5)
# plt.plot([0.04, 0.48],[3, 3], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([0.07, 0.40],[2, 2], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.09, 0.50],[1, 1], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')

# # plt.plot([0.13, 0.13],[3.5, 4.5], color = "k", linewidth = 5.0, alpha = 1, marker="star") 
# plt.scatter(0.13,4, color = "k", marker="*", s = 300) 

# plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed") 
# plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed") 

# plt.plot([0, 1],[10, 10], color = "k", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[8, 8], color = "k", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[6, 6], color = "k", linewidth = 35.0, alpha = 0.1)

# plt.plot([0, 1],[9, 9], color = "orange", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[7, 7], color = "orange", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[5, 5], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[4, 4], color = "r", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[3, 3], color = "r", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[2, 2], color = "b", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[1, 1], color = "b", linewidth = 35.0, alpha = 0.1)


# plt.plot([0, 1],[9.5, 9.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[8.5, 8.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[7.5, 7.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[5.5, 5.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[3.5, 3.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[1.5, 1.5], color = "k", linewidth = 1.0, alpha = 1)
# # plt.plot([0, 1],[0.5, 0.5], color = "k", linewidth = 1.0, alpha = 1)

# plt.xlabel('Radial velocity [c]', fontsize=20)
# plt.ylabel('y', fontsize=20)
# plt.title('STD')
# plt.xlim(0.0, 1)
# plt.ylim(0.5, 12)
# plt.xticks(fontsize=16)
# ax = plt.gca()
# # ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# plt.show()



## テストデータの作成
#SE使用
x = [0.19, 0.20, 0.15, 0.17] # 変数を初期化
y = [10,6,4,2]
x_err = [0.015, 0.05, 0.0034, 0.012]



x_1 = [0.33, 0.28, 0.54, 0.33, 0.18, 0.22] # 変数を初期化
y_1 = [9,8,7,5,3,1]
x_err_1 = [0.025, 0.022, 0.042, 0.084, 0.0039, 0.015]

plt.figure(figsize=(6,7))
# plt.plot([0.09, 0.33],[10, 10], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.16, 0.55],[9, 9], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.15, 0.49],[8, 8], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.26, 0.94],[7, 7], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([0.11, 0.36],[6, 6], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.18, 0.60],[5, 5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([0.03, 0.41],[4, 4], color = "k", linewidth = 5.0, alpha = 0.5)
# plt.plot([0.04, 0.48],[3, 3], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([0.07, 0.40],[2, 2], color = "k", linewidth = 5.0, alpha = 0.5) 
# plt.plot([0.09, 0.50],[1, 1], color = "k", linewidth = 5.0, alpha = 0.5) 

plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=5, ecolor='r', markeredgecolor = "r", color='r')
plt.errorbar(x_1, y_1, xerr = x_err_1, capsize=5, fmt='o', markersize=5, ecolor='black', markeredgecolor = "black", color='w')

# plt.plot([0.13, 0.13],[3.5, 4.5], color = "k", linewidth = 5.0, alpha = 1, marker="star") 
plt.scatter(0.13,4, color = "k", marker="*", s = 300) 

plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")
plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed")

plt.plot([0, 1],[10, 10], color = "orange", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[8, 8], color = "orange", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[6, 6], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)

plt.plot([0, 1],[9, 9], color = "orange", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[7, 7], color = "orange", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[5, 5], color = "deepskyblue", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[4, 4], color = "r", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[3, 3], color = "r", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[2, 2], color = "b", linewidth = 35.0, alpha = 0.1)
plt.plot([0, 1],[1, 1], color = "b", linewidth = 35.0, alpha = 0.1)


plt.plot([0, 1],[9.5, 9.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[8.5, 8.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[7.5, 7.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[5.5, 5.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[3.5, 3.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([0, 1],[1.5, 1.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[0.5, 0.5], color = "k", linewidth = 1.0, alpha = 1)

plt.xlabel('Radial velocity [c]', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.title('95% CI')
plt.xlim(0.0, 0.62)
plt.ylim(0.5, 12)
plt.xticks(fontsize=16)
ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()