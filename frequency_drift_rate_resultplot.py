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
# x = [6.3, 6.4, 5.7, 6.5] # 変数を初期化
# y = [8.5, 5.5, 3.5, 1.5]
# x_err = [2.0, 2.7, 1.9, 2.7] # 誤差範囲を乱数で生成

# plt.figure(figsize=(4,7))
# plt.plot([3.0, 11],[8.5, 8.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([3.5, 12],[5.5, 5.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([1.3, 16],[3.5, 3.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([2.7, 15],[1.5, 1.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=10, ecolor='black', markeredgecolor = "black", color='w')

# # plt.plot([0.13, 0.13],[3.5, 4.5], color = "k", linewidth = 5.0, alpha = 1, marker="star") 
# # plt.scatter(0.13,4, color = "k", marker="*", s = 300) 

# # plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed") 
# # plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed") 

# # plt.plot([0, 1],[10, 10], color = "k", linewidth = 35.0, alpha = 0.1)
# # plt.plot([0, 1],[8, 8], color = "k", linewidth = 35.0, alpha = 0.1)
# # plt.plot([0, 1],[6, 6], color = "k", linewidth = 35.0, alpha = 0.1)

# plt.plot([1, 17],[10, 10], color = "orange", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[9, 9], color = "orange", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[8, 8], color = "orange", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[7, 7], color = "orange", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[6, 6], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[5, 5], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[4, 4], color = "r", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[3, 3], color = "r", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[2, 2], color = "b", linewidth = 33.0, alpha = 0.1)
# plt.plot([1, 17],[1, 1], color = "b", linewidth = 33.0, alpha = 0.1)



# plt.plot([1, 17],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([1, 17],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([1, 17],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)

# plt.plot([6.3, 6.3],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed")
# # plt.plot([1, 17],[1.5, 1.5], color = "k", linewidth = 1.0, alpha = 1)
# # plt.plot([0, 1],[0.5, 0.5], color = "k", linewidth = 1.0, alpha = 1)

# plt.xlabel('Frequency drift rates [MHz/s]', fontsize=20)
# plt.ylabel('y', fontsize=20)
# plt.xlim(1, 17)
# plt.ylim(0.5, 12)
# plt.xticks(fontsize=16)
# plt.title('STD')
# ax = plt.gca()
# # ax.axes.xaxis.set_visible(False)
# ax.axes.yaxis.set_visible(False)
# plt.show()


## テストデータの作成
#SE使用
x = [6.3, 6.4, 5.7, 6.5] # 変数を初期化
y = [8.5, 5.5, 3.5, 1.5]
x_err = [0.49, 1.7, 0.12, 0.45] # 誤差範囲を乱数で生成

plt.figure(figsize=(4,7))
# plt.plot([3.0, 11],[8.5, 8.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([3.5, 12],[5.5, 5.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([1.3, 16],[3.5, 3.5], color = "k", linewidth = 5.0, alpha = 0.5) 

# plt.plot([2.7, 15],[1.5, 1.5], color = "k", linewidth = 5.0, alpha = 0.5) 

plt.errorbar(x, y, xerr = x_err, capsize=5, fmt='o', markersize=6, ecolor='r', markeredgecolor = "r", color='r')

# plt.plot([0.13, 0.13],[3.5, 4.5], color = "k", linewidth = 5.0, alpha = 1, marker="star") 
# plt.scatter(0.13,4, color = "k", marker="*", s = 300) 

# plt.plot([0.04, 0.04],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed") 
# plt.plot([0.60, 0.60],[4.5, 11], color = "k", linewidth = 5.0, alpha = 1, linestyle = "dashed") 

# plt.plot([0, 1],[10, 10], color = "k", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[8, 8], color = "k", linewidth = 35.0, alpha = 0.1)
# plt.plot([0, 1],[6, 6], color = "k", linewidth = 35.0, alpha = 0.1)

plt.plot([1, 17],[10, 10], color = "orange", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[9, 9], color = "orange", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[8, 8], color = "orange", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[7, 7], color = "orange", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[6, 6], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[5, 5], color = "deepskyblue", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[4, 4], color = "r", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[3, 3], color = "r", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[2, 2], color = "b", linewidth = 33.0, alpha = 0.1)
plt.plot([1, 17],[1, 1], color = "b", linewidth = 33.0, alpha = 0.1)



plt.plot([1, 17],[6.5, 6.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([1, 17],[4.5, 4.5], color = "k", linewidth = 1.0, alpha = 1)
plt.plot([1, 17],[2.5, 2.5], color = "k", linewidth = 1.0, alpha = 1)

plt.plot([6.3, 6.3],[0, 10.5], color = "k", linewidth = 2.0, alpha = 1, linestyle = "dashed")
# plt.plot([1, 17],[1.5, 1.5], color = "k", linewidth = 1.0, alpha = 1)
# plt.plot([0, 1],[0.5, 0.5], color = "k", linewidth = 1.0, alpha = 1)

plt.xlabel('Frequency drift rates [MHz/s]', fontsize=20)
plt.ylabel('y', fontsize=20)
plt.xlim(4, 9)
plt.ylim(0.5, 12)
plt.xticks(fontsize=16)
plt.title('95% CI')
ax = plt.gca()
# ax.axes.xaxis.set_visible(False)
ax.axes.yaxis.set_visible(False)
plt.show()