#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:52:27 2021

@author: yuichiro
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pynverse import inversefunc

def numerical_diff_allen(factor, r):

    ne = factor * (2.99*(((r))**(-16))+1.55*(((r))**(-6))+0.036*(((r))**(-1.5)))*1e8
    return ne

def numerical_diff_allen_velocity(factor, r):
    h = 1e-5
    ne_1 = factor * (2.99*(((r+h))**(-16))+1.55*(((r+h))**(-6))+0.036*(((r+h))**(-1.5)))*1e8
    ne_2 = factor * (2.99*(((r-h))**(-16))+1.55*(((r-h))**(-6))+0.036*(((r-h))**(-1.5)))*1e8
    return ((ne_1 - ne_2)/(2*h))


xs = np.flipud(np.arange(0,901,1)/300)
ys = np.arange(0,901,1)/300

plot_list = np.ones((len(xs),len(ys)))

factor = 5


for y in ys:
    for x in xs:
        if x**2 + y**2 >= 1:
            r = np.sqrt(x**2 + y**2)
            plot_list[np.where(xs == x)[0][0]][np.where(ys == y)[0][0]] = numerical_diff_allen(factor, r)
            if ((numerical_diff_allen(factor, r) >= 19451086) & (numerical_diff_allen(factor, r) <= 20073087)):
                # print (r)
                plot_list[np.where(xs == x)[0][0]][np.where(ys == y)[0][0]] = np.nan
            if ((numerical_diff_allen(factor, r) >= 4908271.604938271) & (numerical_diff_allen(factor, r) <= 4968271.604938271)):
                print (r)
                plot_list[np.where(xs == x)[0][0]][np.where(ys == y)[0][0]] = np.nan
        else:
            plot_list[np.where(xs == x)[0][0]][np.where(ys == y)[0][0]] = np.nan

# print (np.nanmin(plot_list), np.nanmax(plot_list))

fig, ax = plt.subplots()
norm = mcolors.SymLogNorm(linthresh=1.0, linscale=1.0, vmin=np.nanmin(plot_list), vmax=np.nanmax(plot_list))
# plt.imshow(plot_list, vmax = np.nanmax(plot_list), vmin = np.nanmin(plot_list), extent = [0, 2.5,  0, 2.5])
ax1 = ax.imshow(plot_list, norm=norm, extent = [0, 3,  0, 3])
# ax.colorbar(aspect=40, pad=0.08, orientation='vertical')
ax.set_title('Factor' + str(factor))
plt.xlabel('Solar radius [Rs]', fontsize = 10)
cbar = plt.colorbar(ax1)
# cbar.ax.tick_params(labelsize=ticksize)
cbar.set_label('Density[/cc]', size=10)




plt.show()


# 3104436.735215995 2288000000.0
# 620887.347043199 457599999.99999994