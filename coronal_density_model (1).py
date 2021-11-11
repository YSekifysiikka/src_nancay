# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:23:01 2019

@author: fujimoto
"""

import numpy as np
import matplotlib.pyplot as plt


Rs=6.96*10**10 #cm
dist=Rs*np.arange(1,220,0.01)
AU=150e11 #[cm]
factor = 3
#dens=4.2*10**4*10**(-4.32*dist/Rs)
dens1=4.2*10**4*10**(4.32*(Rs/dist))#newkirk
dens2=0.01*10**8*(2.99*(dist/Rs)**-16+1.55*(dist/Rs)**-6+0.036*(dist/Rs)**-1.5) #baumbach-allen



freq=9*np.sqrt(dens2)/10**3

plt.close('all')
fig=plt.figure(1,figsize=(14,10))

ax1=fig.add_subplot(1,1,1)
#ln1=ax1.plot(dist/Rs,dens1,linewidth=4)
ln2=ax1.plot(dist/Rs,dens2,linewidth=4)
plt.loglog()
#plt.xscale('log')
plt.tick_params(labelsize=25)
plt.tick_params(width=2,length=10)
plt.tick_params(which='minor',width=1,length=5)
plt.title(r'%i$\times$Baumbach-Allen density model'%factor,fontsize=40)
plt.xlabel('Distance [Rs]',fontsize=30)
plt.ylabel('electron density[/cc]',fontsize=30)
plt.grid(which='both')


ax2=ax1.twinx()
ln3=ax2.plot(dist/Rs,freq)
plt.yscale('log')
plt.tick_params(labelsize=20)
plt.tick_params(which='minor',labelsize=20)
plt.tick_params(width=2,length=10)
plt.tick_params(which='minor',width=1,length=5)
plt.ylabel('plasma frequency [MHz]',fontsize=30)

plt.subplots_adjust(bottom=0.13,right=0.87,top=0.9)
plt.show()




# plt.close('all')
# fig=plt.figure(1,figsize=(14,10))

# ax1=fig.add_subplot(1,1,1)
# #ln1=ax1.plot(dist/Rs,dens1,linewidth=4)
# ln2=ax1.plot(h/Rs,fp1_newkirk, label = '1×Newkirk')
# # plt.loglog()
# # plt.yscale('log')
# plt.yscale('log')

# #plt.xscale('log')
# plt.tick_params(labelsize=25)
# plt.tick_params(width=2,length=10)
# plt.tick_params(which='minor',width=1,length=5)
# plt.title('Electron density model',fontsize=50)
# plt.xlabel(' Radial distance (R$_{Sun}$)',fontsize=40)
# plt.ylabel('plasma frequency [MHz]',fontsize=40)
# plt.grid(which='both')
# plt.ylim(10,300)
# plt.xlim(1,3)
# plt.tick_params(labelsize=32)


# ax2=ax1.twinx()
# ln3=ax2.plot(h/Rs,newkirk,linewidth=4)
# plt.yscale('log')
# # plt.loglog()
# plt.tick_params(labelsize=20)
# plt.tick_params(which='minor',labelsize=20)
# plt.tick_params(width=2,length=10)
# plt.tick_params(which='minor',width=1,length=5)
# plt.ylabel('electron density[/cc]',fontsize=40)
# plt.ylim(1234567.9012345679,1111111111.1111112)
# plt.xlim(1,3)
# plt.tick_params(labelsize=32)
# # plt.ylim(30,80)


# plt.subplots_adjust(bottom=0.13,right=0.87,top=0.9)
# ax1.legend(fontsize = 30)
# plt.show()
