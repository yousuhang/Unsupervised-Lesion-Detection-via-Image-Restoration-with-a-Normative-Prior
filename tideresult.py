#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:12:35 2018

Combine .npy restoration files from test dataset to hdf5

@author: syou
"""

import os
import numpy as np
import h5py
from utils import num2str
datas = ['LGG','HGG']

sequential_number = 26
model = 'GMVAE' # or 'VanillaVAE
def dtop(data): 
    if data == 'HGG':
        nslice = 28200
        batchsize = 60
        head = 'BraTS'
    elif data == 'LGG':
        nslice = 10020
        batchsize = 60
        head = 'BraTS'
    return nslice, batchsize, head
rhos = np.arange(20)/5.0
for data in datas:
    nslice, batchsize, head = dtop(data)
    for rho in rhos:
        print 'make data for rho', rho
        for k in range(1):
#            print 'make data for step', k*50+49
            datapath = '/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/' + head + data +'/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ str(rho)+'/'
            
            savepath = '/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/' + head + data +'/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ "{0:.1f}".format(rho)+'/'
            if not os.path.exists(savepath):
                os.makedirs(savepath)            
            h5f_test = h5py.File( savepath + 'restored_images.hdf5', 'w')
            for i in range(nslice/batchsize + 1  ):
                
                restored_images = np.load(datapath + 'restored_images' + str(i) + '.npy')[:,:,:,k]
                print 'batch max intensity for i = ', i, ' is ', np.max(restored_images)
                if i == 0:
                    h5f_test.create_dataset('Restore', data=restored_images, maxshape=(None, restored_images.shape[1], restored_images.shape[2]))
                elif i < nslice/batchsize:
              
                    h5f_test['Restore'].resize((h5f_test['Restore'].shape[0] + restored_images.shape[0]), axis = 0)
                    h5f_test['Restore'][-restored_images.shape[0]:] = restored_images 
                else :
                    if data == 'HGG':
                        restored_images = np.load(datapath + '/restored_images470.npy')[:,:,:,k]
                    elif data == 'LGG':
                        restored_images = np.load(datapath + '/restored_images167.npy')[1:,:,:,k]
                    h5f_test['Restore'].resize((h5f_test['Restore'].shape[0] + restored_images.shape[0]), axis = 0)
                    h5f_test['Restore'][-restored_images.shape[0]:] = restored_images    

                if i%10 == 0 or i == nslice/batchsize:
                    print h5f_test['Restore'].shape
