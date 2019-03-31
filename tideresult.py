#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 17:12:35 2018

Combine .npy restoration files to hdf5

@author: syou
"""

import numpy as np
import h5py
from utils import num2str
restoredata = 'CamCANT2' # 'CamCANT2' is the training data
						 # 'BraTSHGG' and 'BraTSLGG' are test data

if restoredata  == 'BraTSHGG':
    nslice = 28200
	batchsize = 60
	batch = nslice/batchsize + 1
elif restoredata  == 'BraTSLGG':
    nslice = 10020
	batchsize = 60
	batch = nslice/batchsize + 1
elif restoredata == 'CamCANT2'
	batch = 97
rhos = np.array([1.8,2.2,2.4])# the lambda chosen
for rho in rhos:
    print 'make data for rho', rho
    savepath = '/scratch_net/biwidl104/syou/thesis/extension/GMVAE/' + restoredata  +'/Datasliceuhe0.06FsTVRestoration' + num2str(1) +'/'+ str(rho)+'/' # directory of the restored images
																																					 # saved as .npy files
    h5f_test = h5py.File( savepath + 'restored_images.hdf5', 'w')
    for i in range(batch):
        restored_images = np.load(savepath + 'restored_images' + str(i) + '.npy')[:,:,:,-1]
        if i == 0:
            h5f_test.create_dataset('Restore', data=restored_images, maxshape=(None, restored_images.shape[1], restored_images.shape[2]))
        else:
            h5f_test['Restore'].resize((h5f_test['Restore'].shape[0] + restored_images.shape[0]), axis = 0)
            h5f_test['Restore'][-restored_images.shape[0]:] = restored_images   
	if restoredata == 'BraTSHGG'	
		with h5py.File( savepath + '/restored_images.hdf5', 'a') as h5f_test:
			
			restored_images = np.load(savepath + '/restored_images470.npy')[:,:,:,-1]
			h5f_test['Restore'].resize((h5f_test['Restore'].shape[0] + restored_images.shape[0]), axis = 0)
			h5f_test['Restore'][-restored_images.shape[0]:] = restored_images    
		 
			print h5f_test['Restore'].shape
	if restoredata == 'BraTSHGG'	
		with h5py.File( savepath + '/restored_images.hdf5', 'a') as h5f_test:
			
			restored_images = np.load(savepath + '/restored_images170.npy')[1,:,:,-1]
			h5f_test['Restore'].resize((h5f_test['Restore'].shape[0] + restored_images.shape[0]), axis = 0)
			h5f_test['Restore'][-restored_images.shape[0]:] = restored_images    
		 
			print h5f_test['Restore'].shape