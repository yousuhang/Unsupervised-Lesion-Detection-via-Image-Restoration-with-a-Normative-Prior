#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:33:33 2018

Calculate Corresponded L1 for chosen lambda

@author: syou
"""
import numpy as np
import sys
sys.path.append("/scratch_net/biwidl211_second/syou/thesis/extension/") # include this file
#from utils import roc_score2, dsc
import h5py
from utils import num2str
import os
import tensorflow as tf

batchsize = 68 
imageshape = [158,198]
origin =  tf.placeholder(tf.float32, [None, 158,198,1], name='ori')  
restored =   tf.placeholder(tf.float32, [None, 158,198,1], name='res')  

L1 = tf.reduce_sum(tf.abs(tf.subtract(origin, restored)), axis = [1,2,3])

batch = 97
sequential_number = 1
mode = 'FsTVRestoration'
sess = tf.Session()


rhos = np.arange(20)/5.0# The lambda chosen

camdata = h5py.File('/scratch_net/biwidl211_second/syou/thesis/extension/data/CamCANT2_data.hdf5','r') # Directory of the training data

L1S = []

for rho in rhos:
    respath = '/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/'+ str(rho) # Directory of the restored training images
    savepath = os.path.join('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/camstats', mode  + num2str(sequential_number), str(rho)) # Directory of L1s to save
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for j in range(1): 
        print 'calculating statistics for '+ mode+'rho = ', rho, 'niter', j
        value_to_save = []
        for i in range(batch):
            ori = np.expand_dims(camdata['Test'][i*batchsize:(i+1)*batchsize,22:180,17:215], axis = -1)
            res = np.expand_dims(np.load(respath +  '/restored_images' + str(i) +'.npy')[...,j], axis = -1)
            print 'batch = ' ,i, ' intensity > 2 ', np.sum(res > 2.0),' intensity > 1.5 ', np.sum(res > 1.5), ' intensity > 1.2 ', np.sum(res > 1.2), ' intensity > 1.0 ', np.sum(res > 1.0)
			l1 = sess.run(L1, feed_dict={origin: ori, restored: res})
			value_to_save += [l1]
        np.save(savepath + '/l1' + str(j) + '.npy', np.array(value_to_save))
        l1s = np.load(savepath + '/l1' + str(j) + '.npy')
        L1S += [np.mean(l1s)]

if not os.path.exists('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/'):
    os.makedirs('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/')        
np.save('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/l1s_' + str(j) + '.npy',np.array(L1S))
