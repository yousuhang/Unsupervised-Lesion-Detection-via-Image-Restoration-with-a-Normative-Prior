#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:33:33 2018

Output the TP TN FP FN for a range of threshold covering all intensity difference

@author: syou
"""
import numpy as np
from utils import roc_score2, dsc, STAT
import h5py
from utils import num2str
import os

model = 'GMVAE' # or 'VanillaVAE'

sequential_number = 1
rhos = np.arange(20)/5.0

LGdata = h5py.File('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSLGG_datan.hdf5', 'r')
LGlabel = h5py.File('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSLGG_label.hdf5', 'r')
HGdata = h5py.File('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSHGG_datan.hdf5', 'r')
HGlabel = h5py.File('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSHGG_label.hdf5', 'r')
LGGindex = np.load('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSLGGindeces.npy')
HGGindex = np.load('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSHGGindeces.npy')

for rho in rhos:
    for k in range(1):
        LGGpath = '/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/BraTSLGG/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ "{0:.1f}".format(rho)
        HGGpath = '/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/BraTSHGG/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ "{0:.1f}".format(rho)
        savepath = os.path.join('/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/abs3statshe0.06FsTVRestoration'  + num2str(sequential_number), "{0:.1f}".format(rho))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        print 'calculating statistics for rho = ', rho
#        print 'calculating statistics for step = ', k*50+49
        LGrestore = h5py.File(LGGpath + '/restored_images.hdf5', 'r')      
        HGrestore = h5py.File(HGGpath + '/restored_images.hdf5', 'r')
        print 'starting for LGG'
        startind = 0
        LGGauc = np.zeros(len(LGGindex))
        LGGroc = []
        LGGdsc = []
        threshs = np.append(np.linspace(0,0.1,201),np.append(np.linspace(0.1,0.4,151)[1:],np.linspace(0.4,1,51)[1:])) #np.linspace(0, 1, 401)
        LGsta = []
        for i in range(len(LGGindex)):
            nslice = LGGindex[i].astype(int)
            endind = startind + nslice
            LGmask = LGdata['Mask'][startind:endind,22:180,17:215].astype(bool) 
            pmask = LGlabel['Label'][startind:endind,22:180,17:215]
            ori = LGdata['Test'][startind:endind,22:180,17:215]
            restore = LGrestore['Restore'][startind:endind]
            dif = -restore + ori        
            dif = dif[LGmask]
    #        print 'batch 99.9% intensity of dif for i = ', i, ' is ', np.percentile(dif,99.9)
            dif = np.abs(dif)
            print 'batch 99.9% intensity of abs dif for i = ', i, ' is ', np.percentile(dif,99.9)
            pmask = pmask[LGmask] > 0   
            LGsta += [STAT(threshs, dif, pmask.astype(int))]
            startind = endind
         np.save(savepath + '/LGsta.npy', np.array(LGsta))
        print 'starting for HGG'
        startind = 0
        HGGauc = np.zeros(len(HGGindex))
        HGGroc = []
        HGGdsc = []   
        HGsta = []
        for i in range(len(HGGindex)):
            nslice = HGGindex[i].astype(int)
            endind = startind + nslice
            HGmask = HGdata['Mask'][startind:endind,22:180,17:215].astype(bool) 
            pmask = HGlabel['Label'][startind:endind,22:180,17:215]
            ori = HGdata['Test'][startind:endind,22:180,17:215]
            restore = HGrestore['Restore'][startind:endind]
            dif = -restore + ori
            dif = dif[HGmask]
    #        print 'batch 99.9% intensity of dif for i = ', i, ' is ', np.percentile(dif,99.9)
            dif = np.abs(dif)
            print 'batch 99.9% intensity of abs dif for i = ', i, ' is ', np.percentile(dif,99)
            pmask = pmask[HGmask] > 0 
            HGsta += [STAT(threshs, dif, pmask.astype(int))]
            startind = endind
        np.save(savepath + '/HGsta.npy', np.array(HGsta))
