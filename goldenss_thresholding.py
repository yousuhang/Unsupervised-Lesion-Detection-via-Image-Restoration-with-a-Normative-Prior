#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:26:12 2018

Find thresholds according to 1, 5, 10 percent of FP and output TP TN FP FN

@author: syou
"""
import numpy as np
import sys
sys.path.append("...") % the working folder
#from utils import roc_score2, dsc
import h5py
from utils import num2str, determine_threshold, gss, STAT
import os
    
batchsize = 68 
imageshape = [158,198]
batch = 97
sequential_number = 1

model = 'GMVAE' # or 'VanillaVAE
rhos = np.arange(20)/5.0

camdata = h5py.File('% the data folder/CamCANT2_data.hdf5','r') % loadling camcant2 dataset
fprs = np.array([0.01,0.05,0.10])

# find thresholds from restorations of training data
for rho in rhos:
    thresholds = []
    respath = ' % the working folder' + model + '/CamCANT2/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ str(rho)
    savepath = os.path.join(' % the working folder' + model + '/camstatsFsTVRestoration',num2str(sequential_number), str(rho))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    for j in range(1): 
        print 'calculating statistics for '+ mode+'rho = ', rho, 'niter', j
        resi = []
        mask = []
        for i in range(batch):
            print 'loading batch = ' ,i
            ori = np.expand_dims(camdata['Test'][i*batchsize:(i+1)*batchsize,22:180,17:215], axis = -1)
            res = np.expand_dims(np.load(respath +  '/restored_images' + str(i) +'.npy')[...,j], axis = -1)
            mask += [ori > 0]
            resi_temp = np.abs(ori - res)
            resi += [resi_temp]
        resi = (np.array(resi)[np.array(mask)]).reshape(-1)
        for fpr in fprs:
            threshold_ = determine_threshold(resi, fpr)
            thresholds += [threshold_]
            print str(fpr*100) + '% fpr threshold = ', threshold_ ,'for '+ mode
    np.save(savepath + '/fprthresholds.npy',np.array(thresholds))        

# Calculate DSCs for restorations of test data

LGdata = h5py.File('%the data folder/BraTSLGG_data.hdf5', 'r') % load BRATSLGG dataset
LGlabel = h5py.File('%the data folder/BraTSLGG_label.hdf5', 'r') % The ground truth
HGdata = h5py.File('%the data folder/BraTSHGG_data.hdf5', 'r') % load BRATSHGG dataset
HGlabel = h5py.File('%the data folder/BraTSHGG_label.hdf5', 'r') % The ground truth

LGGindex = np.load('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSLGGindeces.npy') # Index of test data after leaving out pure background slices.
HGGindex = np.load('/scratch_net/biwidl211_second/syou/thesis/extension/data/BraTSHGGindeces.npy')

        
for rho in rhos:
    for k in range(1):
        
        LGGpath = '% the working folder' + model + '/BraTSLGG/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ "{0:.1f}".format(rho)
        HGGpath = '% the working folder' + model + '/BraTSHGG/Dataslicehe0.06FsTVRestoration' + num2str(sequential_number) +'/'+ "{0:.1f}".format(rho)
        savepath = os.path.join('% the working folder' + model + '/abs3statshe0.06FsTVRestoration'  + num2str(sequential_number), "{0:.1f}".format(rho))
        threshpath = os.path.join('% the working folder' + model + '/camstatsFsTVRestoration', num2str(sequential_number), str(rho))
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
        threshs = np.load(threshpath + '/fprthresholds.npy')
        LGsta = []
        for i in range(len(LGGindex)):
            nslice = LGGindex[i].astype(int)
            endind = startind + nslice
            LGmask = LGdata['Mask'][startind:endind,22:180,17:215].astype(bool) 
            pmask = LGlabel['Label'][startind:endind,22:180,17:215]
            ori = LGdata['Test'][startind:endind,22:180,17:215]
            restore = LGrestore['Restore'][startind:endind]
            dif =  -restore + ori
            dif = dif[LGmask]
            dif = np.abs(dif)
            pmask = pmask[LGmask] > 0   
            LGsta += [STAT(threshs, dif, pmask.astype(int))]
            startind = endind
        np.save(savepath + '/LGstag.npy', np.array(LGsta))
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
            dif = restore - ori
            dif = dif[HGmask]
            dif =  np.abs(dif)
            pmask = pmask[HGmask] > 0 
            HGsta += [STAT(threshs, dif, pmask.astype(int))]
            startind = endind
        np.save(savepath + '/HGstag.npy', np.array(HGsta))
