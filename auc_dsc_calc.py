#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:20:25 2018

Calcultate AUC and DSC_AUC

@author: syou
"""

import numpy as np
from utils import num2str
import sys
sys.path.append('/scratch_net/biwidl211_second/syou/thesis/extension/')
import pandas as pd
import os

def auc(fpr, tpr):
    if len(fpr.shape) == 1:
        interval = fpr[::-1][1:] - fpr[::-1][:-1]
        height = (tpr[::-1][1:] + tpr[::-1][:-1])/2.0
        auc = np.sum(interval*height)
        return auc
    elif len(fpr.shape) == 2:
        interval = fpr[:,::-1][:,1:] - fpr[:,::-1][:,:-1]
        height = (tpr[:,::-1][:,1:] + tpr[:,::-1][:,:-1])/2.0
        auc = np.sum(interval*height, axis = -1)
        return auc
    else: 
        raise ValueError('In correct fpr or tpr shape')
        
modes = ['FsTVRestoration']
name = ['Lambda', 'AUC','DSC_mean','DSC_std','Threshold','FPR','FNR']
names = ['TV','SSIM']

model = 'GMVAE'
sequential_number = 26

	threshs = np.append(np.linspace(0,0.1,201),np.append(np.linspace(0.1,0.4,151)[1:],np.linspace(0.4,1,51)[1:]))	
	PAUCA = []
	TTHM = []
	TTHM_d = []
	TTHM_ad = []
	MMDSC = []
	SMDSC = []
	MMDSC_d = []
	SMDSC_d = []
	MMDSC_ad = []
	SMDSC_ad = []
	FPR = []
	FNR = []

	for rho in rhos:
		statpath = os.path.join('/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/abs3statshe0.06FsTVRestoration', num2str(sequential_number),"{0:.1f}".format(rho))
		LGsta = np.load(statpath + '/LGsta.npy')
		HGsta = np.load(statpath + '/HGsta.npy')
		sta  = np.concatenate((LGsta,HGsta), axis = 0)
		tp = sta[:,0]
		atp = np.sum(tp, axis = 0)
		tn = sta[:,1]
		atn = np.sum(tn, axis = 0)
		fp = sta[:,2]
		afp = np.sum(fp, axis = 0)
		fn = sta[:,3]
		afn = np.sum(fn, axis = 0)
		atpr = atp*1.0/(atp +afn)
		afpr = afp*1.0/(afp + atn)
		adpr = atpr - afpr
		dsc = 2.0*tp/(2.0*tp + fp + fn)
		dsc_adpr = dsc[:,np.argmax(adpr)]
		mdsc_adpr = np.mean(dsc_adpr)
		sdsc_adpr = np.std(dsc_adpr)		
		thdad = threshs[np.argmax(adpr)]
		manual_auc_all = auc(afpr, atpr)
		fnr_ad = 1-atpr[np.argmax(adpr)]
		fpr_ad = afpr[np.argmax(adpr)]
		
		PAUCA += [manual_auc_all] 
		TTHM_ad += [thdad]
		MMDSC_ad += [mdsc_adpr]
		SMDSC_ad += [sdsc_adpr]
		FPR += [fpr_ad]
		FNR += [fnr_ad]
	
	data = np.concatenate((np.array(rhos).reshape(1,-1),np.array(PAUCA).reshape(1,-1),
						   np.array(MMDSC_ad).reshape(1,-1),np.array(SMDSC_ad).reshape(1,-1),
						   np.array(TTHM_ad).reshape(1,-1),np.array(FPR).reshape(1,-1),
						   np.array(FNR).reshape(1,-1)
							))
	df = pd.DataFrame(data)
	df['@step'+str(499)] = pd.Series(name, index=df.index)
	savepath = '/scratch_net/biwidl211_second/syou/thesis/extension/' + model + '/abs3stats/'
	writer = pd.ExcelWriter(savepath + 'he0.06FsTVRestoration'+ num2str(sequential_number)+'aucs.xlsx')
	df.to_excel(writer)
	writer.save()    