#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:31:08 2018

@author: syou

"""

import numpy as np
import random
import h5py
from sklearn import metrics



def num2str(num):
    return str(num/10000)+ str(num%10000/1000) + str(num%1000/100) + str(num%100/10) + str(num%10)
 
def path_make(model,database,app):
    import os
    script_path = os.getcwd()
    return os.path.join(script_path,model,database,app,'')

def random_data(data, batchsize = 5):
    slice_number = random.sample(range(np.shape(data)[0]), batchsize)      
    s = np.sort(slice_number)
    return data[s,:,:] 


# reconstruction
def recon(recon_mean, recon_logstd, sess, x, test_image, savepath, step):    
    reconstr_mean, reconstr_logstd = sess.run([recon_mean, recon_logstd], feed_dict={x: test_image})
    reconstr_std = np.exp(reconstr_logstd)    
    np.save(savepath+'reconstr_mean'+ num2str(step) +'.npy',reconstr_mean)
    np.save(savepath+'reconstr_std'+ num2str(step) +'.npy',reconstr_std)

def data_load(process = 'Train', dataset = 'HCP', datapath = 'default'):    
    if datapath == 'default':
        raise ValueError('Please input the correct data path')
    elif dataset == 'CamCANT2':
        h5f = h5py.File( datapath + 'CamCANT2_data.hdf5', 'r')
        slice = h5f[process]
        return slice
    elif dataset == 'BraTSLGG':
        h5f = h5py.File( datapath + 'BraTSLGG_data.hdf5', 'r')
        slice = h5f[process]
        label = h5f['Label']
        return slice, label
    elif dataset == 'BraTSHGG':
        h5f = h5py.File( datapath + 'BraTSHGG_data.hdf5', 'r')
        slice = h5f[process]
        label = h5f['Label']
        return slice, label    
    else: raise ValueError('Error in dataset, should be CamCANT2, BraTSLGG and BraTSHGG')

def roc_score2(pmask, ratio):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 101)
    spes = []
    fprs = []

    fpr, tpr, thresholds = metrics.roc_curve(pmask, ratio)
    if not np.isnan(tpr).any():
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        spes.append(tpr)
        fprs.append(fpr)
        tprs[-1][0] = 0.0
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)    
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    return [spes, fprs], [mean_fpr, mean_tpr, std_tpr], [tprs_upper, tprs_lower], [aucs, mean_auc, std_auc]

def dsc(pred, target):
     if np.sum(pred) + np.sum(target) == 0:
         return 1.
     return (2 * np.sum(pred [target == 1])) / (np.sum(pred) + np.sum(target))
   
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
        
def STAT(thresholds, dif_image, label):
    TP = []
    TN = []
    FP = []
    FN = []
    for threshold in thresholds:
        TP += [np.sum((dif_image > threshold)*label)]
        TN += [np.sum((dif_image <= threshold)*(1-label))]
        FP += [np.sum((dif_image > threshold)*(1-label))]
        FN += [np.sum((dif_image <= threshold)*label)]
        TFPN = np.stack((np.array(TP),np.array(TN),np.array(FP),np.array(FN)), axis = 0)
    return TFPN

def determine_threshold(phi, fprate):
    phi = np.asarray(phi)
    # determines the lowest threshold on Phi that provides at max FP rate on the Phi values.
    # all the samples need to be controls for this function
    nums = len(phi)
    #numf = phi.shape[1]
    def func(threshold):
        phi_ = phi > threshold
        fprate_ = np.sum(phi_) / np.float(nums)
        return np.sqrt((fprate - fprate_) ** 2)
    return gss(func, phi.min(), phi.mean(), phi.max(), tau=1e-8)

def gss(f, a, b, c, tau=1e-3):
    '''
    Python recursive version of Golden Section Search algorithm

    tau is the tolerance for the minimal value of function f
    b is any number between the interval a and c
    '''
    goldenRatio = (1 + 5 ** 0.5) / 2
    if (c - b > b - a):
        x = b + (2 - goldenRatio) * (c - b)
    else:
        x = b - (2 - goldenRatio) * (b - a)
    if (abs(c - a) < tau * (abs(b) + abs(x))): return (c + a) / 2
    if (f(x) < f(b)):
        if (c - b > b - a):
            return gss(f, b, x, c, tau)
        return gss(f, a, x, b, tau)
    else:
        if (c - b > b - a):
            return gss(f, a, b, x, tau)
        return gss(f, x, b, c, tau)        