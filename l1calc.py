#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:33:33 2018

@author: syou
"""
import numpy as np
import sys
sys.path.append("/scratch_net/biwidl211_second/syou/thesis/extension/")
#from utils import roc_score2, dsc
import h5py
from utils import num2str
import os
import tensorflow as tf

def tvr_ssim(tvorssim):
    if len(tvorssim.shape) == 3:
        r = tvorssim[:,0,:]/(tvorssim[:,1,:] + 1e-8)
        return np.mean(r)
    elif len(tvorssim.shape) == 2:
        return np.mean(tvorssim)


batchsize = 68 
imageshape = [158,198]
origin =  tf.placeholder(tf.float32, [None, 158,198,1], name='ori')  
restored =   tf.placeholder(tf.float32, [None, 158,198,1], name='res')  
TV1 = tf.image.total_variation(tf.subtract(origin, restored))
#TV2 = tf.image.total_variation(origin)
#SSIM = tf.image.ssim(origin, restored, max_val=1.0)
L1 = tf.reduce_sum(tf.abs(tf.subtract(origin, restored)), axis = [1,2,3])
L1r = tf.reduce_sum(tf.abs(tf.subtract(origin, restored)/(origin+1e-8)), axis = [1,2,3])
batch = 97
sequential_number = 1
mode = 'FsTVRestoration'
sess = tf.Session()

if mode == 'FsTVRestoration':
    rhos = np.array([2.6,2.8,3.0,3.2,3.4,3.6])#(np.arange(10,15)/5.0)#[1::2]#np.append(np.arange(10)/5.0, np.array([2.5,3.0,3.5,4.0]))# np.append(np.arange(10)/10.0,np.arange(5,25)/5.0) #,np.append(np.arange(5,12)/5.0,np.array([3.0,3.6,4.0,6.0,10.0])))
elif mode == 'FsSSIMRestoration':
    rhos = np.array([2750.0])#np.arange(13)*2500.0#np.append(np.arange(4,19)*500.0 ,np.arange(3,7)*5000.0)

camdata = h5py.File('/scratch_net/biwidl211_second/syou/thesis/extension/data/CamCANT2_data.hdf5','r')
TVS = []
L1S = []
L1R = []
L1Rr = []
#aa = []
for rho in rhos:
    respath = '/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/'+ str(rho)
    
#    HGGpath = '/media/syou/Elements/paperresult/thesis/extension/VanillaVAE/BraTSHGG/Dataslice' + HE + mode + num2str(7) +'/'+ str(rho)
    savepath = os.path.join('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/camstats', mode  + num2str(sequential_number), str(rho))
    
#    aa += [np.load(savepath + '/tvorssim.npy' )]
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
#    
#    camrestore = h5py.File(respath + '/restored_images.hdf5', 'r')
##    HGrestore = h5py.File(HGGpath + '/restored_images.hdf5', 'r')
#    print 'starting for LGG'
#    startind = 0
#    LGGauc = np.zeros(len(LGGindex))
#    LGGroc = []
#    LGGdsc = []
#    threshs = np.linspace(0, 1, 201)
#    LGsta = []

    for j in range(1): 
        print 'calculating statistics for '+ mode+'rho = ', rho, 'niter', j
        value_to_save = []
        value_to_save2 = []
        value_to_save3 = []
        value_to_save4 = []
        for i in range(batch):
            
    #        nslice = LGGindex[i].astype(int)
    #        endind = startind + nslice
            ori = np.expand_dims(camdata['Test'][i*batchsize:(i+1)*batchsize,22:180,17:215], axis = -1)
    #        mask = ori > 0
            res = np.expand_dims(np.load(respath +  '/restored_images' + str(i) +'.npy')[...,j], axis = -1)
            print 'batch = ' ,i, ' intensity > 2 ', np.sum(res > 2.0),' intensity > 1.5 ', np.sum(res > 1.5), ' intensity > 1.2 ', np.sum(res > 1.2), ' intensity > 1.0 ', np.sum(res > 1.0)
            if mode == 'FsTVRestoration':
                tv , l1 = sess.run([TV1,L1], feed_dict={origin: ori, restored: res})
                l1r = l1/np.sum(ori, axis = (1,2,3))
                l1r_ = sess.run(L1r, feed_dict={origin: ori, restored: res})
#                tvs = np.stack((tv1,tv2), axis = 0)
#                tvssim = sess.run(SSIM, feed_dict={origin: ori, restored: res})
                value_to_save += [tv]
                value_to_save2 += [l1]
                value_to_save3 += [l1r]
                value_to_save4 += [l1r_]
#            elif mode == 'FsSSIMRestoration':
#                ssim = sess.run(SSIM, feed_dict={origin: ori, restored: res})
#                value_to_save += [ssim]
            
    #        _, meanstd, _, aucs = roc_score2(pmask.astype(int), dif)  
    #        LGGauc[i] = aucs[0][0]
    #        LGGroc += [meanstd]        
    #        cur_dsc = np.array([dsc((dif > thresh).astype(np.float), pmask) for thresh in threshs])
    #        LGGdsc += [cur_dsc]
    
    #    np.save(savepath + '/LGGroc.npy', np.array(LGGroc))
    #    np.save(savepath + '/LGGauc.npy', LGGauc)
    #    np.save(savepath + '/LGGdsc.npy', np.array(LGGdsc))
        np.save(savepath + '/tv' + str(j) + '.npy', np.array(value_to_save))
        np.save(savepath + '/l1' + str(j) + '.npy', np.array(value_to_save2))
        np.save(savepath + '/l1r' + str(j) + '.npy', np.array(value_to_save3))
        np.save(savepath + '/nl1r' + str(j) + '.npy', np.array(value_to_save4))
#        if mode == 'FsTVRestoration':
#            np.save(savepath + '/ssim' + str(j) + '.npy', np.array(value_to_save2))
        tvs = np.load(savepath + '/tv' + str(j) + '.npy')
        l1s = np.load(savepath + '/l1' + str(j) + '.npy')
        l1r = np.load(savepath + '/l1r' + str(j) + '.npy')
        nl1r = np.load(savepath + '/l1r' + str(j) + '.npy')
        TVS += [np.mean(tvs)]
        L1S += [np.mean(l1s)]
        L1R += [np.mean(l1r)]
        L1Rr += [np.mean(nl1r)]
if not os.path.exists('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/'):
    os.makedirs('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/')        
#        np.save('/scratch_net/biwidl211_second/syou/thesis/extension/VanillaVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/tv_' + str(rho)  + str(j) + '.npy',np.mean(tvs))
#        np.save('/scratch_net/biwidl211_second/syou/thesis/extension/VanillaVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/l1_' + str(rho)  + str(j) + '.npy',np.mean(l1s))
np.save('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/tvs_' + str(j) + '.npy',np.array(TVS))
np.save('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/l1s_' + str(j) + '.npy',np.array(L1S))
np.save('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/l1r_' + str(j) + '.npy',np.array(L1R))
np.save('/scratch_net/biwidl211_second/syou/thesis/extension/GMVAE/CamCANT2/Dataslicehe0.06' + mode + num2str(sequential_number) +'/l1rr_' + str(j) + '.npy',np.array(L1Rr))
#        if mode == 'FsTVRestoration':
#            ssim = np.load(savepath + '/ssim' + str(j) + '.npy')
#            p2 = tvr_ssim(ssim)
#            np.save('/scratch_net/biwidl211_second/syou/thesis/extension/VanillaVAE/CamCANT2/Dataslice' + mode + num2str(sequential_number) +'/ratiossim_' + str(rho)  + str(j) + '.npy',p2)
