#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:26:16 2018

Restoration on VAE

@author: syou
"""

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from vars_layers import input_set
from vae_graph import q_zx, p_xz, loss
from utils import num2str, path_make, data_load
from datetime import datetime
import h5py

# Assign GPU no
os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU'] #<GPU_NO>

# Parameter Settings

sequence_number = 1
mode = 'FsTVRestoration'
model = 'VanillaVAE'
start_time = datetime.utcnow().isoformat()
rho = np.array([2.2])
batchsize = 60 # restoration batch
imageshape = [158,198]
dim_z = 1 # latent variable z's channel
dim_x = imageshape[0]*imageshape[1]  # dimension of input
clipstd = [0.0,1.0]
interval = 500 # the interval to restore the output
niter = 1 # number of intervals
restore_steps = niter * interval
stepa = 99 # step of changing alpha
alphas = np.array([5e-3,2e-3]) # step size
gradient_clip = True
clipbound = 50
datapath = path_make('','data','')
"""
The same VAE graph
"""

# input

tf.reset_default_graph()
x, x_reshape = input_set(dim_x, imageshape[0], imageshape[1])
x_p, x_p_reshape = input_set(dim_x, imageshape[0], imageshape[1])
bmask_in = tf.placeholder(shape = [None]+ imageshape , dtype = tf.float32)
bmask_reshape = tf.reshape(bmask_in, shape = [-1, dim_x])
rho_ = tf.placeholder(tf.float32, shape=())

# the same graph applied 
qzx_kernels = np.tile([3,3,3,3,3,3,1], [2,1]) # 
qzx_channels = [64,64,64,64,64,64]
z_sampled, z_mean, z_std = q_zx(x_reshape,dim_z,qzx_kernels[0],qzx_kernels[1],qzx_channels)
pxz_kernels = np.tile([1,3,3,3,3,3,3], [2,1]) 
pxz_channels = [64, 64, 64, 64, 64, 64, 1]
xz_mean, xz_logvarinv  = p_xz(z_sampled, pxz_kernels[0], pxz_kernels[1], pxz_channels, clipstd) 
_, z_loss, _ = loss(z_mean, z_std, xz_mean, xz_logvarinv, x_reshape)
l2 = tf.reduce_sum(tf.squared_difference(tf.reshape(xz_mean, [-1, dim_x]), x), axis = 1)
xz_std = tf.exp(- 0.5 * xz_logvarinv)
# the loss for the restoration

TV = tf.image.total_variation(tf.subtract(x_p_reshape, x_reshape))
loss1 = -1*(l2 + z_loss)
loss2 = - rho_ * TV
# Gradient dloss/dy
grads = tf.gradients([loss1,loss2], [x])[0]
upperbound = tf.cast(tf.fill(tf.shape(grads), clipbound), dtype = tf.float32)
lowerbound = tf.cast( tf.fill(tf.shape(grads),-1*clipbound), dtype = tf.float32)
clipgrads = tf.clip_by_value(grads, lowerbound , upperbound, name = 'cliped_updating_gradient')  
gradimage = tf.reshape(clipgrads,shape = tf.stack([-1] + imageshape))
gradsmasked =  tf.multiply(clipgrads , bmask_reshape)

# tensorflow saver setting

saver = tf.train.Saver(tf.trainable_variables())   
tvars = tf.trainable_variables()
sess = tf.Session()

# set trained model path

modelpath = os.path.join(path_make(model,'CamCANT2','TrainModel'+ num2str(sequence_number)) , 'model','')

# load trained model
if os.path.exists(modelpath):
           
    file = open(modelpath+'model/checkpoint','r')  # this will read the first line of the save models. change if one will
    checkpoint = file.readline()
    file.close()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,checkpoint[24:-2])
    ckpt = tf.train.get_checkpoint_state(modelpath)
    trained_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])	

else: 
	raise ValueError('No Trained Model Imported')
    
# setup tensorboard log    

merged = tf.summary.merge_all()    

# gathering information of parameter settings

    
info = [['mode', 'slice' + mode], ['sequence number',sequence_number],
        ['batch size',batchsize], ['dimension of latent variable',dim_z],
        ['dimension of input image patch',dim_x], ['std lowerbound @ p_x_z output',clipstd],
        ['trained step of the model',trained_step], ['lambda(s)', rho], 
        ['step size, alpha', alphas], ['restoration steps', restore_steps],
        ['gradient_clip', gradient_clip], ['simulation',True], 
        ['clipbound for restoration gradient', clipbound], 
        ['model path', modelpath], ['start_time(UTC)',start_time]]
infodf = pd.DataFrame(info, columns = ['name','value'])
	
# Restore BraTSLGG      

restoredata = 'BraTSLGG'
MRtest, MRlabel = data_load(process = 'Test', dataset = 'BraTSLGG', datapath = datapath)
bmasks = h5py.File( datapath + 'BraTSLGG_data.hdf5', 'r')

for rho0 in rho:
    batchsize = 60 
    batch = bmasks['Mask'].shape[0]/batchsize
    indeces = np.arange(batch*batchsize)
    print 'restore for rho ', rho0
    savepath = path_make(model,restoredata, 'Dataslicehe0.06' + mode + num2str(sequence_number)+ '/' + str(rho0)) 
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    infodf.to_csv(savepath + 'Parameter_settings.csv', index=False)	
    np.save(os.path.join(savepath, 'index.npy'), indeces)
    for k in np.arange(batch+1):
        if k == batch:
            batchsize2 = bmasks['Label'].shape[0] - batch*batchsize +1
            index = range(bmasks['Label'].shape[0]-batchsize2, bmasks['Label'].shape[0])
            batchsize = batchsize2
        else:
            index = list(indeces[k*batchsize:(k+1)*batchsize] )  
        bmask = bmasks['Mask'][list(index)][:,22:180,17:215]
        rawData = MRtest[list(index)][:,22:180,17:215]      
        labels = MRlabel[list(index)][:,22:180,17:215]   
        step_restore =  np.zeros(list(rawData.shape) + [niter]) 
        MC2restore = rawData.reshape(-1,dim_x)
        MC_paint = MC2restore.copy()
        
        print 'Start restoration of image batch ', k
        for step in range(restore_steps):
            if step > stepa:
                alpha = alphas[-1]
            else: 
                alpha = alphas[0] 
            MC2restore += alpha * sess.run(gradsmasked,feed_dict = {x: MC2restore, x_p: MC_paint, rho_: rho0, bmask_in: bmask})
            if step == restore_steps - 1:
                step_restore[:,:,:,-1] = MC2restore.reshape(batchsize,imageshape[0],imageshape[1])           
        np.save(os.path.join(savepath, 'restored_images' + str(k) +'.npy'), step_restore) 
        
# Restore BraTSHGG      
        
restoredata = 'BraTSHGG'
MRtest, MRlabel = data_load(process = 'Test', dataset = 'BraTSHGG', datapath = datapath)

for rho0 in rho:
    batchsize = 60 
    batch = bmasks['Mask'].shape[0]/batchsize
    indeces = np.arange(batch*batchsize)
    print 'restore for rho ', rho0
    savepath = path_make(model,restoredata, 'Dataslicehe0.06' + mode + num2str(sequence_number)+ '/' + str(rho0)) 
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    infodf.to_csv(savepath + 'Parameter_settings.csv', index=False)	
    np.save(os.path.join(savepath, 'index.npy'), indeces)
    for k in np.arange(batch+1):
        if k == batch:
            batchsize2 = bmasks['Label'].shape[0] - batch*batchsize
            index = range(bmasks['Label'].shape[0]-batchsize2, bmasks['Label'].shape[0])
            batchsize = batchsize2
        else:
            index = list(indeces[k*batchsize:(k+1)*batchsize] )      
        bmask = bmasks['Mask'][list(index)][:,22:180,17:215]
        rawData = MRtest[list(index)][:,22:180,17:215]      
        labels = MRlabel[list(index)][:,22:180,17:215]   
        step_restore =  np.zeros(list(rawData.shape) + [niter]) 
        MC2restore = rawData.reshape(-1,dim_x)
        MC_paint = MC2restore.copy()        
        print 'Start restoration of image batch ', k
        for step in range(restore_steps):
            if step > stepa:
                alpha = alphas[-1]
            else: 
                alpha = alphas[0]        
            MC2restore += alpha * sess.run(gradsmasked,feed_dict = {x: MC2restore, x_p: MC_paint, rho_: rho0, bmask_in: bmask})    
            if step == restore_steps - 1:
                step_restore[:,:,:,-1] = MC2restore.reshape(batchsize,imageshape[0],imageshape[1])          
        np.save(os.path.join(savepath, 'restored_images' + str(k) +'.npy'), step_restore) 
print 'Restoration Finished'
sess.close()

