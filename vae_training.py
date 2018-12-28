#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 14:37:03 2018

Training VAE

@author: syou
"""

import numpy as np
import tensorflow as tf
import os
import pandas as pd
from vars_layers import input_set
from vae_graph import q_zx, p_xz, loss
from utils import num2str, path_make, data_load, recon , random_data
from datetime import datetime

# Assign GPU no
os.environ["CUDA_VISIBLE_DEVICES"]=os.environ['SGE_GPU'] #<GPU_NO>

# Parameter Settings

sequence_number = 1                           #  path
mode = 'TrainModel'                           #  settings
model = 'VanillaVAE'                          #
start_time = datetime.utcnow().isoformat()    #


batchsize = 16 # training batch
imageshape = [158,198] # this is calculated by bounding box of the whold dataset and network setting
dim_z = 1 # latent variable z's channel
dim_x = imageshape[0]*imageshape[1]  # dimension of image in vectors
clipstd = [0.0,1.0] # bound of std
start_step = 0 # default starting step
training_step = 400*4000 # how many steps to train at a time
train_rate = 5e-5

# Load Data

dataset = 'CamCANT2' 
datapath = path_make('','data','')
MRtrain = data_load(process = 'Train', dataset = dataset, datapath = datapath)
MRtest = data_load(process = 'Test', dataset = dataset, datapath = datapath)
test_image = MRtest[[73,  107,  185,  199,  382,  419,  443,  472,  509,  540,  554,    # some random images chosen for reconstruction during training
        713,  746,  757,  801,  861,  893,  943, 1091, 1104, 1137, 1164,                # indeces are random generated 
       1208, 1318, 1333, 1386, 1406, 1445, 1557, 1630, 1659, 1695, 1753,                # the cropping location is found by bounding box before hand
       1791, 1842, 1916, 2033, 2053, 2093, 2193, 2207, 2214, 2232, 2380,
       2387, 2413, 2441, 2521, 2539, 2776, 2809, 3025, 3050, 3055, 3121,
       3136, 3173, 3226, 3257, 3274]][:batchsize,22:22 + imageshape[0],17: 17+ imageshape[1]]  

# Set the Savepath

savepath = path_make(model,dataset, model + num2str(sequence_number))

if not os.path.exists(savepath):    
    os.makedirs(savepath)    
    
# gathering information of parameter settings

info = [['model', model], ['mode', mode], ['sequence_number',sequence_number],
        ['batchsize',batchsize], ['dim_z',dim_z],['dim_x',dim_x],
        ['start_step',start_step], ['training_step',training_step], 
        ['training_rate',train_rate], ['start_time(UTC)',start_time]]
infodf = pd.DataFrame(info, columns = ['name','value'])
infodf.to_csv(savepath + 'Parameter_settings.csv', index=False)	

"""
 set the Network

"""

# input

tf.reset_default_graph()
x, x_reshape = input_set(dim_x, imageshape[0], imageshape[1])

# endcoder parameter

qzx_kernels = np.tile([3,3,3,3,3,3,1], [2,1]) # 
qzx_channels = [64,64,64,64,64,64]

# encoding

z_sampled, z_mean, z_std = q_zx(x_reshape,dim_z,qzx_kernels[0],qzx_kernels[1],qzx_channels)

# decoder parameter

pxz_kernels = np.tile([1,3,3,3,3,3,3], [2,1]) 
pxz_channels = [64, 64, 64, 64, 64, 64, 1]

# decoding
 
xz_mean, xz_logvarinv  = p_xz(z_sampled, pxz_kernels[0], pxz_kernels[1], pxz_channels, clipstd) 

# loss function

t_loss, z_loss, l2_loss = loss(z_mean, z_std, xz_mean, xz_logvarinv, x_reshape)

# set optimizer

with tf.name_scope('train'):
    
	optimization = tf.train.AdamOptimizer(train_rate).minimize(t_loss)
    
# setup tensorboard log

merged = tf.summary.merge_all()

# saving model 

saver = tf.train.Saver(tf.trainable_variables())   
tvars = tf.trainable_variables()
sess = tf.Session()

# check whether to start training or continue training
if os.path.isdir(savepath+'model') == 1:
           
    file = open(savepath+'model/checkpoint','r')
    checkpoint = file.readline()
    file.close()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,checkpoint[24:-2])
    ckpt = tf.train.get_checkpoint_state(savepath+'model/')
    start_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
    adam_initializers = [var.initializer for var in tf.global_variables() if 'Adam' in var.name]
    sess.run(adam_initializers)

else: 
	start_step = 0
	sess.run(tf.global_variables_initializer())    
    
# set tensorboard output	
    
train_writer = tf.summary.FileWriter(savepath+'train', sess.graph)
test_writer = tf.summary.FileWriter(savepath+'test')

""" 
training session

"""

weight = z_mean.get_shape().as_list()[1] * z_mean.get_shape().as_list()[2] # this is to normalize the loss shown in the terminal against the receptive field

for step in range(start_step, start_step + training_step):
    if step % 20000 == 0 or step == start_step:
		print('Starting @step %s' % step)
        
    batch = random_data(MRtrain, batchsize = batchsize)[:batchsize,22:22 + imageshape[0],17: 17+ imageshape[1]]  
    batch = np.expand_dims(batch, axis = -1)
    
    t_los, z_los, l2_los, _ = sess.run([t_loss, z_loss, l2_loss, optimization], feed_dict={x: batch.reshape(batchsize,-1)})       
    
    if step == 0 or os.path.isdir(savepath+'model') == 0:
        
            os.makedirs(os.path.join(savepath,'model',''))        
        
    if step % 5000 == 0 and step >= start_step or step == start_step + training_step - 1:
        
            model_path = saver.save(sess, savepath+'model/model.ckpt',global_step=step)
            print("Model saved in file: %s" % model_path)
        
    if ((step % 200 == 0) and (step < 8000) ) or (step % 1000 == 0) or step == start_step + training_step - 1: # save to tensorboard
            summary = sess.run(merged,feed_dict={x: batch.reshape(batchsize,-1)})       
            train_writer.add_summary(summary, step)
            print('weighted train total loss, z loss, l2 loss @ step %s: %s || %s || %s' % (step, t_los/weight, z_los/weight, l2_los/weight))

    if ((step % 400 == 0) and (step <8000)) or (step % 1000 == 0) or step == start_step + training_step - 1:  # test, using random batch in test data
        
            batch_test = random_data(MRtest, batchsize = batchsize)[:batchsize,22:22 + imageshape[0],17: 17+ imageshape[1]]  
            batch_test = np.expand_dims(batch_test, axis = -1)

            summary, t_t_los, t_z_los, t_l2_los = sess.run([merged, t_loss, z_loss, l2_loss], feed_dict={x: batch_test.reshape(batchsize,-1)})
            test_writer.add_summary(summary, step)
            print('weighted test total loss, z loss, l2 loss @ step %s: %s || %s || %s' % (step, t_t_los/weight, t_z_los/weight, t_l2_los/weight))

    if step % 5000 == 0 or step == start_step + training_step - 1:  # Record execution stats
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary = sess.run(merged, feed_dict={x: batch_test.reshape(batchsize,-1)}, options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
            train_writer.add_summary(summary, step)
            print('Adding run metadata @step %s'% step)  
        
    if step % 5000 == 0 or step == start_step + training_step - 1 :
        reconpath = os.path.join(savepath,'reconstructions','')
        if not os.path.exists(reconpath):    
            os.makedirs(reconpath)  
        recon(xz_mean, -0.5*xz_logvarinv, sess, x, test_image[:batchsize].reshape(batchsize,-1), reconpath, step)        
        
print('Finished optimisation @step %s' % step)    
#train_writer.close()
#test_writer.close()
sess.close()
        