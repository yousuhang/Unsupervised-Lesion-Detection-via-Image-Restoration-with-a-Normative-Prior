#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:04:58 2018

Network Structure for VAE and GMVAE

Plus related functions

@author: syou
"""

import numpy as np
import tensorflow as tf
from vars_layers import  conv_layer, upconv_layer

"""

Maths involved in tensorflow

"""
# loglikelihood using inverse of log variance

def log_likelihood_inv(x, mu, logvarinv, axis=-1):   
    
    return -0.5 *tf.reduce_sum(tf.log(2 * np.pi)  - logvarinv , axis), -0.5 *tf.reduce_sum(tf.square(x - mu) * tf.exp(logvarinv), axis)

"""

Network Structure of VAE

"""
# the encoder q(z|x)

def q_zx(input_x, dim_z,kernelx, kernely, output_channel):
    
    conv_1 = conv_layer(input_x, kernelx[0], kernely[0], output_channel[0], 'q_wz_x/3x3convlayer', stride = 2, padding = 'VALID')
    conv_2 = conv_layer(conv_1, kernelx[1], kernely[1], output_channel[1], 'q_wz_x/3x3convlayer1', stride = 1, padding = 'VALID') 
    conv_3 = conv_layer(conv_2, kernelx[2], kernely[2], output_channel[2], 'q_wz_x/3x3convlayer2', stride = 1, padding = 'VALID')    
    conv_4 = conv_layer(conv_3, kernelx[3], kernely[3], output_channel[3], 'q_wz_x/3x3convlayer3', stride = 2, padding = 'VALID')    
    conv_5 = conv_layer(conv_4, kernelx[4], kernely[4], output_channel[4], 'q_wz_x/3x3convlayer4', stride = 1, padding = 'VALID') 
    conv_6 = conv_layer(conv_5, kernelx[5], kernely[5], output_channel[5], 'q_wz_x/3x3convlayer5', stride = 1, padding = 'VALID')    
    z_mean = conv_layer(conv_6, kernelx[6], kernely[6], dim_z , 'q_zx/z_mean', act=tf.identity)
    z_logvar = conv_layer(conv_6, kernelx[6], kernely[6], dim_z , 'q_zx/z_logvar', act=tf.identity)
    upsilon_z = tf.random_normal(tf.shape(z_mean), 0, 1,dtype=tf.float32) 
    z_sampled = z_mean + (tf.exp(0.5*z_logvar) * upsilon_z) #here now as input of decoder
    tf.summary.histogram('z_sampled', z_sampled)
    print conv_1.get_shape().as_list(), conv_2.get_shape().as_list(), conv_3.get_shape().as_list(), conv_4.get_shape().as_list(), conv_5.get_shape().as_list(), conv_6.get_shape().as_list(), z_sampled.get_shape().as_list()
    return z_sampled, z_mean, z_logvar    

# the decoder  p(x|z)
    
def p_xz(input_z, kernelx, kernely, output_channel, clipstd):  
    pad = tf.constant([[0,0],[2, 2,], [2, 2],[0,0]])
    
    conv_1 = conv_layer(input_z, kernelx[0], kernely[0], output_channel[0], 'p_xz/1x1convlayer')
    
    outputshape1 = [conv_1.get_shape().as_list()[0], conv_1.get_shape().as_list()[1] -1 + kernelx[1],
                    conv_1.get_shape().as_list()[2] -1  + kernely[1],  output_channel[1] ]
    
    conv_2 = upconv_layer(conv_1, kernelx[1], kernely[1], outputshape1, 'p_xz/3x3upconvlayer1', stride = 1)
    
    outputshape2 = [conv_2.get_shape().as_list()[0], 1 * (conv_2.get_shape().as_list()[1] -1 ) + kernelx[2],
                    1 * (conv_2.get_shape().as_list()[2] -1 ) + kernely[2],  output_channel[2] ]
    
    conv_3 = upconv_layer(conv_2, kernelx[2], kernely[2], outputshape2, 'p_xz/3x3upconvlayer2', stride = 1) 
        
    outputshape3 = [2 * conv_3.get_shape().as_list()[1], 2*conv_3.get_shape().as_list()[2]]
    
    upconv_3 = tf.image.resize_images(conv_3, outputshape3 ,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    padupconv_3 = tf.pad(upconv_3, pad , 'CONSTANT')
    
    conv_4 = conv_layer(padupconv_3, kernelx[3], kernely[3], output_channel[3], 'p_xz/3x3upconvlayer3', stride = 1, padding = 'VALID')     
    
    outputshape4 = [conv_4.get_shape().as_list()[0], 1 * (conv_4.get_shape().as_list()[1] -1 ) + kernelx[4],
                    1 * (conv_4.get_shape().as_list()[2] -1 ) + kernely[4], output_channel[4]]
    conv_5 = upconv_layer(conv_4, kernelx[4], kernely[4], outputshape4, 'p_xz/3x3upconvlayer4', stride = 1) 
    
    outputshape5 = [conv_5.get_shape().as_list()[0], 1 * (conv_5.get_shape().as_list()[1] -1 ) + kernelx[5],
                    1 * (conv_5.get_shape().as_list()[2] -1 ) + kernely[5], output_channel[5] ]
    
    conv_6 = upconv_layer(conv_5, kernelx[5], kernely[5], outputshape5, 'p_xz/3x3upconvlayer5', stride = 1)    

    
    outputshape6 = [2 * conv_6.get_shape().as_list()[1], 2*conv_6.get_shape().as_list()[2]]
    
    upconv_6 = tf.image.resize_images(conv_6, outputshape6 ,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    padupconv_6 = tf.pad(upconv_6, pad , 'CONSTANT') 
    
    out_y_mean = conv_layer(padupconv_6, kernelx[6], kernely[6], output_channel[6], 'p_xz/y_mean', act = tf.identity, padding = 'VALID')
    out_y_logvarinv_no_clip = conv_layer(padupconv_6, kernelx[6], kernely[6], output_channel[6], 'p_xz/y_logvarinv', act = tf.identity, bias = 0.1, padding = 'VALID')

    upperbound = tf.cast(tf.fill(tf.shape(out_y_logvarinv_no_clip), -2*np.log(clipstd[0]+1e-8)), dtype = tf.float32)
    lowerbound = tf.cast( tf.fill(tf.shape(out_y_logvarinv_no_clip),-2*np.log(clipstd[-1])), dtype = tf.float32)
    out_y_logvarinv = tf.clip_by_value(out_y_logvarinv_no_clip, lowerbound , upperbound, name = 'cliped_y_logvarinv')  
    print conv_1.get_shape().as_list(), conv_2.get_shape().as_list(), conv_3.get_shape().as_list(), conv_4.get_shape().as_list(), conv_5.get_shape().as_list(), conv_6.get_shape().as_list(), out_y_mean.get_shape().as_list()
    
    tf.summary.histogram('p_xz/output_y_mean', out_y_mean)
    tf.summary.histogram('p_xz/output_y_logstd', out_y_logvarinv)
    return out_y_mean, out_y_logvarinv

# loss function of VAE
    
def loss(z_mean,  z_logvar  , recon_mean, recon_logvarinv, input):
    
    with tf.name_scope('loss'):
        z_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(z_logvar) - z_logvar  - 1, axis = [1,2,3]) 

        meadim_z_loss = tf.reduce_mean(z_loss)
        tf.summary.scalar('z_loss',meadim_z_loss)     
        l2_loss1, l2_loss2 = log_likelihood_inv(input, recon_mean, recon_logvarinv, axis = [1,2,3])
        l2_loss = -1.0*l2_loss1-1.0*l2_loss2
        mean_l2_loss = tf.reduce_mean(l2_loss)
        tf.summary.scalar('l2_loss',mean_l2_loss)
        	
        t_loss = tf.reduce_mean(z_loss + l2_loss)
        tf.summary.scalar('t_loss',t_loss)
    return t_loss, meadim_z_loss, mean_l2_loss

"""

Network Structure of GMVAE

"""

# then encoder q(z|x) and q(w|x)

def q_wz_x(input_x, dim_z, dim_w, dim_c, kernelx, kernely, output_channel):
    
    conv_1 = conv_layer(input_x, kernelx[0], kernely[0], output_channel[0], 'q_wz_x/3x3convlayer', stride = 2, padding = 'VALID')
    conv_2 = conv_layer(conv_1, kernelx[1], kernely[1], output_channel[1], 'q_wz_x/3x3convlayer1', stride = 1, padding = 'VALID') 
    conv_3 = conv_layer(conv_2, kernelx[2], kernely[2], output_channel[2], 'q_wz_x/3x3convlayer2', stride = 1, padding = 'VALID')    
    conv_4 = conv_layer(conv_3, kernelx[3], kernely[3], output_channel[3], 'q_wz_x/3x3convlayer3', stride = 2, padding = 'VALID')    
    conv_5 = conv_layer(conv_4, kernelx[4], kernely[4], output_channel[4], 'q_wz_x/3x3convlayer4', stride = 1, padding = 'VALID') 
    conv_6 = conv_layer(conv_5, kernelx[5], kernely[5], output_channel[5], 'q_wz_x/3x3convlayer5', stride = 1, padding = 'VALID')    
    w_mean = conv_layer(conv_6, kernelx[6], kernely[6], dim_w, 'q_wz_x/w_mean', act=tf.identity)
    w_logvar = conv_layer(conv_6, kernelx[6], kernely[6], dim_w, 'q_wz_x/w_logvar', act=tf.identity)
    upsilon_w = tf.random_normal(tf.shape(w_mean), 0, 1, dtype=tf.float32)
    w_sampled = w_mean + (tf.exp(0.5*w_logvar) * upsilon_w)
    tf.summary.histogram('q_wz_x/w_sampled', w_sampled)
    z_mean = conv_layer(conv_6, kernelx[6], kernely[6], dim_z , 'q_wz_x/z_mean', act=tf.identity)
    z_logvar = conv_layer(conv_6, kernelx[6], kernely[6], dim_z , 'q_wz_x/z_logvar', act=tf.identity)
    upsilon_z = tf.random_normal(tf.shape(z_mean), 0, 1,dtype=tf.float32) 
    z_sampled = z_mean + (tf.exp(0.5*z_logvar) * upsilon_z) 
    tf.summary.histogram('z_sampled', z_sampled)
#    print conv_1.get_shape().as_list(), conv_2.get_shape().as_list(), conv_3.get_shape().as_list(), conv_4.get_shape().as_list(), conv_5.get_shape().as_list(), conv_6.get_shape().as_list(), z_sampled.get_shape().as_list()
    return w_sampled, w_mean, w_logvar, z_sampled, z_mean, z_logvar    

# the decoder p(x|z,w)
    
def p_x_z(input_z, kernelx, kernely, output_channel, clipstd):  
    pad = tf.constant([[0,0],[2, 2,], [2, 2],[0,0]])
    
    conv_1 = conv_layer(input_z, kernelx[0], kernely[0], output_channel[0], 'p_x_z/1x1convlayer')
    
    outputshape1 = [conv_1.get_shape().as_list()[0], conv_1.get_shape().as_list()[1] -1 + kernelx[1],
                    conv_1.get_shape().as_list()[2] -1  + kernely[1],  output_channel[1] ]
    
    conv_2 = upconv_layer(conv_1, kernelx[1], kernely[1], outputshape1, 'p_x_z/3x3upconvlayer1', stride = 1)
    
    outputshape2 = [conv_2.get_shape().as_list()[0], 1 * (conv_2.get_shape().as_list()[1] -1 ) + kernelx[2],
                    1 * (conv_2.get_shape().as_list()[2] -1 ) + kernely[2],  output_channel[2] ]
    
    conv_3 = upconv_layer(conv_2, kernelx[2], kernely[2], outputshape2, 'p_x_z/3x3upconvlayer2', stride = 1) 
        
    outputshape3 = [2 * conv_3.get_shape().as_list()[1], 2*conv_3.get_shape().as_list()[2]]
    
    upconv_3 = tf.image.resize_images(conv_3, outputshape3 ,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    padupconv_3 = tf.pad(upconv_3, pad , 'CONSTANT')
    
    conv_4 = conv_layer(padupconv_3, kernelx[3], kernely[3], output_channel[3], 'p_x_z/3x3upconvlayer3', stride = 1, padding = 'VALID')     
    
    outputshape4 = [conv_4.get_shape().as_list()[0], 1 * (conv_4.get_shape().as_list()[1] -1 ) + kernelx[4],
                    1 * (conv_4.get_shape().as_list()[2] -1 ) + kernely[4], output_channel[4]]
    conv_5 = upconv_layer(conv_4, kernelx[4], kernely[4], outputshape4, 'p_x_z/3x3upconvlayer4', stride = 1) 
    
    outputshape5 = [conv_5.get_shape().as_list()[0], 1 * (conv_5.get_shape().as_list()[1] -1 ) + kernelx[5],
                    1 * (conv_5.get_shape().as_list()[2] -1 ) + kernely[5], output_channel[5] ]
    
    conv_6 = upconv_layer(conv_5, kernelx[5], kernely[5], outputshape5, 'p_x_z/3x3upconvlayer5', stride = 1)    

    
    outputshape6 = [2 * conv_6.get_shape().as_list()[1], 2*conv_6.get_shape().as_list()[2]]
    
    upconv_6 = tf.image.resize_images(conv_6, outputshape6 ,method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    
    padupconv_6 = tf.pad(upconv_6, pad , 'CONSTANT') 
    
    out_y_mean = conv_layer(padupconv_6, kernelx[6], kernely[6], output_channel[6], 'p_x_z/y_mean', act = tf.identity, padding = 'VALID')
    out_y_logvarinv_no_clip = conv_layer(padupconv_6, kernelx[6], kernely[6], output_channel[6], 'p_x_z/y_logvarinv', act = tf.identity, bias = 0.1, padding = 'VALID')

    upperbound = tf.cast(tf.fill(tf.shape(out_y_logvarinv_no_clip), -2*np.log(clipstd[0]+1e-8)), dtype = tf.float32)
    lowerbound = tf.cast( tf.fill(tf.shape(out_y_logvarinv_no_clip),-2*np.log(clipstd[-1])), dtype = tf.float32)
    out_y_logvarinv = tf.clip_by_value(out_y_logvarinv_no_clip, lowerbound , upperbound, name = 'cliped_y_logvarinv')  
#    print conv_1.get_shape().as_list(), conv_2.get_shape().as_list(), conv_3.get_shape().as_list(), conv_4.get_shape().as_list(), conv_5.get_shape().as_list(), conv_6.get_shape().as_list(), out_y_mean.as_list()
    
    tf.summary.histogram('p_x_z/output_y_mean', out_y_mean)
    tf.summary.histogram('p_x_z/output_y_logstd', out_y_logvarinv)
    return out_y_mean, out_y_logvarinv

# posterior p(z|w,c)

def p_z_wc(input_w, dim_c , dim_z, kernelx, kernely, output_channel): #output_channel[1] = dim_z * dim_c
    conv_1 = conv_layer(input_w, kernelx[0], kernely[0], output_channel[0], 'p_z_wc/1x1convlayer')
    z_wc_mean = conv_layer(conv_1, kernelx[0], kernely[0], output_channel[1], 'p_z_wc/z_wc_mean', act = tf.identity)
    z_wc_logvarinv = conv_layer(conv_1, kernelx[0], kernely[0], output_channel[1], 'p_z_wc/z_wc_logvarinv', act = tf.identity, bias = 0.1)
    z_wc_means = tf.reshape(z_wc_mean, [-1, z_wc_mean.get_shape().as_list()[1],
                                        z_wc_mean.get_shape().as_list()[2],dim_z, dim_c])
    z_wc_logvarinvs = tf.reshape(z_wc_logvarinv, [-1, z_wc_logvarinv.get_shape().as_list()[1],
                                                  z_wc_logvarinv.get_shape().as_list()[2],dim_z, dim_c])
    upsilon_z_wc = tf.random_normal(tf.shape(z_wc_means), 0, 1,dtype=tf.float32) 
    z_wc_sampled = z_wc_means + (tf.exp(-0.5*z_wc_logvarinvs) * upsilon_z_wc)
    return z_wc_sampled, z_wc_means, z_wc_logvarinvs

# prior p(c)

def p_c(z_sample, pzc_mean, pzc_logvarinv, dim_c):
    
    z_sample = tf.tile(tf.expand_dims(z_sample, -1), [1,1,1,1, dim_c])
    loglh = -0.5 * (tf.squared_difference(z_sample, pzc_mean)*tf.exp(pzc_logvarinv)) - pzc_logvarinv + tf.log(np.pi)
    loglh_sum = tf.reduce_sum(loglh, 3)
    pc_logit = loglh_sum
    pc = tf.nn.softmax(loglh_sum)
    return pc_logit, pc

# loss function of GMVAE
    
# loss functions
# the loss function consist of 4 parts
# 1. the reconstruction loss
def gmloss(z_mean, z_logvar, z_c_mean, z_c_logvarinv, w_mean, w_logvar, dim_c, c, logc, recon_mean, recon_logvarinv, input_tensor, z_sample, clambda = 0):

    p_loss_part1,  p_loss_part2= log_likelihood_inv(input_tensor, recon_mean, recon_logvarinv, axis = [1,2,3])
    p_loss_part1 = - p_loss_part1
    p_loss_part2 = - p_loss_part2 
    p_loss = p_loss_part1 +  p_loss_part2
    mean_p_loss = tf.reduce_mean(p_loss)
                                
        
# 2. E_c_w[KL(q(z|x)|| p(z|w, c))]      
# calculate KL for each cluster
# KL  = 1/2(  logvar2 - logvar1 + (var1 + (m1-m2)^2)/var2  - 1 ) here dim_c clusters, then we have batchsize * dim_z * dim_c
# then [batchsize * dim_z* dim_c] * [batchsize * dim_c * 1]  = batchsize * dim_z * 1, squeeze it to batchsize * dim_z
    z_mean = tf.tile(tf.expand_dims(z_mean, -1), [1, 1,1,1, dim_c])
    z_logvar = tf.tile(tf.expand_dims(z_logvar, -1), [1, 1,1,1, dim_c])  
    d_mu_2 = tf.squared_difference(z_mean, z_c_mean)
    d_var = (tf.exp(z_logvar)+d_mu_2)*(tf.exp(z_c_logvarinv)+1e-6)
    d_logvar = -1*(z_c_logvarinv+z_logvar)   
    KL = (d_var + d_logvar -1) *0.5         
    con_prior_loss =  tf.reduce_sum(tf.squeeze(tf.matmul(KL,tf.expand_dims(c,-1)),-1), [1,2,3])
    mean_con_loss = tf.reduce_mean(con_prior_loss)
        
# 3. KL(q(w|x)|| p(w) ~ N(0, I))
# KL = 1/2 sum( mu^2 + var - logvar -1 )

    w_loss = 0.5 * tf.reduce_sum(tf.square(w_mean) + tf.exp(w_logvar) - w_logvar -1,[1,2,3]) 
    mean_w_loss = tf.reduce_mean(w_loss)        
    
# 4. KL(q(c|z)||p(c)) =  - sum_k q(k) log p(k)/q(k) , k = dim_c
#let p(k) = 1/K#
    
    closs1 = tf.reduce_sum(tf.multiply(c,tf.log(c*dim_c+1e-8)), [3])
    c_lambda = tf.cast(tf.fill(tf.shape(closs1), clambda), dtype = tf.float32)
    c_loss = tf.maximum(closs1, c_lambda)
    c_loss = tf.reduce_sum(c_loss, [1,2])
    mean_c_loss = tf.reduce_mean(c_loss)    
    
    t_loss = mean_p_loss + mean_con_loss + mean_w_loss + mean_c_loss
    p_t_loss = t_loss/c.get_shape().as_list()[1]/c.get_shape().as_list()[2]
    p_mean_p_loss = mean_p_loss/c.get_shape().as_list()[1]/c.get_shape().as_list()[2]
    p_mean_con_loss = mean_con_loss/c.get_shape().as_list()[1]/c.get_shape().as_list()[2]
    p_mean_c_loss = mean_c_loss/c.get_shape().as_list()[1]/c.get_shape().as_list()[2]
    p_mean_w_loss = mean_w_loss/c.get_shape().as_list()[1]/c.get_shape().as_list()[2]
    with tf.name_scope('loss'):
        tf.summary.scalar('recon',mean_p_loss)
        tf.summary.scalar('conditional_prior_loss',mean_con_loss)      
        tf.summary.scalar('w_prior_loss',mean_w_loss)
        tf.summary.scalar('c_prior_loss',mean_c_loss)
        tf.summary.scalar('total_loss',t_loss)
    with tf.name_scope('patchwise_loss'):
        tf.summary.scalar('recon',p_mean_p_loss)
        tf.summary.scalar('conditional_prior_loss',p_mean_con_loss)      
        tf.summary.scalar('w_prior_loss',p_mean_w_loss)
        tf.summary.scalar('c_prior_loss',p_mean_c_loss)
        tf.summary.scalar('total_loss',p_t_loss)
    return t_loss, mean_p_loss, mean_con_loss, mean_w_loss, mean_c_loss