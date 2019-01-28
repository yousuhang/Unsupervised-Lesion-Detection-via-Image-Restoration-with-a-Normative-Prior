# Implementation for 'Unsupervised Lesion Detection via Image Restoration with a Normative-Prior'
(https://openreview.net/forum?id=S1xg4W-leV) by Suhang You, Kerem Tezcan, Xiaoran Chen, Ender Konukoglu

# Abstract
While human experts excel in and rely on identifying an abnormal structure when assessing a medical scan, without necessarily specifying the type, current unsupervised abnormality detection methods are far from being practical. Recently proposed deep-learning (DL) based methods were initial attempts showing the capabilities of this approach. In this work, we propose an outlier detection method combining image restoration with unsupervised learning based on DL. A normal anatomy prior is learned by training a Gaussian Mixture Variational Auto-Encoder (GMVAE) on images from healthy individuals. This prior is then used in a Maximum-A-Posteriori (MAP) restoration model to detect outliers. Abnormal lesions, not represented in the prior, are removed from the images during restoration to satisfy the prior and the difference between original and restored images form the detection of the method. We evaluated the proposed method on Magnetic Resonance Images (MRI) of patients with brain tumors and compared against previous baselines. Experimental results indicate that the method is capable of detecting lesions in the brain and achieves improvement over the current state of the art.
# Requirement
python 2.7
tensorflow 1.8
sklearn
numpy
pandas
# Brief on codes
vars_layers.py contains variable/tensor definition and basic layer construction
vae_graph.py contains main structure of vanilla VAE and GMVAE
vae_training.py and gmvae_training.py are codes for training session. The parameter setting of can be changed accordingly
vae_restore.py and gmvae_restore.py are codes for restoration process
# Acknowledgement
The tensorflow version of GMVAE implementation is inspired by Nat Dilokthanakul https://github.com/Nat-D/GMVAE
