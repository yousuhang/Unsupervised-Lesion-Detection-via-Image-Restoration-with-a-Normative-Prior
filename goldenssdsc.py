#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:20:25 2018

Calculate DSCs for 1, 5, 10 percent fpr

@author: syou
"""

import numpy as np
from utils import num2str
import sys
sys.path.append('...') % working folder
import pandas as pd
import os      


name2 = ['Lambda'] + [str(i) + ' mean' for i in[1.0,5.0,10.0]] + [str(i) + ' sd' for i in [1.0,5.0,10.0] ]
model = 'GMVAE'
statmode = 'abs3stats'
sequential_number = 1
rhos = np.arange(20)/5.0

ASDSC = []
MADSC = []
SADSC = []
for rho in rhos:		
	    statpath = os.path.join('%the working folder' + model + '/abs3statshe0.06FsTVRestoration', num2str(sequential_number),"{0:.1f}".format(rho))

	    HGstag = np.load(statpath + '/HGstag.npy')
	    LGstag = np.load(statpath + '/LGstag.npy')
	    stag  = np.concatenate((LGstag,HGstag), axis = 0)
            
            stp = stag[:,0]
            astp = np.sum(stp, axis = 0)
            stn = stag[:,1]
            astn = np.sum(stn, axis = 0)
            sfp = stag[:,2]
            asfp = np.sum(sfp, axis = 0)
            sfn = stag[:,3]
            asfn = np.sum(sfn, axis = 0)

            asdsc = 2.0*astp/(2.0*astp + asfp + asfn)
            madsc = np.mean(sdsc, axis = 0)
            sadsc = np.std(sdsc, axis = 0)

            if rho == rhos[0]:
                ASDSC = asdsc.reshape(1,-1)
                MADSC = madsc.reshape(1,-1)
                SADSC = sadsc.reshape(1,-1)
            else: 
                ASDSC = np.concatenate((ASDSC,asdsc.reshape(1,-1)),axis = 0)
                MADSC = np.concatenate((MADSC,madsc.reshape(1,-1)),axis = 0)
                SADSC = np.concatenate((SADSC,sadsc.reshape(1,-1)),axis = 0)


data2 = np.concatenate((np.array(rhos).reshape(1,-1),MADSC.T, SADSC.T
			))#np.array(rauc).reshape(1,-1)        

df2 = pd.DataFrame(data2)
df2['FPR percent']  = pd.Series(name2, index=df2.index)

savepath = '%working folder' + model + '/abs3stats/'
	writer2 = pd.ExcelWriter(savepath + 'he0.06FsTVRestoration'+ num2str(sequential_number)+'fprdscs5.xlsx')

df2.to_excel(writer2)
writer2.save()        
        
