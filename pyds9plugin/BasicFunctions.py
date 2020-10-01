#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:52:08 2020

@author: Vincent
"""

import os
import sys
import numpy as np
from pyds9 import DS9
from  pkg_resources  import resource_filename

try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()
#from shutil import which
#import matplotlib.pyplot as plt


       
#def verboseprint(*args, verbose=bool(int(np.load(os.path.join(resource_filename('pyds9plugin', 'config'),'verbose.npy'))))):
#    """Print function with a boolean verbose argument
#    
#    """
#    if bool(int(verbose)):
#        print(*args)
#    else:
#        pass
    
 

def RunFunction(Fonction, args, return_dict):
    """Run a function in order to performe multi processing
    """
    #print(*args)
    out = Fonction(*args)
    #verboseprint(out)
    return_dict['output'] = out
    return 




def FB_ADU2Flux(ADU, EMgain=1370, ConversionGain=1.8, dispersion=46.6/10):#old emgain=453
    """
    Convert FB2 ADUs/sec into photons per seconds per angstrom
    print((galex2Ph_s_A()-FB_ADU2Flux(300/50))/galex2Ph_s_A())
    """
    Flux = ADU * ConversionGain * dispersion / EMgain
    return Flux

#242/50/1370
#def(3600*np.arctan(0.010/500)*180/np.pi):
    


def Flux2FBADU(Flux, EMgain=453, ConversionGain=1.8, dispersion=46.6/10):
    """
    Convert FB2 ADUs into photons per seconds per angstrom
    """
    ADU = Flux * EMgain / ConversionGain / dispersion
    #Flux = ADU * ConversionGain * dispersion / EMgain / 10
    return ADU


def galex2Ph_s_A(f200=2.9e-4, atm=0.37, throughput=0.13, QE=0.5, area=7854):
    """
    Convert galex fluxes into photons [pupdate: electrons non???] per seconds per angstrom
    """
    Ph_s_A = f200*(throughput*atm* QE*area)
    return Ph_s_A



    # def connect(host='http://google.com'):
    # import urllib.request
    # try:
    #     urllib.request.urlopen(host) #Python 3.x
    #     return True
    # except:
    #     return False