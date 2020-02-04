#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:52:08 2020

@author: Vincent
"""
#import time
#import glob
import os
import sys
import numpy as np
from pyds9 import DS9
#import datetime
#from  pkg_resources  import resource_filename
#from astropy.table import Table
try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()
#from shutil import which
#import matplotlib.pyplot as plt


       
def verboseprint(args, verbose=True):
    """Print function with a boolean verbose argument
    """
    if bool(int(verbose)):
        print(*args)
    else:
        pass
    
    

def FitsExt(fitsimage):
    ext = np.where(np.array([type(ext.data) == np.ndarray for ext in fitsimage])==True)[0][0]
    print('Taking extension: ',ext)
    return ext

def ExecCommand(filename, path2remove, exp, config, eval_=False):
    """Combine two images and an evaluable expression 
    """
    import numpy as np
    from simpleeval import simple_eval, EvalWithCompoundTypes
    import astropy
    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel, convolve
    type_ = 'exec'
    fitsimage = fits.open(filename)
    ext = FitsExt(fitsimage) 
    fitsimage = fitsimage[ext]
    ds9 = fitsimage.data
    if os.path.isfile(path2remove) is False:
        if 'image'in exp:
           # from tkinter import messagebox
           # messagebox.showwarning( "File error","Image not found, please verify your path!")     
            d = DS9();d.set('analysis message {Image not found, please verify your path!}')
        else:
            image = 0
    else:
        fitsimage2 = fits.open(path2remove)[ext]
        image = fitsimage2.data

    #fitsimage.data = image - a * image2substract - b 
    #fitsimage.data = simple_eval(exp, names={"ds9": ds9, "image": image, "np":np})
    names = {"ds9": ds9, "image": image, "np":np}
    functions = {"convolve":convolve}
    print('Expression = ',exp)
    if type_ == 'eval': 
        try:
            fitsimage.data = EvalWithCompoundTypes(exp, names=names, functions=functions).eval(exp)
        except (TypeError) as e:
            if eval_:
                print('Simple_eval did not work: ', e)
                try:
                    fitsimage.data = eval(exp)
                    print('Using eval!')
                except SyntaxError:
                    print('Using exec!')
                    print(ds9)
                    exec(exp, globals=globals(), locals=locals())
                    print(ds9)
                    fitsimage.data = ds9
                    
            else:
                print(e)
                sys.exit()
    else:
        print('Using exec!')
        print(ds9)
        #import IPython; IPython.embed()
        ldict={'ds9':ds9,'image':image,'convolve':convolve}
        #exec("import IPython;IPython.embed()", globals(), ldict)#, locals(),locals())
        exec(exp, globals(), ldict)#, locals(),locals())
        #IPython.embed()
        ds9 = ldict['ds9']
        print(ds9)
        fitsimage.data = ds9#ds9
    name = filename[:-5] + '_modified.fits'
    #exp1, exp2 = int(fitsimage.header[my_conf.exptime[0]]), int(fitsimage2.header[my_conf.exptime[0]])
    fitsimage.header['DS9'] = filename
    fitsimage.header['IMAGE'] = path2remove
    try:
        fitsimage.header['COMMAND'] = exp
    except ValueError as e:
        print(e)
        print(len(exp))
        fitsimage.header.remove('COMMAND')
    fitsimage.writeto(name, overwrite=True)
    return fitsimage.data, name  


def DS9lock(xpapoint):
    """Lock all the images in DS9 together in frame, smooth, limits, colorbar
    """
    d = DS9(xpapoint)
    l = sys.argv[-3:]
    ll = np.array(l,dtype='U3')
    print(l,ll)
    l = np.array(l,dtype=int)
    ll[l==1]='yes'
    ll[l==0]='no'
#    if ll[0] == 'no':
#        d.set("lock frame %s"%(ll[0]))
#    else:
    d.set("lock frame %s"%(sys.argv[-5]))
    d.set("lock crosshair %s"%(sys.argv[-4]))
        
    d.set("lock scalelimits  %s"%(ll[-3]))
#    if ll[2] == 'no':
#    else:
#        d.set("lock crosshair physical")
    d.set("lock smooth  %s"%(ll[-2]))
    d.set("lock colorbar  %s"%(ll[-1]))

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
    Convert galex fluxes into photons per seconds per angstrom
    """
    Ph_s_A = f200*(throughput*atm* QE*area)
    return Ph_s_A



def connect(host='http://google.com'):
    import urllib.request
    try:
        urllib.request.urlopen(host) #Python 3.x
        return True
    except:
        return False