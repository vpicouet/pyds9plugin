#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:30:33 2020

@author: Vincent
"""
import numpy as np
import os, glob, sys
from pyds9plugin.DS9Utils import *

DS9_BackUp_path = '/Users/Vincent/DS9QuickLookPlugIn'

def main():
    np.savetxt(DS9_BackUp_path + '/.verbose.txt',[int(0)])
    d=DS9()
    name = d.get('xpa').split('\t')[-1]
    print('\n    Setup    \n' )
    os.system('DS9Utils %s open  "/Users/Vincent/Desktop/stack.fits" Slice  0 '%(name))
    os.system('DS9Utils %s  lock   image image 0  0 0    '%(name))
    os.system('DS9Utils %s  lock   wcs wcs 1  1 1    '%(name))
    os.system('DS9Utils %s  lock   none none 1  0 1    '%(name))
    
    os.system('DS9Utils %s setup  "Log"  50-99.9  0 cool 0 0 '%(name))
    d.set('regions command "circle 100 100 20"')
    d.set('regions select all')
    os.system('DS9Utils %s setup  "Log"  50-99.9  0 cool 0 0 '%(name))
    os.system('DS9Utils %s OriginalSettings'%(name))
    d.set('regions delete all')
    d.set('regions command "circle 100 100 200"')
    d.set('regions select all')
    os.system('DS9Utils %s setup  "Log"  50-99.9  0 cool 0 0 '%(name))
    
    print('\n    Header    \n' )
    os.system('DS9Utils %s CreateHeaderCatalog "/Users/Vincent/Github/DS9functions/pyds9plugin/Images/*.fits"  0 - '%(name))
    os.system('DS9Utils %s DS9createSubset %s all Directory,BITPIX "NAXIS>1 & FileSize_Mo>10"  &'%(name,glob.glob('/Users/Vincent/DS9QuickLookPlugIn/HeaderDataBase/*.csv')[0]))
    os.system('DS9Utils %s AddHeaderField test 1 no_comment "/Users/Vincent/Desktop/*.fits"  '%(name))
    
    
    print('\n    Regions    \n' )
    
    
    os.system('DS9Utils %s DS9Region2Catalog  "/Users/Vincent/test.csv" '%(name))
    d.set('regions delete all')
    os.system('DS9Utils %s DS9Region2Catalog  "/Users/Vincent/test.csv" '%(name))
    os.system('DS9Utils %s DS9Catalog2Region    "/Users/Vincent/test.csv"  xcentroid,ycentroid - circle 10 0 "-"  '%(name))
    d.set('regions select all')
    os.system('DS9Utils %s ReplaceWithNans nan 0 '%(name))
    
    print('\n    Image Processing     \n' )
    
    os.system('DS9Utils %s SubstractImage  -  "ds9+=np.random.normal(0,0.5*np.nanstd(ds9),size=ds9.shape)"  0  - 0 '%(name))
    os.system('DS9Utils %s SubstractImage  -  "ds9+=np.random.normal(0,0.5*np.nanstd(ds9),size=ds9.shape)"  0  "/Users/Vincent/Desktop/stack*.fits" 0 '%(name))
    
    DS9Plot(d,path='/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat')
    os.system('DS9Utils %s BackgroundFit1D  x none none 1 1 1 0 none'%(name))
    d.set('regions delete all')
    d.set('regions command "circle 477 472 20"')
    d.set('regions select all')
    os.system('DS9Utils %s fitsgaussian2D 0 '%(name))
    
    
    os.system('DS9Utils %s InterpolateNaNs %s '%(name, d.get('file')))
    d.set('regions delete all')
    d.set('regions command "box 477 472 100 99"')
    d.set('regions select all')
    
    os.system('DS9Utils %s Trimming Image  - '%(name))
    
    
    print('\n    Other     \n' )
    
    
    d.set('regions command "circle 50 50 50"')
    d.set('regions select all')
    os.system('DS9Utils %s PlotArea3D & '%(name))
    
    
    print('\n    INSTRUMENTATION AIT     \n' )
    os.system('DS9Utils %s radial_profile Maximum 0 0  '%(name))
    os.system('DS9Utils %s radial_profile 2D-Gaussian-fitting 10 0  '%(name))

    os.system('DS9Utils %s centering Maximum 0  '%(name))
    d.set('regions select all')
    os.system('DS9Utils %s centering 2D-Gaussian-fitting 0  '%(name))
    d.set('regions select all')
    os.system('DS9Utils %s centering 2x1D-Gaussian-fitting 0  '%(name))
    d.set('regions select all')
    os.system('DS9Utils %s centering Center-of-mass 0  '%(name))
    
    
    
    print('\n    Astronomical software     \n' )
    os.system('DS9Utils %s open  "/Users/Vincent/Desktop/stack.fits" Slice  0 '%(name))
    os.system('DS9Utils %s RunSextractor   NUMBER - - FITS_1.0 sex_vignet.param CCD 10 0 RELATIVE 0.8 2.0 1 gauss_4.0_7x7.conv 64 0.0003 1 1.0 CORRECT NONE 1 - 1 NONE OR 6,12,18 2.5,4.0 2.0,4.0 0.3,0.5,0.9 50000.0 SATURATE 0.0 4.0 GAIN 0 0.8 default.nnw NONE AUTO 0.0 64 3 LOCAL 24 0.0 3000 300000 1024 1 '%(name))

    
    np.savetxt(DS9_BackUp_path + '/.verbose.txt',[int(1)])





    return


# d.set('plot current add ')#line|bar|scatter
# d.set('plot load %s %s'%('/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat', 'xy'))
# d.set('plot  current graph 1')
# d.set('plot load %s %s'%('/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat', 'xy'))
# d.set("plot title {%s}"%(title))
# d.set("plot title x {%s}"%(xlabel))
# d.set("plot title y {%s}"%(ylabel))
    


if __name__ == '__main__':
    try:
        a = main()
    finally:
        np.savetxt(DS9_BackUp_path + '/.verbose.txt',[int(1)])
        

# os.system('DS9Utils %s '%(name))
# os.system('DS9Utils %s '%(name))
