#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:30:33 2020

@author: Vincent
"""
import numpy as np
import os, glob, sys
from pyds9plugin.DS9Utils import *
from shutil import copyfile, rmtree
from  pkg_resources  import resource_filename
from IPython import get_ipython
ipython = get_ipython()

DS9_BackUp_path = os.environ['HOME'] + '/DS9QuickLookPlugIn'
test_folder = '/Users/Vincent/Github/pyds9plugin/pyds9plugin/testing'#resource_filename('pyds9plugin', 'testing')
im  = '/Users/Vincent/Github/pyds9plugin/pyds9plugin/images/stack.fits'#resource_filename('pyds9plugin', 'images/stack.fits')
if os.path.exists(test_folder) is False:
    os.mkdir(test_folder)
files_folder = test_folder + '/files'
if os.path.exists(files_folder) is False:
    os.mkdir(files_folder)
def main():
    d=DS9n()
    name = d.get('xpa').split('\t')[-1]
    print('\n    Setup    \n' )
    os.system('DS9Utils open_file -x %s   -p " %s/stack.fits" -t Slice  -c 0 '%(name,files_folder))
    # ipython.magic("lprun -u 1e-1  -T /tmp/prun_open.py -s -r -f DS9open -f get DS9open('%s','%s/stack.fits')  "%(name,files_folder))
    # ipython.magic("lprun -u 1e-1  -T /tmp/prun_get.py -s -r -f d.get  d.get('file')" )

    os.system('DS9Utils lock -x %s  -f image -c  image -l 0 -l  0 -m 0    '%(name))
    os.system('DS9Utils lock -x %s  -f wcs  -c   wcs -l 1  -s 1  -m 1    '%(name))
    os.system('DS9Utils lock -x %s  -f none -c none  -l 1  -m 0  -m 1    '%(name))

    sys.exit()
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
    os.system('DS9Utils %s CreateHeaderCatalog "%s"  0 - '%(name,  resource_filename('pyds9plugin', 'Images/*.fits')))
    os.system('DS9Utils %s DS9createSubset %s all Directory,BITPIX "NAXIS>1 & FileSize_Mo>10"  &'%(name,glob.glob('/Users/Vincent/DS9QuickLookPlugIn/HeaderDataBase/*.csv')[0]))
    os.system('DS9Utils %s AddHeaderField test 1 no_comment "%s/Desktop/*.fits"  '%( os.environ['HOME'], name))


    print('\n    Regions    \n' )


    os.system('DS9Utils %s DS9Region2Catalog  "%s/test.csv" '%( name, files_folder))
    d.set('regions delete all')
    os.system('DS9Utils %s DS9Region2Catalog  "%s/test.csv" '%( name, files_folder))
    os.system('DS9Utils %s DS9Catalog2Region    "%s/test.csv"  x,y - circle 10 0 "-"  '%( name, files_folder))
    d.set('regions select all')
    os.system('DS9Utils %s ReplaceWithNans nan 0 '%(name))

    print('\n    Image Processing     \n' )

    os.system('DS9Utils %s SubstractImage  -  "ds9+=np.random.normal(0,0.5*np.nanstd(ds9),size=ds9.shape)"  0  - 0 '%(name))
    os.system('DS9Utils %s SubstractImage  -  "ds9+=np.random.normal(0,0.5*np.nanstd(ds9),size=ds9.shape)"  0  "%s/Desktop/stack*.fits" 0 '%( name, os.environ['HOME']))

    DS9Plot(d,path='%s/test.dat'%(test_folder))
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
    d.set('regions command "circle 50 50 50"');d.set('regions select all')
    # os.system('DS9Utils %s radial_profile Maximum 0 0  & '%(name))
    ipython.magic("lprun -T /tmp/prun_radial_profile.py -s -r -f DS9rp -f DS9plot_rp_convolved -f radial_profile_normalized DS9rp(%s,center_type='Maximum',fibersize=0)  "%(xpapoint))

    d.set('regions command "circle 50 50 50"');d.set('regions select all')
    os.system('DS9Utils %s radial_profile 2D-Gaussian-fitting 10 0 &  '%(name))

    d.set('regions command "circle 50 50 50"');d.set('regions select all')
    os.system('DS9Utils %s centering Maximum 0  '%(name))
    d.set('regions command "circle 50 50 50"');d.set('regions select all')
    os.system('DS9Utils %s centering 2D-Gaussian-fitting 0  '%(name))
    d.set('regions command "circle 50 50 50"');d.set('regions select all')
    os.system('DS9Utils %s centering 2x1D-Gaussian-fitting 0  '%(name))
    d.set('regions command "circle 50 50 50"');d.set('regions select all')
    os.system('DS9Utils %s centering Center-of-mass 0  '%(name))



    print('\n    Astronomical software     \n' )
    os.system('DS9Utils %s open  "%s/stack.fits" Slice  0 '%( name, files_folder))
    os.system('DS9Utils %s RunSextractor   NUMBER - - FITS_1.0 sex_vignet.param CCD 10 0 RELATIVE 0.8 2.0 1 gauss_4.0_7x7.conv 64 0.0003 1 1.0 CORRECT NONE 1 - 1 NONE OR 6,12,18 2.5,4.0 2.0,4.0 0.3,0.5,0.9 50000.0 SATURATE 0.0 4.0 GAIN 0 0.8 default.nnw NONE AUTO 0.0 64 3 LOCAL 24 0.0 3000 300000 1024 1 '%(name))
    print('\n    TEST COMPLETED 100%     \n' )
    os.system('echo 1 > %s'%(DS9_BackUp_path + '.verbose.txt'))
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
        os.system('echo 0 > %s'%(DS9_BackUp_path + '/.verbose.txt'))
        os.system('echo 0 > %s'%(DS9_BackUp_path + '/.message.txt'))
        copyfile(im, test_folder + '/files/' + os.path.basename(im))
        a = main()
        [os.remove(file) for file in glob.glob(os.environ['HOME'] +  '/Github/DS9functions/pyds9plugin/testing/files/*')]

    finally:
        os.system('echo 1 > %s'%(DS9_BackUp_path + '/.verbose.txt'))
        os.system('echo 1 > %s'%(DS9_BackUp_path + '/.message.txt'))


# os.system('DS9Utils %s '%(name))
# os.system('DS9Utils %s '%(name))
