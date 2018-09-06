#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:13 2018

@author: Vincent
"""

from __future__ import print_function
import timeit
import glob
import os
import  sys
import json
import numpy as np
from pyds9 import DS9
import datetime
sys.path.append(os.path.dirname(os.path.realpath(__file__)))     
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

#print(sys.path)
#from collections import namedtuple
#from pyds9 import *

#from astropy.io import fits

#import matplotlib.pyplot as plt
#from astropy.table import Table

#from scipy.optimize import curve_fit
#from scipy import interpolate
#from scipy import stats
#from scipy import ndimage

#start = timeit.default_timer()
#from focustest import ConvolveBoxPSF
#from focustest import AnalyzeSpot
#from focustest import plot_rp2_convolved_wo_latex
#from focustest import  radial_profile_normalized
#from focustest import stackImages  
#from focustest import ConvolveDiskGaus2D
#from focustest import twoD_Gaussian
#from focustest import estimateBackground
#from focustest import create_DS9regions2
#from focustest import Gaussian  
#from focustest import ConvolveSlit2D_PSF

#stop = timeit.default_timer()
#print('Packages imported...')
#print('Import time = {}s'.format(stop-start))

#print(print_function)
#print(glob, os, sys, json)
#print(np, fits, namedtuple, curve_fit)
#print(DS9, interpolate, stats, ndimage)
#print(AnalyzeSpot,ConvolveBoxPSF,plot_rp2_convolved_wo_latex,radial_profile_normalized)
#print(ConvolveDiskGaus2D,twoD_Gaussian,estimateBackground,create_DS9regions2)
#print(stackImages,Gaussian,Focus,ConvolveSlit2D_PSF)

#from __future__ import division
#from __future__ import print_function
#from __future__ import absolute_import
#
#from builtins import input
#from builtins import map
#from past.utils import old_div
#

def rot_matrix(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return R
def warp_matix(angle,t1,t2):
    rot_matrix(angle)
    mat = np.hstack((rot_matrix(30),np.array([[t1],[t2]])))
    return mat

def Compute_transformation(path1,path2):
    import imageio
    from astropy.io import fits
    import cv2
#path1= '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180612/image-000085-000094-Zinc-with_dark-117-stack.fits'
#path1= '/Users/Vincent/Nextcloud/Work/MaskMoves/Sub12June/Tostack/StackedImage_8309957-8313557_Inverse.fits'
#path2 = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180612/image-000075-000084-Zinc-with_dark-121-stack.fits'
#path2= '/Users/Vincent/Nextcloud/Work/MaskMoves/SubAugust/StackedImage_1000700-9974457_Inverse.fits'
    img1 = fits.open(path1)[0].data#[:,1100:2000]
    img2 = fits.open(path2)[0].data#[:,1100:2000]
    imageio.imwrite('/tmp/img1.jpg', img1)
    imageio.imwrite('/tmp/img2.jpg', img2)
    im1 =  cv2.imread('/tmp/img1.jpg');
    im2 =  cv2.imread("/tmp/img2.jpg");
    im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

    res = cv2.resize(im1,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
    warp_mode = cv2.MOTION_EUCLIDEAN#MOTION_TRANSLATION
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Specify the number of iterations.
    number_of_iterations = 5000;
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10;
    
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
 
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria)
    sz = im1.shape
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    return warp_matrix
#    warp_matrix_test = warp_matix(30,30,100)
 #   im2_aligned = cv2.warpAffine(im2, warp_matrix_test, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP); 
 #   imshow(im1);show()
 #   imshow(im2);show()
 #   imshow(im2_aligned)    


def DS9setup(xpapoint, filename=None, Internet=False, smooth=2, 
                 regions=True, centering=False, rot=0):
    '''Load an image
    '''
    from astropy.io import fits
    d = DS9()
    if filename == None:
        filename = d.get("file")
    fitsimage = fits.open(filename)
    if fitsimage[0].header['BITPIX'] == -32:
        Type = 'guider'  
    else:
        Type = 'detector'
    print (Type)   
    if Type == 'guider':
        d.set("scale limits {} {} ".format(np.percentile(fitsimage[0].data,1),
              np.percentile(fitsimage[0].data,99.4)))
        d.set("rotate 0") 
        try:
#            d.set("file {}".format(filename[:-5]+ '_wcs.fits'))   
            d.set("grid")
#            if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
            d.set("lock scalelimits no")
            d.set("dsssao")
            d.set("lock frame wcs")
        except ValueError:
            d.set("file {}".format(filename))            
            d.set("grid no")
    if Type == 'detector':
        d.set("file {}".format(filename))            
        d.set("grid no") 
        d.set("rotate %0.2f"%(rot)) 
        d.set("scale limits {} {} ".format(np.percentile(fitsimage[0].data,9),
              np.percentile(fitsimage[0].data,99.6)))
        d.set("lock frame physical")
        d.set("lock scalelimits yes") 
    if regions:
        if os.path.isfile(filename[:-5]+ '.reg'):   
            d.set("region {}".format(filename[:-5]+ '.reg'))  
        if os.path.isfile(filename[:-5]+ 'predicted.reg'):
            d.set("region {}".format(filename[:-5]+ 'predicted.reg'))
    if centering:
        d.set("region {}".format('/tmp/centers.reg'))  
    d.set("cmap Cubehelix0")
    d.set("smooth yes")
    d.set("smooth radius {}".format(2))
    d.set("smooth yes")
    #print('test')
    return fitsimage


def DS9guider(xpapoint):
    """
    """
    from astropy.io import fits
    d = DS9()
    filename = d.get("file")
    header = fits.open(filename)[0].header
    if ('WCSAXES' in header):
        print('WCS header existing, checking Image servers')
        d.set("grid")
        d.set("scale mode 99.5")#vincent
        try:# if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
            d.set("dsssao")
        except:
            pass
        d.set("lock scalelimits no")
        d.set("lock frame wcs")
    else:
        print ('Nop header WCS - Applying lost in space algorithm: Internet needed!')
        print ('Processing might take a few minutes ~5-10')
        PathExec = os.path.dirname(os.path.realpath(__file__)) + '/astrometry-net.py'
        Newfilename = filename[:-5] + '_wcs.fits'
        CreateWCS(PathExec, filename, Newfilename)
        filename = d.set("file {}".format(Newfilename))
        filename = d.get("file")
        header = fits.open(filename)[0].header
        if header['WCSAXES'] == 2:
            print('WCS header existing, checking Image servers')
            d.set("grid")
            d.set("scale mode 99.5")#vincent
            try:# if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
                d.set("dsssao")
            except:
                pass
            d.set("lock scalelimits no")
            d.set("lock frame wcs")        
    return

def CreateWCS(PathExec, filename, Newfilename):
    import subprocess
    print(filename)
    start = timeit.default_timer()
    print('''\n\n\n\n      Start lost in space algorithm - might take a few minutes \n\n\n\n''')
    subprocess.check_output("python " + PathExec + " --apikey apfqmasixxbqxngm --newfits --wait --upload " + filename,shell=True)
    try:
        os.rename(os.path.dirname(PathExec) + "/--wait", Newfilename)
    except OSError:
        os.rename(os.path.dirname(PathExec) + "/--wait.fits", Newfilename)
    stop = timeit.default_timer()
    print('File created')
    print('Lost in space duration = {} seconds'.format(stop-start))
    return

def DS9setup2(xpapoint):
    """
    """
    from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = d.get("file")
    fitsimage = fits.open(filename)
    if d.get("lock bin") == 'no':
        d.set("grid no") 
#        d.set("scale limits {} {} ".format(np.percentile(fitsimage[0].data,9),
#              np.percentile(fitsimage[0].data,99.6)))
        d.set("scale limits {} {} ".format(np.percentile(fitsimage[0].data,50),
              np.percentile(fitsimage[0].data,99.95)))
        d.set("scale asinh")
        d.set("cmap bb")
#        d.set("smooth yes")
        #sd.set("cmap Cubehelix0")
#        d.set("smooth radius {}".format(2))
#        d.set("smooth yes")
        d.set("lock bin yes")
    elif d.get("lock bin") == 'yes':
        #d.set("regions delete all")
        d.set("cmap grey")
        d.set("scale linear")
        d.set("scale mode minmax")
        d.set("grid no")
        d.set("smooth no")
        d.set("lock bin no")
    return filename



def parse_data(data, nan=np.nan, map=map, float=float):
    vals = []
    xy = []
    for s in data.split('\n'):
        coord,val = s.split('=')
        val = val.strip() or nan
        xy.append(list(map(float, coord.split(','))))
        vals.append(float(val))
    vals = np.array(vals)
    xy = np.floor(np.array(xy)).astype(int)
    x,y = xy[:,0], xy[:,1]
    w = x.ptp() + 1
    h = y.ptp() + 1
    arr = np.empty((w, h))
    X = np.empty((w, h))
    Y = np.empty((w, h))
    indices = x - x.min(), y - y.min()
    arr[indices] = vals
    X[indices] = x
    Y[indices] = y
    return X.T, Y.T, arr.T

def process_region(region, win):
    from collections import namedtuple
    name, info = region.split('(')
    coords = [float(c) for c in info.split(')')[0].split(',')]
    print(coords)
    if name == 'box':
        xc,yc,w,h,angle = coords
        dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
        X,Y,arr = parse_data(dat)
        box = namedtuple('Box', 'data xc yc w h angle')
        return box(arr, xc, yc, w, h, angle)
    elif name == 'bpanda':
        xc, yc, a1, a2, a3, a4,a5, w, h,a6,a7 = coords
        dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
        X,Y,arr = parse_data(dat)
        box = namedtuple('Box', 'data xc yc w h angle')
        return box(arr, xc, yc, w, h, 0)
    elif name == 'circle':
        xc,yc,r = coords
        dat = win.get("data physical %s %s %s %s no" % (xc - r, yc - r, 2*r, 2*r))
        X,Y,arr = parse_data(dat)
        Xc,Yc = np.floor(xc), np.floor(yc)
        inside = (X - Xc)**2 + (Y - Yc)**2 <= r**2
        circle = namedtuple('Circle', 'data databox inside xc yc r')
        return circle(arr[inside], arr, inside, xc, yc, r)
#    elif name == 'annulus':
#        xc,yc,r1,r2 = coords
#        w = 2*r2
#        h = 2*r2
#        dat = win.get("data physical %s %s %s %s no" % (xc-r2,yc-r2,w,h))
#        X,Y,arr = parse_data(dat)
#        Xc,Yc = np.floor(xc), np.floor(yc)
#        inside = between((X - Xc)**2 + (Y - Yc)**2, r1**2, r2**2)
#        annulus = namedtuple('Annulus', 'data databox inside xc yc r1 r2')
#        return annulus(arr[inside], arr, inside, xc, yc, r1, r2)
    elif name == 'ellipse':
        if len(coords) == 5:
            xc, yc, a2, b2, angle = coords
        else:
            xc, yc, a1, b1, a2, b2, angle = coords
        w = 2*a2
        h = 2*b2
        dat = win.get("data physical %s %s %s %s no" % (xc - a2,yc - b2,w,h))
        X,Y,arr = parse_data(dat)
        Xc,Yc = np.floor(xc), np.floor(yc)
        inside = ((X - Xc)/a2)**2 + ((Y - Yc)/b2)**2 <= 1
        if len(coords) == 5:
            ellipse = namedtuple('Ellipse',
                                 'data databox inside xc yc a b angle')
            return ellipse(arr[inside], arr, inside, xc, yc, a2, b2, angle)

        inside &= ((X - Xc)/a1)**2 + ((Y - Yc)/b1)**2 >= 1
        annulus = namedtuple('EllipticalAnnulus',
                             'data databox inside xc yc a1 b1 a2 b2 angle')
        return annulus(arr[inside], arr, inside, xc, yc, a1, b1, a2, b2, angle)
    else:
        raise ValueError("Can't process region %s" % name)

def getregion(win, debug=False):
    """ Read a region from a ds9 instance.

    Returns a tuple with the data in the region.
    """
    win.set("regions format ds9")
    win.set("regions system physical")
    #rows = win.get("regions list")
    rows = win.get("regions all")
    rows = [row for row in rows.split('\n') if row]
    if len(rows) < 3:
        print( "No regions found")
        sys.exit()
    #units = rows[2]
    #assert units == 'physical'
    if debug:
        print (rows[4])
        if rows[5:]:
            print('discarding %i regions' % len(rows[5:]) )
    return process_region(rows[-1], win)


def create_PA(A=15.45,B=13.75,C=14.95,pas=0.15,nombre=11):
    a = np.linspace(A-int(nombre/2)*pas, A+int(nombre/2)*pas, nombre)
    b = np.linspace(B-int(nombre/2)*pas, B+int(nombre/2)*pas, nombre)
    c = np.linspace(C-int(nombre/2)*pas, C+int(nombre/2)*pas, nombre)
    return a[::-1],b[::-1],c[::-1]
#ENCa, ENCb, ENCc = create_PA()

def ENC(x,ENCa):
    a = (ENCa[-1]-ENCa[0])/(len(ENCa)-1) * x + ENCa[0]
    #b = (ENCb[10]-ENCb[0])/(10) * x + ENCb[0]
    #c = (ENCc[10]-ENCc[0])/(10) * x + ENCc[0]
    return a#, b, c

def throughfocus(center, files,x=None, 
                 fibersize=0, center_type='barycentre', SigmaMax= 4,Plot=True, Type=None, ENCa_center=None, pas=None, WCS=False):
    """
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import matplotlib.pyplot as plt
    from focustest import AnalyzeSpot
    from focustest import estimateBackground
    from scipy.optimize import curve_fit
    fwhm = []
    EE50 = []
    EE80 = [] 
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    sec=[]
    images=[]
    ENCa = []
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[0]
            image = fitsfile.data
        time = fitsfile.header['DATE']
        if Type == 'guider':    
            ENCa.append(fitsfile.header['LINAENC'])
        else:
            nombre = 5
            if ENCa_center is not None:
                print('Actuator given: Center = {} , PAS = {}'.format(ENCa_center,pas))
                ENCa = np.linspace(ENCa_center-nombre*pas, ENCa_center+nombre*pas, 2*nombre+1)[::-1]
#            else:
#                ENCa = np.linspace(100,100, 2*nombre+1)[::-1]
        day,h, m, s = float(time[-11:-9]),float(time[-8:-6]), float(time[-5:-3]), float(time[-2:])
        sec.append(t2s(h=h,m=m,s=s,d=day))

        background = 1*estimateBackground(image,center)
        n = 25
        subimage = (image-background)[int(center[1]) - n:int(center[1]) + n, int(center[0]) - n:int(center[0]) + n]
        images.append(subimage)
        d = AnalyzeSpot(image,center=center,fibersize=fibersize,
                        center_type=center_type, SigmaMax = SigmaMax)
        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d['Sigma'])
        EE50.append(d['EE50'])
        EE80.append(d['EE80'])
        xo.append(d['Center'][0])
        yo.append(d['Center'][1])        
        maxpix.append(max20[-20:].mean())
        sumpix.append(d['Flux'])
        varpix.append(subimage.var())
    f = lambda x,a,b,c: a * (x-b)**2 + c#a * np.square(x) + b * x + c
    x = np.arange(len(files))
    fig, axes = plt.subplots(4, 2, figsize=(10,6),sharex=True)
    xtot = np.linspace(x.min(),x.max(),200)
#    if ENCa_center is not None:
#        axes2 = axes[0,0].twinx()
#        X2tick_location= axes[0,0].xaxis.get_ticklocs() #Get the tick locations in data coordinates as a numpy array
#        axes2.set_xticks(X2tick_location)
#        axes2.set_xticklabels(ENCa)
    print(ENCa)
    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted')
        bestx1 = xtot[np.argmin(f(xtot,*opt1))]
        axes[0,0].plot(np.ones(2)*bestx1,[min(fwhm),max(fwhm)])
        if len(ENCa) > 0:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx1,ENC(bestx1,ENCa)))
        else:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx1))
            
    except RuntimeError:
        opt1 = [0,0,0]
        bestx1 = np.nan
        pass 
    try:
        opt2,cov2 = curve_fit(f,x,EE50)
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted')
        bestx2 = xtot[np.argmin(f(xtot,*opt2))]
        if len(ENCa) > 0:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx2,ENC(bestx2,ENCa)))
        else:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx2))
        axes[1,0].plot(np.ones(2)*bestx2,[min(EE50),max(EE50)])
    except RuntimeError:
        opt2 = [0,0,0]
        bestx2 = np.nan
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[2,0].plot(xtot,f(xtot,*opt3),linestyle='dotted')
        bestx3 = xtot[np.argmin(f(xtot,*opt3))]
        if len(ENCa) > 0:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx3,ENC(bestx3,ENCa)))
        else:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx3))
        axes[2,0].plot(np.ones(2)*bestx3,[min(EE80),max(EE80)])
    except RuntimeError:
        opt3 = [0,0,0]
        bestx3 = np.nan
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted')
        bestx4 = xtot[np.argmax(f(xtot,*opt4))]
        axes[0,1].plot(np.ones(2)*bestx4,[min(maxpix),max(maxpix)])
        if len(ENCa) > 0:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx4,ENC(bestx4,ENCa)))
        else:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx4))
    except RuntimeError:
        opt4 = [0,0,0]
        bestx4 = np.nan
        pass            
    try:
        opt5,cov5 = curve_fit(f,x,sumpix)
        axes[1,1].plot(xtot,f(xtot,*opt5),linestyle='dotted')
        bestx5 = xtot[np.argmax(f(xtot,*opt5))]
        if len(ENCa) > 0:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx5,ENC(bestx5,ENCa)))
        else:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx5))
        axes[1,1].plot(np.ones(2)*bestx5,[min(sumpix),max(sumpix)])
    except RuntimeError:
        opt5 = [0,0,0]
        bestx5 = np.nan
        pass
    try:
        opt6,cov6 = curve_fit(f,x,varpix)
        axes[2,1].plot(xtot,f(xtot,*opt6),linestyle='dotted')
        bestx6 = xtot[np.argmax(f(xtot,*opt6))]
        if len(ENCa) > 0:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx6,ENC(bestx6,ENCa)))
        else:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx6))
        axes[2,1].plot(np.ones(2)*bestx6,[min(varpix),max(varpix)])
    except RuntimeError:
        opt6 = [0,0,0]
        bestx6 = np.nan
        pass
    axes[0,0].plot(x,fwhm, '-o')
    axes[0,0].set_ylabel('Sigma')

    axes[1,0].plot(x,EE50, '-o')
    axes[1,0].set_ylabel('EE50')

    axes[2,0].plot(x,EE80, '-o')
    axes[2,0].set_ylabel('EE80')

    axes[3,0].plot(x,xo, '-o')
    axes[3,0].set_ylabel('y center')
    axes[3,0].grid()

    axes[0,1].plot(x,maxpix, '-o')
    axes[0,1].set_ylabel('Max pix')

    axes[1,1].plot(x,sumpix, '-o')
    axes[1,1].set_ylabel('Flux')
    
    axes[2,1].plot(x,varpix, '-o')
    axes[2,1].set_ylabel('Var pix (d=50)')
    axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
    axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
    #axes[3,1].grid()
    axes[3,1].set_ylabel('y center')
   

    name = '{} - {} - {}'.format(os.path.basename(filename),[int(center[0]),int(center[1])],fitsfile.header['DATE'])
    fig.tight_layout()
    fig.suptitle(name, y=1.01)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
    plt.show()
    print(name) 
    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix','Best sigma','Best EE50','Best EE80','Best Maxpix','Best Varpix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    t.add_row((os.path.basename(filename),float(os.path.basename(filename)[-11:-4]),
               t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],min(fwhm),
               min(EE50),min(EE80),min(maxpix),min(sumpix),max(varpix),
               ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)))
    try:
        print(os.path.dirname(filename) + '/Throughfocus.csv')
        OldTable = Table.read(os.path.dirname(filename) + '/Throughfocus.csv')
        print ('old',OldTable)
    except IOError:
        t.write(os.path.dirname(filename) + '/Throughfocus.csv')
    else:
        t = vstack((OldTable,t))
        #print ('new',newTable)
        t.write(os.path.dirname(filename) + '/Throughfocus.csv',overwrite=True)
    if Plot:        
        fig, axes = plt.subplots(1, 11,figsize=(24,3),sharey=True)
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('equal')
            subname = os.path.basename(files[i])[6:-5] 
            try:
                axes[i].set_xlabel('%i - %0.2f'%(int(subname),float(ENCa[i])))
                #axes[i].set_title(float(ENCa[i]))
            except:
                axes[i].set_xlabel(int(subname))
                pass
        fig.suptitle(name)
        fig.subplots_adjust(top=0.88)   
        fig.tight_layout()
        #plt.axis('equal')
        fig.savefig(os.path.dirname(filename) + '/ThroughfocusImage-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
        #fig.show()
    return fwhm, EE50, EE80
#throughfocus(center = [F.table['X_IMAGE'][i],F.table['Y_IMAGE'][i]], 
#             files=path,x = x,fibersize=0,
#             center_type=None,SigmaMax=6,Plot=True)
def throughfocusWCS(center, files,x=None, 
                 fibersize=0, center_type='barycentre', SigmaMax= 4,Plot=True, Type=None, ENCa_center=None, pas=None, WCS=False):
    """
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import matplotlib.pyplot as plt
    from focustest import AnalyzeSpot
    from focustest import estimateBackground
    from scipy.optimize import curve_fit
    fwhm = []
    EE50 = []
    EE80 = [] 
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    sec=[]
    images=[]
    ENCa = []
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[0]
            image = fitsfile.data
        header = fitsfile.header
        time = header['DATE']
        if Type == 'guider':    
            ENCa.append(header['LINAENC'])
        else:
            nombre = 5
            if ENCa_center is not None:
                print('Actuator given: Center = {} , PAS = {}'.format(ENCa_center,pas))
                ENCa = np.linspace(ENCa_center-nombre*pas, ENCa_center+nombre*pas, 2*nombre+1)[::-1]
#            else:
#                ENCa = np.linspace(100,100, 2*nombre+1)[::-1]
        day,h, m, s = float(time[-11:-9]),float(time[-8:-6]), float(time[-5:-3]), float(time[-2:])
        sec.append(t2s(h=h,m=m,s=s,d=day))
        if WCS:
            from astropy import units as u
            from astropy import wcs

            print ('Using WCS')
            w = wcs.WCS(header)
            center_wcs = center
            center_pix = w.all_world2pix(center_wcs[0]*u.deg, center_wcs[1]*u.deg,0, )
            center_pix = [int(center_pix[0]), int(center_pix[1])]
            print('CENTER PIX= ' , center_pix)
        else:
            center_pix = center
        d = AnalyzeSpot(image,center=center_pix,fibersize=fibersize,
                        center_type=center_type, SigmaMax = SigmaMax)
#        else:
#            d = AnalyzeSpot(image,center=center,fibersize=fibersize,
#                            center_type=center_type, SigmaMax = SigmaMax)
        background = 1*estimateBackground(image,center)
        n = 25
        subimage = (image-background)[int(center_pix[1]) - n:int(center_pix[1]) + n, int(center_pix[0]) - n:int(center_pix[0]) + n]
        images.append(subimage)

        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d['Sigma'])
        EE50.append(d['EE50'])
        EE80.append(d['EE80'])
        xo.append(d['Center'][0])
        yo.append(d['Center'][1])        
        maxpix.append(max20[-20:].mean())
        sumpix.append(d['Flux'])
        varpix.append(subimage.var())
    f = lambda x,a,b,c: a * (x-b)**2 + c     #a * np.square(x) + b * x + c
    if Type == 'guider':
        x = np.array(ENCa)
        xtot = np.linspace(x.min(),x.max(),200)
        ENC = lambda x,a : x
    if Type == 'detector':
        x = np.arange(len(files))
        xtot = np.linspace(x.min(),x.max(),200)
        if len(ENCa)==0:
            ENC = lambda x,a : 0
        else :
            ENC = lambda x,a : (ENCa[-1]-ENCa[0])/(len(ENCa)-1) * x + ENCa[0]

   
    #x = np.array(ENCa)
    fig, axes = plt.subplots(4, 2, figsize=(10,6),sharex=True)
    
#    if ENCa_center is not None:
#        axes2 = axes[0,0].twinx()
#        X2tick_location= axes[0,0].xaxis.get_ticklocs() #Get the tick locations in data coordinates as a numpy array
#        axes2.set_xticks(X2tick_location)
#        axes2.set_xticklabels(ENCa)
    #print(ENCa)
    #print(ENC(,ENCa))
    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted')
        bestx1 = xtot[np.argmin(f(xtot,*opt1))]
        axes[0,0].plot(np.ones(2)*bestx1,[min(fwhm),max(fwhm)])
        if len(ENCa) > 0:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx1,ENC(bestx1,ENCa)))
        else:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx1))
            
    except RuntimeError:
        opt1 = [0,0,0]
        bestx1 = np.nan
        pass 
    try:
        opt2,cov2 = curve_fit(f,x,EE50)
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted')
        bestx2 = xtot[np.argmin(f(xtot,*opt2))]
        if len(ENCa) > 0:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx2,ENC(bestx2,ENCa)))
        else:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx2))
        axes[1,0].plot(np.ones(2)*bestx2,[min(EE50),max(EE50)])
    except RuntimeError:
        opt2 = [0,0,0]
        bestx2 = np.nan
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[2,0].plot(xtot,f(xtot,*opt3),linestyle='dotted')
        bestx3 = xtot[np.argmin(f(xtot,*opt3))]
        if len(ENCa) > 0:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx3,ENC(bestx3,ENCa)))
        else:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx3))
        axes[2,0].plot(np.ones(2)*bestx3,[min(EE80),max(EE80)])
    except RuntimeError:
        opt3 = [0,0,0]
        bestx3 = np.nan
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted')
        bestx4 = xtot[np.argmax(f(xtot,*opt4))]
        axes[0,1].plot(np.ones(2)*bestx4,[min(maxpix),max(maxpix)])
        if len(ENCa) > 0:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx4,ENC(bestx4,ENCa)))
        else:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx4))
    except RuntimeError:
        opt4 = [0,0,0]
        bestx4 = np.nan
        pass            
    try:
        opt5,cov5 = curve_fit(f,x,sumpix)
        axes[1,1].plot(xtot,f(xtot,*opt5),linestyle='dotted')
        bestx5 = xtot[np.argmax(f(xtot,*opt5))]
        if len(ENCa) > 0:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx5,ENC(bestx5,ENCa)))
        else:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx5))
        axes[1,1].plot(np.ones(2)*bestx5,[min(sumpix),max(sumpix)])
    except RuntimeError:
        opt5 = [0,0,0]
        bestx5 = np.nan
        pass
    try:
        opt6,cov6 = curve_fit(f,x,varpix)
        axes[2,1].plot(xtot,f(xtot,*opt6),linestyle='dotted')
        bestx6 = xtot[np.argmax(f(xtot,*opt6))]
        if len(ENCa) > 0:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx6,ENC(bestx6,ENCa)))
        else:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx6))
        axes[2,1].plot(np.ones(2)*bestx6,[min(varpix),max(varpix)])
    except RuntimeError:
        opt6 = [0,0,0]
        bestx6 = np.nan
        pass
    axes[0,0].plot(x,fwhm, '-o')
    axes[0,0].set_ylabel('Sigma')

    axes[1,0].plot(x,EE50, '-o')
    axes[1,0].set_ylabel('EE50')

    axes[2,0].plot(x,EE80, '-o')
    axes[2,0].set_ylabel('EE80')

    axes[3,0].plot(x,xo, '-o')
    axes[3,0].set_ylabel('y center')
    axes[3,0].grid()

    axes[0,1].plot(x,maxpix, '-o')
    axes[0,1].set_ylabel('Max pix')

    axes[1,1].plot(x,sumpix, '-o')
    axes[1,1].set_ylabel('Flux')
    
    axes[2,1].plot(x,varpix, '-o')
    axes[2,1].set_ylabel('Var pix (d=50)')
    axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
    axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
    #axes[3,1].grid()
    axes[3,1].set_ylabel('y center')
   

    fig.tight_layout()

    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix','Best sigma','Best EE50','Best EE80','Best Maxpix','Best Varpix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    t.add_row((os.path.basename(filename),os.path.basename(filename)[5:11],
               t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],min(fwhm),
               min(EE50),min(EE80),min(maxpix),min(sumpix),max(varpix),
               ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)))
    mean = np.nanmean(np.array([ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)]))
    print(mean)
    name = '%s - %i - %i - %s - %0.3f'%(os.path.basename(filename),int(center_pix[0]),int(center_pix[1]),fitsfile.header['DATE'],mean)
    #name = '%s - %i - %i'%(os.path.basename(filename),int(center_pix[0]),int(center_pix[1]),fitsfile.header['DATE'],mean)
    print(name) 
    fig.suptitle(name, y=0.99)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center_pix[0]) ,int(center_pix[1]), fitsfile.header['DATE']))
    plt.show()
    try:
        print(os.path.dirname(filename) + '/Throughfocus.csv')
        OldTable = Table.read(os.path.dirname(filename) + '/Throughfocus.csv')
        print ('old',OldTable)
    except IOError:
        t.write(os.path.dirname(filename) + '/Throughfocus.csv')
    else:
        t = vstack((OldTable,t))
        #print ('new',newTable)
        t.write(os.path.dirname(filename) + '/Throughfocus.csv',overwrite=True)
    if Plot:        
        fig, axes = plt.subplots(1, 11,figsize=(24,3),sharey=True)
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('equal')
            subname = os.path.basename(files[i])[6:11] 
            try:
                axes[i].set_xlabel('%i - %0.2f'%(int(subname),float(ENCa[i])))
                #axes[i].set_title(float(ENCa[i]))
            except:
                axes[i].set_xlabel(int(subname))
                pass
        fig.suptitle(name)
        fig.subplots_adjust(top=0.88)   
        fig.tight_layout()
        #plt.axis('equal')
        fig.savefig(os.path.dirname(filename) + '/ThroughfocusImage-{}-{}-{}.png'.format( int(center_pix[0]) ,int(center_pix[1]), fitsfile.header['DATE']))
        #fig.show()
    return fwhm, EE50, EE80

def throughfocus2(center, files,x=np.linspace(11.95,14.45,11)[::-1][3:8], 
                 fibersize=100, center_type='barycentre', SigmaMax= 4, box=25):
    """
    """
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from focustest import AnalyzeSpot
    from focustest import estimateBackground
    from scipy.optimize import curve_fit
    from astropy.table import Table, vstack
    fwhm = []
    EE50 = []
    EE80 = [] 
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    sec = []
    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    n = box
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[0]
            image = fitsfile.data
        time = fitsfile.header['DATE']
        day,h, m, s = float(time[-11:-9]),float(time[-8:-6]), float(time[-5:-3]), float(time[-2:])
        sec.append(t2s(h=h,m=m,s=s,d=day))
        #image = fitsfile.data
        background = 1*estimateBackground(image,center)
        subimage = (image-background)[int(center[1]) - n:int(center[1]) + n, int(center[0]) - n:int(center[0]) + n]
        d = AnalyzeSpot(image,center=center,fibersize=fibersize,
                        center_type=center_type, SigmaMax = SigmaMax)
        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d['Sigma'])
        EE50.append(d['EE50'])
        EE80.append(d['EE80'])
        xo.append(d['Center'][0])
        yo.append(d['Center'][1])        
        maxpix.append(max20[-20:].mean())
        sumpix.append(d['Flux'])
        varpix.append(subimage.var())
        t.add_row((os.path.basename(filename),float(os.path.basename(filename)[-11:-4]), t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],d['Sigma'],d['EE50'],d['EE80'],max20[-20:].mean(),subimage.sum(),subimage.var() ))
    try:
        print(os.path.dirname(filename) + '/Analysis.csv')
        OldTable = Table.read(os.path.dirname(filename) + '/Analysis.csv')
        print ('old',OldTable)
    except IOError:
        t.write(os.path.dirname(filename) + '/Analysis.csv')
    else:
        t = vstack((OldTable,t))
        #print ('new',newTable)
        t.write(os.path.dirname(filename) + '/Analysis.csv',overwrite=True)
    sec = np.array(sec)
    sec = (sec - sec.min())/60
    xtot = np.linspace(sec.min(),sec.max(),200)
    x = sec
    f = lambda x,a,b,c: a * (x-b)**2 + c#a * np.square(x) + b * x + c

    fig, axes = plt.subplots(4, 2, figsize=(10,6))


    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted')
        axes[0,0].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt1))],[min(fwhm),max(fwhm)])
        axes[0,0].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt1))]))
    except RuntimeError:
        pass 
    try:
        opt2,cov2 = curve_fit(f,x,EE50)
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted')
        axes[1,0].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt2))]))
        axes[1,0].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt2))],[min(EE50),max(EE50)])
    except RuntimeError:
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[2,0].plot(xtot,f(xtot,*opt3),linestyle='dotted')
        axes[2,0].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt3))]))
        axes[2,0].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt3))],[min(EE80),max(EE80)])
    except RuntimeError:
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted')
        axes[0,1].plot(np.ones(2)*xtot[np.argmax(f(xtot,*opt4))],[min(maxpix),max(maxpix)])
        axes[0,1].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmax(f(xtot,*opt4))]))
    except RuntimeError:
        pass            
    try:
        opt5,cov5 = curve_fit(f,x,sumpix)
        axes[1,1].plot(xtot,f(xtot,*opt5),linestyle='dotted')
        axes[1,1].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt5))]))
        axes[1,1].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt5))],[min(sumpix),max(sumpix)])
    except RuntimeError:
        pass
    try:
        opt6,cov6 = curve_fit(f,x,varpix)
        axes[2,1].plot(xtot,f(xtot,*opt6),linestyle='dotted')
        axes[2,1].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmax(f(xtot,*opt6))]))
        axes[2,1].plot(np.ones(2)*xtot[np.argmax(f(xtot,*opt6))],[min(varpix),max(varpix)])
    except RuntimeError:
        pass
    axes[0,0].plot(x,fwhm, '-o')
    axes[0,0].set_ylabel('Sigma')

    axes[1,0].plot(x,EE50, '-o')
    axes[1,0].set_ylabel('EE50')

    axes[2,0].plot(x,EE80, '-o')
    axes[2,0].set_ylabel('EE80')

    axes[3,0].plot(x,xo, '-o')
    axes[3,0].set_ylabel('y center')
    axes[3,0].grid()

    axes[0,1].plot(x,maxpix, '-o')
    axes[0,1].set_ylabel('Max pix')

    axes[1,1].plot(x,sumpix, '-o')
    axes[1,1].set_ylabel('Flux')
    
    axes[2,1].plot(x,varpix, '-o')
    axes[2,1].set_ylabel('Var pix (d=50)')
    axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
    axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
    axes[3,1].grid()
    axes[3,1].set_ylabel('y center')
   

    name = '{} - {} - {}'.format(os.path.basename(filename),[int(center[0]),int(center[1])],fitsfile.header['DATE'])
    fig.tight_layout()
    fig.suptitle(name, y=1.)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
    plt.show()
    print(name) 
    return t#fwhm, EE50, EE80



def DS9throughfocus(xpapoint):
    """
    """
    from astropy.io import fits
    from focustest import AnalyzeSpot

    print('''\n\n\n\n      START THROUGHFOCUS \n\n\n\n''')
    d = DS9(xpapoint)
    filename = d.get("file ")
    path = Charge_path(xpapoint)
    try:
        ENCa_center, pas = sys.argv[4].split('-')
        ENCa_center, pas = float(ENCa_center), float(pas)
    except ValueError:
        print('No actuator given, taking header ones for guider images, none for detector images')
        ENCa_center, pas = None, None
    except IndexError :
        print('No actuator given, taking header ones for guider images, none for detector images')
        ENCa_center, pas = None, None        
    x = np.arange(len(path))
    
    a = getregion(d)
    image = fits.open(filename)[0]
    rp = AnalyzeSpot(image.data,center = [np.int(a.xc),
                     np.int(a.yc)],fibersize=0)
    x,y = rp['Center']    
    #d.set('regions delete all')#testvincent
    d.set('regions system image')
    #d.set('regions command "circle %0.3f %0.3f %0.3f # color=red"' % (x,y,10))#testvincent
    print('\n\n\n\n     Centring on barycentre of the DS9 image '
          '(need to be close to best focus) : %0.1f, %0.1f'
          '--> %0.1f, %0.1f \n\n\n\n' % (a.xc,a.yc,rp['Center'][0],rp['Center'][1]))
    print('Applying throughfocus')
    if image.header['BITPIX'] == -32:
        Type = 'guider'
    else:
        Type = 'detector'        
    if d.get('wcs lock frame') == 'wcs':
        from astropy import wcs
        print ('Using WCS')
        w = wcs.WCS(image.header)
        center_wcs = w.all_pix2world(x, y,0)
                #d.set('crosshair {} {} physical'.format(x,y))
        alpha, delta = float(center_wcs[0]), float(center_wcs[1])
        print('alpha, delta = ',alpha, delta)
        
        #alpha, delta = float(alpha), float(delta)
        throughfocusWCS(center = [alpha,delta], files=path,x = x,fibersize=0,
                     center_type=None,SigmaMax=6, Plot=True, Type=Type,ENCa_center=ENCa_center, pas=pas,WCS=True)
        
    else:
        throughfocus(center = rp['Center'], files=path,x = x,fibersize=0,
                     center_type=None,SigmaMax=6, Plot=True, Type=Type,ENCa_center=ENCa_center, pas=pas)

    return 



def back(xpapoint):#,filename = None,Internet =False, smooth=2, regions=True, centering=False,rot=0):
    d = DS9(xpapoint)#DS9(xpapoint)
    d.set("regions delete all")
    d.set("cmap grey")
    d.set("scale linear")
    d.set("scale mode minmax")
    d.set("grid no")
    d.set("smooth no")
    return



def DS9rp(xpapoint):#,filename = None,Internet =False, smooth=2, regions=True, centering=False,rot=0):
    """
    """
    from astropy.io import fits
    import matplotlib.pyplot as plt
    d = DS9(xpapoint)#DS9(xpapoint)
    try:
        fibersize = sys.argv[3]
    except IndexError:
        print('No fibersize, Using point source object')
        fibersize = 0
    if fibersize == '':
        fibersize = 0
    filename = d.get("file ")
    a = getregion(d)
    fitsfile = fits.open(filename)[0]
    spot = DS9plot_rp_convolved(data=fitsfile.data,
                                center = [np.int(a.xc),np.int(a.yc)],
                                fibersize=fibersize)    
    plt.title('{} - {} - {}'.format(os.path.basename(filename),[np.int(a.xc),np.int(a.yc)],fitsfile.header['DATE']))
    #plt.savefig(os.path.basename(filename),[np))
    plt.show()
    #d.set('regions delete select')
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=red"' % (spot['Center'][0]+1,spot['Center'][1]+1,10))#testvincent
    return


def DS9plot_rp_convolved(data, center, size=40, n=1.5, anisotrope=False, angle=30, radius=40, ptype='linear', fit=True, center_type='barycentre', maxplot=0.013, minplot=-1e-5, radius_ext=12, platescale=None,fibersize = 100,SigmaMax=4):
  """Function used to plot the radial profile and the encircled energy of a spot,
  Latex is not necessary
  """
  from focustest import  radial_profile_normalized
  import matplotlib.pyplot as plt
  from focustest import ConvolveDiskGaus2D
  from focustest import gausexp
  from scipy.optimize import curve_fit
  from scipy import interpolate
  if anisotrope == True:
      spectral, spatial, EE_spectral, EE_spatial = radial_profile_normalized(data, center, anisotrope=anisotrope, angle=angle, radius=radius, n=n, center_type=center_type)
      spectral = spectral[~np.isnan(spectral)]
      spatial = spatial[~np.isnan(spatial)]
      #min1 = min(spatial[:size])
      #min2 = min(spectral[:size])
      norm_spatial = spatial[:size]#(spatial[:n] - min(min1,min2)) / np.sum((spatial[:n] - min(min1,min2) ))
      norm_spectral = spectral[:size]#(spectral[:n] - min(min1,min2)) / np.sum((spectral[:n] - min(min1,min2) ))              
      if ptype == 'linear':
          popt1, pcov1 = curve_fit(gausexp, np.arange(size), norm_spatial)       
          popt2, pcov2 = curve_fit(gausexp, np.arange(size), norm_spectral) 
          plt.plot(np.arange(size), norm_spectral, label='spectral direction')   
          plt.plot(np.arange(size), norm_spatial, label='spatial direction') 
          if fit==True:
              plt.plot(np.linspace(0,size,10*size), gausexp(np.linspace(0,size,10*size), *popt1), label='Spatial fit')            
              plt.plot(np.linspace(0,size,10*size), gausexp(np.linspace(0,size,10*size), *popt2), label='Spectral fit')                       
              plt.figtext(0.5,0.5,'Sigma = %0.3f-%0.3f pix \nLambda = %0.3f-%0.3f pix \npcGaus = %0.0f-%0.3fpc' % (popt1[2],popt2[2],popt1[3],popt2[3],100*popt1[0]/(popt1[1] + popt1[0]),100*popt2[0]/(popt2[1] + popt2[0])), fontsize=11,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
      else:
          plt.semilogy(np.arange(size),norm_spectral, label='spectral direction')   
          plt.semilogy(np.arange(size),norm_spatial, label='spatial direction')           
      return popt1, popt2
  else: 
      rsurf, rmean, profile, EE, NewCenter = radial_profile_normalized(data, center, anisotrope=anisotrope, angle=angle, radius=radius, n=n, center_type=center_type)
      profile = profile[:size]#(a[:n] - min(a[:n]) ) / np.sum((a[:n] - min(a[:n]) ))
      fig, ax1 = plt.subplots(figsize=(8, 4))
          #popt, pcov = curve_fit(ConvolveDiskGaus2D, np.linspace(0,size,size), profile, p0=[2,2,2])#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):  3.85  
      fiber = float(fibersize) #/ (2*1.08*(1/0.083))
      if fiber == 0:
          gaus = lambda x, a, sigma: a**2 * np.exp(-np.square(x / sigma) / 2)
          popt, pcov = curve_fit(gaus, rmean[:size], profile, p0=[1, 2])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
          ax1.plot(np.linspace(0,size,10*size), gaus(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
          ax1.fill_between(rmean[:size], profile - 1.5*np.abs(profile - gaus(rmean[:size], *popt)), 
                           profile + 1.5*np.abs(profile - gaus(rmean[:size], *popt)), alpha=0.3, label=r"3*Residuals")
          
      else:
          popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[1,fiber,2, np.mean(profile)],bounds=([0,0.95*fiber-1e-5,1,-1],[2,1.05*fiber+1e-5,SigmaMax,1]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
          ax1.plot(np.linspace(0,size,10*size), ConvolveDiskGaus2D(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
          ax1.fill_between(rmean[:size], profile - 1.5*np.abs(profile - ConvolveDiskGaus2D(rmean[:size], *popt)), 
                           profile + 1.5*np.abs(profile - ConvolveDiskGaus2D(rmean[:size], *popt)), alpha=0.3, label=r"3*Residuals")
      ax1.set_xlabel('Distance to center [pix]', fontsize=12)                      
      #ax1.plot(rmean[:size], profile, '+', c='black', label='Normalized isotropic profile')
      ax1.plot(rmean[:size], profile, '+', c='black', label='Normalized isotropic profile')
      #ax1.plot(np.linspace(0, size, size), profile, '+', c='black', label='Normalized isotropic profile')
      #ax1.plot(np.linspace(0, size, 10*size), exp(np.linspace(0, size, 10*size), *popt), c='navy')
      #ax1.plot(np.linspace(0, size, 10*size), gaus(np.linspace(0, size, 10*size), *popt), c='blue')
      ax1.set_ylabel('Radial Profile', color='b', fontsize=12)
      ax1.tick_params('y', colors='b')
      ax1.set_ylim((minplot, np.max([np.max(1.1*(profile)), maxplot])))
      ax2 = ax1.twinx()
      #ax2.plot(np.linspace(0, size, len(norm)), EE[:len(norm)], 'r--x')
      EE_interp = interpolate.interp1d(rsurf[:size], EE[:size],kind='cubic')
      ninterp = 10
      xnew = np.linspace(rsurf[:size].min(),rsurf[:size].max(),ninterp*len(rsurf[:size]))
      ax2.plot(xnew,EE_interp(xnew),linestyle='dotted',c='r')
#                  x = np.linspace(0,size,100*size)
#                  aire = np.pi * np.square(np.linspace(0,size,100*size))
#                  rp2 = ConvolveDiskGaus2D(np.linspace(0, size, 100*size), *popt) - ConvolveDiskGaus2D(np.linspace(0, size, 100*size), *popt).min()
 #                 ee = np.cumsum(aire*rp)
#                  ax2.plot(x,100*(ee/ee.max()),linestyle='dotted')
      
      ax2.plot(rsurf[:size], EE[:size], 'rx')
#                    print(np.linspace(0,size,len(norm)))

      mina = min(xnew[EE_interp(xnew)[:ninterp*size]>79])
      minb = min(xnew[EE_interp(xnew)[:ninterp*size]>49])

#                    print(mina)
      ax2.plot(np.linspace(minb, minb, 2), np.linspace(0, 50, 2), 'r-o')                    
      ax2.plot(np.linspace(minb, size, 2), np.linspace(50, 50, 2), 'r-o')
      ax2.plot(np.linspace(mina, mina, 2), np.linspace(0, 80, 2), 'r-o')                    
      ax2.plot(np.linspace(mina, size, 2), np.linspace(80, 80, 2), 'r-o')
      #EE_gaus = np.cumsum(gaus(np.linspace(0,size,100*size),*popt) *2 * np.pi * np.linspace(0,size,100*size)**1)
      #EE_exp = np.cumsum(exp(np.linspace(0,size,100*size),*popt) * 2 * np.pi * np.linspace(0,size,100*size)**1)
      ax2.set_ylim((0, 110))
      ax2.set_ylabel('Encircled Energy', color='r', fontsize=12)
      ax2.tick_params('y', colors='r')
      fig.tight_layout()
      ax1.xaxis.grid(True)
      ax1.tick_params(axis='x', labelsize=12)
      ax1.tick_params(axis='y', labelsize=12)
      ax2.tick_params(axis='y', labelsize=12)                    
#                  e_gaus = np.sum(gaus(np.linspace(0,size,100*size),*popt) *2 * np.pi * np.linspace(0,size,100*size)**1)
#                  e_exp = np.sum(exp(np.linspace(0,size,100*size),*popt) * 2 * np.pi * np.linspace(0,size,100*size)**1)
      ax1.legend(loc = (0.54,0.05),fontsize=12)
      if fiber == 0:
          flux = 2*np.pi*np.square(popt[1])*np.square(popt[0])
          plt.figtext(0.53,0.53,"Flux = %0.0f\nRadius = %0.3f pix \nSigmaPSF = %0.3f pix \nEE50-80 = %0.2f - %0.2f p" % (flux,0,abs(popt[1]),minb,mina), 
                      fontsize=14,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})#    norm_gaus = np.pi*sigma    norm_exp = 2*np.pi * lam**2 * gamma(2/alpha)/alpha
          d = {"Flux":flux,"SizeSource":0,"Sigma":popt[1],"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
          print("Flux = {}\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(flux,0,popt[1],minb,mina,platescale,NewCenter))
      else:
          plt.figtext(0.53,0.53,"Amp = %0.3f\nRadius = %0.3f pix \nSigmaPSF = %0.3f pix \nEE50-80 = %0.2f - %0.2f p" % (popt[0],popt[1],popt[2],minb,mina), 
                      fontsize=14,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})#    norm_gaus = np.pi*sigma    norm_exp = 2*np.pi * lam**2 * gamma(2/alpha)/alpha
          d = {"Flux":0,"SizeSource":popt[1],"Sigma":popt[2],"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
          print("Flux = 0\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(popt[1],popt[2],minb,mina,platescale,NewCenter))
      return d
  #                plt.figtext(0.74,0.18,r'$\displaystyle\sigma =$ %0.3f pix \n$\displaystyle\lambda =$ %0.3f pix \n$\displaystyle pGaus = \%$%0.2f\n$\displaystyle\alpha = $%0.1f' % (popt[2],popt[3],100*e_gaus/(e_gaus + e_exp),popt[4]), fontsize=18,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})
  #                plt.show()




def DS9open(xpapoint, filename=None):
    """
    """
    if filename is None:
        filename = sys.argv[3]
    if os.path.isfile(filename):
        print('Opening = ',filename)
        d = DS9(xpapoint)#DS9(xpapoint)
        d.set('grid no')
        d.set('frame new')
        d.set("file {}".format(filename))#a = OpenFile(xpaname,filename = filename)
    else:
        print('File not found, please verify your path')
        sys.exit()
    return

#def OpenFile(xpapoint,filename):#,filename = None,Internet =False, smooth=2, regions=True, centering=False,rot=0):
#    d = DS9(xpapoint)#DS9(xpapoint)
#    d.set("file {}".format(filename))
#    return

def Charge_path(xpapoint):
    from astropy.io import fits
    #'7739194-7741712'#''#sys.argv[3]#'7739194-7741712'#sys.argv[3]#'7738356-7742138  '#sys.argv[3]#'7738356-7742135  '#sys.argv[3]
#    try:
#        entry = sys.argv[3]#'7739194-7741712'#sys.argv[3]
#        try:
#            print(entry)
#            n1, n2 = entry.split('-')
#        except:
#            n1=''
#            n2=''
#    except IndexError:
#        n1=''
#        n2=''
#    print('N1 = {}, N2 = {}'.format(n1,n2))
#    #print(type(n1))
#    try:
#        n1, n2 = int(n1), int(n2)
#    except ValueError:
#        pass
    try:
        entry = sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# 
        print('Entry 1 = ', entry)
    except:
        n1=''
        n2='' 
        numbers = ['']
        entry = ''
    numbers = entry.split('-')
#    if numbers in None:
#        pass
    if len(numbers) == 1:
        numbers = None
    elif len(numbers) == 2:
        n1, n2 = entry.split('-')
        n1, n2 = int(n1), int(n2)
        numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
    print('Numbers used: {}'.format(numbers))           
    

    d = DS9(xpapoint)
    filename = d.get("file")
    fitsimage = fits.open(filename)
    if fitsimage[0].header['BITPIX'] == -32:
        Type = 'guider'
    else:
        Type = 'detector'
    print ('Type = {}'.format(Type))
    path = []
    if Type == 'detector':
        if numbers is not None:
            print('Specified numbers are integers, opening corresponding files ...')
            for number in numbers:
                #path = os.path.dirname(filename) + '/image%06d.fits' % (number)
                path.append(os.path.dirname(filename) + '/image%06d.fits' % (int(number)))
               # x = np.arange(n1,n2+1)
        if numbers is None:
            print('Not numbers, taking all the .fits images from the current repository')
            path = glob.glob(os.path.dirname(filename) + '/image*.fits')
            #x = np.arange(len(path))
    if Type == 'guider':
        files = glob.glob(os.path.dirname(filename) + '/stack*.fits')
        im_numbers = []
        try:
            for file in files:
                name = os.path.basename(file)
                im_numbers.append(int(name[5:12]))
            im_numbers = np.array(im_numbers)
        except:
            pass
        map
        if numbers is None:
            print('Not numbers, taking all the .fits images from the current repository')
            path = glob.glob(os.path.dirname(filename) + '/stack*.fits')
        elif len(numbers) == 2:
            print('Two numbers given, opening files in range ...')
            #print(np.arange(n1,n2+1))

            path = []
            for i, im_number in enumerate(im_numbers):
                if (im_number >= n1) & (im_number <= n2):
                    path.append(files[i])
                    print(files[i])
        elif len(numbers) > 2:
            print('More than 2 numbers given, opening the corresponding files ...')
            path = []
            numbers = [int(number) for number in numbers]
            #print (im_numbers)
            #print (numbers)
            for i, im_number in enumerate(im_numbers):
                if im_number in numbers:
                    path.append(files[i])
    for file in path:
        if 'table' in file:
            path.remove(file)
    path = np.sort(path)
    print(path)   
    return path
    
def DS9visualisation_throughfocus(xpapoint):
    """
    """
    d = DS9(xpapoint)
    path = Charge_path(xpapoint)        
    d.set('tile yes')
    #d.set("cmap Cubehelix0")
    d.set("frame delete")
    d.set("smooth no")
    for filen in path[:]:
        #d.set("file {}".format(filen)) 
        d.set('frame new')
        d.set("fits {}".format(filen))        
    try:
        a = getregion(d)
        d.set('pan to %0.3f %0.3f physical' % (a.xc,a.yc))
    except:
        pass
    d.set("lock frame physical")
    d.set("lock scalelimits yes") 
    d.set("lock smooth yes") 
    d.set("lock colorbar yes") 
    #d.set("lock crosshair %f %f"%(a.xc,a.yc))
    #d.set("scale mode 99.5")#vincent
    return

    
#xpapoint,filename ='ac148f06:51460', '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/FBGuider2018/stack8103716_pa+078_2018-06-11T06-21-24_new.fits'


def plot_hist2(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,
               ygaussfit,n_bias,n_log,threshold0,threshold55,plot_flag=False):    
    import matplotlib.pyplot as plt
    if plot_flag:    
        #plt.close("all")
        #plt.clf()
        fig = plt.figure(figsize=(12,4.5))
        fig.add_subplot(111)
        #plt.rc('text', usetex=True)
        #plt.rc('font',**{'family':'sans-serif', 'sans-serif':['Times']})
        plt.xlabel("Pixel Value [ADU]",fontsize=15)
        plt.ylabel("Log10(\# Pixels)",fontsize=15)
        #plt.axis([0,np.max(n_log),0,bins[bins.size-1]])
        fig = plt.plot(bin_center, n_log, "rx", label="Histogram")
        fig = plt.plot(xgaussfit,np.log10(ygaussfit), "b-", label="Gaussian")
        fig = plt.plot(np.ones(len(n_log))*threshold0,n_log, "b--", label="Bias")
        fig = plt.plot(np.ones(len(n_log))*threshold55, n_log, "k--", label="5.5 Sigma")
        fig = plt.plot(xlinefit,ylinefit, "g--", label="EM gain fit")
        plt.figtext(.43, .70, 'Bias value = %0.3f DN \nSigma = %0.3f DN \n '
                    'EM gain = %0.3f e/e' % (bias, sigma, emgain),
                    fontsize=15,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
        plt.legend(loc="upper right",fontsize=15)   
        plt.grid(b=True, which='major', color='0.75', linestyle='--')
        plt.grid(b=True, which='minor', color='0.75', linestyle='--')
        plt.tick_params(axis='x', labelsize=13)
        plt.tick_params(axis='y', labelsize=13)
        axes = plt.gca()
        axes.set_ylim([0,np.log10(n_bias) +0.1])
        a = np.isfinite(n_log)
        axes.set_xlim((np.percentile(bin_center[a],0.1),np.percentile(bin_center[a],80)))#([10**4,10**4.3])    
        #fn =  directory + 'figures/'
        #fig_dir = os.path.dirname(fn) 
        #print(fn)
        #if not os.path.exists(fig_dir):# create data directory if needed
        #    os.makedirs(fig_dir)
        #    plt.savefig(fn + image.replace('.fits', '.hist.png'), dpi = 100, bbox_inches = 'tight')
        plt.show()
        #plt.savefig(image.replace('.fits', '.hist.png'), dpi = 100, bbox_inches = 'tight')

#plot(bin_center,n_log);xlim((np.percentile(bin_center[a],0.1),np.percentile(bin_center[a],80)))
def calc_emgain(image, area,plot_flag=True):
    """
    """
	# Read data from FITS image
#    try:
#        img_data = fits.open(image)[0].data
#    except IOError:
#        raise IOError("Unable to open FITS image %s" %(image))	
#    if np.ndim(img_data) == 3:
#		# Image dimension
#        zsize, ysize, xsize = img_data.shape
#        img_section = img_data[:,area[0]:area[1],area[2]:area[3]]
#        stddev = np.std(img_data[:,area[0]:area[1],area[2]:area[3]])
#        img_size = img_section.size
#    else:
#		# Image dimension
#        ysize, xsize = img_data.shape  
#        img_section = img_data[area[0]:area[1],area[2]:area[3]]
#        stddev = np.std(img_data[area[0]:area[1],area[2]:area[3]]	)	
#        img_size = img_section.size    
    img_data = image
    ysize, xsize = img_data.shape  
    img_section = img_data[area[0]:area[1], area[2]:area[3]]
    stddev = np.std(img_data[area[0]:area[1], area[2]:area[3]])	
    img_size = img_section.size 
    nbins = 1000
    readnoise = 60
    gain=1.3

	# Histogram of the pixel values
    n, bins = np.histogram(np.array(img_section), bins=nbins)
    bin_center = 0.5 * (bins[:-1] + bins[1:])#center of each bin
    y0 = np.min(n)		
    n_log = np.log10(n)	
    # What is the mean bias value?
    idx = np.where(n == n.max())
    bias = bin_center[idx][0]
    n_bias = n[idx][0]  #number of pixels with this value of pixel
    
    # Range of data in which to fit the Gaussian to calculate sigma before -1.5 and 2.5
    bias_lower = bias - float(1.5) * readnoise #if you get an error this value will need adjusting
    bias_upper = bias + float(2.5) * readnoise #if you get an error this value will need adjusting
    #print (bias_lower,bias_upper, readnoise, bias)
    #print (bin_center)
    idx_lower = np.where(bin_center >= bias_lower)[0][0]
    #print (idx_lower)
    idx_upper = np.where(bin_center >= bias_upper)[0][0]

#   gauss_range = np.where(bin_center >= bias_lower)[0][0] ne sert a rien je crois
    
    valid_idx = np.where(n[idx_lower:idx_upper] > 0)

    amp, x0, sigma = gaussianFit(bin_center[idx_lower:idx_upper][valid_idx], 
                                 n[idx_lower:idx_upper][valid_idx], [n_bias, bias, readnoise])
            
    #plt.figure()
    #plt.plot(bin_center[idx_lower:idx_upper], n[idx_lower:idx_upper], 'r.')
    #plt.show()

    # Fitted frequency values
    xgaussfit = np.linspace(bin_center[idx_lower], bin_center[idx_upper], 1000)
    #print xgaussfit
    ygaussfit = gaussian(xgaussfit, amp, x0, sigma)
    #print ygaussfit

    # Define index of "linear" part of the curve: before 10 and 50
    threshold_min = bias + (float(8.0) * sigma) #lower limit to fit line to measure slope for flat part of histogram --might need to adjust 10.0 to 8.0
    threshold_max = bias + (float(40.0) * sigma) #upper limit to fit line to measure slope for flat part of histogram --might need to adjust 50.0 from 30.0 to 80.0
    
    # Lines for bias, 5.5*sigma line
    
    n_line = n_log.size
    zeroline = np.zeros([n_line], dtype = np.float32)
    threshold0 = int(bias)
    threshold55 = int(bias + 5.5*sigma)
    thresholdmin = int(threshold_min)
    thresholdmax = int(threshold_max)
    
    idx_threshmin = np.array(np.where(bin_center >= threshold_min))[0,0]
#    idx_threshmax = np.array(np.where(bin_center >= threshold_max))[0,0]
    idx_threshmax = np.array(np.where(bin_center >= threshold_max))[0,0]

    valid_idx2 = np.where(n[idx_threshmin:idx_threshmax] > 0)
    
    slope, intercept = fitLine(bin_center[idx_threshmin:idx_threshmax][valid_idx2], 
                               n_log[idx_threshmin:idx_threshmax][valid_idx2]) 
#        slope, intercept = fitLine(bin_center[idx_threshmin:idx_threshmax], n_log[idx_threshmin:idx_threshmax])  
    # Fit line
#        xlinefit = np.linspace(bias, bin_center[idx_threshmax], 1000)
    xlinefit = np.linspace(threshold_min, threshold_max, 1000)
    ylinefit = linefit(xlinefit, slope, intercept)
    #plt.plot(xlinefit,ylinefit);plt.plot(bin_center[idx_threshmin:idx_threshmax][valid_idx2],n_log[idx_threshmin:idx_threshmax][valid_idx2])
    emgain = (-1./slope) * (gain)
    hist = open('histplot.txt', 'w')
    hist.write('%0.0f/%0.0f/%0.0f/' % (emgain,bias,sigma) +json.dumps(list(bin_center)) + '/' +json.dumps(list(n.astype(float))) + '/' +json.dumps(list(xlinefit)) + '/' +json.dumps(list(ylinefit)) + '/' +json.dumps(list(xgaussfit)) + '/' +json.dumps(list(ygaussfit)) + '/%0.0f/' % (n_bias) + json.dumps(list(np.nan_to_num(n_log))) + '/%0.0f/%0.0f' % (threshold0,threshold55))
    hist.close()  
    if plot_flag:
        plot_hist2(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,
                   ygaussfit,n_bias,n_log,threshold0,threshold55,plot_flag=plot_flag)
    #if area == image_area:
    print('This needs to be corrected: image area and overscan area')
    print ('pCIC + sCIC = ' , float(len(img_section[img_section>bias+5.5*sigma]))/len(img_section.flatten()))
    #if area == overscan_area:
    print ('sCIC = ', float(len(img_section[img_section>bias+5.5*sigma]))/len(img_section.flatten()))
    return (emgain,bias,sigma,amp,slope,intercept) 

 
def apply_pc(image,output,area=0):
    """Put image pixels to 1 if superior to threshold and 0 else
    """
    cutoff = int(output[1]) + int(output[2])*5.5
    idx = image > cutoff - 1        
    image[idx] = np.ones(1, dtype = np.uint16)[0]
    image[~idx] = np.zeros(1, dtype = np.uint16)[0]
    return image

def DS9photo_counting(xpapoint):
    """Calculate threshold of the image and apply phot counting
    """
    from astropy.io import fits
    d = DS9(xpapoint)
    filename = d.get("file")
    image_area = [0,2069,1172,2145]
    if os.path.isfile(filename[:-5]+ '_pc.fits'):
        d.set('frame new')
        d.set("file {}".format(filename[:-5]+ '_pc.fits'))   
    else:
        fitsimage = fits.open(filename)
        image = fitsimage[0].data
        output = calc_emgain(image,area=image_area,plot_flag=True)
        emgain,bias,sigma,amp,slope,intercept = output
        new_image = apply_pc(image,output,area=0)
        print (new_image.shape)
        fitsimage[0].data = new_image
        if 'NAXIS3' in fitsimage[0].header:
            fitsimage[0].header.remove('NAXIS3')
        fitsimage.writeto('/tmp/pc.fits', overwrite=True)
        d.set('frame new')
        d.set('file /tmp/pc.fits')    #        d.set("file {}".format(filename[:-5]+ '_pc.fits'))   
    return


def gaussian(x, amp, x0, sigma):
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussianFit(x, y, param):
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(gaussian, x, y, p0=param)   
    amp, x0, sigma = popt   
    return (amp, x0, sigma) 


def linefit(x, A, B):
    return A*x + B


def fitLine(x, y, param=None):
    """
    """
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(linefit, x, y, p0 = param)
    a, b = popt
    return (a, b)

def ind2sub(array_shape, ind):
    """
    """
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def DS9next(xpapoint):
    """
    """
    d = DS9(xpapoint)
    filename = d.get("file")
    files = glob.glob(os.path.dirname(filename) + '/*.fits')
    files.sort()
    index = files.index(filename)#np.where(files == '%s' % (filename))
    print(files,filename,index)
    d.set('tile no')
    try:
        d.set('frame new')
        #d.set("fits {}".format(filen)) 
        d.set("file {}".format(files[index+1]))
    except IndexError:
        print('No more files')
        sys.exit()
    return
                         
def DS9previous():
    return                         
                         
def create_multiImage(xpapoint, w=0.20619, n=30, rapport=1.8, continuum=False):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """
    from astropy.table import Table
    from astropy.io import fits
    line = sys.argv[3]#'f3 names'#sys.argv[3]
    if '202' in line:
        w = 0.20255
    if '206' in line:
        w = 0.20619
    if '213' in line:
        w = 0.21382
    if 'lya' in line:
        w = None        
        
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsfile = fits.open(filename)
    image = fitsfile[0].data
    try:
        table = Table.read(filename[:-5] + '_table.csv')
    except IOError:
        print('No csv table found, Trying fits table')
        try:
            table = Table.read(filename[:-5] + '_table.fits')
        except IOError:
            print('No fits table found, Please run focustest')
            sys.exit() 
    if w is None:
        table = table [(table['wavelength'] != 0.20255) & (table['wavelength'] != 0.20619) & (table['wavelength'] != 0.21382)
        & (table['wavelength'] != 0.0) & (table['wavelength'] != -1.0)]
    else:
        table = table [table['wavelength'] == w]
    x,y = table['X_IMAGE'], table['Y_IMAGE']
    n1, n2 = n, n
    if continuum:
        imagettes=[]
        for y,x in zip(x,y):
            imagettes.append(image[int(x)-n1:int(x) +n1,int(y)-n2:int(y) +n2])
            imagettes.append(image[int(x)-n1+80:int(x) +n1+80,int(y)-n2:int(y) +n2])
    else:
        imagettes = [image[int(x)-n1:int(x) +n1,int(y)-n2:int(y) +n2] for y,x in zip(x,y)]
    v1,v2 = 6,14
#    fig, axes = plt.subplots(v1, v2, figsize=(v2,v1),sharex=True)
#    for i, ax in enumerate(axes.ravel()): 
#        try:
#            ax.imshow(imagettes[i][:, ::-1])
#            ax.get_yaxis().set_ticklabels([])
#        except IndexError:
#            pass
    #size = len(table)
    try:
        new_image = np.ones((v1*(2*n) + v1,v2*(2*n) + v2))*np.min(imagettes)
    except ValueError:
        print ('No matching in the catalog, please run focustest before using this function')
        sys.exit()
    for index,imagette in enumerate(imagettes):
        j,i = index%v2,index//v2
        centrei, centrej = 1 + (2*i+1) * n,1 + (2*j+1) * n
        print (i,j)
        print (centrei,centrej)
        new_image[centrei-n:centrei+n,centrej-n:centrej+n] = imagette
    new_image[1:-1:2*n, :] = np.max(np.array(imagettes))
    new_image[:,1:-1:2*n] = np.max(np.array(imagettes))
    if continuum:
        new_image[0:-2:2*n, :] = np.max(np.array(imagettes))
        new_image[:,0:-2:4*n] = np.max(np.array(imagettes))
    fitsfile[0].data = new_image[::-1, :]
    fitsfile.writeto('/tmp/imagettes.fits', overwrite=True)
    d.set('frame new')
    d.set("file /tmp/imagettes.fits")
    d.set('scale mode 90')
    return
#createMultiImage(filename,continuum=False,w=0.20619)

def DS9tsuite(xpapoint):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """
    path = os.path.dirname(os.path.realpath(__file__))    
    d = DS9(xpapoint)
    d.set('frame delete all')
    #d.set('frame new')

    print('''\n\n\n\n      TEST: Open    \n\n\n\n''')
    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')

    print('''\n\n\n\n      TEST: Setup   \n\n\n\n''')
    DS9setup2(xpapoint)

    print('''\n\n\n\n      TEST: Back   \n\n\n\n''')
    back(xpapoint)

    print('''\n\n\n\n      TEST: Visualization Detector   \n\n\n\n''')
    print('''\n\n\n\n      TEST: Stacking Detector   \n\n\n\n''')
    sys.argv.append('');sys.argv.append('');sys.argv.append('');sys.argv.append('')
    sys.argv[3] = ''
    DS9visualisation_throughfocus(xpapoint)
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
    sys.argv[3] = '402-404'
    sys.argv[4] = '407-408'
    DS9visualisation_throughfocus(xpapoint)
    DS9stack(xpapoint)    
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
    sys.argv[3] = '402-404-406'    #sys.argv[4] = '403-401-407'
    DS9visualisation_throughfocus(xpapoint)
    DS9stack(xpapoint)    


    print('''\n\n\n\n      TEST: Visualization Guider   \n\n\n\n''')
    print('''\n\n\n\n      TEST: stacking Guider   \n\n\n\n''')
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack7662505_pa+119_2018-06-10T18-34-51.fits')
    sys.argv[3] = ''
    DS9visualisation_throughfocus(xpapoint)
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack7662505_pa+119_2018-06-10T18-34-51.fits')
    sys.argv[3] = '7662504-7662914'
    DS9visualisation_throughfocus(xpapoint)
    DS9stack(xpapoint)    
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack7662505_pa+119_2018-06-10T18-34-51.fits')
    sys.argv[3] = '7662505-7662913-7664160'
    DS9visualisation_throughfocus(xpapoint)
    DS9stack(xpapoint)    

    print('''\n\n\n\n      TEST: Next Guider   \n\n\n\n''')
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack7662505_pa+119_2018-06-10T18-34-51.fits')
    print('''\n\n\n\n      TEST: Throughfocus Guider   \n\n\n\n''')
    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (812,783.2,40))
    d.set('regions select all') 
    sys.argv[3] = ''
    DS9throughfocus(xpapoint)
    sys.argv[3] = '7662505-7663744'
    DS9throughfocus(xpapoint)    

    print('''\n\n\n\n      TEST: Next detector   \n\n\n\n''')

    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
    print('''\n\n\n\n      TEST: Throughfocus detector   \n\n\n\n''')

    DS9next(xpapoint)
    print('''\n\n\n\n      TEST: Throughfocus detector   \n\n\n\n''')
    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (1677,1266.2,40))
    d.set('regions select all') 
    sys.argv[3] = ''
    DS9throughfocus(xpapoint)
    sys.argv[3] = '404-408'
    DS9throughfocus(xpapoint)  

    print('''\n\n\n\n      TEST: Radial profile   \n\n\n\n''') 
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/TestImage.fits')    
    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (1001,900.2,40))
    d.set('regions select all') 
    sys.argv[3] = '3'
    DS9rp(xpapoint)  
    sys.argv[3] = ''
    DS9rp(xpapoint)  
    print('''\n\n\n\n      TEST: Centering spot   \n\n\n\n''') 
    DS9center(xpapoint)  

    print('''\n\n\n\n      TEST: Centering slit   \n\n\n\n''') 
 
    
    d.set('regions delete all') 
    d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=yellow"' % (1502,901,20,10))
    d.set('regions select all')
    DS9center(xpapoint)      
    
    print('''\n\n\n\n      TEST: Show slit regions   \n\n\n\n''') 
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/detector/image-000075-000084-Zinc-with_dark-121-stack.fits')    
    sys.argv[3] = 'f3'
    Field_regions(xpapoint)
    d.set('regions delete all') 
    sys.argv[3] = 'f4-lya'
    Field_regions(xpapoint)
    d.set('regions delete all') 
    sys.argv[3] = 'f1-names'
    Field_regions(xpapoint)
    d.set('regions delete all') 
    
    print('''\n\n\n\n      TEST: diffuse focus test analysis   \n\n\n\n''') 
    sys.argv[3] = 'f3'
    DS9focus(xpapoint)

    print('''\n\n\n\n      TEST: Imagette lya   \n\n\n\n''') 
    sys.argv[3] = '206'
    create_multiImage(xpapoint)

    print('''\n\n\n\n      TEST: Photocounting   \n\n\n\n''') 
    d.set('frame delete all')
    d.set('frame new')
    DS9open(xpapoint,path + '/test/detector/image-000075-000084-Zinc-with_dark-121-stack.fits')    
    DS9photo_counting(xpapoint)
    
    print('''\n\n\n\n      TEST: Guider WCS   \n\n\n\n''') 
    d.set('frame delete all')
    d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/stack8103716_pa+078_2018-06-11T06-21-24_wcs.fits')    
    DS9guider(xpapoint)  

    print('''\n\n\n\n      END OF THE TEST: EXITED OK   \n\n\n\n''') 

    return

    
    
#    DS9next(xpapoint)
#    #create a circle center on a line
#    d.set('regions command "circle %i %i %0.1f # color=red"' % (1843.93,615.27,40))
#    d.set('regions select all')
#    DS9center(xpapoint)
#    DS9rp(xpapoint)
#    DS9throughfocus(xpapoint)
#    DS9visualisation_throughfocus(xpapoint)
#    d.set('frame delete all')
#    d.set('frame new')
#    DS9open(xpapoint,'/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/FBGuider2018/stack8103716_pa+078_2018-06-11T06-21-24_new.fits')
#    DS9guider(xpapoint)
#    d.set('frame delete all')
#    d.set('frame new')
#    DS9open(xpapoint,'/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180614/photon_counting/image000016.fits')
#    DS9photo_counting(xpapoint)
#    d.set('frame delete all')
#    d.set('frame new')
#    DS9open(xpapoint,'/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180612/image-000025-000034-Zinc-with_dark-161-stack.fits')
#    create_multiImage(xpapoint)
#    print('Test completed: OK')
    return 

def Field_regions(xpapoint, mask=''):
    from astropy.io import fits
    d = DS9(xpapoint)
    #d.set("regions format ds9")
    #d.set("regions system image")
    path = d.get("file")
    ImageName = os.path.basename(path)
    if ImageName[:5] == 'image':
        Type = 'detector'
    if ImageName[:5] == 'stack':
        Type = 'guider'
    print('Type = ', Type)
    if mask == '':
        try:
            mask = sys.argv[3]#'f3 names'#sys.argv[3]
        except:
            mask = ''
    print ('Masks = ', mask)
    mask = mask.lower()
    if Type == 'detector':
        if ('f1' in mask):
            if ('lya' in mask):
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F1_119_Lya.reg'
            else:
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F1_119_Zn.reg'
            #if ('name' in mask):
                #   filename2 = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F1_119_names.reg'        
        if ('f2' in mask):
            if ('lya' in mask):
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F2_-161_Lya.reg'
            else:
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F2_-161_Zn.reg'
            if ('name' in mask):
                filename2 = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F2_-161_names.reg'
        if ('f3' in mask):
            if ('lya' in mask):
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F3_-121_Lya.reg'
            else:
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F3_-121_Zn.reg'
            if ('name' in mask):
                filename2 = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F3_-121_names.reg'
        if ('f4' in mask):
            if ('lya' in mask):
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F4_159_Lya.reg'
            else:
                filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F4_159_Zn.reg'
            #if ('name' in mask):
                #filename2 = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F4_159_names.reg'
        if ('grid' in mask):
            filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/grid_Zn.reg' 
        d.set("region {}".format(filename))
        d.set('frame last')
        for i in range(int(d.get('frame'))-1):
            d.set('frame next')
            d.set('regions ' + filename)



    if ('d' in mask):
        filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/DetectorFrame.reg' 
    if ('g' in mask):
        filename = os.path.dirname(os.path.realpath(__file__)) + '/Slits/GuiderFrame.reg' 
    if Type == 'guider':
        if mask == 'no':
            d.set('frame last')
            for i in range(int(d.get('frame'))-1):
                d.set('frame next')
                d.set('contour clear')
                
        else:
            d.set('contour clear')
            #d.set('contour yes')
            pa = int(fits.open(path)[0].header['ROTENC'])
            print('Position angle = ',pa)
            if (pa>117) & (pa<121):
                name1 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/GSF1.reg'
                name2 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F1.ctr'
    #            d.set('regions /Users/Vincent/Documents/FireBallPipe/Calibration/Slits/GSF1.reg')
    #            d.set('contour load /Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F1.ctr')
            if (pa>-163) & (pa<-159):
                name1 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/GSF2.reg'
                name2 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F2.ctr'
            if (pa>-123) & (pa<-119):
                name1 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/GSF3.reg'
                name2 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F3.ctr'
            if (pa>157) & (pa<161):
                name1 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/GSF4.reg'
                name2 = '/Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F4.ctr'
            d.set('regions ' + name1)
            d.set('contour load ' + name2)
            d.set('frame last')
            for i in range(int(d.get('frame'))-1):
                d.set('frame next')
                d.set('regions ' + name1)
                d.set('contour load ' + name2)



    print('Putting regions, filename = ', filename)
            
    #d.set('contour load /Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F4.ctr')

        
        
    try:
        d.set('regions {}'.format(filename2))
    except:
        pass
    print('Test completed: OK')
    return

def DS9XYAnalysis(xpapoint):
    from matplotlib import pyplot as plt
    d = DS9(xpapoint)    
    filename = d.get('file')
    d.set('frame first')
    n1 = int(d.get('frame'))
    
    d.set('frame last')
    n2 = int(d.get('frame'))
    n=n2-n1+1
    print('Number of frame = ',n)
    d.set('frame first')
    Centers = np.zeros((n,2))
    SlitPos = np.zeros((n,2))
    SlitDetectedPos = np.zeros((n,2))
    DetectedRegFile = glob.glob(os.path.dirname(filename) + '/*detected.reg')

    mask = sys.argv[3]
    print(n)
    for frame in range(n):
        x, y = d.get('pan image').split(' ')
        xc, yc = int(float(x)), int(float(y))
        d.set('regions delete all')
        d.set('regions command "circle %0.3f %0.3f %0.3f # color=yellow"' % (xc,yc,30))
        d.set('regions select all')
        xnew, ynew = DS9center(xpapoint,Plot=False)#;plt.show()
        Centers[frame,:] = xnew, ynew
        if len(DetectedRegFile) == 1:
            d.set('regions file ' + DetectedRegFile[0])
        d.set('frame next')
    d.set('frame last')
    Field_regions(xpapoint, mask = mask)
        

    d.set('frame first')
    slitnames = []
    MappedRegions = []
    for frame in range(n):    
        d.set('regions select all')
        regions = d.get('regions').split('\n')
        SlitsOnImage = []
        for region in regions:
            if 'bpanda' in region:
                MappedRegions.append(region)
        regionsxy = np.zeros((len(MappedRegions),2))

        for i,region in enumerate(MappedRegions):
            a, reg, b = region.replace('(',')').split(')')
            a,name,b = b.replace('{','}').split('}')
            x,y,a1,a1,a1,a1,a1,a1,a1,a1,a1 = reg.split(',')
            x,y = float(x), float(y)
            regionsxy[i,:] = x, y
            SlitsOnImage.append(name)            
        distance = np.sqrt(np.square((Centers[frame,:] - regionsxy)).sum(axis=1))
        index = np.argmin(distance)
        SlitPos[frame] = regionsxy[index]
        slitnames.append(SlitsOnImage[index])


    detectedregions = []
    if len(DetectedRegFile) == 1:
        d.set('frame first')
        for frame in range(n):    
            d.set('regions file ' + DetectedRegFile[0])
            d.set('regions select all')
            regions = d.get('regions').split('\n')
            for region in regions:
                if 'circle' in region:
                    detectedregions.append(region)
            regions_detected_xy = np.zeros((len(detectedregions[1:-1]),2))
            SlitsOnImage = []
            for i,region in enumerate(detectedregions[1:-1]):
                a, reg, b = region.replace('(',')').split(')')
                #a,name,b = b.replace('{','}').split('}')
                x,y,r = reg.split(',')
                x,y,r = float(x), float(y), float(r)
                regions_detected_xy[i,:] = x, y
                #SlitsOnImage.append(name)

            
            distance = np.sqrt(np.square((Centers[frame,:] - regions_detected_xy)).sum(axis=1))
            index = np.argmin(distance)
            SlitDetectedPos[frame] = regions_detected_xy[index]
            
        print('Slit Detected positions are :')
        print(repr(SlitDetectedPos))    



    print('Slit predicted positions are :')
    print(repr(SlitPos))

    print('Centers of the spots are :')
    print(repr(Centers))
    
    plt.figure(figsize=(5,7))
    plt.title('Distances to slit')
    plt.xlabel('x pixel detector')
    plt.ylabel('y pixel detector')
    Q = plt.quiver(Centers[:,0],Centers[:,1],SlitPos[:,0] - Centers[:,0],SlitPos[:,1] - Centers[:,1],scale=30,label='Dist to mapped mask')
    try:
        Q = plt.quiver(Centers[:,0],Centers[:,1],SlitDetectedPos[:,0] - Centers[:,0],SlitDetectedPos[:,1] - Centers[:,1],scale=30,color='blue',label='Dist to detected slit')
    except:
        pass
    #Q = plt.quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - Centers[:,0],0,color='blue',scale=30)#, width=0.003)#, scale=Q)
    #Q = plt.quiver(SlitPos[:,0],SlitPos[:,1],0,SlitPos[:,1] - Centers[:,1],color='blue',scale=30)#, width=0.003)#, units='x')
    plt.quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
    plt.axis('equal')
    plt.gca().set_xlim(900,2300)
    plt.gca().set_ylim(-150,2300)
    #plt.xlim((1000,2000))
    plt.ylim((0,2000))
    plt.figtext(.15, .65,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitPos[:,0] - Centers[:,0]).mean(), (SlitPos[:,1] - Centers[:,1]).mean(),(SlitPos[:,0] - Centers[:,0]).std(), (SlitPos[:,1] - Centers[:,1]).std(), max((SlitPos[:,0] - Centers[:,0]), key=abs), max((SlitPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'black')
    plt.figtext(.15, .35,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitDetectedPos[:,0] - Centers[:,0]).mean(), (SlitDetectedPos[:,1] - Centers[:,1]).mean(),(SlitDetectedPos[:,0] - Centers[:,0]).std(), (SlitDetectedPos[:,1] - Centers[:,1]).std(), max((SlitDetectedPos[:,0] - Centers[:,0]), key=abs), max((SlitDetectedPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'blue')
    for i in range(len(slitnames)):
        plt.text(SlitPos[i,0],SlitPos[i,1], slitnames[i], color='red',fontsize=14)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig(os.path.dirname(filename) + '/XYCalibration.png')
    plt.show()
    
    
#    fig, ax = plt.subplots(1,2,figsize=(8, 8), sharex = True, sharey=True)
#    fig.suptitle('Distances to slit')
#    ax[0].set_xlabel('x pixel detector')
#    ax[0].set_ylabel('y pixel detector')
#    Q = ax[0].quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - Centers[:,0],SlitPos[:,1] - Centers[:,1],scale=30)
#    #Q = plt.quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - Centers[:,0],0,color='blue',scale=30)#, width=0.003)#, scale=Q)
#    #Q = plt.quiver(SlitPos[:,0],SlitPos[:,1],0,SlitPos[:,1] - Centers[:,1],color='blue',scale=30)#, width=0.003)#, units='x')
#    ax[0].quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
#    ax[0].axis('equal')
#    ax[0].set_xlim(1000,2000)
#    ax[0].set_ylim(0,2000)
#    #plt.xlim((1000,2000))
#    #ax[0].ylim((0,2000))
#    plt.figtext(.15, .65,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitPos[:,0] - Centers[:,0]).mean(), (SlitPos[:,1] - Centers[:,1]).mean(),(SlitPos[:,0] - Centers[:,0]).std(), (SlitPos[:,1] - Centers[:,1]).std(), max((SlitPos[:,0] - Centers[:,0]), key=abs), max((SlitPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'blue')
#    for i in range(len(slitnames)):
#        ax[0].text(SlitPos[i,0],SlitPos[i,1], slitnames[i], color='red',fontsize=14)
#    ax[0].grid(linestyle='dotted')
#
#
#    Q = ax[1].quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - SlitDetectedPos[:,0],SlitPos[:,1] - SlitDetectedPos[:,1],scale=30)
#    ax[1].quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
#    ax[1].axis('equal')
#    ax[1].set_xlim(1000,2000)
#    ax[1].set_ylim(0,2000)
#    #plt.xlim((1000,2000))
#    #ax[0].ylim((0,2000))
#    plt.figtext(.15, .65,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitPos[:,0] - Centers[:,0]).mean(), (SlitPos[:,1] - Centers[:,1]).mean(),(SlitPos[:,0] - Centers[:,0]).std(), (SlitPos[:,1] - Centers[:,1]).std(), max((SlitPos[:,0] - Centers[:,0]), key=abs), max((SlitPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'blue')
#    for i in range(len(slitnames)):
#        ax[1].text(SlitPos[i,0],SlitPos[i,1], slitnames[i], color='red',fontsize=14)
#    ax[1].grid(linestyle='dotted')
#
#
#    plt.savefig(os.path.dirname(filename) + '/XYCalibration.png')
#    plt.show()
    
    
    
    return Centers


#    


def DS9stack(xpapoint):
    from astropy.io import fits
    from focustest import stackImages
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsimage = fits.open(filename)[0]
    if fitsimage.header['BITPIX'] == -32:
        Type = 'guider'  
    else:
        Type = 'detector' 
    print('Type = ', Type)

    if Type == 'guider':
        files = Charge_path(xpapoint)
        print(files)
        n = len(files)
        image = fitsimage
        lx,ly = image.data.shape
        stack = np.zeros((lx,ly,n))
        print('\nReading fits files...')
        numbers = []
        for i,file in enumerate(files):
            name = os.path.basename(file) 
            numbers.append(int(name[5:12]))
            with fits.open(file) as f:
                stack[:,:,i] = f[0].data
        image.data = np.mean(stack,axis=2)
        fname = os.path.dirname(filename)                
        filename = '{}/StackedImage_{}-{}.fits'.format(fname, np.array(numbers).min(), np.array(numbers).max())
        image.writeto(filename ,overwrite=True)
        print('Images stacked:',filename)            
        
    if Type == 'detector':
        entry = sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# 
        print('Entry 1 = ', entry)
        try:
            number_dark = sys.argv[4] #''#sys.argv[4] #''#'sys.argv[4] #'365'#'365-374'#''#sys.argv[4] 
        except:
            number_dark = ''
        print('Entry 2 = ', number_dark)
        numbers = entry.split('-')
    
        if len(numbers) == 2:
            print('2 numbers given for the stack')
            n1, n2 = entry.split('-')
            n1, n2 = int(n1), int(n2)
            numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
        print('Numbers used: {}'.format(numbers))           
    
        #filename = d.get("file")
        path = os.path.dirname(filename)
        try:
            d1,d2 = number_dark.split('-') 
            print('2 numbers given for the dark')
            print(d1,d2)
            dark = stackImages(path,all=False, DS=0, function = 'mean', numbers=np.arange(int(d1),int(d2)), save=True, name="Dark")[0]
            Image, filename = stackImages(path,all=False, DS=dark, function = 'mean', numbers=np.array([int(number) for number in numbers]), save=True, name="Dark_{}-{}".format(int(d1),int(d2)))
        except ValueError:
            d1 = number_dark
            if d1 == '':
                dark = 0
                print('No dark')
                Image, filename = stackImages(path,all=False, DS=dark, function = 'mean', numbers=numbers, save=True, name="NoDark")
            else:
                dark = fits.open(path + '/image%06d.fits' % (int(d1)))[0].data
                Image, filename = stackImages(path,all=False, DS=dark, function = 'mean', numbers=numbers, save=True, name="Dark_{}".format(d1))

    d.set('tile yes')
    d.set('frame new')
    #d.set("lock scalelimits yes") 
    d.set("file {}".format(filename))   
    #d.set("scale mode 99.5")#vincent
    d.set("lock frame physical")
    
    return

def DS9focus(xpapoint):
#    sys.path.append('/Users/Vincent/Documents/FireBallIMO')
#    print(sys.path) 
#    from FireBallIMO.PSFInterpoler.PSFImageHyperCube import PSFImageHyperCube
#    from FireBallIMO.PSFInterpoler.PSFInterpoler import PSFInterpoler
#    from FireBallIMO.PSFInterpoler.SkySlitMapping        import SkySlitMapping
#    try:
#        from FireBallIMO.PSFInterpoler.PSFImageHyperCube import PSFImageHyperCube
#        from FireBallIMO.PSFInterpoler.PSFInterpoler import PSFInterpoler
#        from FireBallIMO.PSFInterpoler.SkySlitMapping        import SkySlitMapping
#    except:
#        pass
    from focustest import Focus  
    d = DS9(xpapoint)
    filename = d.get("file")
    #image = fitsfile[0].data
    try:
        entry = sys.argv[3] #f3 -121'#sys.argv[3] #''#sys.argv[4] #''#'sys.argv[4] #'365'#'365-374'#''#sys.argv[4] 
    except:
        F = Focus(filename = filename, HumanSupervision=False, source='Zn', shape='holes', windowing=False, peak_threshold=50,plot=False)
        d.set('regions {}'.format(filename[:-5]+'detected.reg'))

    #print ('Entry = ',entry)
    else:
        try:
            mask, pa = entry.split(' ')
            F = Focus(filename = filename, quick=False, threshold = [7], fwhm = [9,12.5],
                  HumanSupervision=False, reversex=False, source='all',
                  shape='slits', windowing=True, mask=mask.capitalize(), pa=int(pa) ,MoreSources=0,peak_threshold=50,plot=False)
        except ValueError:
            mask = entry
            F = Focus(filename = filename, quick=False, threshold = [7], fwhm = [9,12.5],
                  HumanSupervision=False, reversex=False, source='all',
                  shape='slits', windowing=True, mask=mask.capitalize(),MoreSources=0,peak_threshold=50,plot=False)
    
        d.set('regions {}'.format(filename[:-5]+'detected.reg'))

    return F


def DS9throughslit(xpapoint):#, nimages=np.arange(2,15), pos_image=np.arange(2,15), radius=15, center=[933, 1450], n_bg=1.3, sizefig=4):#, center_bg=[500,500]
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from focustest import estimateBackground
    from focustest import Gaussian
    from scipy.optimize import curve_fit

    print('''\n\n\n\n      START THROUGHSLIT \n\n\n\n''')
    try:
        entry = sys.argv[3]#'2-15'#'2-7-8-9-11-14-15'#'2-15'#'2-4-6-8-9'#sys.argv[3]
        numbers = entry.split('-')#[::-1]
        if len(numbers) == 2:
            n1,n2 = entry.split('-')
            n1,n2 = int(n1), int(n2)
            numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
        print(numbers)
        
    except IndexError:
        n1=''
        n2=''
    d = DS9(xpapoint)
    filename = d.get("file")
    path = []
    if (type(numbers) == list) or (type(numbers) == np.ndarray):
        print('Specified numbers are integers, opening corresponding files ...')
        for number in numbers:
            print (number)
            #path = os.path.dirname(filename) + '/image%06d.fits' % (number)
            path.append(os.path.dirname(filename) + '/image%06d.fits' % (int(number)))
            x = [int(i) for i in numbers]
        print (x)
    else:
        print('Not numbers, taking all the .fits images from the current repository')
        path = glob.glob(os.path.dirname(filename) + '/*.fits')
        x = np.arange(len(path))
    #    with fits.open(path) as f:
    #        files.append(f[0].data)        
    #    os.path.dirname(filename)
    #path = np.sort(path)
    print(path)
    
    
    a = getregion(d)

    radius = 15
    print('Sum pixel is used (another estimator may be prefarable)')
    fluxes=[]
    #n=radius
    
    for file in path:
        print (file)
        fitsfile = fits.open(file)[0]
        image = fitsfile.data
        #plt.figure(figsize=(sizefig,sizefig))
        #plt.imshow(image[int(a.yc)-radius:int(a.yc)+radius, int(a.xc)-radius:int(a.xc)+radius])#;plt.colorbar();plt.show()
        #plt.show()
        subimage = image[int(a.yc)-radius:int(a.yc)+radius, int(a.xc)-radius:int(a.xc)+radius]
        background = estimateBackground(image, [a.xc,a.yc], radius=30, n=1.8)
        #flux = np.sum(image[center[0]-n:center[0]+n,center[1]-n:center[1]+n])-np.sum(image[center_bg[0]-n:center_bg[0]+n,center_bg[1]-n:center_bg[1]+n])
        flux = np.sum(subimage - background) #- estimateBackground(image, center, radius, n_bg)
        fluxes.append(flux)
    fluxesn = (fluxes - min(fluxes)) / max(fluxes - min(fluxes))
#    maxf = x[np.where(fluxes==np.max(fluxes))[0][0]]#[0]
    
    x = np.arange(len(numbers))+1
    popt, pcov = curve_fit(Gaussian, x, fluxesn, p0=[1, x.mean(),3,0])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
    xl = np.linspace(x.min(),x.max(),100)
    maxf = xl[np.where(Gaussian(xl,*popt)==np.max(Gaussian(xl,*popt)))[0][0]]#[0]
    plt.figure()
    plt.plot(x, fluxesn,'o')
    plt.plot(xl, Gaussian(xl,*popt),'--')
    plt.plot(np.linspace(maxf, maxf, len(fluxes)), fluxesn/max(fluxesn))
    plt.grid()
    plt.xlabel('# image')
    #plt.title('Best image : {}'.format(os.path.basename(path[maxf])))
    plt.title('Best image : {}'.format(maxf))
    plt.ylabel('Sum pixel') 
    name = '%0.3f - %s - %s'%(maxf,[int(a.xc),int(a.yc)],fitsfile.header['DATE'])
    print(name) 
    plt.title(name)
    plt.savefig(os.path.dirname(file) + '/' + name + '.jpg')
    plt.show()
    return 

def DS9snr(xpapoint):
    from astropy.io import fits
    from focustest import create_DS9regions2
    n1 = 1.2
    n2 = 1.8
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsfile = fits.open(filename)
    image = fitsfile[0].data
    
    region = getregion(d)
    y, x = np.indices((image.shape))
    r = np.sqrt((x - region.xc)**2 + (y - region.yc)**2)  
    r = r.astype(np.int)
#    if hasattr(region, 'h'):
#        Xinf = int(region.yc - region.h/2)
#        Xsup = int(region.yc + region.h/2)
#        Yinf = int(region.xc - region.w/2)
#        Ysup = int(region.xc + region.w/2)
#        signal = fits.open(filename)[0].data[Xinf:Xsup,Yinf:Ysup]  
#
#        Xinfbi = int(region.yc - n1 * region.h/2)
#        Xsupbi = int(region.yc + n1 * region.h/2)
#        Yinfbi = int(region.xc - n1 * region.w/2)
#        Ysupbi = int(region.xc + n1 * region.w/2)
#
#        Xinfbs = int(region.yc - n2 * region.h/2)
#        Xsupbs = int(region.yc + n2 * region.h/2)
#        Yinfbs = int(region.xc - n2 * region.w/2)
#        Ysupbs = int(region.xc + n2 * region.w/2)
#        
#        background_mask_ext = (x >= Xinfbs) & (x <= Xinfbs) & (y <= Ysupbs) & (y >= Yinfbs)
#        background_mask_inf = (x >= Xinfbi) & (x <= Xinfbi) & (y <= Ysupbi) & (y >= Yinfbi)
#        maskBackground = background_mask_ext & ~background_mask_inf
#             
#        np.sum(background_mask_ext)
#        np.sum(background_mask_inf)
#        np.sum(maskBackground)
        
    if hasattr(region, 'r'):
        signal = image[r<region.r]
        maskBackground = (r >= n1 * region.r) & (r <= n2 * region.r)
    else:
        print('Need to be a circular region')
    signal_max = np.percentile(signal,95)
    noise = np.sqrt(np.nanvar(image[maskBackground]))
    background = np.nanmean(image[maskBackground])
    SNR = (signal_max - background) / noise
    print('Signal = ', signal_max)
    print('Background = ', background)
    print('Noise = ', noise)
    print('SNR = ', SNR)
    #d.set('regions command "text %i %i # text={SNR = %0.2f}"' % (region.xc+10,region.yc+10,SNR))
    try:
        os.remove('/tmp/centers.reg')
    except OSError:
        pass
    create_DS9regions2([region.xc+n2*region.r+10],[region.yc], form = '# text',
                       save=True,color = 'yellow', savename='/tmp/centers',
                       text = ['SNR = %0.2f' % (SNR)])
    d.set('regions /tmp/centers.reg')
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=yellow"' % (region.xc,region.yc,n1 * region.r))
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=yellow"' % (region.xc,region.yc,n2 * region.r))
    return
    
#
#    image = np.ones((100,100))
#    y, x = np.indices((image.shape))
#    r = np.sqrt((x - 30)**2 + (y - 30)**2)  
#    r = r.astype(np.int)
#    
#    imshow(r)
#    mask = (r >= 10) & (r <= 15)
#    imshow(mask)
#    


#np.arange(int(n1),int(n2))
 

def create_test_image():
    """
    Should test: centering (gaussian, slit), radial profile, stack dark, SNR, 
    throughfocus, 
    Du coup dans l'image il faut quil y ai: du bruit, une gaussienne, une fente convoluee avec une gausienne
    Radial profile: OK
    centering spot: OK
    centering slit: OK
    through focus
    throughlit
    """
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from focustest import twoD_Gaussian
    from focustest import ConvolveSlit2D_PSF
    n=20
    fitstest = fits.open('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180612/image000365.fits')
    #fitstest[0].data *= 0`
    lx, ly = fitstest[0].data.shape
    x, y = np.arange(lx), np.arange(ly)
    xy = np.meshgrid(x,y)
    new_image = fitstest[0].data
    for xi, ampi in zip((np.linspace(100,1900,10)),(np.linspace(10,1000,10))):
        slit = (1/0.006)*ConvolveSlit2D_PSF(xy, ampi, 3, 9, int(xi), 1500, 3,3).reshape(ly,lx).T
        new_image = new_image + slit# + gaussian2
        plt.imshow(slit[int(xi)-n:int(xi)+n,1500-n:1500+n]);plt.plt.colorbar();plt.show()
        plt.imshow(new_image[int(xi)-n:int(xi)+n,1500-n:1500+n]);plt.colorbar();plt.show()
    for xi, ampi in zip((np.linspace(100,1900,10)),(np.linspace(10,1000,10))):
        gaussian = twoD_Gaussian(xy, ampi, int(xi), 1000, 5, 5, 0).reshape(ly,lx).T
        new_image = new_image + gaussian# + gaussian2
        #imshow(new_image[int(xi)-n:int(xi)+n,1000-n:1000+n]);colorbar();plt.show()

#        
#    imshow(fitstest[0].data);colorbar()
#    imshow(new_image[1000-n:1000+n,1000-n:1000+n]);colorbar();plt.show()  # + gaussian[1000-n:1000+n,1000-n:1000+n]);colorbar()  
#    imshow(new_image[1500-n:1500+n,1000-n:1000+n]);colorbar();plt.show()  # + gaussian2[1000-n:1000+n,2000-n:2000+n]);colorbar()  
#    imshow(new_image[1900-n:1900+n,1000-n:1000+n]);colorbar();plt.show()  # + gaussian2[1000-n:1000+n,2000-n:2000+n]);colorbar()  
#    
    
    fitstest[0].data = new_image
    try:
        fitstest[0].header.remove('NAXIS3')  
    except KeyError:
        pass
    fitstest.writeto('/Users/Vincent/Documents/FireBallPipe/Calibration/test/TestImage.fits',overwrite = True)
    #imshow(fits.open('/Users/Vincent/Documents/FireBallPipe/Calibration/TestImage.fits')[0].data)
#    plt.figure()DS
#    plt.plot(fitstest[0].data[1000-n:1000+n,2000])
#    plt.plot(fitstest[0].data[1000,1000-n:1000+n])
#    plt.show()
    return





def DS9meanvar(xpapoint):
    """
    """
    from astropy.io import fits
    from scipy import stats
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = d.get("file")
    region = getregion(d)
    if hasattr(region, 'h'):
        xc, yc, w, h = int(region.xc), int(region.xc), int(region.w), int(region.h) 
        Xinf = yc - h/2
        Xsup = yc + h/2#int(region.yc + region.h/2)
        Yinf = xc - w/2 #int(region.xc - region.w/2)
        Ysup = xc + w/2 #int(region.xc + region.w/2)
    if hasattr(region, 'r'):
        xc, yc, r = int(region.xc), int(region.xc), int(region.r)
        Xinf = yc - r
        Xsup = yc + r
        Yinf = xc - r
        Ysup = xc + r
    image = fits.open(filename)[0].data[Xinf:Xsup,Yinf:Ysup]
    print ('Image : {}'.format(filename))
    print ('Mean : {}'.format(image.mean()))
    print ('Standard deviation : {}'.format(image.std()))
    print ('Skewness: {}'.format(stats.skew(image,axis=None)))
    return


def DS9lock(xpapoint):
    """
    """
    d = DS9(xpapoint)#DS9(xpapoint)
    lock = d.get("lock scalelimits")
    if lock == 'yes':
        d.set("lock frame no")
        d.set("lock scalelimits no")
        d.set("lock crosshair no")
        d.set("lock smooth no")
        d.set("lock colorbar noe")
    if lock == 'no':
        d.set("lock frame physical")
        d.set("lock scalelimits yes")
        d.set("crosshair lock physical")
        d.set("lock crosshair physical")
        d.set("lock smooth yes")
        d.set("lock colorbar yes")
    return


def DS9inverse(xpapoint):
    """
    """
    from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = d.get("file")
    fitsfile = fits.open(filename)[0]

    image = fitsfile.data
    new_image = - image + image.max()
    fitsfile.data = new_image
    fitsfile.data = new_image
    fitsfile.writeto(filename[:-5] + '_Inverse.fits',overwrite=True)
    d.set('frame new')
    d.set('file {}'.format(filename[:-5] + '_Inverse.fits'))
    return


def DS9center(xpapoint,Plot=True):
    """
    """
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from focustest import ConvolveBoxPSF
    from focustest import twoD_Gaussian
    from focustest import create_DS9regions
    from focustest import create_DS9regions2
    from focustest import estimateBackground
    from scipy.optimize import curve_fit
    from focustest import Gaussian
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = d.get("file")
    region = getregion(d)
    if hasattr(region, 'h'):
        xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
        print('W = ', w)
        print('H = ', h)
        if w <= 2:
            w = 2
        if h <= 2:
            h = 2
        Xinf = yc - h/2 -1
        Xsup = yc + h/2 -1 
        Yinf = xc - w/2 -1
        Ysup = xc + w/2 -1
        imagex = fits.open(filename)[0].data[Xinf-15:Xsup+15,Yinf:Ysup].sum(axis=1)
        imagey = fits.open(filename)[0].data[Xinf:Xsup,Yinf-15:Ysup+15].sum(axis=0)
        #lx, ly = image.shape
        model = ConvolveBoxPSF
        x = np.arange(-len(imagex)/2,len(imagex)/2)
        y = np.arange(-len(imagey)/2,len(imagey)/2)
        try:
            poptx, pcovx = curve_fit(model, x, imagex, p0=[imagex.max(), 20, 0., 10., np.median(imagex)])#,  bounds=bounds)
            popty, pcovy = curve_fit(model, y, imagey, p0=[imagey.max(), 10, 0., 10., np.median(imagey)])#,  bounds=bounds)
            ampx, lx, x0x, sigma2x, offsetx = poptx
            ampy, ly, x0y, sigma2y, offsety = popty
        except RuntimeError:
            print('Optimal parameters not found: Number of calls to function has reached maxfev = 1400.')
        print('Poptx = ', poptx)
        print('Popty = ', popty)
        
        newCenterx = xc + x0y#popty[2]
        newCentery = yc + x0x#poptx[2]
        d.set('regions delete select')
        print('''\n\n\n\n     Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f] \n\n\n\n''' % (region.yc,region.xc,newCentery,newCenterx))
        #d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=yellow"' % (newCenterx+1,newCentery+1,region.w,region.h))
        d.set('regions command "box %0.3f %0.3f %0.2f %0.2f # color=yellow"' % (newCenterx,newCentery,region.w,region.h))
        #d.set('regions command "circle %i %i %0.1f # color=yellow"' % (newCenterx+1,newCentery+1,2))
        #d.set('regions command "circle %0.3f %0.3f %0.2f # color=yellow"' % (newCenterx,newCentery,2))

        try:
            os.remove('/tmp/centers.reg')
        except OSError:
            pass
        #create_DS9regions([newCenterx],[newCentery], radius=2, save=True, savename="/tmp/centers", form=['circle'], color=['yellow'], ID='test')
        create_DS9regions2([newCenterx],[newCentery-15], form = '# text',
                           save=True,color = 'yellow', savename='/tmp/centers',
                           text = ['%0.2f - %0.2f' % (newCenterx,newCentery)])
#        create_DS9regions([newCenterx-1],[newCentery-1], radius=2, save=True, savename="/tmp/centers", form=['circle'], color=['yellow'], ID=[['%0.2f - %0.2f' % (newCenterx,newCentery)]])

        d.set('regions /tmp/centers.reg')
        
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6,6))
        axes[0].plot(x,imagex, 'bo', label='Spatial direction')
        axes[1].plot(y,imagey, 'ro', label='Spectral direction')
        axes[0].plot(x, model(x, *poptx), color='b')#,label='Spatial direction')
        axes[1].plot(y, model(y, *popty), color='r')#,label='Spatial direction')
        #plt.ylabel('Fitted profiles')
        axes[0].set_ylabel('Spatial direction');axes[1].set_ylabel('Spectral direction')
        axes[0].plot(x, Gaussian(x, imagex.max()-offsetx, x0x, sigma2x, offsetx), ':b',label='Deconvolved PSF') # Gaussian x, amplitude, xo, sigma_x, offset
        #axes[0].plot(x, Gaussian(x, ampx, x0x, sigma2x, offsetx), ':b',label='Deconvolved PSF') # Gaussian x, amplitude, xo, sigma_x, offset
        xcc = x - x0x
        #axes[0].plot(x, np.piecewise(x, [xc < -lx, (xc >=-lx) & (xc<=lx), xc>lx], [offsetx, ampx + offsetx, offsetx]), ':r', label='Slit size') # slit (UnitBox)
        axes[0].plot(x, np.piecewise(x, [xcc < -lx, (xcc >=-lx) & (xcc<=lx), xcc>lx], [offsetx, imagex.max() , offsetx]), ':r', label='Slit size') # slit (UnitBox)
        axes[0].plot([x0x, x0x], [imagex.min(), imagex.max()])
        axes[1].plot(y, Gaussian(y, imagey.max() - offsety, x0y, sigma2y, offsety), ':b',label='Deconvolved PSF') # Gaussian x, amplitude, xo, sigma_x, offset
        xcc = x - x0y
        axes[1].plot(x, np.piecewise(x, [xcc < -ly, (xcc >=-ly) & (xcc<=ly), xcc>ly], [offsety, imagey.max(), offsety]), ':r', label='Slit size') # slit (UnitBox)
        axes[1].plot([x0y, x0y], [imagey.min(), imagey.max()])
        plt.figtext(0.66,0.65,'Sigma = %0.2f +/- %0.2f pix\nSlitdim = %0.2f +/- %0.2f pix\ncenter = %0.2f +/- %0.2f' % ( np.sqrt(poptx[3]), np.sqrt(np.diag(pcovx)[3]/2.) , 2*poptx[1],2*np.sqrt(np.diag(pcovx)[1]), x0x, np.sqrt(np.diag(pcovx)[2])),bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})
        plt.figtext(0.67,0.25,'Sigma = %0.2f +/- %0.2f pix\nSlitdim = %0.2f +/- %0.2f pix\ncenter = %0.2f +/- %0.2f' % ( np.sqrt(popty[3]), np.sqrt(np.diag(pcovy)[3]/2.) , 2*popty[1],2*np.sqrt(np.diag(pcovy)[1]), x0y, np.sqrt(np.diag(pcovy)[2])),bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
        plt.show()

        
        pass
    if hasattr(region, 'r'):
        xc, yc, r = int(region.xc), int(region.yc), int(region.r)
        Xinf = yc - r -1#int(region.yc - region.r)
        Xsup = yc + r -1#int(region.yc + region.r)
        Yinf = xc - r -1#int(region.xc - region.r)
        Ysup = xc + r -1#int(region.xc + region.r)
        data = fits.open(filename)[0].data
        image = data[Xinf:Xsup,Yinf:Ysup]
        print('2D fitting with 100 microns fibre, to be updated by allowing each fiber size')
        background =  estimateBackground(data,[region.yc,region.xc],20,1.8 )
        image = image - background
        lx, ly = image.shape
        x = np.linspace(0,lx-1,lx)
        y = np.linspace(0,ly-1,ly)
        x, y = np.meshgrid(x,y)
        yo,xo = np.where(image == image.max())#ndimage.measurements.center_of_mass(image)
        maxx, maxy = xc - (lx/2 - xo), yc - (ly/2 - yo)
        print ('maxx, maxy = {}, {}'.format(maxx,maxy))
#        d.set('regions command "circle %i %i %0.1f # color=yellow"' % (maxy,maxx,1))
#        try:
#            os.remove('/tmp/centers.reg')
#        except OSError:
#            pass
#        create_DS9regions2([maxx],[maxy-10], form = '# text',
#                           save=True,color = 'yellow', savename='/tmp/centers',
#                           text = ['%0.2f - %0.2f' % (maxx,maxy)])
#        d.set('regions /tmp/centers.reg')
        bounds = ([1e-1*np.max(image), xo-10 , yo-10, 0.5,0.5,-1e5], [10*np.max(image), xo+10 , yo+10, 10,10,1e5])#(-np.inf, np.inf)#
        Param = (np.max(image),int(xo),int(yo),2,2,np.percentile(image,15))
        print ('bounds = ',bounds)
        print('\nParam = ', Param)
        try:
            popt,pcov = curve_fit(twoD_Gaussian,(x,y),image.flat,
                                  Param,bounds=bounds)
            print('\nFitted parameters = ', popt)
        except RuntimeError:
            print('Optimal parameters not found: Number of calls to function has reached maxfev = 1400.')
            sys.exit() 
        fit = twoD_Gaussian((x,y),*popt).reshape((ly,lx))
        #imshow(twoD_Gaussian((x,y), *Param).reshape((ly,lx)));colorbar()
        #imshow(image);colorbar()
        #imshow(fit);colorbar()
        newCenterx = xc - (lx/2 - popt[1])#region.xc - (lx/2 - popt[1])
        newCentery = yc - (ly/2 - popt[2])#region.yc - (ly/2 - popt[2])
        print('''\n\n\n\n     Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f] \n\n\n\n''' % (region.yc,region.xc,newCentery,newCenterx))

        print(np.diag(pcov))

        #plt.show()
#        d.set('regions command "text %i %i # text={%0.2f}"' % (newCenterx+10,newCentery+10,newCentery))

        d.set('regions delete select')

        #d.set('regions command "circle %0.2f %0.2f %0.2f # color=yellow"' % (newCenterx,newCentery,5))
        try:
            os.remove('/tmp/centers.reg')
        except OSError:
            pass
#        create_DS9regions2([newCenterx],[newCentery-10], form = '# text',
#                           save=True,color = 'yellow', savename='/tmp/centers',
#                           text = ['%0.2f - %0.2f' % (newCenterx,newCentery)])
        create_DS9regions([newCenterx-1],[newCentery-1], radius=5, save=True, savename="/tmp/centers", form=['circle'], color=['white'], ID=[['%0.2f - %0.2f' % (newCenterx,newCentery)]])
        
        d.set('regions /tmp/centers.reg')
        if Plot:
            plt.plot(image[int(yo), :], 'bo',label='Spatial direction')
            plt.plot(fit[int(yo), :],color='b')#,label='Spatial direction')
            plt.plot(image[:,int(xo)], 'ro',label='Spatial direction')
            plt.plot(fit[:,int(xo)],color='r')#,label='Spatial direction')
            plt.ylabel('Fitted profiles')
            plt.figtext(0.66,0.55,'Sigma = %0.2f +/- %0.2f pix\nXcenter = %0.2f +/- %0.2f\nYcenter = %0.2f +/- %0.2f' % ( np.sqrt(popt[3]), np.sqrt(np.diag(pcov)[3]/2.), lx/2 - popt[1] , np.sqrt(np.diag(pcov)[1]), ly/2 - popt[2], np.sqrt(np.diag(pcov)[2])),bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})
            plt.legend()
    return newCenterx, newCentery

def t2s(h,m,s,d=0):
    return 3600 * h + 60 * m + s + d*24*3600

if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    print(datetime.datetime.now())
    print (path)
    #

#    xpapoint = '7f000001:63777'
#    function = 'xy_calib'
#    sys.argv.append(xpapoint)
#    sys.argv.append(function)
###    
##    
#    sys.argv.append('f3')
    #sys.argv.append('10.49-0.25')#16.45-0.15
#d.set('contour save /Users/Vincent/Documents/FireBallPipe/Calibration/F4.ctr image')
#d.set('contour load /Users/Vincent/Documents/FireBallPipe/Calibration/F2.ctr')

    start = timeit.default_timer()

    DictFunction = {'centering':DS9center, 'radial_profile':DS9rp,
                    'throughfocuss':DS9throughfocus, 'open':DS9open,
                    'setup':DS9setup2,#'back':back,
                    'throughfocus_visualisation':DS9visualisation_throughfocus, 
                    'WCS':DS9guider, 'test':DS9tsuite,
                    'photo_counting':DS9photo_counting, 'lya_multi_image':create_multiImage,
                    'next':DS9next, 'previous':DS9previous,
                    'regions': Field_regions, 'stack': DS9stack,'lock': DS9lock,
                    'snr': DS9snr, 'focus': DS9focus,'inverse': DS9inverse,
                    'throughslit': DS9throughslit, 'meanvar': DS9meanvar,
                    'xy_calib': DS9XYAnalysis}
#    try:
    xpapoint = sys.argv[1]
    function = sys.argv[2]

    print("""
        ********************************************************************
                                     Function = %s         
        ********************************************************************
        """%(function)) 


    DictFunction[function](xpapoint)             



    stop = timeit.default_timer()

    print("""
        ********************************************************************
                            Exited OK, test duration = {}s      
        ********************************************************************
        """.format(stop - start)) 
#    except Exception as e:
#        print(e)
#        pass

#xpapoint='7f000001:57475'
#d = DS9tsuite(xpapoint)
#d.set('contour no')
#
#d.set('contour yes')
#d.set('contour smooth 8')
#d.set('contour smooth 5')
#d.set('contour smooth 5')

#        
#        import numpy
#        from scipy import optimize
#        
#        
#        ## This is y-data:
#        y_data = numpy.array([0.2867, 0.1171, -0.0087, 0.1326, 0.2415, 0.2878, 0.3133, 0.3701, 0.3996, 0.3728, 0.3551, 0.3587, 0.1408, 0.0416, 0.0708, 0.1142, 0, 0, 0])
#        
#        ## This is x-data:
#        t = numpy.array([67., 88, 104, 127, 138, 160, 169, 188, 196, 215, 240, 247, 271, 278, 303, 305, 321, 337, 353])
#        
#        def fitfunc(p, t):
#            """This is the equation"""
#            return p[0] + (p[1] -p[0]) * ((1/(1+np.exp(-p[2]*(t-p[3])))) + (1/(1+np.exp(p[4]*(t-p[5])))) -1)
#        
#        def errfunc(p, t, y):
#            return fitfunc(p,t) -y
#        #    
#        #def jac_errfunc(p, t, y):
#        #    ap = algopy.UTPM.init_jacobian(p)
#        #    return algopy.UTPM.extract_jacobian(errfunc(ap, t, y))
#            
#        guess = numpy.array([0, max(y_data), 0.1, 140, -0.1, 270])
#        p2, C, info, msg, success = optimize.leastsq(errfunc, guess, args=(t, y_data), full_output=1)
#        print('Estimates from leastsq \n', p2,success)
#        print('number of function calls =', info['nfev'])
#        
#        p3, C, info, msg, success = optimize.leastsq(errfunc, guess, args=(t, y_data), full_output=1)
#        print('Estimates from leastsq \n', p3,success)
#        print('number of function calls =', info['nfev'])
#        
#        #
#        lx,ly = 100,100
#        x = np.linspace(0,lx-1,lx)
#        y = np.linspace(0,ly-1,ly)
#        x, y = np.meshgrid(x,y)
#        data = (twoD_Gaussian((x,y),100, 40.3,67.2,1.5,1.8,2)).reshape(lx,ly)
#        #data.max()
#        noise = 0.5*np.random.normal(1,size=(lx,ly))
#        imshow(data+noise);colorbar()
#
#        leastsq(errfunc, args(), full_output=1)
#
#
#        
#        xo,yo = ndimage.measurements.center_of_mass(image)
#        bounds = (-np.inf, np.inf)#([0.1*np.max(image), xo-10 , yo-10, 0.5,0.5,-1e5], [10*np.max(image), xo+10 , yo+10, 10,10,1e5])
#        print('2D fitting with 100 microns fibre, to be updated by allowing each fiber size')
#        popt,pcov = curve_fit(twoD_Gaussian,(x,y),image.flat,
#                              (np.max(image),xo,yo,2,2,np.min(image)),bounds=bounds)
#
#
#def gaussian(height, center_x, center_y, width_x, width_y):
#    """Returns a gaussian function with the given parameters"""
#    width_x = float(width_x)
#    width_y = float(width_y)
#    return lambda x,y: height*exp(
#                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)
#
#def moments(data):
#    """Returns (height, x, y, width_x, width_y)
#    the gaussian parameters of a 2D distribution by calculating its
#    moments """
#    total = data.sum()
#    X, Y = indices(data.shape)
#    x = (X*data).sum()/total
#    y = (Y*data).sum()/total
#    col = data[:, int(y)]
#    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
#    row = data[int(x), :]
#    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
#    height = data.max()
#    return height, x, y, width_x, width_y
#
#def fitgaussian(data):
#    """Returns (height, x, y, width_x, width_y)
#    the gaussian parameters of a 2D distribution found by a fit"""
#    params = moments(data)
#    errorfunction = lambda p: ravel(gaussian(*p)(*indices(data.shape)) -
#                                 data)
#    p, success = optimize.leastsq(errorfunction, params)
#    return p
#
#from pylab import *
## Create the gaussian data
##Xin, Yin = mgrid[0:201, 0:201]
##data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)
#
#
#lx,ly = 100,100
#x = np.linspace(0,lx-1,lx)
#y = np.linspace(0,ly-1,ly)
#x, y = np.meshgrid(x,y)
#data = (twoD_Gaussian((x,y),100, 20,40,1.5,1.8,0)).reshape(lx,ly)
##data.max()
#noise =0.09*np.random.normal(1,size=(lx,ly))
#
#imshow(data+noise)
#
#params = fitgaussian(data+noise)
#fit = gaussian(*params)
#
#contour(fit(*indices(data.shape)), cmap=cm.copper)
#ax = gca()
#(height, x, y, width_x, width_y) = params
#
#text(0.95, 0.05, """
#x : %.1f
#y : %.1f
#width_x : %.1f
#width_y : %.1f""" %(x, y, width_x, width_y),
#        fontsize=16, horizontalalignment='right',
#        verticalalignment='bottom', transform=ax.transAxes)
#
#show()
#           


#plt.plot(1284.26, 662, 'o', label='Autocoll', markersize=20)
#plt.plot(1208, 455.75,'o', label='Source', markersize=20)
#plt.plot(1246.13, 558.875,'P',  label='Axis of the parabola', markersize=20)
#plt.plot(1208, 514.3,'P',  label='Spectro axis', markersize=20)
#plt.axis('equal')
#plt.grid()
#plt.legend()