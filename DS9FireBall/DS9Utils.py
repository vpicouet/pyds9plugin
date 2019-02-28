#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:13 2018

@author: Vincent
"""

from __future__ import print_function, division
import timeit
import glob
import os
import sys
import numpy as np
from pyds9 import DS9
import datetime
from  pkg_resources  import resource_filename
try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()

DS9_BackUp_path = os.environ['HOME'] + '/DS9BackUp/'
def CreateFolders(DS9_BackUp_path=DS9_BackUp_path):
    if not os.path.exists(os.path.dirname(DS9_BackUp_path)):
        os.makedirs(os.path.dirname(DS9_BackUp_path))
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/Plots'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/Plots')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/DS9Regions'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/DS9Regions')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/DS9Curves'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/DS9Curves')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/CSVs'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/CSVs')
    return


    
def DS9MultipleThreshold(xpapoint):
    """
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    filename = d.get("file")
    try:
        region = getregion(d)
    except ValueError:
        image_area = [0,2069,1172,2145]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        image_area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)

    if len(sys.argv) > 3:
        path = Charge_path(xpapoint, entry_point=3)
    else:
        path = [filename]

    d = DS9(xpapoint)
    fitsimage = d.get_fits()[0]
    imasum_noprior = np.zeros(fitsimage.data.shape[:])
    imasum_prior = np.zeros(fitsimage.data.shape[:])
    
    for fn in path:
        print(fn)
        with fits.open(fn) as f:
            ima = f[0].data#.T
            hdr = f[0].header
    
        ima_count_noprior = process_image(ima, hdr)
        ima_count_prior = process_image(ima, hdr, with_prior=True)

        imasum_noprior += ima_count_noprior
        imasum_prior += ima_count_prior
        fitsimage.data = imasum_noprior
        fitswrite(fitsimage,fn[:-5] + '_sum_no_prior.fits')
        fitsimage.data = imasum_noprior
        fitswrite(fitsimage,fn[:-5] + '_sum_with_prior.fits')
    if len(path) == 1:
        d.set('frame new')
        d.set('file ' + fn[:-5] + '_sum_no_prior.fits')  
        d.set('frame new')
        d.set('file ' + fn[:-5] + '_sum_with_prior.fits')  
    return


def normalgamma(nooe, g, r , lamb):
    """
    """
    from scipy.special import hyp1f1, gamma
    lamb2 = lamb/2.
    rsq = r**2
    aux = np.square(-g*nooe + rsq)/(2*g**2*rsq)
    res = 2**(-lamb2) * np.exp(- np.square(nooe)/(2*rsq)) * g**(-lamb) * r**(lamb - 2)
    res *= r  * hyp1f1(lamb2, .5, aux) / np.sqrt(2)*gamma((lamb + 1)/2) + \
            (g*nooe - rsq)*hyp1f1((lamb+1)/2, 3/2, aux) / (g*gamma(lamb2))
    
    return res



def bias_from_right_overscan(ima, left=2135, right=2580):
    """
    """
    from astropy.stats import sigma_clipped_stats
    overscan = ima[left:right,:] 
    bias, median, stdev = sigma_clipped_stats(overscan, mask=np.isnan(overscan), axis=0)
    ima_bias_cor = ima - bias
    return ima_bias_cor, bias, stdev


def plot_histo_bias(data, mean, median, stdev):
    """
    """
    from matplotlib import pyplot as plt    
    # full stat
    fmean = np.nanmean(data)
    fmed = np.nanmedian(data)
    # fitted gaussian
    plt.figure()
    hits, bins, _ = plt.hist(data, bins=100, range=[300,3700])
    binsize = bins[1] - bins[0]
    x = np.linspace(3000, 3500)
    gaussian = 1./(np.sqrt(2*np.pi)*stdev)*np.exp(-.5*np.square(x-mean)/np.square(stdev))
    count = np.count_nonzero((data>=(mean-3*stdev)) & (data<=(mean+3*stdev)))
    amp = count*binsize
    plt.plot(x, amp*gaussian, 'r')
    plt.plot([mean, mean],plt.ylim(), 'r', label='clipped mean')
    plt.plot([fmean, fmean], plt.ylim(), ':r', label='full mean')
    plt.plot([median, median],plt.ylim(), 'g', label='clipped median')
    plt.plot([fmed, fmed], plt.ylim(), ':g', label='full median')
    plt.legend()
    return




def multiple_photon_counting(ima, prior=None):
    """
    """
    tab_path = resource_filename('Calibration', 'PhotoCountingTables')
    tab_prior_name = os.path.join(tab_path, 'tabwithprior-EMgain470-rnoise109.csv')
    tab_woprior_name = os.path.join(tab_path, 'tabnoprior-EMgain470-rnoise109.csv')
    
    if prior is None:
        tab = np.loadtxt(tab_woprior_name, delimiter=',')
    else:
        tab = np.loadtxt(tab_prior_name, delimiter=',')
        
    # thresholding
    if prior is None:
        imac = np.searchsorted(tab, ima)
    else:
        prior_edge_values = tab[:-1,0] + (tab[1:,0] - tab[:-1,0])/2
        prior_index = np.searchsorted(prior_edge_values, prior)
        imac = np.zeros_like(ima, dtype=int)
        for i in range(tab.shape[0]):
            thresholds = tab[i,1:]
            # clean the table at the end (repeated max value or 0)
            last = np.where(thresholds == thresholds.max())[0].min()
            thresholds = thresholds[:last+1]
            mask = (prior_index == i)
            imac[mask] = np.searchsorted(thresholds, ima[mask])
            
    return imac
    

def compute_local_background(ima):
    """
    """
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve
    # remove isolated bright pixels
    kernel  = Gaussian2DKernel(3)
    #bkg = convolve(ima, np.ones((11,11))/121)
    #bkg = gaussian_filter(ima, 3)
    bkg = convolve(ima, kernel)
    return bkg
    
    
def process_image(ima, hdr, with_prior=False):
    """
    """
    if hdr['EMGAIN']==0:
        EMgain = 1
    if hdr['EMGAIN']==9000:
        EMgain = 235
    if hdr['EMGAIN']==9200:
        EMgain = 470
    if hdr['EMGAIN']==9400:
        EMgain = 700
    EMgain /= .32 #Accounting for smearing
    ADU2e = .53    
    print('EM gain = %i'%(EMgain))
    print('Conversion gain = %0.2f'%(ADU2e))

    # sub bias line/line
    ima0, _, _ = bias_from_right_overscan(ima)
    # remove hot pixels
    # compute local bkg (for prior estimation)
    if with_prior:
        bkg = compute_local_background(ima0)
        prior_val = bkg/ADU2e/EMgain
    ima0 /= ADU2e
    # thresholding & saturation at 10xprior
    if with_prior:
        ima_count = multiple_photon_counting(ima0, prior=prior_val)
    else:
        ima_count = multiple_photon_counting(ima0)  
    return ima_count

def StackAllImages(xpapoint):
    """
    """
    from astropy.table import Table
    try:
        cat = Table.read(sys.argv[3])
    except:
        print('Impossible to open table, verify your path.')
    for path in np.unique([os.path.dirname(path) for path in cat['PATH']]):
        print(path)
        for gain in np.unique(cat['EMGAIN']):
            for exp in np.unique(cat['EXPTIME']):
                for temp in np.unique(cat['TEMP']).astype(int):
                    print('Stacking images of GAIN = %i, exposure  = %i, temp = %i and path=%s'%(gain, exp, temp, path))
                    files = cat[([path in cati['PATH'] for cati in cat])  & (cat['EMGAIN'] == gain) & (cat['EXPTIME'] == exp) & (cat['TEMP'].astype(int) == temp) ]['PATH']
                    #print(files)
                    if len(files) > 0:
                        print(files)
                        StackImagesPath(files, fname='-Gain%i-Texp%i-Temp%i'%(gain, exp, temp))
    return
    
    


def Normalization2D(amplitude, xo, yo, sigma_x, sigma_y , offset):
    """Given the parameters of a gaussian function, returns the amplitude so that it is normalized in volume 
    """
    B = dblquad(lambda x, y: twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, 0), -float('inf'), +float('inf'), lambda x: -float('inf'), lambda x: float('inf'))       
    return (amplitude/B[0], xo, yo, sigma_x, sigma_y , 0)


def DS9guider(xpapoint):
    """Display on DS9 image from SDSS at the same location if a WCS header is 
    present in the image. If not, it send the image on the astrometry.net server
    and run a lost in space algorithm to have this header. Processing might take a few minutes
    """
    #from astropy.io import fits
    d = DS9(xpapoint)
    filename = getfilename(d)#d.get("file")
    header = d.get_fits()[0].header
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
        d.set("frame last")
        d.set("scale squared")
        
    else:
        print ('Nop header WCS - Applying lost in space algorithm: Internet needed!')
        print ('Processing might take a few minutes ~5-10')
        PathExec = os.path.dirname(os.path.realpath(__file__)) + '/astrometry-net.py'
        Newfilename = filename[:-5] + '_wcs.fits'
        CreateWCS(PathExec, filename, Newfilename)
        filename = d.set("file {}".format(Newfilename))
        filename = getfilename(d)#d.get("file")
        header = d.get_fits()[0].header
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
    """Sends the image on the astrometry.net server
    and run a lost in space algorithm to have this header. 
    Processing might take a few minutes
    """
    import subprocess
    print(filename, Newfilename)
    print(os.path.dirname(filename) + "/--wait.fits")
    start = timeit.default_timer()
    print('''\n\n\n\n      Start lost in space algorithm - might take a few minutes \n\n\n\n''')
    subprocess.check_output("python " + PathExec + " --apikey apfqmasixxbqxngm --newfits " + Newfilename + " --upload " + filename,shell=True)
#    try:
#        os.rename(os.path.dirname(PathExec) + "/--wait", Newfilename)
#    except OSError:
#        os.rename(os.path.dirname(PathExec) + "/--wait.fits", Newfilename)
    stop = timeit.default_timer()
    print('File created')
    print('Lost in space duration = {} seconds'.format(stop-start))
    return


def DS9setup2(xpapoint):
    """This function aims at giving a quick and general visualisation of the image by applying specific thresholding
        and smoothing parameters. This allows to detect easily:
        •Different spot that the image contains
        •The background low frequency/medium variation
        •If a spot saturates
        •If the image contains some ghost/second pass spots. . .
    """
    #from astropy.io import fits
    d = DS9()#DS9(xpapoint)
#    try:
#        region = getregion(d)
#    except ValueError:
#        image_area = [0,2069,1172,2145]
#        Yinf, Ysup,Xinf, Xsup = image_area
#    else:
#        Yinf, Ysup, Xinf, Xsup = Lims_from_region(region)#[131,1973,2212,2562]
#        print(Yinf, Ysup,Xinf, Xsup)
    image_area = [0,2069,1172,2145]
    Yinf, Ysup,Xinf, Xsup = image_area
    fitsimage = d.get_fits()[0].data
    print(fitsimage)
    image = fitsimage[Xinf: Xsup, Yinf: Ysup]
    print(image)
    
#    if d.get("lock bin") == 'no':
    d.set("grid no") 
#        d.set("scale limits {} {} ".format(np.percentile(fitsimage[0].data,9),
#              np.percentile(fitsimage[0].data,99.6)))
    d.set("scale limits {} {} ".format(np.nanpercentile(image,50),np.nanpercentile(image,99.95)))
    d.set("scale asinh")
    d.set("cmap bb")
#        d.set("smooth yes")
#        d.set("cmap Cubehelix0")
#        d.set("smooth radius {}".format(2))
#        d.set("smooth yes")
    d.set("lock bin yes")
#    elif d.get("lock bin") == 'yes':
#       
#        d.set("cmap grey") #d.set("regions delete all")
#        d.set("scale linear")
#        d.set("scale mode minmax")
#        d.set("grid no")
#        d.set("smooth no")
#        d.set("lock bin no")
    return 


def DS9originalSettings(xpapoint):
    """DS9 original settings
    """
    d = DS9(xpapoint)
    d.set("cmap grey") #d.set("regions delete all")
    d.set("scale linear")
    d.set("scale mode minmax")
    d.set("grid no")
    d.set("smooth no")
    d.set("lock bin no")
    return d


def parse_data(data, nan=np.nan, map=map, float=float):
    """Parse DS9 defined region to give pythonic regions
    """
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

def process_region(regions, win):
    """Process DS9 regions to return pythonic regions
    """
    from collections import namedtuple
    processed_regions = []
    for region in regions:
        name, info = region.split('(')
        coords = [float(c) for c in info.split(')')[0].split(',')]
        print(coords)
        if name == 'box':
            xc,yc,w,h,angle = coords
            dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
            X,Y,arr = parse_data(dat)
            box = namedtuple('Box', 'data xc yc w h angle')
            processed_regions.append(box(arr, xc, yc, w, h, angle))
            #return box(arr, xc, yc, w, h, angle)
        elif name == 'bpanda':
            xc, yc, a1, a2, a3, a4,a5, w, h,a6,a7 = coords
            dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
            X,Y,arr = parse_data(dat)
            box = namedtuple('Box', 'data xc yc w h angle')
            processed_regions.append(box(arr, xc, yc, w, h, 0))
            #return box(arr, xc, yc, w, h, 0)
        elif name == 'circle':
            xc,yc,r = coords
            dat = win.get("data physical %s %s %s %s no" % (xc - r, yc - r, 2*r, 2*r))
            X,Y,arr = parse_data(dat)
            Xc,Yc = np.floor(xc), np.floor(yc)
            inside = (X - Xc)**2 + (Y - Yc)**2 <= r**2
            circle = namedtuple('Circle', 'data databox inside xc yc r')
            processed_regions.append(circle(arr[inside], arr, inside, xc, yc, r))
        elif name == '# vector':
            xc, yc, xl, yl = coords
            #dat = win.get("data physical %s %s %s %s no" % (xc - r, yc - r, 2*r, 2*r))
            #X,Y,arr = parse_data(dat)
            #Xc,Yc = np.floor(xc), np.floor(yc)
            #inside = (X - Xc)**2 + (Y - Yc)**2 <= r**2
            vector = namedtuple('Vector', 'data databox inside xc yc r')
            processed_regions.append(vector(xc, yc, xl, yl,0,0))

#return circle(arr[inside], arr, inside, xc, yc, r)
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
            processed_regions.append(annulus(arr[inside], arr, inside, xc, yc, a1, b1, a2, b2, angle))
            #return annulus(arr[inside], arr, inside, xc, yc, a1, b1, a2, b2, angle)
        else:
            raise ValueError("Can't process region %s" % name)
    if len(processed_regions) == 1:
        return processed_regions[0]
    else:
        return processed_regions

def DS9realignImage(xpapoint):
    """
    """
    #import matplotlib; matplotlib.use('TkAgg')  
    #import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    filename = d.get("file")
    try:
        vector = getregion(d)
    except ValueError:
        pass    
    #fitsimage = fits.open(filename)[0]
    fitsimage = d.get_fits()[0]
    image = fitsimage.data
    xs, ys, angle = FindPixelsPositionsOnVector(vector)
    new_im = RecrateArrayFromIndexList(xs, ys, angle, image)
    fitsimage.data = new_im
    name = filename[:-5] + '_reAlighned.fits'
    fitswrite(fitsimage,name)
    d.set('frame new')
    d.set("file {}".format(name))#a = OpenFile(xpaname,filename = filename)

    return
    
    
def FindPixelsPositionsOnVector(vector):
    """
    """
    xc, yc, length, angle = vector.data, vector.databox, vector.inside, vector.xc
    xs = xc + np.linspace(0,length,length) * np.cos(np.pi*angle/180)
    ys = yc + np.linspace(0,length,length) * np.sin(np.pi*angle/180)
    #plt.plot(xs,ys,'x')
    #plt.axis('equal')
    #plt.show()
    return xs, ys, angle

def RecrateArrayFromIndexList(xs, ys, angle, image):
    """
    """
    xs, ys = xs.astype(int), ys.astype(int)
    n1, n2 = xs.min() - (xs.max()-xs.min())/2, 2936 - xs.max() - (xs.max()-xs.min())/2 #int(1500 - (xs.max()-xs.min())/2)
    #a = array[ys[0]-20:ys[0]+20,xs[0]-n:xs[0]+n]
    n1,n2=700,700
    a = image[ys[0],xs[0]-n1:xs[0]+n2]

    for i in range(len(xs)-1):
        a = np.vstack((a,image[ys[i+1],xs[i+1]-n1:xs[i+1]+n2]))
    #imshow(a)
    if (angle > 270.):
        a = a[::-1,:]
        print(angle)
        up = image[ys[0]:,xs[-1]-n1:xs[-1]+n2][::-1,:]
        down = image[:ys[-1],xs[0]-n1:xs[0]+n2][::-1,:]
        new_image = np.vstack((up,a,down))
    elif (angle > 180.) & (angle < 270.):
        print(angle)
        up = image[ys[0]:,xs[0]-n1:xs[0]+n2][::-1,:]
        down = image[:ys[-1],xs[-1]-n1:xs[-1]+n2][::-1,:]
        new_image = np.vstack((up,a,down))  
    #imshow(new_image)
    return new_image
    
    
    


def getregion(win, debug=False, all=False):
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

    #units = rows[2]
    #assert units == 'physical'
    if debug:
        print (rows[4])
        if rows[5:]:
            print('discarding %i regions' % len(rows[5:]) )
    #print(rows[5:])
    if all:
        return process_region(rows[3:], win)
    else:
        return process_region([rows[-1]], win)
#getregion(d, all=True)
        
def create_PA(A=15.45,B=13.75,C=14.95,pas=0.15,nombre=11):
    """Return encoder steps of FB2 tip-tilds focus
    """
    a = np.linspace(A-int(nombre/2)*pas, A+int(nombre/2)*pas, nombre)
    b = np.linspace(B-int(nombre/2)*pas, B+int(nombre/2)*pas, nombre)
    c = np.linspace(C-int(nombre/2)*pas, C+int(nombre/2)*pas, nombre)
    return a[::-1],b[::-1],c[::-1]
#ENCa, ENCb, ENCc = create_PA()

def ENC(x,ENCa):
    """Return encoder step of FB2 tip-tilds focus A
    """
    a = (ENCa[-1]-ENCa[0])/(len(ENCa)-1) * x + ENCa[0]
    #b = (ENCb[10]-ENCb[0])/(10) * x + ENCb[0]
    #c = (ENCc[10]-ENCc[0])/(10) * x + ENCc[0]
    return a#, b, c



def throughfocus(center, files,x=None, 
                 fibersize=0, center_type='barycentre', SigmaMax= 4,Plot=True,
                 Type=None, ENCa_center=None, pas=None, WCS=False, DS9backUp = DS9_BackUp_path):
    """How to use: Open an image of the through focus which is close to the focus.  Click on region. Then click
    precisely on what you think is the centre of the PSF (the size does not matter). Select the region you created
    and press t (throughfocus) or go in analysis menu: throughfocus. This will open a dialog box that asks what
    is the number of the images of the through focus. You can either put the numbers (eg: "10-21") or only press
    enter if the folder in which is the image contains only the images from the throughfocus.
    This will pick up the center of the region you put, compute the barycenter of the image after removing the
    background. Then it keeps this center and for each image of the throughfocus the code computes the radial
    profile (+encircled energy) and fit it by the radial profile of the 2D convolution of a disk with a gaussian. It
    returns the characteristics of the spots and plot their evolution throughout the though focus (fwhm, EE50,
    EE80
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import AnalyzeSpot
    from .focustest import estimateBackground
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
    return fwhm, EE50, EE80


def throughfocusWCS(center, files,x=None, 
                 fibersize=0, center_type='barycentre', SigmaMax= 4,Plot=True,
                 Type=None, ENCa_center=None, pas=None, WCS=False, DS9backUp = DS9_BackUp_path):
    """Same algorithm than throughfocus except it works on WCS coordinate 
    and not on pixels. Then the throughfocus can be run on stars even
    with a sky drift
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import AnalyzeSpot
    from .focustest import estimateBackground
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

    axes[0,1].plot(x,maxpix, '-o')
    axes[0,1].set_ylabel('Max pix')

    axes[1,1].plot(x,sumpix, '-o')
    axes[1,1].set_ylabel('Flux')
    
    axes[2,1].plot(x,varpix, '-o')
    axes[2,1].set_ylabel('Var pix (d=50)')
    axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
    axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
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

    return fwhm, EE50, EE80

def throughfocus2(center, files,x=np.linspace(11.95,14.45,11)[::-1][3:8], 
                 fibersize=100, center_type='barycentre', SigmaMax= 4, box=25):
    """How to use: Open an image of the through focus which is close to the focus.  Click on region. Then click
    precisely on what you think is the centre of the PSF (the size does not matter). Select the region you created
    and press t (throughfocus) or go in analysis menu: throughfocus. This will open a dialog box that asks what
    is the number of the images of the through focus. You can either put the numbers (eg: "10-21") or only press
    enter if the folder in which is the image contains only the images from the throughfocus.
    This will pick up the center of the region you put, compute the barycenter of the image after removing the
    background. Then it keeps this center and for each image of the throughfocus the code computes the radial
    profile (+encircled energy) and fit it by the radial profile of the 2D convolution of a disk with a gaussian. It
    returns the characteristics of the spots and plot their evolution throughout the though focus (fwhm, EE50,
    EE80
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import AnalyzeSpot
    from .focustest import estimateBackground
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
    """How to use: Open an image of the through focus which is close to the focus.  Click on region. Then click
    precisely on what you think is the centre of the PSF (the size does not matter). Select the region you created
    and press t (throughfocus) or go in analysis menu: throughfocus. This will open a dialog box that asks what
    is the number of the images of the through focus. You can either put the numbers (eg: "10-21") or only press
    enter if the folder in which is the image contains only the images from the throughfocus.
    This will pick up the center of the region you put, compute the barycenter of the image after removing the
    background. Then it keeps this center and for each image of the throughfocus the code computes the radial
    profile (+encircled energy) and fit it by the radial profile of the 2D convolution of a disk with a gaussian. It
    returns the characteristics of the spots and plot their evolution throughout the though focus (fwhm, EE50,
    EE80
    """
    from astropy.io import fits
    from .focustest import AnalyzeSpot

    print('''\n\n\n\n      START THROUGHFOCUS \n\n\n\n''')
    d = DS9(xpapoint)
    filename = getfilename(d)#d.get("file ")
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

def hasNumbers(inputString):
    """Tcheck if number in the string
    """
    return any(char.isdigit() for char in inputString)

def DS9rp(xpapoint):
    """How to use: Click on region and select Circle shape (default one). Then click precisely on what you think is
    the centre of the PSF. Select the region you created and press p or go in analysis menu: radial profile.
    The code will:
    •Pick up the center of the region you put
    •Subtract the background by evaluating it in an annulus around the spot
    •Compute the barycenter of the image
    •Compute the radial profile and encircled energy using this new center
    •Return the characteristics of the spot (see image: EE50, FWHM, source size. . . )  and plot the radial
    profile of the spot 
    """
    #from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    import re
    d = DS9(xpapoint)#DS9(xpapoint)
    try:
        entry = sys.argv[3]
    except IndexError:
        entry = ''
    
    if hasNumbers(entry):
        fibersize = float(re.findall(r'\d+','456')[0])
    else:
        fibersize = 0

#    try:
#        fibersize = entry
#    except IndexError:
#        print('No fibersize, Using point source object')
#        fibersize = 0
#    if fibersize == '':
#        fibersize = 0
    #fibersize = 1
    filename =  getfilename(d)#d.get("file ")
    a = getregion(d)
    #fitsfile = fits.open(filename)[0]
    fitsfile = d.get_fits()[0]

    if 'b' in entry.lower():
        center_type = 'barycentre'
    if 'm' in entry.lower():
        center_type = 'maximum'        
    else:
        center_type = None
    center_type
    spot = DS9plot_rp_convolved(data=fitsfile.data,
                                center = [np.int(a.xc),np.int(a.yc)],
                                fibersize=fibersize, center_type=center_type)    
    try:
        plt.title('{} - {} - {}'.format(os.path.basename(filename),[np.int(a.xc),np.int(a.yc)],fitsfile.header['DATE']),y=0.99)
    except KeyError:
        print('No date in header')
        
    plt.show()
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=red"' % (spot['Center'][0]+1,spot['Center'][1]+1,40))#testvincent
    return


def DS9plot_rp_convolved(data, center, size=40, n=1.5, anisotrope=False,
                         angle=30, radius=40, ptype='linear', fit=True, 
                         center_type='barycentre', maxplot=0.013, minplot=-1e-5, 
                         radius_ext=12, platescale=None,fibersize = 100,SigmaMax=4, DS9backUp = DS9_BackUp_path):
  """Function used to plot the radial profile and the encircled energy of a spot,
  Latex is not necessary
  """
  from .focustest import  radial_profile_normalized
  import matplotlib; matplotlib.use('TkAgg')  
  import matplotlib.pyplot as plt
  from .focustest import ConvolveDiskGaus2D
  from .focustest import gausexp
  from scipy.optimize import curve_fit
  from scipy import interpolate
  if anisotrope == True:
      spectral, spatial, EE_spectral, EE_spatial = radial_profile_normalized(data, center, anisotrope=anisotrope, angle=angle, radius=radius, n=n, center_type=center_type)
      spectral = spectral[~np.isnan(spectral)]
      spatial = spatial[~np.isnan(spatial)]

      norm_spatial = spatial[:size]#(spatial[:n] - min(min1,min2)) / np.nansum((spatial[:n] - min(min1,min2) ))
      norm_spectral = spectral[:size]#(spectral[:n] - min(min1,min2)) / np.nansum((spectral[:n] - min(min1,min2) ))              
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
      profile = profile[:size]#(a[:n] - min(a[:n]) ) / np.nansum((a[:n] - min(a[:n]) ))
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
          #profile /= profile.max()
          #SigmaMax = 10
          #popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[profile.max(),fiber,2, np.nanmean(profile)],bounds=([0,0.95*fiber,1,-1],[2,1.05*fiber,SigmaMax,1]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
          popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[profile.max(),fiber,2, np.nanmean(profile)],bounds=([1e-3*profile.max(),0.95*fiber,1,1e-1*profile.mean()],[1e3*profile.max(),1.05*fiber,SigmaMax,1e1*profile.mean()]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
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
      ax1.set_ylim((minplot, np.nanmax([np.nanmax(1.1*(profile)), maxplot])))
      ax2 = ax1.twinx()
      #ax2.plot(np.linspace(0, size, len(norm)), EE[:len(norm)], 'r--x')
      EE_interp = interpolate.interp1d(rsurf[:size], EE[:size],kind='cubic')
      ninterp = 10
      xnew = np.linspace(rsurf[:size].min(),rsurf[:size].max(),ninterp*len(rsurf[:size]))
      ax2.plot(xnew,EE_interp(xnew),linestyle='dotted',c='r')

      
      ax2.plot(rsurf[:size], EE[:size], 'rx')

      mina = min(xnew[EE_interp(xnew)[:ninterp*size]>79])
      minb = min(xnew[EE_interp(xnew)[:ninterp*size]>49])

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
#                  e_gaus = np.nansum(gaus(np.linspace(0,size,100*size),*popt) *2 * np.pi * np.linspace(0,size,100*size)**1)
#                  e_exp = np.nansum(exp(np.linspace(0,size,100*size),*popt) * 2 * np.pi * np.linspace(0,size,100*size)**1)
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
      csvwrite(np.vstack((rmean[:size], profile,ConvolveDiskGaus2D(rmean[:size], *popt))).T, DS9backUp + 'CSVs/%s_EnergyProfile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
      csvwrite(np.vstack((rsurf, EE)).T, DS9backUp + 'CSVs/%s_RadialProfile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
      return d


def DS9open(xpapoint, filename=None):
    """As the OSX version of DS9 does not allow to enter the path of an image when we want to access some data
    I added this possibility. Then you only need to press o (open) so that DS9 opens a dialog box where you can
    enter the path. Then click OK.
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
        print(bcolors.BLACK_RED + 'File not found, please verify your path' + bcolors.END)
        sys.exit()
    return


    
    
    
    

def Charge_path(xpapoint, entry_point = 3):
    """From the entry gave in DS9 (either nothing numbers or beginning-end),
    reuturns the path of the images to take into account in the asked analysis
    """
    #print('salut')
    #sys.quit()
    #from astropy.io import fits
    import re
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
    d = DS9(xpapoint)
    filename = getfilename(d)
    #fitsimage = fits.open(filename)
    fitsimage = d.get_fits()
    number = re.findall(r'\d+',os.path.basename(filename))[-1]
    filen1, filen2 = filename.split(number)
    print(filen1, filen2 )
    
    if fitsimage[0].header['BITPIX'] == -32:
        Type = 'guider'
    else:
        Type = 'detector'
    print ('Type = {}'.format(Type))
    
    try:
        entry = sys.argv[entry_point]#'325-334'# sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# 
        print('Entry 1 = ', entry)
    except IndexError:
        n1=''
        n2='' 
        numbers = ['']
        entry = ''
    if entry == 'all':
        path = glob.glob(os.path.dirname(filename) + '/*.fits')
        print(path)
        return path
    numbers = entry.split('-')
#    if numbers in None:
#        pass
    if len(numbers) == 1:
        try:
            numbers = int(numbers[0])#None
        except ValueError:
            numbers = None
    elif len(numbers) == 2:
        n1, n2 = entry.split('-')
        n1, n2 = int(n1), int(n2)
        #if Type == 'detector':            
        numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
    #elif len(numbers) > 2:
        
    print('Numbers used: {}'.format(numbers))               
    
    path = []
    if Type == 'detector':
        if numbers is None:
            print('Not numbers, taking all the .fits images from the current repository')
            path = glob.glob(os.path.dirname(filename) + '/image*.fits')
            #path = [filename]
        elif type(numbers) == int:
            print('Single number given, using this image.')
            #path = [os.path.dirname(filename) + '/%s%06d%s' % (os.path.basename(filename)[:5],int(numbers),os.path.basename(filename)[11:])]
            path = [filen1 + '%06d'%(numbers) + filen2]
        elif numbers is not None:
            print('Specified numbers are integers, opening corresponding files ...')
            for number in numbers:
#                path.append(os.path.dirname(filename) + '/%s%06d%s' % (os.path.basename(filename)[:5],int(number),os.path.basename(filename)[11:]))
                path.append(filen1 + '%06d'%(int(number)) + filen2)

    
    if Type == 'guider':
        files = glob.glob(os.path.dirname(filename) + '/stack*.fits')
        im_numbers = []
        try:
            for file in files:
                name = os.path.basename(file)
                im_numbers.append(int(name[5:13]))
            im_numbers = np.array(im_numbers)
            print('Files in folder : ', im_numbers)
        except:
            pass
        map
        if numbers is None:
            print('Not numbers, taking all the .fits images from the current repository')
            path = glob.glob(os.path.dirname(filename) + '/stack*.fits')
            #path = [os.path.dirname(filename) + '/%s%06d%s' % (os.path.basename(filename)[:5],int(number),os.path.basename(filename)[11:])]
        elif type(numbers) == int:
            print('Only one number given, running analysis on this image')
            #path = glob.glob(os.path.dirname(filename) + '/stack*.fits')
            path = [filename]
        elif len(numbers) == 2:
            print('Two numbers given, opening files in range ...')
            path = []
            for i, im_number in enumerate(im_numbers):
                print('ok',n1,n2,im_number)
                if (im_number >= n1) & (im_number <= n2):
                    print('yes')
                    path.append(files[i])
                    print(files[i])
        elif len(numbers) > 2:
            print('More than 2 numbers given, opening the corresponding files ...')
            path = []
            numbers = [int(number) for number in numbers]
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
    try:
        a = getregion(d)
        print('Region found')
    except:
        print('Region not found')
        pass
    d.set("frame delete")
    d.set("smooth no")
    for filen in path[:]:
        d.set('frame new')
        d.set("fits {}".format(filen))        
    d.set("lock frame physical")
    d.set("lock scalelimits yes") 
    d.set("lock smooth yes") 
    d.set("lock colorbar yes") 

    try:
        d.set('pan to %0.3f %0.3f physical' % (a.xc,a.yc))
    except:
        pass
    return


def plot_hist2(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,
               ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain,temp, 
               plot_flag=False, ax=None, save=True, DS9backUp = DS9_BackUp_path):
    """Plot the log histogram of the image used to apply thresholding photocounting
    process
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from  scipy import stats
    from astropy.table import Table
    if ax is None:
        fig = plt.figure(figsize=(12,4.5))
        ax = fig.add_subplot(111)
    else:
        save = False
    ax.set_xlabel("Pixel Value [ADU]",fontsize=12)
    ax.set_ylabel("Log10(\# Pixels)",fontsize=12)
    ax.plot(bin_center, n_log, "rx", label="Histogram")
    ax.plot(xgaussfit,np.log10(ygaussfit), "b-", label="Gaussian")
    ax.plot(np.ones(len(n_log))*threshold0,n_log, "b--", label="Bias")
    ax.plot(np.ones(2) * xlinefit.min(), [0.8 * ylinefit.max(), 1.2 * ylinefit.max()], "k--", label="5.5 Sigma")
    ax.plot(np.ones(2) * xlinefit.max(), [0.8 * ylinefit.min(), 1.2 * ylinefit.min()], "k--")
    for k in [1,10]:
        ax.plot(np.ones(2) * (bias + k * sigma), [0,1], "k--")
        
        flux = np.nansum(10**(n_log[(bin_center > bias + k * sigma) & (np.isfinite(n_log))])) / np.nansum(10**(n_log[np.isfinite(n_log)])) 
        ax.text(bias + k * sigma,1.1, 'k=%i \nRONpc=%0.1f \nF=%0.3fe-/p '%(k, 200*(1-stats.norm(0, 1).cdf(k)),flux),
                fontsize=8)
        
    #ax.plot(np.ones(len(n_log))*threshold55, n_log, "k--", label="5.5 Sigma")
    ax.plot(xlinefit,ylinefit, "g--", label="EM gain fit")
    ax.text(.65, .75, 'Bias value = %0.3f DN\nSigma = %0.3f DN\n '
                'EM gain = %0.3f e/e' % (bias, sigma, emgain/np.log(10)),transform=ax.transAxes,
                fontsize=10,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    ax.text(.83, .25, 'Exposure = %i sec\nGain = %i \n '
                'T det = %0.2f C' % (exposure, gain, float(temp)),transform=ax.transAxes,
                fontsize=10,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
    ax.legend(loc="upper right",fontsize=12)   
    ax.grid(b=True, which='major', color='0.75', linestyle='--')
    ax.grid(b=True, which='minor', color='0.75', linestyle='--')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim([0,np.log10(n_bias) + 0.2 ])
#    ax.set_xlim((-1000,10000))
    #ax.set_xlim((bias - 10 * sigma, bias + 100 * sigma))
    ax.set_xlim((bias - 10 * sigma, 1e4))
    #a = np.isfinite(n_log)
    #ax.set_xlim((np.nanpercentile(bin_center[a],0.1),np.nanpercentile(bin_center[a],80)))#([10**4,10**4.3])    
    #ax.set_xlim((bias - 10 * sigma, bias + 100 * sigma))#([10**4,10**4.3]) 
    
    if save:
        if not os.path.exists(os.path.dirname(image) +'/Histograms'):
            os.makedirs(os.path.dirname(image) +'/Histograms')
        plt.savefig(os.path.dirname(image) +'/Histograms/'+ os.path.basename(image).replace('.fits', '.hist.png'), dpi = 100, bbox_inches = 'tight')
    csvwrite(Table(np.vstack((bin_center,n_log)).T), DS9backUp + 'CSVs/%s_Histogramcsv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    #a = Table(np.vstack((bin_center,n_log)).T)
    #print(a)
    #a.write(DS9backUp + 'CSVs/%s_Histogramcsv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")),format='csv')
    return n_log, bin_center 


def apply_pc(image,bias, sigma,area=0, threshold=5.5):
    """Put image pixels to 1 if superior to threshold and 0 else
    """
    cutoff = int(bias + sigma*threshold)#)5.5)
    idx = image > cutoff - 1        
    image[idx] = np.ones(1, dtype = np.uint16)[0]
    image[~idx] = np.zeros(1, dtype = np.uint16)[0]
    return image

def CountCRevent(paths='/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CosmicRaysFree/image000???.CRv.fits'):
    """
    """
    import glob
    from astropy.ito import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt    
    files = glob.glob(paths)
    files.sort()
    for path in files[::-1]:
        print(path)
        with fits.open(path) as f:
            fitsimage = f[0]
            header = f[0].header
            image = f[0].data.copy()
            image[np.isfinite(image)]=0
            image[~np.isfinite(image)]=1
            CS = plt.contour(image, levels=1, colors='white', alpha=0.5)
            try:
                header['N_CR'] = len(CS.allsegs[0]) - 1 
            except IndexError:
                header['N_CR'] = 0
            header['%NaNs'] = 100*np.nansum(image[:,1033:2053]) / (2069*(2053-1033))
            fitsimage.header = header
            fitsimage.writeto(path,overwrite=True)
    return




def getImage(xpapoint):
    """
    """
    #from astropy.io import fits
    d = DS9(xpapoint)
    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    #fitsimage = fits.open(filename)[0]
    fitsimage = d.get_fits()[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]#picouet
    header = fitsimage.header
    return image, header, area
    

def PlotArea3D(xpapoint):
    """
    Plot the image area defined in DS9 in 3D, should add some kind of 2D
    polynomial fit
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #Needed even if not used
    from polynomial import polyfit2d
    image, header, area = getImage(xpapoint)
    X,Y = np.indices(image.shape)
    x, y = image.shape
    x,y = np.meshgrid(np.arange(x),np.arange(y),indexing='ij')
    x, y = x.ravel(), y.ravel()
#    pol = polyfit2d(np.arange(image.shape[1]), np.arange(image.shape[0]), image, [5,5])
    imager = image.ravel()
    coeff = polyfit2d(x, y, imager, [4,4])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, image, c='r', s=0.1)#, cstride=1, alpha=0.2)
    ax.plot_surface(X, Y, popol2D(x, y , coeff).reshape(X.shape) ,rstride=1, cstride=1, shade=True)#,  linewidth=0)
    plt.title('3D plot, area = %s'%(area))
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlim((0.9 * image.min(), 1.1 * image.max()))
    ax.set_zlabel('Pixels ADU value')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    return
    
def popol2D(x, y , coeff):
    fit = 0
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            fit += np.power(x,i) * np.power(y,j) * coeff[i,j] 
    return fit

def ContinuumPhotometry(xpapoint, DS9backUp = DS9_BackUp_path):
    """Fit a gaussian
    interger the flux on 1/e(max-min) and then add the few percent calculated by the gaussian at 1/e 
    """
    #from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from scipy.optimize import curve_fit
    from .focustest import Gaussian
    d = DS9(xpapoint)
    texp = d.get_fits()[0].header['EXPTIME']
    try:
        path = sys.argv[3]
        a = Table.read(path,format='ascii')
    except IndexError:
        a = Table.read(DS9backUp + '/DS9Curves/ds9.dat',format='ascii')
    x, y = a['col1'], a['col2']
    n=0.3
    fit = np.polyfit(x[y<y.mean()+n*np.std(y)], y[y<y.mean()+n*np.std(y)] , 1)
    #xmax = x[np.argmax(y)]
    fit_fn = np.poly1d(fit) 
    popt, pcov = curve_fit(Gaussian, x,y, p0=[y.max()-y.min(), x[y.argmax()], 5, y.min()])
    print(popt)
    plt.figure(figsize=(10,6))
    plt.xlabel('Spatial direction')
    plt.ylabel('ADU mean value')
    plt.plot(x, y ,'--o', label='Data, F=%0.2fADU/sec/col'%(np.nansum(y-fit_fn(x))/texp))
    plt.plot(x, Gaussian(x, *popt) ,'--', label='Gaussian Fit, F = %0.3f'%(np.sum(Gaussian(x, *popt)-popt[-1])))
    limfit =   Gaussian(x, *popt).min() + (Gaussian(x, *popt).max() -  Gaussian(x, *popt).min())/np.exp(1)
    plt.plot(x[y>limfit], y[y>limfit], 'P', color='red', label='Point for flux calculation: F=%i'%(1.1579*np.sum(y[y>limfit]-popt[-1])))
    plt.plot([x.min(), x.max()], np.ones(2) * limfit,'--', c='black', label='1/e limit')
    plt.plot(x, fit_fn(x),label='Fitted background')
    #plt.plot(x, np.ones(len(x))*(y.mean()+n*np.std(y)),'--',c='grey', label='M+%0.2f*sigma'%(n))
    plt.legend()
    plt.show()
    return

    



def DS9photo_counting(xpapoint, save=True):
    """Calculate threshold of the image and apply phot counting
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    filename = d.get("file")
    try:
        region = getregion(d)
    except ValueError:
        image_area = [0,2069,1172,2145]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        image_area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    try:
        threshold = float(sys.argv[3])
    except IndexError:
        threshold = 5.5
    print('Threshold = %0.2f'%(threshold))
    if len(sys.argv) > 4:
        path = Charge_path(xpapoint, entry_point=4)
    else:
        path = [filename]
    if len(path) == 1:
        plot_flag = True
    else:
        plot_flag = False
    print('plot_flag = ', plot_flag)
        
    for filename in path:
        print(filename)
        try:
            fitsimage = fits.open(filename)
        except IOError as e:
            print(bcolors.BLACK_RED + 'FILE NOT FOUND: ' + bcolors.END, e)
        else:
            image = fitsimage[0].data
            #emgain,bias,sigma,amp,slope,intercept = calc_emgain(image,area=image_area,plot_flag=True)
            print(type(filename))
            a, b = calc_emgainGillian(filename,area=image_area,plot_flag=plot_flag)
            print(len(a),len(b))
            (emgain,bias,sigma,frac_lost) = a
            #image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp = b
            try:    
                a = plot_hist2(*b, ax=None,plot_flag=plot_flag)
            except TypeError:
                pass
            else:
                if plot_flag:
                    plt.show()
                if save: 
                    new_image = apply_pc(image=image, bias=bias, sigma=sigma ,area=0, threshold=threshold)
                    print (new_image.shape)
                    fitsimage[0].data = new_image
#                    if not os.path.exists(os.path.dirname(filename) +'/Thresholded_images'):
#                        os.makedirs(os.path.dirname(filename) +'/Thresholded_images')
                    name = os.path.dirname(filename) +'/Thresholded_images/'+ os.path.basename(filename)[:-5] + '_THRES.fits'
                    #name = '/tmp/test_pc.fits'
                    fitswrite(fitsimage[0], name)
                    if len(path) == 1:
                        d.set('frame new')
                        d.set('file ' + name)  
    return a

def fitswrite(fitsimage, filename):
    """
    """
    if len(filename)==0:
        print('Impossible to save image in filename %s, saving it to /tmp/image.fits'%(filename))
        filename = '/tmp/image.fits'
        fitsimage.writeto(filename,overwrite=True) 
    if 'NAXIS3' in fitsimage.header:
        print('2D array: Removing NAXIS3 from header...')
        fitsimage.header.remove('NAXIS3')
    elif not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if not os.path.exists(os.path.dirname(filename)):
        print('%s not existing: creating folde...'%(os.path.dirname(filename)))
        os.makedirs(os.path.dirname(filename))
    try:
        fitsimage.writeto(filename, overwrite=True)
    except IOError:
        print(bcolors.BLACK_RED + 'Can not write in this repository : ' + filename + bcolors.END)
        filename = '/tmp/' + os.path.basename(filename)
        print(bcolors.BLACK_RED + 'Instead writing new file in : ' + filename + bcolors.END)
        fitsimage.writeto(filename,overwrite=True) 
    print('Image saved: %s'%(filename))
    return filename



def csvwrite(table, filename):
    """
    """
    from astropy.table import Table
    if type(table) == np.ndarray:
        table = Table(table)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    try:                                          
        table.write(filename, overwrite=True, format='csv')
    except UnicodeDecodeError:
        print('UnicodeDecodeError: you should consider change the name of the file/folder...')
        reload(sys); sys.setdefaultencoding('utf8')
    try:
        table.write(filename, overwrite=True, format='csv')
    except IOError:
        print(bcolors.BLACK_RED + 'Can not write in this repository : ' + filename + bcolors.END)
        filename = '/tmp/' + os.path.basename(filename)
        print(bcolors.BLACK_RED + 'Instead writing new file in : ' + filename + bcolors.END)
        table.write(filename, overwrite=True, format='csv')
    print('Table saved: %s'%(filename))
    return table

def gaussian(x, amp, x0, sigma):
    """Gaussian funtion
    """
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussianFit(x, y, param):
    """Fit Gaussian funtion
    """
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(gaussian, x, y, p0=param)   
    amp, x0, sigma = popt   
    return (amp, x0, sigma) 


def linefit(x, A, B):
    """
    """
    return A*x + B


def fitLine(x, y, param=None):
    """Fit line
    """
    from scipy.optimize import curve_fit
    try:
        popt, pcov = curve_fit(linefit, x, y, p0 = param)
    except TypeError as e:
        print(e)
        return (1,1)
    else:
        a, b = popt
        return (a, b)

def ind2sub(array_shape, ind):
    """?
    """
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def DS9next(xpapoint):
    """Load the next alphanumerical image in DS9
    """
    d = DS9(xpapoint)
    filename = d.get("file")
    files = glob.glob(os.path.dirname(filename) + '/*.fits')
    files.sort()
    index = files.index(filename)
    print(files,filename,index)
    d.set('tile no')
    try:
        d.set('frame new')
        d.set("file {}".format(files[index+1]))
    except IndexError:
        print('No more files')
        sys.exit()
    return
                         
def DS9previous():
    """Still to be written, load previous image in DS9
    """
    return                         
                         
def create_multiImage_old(xpapoint, w=None, n=30, rapport=1.8, continuum=False):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """
    from astropy.table import Table
    #from astropy.io import fits
    from .mapping import Mapping
    line = sys.argv[3]#'f3 names'#sys.argv[3]
    print('Entry = ', line)
    line = line.lower()
    if '202' in line:
        w = 0.20255
    if '206' in line:
        w = 0.20619
    if '213' in line:
        w = 0.21382
    if 'lya' in line:
        w = None        
    print('Selected Line is : ', w)

    try:
        slit_dir = resource_filename('DS9FireBall', 'Slits')
        Target_dir = resource_filename('DS9FireBall', 'Slits')
        Mapping_dir = resource_filename('DS9FireBall', 'Mappings')
    except:
        slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        Target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Targets')
        Mapping_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mappings')
        
        
    if ('f1' in line) or ('119' in line):
        csvfile = os.path.join(slit_dir,'F1_119.csv')
        targetfile = os.path.join(Target_dir,'targets_F1.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F1_p2.pkl')
    if ('f2' in line) or ('161' in line):
        csvfile = os.path.join(slit_dir,'F2_-161.csv')
        targetfile = os.path.join(Target_dir,'targets_F2.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F2_p2.pkl')
    if ('f3' in line) or ('121' in line):
        csvfile = os.path.join(slit_dir,'F3_-121.csv')
        targetfile = os.path.join(Target_dir,'targets_F3.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F3_p2.pkl')
    if ('f4' in line) or ('159' in line):
        csvfile = os.path.join(slit_dir,'F4_159.csv')
        targetfile = os.path.join(Target_dir,'targets_F4.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F4_p2.pkl')
    print('Selected field in : ', csvfile)
        
    mapping = Mapping(filename=mappingfile)
    d = DS9(xpapoint)
    #filename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data
    try:
        table = Table.read(csvfile)
        taget_table = Table.read(targetfile, format='ascii')#, delimiter='\t'
        #print(taget_table)
    except IOError:
        print('No csv table found, Trying fits table')
        try:
#            table = Table.read(filename[:-5] + '_table.fits')
            table = Table.read(csvfile)
        except IOError:
            print('No fits table found, Please run focustest')
            sys.exit() 
    if w is None:
        table = table [(table['wavelength'] != 0.20255) & (table['wavelength'] != 0.20619) & (table['wavelength'] != 0.21382)
        & (table['wavelength'] != 0.0) & (table['wavelength'] != -1.0)]
    else:
        table = table [table['wavelength'] == w]
    
    xm, ym, redshift, slit = taget_table['xmask'], taget_table['ymask'], taget_table['Z'], taget_table['Internal-count']
    #print()#0.20619, 0.21382
    if w is None:
        print(1e-4*(1+redshift[redshift<0.8])*1215.67, xm, ym)
        y,x = mapping.map(1e-4*(1+redshift[redshift<0.8])*1215.67, xm[redshift<0.8], ym[redshift<0.8], inverse=False)
    else:
        y,x = mapping.map(w, xm, ym, inverse=False)
    #x, y, slit = table['X_IMAGE'], table['Y_IMAGE'], table['Internal-count']
    n1, n2 = n, n 
    print('n1,n2 = ',n1,n2)
    redshift = redshift.tolist()
    sliti=[]
    redshifti=[]
    imagettes=[]
    xi=[]
    yi=[]
    for i in range(len(x)):
        if (y[i]>1053) & (y[i]<2133) & (x[i]>0) & (x[i]<2070):
            imagettes.append(image[int(x[i])-n1:int(x[i]) +n1,int(y[i])-n2:int(y[i]) +n2])
            redshifti.append(redshift[i])
            sliti.append(slit[i])
            xi.append(x[i]);yi.append(y[i])
            print(y[i],x[i])
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
        new_image = np.ones((v1*(2*n) + v1,v2*(2*n) + v2))*np.nanmin(imagettes[0])
    except ValueError:
        print ('No matching in the catalog, please run focustest before using this function')
        #sys.exit()
    for index,imagette in enumerate(imagettes):
        j,i = index%v2,index//v2
        centrei, centrej = 1 + (2*i+1) * n,1 + (2*j+1) * n
        #print (i,j)
        #print (centrei,centrej)
        try:
            new_image[centrei-n:centrei+n,centrej-n:centrej+n] = imagette
        except:
            pass
    new_image[1:-1:2*n, :] = np.nanmax(np.array(imagettes[0]))
    new_image[:,1:-1:2*n] = np.nanmax(np.array(imagettes[0]))
    if continuum:
        new_image[0:-2:2*n, :] = np.nanmax(np.array(imagettes))
        new_image[:,0:-2:4*n] = np.nanmax(np.array(imagettes))
    fitsfile[0].data = new_image[::-1, :]
    fitswrite(fitsfile, '/tmp/imagettes.fits')
    d.set('frame new')
    d.set("file /tmp/imagettes.fits")
    d.set('scale mode minmax')
    return


def create_multiImage(xpapoint, w=None, n=30, rapport=1.8, continuum=False):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """
    #from astropy.io import fits
    line = sys.argv[3]#'f3 names'#sys.argv[3]
    print('Entry = ', line)
    d = DS9(xpapoint)
    #filename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    
    x, y, redshift, slit, w = returnXY(line)  
    
    n1, n2 = n, n 
    print('n1,n2 = ',n1,n2)
    redshift = redshift.tolist()
    sliti=[]
    redshifti=[]
    imagettes=[]
    xi=[]
    yi=[]
    for i in range(len(x)):
        if (y[i]>1053) & (y[i]<2133) & (x[i]>0+n1) & (x[i]<2070-n1):
            imagettes.append(image[int(x[i])-n1:int(x[i]) +n1,int(y[i])-n2:int(y[i]) +n2])
            redshifti.append(redshift[i])
            sliti.append(slit[i])
            xi.append(x[i]);yi.append(y[i])
            print(y[i],x[i])
    v1,v2 = 6,14
    try:
        new_image = np.ones((v1*(2*n) + v1,v2*(2*n) + v2))*np.nanmin(imagettes[0])
    except ValueError:
        print ('No matching in the catalog, please run focustest before using this function')
        #sys.exit()
    for index,imagette in enumerate(imagettes):
        j,i = index%v2,index//v2
        centrei, centrej = 1 + (2*i+1) * n,1 + (2*j+1) * n
        #print (i,j)
        #print (centrei,centrej)
        try:
            new_image[centrei-n:centrei+n,centrej-n:centrej+n] = imagette
        except:
            pass
    new_image[1:-1:2*n, :] = np.nanmax(np.array(imagettes[0]))
    new_image[:,1:-1:2*n] = np.nanmax(np.array(imagettes[0]))
    if continuum:
        new_image[0:-2:2*n, :] = np.nanmax(np.array(imagettes))
        new_image[:,0:-2:4*n] = np.nanmax(np.array(imagettes))
    fitsfile[0].data = new_image[::-1, :]   
    fitswrite(fitsfile,'/tmp/imagettes.fits')
    
    d.set('frame new')
    d.set("file /tmp/imagettes.fits")
    d.set('scale mode minmax')
    return


def check_if_module_exists(module):
    if sys.version_info.major == 3:
        if sys.version_info.minor > 3:
            from importlib.util import find_spec as check_module
            spam_spec = check_module(module)
            found = spam_spec is not None
        else:
            from importlib import find_loader as check_module
            spam_loader = check_module(module)
            found = spam_loader is not None
        return found
    elif sys.version_info.major == 2:
        from pkgutil import find_loader as check_module    
        eggs_loader = check_module(module)
        found = eggs_loader is not None
        return found
    
class bcolors:
    BLACK_RED = '\x1b[4;30;41m' 
    GREEN_WHITE = '\x1b[0;32;47m' 
    BLACK_GREEN = '\x1b[0;30;42m' 
    END = '\x1b[0m'



def returnXY(field, w = 2060, frame='observed', keyword='lya'):
    """Return redshift, position of the slit, wavelength used for each mask
    given the DS9 entry
    """
    from astropy.table import Table
#    try:
    from .mapping import Mapping
#    except ValueError:
#        from Calibration.mapping import Mapping
    field = field.lower()
    if (w == 'lya') or (w == 'Lya') or (w == 'Lya'):
        w = 1215.67
    w = float(w)
    w *= 1e-4
    print('Selected Line is : %0.4f microns'%( w))

    try:
        #slit_dir = resource_filename('DS9FireBall', 'Slits')
        Target_dir = resource_filename('DS9FireBall', 'Targets')
        Mapping_dir = resource_filename('DS9FireBall', 'Mappings')
    except:
        #slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        Target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Targets')
        Mapping_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mappings')
        
        
    if ('f1' in field) or ('119' in field):
        #csvfile = os.path.join(slit_dir,'F1_119.csv')
        targetfile = os.path.join(Target_dir,'targets_F1.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F1.pkl')#mapping-mask-det-180612-F1.pkl
    if ('f2' in field) or ('161' in field):
        #csvfile = os.path.join(slit_dir,'F2_-161.csv')
        targetfile = os.path.join(Target_dir,'targets_F2.csv')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F2.pkl')
    if ('f3' in field) or ('121' in field):
        #csvfile = os.path.join(slit_dir,'F3_-121.csv')
        targetfile = os.path.join(Target_dir,'targets_F3.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F3.pkl')
    if ('f4' in field) or ('159' in field):
        #csvfile = os.path.join(slit_dir,'F4_159.csv')
        targetfile = os.path.join(Target_dir,'targets_F4.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F4.pkl')
    #print('Selected field in : ', csvfile)
    print('targetfile = ',targetfile)  
    mapping = Mapping(filename=mappingfile)

    try:
        target_table = Table.read(targetfile)#, format='ascii')
    except:
        target_table = Table.read(targetfile, format='ascii', delimiter='\t')

    if 'f1' in field:
        idok = (target_table['slit_length_right'] != 0) &  (target_table['slit_length_left'] !=0)
        target_table = target_table[idok]
        xmask = target_table['xmask'] + (target_table['slit_length_right'] - target_table['slit_length_left'])/2.
        ymask = target_table['ymask'] + target_table['offset']
        z = target_table['z'] 
        internalCount = target_table['Internal-count']

    else:
        xmask = target_table['xmm']
        ymask = target_table['ymm']
        internalCount = target_table['Internal-count']
        z = target_table['Z'] 
        mag = target_table['mgNUVMAG_L'] 
    redshift = z
    slit = internalCount
    
    if frame.lower() == 'rest':
        print('Working in rest frame wevelength' )
        #w = 1215.67
        wavelength = (1 + redshift) * w #* 1e-4
        y,x = mapping.map(wavelength, xmask, ymask, inverse=False)
    if frame.lower() == 'observed':
        print('Working in observed frame wevelength' )
        y,x = mapping.map(w, xmask, ymask, inverse=False)
    w *= 1e4
        
    print(x[0],y[0])
    if keyword is None:
        return x, y, redshift, slit, w  
    elif keyword.lower() == 'lya':
        index = [len(sliti)<3 for sliti in slit]
        x, y, redshift, slit, mag, w = x[index], y[index], redshift[index], slit[index], mag[index], w
       # index = [magi<np.median(mag) for magi in mag]
        index = [magi>np.median(mag) for magi in mag]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'qso':
        index = ['qso' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'lyc':
        index = ['lyc' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'ovi':
        index = ['ovi' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index] 



def returnXY_old(line, w = 0.206, keyword='lya'):
    """Return redshift, position of the slit, wavelength used for each mask
    given the DS9 entry
    """
    from astropy.table import Table
#    try:
    from .mapping import Mapping
#    except ValueError:
#        from Calibration.mapping import Mapping
    line = line.lower()
    if '202' in line:
        w = 0.20255
    if '206' in line:
        w = 0.20619
    if '213' in line:
        w = 0.21382
    if 'lya' in line:
        w = None
    else:
        w *= 1e-3
    print('Selected Line is : ', w)

    try:
        #slit_dir = resource_filename('DS9FireBall', 'Slits')
        Target_dir = resource_filename('DS9FireBall', 'Targets')
        Mapping_dir = resource_filename('DS9FireBall', 'Mappings')
    except:
        #slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        Target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Targets')
        Mapping_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mappings')
        
        
    if ('f1' in line) or ('119' in line):
        #csvfile = os.path.join(slit_dir,'F1_119.csv')
        targetfile = os.path.join(Target_dir,'targets_F1.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F1.pkl')#mapping-mask-det-180612-F1.pkl
    if ('f2' in line) or ('161' in line):
        #csvfile = os.path.join(slit_dir,'F2_-161.csv')
        targetfile = os.path.join(Target_dir,'targets_F2.csv')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F2.pkl')
    if ('f3' in line) or ('121' in line):
        #csvfile = os.path.join(slit_dir,'F3_-121.csv')
        targetfile = os.path.join(Target_dir,'targets_F3.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F3.pkl')
    if ('f4' in line) or ('159' in line):
        #csvfile = os.path.join(slit_dir,'F4_159.csv')
        targetfile = os.path.join(Target_dir,'targets_F4.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F4.pkl')
    #print('Selected field in : ', csvfile)
    print('targetfile = ',targetfile)  
    mapping = Mapping(filename=mappingfile)

    try:
        target_table = Table.read(targetfile)#, format='ascii')
    except:
        target_table = Table.read(targetfile, format='ascii', delimiter='\t')

#        try:
##            table = Table.read(filename[:-5] + '_table.fits')
#            table = Table.read(csvfile)
#        except IOError:
#            print('No fits table found, Please run focustest')
#            sys.exit() 
#    if w is None:
#        table = table [(table['wavelength'] != 0.20255) & (table['wavelength'] != 0.20619) & (table['wavelength'] != 0.21382)
#        & (table['wavelength'] != 0.0) & (table['wavelength'] != -1.0)]
#    else:
#        table = table[table['wavelength'] == w]

    if 'f1' in line:
        idok = (target_table['slit_length_right'] != 0) &  (target_table['slit_length_left'] !=0)
        target_table = target_table[idok]
        xmask = target_table['xmask'] + (target_table['slit_length_right'] - target_table['slit_length_left'])/2.
        ymask = target_table['ymask'] + target_table['offset']
        z = target_table['z'] 
        internalCount = target_table['Internal-count']

    else:
        xmask = target_table['xmm']
        ymask = target_table['ymm']
        internalCount = target_table['Internal-count']
        z = target_table['Z'] 
        mag = target_table['mgNUVMAG_L'] 
        #mag = target_table['mgNUVMAG_L'] 
    redshift = z
    slit = internalCount
    #xm, ym, redshift, slit = taget_table['xmask'], taget_table['ymask'], taget_table['Z'], taget_table['Internal-count']
    #print()#0.20619, 0.21382
    
    
    if 'lya' in line:
        print('Lya given' )
#        y,x = mapping.map(0.20255, xmask, ymask, inverse=False)
#        print(x[0],y[0])
#        w = 1215.67
#        y -= ((1 + redshift) * w - 2025.5 )*46.6/10   #((1 + redshift) * w) 
#        print(x[0],y[0])
        w = 1215.67
        #print(redshift)
        wavelength = (1 + redshift) * w * 1e-4
        y,x = mapping.map(wavelength, xmask, ymask, inverse=False)
    else:
        y,x = mapping.map(w, xmask, ymask, inverse=False)
        w *= 1e4
    print(x[0],y[0])
    if keyword is None:
        return x, y, redshift, slit, w  
    elif keyword.lower() == 'lya':
        index = [len(sliti)<3 for sliti in slit]
        x, y, redshift, slit, mag, w = x[index], y[index], redshift[index], slit[index], mag[index], w
       # index = [magi<np.median(mag) for magi in mag]
        index = [magi>np.median(mag) for magi in mag]
        return x[index], y[index], redshift[index], slit[index], w#[index]  

#        return x[index], y[index], redshift[index], slit[index], w#[index] 
    elif keyword.lower() == 'qso':
        index = ['qso' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'lyc':
        index = ['lyc' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'ovi':
        index = ['ovi' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
        



def DS9plot_spectra_Observed_Frame(xpapoint, w=None, n=30, rapport=1.8, continuum=False, DS9backUp = DS9_BackUp_path):
    """Plot spectra in local frame 
    """
    import matplotlib.pyplot as plt
    from .focustest import create_DS9regions2
    #from astropy.io import fits
    line = sys.argv[3]#'f3 names'#sys.argv[3]
    print('Entry = ', line)
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    
    field, w =  line.split('-')
    x, y, redshift, slit, w = returnXY(field, w=w, frame='Observed')  

    create_DS9regions2(y,x, radius=10, form = 'box',save=True,color = 'yellow', savename='/tmp/centers')
    print('x = ', x)
    print('y = ', y)

    try:
        region = getregion(d)
    except:
        pass
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        mask = (y>Xinf) & (y<Xsup) & (x>Yinf) & (x<Ysup)
        print(mask)
        x, y, redshift, slit = x[mask], y[mask], redshift[mask], slit[mask]
        print('Selected objects are : ', slit)
    
    
    n1, n2 = int(n/4), 700 
    print('n1,n2 = ',n1,n2)
    redshift = redshift.tolist()

    lambdasup = []
    lambdainf = []
    sup = 2100#2133
    inf = 1100#1053
    x2w = 10./46
    flag_visible = (y>inf) & (y<sup) & (x>0) & (x<2070)

    sliti = np.array(slit)[flag_visible]
    redshifti = np.array(redshift)[flag_visible]
    xi=np.array(x)[flag_visible]
    yi=np.array(y)[flag_visible]
    imagettes = []
    for i in range(len(xi)):
        imagettes.append(image[int(xi[i])-n1:int(xi[i]) +n1,int(yi[i])-n2:int(yi[i]) +n2])
        lambdainf.append(  - (sup - yi[i]) * x2w + w)
        lambdasup.append( - (inf - yi[i]) * x2w + w)
    #print(np.array(lambdainf)-np.array(lambdasup))
#    for i in range(len(x)):
#        if (y[i]>inf) & (y[i]<sup) & (x[i]>0) & (x[i]<2070):
#            imagettes.append(image[int(x[i])-n1:int(x[i]) +n1,int(y[i])-n2:int(y[i]) +n2])
#            redshifti.append(redshift[i])
#            sliti.append(slit[i])
#            xi.append(x[i]);yi.append(y[i])
#            lambdainf.append(  (inf - xi) * x2w + w)
#            lambdasup.append(  (sup - xi) * x2w + w)
            #boundsup.append()
    v1,v2 = int(len(xi)/2+1),2
    print('v1,v2=',v1,v2)
    #fig, axes = plt.subplots(v1, v2, figsize=(18,50),sharex=True)
    fig, axes = plt.subplots(v1, v2, figsize=(12.8,70/40*v1),sharex=True, sharey=True)
    xaxis = np.linspace(w-n2*x2w, w+n2*x2w, 2*n2)
    xinf = np.searchsorted(xaxis,lambdainf)
    xsup = np.searchsorted(xaxis,lambdasup)

    fig.suptitle('Spectra, lambda in Oberserved frame',y=1)
    spectras = []
    for i, ax in enumerate(axes.ravel()[1:len(imagettes)+1]): 
        spectra = imagettes[i][:, ::-1].mean(axis=0)
        spectra[:xinf[i]] = np.nan
        spectra[xsup[i]:] = np.nan
        spectras.append(spectra)
        ax.step(xaxis,spectra,label = 'Slit: ' + sliti[i] +'\nz = %0.2f'%(redshifti[i])+'\nx,y = %i - %i'%(yi[i],xi[i]))
        ax.axvline(x=lambdainf[i],color='black',linestyle='dotted')
        ax.axvline(x=lambdasup[i],color='black',linestyle='dotted')
        ax.legend()
        ax.set_xlim(xaxis[[0,-1]])
        ax.tick_params(labelbottom=True)
    ax = axes.ravel()[0]
    print(np.array(imagettes).shape)
    stack = np.nanmean(np.array(spectras),axis=0)
    ax.step(xaxis,stack,label = 'Stack',c='orange')  
    ax.legend()
    ax.set_xlim(xaxis[[0,-1]])
    ax.tick_params(labelbottom=True)
    for ax in axes[-1,:]:
        ax.set_xlabel('Wavelength [A] in observed frame')
    fig.tight_layout()
    fig.savefig(filename[:-5] + '_Spectras.png')
    if v1>12:
        ScrollableWindow(fig)
    else:
        plt.show()
    csvwrite(np.vstack((xaxis,stack)).T, DS9backUp + 'CSVs/%s_SpectraBigRange.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )

    return imagettes



def DS9plot_spectra_Rest_Frame(xpapoint, w=None, n=30, rapport=1.8, continuum=False, DS9backUp = DS9_BackUp_path):
    """Plot spectra in local frame 
    """
    import matplotlib.pyplot as plt
    from .focustest import create_DS9regions2
    #from astropy.io import fits
    line = sys.argv[3]#'f3 names'#sys.argv[3]
    print('Entry = ', line)
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    if '-' in line:
        field, w =  line.split('-')
        x, y, redshift, slit, w = returnXY(field, w=w, frame='rest')  
    else:
        x, y, redshift, slit, w = returnXY(field, w='lya', frame='rest') 
    #x, y = x+93, y-93
    create_DS9regions2(y,x, radius=10, form = 'box',save=True,color = 'yellow', savename='/tmp/centers')
    #d.set('regions /tmp/centers.reg')
    print('x = ', x)
    print('y = ', y)
    #print('redshift = ', redshift)
    #print('slit = ', slit)
    try:
        region = getregion(d)
    except:
        pass
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        mask = (y>Xinf) & (y<Xsup) & (x>Yinf) & (x<Ysup)
        print(mask)
        x, y, redshift, slit = x[mask], y[mask], redshift[mask], slit[mask]
        print('Selected objects are : ', slit)
    
    
    n1, n2 = int(n/4), 700 
    print('n1,n2 = ',n1,n2)
    redshift = redshift.tolist()

    lambdasup = []
    lambdainf = []
    sup = 2100#2133
    inf = 1100#1053
    x2w = 10./46
    lya_z = 0.7
    flag_visible = (y>inf) & (y<sup) & (x>0) & (x<2070)

    sliti = np.array(slit)[flag_visible]
    redshifti = np.array(redshift)[flag_visible]
    xi=np.array(x)[flag_visible]
    yi=np.array(y)[flag_visible]
    imagettes = []
    for i in range(len(xi)):
        imagettes.append(image[int(xi[i])-n1:int(xi[i]) +n1,int(yi[i])-n2:int(yi[i]) +n2])
        lambdainf.append(  - (sup - yi[i]) * x2w / (1*redshift[i]+1) + w)
        lambdasup.append( - (inf - yi[i]) * x2w / (1*redshift[i]+1) + w)
    #print(np.array(lambdainf)-np.array(lambdasup))
#    for i in range(len(x)):
#        if (y[i]>inf) & (y[i]<sup) & (x[i]>0) & (x[i]<2070):
#            imagettes.append(image[int(x[i])-n1:int(x[i]) +n1,int(y[i])-n2:int(y[i]) +n2])
#            redshifti.append(redshift[i])
#            sliti.append(slit[i])
#            xi.append(x[i]);yi.append(y[i])
#            lambdainf.append(  (inf - xi) * x2w + w)
#            lambdasup.append(  (sup - xi) * x2w + w)
            #boundsup.append()
    v1,v2 = int(len(xi)/2+1),2
    print('v1,v2=',v1,v2)
    #fig, axes = plt.subplots(v1, v2, figsize=(18,50),sharex=True)
    fig, axes = plt.subplots(v1, v2, figsize=(12.8,70/40*v1),sharex=True, sharey=True)
    xaxis = np.linspace(w-n2*x2w/(1+lya_z), w+n2*x2w/(1+lya_z), 2*n2)
    xinf = np.searchsorted(xaxis,lambdainf)
    xsup = np.searchsorted(xaxis,lambdasup)

    fig.suptitle('Spectra, lambda in rest frame' ,y=1)
    spectras = []
    for i, ax in enumerate(axes.ravel()[1:len(imagettes)+1]): 
        spectra = imagettes[i][:, ::-1].mean(axis=0)
        spectra[:xinf[i]] = np.nan
        spectra[xsup[i]:] = np.nan
        spectras.append(spectra)
        ax.step(xaxis,spectra,
                label = 'Slit: ' + sliti[i] +'\nz = %0.2f'%(redshifti[i])+'\nx,y = %i - %i'%(yi[i],xi[i]))
        ax.axvline(x=lambdainf[i],color='black',linestyle='dotted')
        ax.axvline(x=lambdasup[i],color='black',linestyle='dotted')
        ax.legend()
        ax.set_xlim(xaxis[[0,-1]])
        ax.tick_params(labelbottom=True)
    ax = axes.ravel()[0]
    print(np.array(imagettes).shape)
    stack = np.nanmean(np.array(spectras),axis=0)
    ax.step(xaxis,stack,label = 'Stack',c='orange')  
    ax.legend()
    ax.set_xlim(xaxis[[0,-1]])
    ax.tick_params(labelbottom=True)
    for ax in axes[-1,:]:
        ax.set_xlabel('Wavelength [A] rest frame')
    fig.tight_layout()
    fig.savefig(filename[:-5] + '_Spectras.png')
    if v1>12:
        ScrollableWindow(fig)
    else:
        plt.show()
    csvwrite(np.vstack((xaxis,stack)).T, DS9backUp + 'CSVs/%s_SpectraBigRange.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )

    return imagettes




#def DS9plot_all_spectra(xpapoint, w=None, n=30, rapport=1.8, continuum=False, DS9backUp = DS9_BackUp_path):
#    """Plot spectra in local frame
#    """
#    import matplotlib.pyplot as plt
#    #from astropy.io import fits
#    line = sys.argv[3]#'f3 names'#sys.argv[3]
#    print('Entry = ', line)
#    d = DS9(xpapoint)
#    #filename = d.get("file")
#    fitsfile = d.get_fits()#fits.open(filename)
#    image = fitsfile[0].data        
#    x, y, redshift, slit, w = returnXY(line)  
#    #x, y = x - 93, y + 92
#    n1, n2 = 1, 500 
#    print('n1,n2 = ',n1,n2)
#    redshift = redshift.tolist()
#    
#    imagettes = [image[int(xi)-n1:int(xi) +n1,1500-n2:1500 +n2] for xi,yi in zip(x,y)]
#
#    v1,v2 = int(len(redshift)/2+1),2
#    print('v1,v2=',v1,v2)
#    #fig, axes = plt.subplots(v1, v2, figsize=(18,50),sharex=True)
#    fig, axes = plt.subplots(v1, v2, figsize=(12.8,70), sharex=True, sharey=True)
#    fig.suptitle("Spectra",y=1)
#    #xaxis = np.linspace(w-n2*(10./46), w+n2*(10./46), 2*n2)
#    xaxis = np.arange(1500-n2,1500+n2)
#    print(len(imagettes),len(x),len(redshift))
#    for i, ax in enumerate(axes.ravel()): 
#        try:
#            #print(i)
#            ax.step(xaxis,imagettes[i][:, ::-1].mean(axis=0),
#                    label = 'Slit: ' + slit[i] )#+'\nz = %0.2f'%(redshift[i])+'\nx,y = %i - %i'%(1500,x[i]))
#            ax.legend()
#            ax.tick_params(labelbottom=True)
#        except IndexError:
#            pass
#    for ax in axes[-1,:]:
#        ax.set_xlabel('Wavelength [pixels] ({:.2f} A/pix)'.format(10./46))
#    fig.tight_layout()
#    try:
#        ScrollableWindow(fig)
#    except:
#        print ('Impossible to run ScrollableWindow')    
#    print(len(imagettes))
#    return imagettes


def DS9tsuite(xpapoint):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """

    path = os.path.dirname(os.path.realpath(__file__))    
    d = DS9(xpapoint)
    d.set('frame delete all')
    #d.set('frame new')
    print('''\n\n\n\n      TEST: diffuse focus test analysis   \n\n\n\n''') 
    d.set('frame delete all')
    sys.argv.append('');sys.argv.append('');sys.argv.append('');sys.argv.append('')

    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
    sys.argv[3] = 'f3'
    DS9focus(xpapoint)
    
    print('''\n\n\n\n      TEST: Open    \n\n\n\n''')
    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')

    print('''\n\n\n\n      TEST: Setup   \n\n\n\n''')
    DS9setup2(xpapoint)


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
    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
    sys.argv[3] = ''
    DS9visualisation_throughfocus(xpapoint)
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
    sys.argv[3] = '18405298-18407298'
    DS9visualisation_throughfocus(xpapoint)
    DS9stack(xpapoint)    
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
    sys.argv[3] = '18405298-18405946-18407582'
    DS9visualisation_throughfocus(xpapoint)
    DS9stack(xpapoint)    

    print('''\n\n\n\n      TEST: Next Guider   \n\n\n\n''')
    d.set('frame delete all')
    #d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
    print('''\n\n\n\n      TEST: Throughfocus Guider   \n\n\n\n''')
    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (812,783.2,40))
    d.set('regions select all') 
    sys.argv[3] = ''
    DS9throughfocus(xpapoint)
    sys.argv[3] = '18405298-18407298'
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
    #DS9rp(xpapoint)  
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
    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
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
    d.set('frame delete all')
    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
    sys.argv[3] = 'f3'
    DS9focus(xpapoint)

    print('''\n\n\n\n      TEST: Imagette lya   \n\n\n\n''') 
    sys.argv[3] = '206'
    create_multiImage(xpapoint)

    print('''\n\n\n\n      TEST: Photocounting   \n\n\n\n''') 
    d.set('frame delete all')
    d.set('frame new')
    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
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


def Field_regions(xpapoint, mask=''):
    """Display on DS9 some region files. GS and holes location if guider images,
    slits' locations if detector iamges.
    """
    #from astropy.io import fits
    d = DS9(xpapoint)
    #d.set("regions format ds9")
    #d.set("regions system image")
    path = d.get("file")
    ImageName = os.path.basename(path)
    if (ImageName[:5].lower() == 'image') or (ImageName[:5]== 'Stack'):
        Type = 'detector'
    if ImageName[:5] == 'stack':
        Type = 'guider'
    print ('Type = ', Type)
    try:
        slit_dir = resource_filename('DS9FireBall', 'Slits')
    except:
        slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        
    print('Type = ', Type)
    if mask == '':
        try:
            mask = sys.argv[3]#'f3 names'#sys.argv[3]
        except:
            mask = ''
    print ('Masks = ', mask)
    mask = mask.lower()
    if ('d' in mask):
        print('ok')
        filename = os.path.join(slit_dir, 'DetectorFrame.reg' )
        d.set("region {}".format(filename))
    if ('g' in mask):
        filename = os.path.join(slit_dir, 'GuiderFrame.reg' )
        d.set("region {}".format(filename))
    if Type == 'detector':
        print('ok')
        if ('f1' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F1_119_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F1_119_Zn.reg')
            #if ('name' in mask):
                #   filename2 = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F1_119_names.reg'        
        if ('f2' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F2_-161_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F2_-161_Zn.reg')
            if ('name' in mask):
                filename2 = os.path.join(slit_dir, 'F2_-161_names.reg')
        if ('f3' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F3_-121_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F3_-121_Zn.reg')
            if ('name' in mask):
                filename2 = os.path.join(slit_dir, '3_-121_names.reg')
        if ('f4' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F4_159_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F4_159_Zn.reg')
            #if ('name' in mask):
                #filename2 = os.path.join(slit_dir, 'F4_159_names.reg')
        if ('grid' in mask):
            filename = os.path.join(slit_dir, 'grid_Zn.reg' )
        d.set("region {}".format(filename))
        if d.get('tile')=='yes':
            d.set('frame last')
            for i in range(int(d.get('frame'))-1):
                d.set('frame next')
                d.set('regions ' + filename)



    if Type == 'guider':
        if mask == 'no':
            d.set('contour clear')
            if d.get('tile')=='yes':
                d.set('frame last')
                for i in range(int(d.get('frame'))-1):
                    d.set('frame next')
                    d.set('contour clear')
                
        else:
            d.set('contour clear')
            #d.set('contour yes')

#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']

#            for line in header:                
#                if line in ['USE0','USE1','USE2','USE3','USE4','USE5','USE6','USE7']:
#                    print(line)
            header = d.get_fits()[0].header
            pa = int(header['ROTENC'])
            #DS9
            print('Position angle = ',pa)
            if (pa>117) & (pa<121):
                name1 = os.path.join(slit_dir, 'GSF1.reg')
                name2 = os.path.join(slit_dir, 'F1.ctr')
    #            d.set('regions /Users/Vincent/Documents/FireBallPipe/Calibration/Slits/GSF1.reg')
    #            d.set('contour load /Users/Vincent/Documents/FireBallPipe/Calibration/Slits/F1.ctr')
            if (pa>-163) & (pa<-159):
                name1 = os.path.join(slit_dir, 'GSF2.reg')
                name2 = os.path.join(slit_dir, 'F2.ctr')
            if (pa>-123) & (pa<-119):
                name1 = os.path.join(slit_dir, 'GSF3.reg')
                name2 = os.path.join(slit_dir, 'F3.ctr')
            if (pa>157) & (pa<161):
                name1 = os.path.join(slit_dir, 'GSF4.reg')
                name2 = os.path.join(slit_dir, 'F4.ctr')
            d.set('regions ' + name1)
            d.set('contour load ' + name2)
            if d.get('tile')=='yes':
                d.set('frame last')
                for i in range(int(d.get('frame'))-1):
                    d.set('frame next')
                    d.set('regions ' + name1)
                    d.set('contour load ' + name2)

                    guidingstars = np.zeros((8,3))
                    #namename = getfilename(d)#
                    header = d.get_fits()[0].header
                    for i in range(8):
                        guidingstars[i] = header['CY%i'%(i)],header['CX%i'%(i)],header['USE%i'%(i)]
                        if (int(guidingstars[i,2]) ==  257) or (int(guidingstars[i,2]) ==  1):
                            d.set('regions command "box %0.3f %0.3f 8 8  # color=yellow"' % (guidingstars[i,1],guidingstars[i,0]))
                print('guiding stars = ',guidingstars)
            else:
                guidingstars = np.zeros((8,3))
                header = d.get_fits()[0].header
                for i in range(8):
                    guidingstars[i] = header['CY%i'%(i)],header['CX%i'%(i)],header['USE%i'%(i)]
                    if (int(guidingstars[i,2]) ==  257) or (int(guidingstars[i,2]) ==  1):
                        d.set('regions command "box %0.3f %0.3f 8 8  # color=yellow"' % (guidingstars[i,1],guidingstars[i,0]))
                print('guiding stars = ',guidingstars)

    try:
        print('Putting regions, filename = ', filename)
    except UnboundLocalError:
        pass            

    try:
        d.set('regions {}'.format(filename2))
    except:
        pass
    print('Test completed: OK')
    return

def DS9XYAnalysis(xpapoint, DS9backUp = DS9_BackUp_path):
    """Analyze images from XY calibration, just need to zoom and the 
    selected spot for each images
    """
    from matplotlib import pyplot as plt
    d = DS9(xpapoint)    
    filename = getfilename(d)#filename = d.get('file')
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
            if 'circle' in region:
                MappedRegions.append(region)
        regionsxy = np.zeros((len(MappedRegions),2))

        for i,region in enumerate(MappedRegions):
            a, reg, b = region.replace('(',')').split(')')
            a,name,b = b.replace('{','}').split('}')
            try:
                x,y,a1,a1,a1,a1,a1,a1,a1,a1,a1 = reg.split(',')
            except ValueError:
                x,y,r = reg.split(',')
            x,y = float(x), float(y)
            regionsxy[i,:] = x, y
            SlitsOnImage.append(name)            
        distance = np.sqrt(np.square((Centers[frame,:] - regionsxy)).sum(axis=1))
        index = np.argmin(distance)
        SlitPos[frame] = regionsxy[index]
        slitnames.append(SlitsOnImage[index])


#    detectedregions = []
#    if len(DetectedRegFile) == 1:
#        d.set('frame first')
#        for frame in range(n):    
#            d.set('regions file ' + DetectedRegFile[0])
#            d.set('regions select all')
#            regions = d.get('regions').split('\n')
#            for region in regions:
#                if 'circle' in region:
#                    detectedregions.append(region)
#            regions_detected_xy = np.zeros((len(detectedregions[1:-1]),2))
#            SlitsOnImage = []
#            for i,region in enumerate(detectedregions[1:-1]):
#                a, reg, b = region.replace('(',')').split(')')
#                #a,name,b = b.replace('{','}').split('}')
#                x,y,r = reg.split(',')
#                x,y,r = float(x), float(y), float(r)
#                regions_detected_xy[i,:] = x, y
#                #SlitsOnImage.append(name)
#
#            
#            distance = np.sqrt(np.square((Centers[frame,:] - regions_detected_xy)).sum(axis=1))
#            index = np.argmin(distance)
#            SlitDetectedPos[frame] = regions_detected_xy[index]
#            
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
#    try:
#        Q = plt.quiver(Centers[:,0],Centers[:,1],SlitDetectedPos[:,0] - Centers[:,0],SlitDetectedPos[:,1] - Centers[:,1],scale=30,color='blue',label='Dist to detected slit')
#    except:
#        pass
    plt.quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
    plt.axis('equal')
    #plt.gca().set_xlim(900,2300)
    #plt.gca().set_ylim(-150,2300)
    #plt.ylim((0,2000))
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
#    plt.savefig(os.path.dirname(filename) + '/XYCalibration.png')
#    plt.show()
    return Centers



def DS9stack(xpapoint):
    """This function aims at stacking images with or without dark substraction (no interpolation of the dark for now).
    By pressing ’s’ a window ask for the detector images to stack: just write the first and the last number of the
    images (separated by ’-’ no space as in figure 5). If the images are not consecutive you have to write all the
    numbers (eg. ’13-26-86-89’). Do the same for the dark, put only one number if there is only one dark image
    or press directly ’ok’ if there is not any dark image. The stack image is saved in the repository containing the
    DS9 image with the number of the images stacked and the dark images used + ’stack.fits’.
    """
    from astropy.io import fits
    from .focustest import stackImages
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
        stack = np.zeros((lx,ly,n),dtype='uint16')
        print('\nReading fits files...')
        numbers = []
        for i,file in enumerate(files):
            name = os.path.basename(file) 
            numbers.append(int(name[5:12]))
            with fits.open(file) as f:
                stack[:,:,i] = f[0].data
        image.data = np.nanmean(stack,axis=2)
        fname = os.path.dirname(filename)                
        filename = '{}/StackedImage_{}-{}.fits'.format(fname, np.array(numbers).min(), np.array(numbers).max())
        image.writeto(filename ,overwrite=True)
        print('Images stacked:',filename)            
        
    if Type == 'detector':
        entry = sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# 
        print('Images to stack = ', entry)
        try:
            number_dark = sys.argv[4] #''#sys.argv[4] #''#'sys.argv[4] #'365'#'365-374'#''#sys.argv[4] 
        except:
            number_dark = ''
        print('Dark to remove = ', number_dark)
        numbers = entry.split('-')
    
        if len(numbers) == 2:
            print('2 numbers given for the stack')
            n1, n2 = entry.split('-')
            n1, n2 = int(n1), int(n2)
            numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
        print('Numbers used: {}'.format(numbers))           
    
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
    d.set("file {}".format(filename))   
    d.set("lock frame physical")
    return

def FlagHighBackgroundImages(stack, std=1):
    """
    """
    det_part = stack[:,1053:2133,:]
    det_mean = np.nanmean(det_part)
    det_std = np.nanstd(det_part)
    means = np.nanmean(det_part, axis=(0,1))
    index = means < det_mean + 1 * det_std
    print(index)
    return index
    

def StackImagesPath(paths, cond_bckd=False, fname=''):
    """Stack images of the files given in the path
    """
    from astropy.io import fits
    import re
    n=len(paths)
    paths.sort()
    image = fits.open(paths[0])
    name_images = [os.path.basename(path) for path in paths]
    lx,ly = image[0].data.shape
    if 'CRv' in paths[0]:
        stack = np.zeros((lx,ly))#,dtype='uint16')
    else:
        stack = np.zeros((lx,ly))#,dtype='uint16')
    for i,file in enumerate(paths):
        try:
            with fits.open(file) as f:
                print(file)
                #f[0].data[~np.isfinite(f[0].data)] = stack[~np.isfinite(f[0].data)]
                stack[:,:] += f[0].data
                del(f)
        except TypeError as e:
            print(e)
            n -= 1
    stack = stack/n        
    if cond_bckd:
        index = FlagHighBackgroundImages(stack, std=1)
        images = '\n'.join(list(np.array(name_images)[index])) 
    else:
        index = True
#    if np.isfinite(stack).all():
#        stack.dtype = 'uint16'
    
    #print('All these images have not been taken into account because of hight background', '\n'.join(list(np.array(name_images)[~index]))  )
    #image[0].data = np.nanmean(stack[:,:,index],axis=2)#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    image[0].data = stack#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    #image[0].header['STK_NB'] =  '-'.join(re.findall(r'\d+',images))#'-'.join(list(np.array(name_images)[index]))   
    try:
        name = '{}/StackedImage_{}-{}{}.fits'.format(os.path.dirname(paths[0]), int(os.path.basename(paths[0])[5:5+6]), int(os.path.basename(paths[-1])[5:5+6]),fname)
    except ValueError:
        name = '{}/StackedImage_{}-{}{}'.format(os.path.dirname(paths[0]), os.path.basename(paths[0]).split('.')[0], os.path.basename(paths[-1]),fname)       
    print('Image saved : %s'%(name))
    fitswrite(image[0], name)
    print('n = ', n)
    return image  


def StackImagesPath_old(paths, cond_bckd=True):
    """Stack images of the files given in the path
    """
    from astropy.io import fits
    import re
    n=len(paths)
    paths.sort()
    image = fits.open(paths[0])
    name_images = [os.path.basename(path) for path in paths]
    lx,ly = image[0].data.shape
    if 'CRv' in paths[0]:
        stack = np.zeros((lx,ly,n))#,dtype='uint16')
    else:
        stack = np.zeros((lx,ly,n),dtype='uint16')
    for i,file in enumerate(paths):
        with fits.open(file) as f:
            print(file)
            stack[:,:,i] = f[0].data
    if cond_bckd:
        index = FlagHighBackgroundImages(stack, std=1)
    else:
        index = True
#    if np.isfinite(stack).all():
#        stack.dtype = 'uint16'
    images = '\n'.join(list(np.array(name_images)[index])) 
    print('All these images have not been taken into account because of hight background', '\n'.join(list(np.array(name_images)[~index]))  )
    image[0].data = np.nanmedian(stack[:,:,index],axis=2)#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    image[0].header['STK_NB'] =  '-'.join(re.findall(r'\d+',images))#'-'.join(list(np.array(name_images)[index]))   
    try:
        name = '{}/StackedImage_{}-{}.fits'.format(os.path.dirname(paths[0]), int(os.path.basename(paths[0])[5:5+6]), int(os.path.basename(paths[-1])[5:5+6]))
    except ValueError:
        name = '{}/StackedImage_{}-{}'.format(os.path.dirname(paths[0]), os.path.basename(paths[0]).split('.')[0], os.path.basename(paths[-1]))       
    print('Image saved : %s'%(name))
    fitswrite(image[0], name)
    return image  

def DS9focus(xpapoint):
    """Apply focus test class to the image
    """
    from .focustest import Focus  
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get("file")
    try:
        entry = sys.argv[3] #f3 -121'#sys.argv[3] #''#sys.argv[4] #''#'sys.argv[4] #'365'#'365-374'#''#sys.argv[4] 
    except:
        F = Focus(filename = filename, HumanSupervision=False, source='Zn', shape='holes', windowing=False, peak_threshold=50,plot=False)
        d.set('regions {}'.format(filename[:-5]+'detected.reg'))
    else:
        try:
            mask, pa = entry.split(' ')
            F = Focus(filename = filename, quick=False, threshold = [7], fwhm = [9,12.5],
                  HumanSupervision=False, reversex=False, source='all',
                  shape='slits', windowing=True, mask=mask.capitalize(), pa=int(pa) ,MoreSources=0,peak_threshold=50,plot=True)
        except ValueError:
            mask = entry
            F = Focus(filename = filename, quick=False, threshold = [7], fwhm = [9,12.5],
                  HumanSupervision=False, reversex=False, source='all',
                  shape='slits', windowing=True, mask=mask.capitalize(),MoreSources=0,peak_threshold=50,plot=True)
    
        d.set('regions {}'.format(filename[:-5]+'detected.reg'))
    return F


def DS9throughslit(xpapoint, DS9backUp = DS9_BackUp_path):
    """How to use: Open one an image of the through focus which is close to the focus. Click on region. Then click
    precisely on what you think is the centre of the PSF. Select the region you created and press t (throughfocus)
    or go in analysis menu: Through slit analysis. This will open a dialog box that asks what is the number of
    the images of the through focus. You can either put the numbers (eg: "10-21"). If the throughslit is not done
    straightforward then you will have to enter all the number of the images in the right order (eg "10-12-32-28-21").
    This will pick up the center of the region you put, compute the sum of the pixels in a 30pix box after
    removing the background in an anulus around the source. Then it plots the flux with respect to the throughslit
    and give the number of the image with the maximum of flux in title. Note that the x axis only give the index
    of the image, not its number.
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from .focustest import estimateBackground
    from .focustest import Gaussian
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
        numbers=None
    d = DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
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
    path.sort()
    #print("\n".join(path)) 
    a = getregion(d)
    radius = 15
    print('Sum pixel is used (another estimator may be prefarable)')
    fluxes=[]
    
    for file in path:
        print (file)
        fitsfile = fits.open(file)[0]
        image = fitsfile.data
        #plt.figure(figsize=(sizefig,sizefig))
        #plt.imshow(image[int(a.yc)-radius:int(a.yc)+radius, int(a.xc)-radius:int(a.xc)+radius])#;plt.colorbar();plt.show()
        #plt.show()
        subimage = image[int(a.yc)-radius:int(a.yc)+radius, int(a.xc)-radius:int(a.xc)+radius]
        background = estimateBackground(image, [a.xc,a.yc], radius=30, n=1.8)
        #flux = np.nansum(image[center[0]-n:center[0]+n,center[1]-n:center[1]+n])-np.nansum(image[center_bg[0]-n:center_bg[0]+n,center_bg[1]-n:center_bg[1]+n])
        flux = np.nansum(subimage - background) #- estimateBackground(image, center, radius, n_bg)
        fluxes.append(flux)
    fluxesn = (fluxes - min(fluxes)) / max(fluxes - min(fluxes))    
    x = np.arange(len(path))+1
    popt, pcov = curve_fit(Gaussian, x, fluxesn, p0=[1, x.mean(),3,0])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
    xl = np.linspace(x.min(),x.max(),100)
    maxf = xl[np.where(Gaussian(xl,*popt)==np.nanmax(Gaussian(xl,*popt)))[0][0]]#[0]
    plt.figure()
    plt.plot(x, fluxesn,'o',label='data')
    plt.plot(xl, Gaussian(xl,*popt),'--',label='Gaussian fit')
    plt.legend()
    plt.plot(np.linspace(maxf, maxf, len(fluxes)), fluxesn/max(fluxesn))
    plt.grid(linestyle='dotted')
    plt.xlabel('# image')
    plt.title('Best image : {}'.format(maxf))
    plt.ylabel('Sum pixel') 
    name = 'Through slit analysis\n%0.3f - %s - %s'%(maxf,[int(a.xc),int(a.yc)],fitsfile.header['DATE'])
    print(name) 
    plt.title(name)
    plt.savefig(os.path.dirname(file) + '/' + name + '.jpg')
    plt.show()
    csvwrite(np.vstack((x, fluxesn,Gaussian(x,*popt))).T, DS9backUp + 'CSVs/%s_ThroughSlit.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )

    return 

def DS9snr(xpapoint):
    """Compute a rough SNR on the selected spot. Needs to be updates
    """
    #from astropy.io import fits
    from .focustest import create_DS9regions2
    n1 = 1.2
    n2 = 1.8
    d = DS9(xpapoint)
    #filename = getfilename(d)#ffilename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
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
#        np.nansum(background_mask_ext)
#        np.nansum(background_mask_inf)
#        np.nansum(maskBackground)
        
    if hasattr(region, 'r'):
        signal = image[r<region.r]
        maskBackground = (r >= n1 * region.r) & (r <= n2 * region.r)
    else:
        print('Need to be a circular region')
    signal_max = np.nanpercentile(signal,95)
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
    


def DS9createImage(xpapoint):
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
    import matplotlib; matplotlib.use('TkAgg')  
   # import matplotlib.pyplot as plt
    #from astropy.io import fits
    from .focustest import ConvolveSlit2D_PSF
    d=DS9()
#    n=20
    #fitstest = fits.open('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180612/image000365.fits')
    #fitstest[0].data *= 0`
    lx, ly = 1000, 1000#fitstest[0].data.shape
    x, y = np.arange(lx), np.arange(ly)
    xy = np.meshgrid(x,y)
    new_image = np.zeros((lx,ly))
    for xi, ampi in zip((np.linspace(100,1900,10)),(np.linspace(10,1000,10))):
        slit = (1/0.006)*ConvolveSlit2D_PSF(xy, ampi, 3, 9, int(xi), 1500, 3,3).reshape(ly,lx).T
        new_image = new_image + slit# + gaussian2
        #plt.imshow(slit[int(xi)-n:int(xi)+n,1500-n:1500+n]);plt.plt.colorbar();plt.show()
        #plt.imshow(new_image[int(xi)-n:int(xi)+n,1500-n:1500+n]);plt.colorbar();plt.show()
    for xi, ampi in zip((np.linspace(100,1900,10)),(np.linspace(10,1000,10))):
        gaussian = twoD_Gaussian(xy, ampi, int(xi), 1000, 5, 5, 0).reshape(ly,lx).T
        new_image = new_image + gaussian
    d.set_np2arr(new_image)
    return 
    
    
    # + gaussian2
        #imshow(new_image[int(xi)-n:int(xi)+n,1000-n:1000+n]);colorbar();plt.show()

#        
#    imshow(fitstest[0].data);colorbar()
#    imshow(new_image[1000-n:1000+n,1000-n:1000+n]);colorbar();plt.show()  # + gaussian[1000-n:1000+n,1000-n:1000+n]);colorbar()  
#    imshow(new_image[1500-n:1500+n,1000-n:1000+n]);colorbar();plt.show()  # + gaussian2[1000-n:1000+n,2000-n:2000+n]);colorbar()  
#    imshow(new_image[1900-n:1900+n,1000-n:1000+n]);colorbar();plt.show()  # + gaussian2[1000-n:1000+n,2000-n:2000+n]);colorbar()  
#    
#    
#    fitstest[0].data = new_image
#
#    fitswrite(fitstest, os.path.dirname(os.path.realpath(__file__)) + '/test/TestImage.fits')
#    #imshow(fits.open('/Users/Vincent/Documents/FireBallPipe/Calibration/TestImage.fits')[0].data)
#    return
#




def DS9meanvar(xpapoint):
    """Compute mean standard deviation and skewness in this parth of the image
    """
    #from astropy.io import fits
    from scipy import stats
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    region = getregion(d)
#    if hasattr(region, 'h'):
#        xc, yc, w, h = int(region.xc), int(region.xc), int(region.w), int(region.h) 
#        Xinf = yc - h/2
#        Xsup = yc + h/2#int(region.yc + region.h/2)
#        Yinf = xc - w/2 #int(region.xc - region.w/2)
#        Ysup = xc + w/2 #int(region.xc + region.w/2)
#    if hasattr(region, 'r'):
#        xc, yc, r = int(region.xc), int(region.xc), int(region.r)
#        Xinf = yc - r
#        Xsup = yc + r
#        Yinf = xc - r
#        Ysup = xc + r
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup, Xinf, Xsup]
    print(area)
    image = d.get_fits()[0].data[area[0]:area[1],area[2]:area[3]]#picouet
    print ('Image : {}'.format(filename))
    print ('Mean : {}'.format(image.mean()))
    print ('Standard deviation : {}'.format(image.std()))
    print ('Skewness: {}'.format(stats.skew(image,axis=None)))
    return


def DS9lock(xpapoint):
    """Lock all the images in DS9 together in frame, smooth, limits, colorbar
    """
    d = DS9(xpapoint)
    lock = d.get("lock scalelimits")
    if lock == 'yes':
        d.set("lock frame no")
        d.set("lock scalelimits no")
        d.set("lock crosshair no")
        d.set("lock smooth no")
        d.set("lock colorbar no")
    if lock == 'no':
        d.set("lock frame physical")
        d.set("lock scalelimits yes")
        d.set("crosshair lock physical")
        d.set("lock crosshair physical")
        d.set("lock smooth yes")
        d.set("lock colorbar yes")
    return


def DS9inverse(xpapoint):
    """Inverse the image in DS9, can be used to then do some positive gaussian fitting, etc
    """
    #from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    fitsfile = d.get_fits()[0]#fits.open(filename)[0]

    image = fitsfile.data
    new_image = - image + image.max()
    fitsfile.data = new_image
    fitsfile.data = new_image
    fitsfile.writeto(filename[:-5] + '_Inverse.fits',overwrite=True)
    d.set('frame new')
    d.set('file {}'.format(filename[:-5] + '_Inverse.fits'))
    return


def DS9Update(xpapoint,Plot=True):
    """Always display last image of the repository and will upate with new ones
    """
    import time
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    flag=0
    while flag<1: 
        files = glob.glob(os.path.dirname(filename)+ '/*.fits')
        latest_file = max(files, key=os.path.getctime)
        if os.path.getctime(latest_file)>os.path.getctime(filename):
            filename = latest_file
            d.set('file ' + filename)
            print('New file!\nDisplaying : ' + latest_file)
        else:
            print('No file more recent top display')
            print('Waiting ...')
        time.sleep(5)
    return
        
        
def Lims_from_region(region):
    """Return the pixel locations limits of a DS9 region
    """
    try:
        xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
    except:
        print('Region is not a box, exracting box using the radius')
        xc, yc, r = int(region.xc), int(region.yc), int(region.r)
        w, h = r, r
    print('W = ', w)
    print('H = ', h)
    if w <= 2:
        w = 2		      
    if h <= 2:																			
        h = 2
    Yinf = int(np.floor(yc - h/2 -1))
    Ysup = int(np.ceil(yc + h/2 -1))
    Xinf = int(np.floor(xc - w/2 -1))
    Xsup = int(np.ceil(xc + w/2 -1))
    print('Xinf, Xsup, Yinf, Ysup = ', Xinf, Xsup, Yinf, Ysup)
    print('data[%i:%i,%i:%i]'%(Yinf, Ysup,Xinf, Xsup))
    return Xinf, Xsup, Yinf, Ysup


def DS9center(xpapoint,Plot=True):
    """How to use:
    This function can be used in two different ways:
    •To find the center of a circular spot (eg. fiber in autocoll for XY calibration): Click on region and select
    Circle shape (default). Then click precisely on what you think is the centre of the PSF. Select the region
    you created and press c (centering) or go in analysis menu: centering.
    •To find the center of a slit: Click on region and select box. Then put the box where you think the image
    of the slit is relatively flat (a few pixels). Select the region you created and press c (centering) or go in
    analysis menu: centering.
    In the first case, the function opens a dialog box that will ask the size of the fiber. Give it in pixels +
    Enter or press directly Enter if you only want to fit a 2D gaussian function. Then, the function will pick up the
    center of the region you put, compute the barycenter of the image and fit the spot by a 2D gaussian convolved
    with a 2D disk of the diameter you gave (or only a gaussian). To be sure that the fit worked correctly, after it
    converged it plots the line and the column of the PSF at its center with the computed 2D function that fits the
    spot (see figure 10).
    In the second case, the function does two 1D analysis. First it will pick up the box region and stack it in
    both directions to create two profiles. Then in will fit the two profiles by the 1D convolution of a sigma-free
    gaussian and a size-free box and plot it (see figure 11).
    In both cases it will return the computed center of the spot and print a new region with the computed
    center. It will also give the characteristics of the spot.
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    #from astropy.io import fits
    from .focustest import ConvolveBoxPSF
    from .focustest import create_DS9regions
    from .focustest import create_DS9regions2
    from .focustest import estimateBackground
    from scipy.optimize import curve_fit
    from .focustest import Gaussian
    d = DS9(xpapoint)#DS9(xpapoint)
    #filename = getfilename(d)#ffilename = d.get("file")
    region = getregion(d)

    if hasattr(region, 'h'):
        xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
        print('W = ', w)
        print('H = ', h)
        if w <= 2:
            w = 2
        if h <= 2:
            h = 2
        Xinf = int(np.floor(yc - h/2 -1))
        Xsup = int(np.ceil(yc + h/2 -1))
        Yinf = int(np.floor(xc - w/2 -1))
        Ysup = int(np.ceil(xc + w/2 -1))
        data = d.get_fits()[0].data
        imagex = data[Xinf-15:Xsup+15,Yinf:Ysup].sum(axis=1)
        imagey = data[Xinf:Xsup,Yinf-15:Ysup+15].sum(axis=0)
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
        d.set('regions /tmp/centers.reg')
        
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8,6))
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
        
    if hasattr(region, 'r'):
        xc, yc, r = int(region.xc), int(region.yc), int(region.r)
        Xinf = yc - r -1#int(region.yc - region.r)
        Xsup = yc + r -1#int(region.yc + region.r)
        Yinf = xc - r -1#int(region.xc - region.r)
        Ysup = xc + r -1#int(region.xc + region.r)
        data = d.get_fits()[0].data
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
        bounds = ([1e-1*np.nanmax(image), xo-10 , yo-10, 0.5,0.5,-1e5], [10*np.nanmax(image), xo+10 , yo+10, 10,10,1e5])#(-np.inf, np.inf)#
        Param = (np.nanmax(image),int(xo),int(yo),2,2,np.percentile(image,15))
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
    """Transform hours, minutes, seconds to seconds [+days]
    """
    return 3600 * h + 60 * m + s + d*24*3600


def calc_emgainGillian(image, area=[0,-1,0,-1],plot_flag=False,save=False, DS9backUp = DS9_BackUp_path):
        """Compute Bias and EMgain on a high voltage image taken with an EMCCD
        """
        from astropy.io import fits
        gain =  1.78
        if (type(image) == np.str) or (type(image) == np.str_) or (type(image) == str):        
            fitsimage = fits.open(image)[0]
            img_data = fitsimage.data
            ysize, xsize = img_data.shape
            img_section = img_data[area[0]:area[1],area[2]:area[3]]
        else:
            fitsimage = image#fits.open(image)[0]
            ysize, xsize = fitsimage.data.shape
            img_section = fitsimage.data[area[0]:area[1],area[2]:area[3]]
        nbins = 1000
        readnoise = 60
        #gain = float(gain)

        # Histogram of the pixel values
        n, bins = np.histogram(np.array(img_section[np.isfinite(img_section)]), bins = nbins, range=(-200,11800))
        bin_center = 0.5 * (bins[:-1] + bins[1:])
        #y0 = np.nanmin(n)

        n_log = np.log10(n)

        # What is the mean bias value?
        idx = np.where(n == n.max())
        bias = bin_center[idx][0]
        n_bias = n[idx][0]
    
        # Range of data in which to fit the Gaussian to calculate sigma
        bias_lower = bias - float(1.5) * readnoise
        bias_upper = bias + float(2.0) * readnoise
   
        idx_lower = np.where(bin_center >= bias_lower)[0][0]
        idx_upper = np.where(bin_center >= bias_upper)[0][0]
   
        #gauss_range = np.where(bin_center >= bias_lower)[0][0]
    
        valid_idx = np.where(n[idx_lower:idx_upper] > 0)
        try:
            amp, x0, sigma = gaussianFit(bin_center[idx_lower:idx_upper][valid_idx], n[idx_lower:idx_upper][valid_idx], [n_bias, bias, readnoise])
        except RuntimeError as e:
            print(e)
            return 0,0,0,0
        #plt.figure()
        #plt.plot(bin_center[idx_lower:idx_upper], n[idx_lower:idx_upper], 'r.')
        #plt.show()

        # Fitted frequency values
        xgaussfit = np.linspace(bin_center[idx_lower], bin_center[idx_upper], 1000)
        #print xgaussfit
        ygaussfit = gaussian(xgaussfit, amp, x0, sigma)
        #print ygaussfit

        # Define index of "linear" part of the curve
        Tmin = 10
        Tmax = 30
        threshold_min = bias + (float(Tmin) * sigma)
        threshold_max = bias + (float(Tmax) * sigma)
        threshold_55 = bias + (float(5.5) * sigma)

        #print threshold_max

        # Lines for bias, 5.5*sigma line

        n_line = n_log.size
        zeroline = np.zeros([n_line], dtype = np.float32)
        threshold0 = zeroline + int(bias)
        threshold55 = zeroline + int(bias + 5.5*sigma)
        #thresholdmin = zeroline + int(threshold_min)
        #thresholdmax = zeroline + int(threshold_max)
        try:
            idx_threshmin = np.array(np.where(bin_center >= threshold_min))[0,0]
        except IndexError as e:
            print('Error, threshold min was %i and first bin is %i-%i'%(threshold_min,bin_center[0],bin_center[1]))
            print(e)
            return (0,0,0,0), (0,0)
        try:
            idx_threshmax = np.array(np.where(bin_center >= threshold_max))[0,0]
        except IndexError:
            print('Error, threshold max was %i and last bin was %i'%(threshold_max,bin_center.max()))
            idx_threshmax = np.array(np.where(bin_center == bin_center.max()))[0,0]
            
        valid_idx = np.where(n[idx_threshmin:idx_threshmax] > 0)

        slope, intercept = fitLine(bin_center[idx_threshmin:idx_threshmax][valid_idx], n_log[idx_threshmin:idx_threshmax][valid_idx])

        # Fit line
        #xlinefit = np.linspace(bias-20*sigma, bin_center[idx_threshmax], 1000)
        xlinefit = np.linspace(bin_center[idx_threshmin], bin_center[idx_threshmax], 1000)
        ylinefit = linefit(xlinefit, slope, intercept)


        emgain = (-1./slope) * (gain)

        #print xlinefit
        embias_idx = np.where(xlinefit >= bias)
        #print embias_idx
        embias_y1 = ylinefit[embias_idx[0][0]]
        #print embias_y1
        em5_idx = np.where(xlinefit >= bias+(5.5*sigma))
        #print em5_idx
        em5_y1 = ylinefit[em5_idx[0][0]]
        #print em5_y1

        total_counts = 0.5*(threshold_max-bias)*(embias_y1)
        thres_counts = 0.5*(threshold_max-threshold_55)*(em5_y1)
        #print total_counts, thres_counts
        frac_lost = (total_counts-thres_counts)/thres_counts
        print(frac_lost)
        exposure, gain, temp = fitsimage.header['EXPTIME'], fitsimage.header['EMGAIN'], fitsimage.header['TEMP']
#        plot_hist2(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,
#                   ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp,plot_flag=plot_flag, ax=None)
        try:
            fitsimage.header['C_egain'] = emgain
        except ValueError as e:
            print (e)
            fitsimage.header['C_egain'] = -1
        fitsimage.header['C_bias'] = bias
        fitsimage.header['C_sigR0'] = sigma
        try:
            fitsimage.header['C_flost'] = frac_lost
        except ValueError:
            pass
        if save:
            fitswrite(fitsimage, image)
        return (emgain,bias,sigma,frac_lost), (image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit, ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp)





def DS9DetectCosmics(xpapoint,filen=None,T=6*1e4):
    from astropy.io import fits
    from .focustest import create_DS9regions
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]#]fd.get("file")]
        
#    try:
#        region = getregion(d)
#        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
#        area = [Yinf, Ysup,Xinf, Xsup]
#        print(Yinf, Ysup,Xinf, Xsup)
#    except ValueError:
#        area = [0,-1,1053,2133]
    try:
        region = getregion(d)
    except ValueError:
        area = [0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)  
        
    for filename in path:
        print(filename)
        #from astropy.io import fits
        fitsimage =  fits.open(filename)[0] 
        image = fitsimage.data
        cosmicRays = detectCosmics(image,T=T,area=area)
        if len(cosmicRays)>11000:
            print('Too many cosmics to detected, Please make sure the image is not saturated or change the detection threshold.')
            return 1
        else:
            #print('Deleting duplications...')
            #cosmicRays = delete_doublons_CR(cosmicRays,dist=5)
            print('Assigning detections to single cosmic-ray events...')
            cosmicRays = assign_CR(cosmicRays,dist=10)
    #        plot(cosmicRays[cosmicRays['id']==-1]['xcentroid'],cosmicRays[cosmicRays['id']==-1]['ycentroid'],'.')
            print('Distinguishing  Dark From CR...')
            cosmicRays = DistinguishDarkFromCR(cosmicRays, T=T, number=2)
            print('Determine cosmic-ray front')
            cosmicRays = Determine_front(cosmicRays)
            a=cosmicRays[cosmicRays['front']==1]
            a = a[(a['xcentroid']>1000) & (a['xcentroid']<2200)] 
            a['filename'] = path
            name = os.path.dirname(filename)+'/' +os.path.basename(filename)[:-5]+'.CRv.fits'
            csvwrite(a,name[:-5] + '.csv')
        
    if len(path) < 2:    
        create_DS9regions([list(a['xcentroid'])],[list(a['ycentroid'])], form=['circle'], radius=10, save=True, 
                   savename='/tmp/cr', color = ['yellow'],ID=None)
        d.set('region delete all')
        d.set('region {}'.format('/tmp/cr.reg'))                                        
    return cosmicRays







def DS9removeCRtails2(xpapoint,filen=None):
    """Replace cosmic ray tails present in the image by NaN values and save
    it with the same name with .CVr.fits. Some primary particules hits might
    not be completely removed and would need some other specific care with
    the ReplaceByNaNs function
    """
    d = DS9(xpapoint)
    try:
        length = float(sys.argv[3])
    except IndexError:
        length = 20
    if len(sys.argv) > 4:
        path = Charge_path(xpapoint,entry_point=4)
    else:
        path = [getfilename(d)]#]fd.get("file")]
        
    try:
        region = getregion(d)
    except ValueError:
        area = [0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)   
        
    for filename in path:
        print(filename, length)
        result, name, cosmicRays = RemoveCRtail(filename,T=6*1e4, length=length, area=area)#2.1*e4, area=area) 
    if (len(path)<2) & (len(cosmicRays)>1):    
        d.set('region delete all')
        d.set('region {}'.format('/tmp/cr.reg'))                                        
        d.set('frame new')
        d.set('tile yes')
        d.set('lock frame physical')
        d.set('file ' + name)


    return cosmicRays



def DS9Catalog2Region(xpapoint):
    """vpicouet
    """
    from astropy.table import Table
    from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    path = sys.argv[3]
    x, y = sys.argv[4].replace(',','-').split('-')
    cat = Table.read(path)
    print(cat)
    create_DS9regions2(cat[x],cat[y], radius=3, form = 'circle', save=True,color = 'yellow', savename='/tmp/centers')
    d.set('regions /tmp/centers.reg')    
    return

def DS9Region2Catalog(xpapoint, name=None):
    """
    """
    from astropy.table import Table
    d = DS9(xpapoint)
    regions = getregion(d, all=True)
    #print(regions)
    #print(len(regions))
    filename = d.get('file')
    try:
        x, y = np.array([r.xc for r in regions]), np.array([r.yc for r in regions])
    except AttributeError:
        x, y = np.array([r.xc for r in [regions]]), np.array([r.yc for r in [regions]])
    cat = Table((x,y),names=('ycentroid','xcentroid'))    
    print(cat)
    if name is None:
        csvwrite(cat,filename[:-5] + '_regions.reg')
    else:
        csvwrite(cat,name)
    return cat


def DS9MaskRegions(xpapoint):
    """
    """
#    from astropy.io import fits
    try:
        length = float(sys.argv[3])
    except IndexError:
        length = 20
    d=DS9()
    path = d.get('file')
    fitsimage = d.get_fits()[0]
    cosmicRays = DS9Region2Catalog(xpapoint, name='/tmp/cat.csv')
    cosmicRays['front'] = 1
    cosmicRays['dark'] = 0
    cosmicRays['id'] = np.arange(len(cosmicRays))
    print(cosmicRays)
    maskedimage = MaskCosmicRays(fitsimage.data, cosmics=cosmicRays,length=length, all=True, cols=None)
    fitsimage.data = maskedimage
    name = path[:-5]+'_masked.fits'
    fitsimage.header['N_CR'] = cosmicRays['id'].max()
    fitsimage.header['MASK'] = len(np.where(maskedimage==np.nan)[0])   
    fitswrite(fitsimage, name)
    d.set('frame new')
    d.set('tile yes')
    d.set('file ' + name)
    return 




def DS9removeCRtails_CS(xpapoint): 
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from astropy.table import Table, Column
    from .focustest import create_DS9regions
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint,entry_point=3)
    else:
        path = [getfilename(d)]#]fd.get("file")]
        
    try:
        region = getregion(d)
    except ValueError:
        area = [0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)   
   
    for filename in path:
        fitsimage = fits.open(filename)[0]
        image = fitsimage.data
        CS = plt.contour(image, levels=15000, colors='white', alpha=0.5)
        names = ('id','sizex','sizey','len_contour','max_x','min_y', 'max_y')
        cosmics = Table(np.zeros((len(CS.allsegs[0]),len(names))),names=names)
        cosmics['id'] = np.arange(len(CS.allsegs[0]))
        cosmics['sizex'] = [cs[:,0].max() - cs[:,0].min()     for cs in CS.allsegs[0] ]
        cosmics['sizey'] = [cs[:,1].max() - cs[:,1].min()     for cs in CS.allsegs[0] ]
        cosmics['len_contour'] = [ len(cs[:,1])   for cs in CS.allsegs[0] ]
        cosmics['max_x'] = [ int(cs[:,0].max()+1)   for cs in CS.allsegs[0] ]
        cosmics['min_y'] = [ int(cs[:,1].min()-1)   for cs in CS.allsegs[0] ]
        cosmics['max_y'] = [ int(cs[:,1].max()+2)   for cs in CS.allsegs[0] ]
        cosmics['mean_y'] = [ int((cs[:,1].max()+cs[:,1].max())/2)   for cs in CS.allsegs[0] ]
        cosmics['size'] = [ 50   for cs in CS.allsegs[0] ]
        cosmics['size_opp'] = [ 1 for cs in CS.allsegs[0] ]
        mask1 = (cosmics['len_contour']<=20)
        mask2 = (cosmics['len_contour']>20) & (cosmics['len_contour']<2000)
        mask3 = (cosmics['len_contour']>50) & (cosmics['len_contour']<2000)
        mask4 = (cosmics['len_contour']>200) & (cosmics['len_contour']<2000)
        cosmics['size'][mask2] = 200
        cosmics['size'][mask3] = 3000
        cosmics['size_opp'][mask4] = 3000
        cosmics['min_y'][(cosmics['len_contour']>200) & (cosmics['len_contour']<2000)] -= 20
        cosmics['max_y'][(cosmics['len_contour']>200) & (cosmics['len_contour']<2000)] += 20
        a = cosmics
        maskedimage = MaskCosmicRaysCS(image, cosmics=cosmics)
        csvwrite(a,os.path.dirname(filename)+'/CosmicRayFree/CSVs/' +os.path.basename(filename)[:-5] + '.csv')  
        fitsimage.data = maskedimage
        name = os.path.dirname(filename)+'/CosmicRayFree/' +os.path.basename(filename)[:-5]+'.CRv_cs.fits'
        fitsimage.header['N_CR'] = len(cosmics)
        fitsimage.header['N_CR1'] = len(cosmics[mask1])
        fitsimage.header['N_CR2'] = len(cosmics[mask2])
        fitsimage.header['N_CR3'] = len(cosmics[mask3])
        fitsimage.header['N_CR4'] = len(cosmics[mask4])
        try:
            fitsimage.header['MASK'] = 100 * float(np.sum(~np.isfinite(maskedimage[:,1053:2133]))) / (maskedimage[:,1053:2133].shape[0]*maskedimage[:,1053:2133].shape[1])
        except ZeroDivisionError:
            fitsimage.header['MASK'] = 100 * float(np.sum(~np.isfinite(maskedimage))) / (maskedimage.shape[0]*maskedimage.shape[1])
            
        fitswrite(fitsimage,name)
    if (len(path)<2) & (len(cosmics)>1):    
        create_DS9regions([list(a['max_x'])],[list(a['mean_y'])], form=['circle'], radius=10, save=True, 
                   savename='/tmp/cr', color = ['yellow'],ID=[list(a['len_contour'])])
        d.set('region delete all')
        d.set('regions /tmp/cr.reg')
        d.set('frame new')
        d.set_np2arr(maskedimage)
        d.set('tile yes')
    return  

 
def RemoveCRtail(path,T=6*1e4, length=1000,area=[0,2069,1053,2133]):
    """
    """
    from .focustest import create_DS9regions
    from astropy.io import fits
    fitsimage =  fits.open(path)[0] #d.get_fits()[0]#fits.open(path)[0] 
    image = fitsimage.data
    cosmicRays = detectCosmics(image,T=T, area=area)
    if len(cosmicRays)==0:
        return fitsimage, 'no cosmic detected', cosmicRays
    else:
        if len(cosmicRays)>11000:
            print('Too many cosmics to detected, Please make sure the image is not saturated or change the detection threshold.')
            return 1
        else:
            #print('Deleting duplications...')
            #cosmicRays = delete_doublons_CR(cosmicRays,dist=5)
            print('Assigning detections to single cosmic-ray events...')
            cosmicRays = assign_CR(cosmicRays,dist=20)
            print('Distinguishing  Dark From CR...')
            cosmicRays = DistinguishDarkFromCR(cosmicRays, T=T, number=2)
            print('Determine cosmic-ray front')
            cosmicRays = Determine_front(cosmicRays)
            a=cosmicRays[cosmicRays['front']==1]
            a['filename'] = path
            #a = a[(a['xcentroid']>1000) & (a['xcentroid']<2200)]             
            create_DS9regions([list(a['xcentroid'])],[list(a['ycentroid'])], form=['circle'], radius=10, save=True, 
                       savename='/tmp/cr', color = ['yellow'],ID=[list(a['id'])])
            print('Masking cosmic-ray events...')
            maskedimage = MaskCosmicRays(image, cosmics=cosmicRays,all=False, cols=1, length=length)
            fitsimage.data = maskedimage
            name = os.path.dirname(path)+'/CosmicRayFree/' +os.path.basename(path)[:-5]+'.CRv.fits'
            fitsimage.header['N_CR'] = cosmicRays['id'].max()
            fitsimage.header['MASK'] = len(np.where(maskedimage==np.nan)[0])
            fitswrite(fitsimage,name)
            csvwrite(a,name[:-5] + '.csv')
            return fitsimage, name, cosmicRays


def DetectHotPixels(xpapoint, DS9backUp = DS9_BackUp_path, T1=None, T2=None):
    """
    """
    from .focustest import create_DS9regions
    #from astropy.io import fits
    if T1 is None:
        try:
            entry = sys.argv[3]
        except IndexError:
            T1, T2 = 2*1e4, 4*1e4
        else:
            try:
                T1, T2 = entry.split('-')
            except ValueError:
                T1, T2 = entry, 7e4

        T1, T2 = int(T1), int(T2)
        T1, T2 = np.min([T1,T2]), np.max([T1,T2])
    print('Threshold = %i % i'%(T1,T2))
    d=DS9()
    try:
        region = getregion(d)
    except ValueError:
        area = [0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)    

    fitsimage = d.get_fits()[0]#fits.open(path)[0] 
    image = fitsimage.data
    cosmicRays = detectCosmics(image,T=T1 ,T2=T2, area=area)
    #cosmicRays = cosmicRays[(cosmicRays['xcentroid']<2200) & (cosmicRays['xcentroid']>1700) & (cosmicRays['ycentroid']<1000)]
    #plt.plot(cosmicRays['xcentroid'],cosmicRays['ycentroid'],'x')
    cosmicRays = assign_CR(cosmicRays,dist=5)
    cosmicRays = Determine_front(cosmicRays)
    #plt.plot(cosmicRays[(cosmicRays['front']==1)]['xcentroid'],cosmicRays[(cosmicRays['front']==1)]['ycentroid'],'x')
    cosmicRays = cosmicRays[cosmicRays['front']==1]
    csvwrite(cosmicRays, d.get('file')[:-5] + '_HotPixels.csv')
    create_DS9regions([list(cosmicRays['xcentroid'])],[list(cosmicRays['ycentroid'])], form=['circle'], radius=3, save=True, savename=DS9backUp + 'DS9Regions/HotPixels%i-%i'%(T1,T2), color = ['yellow'],ID=None)
    d.set('region delete all')
    d.set('region {}'.format(DS9backUp + 'DS9Regions/HotPixels%i-%i.reg'%(T1,T2)))                                        
    return
#    
#    if len(cosmicRays)==0:
#        return fitsimage, 'no cosmic detected', cosmicRays
#    else:
#        if len(cosmicRays)>11000:
#    

       
def detectCosmics(image,T=6*1e4, T2=None,area=[0,2069,1053,2133]):
    """Detect cosmic rays, for FIREBall-2 specfic case it is a simplistic case
    where only thresholding is enough
    """
    from astropy.table import Table, Column
    if T2 is None:
        y, x = np.where(image>T)
        value = image[image>T]
    else:
        y, x = np.where((image>T) & (image<T2))
        value = image[(image>T) & (image<T2)]
    index = (x>area[2]) & (x<area[3]) & (y>area[0]) & (y<area[1])
    x, y, value = x[index], y[index], value[index]  
    cosmicRays = Table([x, y, value], names = ('xcentroid', 'ycentroid', 'value'))
    if len(cosmicRays)==0:
        print('No cosmic rays detected... Please verify the detection threshold')
        #sys.exit()
        cosmicRays.add_columns([Column(name='doublons'),Column(name='dark'),Column(name='id'),Column(name='distance')])
        return cosmicRays
    else:
        cosmicRays['doublons']=0
        cosmicRays['dark'] = -1
        cosmicRays['id'] = -1
        cosmicRays['distance'] = -1
        cosmicRays['sum_10'] = -1
        cosmicRays['contour'] = -1
        cosmicRays['Nb_saturated'] = -1
        print(len(cosmicRays), ' detections, youpi!')
    return cosmicRays

def delete_doublons_CR(sources, dist=4):
    """Function that delete doublons detected in a table, 
    the initial table and the minimal distance must be specifies
    """
    from tqdm import tqdm
    for i in tqdm(range(len(sources))):
        distances = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i])
        a = distances >= dist
        #a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
        #a = distance2(sources[sources['doublons']==0]['xcentroid','ycentroid'],sources['xcentroid','ycentroid'][i]) > dist
        a = list(1*a)
        a.remove(0)
        if np.nanmean(a)<1:
            sources['doublons'][i]=1
            sources['distance'][i]= np.nanmin(distances[distances>0])
    #y, x = np.indices((image.shape))
    #mask1 = (sources['xcentroid']<1053) 
    #mask2 = (sources['xcentroid']>2133)
    #sources['doublons'][~mask1]=1
    #sources['doublons'][~mask2]=1
    print(len(sources[sources['doublons']==0]), ' Comsic rays detected, youpi!')
    return sources


def plotOSregion(xpapoint, DS9backUp = DS9_BackUp_path):
    """
    """    
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    data = d.get_arr2np()
    header = d.get_fits()[0].header
    filename = d.get('file')
    y = np.nanmean(data, axis=0)[:1070][::-1]
    x = np.arange(len(y))
    exp = lambda x, xo, a, b, offset: offset + b*np.exp(-a*(x-xo)) 
    end = 30
    p0=[0,5e-1,y.max()-y.min(),y.min()]
    popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0) 
    xo, a, b, offset = popt
    plt.figure(figsize=(10,6))
#    plt.plot(x, y,'o', label='DS9 values')
#    plt.plot(x[:end], exp(x[:end],*popt), label='Exp Fit: %i*exp(-(x-%i)/%0.2f)+%i'%(b, xo, 1/a, offset))
    plt.plot(x, np.log10(y - offset),'o', label='DS9 values')
    plt.plot(x[:end], np.log10(exp(x[:end],*popt)-offset), label='Exp Fit: %i*exp(-(x-%i)/%0.2f)+%i'%(b, xo, 1/a, offset))

    plt.grid(True, linestyle='dotted')
    plt.title('Smearing analysis: Overscan profile - T=%s, gain=%i'%(header['TEMP'],header['EMGAIN']))
    plt.legend()
    plt.xlabel('Pixels - %s'%(filename))
    plt.ylabel("ADU value")
    plt.show() 
    csvwrite(np.vstack((x,y)).T, DS9backUp + 'CSVs/%s_OSprofile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return    

def SmearingProfile(xpapoint, DS9backUp=DS9_BackUp_path, name=''):
    """
    plot a stack of the cosmic rays
    """
    from astropy.io import fits
    from astropy.table import Table
    from .focustest import create_DS9regions2
    from scipy.optimize import curve_fit
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    
    d = DS9(xpapoint)
    print(sys.argv)
    if len(sys.argv) < 4:
        path = ''
    elif sys.argv[3].lower() == 'os':
        path = ''
        plotOSregion(xpapoint)
        sys.exit()
    else:
        path = Charge_path(xpapoint, entry_point=3)
        
        
    if 'Merged' in os.path.basename(path):
        cat = Table.read(path)
        lx,ly, offset = 5,100, 20
        cr_image = np.nan * np.zeros((2*lx,2*ly+offset,len(cat)))
        n=len(cat)
        for image in np.unique(cat['filename'])  :
            #print(image)
            data = fits.open(image)[0].data
            cr2 = delete_doublons_CR(cat[cat['filename']==image],dist=30)
            x, y = cr2['xcentroid'], cr2['ycentroid']
    #        x, y = cr2['xcentroid'][cr2['distance']>10], cr2['ycentroid'][cr2['distance']>10]
            index = (np.sum(np.isnan(cr_image), axis=(0,1))==2*lx*(2*ly+offset)).argmax()
        
            for k in range(len(x)):
                j = x[k]
                i = y[k]
                print('k+index = ', k + index)
                try:
                    cr_image[:,:,k+index] = data[i-lx:i+lx,j-2*ly:j+offset]
                except ValueError as e:
                    print(e)
                    n -= 1
                    pass
        cr_im = np.nansum(cr_image,axis=2)
        cr_im /= n
    
    else:
        image = d.get_fits()[0].data#fits.open(path[0])[0].data
        header = d.get_fits()[0].header#fits.open(path[0])[0].data
        filename = d.get('file')#fits.open(path[0])[0].data
            
        region = getregion(d, all=True)
        #print(region)
        if len(region)>0:
            x, y = np.array([int(reg.xc) for reg in region]), np.array([int(reg.yc) for reg in region])
        else:                                              
            cr = Table.read(path[0][:-5] + '.CRv.csv')    
            
            cr2 = delete_doublons_CR(cr,dist=30)
            x, y = cr2['xcentroid'][cr2['distance']>10], cr2['ycentroid'][cr2['distance']>10]
            #plot(cr2['xcentroid'][cr2['distance']>10],cr2['ycentroid'][cr2['distance']>10],'x')
            create_DS9regions2(x,y, radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers')
            d.set('regions /tmp/centers.reg')
        
        lx,ly, offset = 5,100, 20
        cr_image = np.nan * np.zeros((2*lx,2*ly+offset,len(x)))
        for k in range(len(x)):
            j = x[k]
            i = y[k]
            try:
                cr_image[:,:,k] = image[i-lx:i+lx,j-2*ly:j+offset]
            except ValueError:
                pass
        cr_im = np.nanmean(cr_image,axis=2)
    
    #d.set('frame new')
    #d.set_np2arr(cr_im)
    y = cr_im[4,:]
    y = y[:np.argmax(y)]
    y = y[::-1]
    x = np.arange(len(y))
    
    exp = lambda x, xo, a, b, offset: offset + b*np.exp(-a*(x-xo)) 
    end = 5
    p0=[0,5e-1,y.max()-y.min(),y.min()]

    plt.figure(figsize=(10,6))
    #plt.vlines(median(x),y.min(),y.max())
    #plt.text(median(x)+50,mean(y),'Amplitude = %i'%(max(y)-min(y)))
    try:
        popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0) 
    except RuntimeError:
        offset = np.min(y)
        pass
    else:   
        xo, a, b, offset = popt
        plt.plot(x[:end], np.log10(exp(x[:end],*popt)-offset), label='Exp Fit: %i*exp(-(x-%i)/%0.2f)+%i'%(b, xo, 1/a, offset))
    plt.plot(x, np.log10(y-offset) ,'o', label='DS9 values - %i images'%(cr_image.shape[-1]))
    #plt.plot(x, exp(x,*p0), label='p0')
    plt.grid(True, linestyle='dotted')
    plt.title('Smearing analysis: Hot pixel / Cosmic rays profile - T=%s, gain=%i'%(header['TEMP'],header['EMGAIN']))
    plt.legend()
    plt.xlabel('Pixels - %s'%(filename))
    plt.ylabel("Log ADU value")
    plt.savefig(DS9backUp + 'Plots/%s_CR_HP_profile%s.png'%(datetime.datetime.now().strftime("%y%m%d"), name) )
    plt.show()    
#    f, (a,b,c) = plt.subplots(3,1,figsize=(8,6))
#    im=c.imshow(cr_im);plt.colorbar(im, orientation='horizontal')
#    a.semilogy(np.nanmean(cr_im[:,175:-25],axis=1))
#    b.semilogy(np.nanmean(cr_im[3:-3,:],axis=0));b.grid()
#    plt.title('Cosmic profile, n = %i'%(len(region)))
#    plt.show()
    csvwrite(np.vstack((x,y)).T, DS9backUp + 'CSVs/%s_CR_HP_profile%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"), name) )
    return


def assign_CR(sources, dist=10):
    """Assign all the cosmic detected pixel (superior to the threshold) 
    to one cosmic ray hit event
    """
    from tqdm import tqdm
    groups = sources[sources['doublons']==0]
    groups = sources
#    for i, cr in tqdm(enumerate(groups)):
#        x,y = cr['xcentroid'],cr['ycentroid']
    for i in tqdm(range(len(sources))):
        x = groups[i]['xcentroid']
        y = groups[i]['ycentroid']
        index = (sources['xcentroid']>x-dist) & (sources['xcentroid']<x+dist) & (sources['ycentroid']>y-dist) & (sources['ycentroid']<y+dist)
        sources['id'][index]=i
    return sources

def DistinguishDarkFromCR(sources, T, number=2):
    """When only one pixel of the detection is above threshold it must be dark
    """
    from tqdm import tqdm
    for i in tqdm(range(len(np.unique(sources['id'])))):
        idd = np.unique(sources['id'])[i]
        sources['Nb_saturated'][(sources['id'] == idd)] = len(sources[(sources['id'] == idd) & (sources['value'] > T)])
#        if len(sources[(sources['id'] == idd) & (sources['value'] > T)]) < 2:
#            sources['dark'][sources['id'] == idd] = 1
#        else:
#            sources['dark'][sources['id'] == idd] = 0
            
    return sources


def PlotCR(sources):
    """Visualization of the CRs
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,10))
    plt.plot(sources[sources['dark']<1]['xcentroid'],sources[sources['dark']<1]['ycentroid'],'.',label='CR')
    plt.plot(sources[sources['dark']==1]['xcentroid'],sources[sources['dark']==1]['ycentroid'],'.',label='Dark')
    plt.axis('equal');plt.legend()
    plt.xlim((1000,2100))
    #plt.xlim((1000,2100))
    for idd in np.unique(sources['id']):
        plt.annotate('%i'%(idd), (sources[sources['id']==idd]['xcentroid'][0]+10,sources[sources['id']==idd]['ycentroid'][0]+10))
        plt.scatter(sources[sources['id']==idd]['xcentroid'][0],sources[sources['id']==idd]['ycentroid'][0], s=180, facecolors='none', edgecolors='r')
    return
   #plt.scatter(sources[sources['id']==idd]['xcentroid'][0],sources[sources['id']==idd]['ycentroid'][0], s=10*len(sources[sources['id']==idd]['xcentroid']), facecolors='none', edgecolors='r')
    

#for idd in np.unique(sources['id']):
#    print(sources[(sources['id'] == idd) & (sources['value'] > T)])
#plot(a[a['dark']<1]['xcentroid'],a[a['dark']<1]['ycentroid'],'.')
#plot(a[a['dark']==1]['xcentroid'],a[a['dark']==1]['ycentroid'],'.')
#
##
#sources=a
#for idd in sources['id']:
#    plot(len(sources[(sources['id'] == idd) & (sources['value'] > T)]),
#         sources[(sources['id'] == idd) & (sources['value'] > T)]['dark'].mean(),'.')
#    
    
#plot(sources[sources['id']==-1]['xcentroid'],sources[sources['id']==-1]['ycentroid'],'.')  
        
#def delete_doublons_CR1(sources, dist=10):
#    """Function that delete doublons detected in a table, 
#    the initial table and the minimal distance must be specifies
#    """
#    from tqdm import tqdm
#    sources['doublons']=0
#    for i in tqdm(range(len(sources))):
#        a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
#        #a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
#        #a = distance2(sources[sources['doublons']==0]['xcentroid','ycentroid'],sources['xcentroid','ycentroid'][i]) > dist
#        a = list(1*a)
#        a.remove(0)
#        if np.nanmean(a)<1:
#            sources['doublons'][i]=1
#    print(len(sources[sources['doublons']==0]), ' Comsic rays detected, youpi!')
#    return sources

def Determine_front(sources):
    """Function that delete doublons detected in a table, 
    the initial table and the minimal distance must be specifies
    """
    from tqdm import tqdm
    sources['front'] = 0
    a = sources[(sources['doublons']==0)]# & (sources['dark']<1)]
    for id in tqdm(range(len(a))):
        #print('Id = ',id)
        index = sources['id']==id
        #print('Number of high value pixel in cosmic: ', len(sources[index]))
        for pixel in sources[index]:
            y = pixel['ycentroid']
            #x, y = pixel['xcentroid'], pixel['ycentroid']
            line = sources['ycentroid']==y
            frontmask = index & line  & (sources['xcentroid']==sources[line&index]['xcentroid'].max())
            sources['front'][frontmask]=1
            #sources[frontmask]#=1
        #print('Front pixels: ', sources[sources['id']==id]['front'].sum())
    #print(len(sources[sources['front']==1]), ' Front Comsic ray pixels detected, youpi!')
    return sources




def MaskCosmicRaysCS(image, cosmics,all=False):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    from tqdm import tqdm
    y, x = np.indices((image.shape))
    image = image.astype(float)
    for i in tqdm(range(len(cosmics))):
        image[(y>cosmics[i]['min_y']) & (y<cosmics[i]['max_y']) & (x<cosmics[i]['max_x']+cosmics[i]['size_opp']) & (x>-cosmics[i]['size'] + cosmics[i]['max_x'])] = np.nan#0#np.inf#0#np.nan
    return image


def MaskCosmicRays(image, cosmics, length = 1000, cols=None,all=False):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    from tqdm import tqdm
    y, x = np.indices((image.shape))
    image = image.astype(float)
    if all is False:   
        cosmics = cosmics[(cosmics['front']==1) & (cosmics['dark']<1)]
    if cols is None:
        for i in tqdm(range(len(cosmics))):#range(len(cosmics)):
            print(i)
            #image[(y==cosmics[i]['ycentroid']) & (x<cosmics[i]['xcentroid']+4) ] = np.nan
            image[(y==cosmics[i]['ycentroid']) & (x<cosmics[i]['xcentroid']+1) & (x>-length + cosmics[i]['xcentroid'])] = np.nan
            print(image[(y==cosmics[i]['ycentroid']) & (x<cosmics[i]['xcentroid']+1) & (x>-length + cosmics[i]['xcentroid'])])
    else:
        for i in tqdm(range(len(cosmics))):#range(len(cosmics)):
            #print(i)
            #image[(y>cosmics[i]['ycentroid']-cols-0.1) & (y<cosmics[i]['ycentroid']+cols+0.1) & (x<cosmics[i]['xcentroid']+4)] = np.nan
            image[(y>cosmics[i]['ycentroid']-cols-0.1) & (y<cosmics[i]['ycentroid']+cols+0.1) & (x<cosmics[i]['xcentroid']+4) & (x>-length + cosmics[i]['xcentroid'])] = np.nan
    return image

def distance(x1,y1,x2,y2):
    """
    Compute distance between 2 points in an euclidian 2D space
    """
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

def DS9RemoveImage(xpapoint):
    """Substract an image, for instance bias or dark image, to the image
    in DS9. Can take care of several images
    """
    d = DS9(xpapoint)
    path2remove = sys.argv[3]
    a, b = sys.argv[4].split(',')
    print(a,b)
    a, b = float(a), float(b)
    if len(sys.argv) > 5:
        path = Charge_path(xpapoint, entry_point=5)
    else:
        path = [getfilename(d)]#fd.get("file")]
        
    for filename in path:
        print(filename)
        result, name = SubstractImage(filename, path2remove=path2remove, a=a, b=b) 
                                        
    if len(path) < 2:
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return



def DS9_2D_autocorrelation(xpapoint):
    """Return 2D_autocorrelation plot
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint, entry_point=3)
    else:
        path = [getfilename(d)]#fd.get("file")]
    plot_flag = (len(path) == 1)
    print('flag for plot = ', plot_flag)
    try:
        region = getregion(d)
    except ValueError:
        area = [0,-1,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
        if plot_flag:
            d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=yellow"' % (region.xc ,region.yc, 3*region.w, 3*region.h))
    for filename in path:
        print(filename)
        corr = TwoD_autocorrelation(filename, save=True, area=area, plot_flag=plot_flag)
    return corr
#imshow(np.log10(b),vmin=np.percentile(np.log10(b),40),vmax=np.percentile(np.log10(b),60));
#imshow(b,vmin=np.percentile(b,40),vmax=np.percentile(b,60));
#colorbar()
#c = b.flatten()
#plt.hist(c)

        
def TwoD_autocorrelation(filename, save=True, area=None, plot_flag=True, DS9backUp = DS9_BackUp_path):
    """Return 2D_autocorrelation plot
    """
    from scipy import signal#from scipy import misc
    from astropy.io import fits
    #import matplotlib; matplotlib.use('TkAgg')  
    #import matplotlib.pyplot as plt
#    lena = misc.lena() - misc.lena().mean()
#    template = np.copy(lena[235:295, 310:370]) # right eye
#    template -= template.mean()
#    lena = lena + np.random.randn(*lena.shape) * 50 # add noise
#    corr = signal.correlate2d(lena, template, boundary='symm', mode='same')
#    y, x = np.unravel_index(np.argmax(corr), corr.shape) # find the match
    new_area = [area[0] - (area[1] - area[0]),area[1] + (area[1] - area[0]),
                area[2] - (area[3] - area[2]),area[2] + (area[3] - area[2])]
    data = fits.open(filename)[0].data
    template = np.copy(data[area[0]:area[1], area[2]:area[3]])
    image = np.copy(data[new_area[0]:new_area[1], new_area[2]:new_area[3]])
    corr = signal.correlate2d(image, template, boundary='symm', mode='same')
#    plt.imshow(corr, interpolation = 'none')
#    plt.title('2D auto-correlation')
#    plt.colorbar()
#    plt.savefig('/tmp/Autocorr_%s.png'%(os.path.basename(filename[:-5])))
#    plt.show()
    #np.savetxt(DS9backUp + 'Arrays/%s_Autocorr.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")), corr)
    d = DS9()
    d.set_np2arr(corr)
    return corr


def DS9AnalyzeImage(xpapoint):
    """Return some plot to analyze the image
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint, entry_point=3)
    else:
        path = [getfilename(d)]#fd.get("file")]
    plot_flag = (len(path) == 1)
    print('flag for plot = ', plot_flag)
    for filename in path:
        print(filename)
        AnalyzeImage(filename, save=True, area=None, plot_flag=plot_flag)  
    return

def AnalyzeImage(filename, save=True, area=None, plot_flag=True):
    """Return some plot to analyze the image
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats 
    import scipy.ndimage as ndimage
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data#.astype('uint64')
    header = fitsimage.header
    if os.path.getsize(filename)>5638240:
        Type = 'detector'
        if area is None:
            area = [0,-1,1050,2150]
#        new_image = image[area[0]:area[1],area[2]:area[3]]
        width_ratios = (1,2.8)

    else:
        Type = 'guider'
        if area is None:
            area = [0,-1,0,-1] 
        width_ratios = (1,0.9)
    new_image = image[area[0]:area[1],area[2]:area[3]]
    lx, ly = image.shape
 #   width_ratios = (1,ly/lx+0.5)

    fig = plt.figure(figsize=(15,8))    
    gs = GridSpec(6, 2, width_ratios=width_ratios,height_ratios=(1,1,1,1,0.3,0.2))
    ax1 = fig.add_subplot(gs[:-1, :1])
    axc = fig.add_subplot(gs[-1, :1])
    ax2 = fig.add_subplot(gs[0:2, 1])
    #ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[2:-2, 1])
    #ax5 = fig.add_subplot(gs[-2:, 1])
    amp = 0    
    print(np.nanmean(image,axis=1), image)
    if Type == 'detector':
        offset = 20
        OScorr_n = ComputeOSlevel1(new_image, OSR1=[offset,-offset,offset,400], OSR2=[offset,-offset,2200,2400]) 
        OScorr = ComputeOSlevel1(image, OSR1=[offset,-offset,offset,400], OSR2=[offset,-offset,2200,2400]) 
        new_image = (new_image - OScorr_n)# / header['EXPTIME']
        #ax2.set_ylim((0,1.1*np.nanmean(new_image,axis=1).max()))
        #ax3.set_ylim((0,1.1*np.nanmean(new_image,axis=0).max()))
        if header['EMGAIN'] == 0:
            fitsimage.data = image - OScorr
            amp=1
            a, b = calc_emgainGillian(fitsimage,area=area)
            emgain,bias,sigma,frac_lost = a
            try:
                plot_hist2(*b, ax=ax4,plot_flag=True)
            except TypeError:
                ax4.hist(new_image.flatten(),bins=1000,histtype='step',log=True)
                ax4.set_xlabel('Pixel value')
                ax4.set_ylabel('Log10(#Pixels)')                
            #ax4.set_xlim((-1e3,10000))
    if amp == 0:
        ax4.hist(new_image.flatten(),bins=1000,histtype='step',log=True)
        ax4.set_xlabel('Pixel value')
        ax4.set_ylabel('Log10(#Pixels)')
    cmap = plt.cm.cubehelix
    cmap.set_bad('black',0.9)
    new_image_sm = ndimage.gaussian_filter(new_image, sigma=(1.3, 1.3), order=0)

    im = ax1.imshow(new_image[::-1,:], cmap=cmap,vmin=np.nanpercentile(new_image_sm,10),vmax=np.nanpercentile(new_image_sm,99.99))
    print(np.nanmean(new_image,axis=1), new_image)
    #ax1.axis('equal')
    ax2.plot(np.nanmean(new_image_sm,axis=1), label='Column values [ADU/s]')
    ax2.plot(np.nanmean(new_image_sm,axis=0), label='Line values [ADU/s]')
    ax2.legend()
    #ax2.set_ylabel('Column/Line value')
    #ax3 = ComputeEmGain(filename, Path2substract=None, save=False, plot_flag=False, d=None, ax=ax3)
    #ax3.set_ylabel('Variance Intensity plot')
    fig.colorbar(im, cax=axc, orientation='horizontal');
    area = [500,600,500,600]
    im = image[area[0]:area[1],area[2]:area[3]]
    if Type == 'guider':
        ax1.set_title("Image Analysis - " + os.path.basename(filename) + ' - ' + header['DATE'])
        plt.figtext(.54, .1, 'Az = %0.3f deg\nEl = %0.3f deg\n'
                'Exp = %0.1f sec' % (header['AZ'], header['EL'], header['EXPOSURE']),
                fontsize=10,bbox={'facecolor':'orange', 'alpha':0.9, 'pad':10})
        plt.figtext(.64, .1, 'Press = %i mB\nRotenc = %i \n'
                'Mrot = %0.1f deg' % (header['PRESSURE'], header['ROTENC'], header['MROT']),
                fontsize=10,bbox={'facecolor':'orange', 'alpha':0.9, 'pad':10})
        plt.figtext(.75, .1, 'LINAENC = %0.2f \nLINBENC = %0.2f \n'
                'LINCENC = %0.2f' % (header['LINAENC'], header['LINBENC'], header['LINCENC']),
                fontsize=10,bbox={'facecolor':'orange', 'alpha':0.9, 'pad':10})
        plt.figtext(.87, .1, 'Mean = %0.1f e-/s\nStd = %0.1f e-/s\n'
                'SKew = %0.2f e-/s \nStdovers^2/<Im> = %0.3f' % (np.nanmean(new_image), np.nanstd(new_image), stats.skew(new_image,axis=None, nan_policy='omit'),
                                                                np.nanstd(im)**2/np.nanmean(im)),
                fontsize=10,bbox={'facecolor':'blue', 'alpha':0.3, 'pad':10})    
    else:
        fig.suptitle("Image Analysis [OS corrected] - " + os.path.basename(filename) + ' - ' + header['DATE'], y=0.99)
        plt.figtext(.4, .1, 'Exposure = %i sec\nGain = %i \n'
                'T det = %0.2f C' % (header['EXPTIME'], header['EXPTIME'], float(header['TEMP'])),
                fontsize=12,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
        plt.figtext(.6, .1, 'Mean = %0.5f \nStd = %0.3f \n'
                'SKew = %0.3f ' % (np.nanmean(new_image), np.nanstd(new_image),  stats.skew(new_image,axis=None, nan_policy='omit')),
                fontsize=12,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
        plt.figtext(.8, .1, 'Gain = %i\nStd^2/<Im> = %0.3f \n'
                'Bias = %i' % (header['EMGAIN'], np.nanstd(im)**2/np.nanmean(im), OScorr.mean() ),
                fontsize=12,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
        


    fig.tight_layout()
    name = os.path.dirname(filename) + '/ImageAnalysis/'
    if save:
        try: 
            os.makedirs(name)
        except OSError:
            pass
        plt.savefig(name + os.path.basename(filename)[:-5] + '.png')        
    if plot_flag:
        plt.show() 
    print(np.nanmean(new_image,axis=1), new_image)
    #plt.figure()
    #plt.plot(np.nanmean(new_image_sm,axis=1), label='Column values [ADU/s]')
    #plt.show()
    return
#AnalyzeImage('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/gobc_data/bin_r13_flight_only/img_save/stack25207657.fits')   
#AnalyzeImage('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/data/snape/180815/ptc/image000032.fits')

    
#AnalyzeImage('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CosmicRaysFree/image000093.CRv.fits')



def SubstractImage(filename, path2remove, a , b):
    """Substract an image, for instance bias or dark images,
    and save it
    """
    from astropy.io import fits
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    fitsimage2 = fits.open(path2remove)[0]
    image2substract = fitsimage2.data
    fitsimage.data = image - a * image2substract - b 
    name = filename[:-5] + '_subtracted_' + os.path.basename(path2remove)
    exp1, exp2 = int(fitsimage.header['EXPTIME']), int(fitsimage2.header['EXPTIME'])
    fitsimage.header['EXPTIME'] = np.nanmax([exp2,exp1]) - a * (np.nanmin([exp2,exp1]))
    fitswrite(fitsimage, name)
    return fitsimage.data, name    

def DS9NoiseMinimization(xpapoint, radius=200):
    """Substract a linear combination of images
    """
    from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    if len(sys.argv) > 4:
        path = Charge_path(xpapoint, entry_point=4)
    else:
        path = [getfilename(d)]#[d.get("file")]
    Path2substract = sys.argv[3]
    if sys.version_info.major == 3:
        images2substract = glob.glob(os.path.join(Path2substract, '**/*.fits'), recursive=True)
    if sys.version_info.major == 2:
        images2substract = glob.glob(os.path.join(Path2substract, '**/*.fits'))
        images2substract +=  glob.glob(os.path.join(Path2substract, '*.fits'))
    images2substract.sort()    
    for filename in path:
        print(filename)
        name, areas = NoiseMinimization(filename, images2substract=images2substract, Path2substract=Path2substract, save=True, radius=radius)  
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
        areas = np.array(areas)
        create_DS9regions2(areas[:,2]+float(radius)/2,areas[:,0]+float(radius)/2, radius=radius, form = 'box',
                           save=True,color = 'yellow', savename='/tmp/centers')
        d.set('regions /tmp/centers.reg')
    return    
    #vpicouet

def NoiseMinimization(filename, images2substract, save=True, radius=50, Path2substract=''):
    """
    """
    from scipy.optimize import leastsq
    from astropy.io import fits
    images2substract.sort()
    im0 = fits.open(filename)[0].data
    lx, ly = im0.shape
    images = np.zeros((lx, ly, len(images2substract)))
    print('Images used to minimize standard deviation: \n', '\n'.join(images2substract))
    for i, im in enumerate(images2substract):
        images[:,:,i] = fits.open(im)[0].data
    areas = CreateAreas(im0, area=None, radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    result, cov_x = leastsq(func=residual, x0=0.3*np.ones(len(images2substract)), args=(im0,images,areas,radius))        
    image, name = SubstractImages(param=np.square(result), image_name=filename, images=images, save=True, Path2substract=Path2substract)
    std0 = np.std(residual(np.zeros(len(images2substract)),im0,images,areas,radius))
    std_end = np.std(residual(result,im0,images,areas,radius))
    print(bcolors.BLACK_RED + 'Standard deviation:  %0.2f  =====>   %0.2f '%(std0,std_end) + bcolors.END)
    return name, areas

    
   
def SubstractImages(param, image_name, images, save=True, Path2substract=''):
    """
    """
    from astropy.io import fits
    image = fits.open(image_name)[0]
    new_image = image.data - np.nansum(param * images,axis=2)
    if save:
        image.data = new_image
        par = '_{:0.2f}'*len(param)
        name = image_name[:-5] + par.format(*param) + '.fits'
        print('save : %s'%(name))
        image.header['PATHSUBS'] = Path2substract
        image.writeto(name,overwrite=True)        
    return new_image.ravel() - np.nanmean(new_image), name

def residual(param, image, images, areas, radius):
    """
    """
  #  n=300
  #  xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  
  #  yis = [1100, 1450, 1800]*3
  #  areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]
    #area=[40,330,1830,2100]

    im_tot = np.ones((radius,radius,len(areas)))
    for i, area in enumerate(areas):
        new_image = image[area[0]:area[1],area[2]:area[3]] - np.nansum(np.square(param) * images[area[0]:area[1],area[2]:area[3],:],axis=2)
        new_image -= np.nanmean(new_image)
        im_tot[:,:,i] = new_image
    
    return im_tot[np.isfinite(im_tot)] #- new_image.mean()



def DS9ComputeEmGain(xpapoint):
    """Compute EMgain with the variance intensity method
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 4:
        path = Charge_path(xpapoint, entry_point=4)
    else:
        path = [getfilename(d)]#[d.get("file")]
    try:
        Path2substract = None#sys.argv[3]
    except IndexError:
        Path2substract = None
    for filename in path:
        print(filename)
        ComputeEmGain(filename, Path2substract=Path2substract, save=False, d=d)  
    return


def ComputeEmGain(filename, Path2substract=None, save=False, plot_flag=True, d=None, ax=None, radius=[300,40]):
    """Compute EMgain with the variance intensity method
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import create_DS9regions2
    from astropy.io import fits
    import glob
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    offset = 20
    image = image - ComputeOSlevel1(image, OSR1=[offset,-offset,offset,400],OSR2=[offset,-offset,2200,2400])
    if Path2substract is None:
        #images = glob.glob(os.path.dirname(filename) + '/%s??????.fits'%(os.path.basename(filename)[:6]))
        images = glob.glob(os.path.dirname(filename) + '/%s??????%s'%(os.path.basename(filename)[:6],os.path.basename(filename)[12:]))
        images.sort()
        try: 
            name = images[images.index(filename) + 1]
            print('Path substracted :', name)
            image_sub = image - fits.open(name)[0].data            
        except IndexError:
            name = images[images.index(filename) - 1]
            print('Path substracted :', name)
            image_sub = image - fits.open(name)[0].data   
    else:
        image_sub = image - fits.open(Path2substract)[0].data 
#    image_sub = image
#    fitsimage.data = image_sub
#    imagesub, name = BackgroundSubstraction(fitsimage, stddev=13, save=False) 
#    image_sub = imagesub.data    
    areas = CreateAreas(image, area=None, radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    areasOS1 = CreateAreas(image, area=[0, 400, 0, 2069], radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    areasOS2 = CreateAreas(image, area=[2200, 2400, 0, 2069], radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    len_area_det = len(areas)
    areas = areas + areasOS1 + areasOS2
    print('Number of regions : ', len(areas))
    var = []
    intensity = []
    OSarea = [0,-1,0,400]
    var.append(np.nanvar(image_sub[OSarea[0]:OSarea[1],OSarea[2]:OSarea[3]])/2)
    intensity.append(np.nanmean(image[OSarea[0]:OSarea[1],OSarea[2]:OSarea[3]]))
    for i, area in enumerate(areas):
      #  print(area)
        #print(image_sub[area[0]:area[1],area[2]:area[3]])
       # print(np.var(image_sub[area[0]:area[1],area[2]:area[3]]))
        vari = np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])/2
        var.append(vari)
        #print(vari)
        intensity.append(np.nanmean(image[area[0]:area[1],area[2]:area[3]]))
    
    #print(np.nanvar(imagesub.data[area[0]:area[1],area[2]:area[3]])/np.sqrt(1) - np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])/np.sqrt(1))
    #print(np.nanvar(imagesub.data)/np.sqrt(1) - np.nanvar(image_sub)/np.sqrt(1))
    #print(intensity, var)      

    var = np.array(var).flatten()
    Indexg = var < np.nanmedian(var) + 100 * var.std()
    var = var[Indexg]
    intensity = np.array(intensity)[Indexg].flatten()
    #print(len(intensity), len(var))      
    
    areas = np.array(areas)
    if type(radius)==int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    if d is not None:
        #create_DS9regions2(np.array(yis)+float(n)/2,np.array(xis)+float(n)/2, radius=n, form = 'box',
#        create_DS9regions2(areas[:,2]+float(radius)/2,areas[:,0]+float(radius)/2, radius=radius, form = 'box',
#                           save=True,color = 'yellow', savename='/tmp/centers')
        create_DS9regions2(areas[:,2]+float(r1)/2,areas[:,0]+float(r2)/2, radius=radius, form = 'box',
                           save=True,color = 'yellow', savename='/tmp/centers')
        d.set('regions /tmp/centers.reg')
    emgain = fitsimage.header['EMGAIN'] 
    ax = PlotComputeEmGain(intensity, var, emgain, r1*r2, filename=filename, len_area_det=len_area_det, ax=ax)
    if save:
        if not os.path.exists(os.path.dirname(filename) +'/VarIntensDiagram'):
            os.makedirs(os.path.dirname(filename) +'/VarIntensDiagram')
        plt.savefig(os.path.dirname(filename) +'/VarIntensDiagram/' + os.path.basename(filename)[:-5] + '.png')
    if plot_flag:
        plt.show()
    return ax


def CreateAreas_old(image, area=None, radius=100, offset=20):
    """Create areas in the given image
    """
    #image = a.data
    ly, lx = image.shape
    if area is None:
        if ly == 2069:
            xmin, xmax = 1053, 2121#0,3000#1053, 2121
            rangex = xmax - xmin
        xi = np.arange(offset,rangex - offset - radius, radius)
        yi = np.arange(offset,ly - offset - radius, radius)
        xx, yy = np.meshgrid(xi,yi)
        areas = [[a, a + radius, b + xmin, b + radius+ xmin] for a,b in zip(yy.flatten(),xx.flatten())]
        return areas
    else:
        print(1)
        
def CreateAreas(image, area=None, radius=100, offset=20):
    """Create areas in the given image
    """
    #image = a.data$
    if type(radius)==int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    print('r1,r2=',r1,r2)
    ly, lx = image.shape
    if area is None:
        if ly == 2069:
            xmin, xmax = 1053, 2121#0,3000#1053, 2121
            ymin, ymax = 0,ly
           # rangex = xmax - xmin
        else:
            xmin, xmax = 0, lx
            ymin, ymax = 0, ly            
    else:
        xmin, xmax = area[0], area[1]
        ymin, ymax = area[2], area[3]
    xi = np.arange(offset + xmin, xmax - offset - r1, r1)
    yi = np.arange(offset + ymin, ymax - offset - r2, r2)
    xx, yy = np.meshgrid(xi,yi)
    areas = [[a, a + r2, b, b + r1] for a,b in zip(yy.flatten(),xx.flatten())]
    #print(areas[0])
    return areas







def PlotComputeEmGain(intensity, var, emgain, n, filename, len_area_det, ax=None, DS9backUp = DS9_BackUp_path):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    if emgain > 0:
        cst = 2
    else:
        cst = 1
    print(intensity, var)      
    fit = np.polyfit(intensity,var/cst,1)
    fit_fn = np.poly1d(fit) 
    fit_wo0 = np.polyfit(intensity[:len_area_det],var[:len_area_det]/cst,1)
    fit_fn_wo0 = np.poly1d(fit_wo0) 
    if ax is None:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        
    ax.plot(intensity, var/cst, 'x', label='Data in %i pix side area, G = %i'%(n,emgain))
    ax.plot(intensity, fit_fn(intensity), '--', label='Linear regression, Engain = %0.3f'%(fit[0]))
    ax.plot(intensity, fit_fn_wo0(intensity), '--', label='Linear regression, Engain = %0.3f\n[WO OS region]'%(fit_wo0[0]))
    ax.set_ylabel('Variance [ADU] / %i'%(cst))
    ax.text(0.5,0.1,'y = %0.2f *u x + %0.2f [W OS]'%(fit[0], fit[1]),transform=ax.transAxes)
    ax.text(0.5,0.2,'y = %0.2f * x + %0.2f [WO OS]'%(fit_wo0[0], fit_wo0[1]),transform=ax.transAxes)
    ax.legend(loc='upper left')
    ax.set_xlabel('Intensity [ADU]')
    ax.grid(linestyle='dotted')
    ax.set_title('Variance intensity diagram - %s'%(os.path.basename(filename)))    
    csvwrite(np.vstack((intensity,var/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) ) 
    return ax





def DS9TotalReductionPipeline(xpapoint):
    """Run the total reduction pipeline defined for fireball
    """
    d = DS9(xpapoint)
    if len(sys.argv) == 7:
        path = Charge_path(xpapoint, entry_point=6)
    elif (len(sys.argv) ==  6) :
        if (os.path.isdir(sys.argv[5]) is False) & (os.path.isfile(sys.argv[5]) is False) :
            path = Charge_path(xpapoint, entry_point=5)
    elif (len(sys.argv) <  6) :
        path = [getfilename(d)]#[d.get("file")]
    print('Path to reduce: ', path)
    masterBias = sys.argv[3]
    masterDark = sys.argv[4]
    try:
        masterFlat = sys.argv[5]
    except IndexError:
        masterFlat = None
            
    for filename in path:
        print(filename)
        result, name = ApplyTotalReductionPipeline(filename, masterBias, masterDark, masterFlat) 
                                        
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return

def ApplyTotalReductionPipeline(filename, masterBias, masterDark, masterFlat):
    """Warning, you are about to run the total reduction piepeline. 
    Please make sure you are aware about how it works. 
    In particular note that master bias and dark should be stacked images (or folders containing the images),
    at the same gains/temp. Also make sure the exposure time is saved in the header of your images as 'EXPTIME'.
    If not flat path is given, flat removal will not be performed. 
    """
    from astropy.io import fits
    from astropy.convolution import convolve
    masterDark = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Dark/StackedImage_StackedImage_207-211_100s_9000_NaNsFree-StackedImage_247-251_100s_9400_NaNsFree.fits'
    masterBias = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Bias/StackedImage_0-1.fits'
    masterFlat = None
    filename = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CRfree_and_NaNsFree/StackedImage_368-379.fits'
    if os.path.isdir(masterBias):
        masterBias = StackImagesPath(glob.glob(os.path.join(masterBias, 'image*.fits')))
    elif os.path.isfile(masterBias):
        masterBias = fits.open(masterBias)
    if os.path.isdir(masterDark):
        masterDark = StackImagesPath(glob.glob(os.path.join(masterDark, 'image*.fits')))
    elif os.path.isfile(masterDark):
        masterDark = fits.open(masterDark)

    Bias_OScorr, Bias_OScorrName = ApplyOverscanCorrection(masterBias.filename(), stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
    Bias_OScorr.data = convolve(Bias_OScorr.data, kernel=np.array([[1.,1.,0],[1.,1.,0],[0.,0.,0]])/4)
    Bias_OScorr.writeto('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Bias/StackedImage_0-1_OScorr_convolved.fits')
    Dark_OScorr, Dark_OScorrName = ApplyOverscanCorrection(masterDark.filename(), stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
    
    
    Dark_OScorr_Biascorr = Dark_OScorr.data - Bias_OScorr.data  #SubstractImage(DarkName, Bias_OScorrName)
    if masterDark[0].header['EXPTIME'] != 0:
        Dark_OScorr_Biascorr_per_sec = Dark_OScorr_Biascorr / masterDark[0].header['EXPTIME']
    else:
        print('Exposure time is null for master dark, can not proceed')
        sys.exit()
    if masterFlat is not None:
        name = '_OScorr_Biascorr_Darkcorr_Flatcorr.fits'
        if os.path.isdir(masterFlat):
            masterFlat = StackImagesPath(glob.glob(os.path.join(masterFlat, 'image*.fits')))        
        elif os.path.isfile(masterFlat):
            masterFlat = fits.open(masterFlat)
            
        Flat_OScorr, Flat_OScorrName = ApplyOverscanCorrection(masterFlat, stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
        Flat_OScorr_Biascorr = Flat_OScorr - Bias_OScorr#SubstractImage(DarkName, Bias_OScorrName)
        Flat_OScorr_Biascorr_Darkcorr = Flat_OScorr_Biascorr - (Dark_OScorr_Biascorr * masterFlat[0].header['EXPTIME'])#SubstractImage(DarkName, Bias_OScorrName)
    else:
        name = '_OScorr_Biascorr_Darkcorr.fits'
        Flat_OScorr_Biascorr_Darkcorr = 0
    finalImage = fits.open(filename)   
    FinalImage_OScorr, FinalImage_OScorrName = ApplyOverscanCorrection(finalImage.filename(), stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
    FinalImage_OScorr_Biascorr = FinalImage_OScorr.data - Bias_OScorr.data
    FinalImage_OScorr_Biascorr_Darkcorr = FinalImage_OScorr_Biascorr - (Dark_OScorr_Biascorr_per_sec * finalImage[0].header['EXPTIME'] )
    FinalImage_OScorr_Biascorr_Darkcorr_Flatcorr = FinalImage_OScorr_Biascorr_Darkcorr - Flat_OScorr_Biascorr_Darkcorr
    
    finalImage[0].data = FinalImage_OScorr_Biascorr_Darkcorr_Flatcorr
    fitswrite(finalImage[0], finalImage.filename()[:-5] + name)
    return finalImage, finalImage.filename()[:-5] + name



def DS9Trimming(xpapoint):
    """Crop the image to have only the utility area
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]#[d.get("file")]
        
    try:
        region = getregion(d)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    except ValueError:
        area = [0,-1,1053,2133]
        
    for filename in path:
        print(filename)
        result, name = ApplyTrimming(filename, area=area) 
                                        
    if len(path) < 2:
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return

def ApplyTrimming(path, area=[0,-0,1053,2133]):
    """Apply overscan correction in the specified region, given the two overscann areas
    """
    from astropy.io import fits
    fitsimage = fits.open(path)[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]
    #image = image[area[0]:area[1],area[2]:area[3]]
    fitsimage.data = image
    name = path[:-5] + '_Trimm.fits'
    fitswrite(fitsimage, name)
    return fitsimage, name

def DS9CLcorrelation(xpapoint):
    """
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]#[d.get("file")]
    try:
        region = getregion(d)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    except ValueError:
        area = [0,-1,1053,2133]
        
    for filename in path:
        print(filename)
        CLcorrelation(filename, area=area) 
    return

def CLcorrelation(path, area=[0,-1,1053,2133], DS9backUp = DS9_BackUp_path):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    from astropy.io import fits
    from matplotlib import pyplot as plt
    #path = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Bias/image000000Bias9000.fits'
    fitsimage = fits.open(path)[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]
    #image = image[area[0]:area[1],area[2]:area[3]]
    imagex = np.nanmean(image, axis=1)
    imagey = np.nanmean(image, axis=0)
    nbins=100
    fig, ax = plt.subplots(2, 2, figsize=(12,7))
    ax[0,0].hist(imagex[1:]-imagex[:-1],bins=10*nbins,histtype='step',label='Lines')
    ax[1,0].hist(imagey[1:]-imagey[:-1],bins=10*nbins,histtype='step',label='Column', color='orange')
#    ax[0,1].hist((image[1:,:] - image[:-1,:]).flatten(),bins=nbins,histtype='step',label='Column', color='orange')
    #ax[0,1].hist((image[1:,:] - image[:-1,:]).flatten(),bins=10*nbins,histtype='step',label='Column', color='orange')
    x = (image[:,1:] - image[:,:-1]).flatten()
    y = (image[1:,:] - image[:-1,:]).flatten()
    ax[0,1].hist(x[np.isfinite(x)],bins=30*nbins,histtype='step',label='Line')
    ax[1,1].hist(x[np.isfinite(x)],bins=30*nbins,histtype='step',label='Line')
    ax[1,1].hist(y[np.isfinite(y)],bins=30*nbins,histtype='step',label='Column', color='orange')
    ax[1,1].set_xlim((-100,100))
    ax[0,1].set_xlim((-100,100))
    #ax[1,1].set_ylim((-10,40000))
    #ax[0,1].set_ylim((-10,40000))
    for axx in ax.flatten():
        axx.grid()
    ax[0,0].legend();    ax[0,1].legend()
    ax[1,0].legend();    ax[1,1].legend()
    ax[1,0].set_xlabel('abs(CL(n) - CL(n-1))')
    ax[1,1].set_xlabel('abs(P(n) - P(n-1))')
    print('ok')
    fig.suptitle('Histograms for Columns/Lines correlation analysis \n' + os.path.basename(path), y=0.95)
    plt.show()
    print('ok')
    return





def DS9SubstractBackground(xpapoint, save=True):
    """Apply the background substraction fonction to the DS9 images or several if given
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]#[d.get("file")]
    for filename in path:
        print(filename)
        result, name = BackgroundSubstraction(filename, stddev=10, save=save)
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name) 
    return 

def BackgroundSubstraction(path, stddev=13, save=False):
    """Generate a background of the given image using a 2D gaussian convolution that smoothes the signal
    Background is saved and then substracted from the given image. The bckd substracted image is saved as _BCKDsubstracted.fits
    """
    from astropy.io import fits
    from scipy import ndimage
    from astropy.convolution import interpolate_replace_nans
    from astropy.convolution import Gaussian2DKernel 
    if type(path) == str:
        print(path)
        image = fits.open(path)[0]
    else:
        image = path
    imagesub = image.copy()
    data = image.data
    if np.isfinite(image.data).all() is False:       
        kernel = Gaussian2DKernel(stddev=stddev)
        image.data = interpolate_replace_nans(image.data, kernel)    
    smoothed_image = ndimage.gaussian_filter(image.data, sigma=(stddev, stddev), order=0)

    imagesub.data = data.astype(np.int16) - smoothed_image.astype(np.int16)
    name = ''
    if save:

        name = path[:-5] + '_BCKDsubstracted.fits'
        fitswrite(imagesub, name)
        image.data = smoothed_image.astype('uint16')
        fitswrite(image,path[:-5] + '_smoothed.fits')
    return imagesub, name


def reject_outliers(data, stddev = 3.):
    """Reject outliers above std=m, taking care of nan values
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return data[s<stddev]


def DS9ComputeStandardDeviation(xpapoint,radius=50):
    """
    """
    #from astropy.io import fits
    from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    #path = getfilename(d)#d.get("file")
    fitsimage = d.get_fits()[0]#fits.open(path)[0] 
    areas = CreateAreas(fitsimage.data, area=None, radius=radius)
    lx, ly = fitsimage.data.shape
    image3D = residual([0,0], fitsimage.data, np.zeros((lx,ly,2)), areas, radius)

    
    areas = np.array(areas)
    create_DS9regions2(areas[:,2]+float(radius)/2,areas[:,0]+float(radius)/2, radius=radius, form = 'box',
                       save=True,color = 'yellow', savename='/tmp/centers')
    d.set('regions /tmp/centers.reg')
    print(bcolors.BLACK_RED + 'Standard deviation = %0.1f'%(np.nanstd(image3D)) + bcolors.END)
    return image3D

def DS9HistogramDifferences(xpapoint):
    """Histogram differences
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 4:
        path = Charge_path(xpapoint, entry_point=4)
    else:
        path = [getfilename(d)]#[d.get("file")]
    im2substract = sys.argv[3]
    for filename in path:
        print(filename)
        result = HistogramDifferences(filename, im2substract)
    return result

def HistogramDifferences(filename, im2substract, DS9backUp = DS9_BackUp_path):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    from matplotlib import pyplot as plt
    from astropy.io import fits
    im1 = fits.open(filename)[0].data
    im2 = fits.open(im2substract)[0].data
    nbins = 150#228
    nrange = (np.nanpercentile(np.hstack((im1,im2)),0.1), np.nanpercentile(np.hstack((im1,im2)),99.9)   )
    hist1, x1 = np.histogram(im1.flatten(), bins=nbins, range=nrange)
    hist2, x2 = np.histogram(im2.flatten(),  bins=nbins, range=nrange)
    c1 = (x1[1:] + x1[:-1]) / 2
    n = 10
    beg1 = 6 * int(nbins/n)
    end1 = 8 * int(nbins/n)
    beg2 = 1 * int(nbins/n)
    end2 = 3 * int(nbins/n)
    if np.nansum((hist1-hist2)[beg1:end1]) > np.nansum((hist2-hist1)[beg1:end1]):
        slope1, intercept1 = fitLine(c1[beg1:end1], np.log(hist1-hist2)[beg1:end1])
    else:
        slope1, intercept1 = fitLine(c1[beg1:end1], np.log(hist2-hist1)[beg1:end1])
    
    if np.nansum((hist1-hist2)[beg2:end2]) > np.nansum((hist2-hist1)[beg2:end2]):
        slope2, intercept2 = fitLine(c1[beg2:end2], np.log(hist1-hist2)[beg2:end2])
    else:
        slope2, intercept2 = fitLine(c1[beg2:end2], np.log(hist2-hist1)[beg2:end2])



    xlinefit1 = np.linspace(c1[beg1], c1[end1], 1000)
    ylinefit1 = linefit(xlinefit1, slope1, intercept1)
    xlinefit2 = np.linspace(c1[beg2], c1[end2], 1000)
    ylinefit2 = linefit(xlinefit2, slope2, intercept2)


    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(8,6))
    ax1.step(c1,np.log(hist1), label='H1')
    ax1.step(c1,np.log(hist2), label='H2')
    ax1.legend()
    #f.suptitle('Histogram difference')
    ax1.set_ylabel('Histograms')
    ax2.set_ylabel("Histograms' difference")
    ax2.set_xlabel("Pixel value [ADU]")
    ax1.set_title('%s\n%s'%(os.path.basename(filename), os.path.basename(im2substract)), fontsize=8)
    ax2.step(c1,np.log(hist1-hist2), label='H1 - H2')
    ax2.step(c1,np.log(-hist1+hist2), label='H2 - H1')
    ax2.plot(xlinefit1,ylinefit1, label='Linear fit')
    ax2.plot(xlinefit2,ylinefit2, label='Linear fit')
    ax2.text(-200,4.5,'y = %0.4f x + %0.2f  ==> Gtot = %0.1f'%(slope1, intercept1, -1./slope1))
    ax2.text(-200,3.5,'y = %0.4f x + %0.2f  ==> Gtot = %0.1f'%(slope2, intercept2, 1./slope2))
    ax2.legend()
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
    f.tight_layout()
    plt.show()
    print('Done')
    csvwrite(np.vstack((c1,np.log(hist1-hist2),np.log(-hist1+hist2))).T, DS9backUp + 'CSVs/%s_HistogramDifference.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return

#os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
#from Cocoa import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
#app = NSRunningApplication.runningApplicationWithProcessIdentifier_(os.getpid())
#app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

def DS9CreateHeaderCatalog(xpapoint):
    """
    """
    from astropy.table import vstack
    d = DS9(xpapoint)
    #files = sys.argv[1:]
    if os.path.isdir(sys.argv[3]):
        path = sys.argv[3] #'/Users/Vincent/Nextcloud/FIREBALL/2019/190225/'#sys.argv[1]            
        files = glob.glob(path + '/*/*.fits', recursive=True)
        paths = np.unique([os.path.dirname(f) for f in files ])
        t1s = [CreateCatalog(glob.glob(path + '/image*.fits')) for path in paths] 
        t2s = [CreateCatalogInfo(glob.glob(path + '/image*.fits'),t) for t,path in zip(t1s,paths)] 
        t2 = vstack(t2s)
        csvwrite(t2, os.path.join(sys.argv[3],'TotalMergedCatalog.csv') )
#        for path in paths:
#            files = glob.glob(path + '/image*.fits')
#            print(files)
#            t1 = CreateCatalog(files)
#            t2 = CreateCatalogInfo(files, t1)            
    elif len(sys.argv) > 3:
        path = Charge_path(xpapoint, entry_point=3)
        t1 = CreateCatalog(path)
        t2 = CreateCatalogInfo(path, t1) 
    else:
        path = [getfilename(d)]#[d.get("file")]
        t1 = CreateCatalog(path)
        t2 = CreateCatalogInfo(path, t1) 
    return t2


def CreateCatalog(files):
    """
    """
    from astropy.table import Table
    from astropy.io import fits
    files.sort()
    path = files[0]
#    path= '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/data/snape/180830/photon_counting/image000152.fits'
    #files = glob.glob(os.path.dirname(path) + '/*.fits')
    #files.sort()
    header = fits.getheader(path)
    fields = list(set(header.keys())) + ['PATH'] 
    #t = Table(names=fields,dtype=('S20,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    t = Table(names=fields,dtype=('S20,'*len(fields[:-1])).split(',')[:-1] + ['S100'])#,dtype=('S10,'*30).split(',')[:-1])
    for i, file in enumerate(files):
        t.add_row()
        t['PATH'][i] = file
        print(100*i/len(files),'%')
        header = fits.getheader(file)
        fields_image = list(set(header.keys())) 
        for field in fields_image:
            if field in fields:
                try:
                    t[field][i] = float(header[field])
                except (TypeError, ValueError) as e:
                    print(e)
                    try:
                        
                        t[field][i] = header[field]
                    except ValueError as e:
                        print(e)
                        t[field][i] = -1 
                    except KeyError as e:
                        print(e)
                        pass 
    t['index'] = np.arange(len(t))
    #t.remove_columns(['EXTEND','SIMPLE','NAXIS','COMMENT','NAXIS3','SHUTTER','VSS','BITPIX','BSCALE'])
    for col in ['EXTEND','SIMPLE','NAXIS','ROS','NAXIS1','NAXIS2','NAXIS3','BSCALE']:
        try:
            t.remove_column(col)
        except KeyError:
            pass
    csvwrite(t,os.path.dirname(path) + '/HeaderCatalog.csv')
    #t.write(os.path.dirname(path) + '/HeaderCatalog.csv', overwrite=True)
    try:
        file = open(os.path.dirname(path) + '/Info.log','w') 
        file.write('##########################################################################\n') 
        file.write('\nNumber of images : %i'%(len(t))) 
        file.write('\nGains: ' + '-'.join([gain for gain in np.unique(t['EMGAIN'])]))
        file.write('\nExposures: ' + '-'.join([exp for exp in np.unique(t['EXPTIME'])]))
        file.write('\nTemps: ' + '-'.join([te for te in np.unique(t['TEMP'])]))
        file.write('\nNumber of images per gain and exposure: %0.2f'%(len(t[(t['EMGAIN']==t['EMGAIN'][0]) & (t['EXPTIME']==t['EXPTIME'][0])])) )
        file.write('\n\n##########################################################################')      
        file.close() 
    except:
        pass
    return t


def CreateCatalogInfo(files, t1):
    """
    """
    from astropy.table import Table
    from astropy.io import fits
    from astropy.table import hstack
    files.sort()
    path = files[0]
#    path= '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/data/snape/180830/photon_counting/image000152.fits'
    #files = glob.glob(os.path.dirname(path) + '/*.fits')
    #files.sort()
    fields = ['Col2ColDiff', 'Line2lineDiff', 'OverscannRight', 'OverscannLeft',
              'TopImage', 'BottomImage', 'MeanADUValue', 'SaturatedPixels',
              'stdXY', 'stdY', 'stdX', 'MeanADUValueTR', 'MeanADUValueBR', 'MeanADUValueBL', 'MeanADUValueTL']
    #t = Table(names=fields,dtype=('S20,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    t = Table(names=fields,dtype=('float,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    for i, file in enumerate(files):
        t.add_row()
        print(i)
        data = fits.getdata(file)
        column = np.nanmean(data[2:1950,1100:2100], axis=1)
        line = np.nanmean(data[2:1950,1100:2100], axis=0)
        offset = 20
        OSR1 = [offset,-offset,offset,400]
        OSR2 = [offset,-offset,2200,2400]
        OSR = data[OSR2[0]:OSR2[1],OSR2[2]:OSR2[3]]
        OSL = data[OSR1[0]:OSR1[1],OSR1[2]:OSR1[3]]
        t[i]['Col2ColDiff'] =  np.nanmean(abs(line[1:] - line[:-1])) 
        t[i]['Line2lineDiff'] = np.nanmean(abs(column[1:] - column[:-1])) 
        t[i]['OverscannRight'] = np.nanmean(OSR)
        t[i]['OverscannLeft'] = np.nanmean(OSL)
        t[i]['TopImage'] = np.nanmean(column[:20])
        t[i]['BottomImage'] = np.nanmean(column[-20:])
        t[i]['MeanADUValue'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1950,1100:2100])
        t[i]['MeanADUValueTR'] =  np.nanmean((data - ComputeOSlevel1(data))[1000:1950,1600:2100])
        t[i]['MeanADUValueBR'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1000,1600:2100])
        t[i]['MeanADUValueBL'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1000,1100:1600])
        t[i]['MeanADUValueTL'] =  np.nanmean((data - ComputeOSlevel1(data))[1000:1950,1100:1600])
        t[i]['SaturatedPixels'] = 100*float(np.sum(data[2:1950,1100:2100]>60000)) / np.sum(data[2:1950,1100:2100]>0)
        t[i]['stdXY'] = np.nanstd(data[2:1950,1100:2100])
        t[i]['stdX'] = np.nanstd(data[1500,1100:2100])
        t[i]['stdY'] = np.nanstd(data[2:1950,1500])
    new_cat = hstack((t1,t))
    #t.remove_columns(['EXTEND','SIMPLE','NAXIS','COMMENT','NAXIS3','SHUTTER','VSS','BITPIX','BSCALE'])
    #new_cat.write(os.path.dirname(path) + '/HeaderCatalog_info.csv', overwrite=True)
    csvwrite(new_cat,os.path.dirname(path) + '/HeaderCatalog_info.csv')
    return new_cat    
        
def DS9PlotSpatial(xpapoint):
    """Plot the spatial dimension of the detector in order to detect a spectra
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 4:
        path = Charge_path(xpapoint, entry_point=4)
    else:
        path = [getfilename(d)]#[d.get("file")]
    field = sys.argv[3]
#    print('len(path)=',len(path))
    plot_flag = False
    if len(path) < 2:
        plot_flag = True
    for filename in path:
        print(filename)
        result = PlotSpatial1(filename, field=field, plot_flag=plot_flag)#2133+offset,-offset])
#    if len(path) < 2:
#        d.set("lock frame physical")
#        d.set('frame new')
#        d.set('tile yes')
#        d.set('file ' + name)  
    return result


def createFieldMask(image, x, y, xlen, ylen):#vpi
    """With mask centered on 206nm
    """
    mask = image.copy()
    mask[:,:] = 1
    for xi, yi in zip(x,y):
        print(xi, yi)
        mask[int(xi) - xlen : int(xi) + xlen, :int(yi) - ylen] = 0
        mask[int(xi) - xlen : int(xi) + xlen,int(yi) + ylen:] = 0
#        print(mask[int(xi) - xlen : int(xi) + xlen, int(yi) - ylen : int(yi) + ylen])
    image[mask==0] = np.nan
    print(float(np.nansum(image)) / len(image.flatten()))
    return image

def PlotSpatial(filename, field, save=True, plot_flag=True, DS9backUp = DS9_BackUp_path):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
#    from .mapping import Mapping
    from scipy.optimize import curve_fit
    from astropy.convolution import convolve
    #print('Entry = ', field)
    offset = 5
    fitsfile = fits.open(filename)
    image = fitsfile[0].data   
    #image = np.convolve(image, np.ones((41,3))/(41*3), mode='same')#[offset:-offset]
    image = convolve(image, np.ones((offset,41))/(41*offset))
    x, y, redshift, slit, w = returnXY(field,keyword=None)  #vpiki
#    try:
#        Mapping_dir = resource_filename('DS9FireBall', 'Mappings')
#    except:
#        Mapping_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mappings')
#    if ('f1' in field) or ('119' in field):
#        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F1_p2.pkl')
#    if ('f2' in field) or ('161' in field):
#        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F2_p2.pkl')
#    if ('f3' in field) or ('121' in field):
#        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F3_p2.pkl')
#    if ('f4' in field) or ('159' in field):
#        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-180612-F4_p2.pkl')

    #new_im = image[0:1870,1070:2100]
    #if plot_flag:

#    image = np.convolve(np.nanmean(image, np.ones(3)/3, mode='same')[offset:-offset]
    #mapping = Mapping(filename=mappingfile)
    #y,x = mapping.map(0.20619, x, y, inverse=False)
    new_im1 = createFieldMask(image, x, y, xlen=12, ylen=400)
    new_im = new_im1[0:1870,1070:2100]#[offset:-offset]
    plt.imshow(new_im)
    #new_im = np.convolve(np.nanmean(new_im,axis=1), np.ones(3)/3, mode='same')[offset:-offset]
    plt.show()
    new_im = np.nanmean(new_im,axis=1)[offset:-offset]
    n=2*10
    x1 = 360
    xfit1 = np.arange(x1,len(new_im))
    xfit2 = np.arange(x1)
    try:
        fit1 = np.polyfit(xfit1, new_im[x1:], n)
        fit2 = np.polyfit(xfit2, new_im[:x1], n)
    except ValueError as e:
        print ('Error with the fit, Nan value: ',e)
        p1 = lambda x: np.nanmean(new_im) + x * 0
        p2 = p1
    else:
        p1 = np.poly1d(fit1)
        p2 = np.poly1d(fit2)    
    f, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(11,7))
    ax1.plot(new_im, label='Detector columns')
    ax1.plot(xfit1, p1(xfit1), label='Background fit', c='orange')
    ax1.plot(xfit2, p2(xfit2), c='orange')
    ax1.set_ylim((-50,100))
    ax2.set_xlabel('Column index')
    ax1.set_ylabel('Average line Value [ADU] - K3 convolved')
    yy = [new_im[int(xi)] for xi in x]
    #plt.plot(x, yy,'P',label='Slits position')
    x += offset
    ax1.hlines(yy, x - 10, x + 10,label='Slits position', colors='black')
    for i, sliti in enumerate(slit):
        if i% 4 == 1:
            n = - 40
        elif i% 4 == 2:
            n = + 20
        elif i% 4 == 3:
            n =  -20
        elif i% 4 == 0:
            n = 40
        ax1.text(x[i], yy[i] + 1.3 * n,str(sliti),bbox=dict(facecolor='red', alpha=0.3),fontsize=8)
        ax1.vlines(x[i], yy[i] + 1.3 * n, yy[i], linestyles='dotted', colors='red')

    #plt.ylim(0.9*np.nanmin(new_im) -  1.3 * n, 1.1*np.nanmax(new_im)  +  1.3 * n)
    stdd1 = np.nanstd(new_im[x1:]-p1(xfit1))
    mean1 = np.nanmean(new_im[x1:]-p1(xfit1))
    #print('xfit1 = ',xfit1)
    #print('xfit2 = ',xfit2)
    #n=100
 #   argmaxx = int(np.nanmax(new_im[x1+n:-n]-p1(xfit1[n:-n])))
    #plt.plot(new_im[x1+n:-n]-p1(xfit1[n:-n]));plt.show()
    argmaxx = np.argmax(new_im[x1:]-p1(xfit1))
    print('argmax = ', argmaxx)
    popt, pcov = curve_fit(gaussian, xfit1, new_im[x1:]-p1(xfit1), p0=(np.nanmax(new_im[x1:]-p1(xfit1)),xfit1[argmaxx],5))
    print('x0 = ', xfit1[argmaxx])
    print('popt = ', popt)

    
    for k in [3,5,7,10]:
        ax2.plot(np.hstack((xfit1,xfit2)), np.ones(len(np.hstack((xfit1,xfit2)))) * (mean1 + k * stdd1), color='grey', linewidth=0.8, linestyle='dotted')
        ax2.text(500, mean1 + k * stdd1, '%i sigma'%(k), fontsize=9)
    ax2.plot(xfit1, new_im[x1:]-p1(xfit1), color='orange')
    ax2.plot(xfit2, new_im[:x1]-p2(xfit2), color='orange', label='Background substracted')
    ax2.text(xfit1[argmaxx], 30,"Amp = %0.1fADU\nFWHM = %0.2f''"%(popt[0],1.1*2.35*popt[2]),bbox=dict(facecolor='red', alpha=0.3),fontsize=8)
    ax2.plot(xfit1, gaussian(xfit1, *popt),'--', color='grey')
    ax2.plot(xfit2, gaussian(xfit2, *popt),'--', color='grey', label='Gaussian fit')
    ax2.text(1100, 20, 'Std(signal) = %0.1f'%(stdd1))
    ax1.legend()
    ax2.legend()
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
    ax1.set_title(os.path.basename(filename))
    f.tight_layout()
    #ax2.step(xfit1, new_im[x1:]-p1(xfit1), color='orange')
    #ax2.errorbar(xfit1, new_im[x1:]-p1(xfit1), yerr=stdd[x1:], color='orange')
    #plt.show()
    if save:#vpicouet
        if not os.path.exists(os.path.dirname(filename) +'/detection'):
            os.makedirs(os.path.dirname(filename) +'/detection')
        try:
            f.savefig(os.path.dirname(filename) +'/detection/'+ os.path.basename(filename).replace('.fits', '.detect.png'), dpi = 300, bbox_inches = 'tight')
        except ValueError:
            pass
    if plot_flag:
        #f.canvas.manager.window.raise_()
        plt.show()
        #f.canvas.manager.window.activateWindow()
    csvwrite(np.vstack((np.arange(len(new_im)),new_im)).T, DS9backUp + 'CSVs/%s_SpatialPlot.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return x, y, redshift, slit, w



def PlotSpatial1(filename, field, save=True, plot_flag=True, DS9backUp = DS9_BackUp_path):
    """
    Without mask centered on 206nm
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.optimize import curve_fit
    #print('Entry = ', field)
    fitsfile = fits.open(filename)#d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    x, y, redshift, slit, w = returnXY(field,keyword=None)  
    new_im = image[0:1870,1070:2100]
    #if plot_flag:
    offset = 5
    #stdd = np.nanstd(new_im,axis=1)[offset:-offset]
    new_im = np.convolve(np.nanmean(new_im,axis=1), np.ones(3)/3, mode='same')[offset:-offset]
    #new_im = np.nanmean(new_im,axis=1)[5:-5]
    n=2*10
    x1 = 360
    xfit1 = np.arange(x1,len(new_im))
    xfit2 = np.arange(x1)
    try:
        fit1 = np.polyfit(xfit1, new_im[x1:], n)
        fit2 = np.polyfit(xfit2, new_im[:x1], n)
    except ValueError as e:
        print ('Error with the fit, Nan value: ',e)
        p1 = lambda x: np.nanmean(new_im) + x * 0
        p2 = p1
    else:
        p1 = np.poly1d(fit1)
        p2 = np.poly1d(fit2)    
    f, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(11,7))
    ax1.plot(new_im, label='Detector columns')
    ax1.plot(xfit1, p1(xfit1), label='Background fit', c='orange')
    #ax1.set_ylim((-50,100))
    ax1.plot(xfit2, p2(xfit2), c='orange')
    ax2.set_xlabel('Column index')
    ax1.set_ylabel('Average line Value [ADU] - K3 convolved')
    yy = [new_im[int(xi)] for xi in x]
    #plt.plot(x, yy,'P',label='Slits position')
    x += offset
    ax1.hlines(yy, x - 10, x + 10,label='Slits position', colors='black')
    for i, sliti in enumerate(slit):
        if i% 4 == 1:
            n = - 40
        elif i% 4 == 2:
            n = + 20
        elif i% 4 == 3:
            n =  -20
        elif i% 4 == 0:
            n = 40
        ax1.text(x[i], yy[i] + 1.3 * n,str(sliti),bbox=dict(facecolor='red', alpha=0.3),fontsize=8)
        ax1.vlines(x[i], yy[i] + 1.3 * n, yy[i], linestyles='dotted', colors='red')

    #plt.ylim(0.9*np.nanmin(new_im) -  1.3 * n, 1.1*np.nanmax(new_im)  +  1.3 * n)
    stdd1 = np.nanstd(new_im[x1:]-p1(xfit1))
    mean1 = np.nanmean(new_im[x1:]-p1(xfit1))
    #print('xfit1 = ',xfit1)
    #print('xfit2 = ',xfit2)
    #n=100
 #   argmaxx = int(np.nanmax(new_im[x1+n:-n]-p1(xfit1[n:-n])))
    #plt.plot(new_im[x1+n:-n]-p1(xfit1[n:-n]));plt.show()
    argmaxx = np.argmax(new_im[x1:]-p1(xfit1))
    print('argmax = ', argmaxx)
    popt, pcov = curve_fit(gaussian, xfit1, new_im[x1:]-p1(xfit1), p0=(np.nanmax(new_im[x1:]-p1(xfit1)),xfit1[argmaxx],5))
    print('x0 = ', xfit1[argmaxx])
    print('popt = ', popt)

    
    for k in [3,5,7,10]:
        ax2.plot(np.hstack((xfit1,xfit2)), np.ones(len(np.hstack((xfit1,xfit2)))) * (mean1 + k * stdd1), color='grey', linewidth=0.8, linestyle='dotted')
        ax2.text(500, mean1 + k * stdd1, '%i sigma'%(k), fontsize=9)
    ax2.plot(xfit1, new_im[x1:]-p1(xfit1), color='orange')
    ax2.plot(xfit2, new_im[:x1]-p2(xfit2), color='orange', label='Background substracted')
    ax2.text(xfit1[argmaxx], 30,"Amp = %0.1fADU\nFWHM = %0.2f''"%(popt[0],1.1*2.35*popt[2]),bbox=dict(facecolor='red', alpha=0.3),fontsize=8)
    ax2.plot(xfit1, gaussian(xfit1, *popt),'--', color='grey')
    ax2.plot(xfit2, gaussian(xfit2, *popt),'--', color='grey', label='Gaussian fit')
    ax2.text(1100, 20, 'Std(signal) = %0.1f'%(stdd1))
    ax1.legend()
    ax2.legend()
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
    ax1.set_title(os.path.basename(filename))
    f.tight_layout()
    #ax2.step(xfit1, new_im[x1:]-p1(xfit1), color='orange')
    #ax2.errorbar(xfit1, new_im[x1:]-p1(xfit1), yerr=stdd[x1:], color='orange')
    #plt.show()
    if save:#vpicouet
        if not os.path.exists(os.path.dirname(filename) +'/detection'):
            os.makedirs(os.path.dirname(filename) +'/detection')
        try:
            f.savefig(os.path.dirname(filename) +'/detection/'+ os.path.basename(filename).replace('.fits', '.detect.png'), dpi = 300, bbox_inches = 'tight')
        except ValueError:
            pass
    if plot_flag:
        #f.canvas.manager.window.raise_()
        plt.show()
        #f.canvas.manager.window.activateWindow()
    csvwrite(np.vstack((np.arange(len(new_im)),new_im)).T, DS9backUp + 'CSVs/%s_SpatialPlot.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return x, y, redshift, slit, w



def DS9OverscanCorrection(xpapoint):
    """Overscan correction line by line
    """
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]
    for filename in path:
        print(filename)
        offset = 20
        result, name = ApplyOverscanCorrection(filename, stddev=3, 
                                               OSR1=[offset,-offset,offset,400],#1053-offset], 
                                               OSR2=[offset,-offset,2200,2400])#2133+offset,-offset])
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return result


def ApplyOverscanCorrection(path, stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20], save=True):
    """Apply overscan correction line by line using overscan region
    """
    from astropy.io import fits
    fitsimage = fits.open(path)[0]
    image = fitsimage.data.astype(float).copy()
    OScorrection = ComputeOSlevel1(image, OSR1=OSR1, OSR2=OSR2)
    fitsimage.data = image - OScorrection
    name = path[:-5] + '_OScorr.fits'
    if save:
        fitswrite(fitsimage, name)
    return fitsimage, name

def ComputeOSlevel(image, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20]):
    """Apply average overscan correction  (not line by line)
    """
    OSregion = np.hstack((image[OSR1[0]:OSR1[1],OSR1[2]:OSR1[3]],image[OSR2[0]:OSR2[1],OSR2[2]:OSR2[3]]))
    OScorrection = np.nanmedian(OSregion,axis=1)#reject_outliers(OSregion, stddev=3))
    #OScorrection = OScorrection[..., np.newaxis]*np.ones((image.shape))
    return OScorrection

def ComputeOSlevel1(image, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20]):
    """Apply overscan correction line by line using overscan region
    """
    OSregion = np.hstack((image[:,OSR1[2]:OSR1[3]],image[:,OSR2[2]:OSR2[3]]))
    OScorrection = np.nanmedian(OSregion,axis=1)#reject_outliers(OSregion, stddev=3))
    OScorrection = OScorrection[..., np.newaxis]*np.ones((image.shape))
    #print(OScorrection)
    return OScorrection



def DS9replaceNaNs(xpapoint):
    """Replace the pixels in the selected regions in DS9 by NaN values
    """
   # from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#d.get("file")
    regions = getregion(d, all=True)
    fitsimage = d.get_fits()[0]#fits.open(filename)[0]
    image = fitsimage.data.astype(float).copy()
    print(regions)
    try:
        xc, yc, h, w = int(regions.xc), int(regions.yc), int(regions.h), int(regions.w)
        print('Only one region found...')        
        print('W = ', w)
        print('H = ', h)
        Xinf = int(np.floor(yc - h/2 -1))
        Xsup = int(np.ceil(yc + h/2 -1))
        Yinf = int(np.floor(xc - w/2 -1))
        Ysup = int(np.ceil(xc + w/2 -1))
        image[Xinf:Xsup+1,Yinf:Ysup+2] = np.nan
    except AttributeError:
        print('Several regions found...')
        for region in regions:
            xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
            print('W = ', w)
            print('H = ', h)
            Xinf = int(np.floor(yc - h/2 -1))
            Xsup = int(np.ceil(yc + h/2 -1))
            Yinf = int(np.floor(xc - w/2 -1))
            Ysup = int(np.ceil(xc + w/2 -1))
            image[Xinf:Xsup+1,Yinf:Ysup+2] = np.nan
    fitsimage.data = image
    #fitsimage.writeto(filename,overwrite=True)
    filename = fitswrite(fitsimage,filename)
    #d.set('frame new')
    #d.set('tile yes')
    d.set('file '+ filename)
    d.set('pan to %0.3f %0.3f physical' % (xc,yc))
    #d.set('lock yes')
    #d.set("lock frame physical")
    return



def InterpolateNaNs(path, stddev=1):
    """Return int16 fits image with NaNs interpolated
    """
    from astropy.convolution import interpolate_replace_nans
    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel          
    print(path)
    image = fits.open(path)[0]
    result = image.data
    stddev = stddev
    while (~np.isfinite(result).all()):
        print(len(np.where(~np.isfinite(result))[0]), 'NaN values found!')
        print('Infinite values in the image, inteprolating NaNs with 2D Gaussian kernel of standard deviation = ', stddev)
        kernel = Gaussian2DKernel(stddev=stddev)
        result = interpolate_replace_nans(result, kernel)
        stddev += 1

    image.data = result.astype('uint16')
    name = path[:-5] + '_NaNsFree.fits'
    fitswrite(image, name)
    return result, name

def DS9InterpolateNaNs(xpapoint):
    """Replace NaNs value of the image in DS9 (or several if specified) by values interpolated around NaNs value
    """
#    d = DS9(xpapoint)
#    filename = d.get("file")
#    try:
#        n1, n2 = sys.argv[3].split('-')#'f3 names'#sys.argv[3]
#        n1, n2 = int(n1), int(n2)
#        print('n1, n2 = ', n1, n2)
#    except (ValueError, IndexError) as e:
#        print('No value given, taking only DS9 image:', e)
#        n1, n2 = None, None
#    else:
#        numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
#        print('Numbers used: {}'.format(numbers))               
#        path = []
#        print('Specified numbers are integers, opening corresponding files ...')
#        names = os.path.basename(filename).split('.')
#        for number in numbers:
#            if len(names)==2:
#                path.append(os.path.dirname(filename) + '/image%06d.' % (int(number)) + names[-1])
#            if len(names)==3:
#                path.append(os.path.dirname(filename) + '/image%06d.' % (int(number)) + names[-2] + '.' + names[-1])            
#
#    if n1 is None:
#        path = [filename]
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]
        
    for filename in path:
        if os.path.isfile(filename) is False:
            filename = filename[:-5] + '.CRv.fits'#os.path.join(os.path.dirname(filename),os.path.basename(fielname))
        print(filename)
        result, name = InterpolateNaNs(filename, stddev=2)
        if len(path)<2:
            d.set("lock frame physical")
            d.set('frame new')
            d.set('tile yes')
            d.set('file ' + name)  
            #picouet
    return


def AnalyzeBackground(filename):
    """Analyze background in 2 specfic areas of the images for FB2 because
    of caracteristics of the flight and save it in the header
    """
    from astropy.io import fits
    from scipy import stats
    fitsfile = fits.open(filename)[0]
    LowPartimage = fitsfile.data[16:330,1100:2100]
    HighPart = fitsfile.data[500:1950,1100:2100]
    print ('Image Low part: {}'.format(filename))
    print ('Mean : {}'.format(np.nanmean(LowPartimage)))
    print ('Standard deviation : {}'.format(np.nanstd(LowPartimage)))
    #print ('Skewness: {}'.format(float(stats.skew(LowPartimage,axis=None, nan_policy='omit').data)))    
   
    print ('Image Hight part: {}'.format(filename))
    print ('Mean : {}'.format(np.nanmean(HighPart)))
    print ('Standard deviation : {}'.format(np.nanstd(HighPart)))
    #print ('Skewness: {}'.format(float(stats.skew(HighPart,axis=None, nan_policy='omit').data)))   
    
    fitsfile.header['MEANHIGH'] = np.nanmean(HighPart)
    fitsfile.header['STDHIGH'] = np.nanstd(HighPart)
    try:    
        fitsfile.header['SKEWHIGH'] = float(stats.skew(HighPart,axis=None, nan_policy='omit').data)  
    except AttributeError:
        fitsfile.header['SKEWHIGH'] = float(stats.skew(HighPart,axis=None, nan_policy='omit'))  

    fitsfile.header['MEANLOW'] = np.nanmean(LowPartimage)
    fitsfile.header['STDLOW'] = np.nanstd(LowPartimage)
    try:
        fitsfile.header['SKEWLOW'] = float(stats.skew(LowPartimage,axis=None, nan_policy='omit').data)  
    except AttributeError:
        fitsfile.header['SKEWLOW'] = float(stats.skew(LowPartimage,axis=None, nan_policy='omit'))
    fitsfile.header.comments['SKEWHIGH'] = 'Skewness of a data set on [500:1950,1100:2100]'
    fitsfile.header.comments['STDHIGH'] = 'Mean ADU values of a data set on [500:1950,1100:2100]'
    fitsfile.header.comments['MEANHIGH'] = 'Std ADU avlues of a data set on [500:1950,1100:2100]'

    fitsfile.header.comments['SKEWLOW'] = 'Skewness of a data set on [16:330,1100:2100]'
    fitsfile.header.comments['STDLOW'] = 'Mean ADU values of a data set on [16:330,1100:2100]'
    fitsfile.header.comments['MEANLOW'] = 'Std ADU avlues of a data set on [16:330,1100:2100]'
#    try:
#        fitsfile.header.comments['N_CR'] = 'Number of detected cosmic rays'
#        fitsfile.header.comments['MASK'] = 'Number of pixels deleted due to cosmics'
#    except KeyError:
#        fitsfile.header['N_CR'] = 0
#        fitsfile.header['MASK'] = 0
#        fitsfile.header.comments['N_CR'] = 'Number of detected cosmic rays'
#        fitsfile.header.comments['MASK'] = 'Number of pixels deleted due to cosmics'
    fitsfile.writeto(filename, overwrite=True)
    return
 
#for filename in glob.glob('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux'+'/image*.CRv.fits'):
#    AnalyzeBackground(filename)
 
ScrollableWindow = None
def Choose_backend(function):
    """
    """
    global ScrollableWindow 
    if 'spectra' in function:  
        if check_if_module_exists('PyQt5'):
            from PyQt5 import QtWidgets
            #import matplotlib.pyplot as plt;from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas;from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar    
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar    
            
            class ScrollableWindow(QtWidgets.QMainWindow):
                """
                """
                import matplotlib.pyplot as plt    
                #matplotlib.use('Qt5Agg')# Make sure that we are using QT5
                def __init__(self, fig):
                    self.qapp = QtWidgets.QApplication([])
                    QtWidgets.QMainWindow.__init__(self)
                    self.widget = QtWidgets.QWidget()
                    self.setCentralWidget(self.widget)
                    self.widget.setLayout(QtWidgets.QVBoxLayout())
                    self.widget.layout().setContentsMargins(0,0,0,0)
                    self.widget.layout().setSpacing(0)    
                    self.fig = fig
                    self.canvas = FigureCanvas(self.fig)
                    self.canvas.draw()
                    self.scroll = QtWidgets.QScrollArea(self.widget)
                    self.scroll.setWidget(self.canvas)
                    self.nav = NavigationToolbar(self.canvas, self.widget)
                    self.widget.layout().addWidget(self.nav)
                    self.widget.layout().addWidget(self.scroll)
                    self.resize(2000, 1000)
                    self.show()
                    self.qapp.exec_()#exit(self.qapp.exec_()) 
                    return


def DS9LoadCSV(xpapoint):
    from astropy.table import Table
    from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
#    if len(sys.argv) > 3:
#        path = Charge_path(xpapoint)
#    else:
#        path = [d.get("file")]
    sources = Table.read(filename[:-5] + '.csv')
    create_DS9regions2(sources['xcentroid'],sources['ycentroid'], radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers')
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers.reg'))                                        
    return


def DS9ExtractSources(xpapoint):
    """Extract sources for DS9 image and create catalog
    """
    from .focustest import create_DS9regions2
    #x, y = x+93, y-93
    d = DS9(xpapoint)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]
        
    for filename in path:
        print(filename)
        sources = ExtractSources(filename, fwhm=5, threshold=5, theta=0, ratio=1, n=2, quick=True)
        if len(path)<2:
            create_DS9regions2(sources['xcentroid'],sources['ycentroid'], radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers')
            d.set('region delete all')
            d.set('region {}'.format('/tmp/centers.reg')) 
        csvwrite(sources, filename[:-5] + '.csv')
#        try:                                          
#            sources.write(filename[:-5] + '.csv')
#        except UnicodeDecodeError:
#            reload(sys); sys.setdefaultencoding('utf8')
#            sources.write(filename[:-5] + '.csv')            
    return



def ExtractSources(filename, fwhm=5, threshold=8, theta=0, ratio=1, n=2, quick=True):
    """Extract sources for DS9 image and create catalog
    """
    from astropy.io import fits
    from scipy import ndimage
    from astropy.table import Table
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    from .focustest import delete_doublons
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data
    data2 = ndimage.grey_dilation(ndimage.grey_erosion(data, size=(n,n)), size=(n,n))       
    mean, median, std = sigma_clipped_stats(data2, sigma=3.0, iters=5)    
    if quick:    
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std,ratio = ratio, theta = theta)         
        sources0 = daofind(data2 - median) 
    if not quick:
        daofind = DAOStarFinder(fwhm=fwhm[0], threshold=threshold[0]*std,ratio = ratio, theta = theta)        
        sources0 = daofind(data2 - median)
        for i in fwhm: 
            for j in threshold: 
                daofind = DAOStarFinder(fwhm=i, threshold=j*std,ratio =ratio, theta = theta)         
            
                sources1 = daofind(data2 - median) 
                print('fwhm = {}, T = {}, len = {}'.format(i,j,len(sources1)))
                try :
                    sources0 = Table(np.hstack((sources0,sources1)))
                except TypeError:
                    print('catalog empty')
                    if len(sources0)==0:
                        sources0 = sources1
    sources = delete_doublons(sources0, dist = 30)
    return sources


#class Fits(object):
#    """
#    """
#    def __init__(self, ds9):
#        """
#        """
#        from astropy.io import fits
#
#        
#        filename = ds9.get('file')
#        if len(filename) == 0:
#            self.filename = ''
#            self.header = None
#        else:
#            if '[IMAGE]' in os.path.basename(filename):
#                filename = filename.split('[IMAGE]')[0]
#            
#            if filename[0] == '.':
#                new_filename = backup_path + filename[1:]
#            
#            else:
#                new_filename = filename
#            
#            self.data = ds9.get_arr2np()
#            self.filename = new_filename
#            self.header = fits.open(self.filename)[0].header


def getfilename(ds9):
    """
    """
    backup_path = os.environ['HOME'] + '/DS9BackUp'
    if not os.path.exists(os.path.dirname(backup_path)):
        os.makedirs(os.path.dirname(backup_path))
    filename = ds9.get('file')
    
    if '[IMAGE]' in os.path.basename(filename):
        filename = filename.split('[IMAGE]')[0]
    if len(filename)==0:
        new_filename = filename
    elif filename[0] == '.':
        new_filename = backup_path + filename[1:]
    else:
        new_filename = filename
    print(new_filename)
    return new_filename    

def HistogramSums(xpapoint, DS9backUp = DS9_BackUp_path):
    """
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt   
    from astropy.table import Table
    d = DS9(xpapoint)
    #filename = d.get('file')
    filename = getfilename(d)
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]    

    try:
        region = getregion(d)
    except ValueError:
        image_area = [0,2069,1172,2145]
        Yinf, Ysup,Xinf, Xsup = image_area
    else:
        Yinf, Ysup, Xinf, Xsup = Lims_from_region(region)#[131,1973,2212,2562]
        #image_area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)

    #filename='/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CRfree_and_NaNsFree/image000101.CRv_NaNsFree.fits'
    #filename='/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CosmicRaysfree/image000341.CRv.fits'
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data[Xinf: Xsup, Yinf: Ysup]
    #value, bins = np.histogram(data,bins=500,range=(data.min(),np.percentile(data,99.999)))
    value, bins = np.histogram(data,bins=200,range=(np.nanmin(data),np.nanmax(data)))#,np.percentile(data,99.99)))
    bins_c = (bins[1:] + bins[:-1])/2

    for filename in path[1:]: 
        print(filename)
        fitsfile = fits.open(filename)[0]
        data = fitsfile.data[Xinf: Xsup, Yinf: Ysup]
        value1, bins = np.histogram(data,bins=bins, range=(np.nanmin(bins), np.nanmax(bins)))
        value += value1
    plt.figure(figsize=(10,6))
    #plt.step(bins_c,np.log10(value))
    plt.semilogy(bins_c,value)
    plt.grid(True, which="both")
    plt.xlabel('ADU value')
    plt.title("Histogram's sum")
    plt.ylabel('Log(#)')
    plt.grid()
    plt.show()
    csvwrite(Table(np.vstack((bins_c,value)).T), DS9backUp + 'CSVs/%s_HistogramSum.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")))#piki
    return


def DS9BackgroundEstimationPhot(xpapoint, n=2, DS9backUp = DS9_BackUp_path):
    """
    """
    from astropy.io import fits
    from astropy.stats import SigmaClip
    #from scipy import ndimage
#    import matplotlib.pyplot as plt   
    from photutils import Background2D, MedianBackground
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    
    if len(sys.argv) > 3:
        path = Charge_path(xpapoint)
    else:
        path = [getfilename(d)]    

    #filename='/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CRfree_and_NaNsFree/image000101.CRv_NaNsFree.fits'
    #filename='/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CosmicRaysfree/image000341.CRv.fits'
    for filename in path: 
        print(filename)
        fitsfile = fits.open(filename)[0]
        data = fitsfile.data#[400:1700,1200:2000]
        #data2 = data#ndimage.grey_dilation(ndimage.grey_erosion(data, size=(n,n)), size=(n,n))   
        masks = generatemask(data)
        for i, mask in enumerate(masks):
            sigma_clip = SigmaClip(sigma=3.)
            bkg_estimator = MedianBackground()
            bkg = Background2D(mask, (50, 50), filter_size=(3, 3),
                               sigma_clip=sigma_clip, bkg_estimator=bkg_estimator,exclude_percentile=20)
            print('Mask %i, median = %0.2f'%(i,bkg.background_median))  
            print('Mask %i, rms = %0.2f'%(i,bkg.background_rms_median))  
#            fitsfile.data -= bkg.background.astype('uint16')
            #bkg.background[mask==np.nan]=0
            bkg.background[~np.isfinite(mask)]=0#; imshow(data)
            fitsfile.data = fitsfile.data - bkg.background#.astype('uint16')
#            plt.figure()
#            plt.imshow(bkg.background)
#            plt.show()

        diff = np.nanmean(fitsfile.data[np.isfinite(masks[1])]) - np.nanmean(fitsfile.data[np.isfinite(masks[0])])
        fitsfile.data[np.isfinite(masks[1])] -= diff
        if np.isfinite(data).all():
            fitsfile.data = fitsfile.data.astype('uint16')
            
        name = filename[:-5] + 'bkgd_photutils_substracted.fits'
        #fitsfile.writeto(name, overwrite=True)
        fitswrite(fitsfile, name)
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt    
    fig = plt.figure(figsize=(12,5))
    plt.subplot(131)
    plt.title('Data')
    plt.imshow(data[0:-1,1053:2133], vmin=np.percentile(data[0:-1,1053:2133],50), vmax=np.percentile(data[0:-1,1053:2133],99));plt.colorbar()
    plt.subplot(132)
    im = plt.imshow(data[0:-1,1053:2133] - fitsfile.data[0:-1,1053:2133], vmin=np.percentile(data[0:-1,1053:2133] - fitsfile.data[0:-1,1053:2133],50), vmax=np.percentile(data[0:-1,1053:2133] - fitsfile.data[0:-1,1053:2133],99));plt.colorbar()
    plt.title('Extracted background')
    plt.subplot(133)
    im = plt.imshow(fitsfile.data[0:-1,1053:2133], vmin=np.percentile(fitsfile.data[0:-1,1053:2133],50), vmax=np.percentile(fitsfile.data[0:-1,1053:2133],99))#;plt.colorbar()
    fig.colorbar(im)
    plt.title('Data - Background')
    #plt.savefig(filename[:-5] + '.png')
    plt.savefig(DS9backUp + 'Plots/%s_BackgroundSubtraction%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")))    #plt.subplot_tool()
    #plt.show()
#    plt.imshow(data, origin='lower', cmap='Greys_r', norm=norm)
#    bkg.plot_meshes(outlines=True, color='#1f77b4')
    if len(path)<2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return bkg

def generatemask(data):
    """
    """
    if data.shape == (2069, 3216):
        from PIL import Image, ImageDraw
        img1 = Image.new('L', data.shape[::-1], 0)#, Image.new('L2', data.shape, 0)
        img2 = Image.new('L', data.shape[::-1], 0)#, Image.new('L2', data.shape, 0)
        polygon1 = [1907,366,   2122,389,   2122,0,    1065,0,      1065,352]
        polygon2 = [1907,366,   2122,389,   2122,2000, 1065,2000,   1065,352]
        ImageDraw.Draw(img1).polygon(polygon1, outline=1, fill=1)
        ImageDraw.Draw(img2).polygon(polygon2, outline=1, fill=1)
        m1 = np.array(img1)
        m2 = np.array(img2)
        mask1, mask2 = data.copy().astype(float), data.copy().astype(float)
        mask1[m1==0] = np.nan
        mask2[m2==0] = np.nan
        return mask1, mask2
    else:
        print('Data shape not understood, this function works only for FB detector images')
        return data


def generatemask1(data):
    """
    """
    if data.shape == (2069, 3216):
        general_mask = data.copy()
        general_mask[2000:, :] = np.nan
        general_mask[:,:1065] = np.nan
        general_mask[:,2122:] = np.nan
        mask1, mask2 = general_mask.copy(), general_mask.copy()
        mask1[:362,:] = np.nan
        mask2[362:,:] = np.nan
        return mask1, mask2
    else:
        print('Data shape not understood, this function works only for FB detector images')
        return data

def GroupingAlgorithm(xpapoint, n=2, fwhm=5):
    """
    """
    from astropy.io import fits
    #from astropy.stats import gaussian_sigma_to_fwhm
    #from photutils.psf.groupstars import DAOGroup
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data
    #from scipy import ndimage
#    data2 = ndimage.grey_dilation(ndimage.grey_erosion(data, size=(n,n)), size=(n,n)) 
#    daogroup = DAOGroup(crit_separation=2.5*fwhm)
#    fwhm = sigma_psf * gaussian_sigma_to_fwhm
    return data

def AperturePhotometry(xpapoint):
    """
    """
    from astropy.table import Table
    from .focustest import create_DS9regions2
    #from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from photutils import aperture_photometry
    from photutils import CircularAperture, CircularAnnulus
    from astropy.table import hstack
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    fitsfile = d.get_fits()[0]#fits.open(filename)[0]
    data = fitsfile.data
    try:
        source_catalog_name = sys.argv[3]
    except IndexError:
        source_catalog_name = filename[:-5] + '.csv'
    source_catalog = Table.read(source_catalog_name)
    try:        
        positions = [(x,y) for x,y in source_catalog['xcentroid', 'ycentroid']]
    except ValueError:
        positions = [(x,y) for x,y in source_catalog['xcenter', 'ycenter']]
    apertures = CircularAperture(positions, r=5)
    annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)
    annulus_masks = annulus_apertures.to_mask(method='center')
    
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    phot = aperture_photometry(data, apertures, error = 0.1 * data)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * apertures.area()
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    for col in phot.colnames:
        phot[col].info.format = '%.8g'  # for consistent table output
    print(phot) 
    phot = Table(phot)
    create_DS9regions2(phot['xcenter'],phot['ycenter'], radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers1')
    create_DS9regions2(phot['xcenter'],phot['ycenter'], radius=15, form = 'circle',save=True,color = 'red', savename='/tmp/centers2')
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers1.reg'))                                        
    d.set('region {}'.format('/tmp/centers2.reg'))
    new_cat =  hstack((source_catalog,phot))    
    new_cat.remove_columns(['xcentroid','ycentroid','id_2'])                              
    csvwrite(new_cat, filename[:-5] + '.csv')#new_cat.write(filename[:-5] + '.csv', overwrite=True)
    return phot

def BuildingEPSF(xpapoint):
    """
    """
    from astropy.table import Table
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.nddata import NDData
    from photutils.psf import extract_stars
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    try:
        source_catalog_name = sys.argv[3]
    except IndexError:
        source_catalog_name = filename[:-5] + '.csv'
    source_catalog = Table.read(source_catalog_name)
    stars_tbl = Table()
    try:        
        stars_tbl['x'],stars_tbl['y'] = source_catalog['xcentroid'], source_catalog['ycentroid']
    except ValueError:
        stars_tbl['x'],stars_tbl['y'] = source_catalog['xcenter'], source_catalog['ycenter']
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
    data -= median_val    
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=25)   
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                           squeeze=True)
    ax = ax.ravel()
    for i in range(nrows*ncols):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    from photutils import EPSFBuilder
    epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    norm = simple_norm(epsf.data, 'log', percent=99.)
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    plt.colorbar()
    return
   
def MorphologicalProperties(xpapoint):
    """
    """
    from astropy.table import Table
    #from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel
    from photutils import Background2D, MedianBackground
    from photutils import detect_sources
    from astropy.stats import gaussian_fwhm_to_sigma
    from photutils import deblend_sources
    import matplotlib.pyplot as plt
    from photutils import source_properties, EllipticalAperture
    from .focustest import create_DS9regions
    #from astropy.table import hstack

    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    fitsfile = d.get_fits()[0]#fits.open(filename)[0]
    data = fitsfile.data

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    threshold = bkg.background + (2. * bkg.background_rms)
    sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    npixels = 5
    segm = detect_sources(data, threshold, npixels=npixels,
                          filter_kernel=kernel)
    segm_deblend = deblend_sources(data, segm, npixels=npixels,
                                   filter_kernel=kernel, nlevels=32,
                                   contrast=0.001)
    cat = source_properties(data, segm_deblend)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['cxx'].info.format = '.2f'
    tbl['cxy'].info.format = '.2f'
    tbl['cyy'].info.format = '.2f'
    
    r = 3.    # approximate isophotal extent
    apertures = []
    for obj in cat:
        position = (obj.xcentroid.value, obj.ycentroid.value)
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        theta = obj.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
    tbl = Table(tbl)
    print(tbl)
    ly, lx = data.shape
    fig, ax1 = plt.subplots(1, 1, figsize=(7*float(lx)/ly-1, 7))
    ax1.imshow(segm_deblend, origin='lower',
               cmap=segm_deblend.cmap(random_state=12345))
    ax1.set_title('Segmentation Image')
    for aperture in apertures:
        aperture.plot(color='white', lw=0.7, ax=ax1)
    plt.show()
    create_DS9regions([list(tbl['xcentroid'])],[list(tbl['ycentroid'])], more=[3*tbl['semimajor_axis_sigma'],3*tbl['semiminor_axis_sigma'],180*tbl['orientation']/np.pi], form = ['ellipse']*len(tbl),save=True,color = ['red']*len(tbl), savename='/tmp/centers',ID=[list(tbl['id'])])
#    create_DS9regions([list(tbl['xcentroid'])],[list(tbl['ycentroid'])], form=['ellipse'],  more=[2.35*tbl['semimajor_axis_sigma'],2.35*tbl['semiminor_axis_sigma']], radius=10, save=True, savename='/tmp/cr', color = ['yellow'],ID=None)
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers.reg'))                                        
    tbl.write(filename[:-5] + '.csv', overwrite=True)
    return segm_deblend

def EllipticalIsophoteAnalysis(xpapoint):
    """
    """
    from photutils.isophote import EllipseGeometry
    import matplotlib.pyplot as plt
    from photutils.isophote import Ellipse
    #from astropy.io import fits
#    from photutils.datasets import make_noise_image
#    from astropy.modeling.models import Gaussian2D
#    g = Gaussian2D(100., 75, 75, 20, 12, theta=40.*np.pi/180.)
#    ny = nx = 150
#    y, x = np.mgrid[0:ny, 0:nx]
#    noise = make_noise_image((ny, nx), type='gaussian', mean=0.,
#                             stddev=2., random_state=12345)
#    data = g(x, y) + noise
    d = DS9(xpapoint)
    #filename = getfilename(d)#filename = d.get('file')

    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    data = d.get_fits()[0].data[area[0]:area[1],area[2]:area[3]]#fits.open(filename)[0].data[area[0]:area[1],area[2]:area[3]]#picouet  
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                           pa=20.*np.pi/180.)
#    plt.imshow(data);plt.show()
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()
    print(isolist.pa)    
    print(isolist.to_table())  
    plt.figure(figsize=(8, 6))
    #plt.subplots_adjust(hspace=0.35, wspace=0.35)
    plt.subplot(2, 2, 1)
    plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
                 fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('Ellipticity')
    plt.subplot(2, 2, 2)
    plt.errorbar(isolist.sma, isolist.pa/np.pi*180.,
                 yerr=isolist.pa_err/np.pi* 80., fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('PA (deg)')
    plt.subplot(2, 2, 3)
    plt.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('x0')
    plt.subplot(2, 2, 4)
    plt.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('y0')
    plt.tight_layout()
    plt.show()
    table = isolist.to_table()
    table.write('/tmp/isolist_fit.csv',overwrite=True)
    return isolist


def Centroiding(xpapoint):
    """
    """
    #from astropy.io import fits
    from photutils import centroid_com, centroid_1dg, centroid_2dg
    from .focustest import create_DS9regions

    d = DS9(xpapoint)
    #filename = getfilename(d)#filename = d.get('file')

    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    data = d.get_fits()[0][area[0]:area[1],area[2]:area[3]] #fits.open(filename)[0].data[area[0]:area[1],area[2]:area[3]]#picouet 
    x1, y1 = centroid_com(data)
    print((x1, y1))    
    x2, y2 = centroid_1dg(data)
    print((x2, y2))    
    x3, y3 = centroid_2dg(data)
    print((x3, y3))
    create_DS9regions([[x1+Xinf,x2+Xinf,x3+Xinf]], [[y1+Yinf,y2+Yinf,y3+Yinf]], radius=2, form = ['circle']*3,save=True, ID=[['COM:%0.2f - %0.2f'%(x1,y1),'2x1D Gaussian:%0.2f - %0.2f'%(x2,y2),'2D Gaussian:%0.2f - %0.2f'%(x3,y3)]], color = ['yellow', 'red', 'green'], savename='/tmp/centers1')
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers1.reg'))   
   
    
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots(1, 1)
#    ax.imshow(data, origin='lower', interpolation='nearest', cmap='viridis')
#    marker = '+'
#    ms, mew = 30, 2.
#    plt.plot(x1, y1, color='#1f77b4', marker=marker, ms=ms, mew=mew)
#    plt.plot(x2, y2, color='#17becf', marker=marker, ms=ms, mew=mew)
#    plt.plot(x3, y3, color='#d62728', marker=marker, ms=ms, mew=mew)   
#    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#    ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
#    ax2.imshow(data, interpolation='nearest', origin='lower',
#               cmap='viridis', vmin=190, vmax=220)
#    ax2.plot(x1, y1, color='#1f77b4', marker=marker, ms=ms, mew=mew)
#    ax2.plot(x2, y2, color='#17becf', marker=marker, ms=ms, mew=mew)
#    ax2.plot(x3, y3, color='#d62728', marker=marker, ms=ms, mew=mew)
#    ax2.set_xlim(13, 15)
#    ax2.set_ylim(16, 18)
#    mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='0.5')
#    ax2.axes.get_xaxis().set_visible(False)
#    ax2.axes.get_yaxis().set_visible(False)
#    ax.set_xlim(0, data.shape[1]-1)
#    ax.set_ylim(0, data.shape[0]-1)
#    plt.show()
    return


def FollowProbabilityLaw(xpapoint):
    d = DS9()
    lx, ly = 1000, 1000
    law = sys.argv[-1]    #law = 'standard_exponential'
    DictFunction = {'beta':[np.random.beta,1,1],'binomial':[np.random.binomial,10, 0.5],'geometric':[np.random.geometric, 0.5],'pareto':[np.random.pareto, 1],
                     'poisson':[np.random.poisson, 1],'power':[np.random.power, 1],'rand':[np.random.rand],'randint':[np.random.randint, lx],'standard_exponential':[np.random.standard_exponential],
                     'standard_gamma':[np.random.standard_gamma, 1],'standard_normal':[np.random.standard_normal],'standard_t':[np.random.standard_t, 1]}
    if law not in DictFunction.keys():
        print('Please choose a law in the following possibilities: \n','\n'.join(DictFunction.keys()))
    else:
        image = DictFunction[law][0](*DictFunction[law][1:],size=(lx,ly))
        d.set('frame new')       
        d.set_np2arr(image)
    return

def ApplyRealisation(xpapoint):
    d = DS9()
    data = d.get_arr2np()
    law = sys.argv[-1]    #law = 'standard_exponential'
    lx=1
    DictFunction = {'beta':[np.random.beta,1,1],'gamma':[np.random.gamma,1,1],'binomial':[np.random.binomial,10, 0.5],'geometric':[np.random.geometric, 0.5],'pareto':[np.random.pareto, 1],
                     'poisson':[np.random.poisson, 1],'power':[np.random.power, 1],'rand':[np.random.rand],'randint':[np.random.randint, lx],'standard_exponential':[np.random.standard_exponential],
                     'standard_gamma':[np.random.standard_gamma, 1],'standard_normal':[np.random.standard_normal],'standard_t':[np.random.standard_t, 1]}
    if law not in DictFunction.keys():
        print('Please choose a law in the following possibilities: \n','\n'.join(DictFunction.keys()))
    else:
        image = DictFunction[law][0](data)
        d.set('frame new')       
        d.set_np2arr(image)
    return


#        image = np.random.standard_exponential(size=(lx,ly))
#        image =  np.random.beta(1, 1, size=(lx,ly))
#        image =  np.random.binomial(10, 0.5, size=(lx,ly))
#        image =  np.random.geometric(0.5, size=(lx,ly))
#        image =  np.random.pareto(1, size=(lx,ly))
#        image =  np.random.poisson(1, size=(lx,ly))
#        image =  np.random.power(1, size=(lx,ly))
#        image =  np.random.rand(lx,ly)
#        image =  np.random.randint(2, size=(lx,ly))
#        image =  np.random.standard_exponential(size=(lx,ly))
#        image =  np.random.standard_gamma(1, size=(lx,ly))
#        image =  np.random.standard_normal(size=(lx,ly))
#        image =  np.random.standard_t(1,size=(lx,ly))

def CreateImageFromCatalogObject(xpapoint):
    """
    """
    from astropy.table import Table
    from tqdm import tqdm
    from astropy.modeling.functional_models import Gaussian2D


    d = DS9()
    nb = int(1e3)
    lx, ly = 1000, 1000
    try:
        catfile = sys.argv[3]    #law = 'standard_exponential'
    except IndexError:
        x, y, angle = np.random.randint(lx, size=nb), np.random.randint(ly, size=nb), np.random.rand(nb)*2*np.pi - np.pi
        peak = np.random.exponential(10, size=nb)
        sigmax, sigmay = np.random.normal(3, 2, size=nb), np.random.normal(3, 2, size=nb)
        catalog = Table([x, y, peak, sigmax, sigmay, angle], names=('xcenter', 'ycenter', 'peak','sigmax','sigmay', 'angle'))
    else:
        catalog = Table.open(catfile)
    background = 1
    image = np.ones((lx,ly)) * background
    for i in tqdm(range(len(catalog))):
        x = np.linspace(0,lx-1,lx)
        y = np.linspace(0,ly-1,ly)
        x, y = np.meshgrid(x,y)
        image += Gaussian2D.evaluate(x, y, catalog[i]['peak'], catalog[i]['xcenter'], catalog[i]['ycenter'], catalog[i]['sigmax'],  catalog[i]['sigmay'], catalog[i]['angle'])

        #image += twoD_Gaussian(xy,catalog[i]['peak'], catalog[i]['xcenter'], catalog[i]['ycenter'], catalog[i]['sigmax'],  catalog[i]['sigmay'], offset=0, angle=catalog[i]['angle']).reshape(ly,lx).T
    image_real = np.random.poisson(image)
    d.set_np2arr(image_real)
    return

def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, offset=0):
    """Defines a gaussian function in 2D 
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    #A = amplitude/(2*np.pi*sigma_x*sigma_y)    #A to be there?
    g = offset + amplitude * np.exp( - 0.5*(((x-xo)/sigma_x)**2) - 0.5*(((y-yo)/sigma_y)**2))
    return g.ravel()

def galex2Ph_s_A(f200, atm=0.36, throughput=0.13, QE=0.55, area=7854):
    """
    Convert galex fluxes into photons per seconds per angstrom
    """
    Ph_s_A = f200*(throughput*atm* QE*area)
    return Ph_s_A

def FB_ADU2Flux(ADU, EMgain=453, ConversionGain=1.8, dispersion=46.6/10):
    """
    Convert FB2 ADUs into photons per seconds per angstrom
    """
    Flux = ADU * ConversionGain * dispersion / EMgain
    return Flux

def Flux2FBADU(Flux, EMgain=453, ConversionGain=1.8, dispersion=46.6/10):
    """
    Convert FB2 ADUs into photons per seconds per angstrom
    """
    ADU = Flux * EMgain / ConversionGain / dispersion
    #Flux = ADU * ConversionGain * dispersion / EMgain / 10
    return ADU


def BackgroundMeasurement(xpapoint):
    """
    """
    from .focustest import create_DS9regions2
    from decimal import Decimal
    #x, y = x+93, y-93
    d = DS9(xpapoint)
    d.set('frame first')
    n1 = int(d.get('frame'))
    d.set('frame last')
    n2 = int(d.get('frame'))
    n=n2-n1+1
    print('Number of frame = ',n)
    d.set('frame first')
    for frame in range(n):
        data = d.get_fits()[0].data
        texp = d.get_fits()[0].header['EXPTIME']
        xc = [int(2336),int(1500)]
        yc = 1000
        w,l = 300,1900
        create_DS9regions2(xc,[yc]*2, radius=[w,l], form = 'box',save=True,color = 'yellow', savename='/tmp/centers')
        d.set('region delete all')
        d.set('region {}'.format('/tmp/centers.reg'))
        reg = data[int(yc - l/2):int(yc + l/2),int(xc[1] - w/2):int(xc[1] + w/2)]
        regOS = data[int(yc - l/2):int(yc + l/2),int(xc[0] - w/2):int(xc[0] + w/2)]
        print(reg.shape)
        print(reg.shape)
        print(regOS)
        meanADU = np.nanmean(reg) - np.nanmean(regOS)
        stdADU = np.nanstd(reg)
        create_DS9regions2([xc[1]],[yc], radius=[w,l], form = '# text',save=True, text=['mean=%0.3EADU/s/pix - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)],color = 'yellow', savename='/tmp/centers1')
        d.set('region {}'.format('/tmp/centers1.reg')) 
        d.set('frame next')
    return


def SourcePhotometry(xpapoint):
    """vtest
    """
    from .focustest import create_DS9regions
    from decimal import Decimal
    #x, y = x+93, y-93
    d = DS9(xpapoint)
    data = d.get_fits()[0].data
    texp = d.get_fits()[0].header['EXPTIME']
    r = getregion(d, all=True)
    y, x = np.indices((data.shape))
    radius = np.sqrt((x - r.xc)**2 + (y - r.yc)**2)
    physdet = (x>1050) & (x<2050) & (y<2000)

    if hasattr(r, 'w'):
        smallreg = (x>r.xc - r.w/2) & (x<r.xc + r.w/2) & (y>r.yc - r.h/2) & (y<r.yc + r.h/2)
        background = (x>r.xc - r.w) & (x<r.xc + r.w) & (y>r.yc - r.h) & (y<r.yc + r.h)
    if hasattr(r, 'r'):
        smallreg = radius<r.r
        background = radius<2*r.r
    meanADU = np.nansum(data[smallreg & physdet] - np.nanmean(data[~ smallreg & physdet & background]) ) 
    stdADU = np.nanstd(data[~ smallreg & physdet & background])
    if hasattr(r, 'r'):
        create_DS9regions([r.xc], [r.yc], radius=int(2*r.r), form = ['circle'],save=True,color = ['yellow'], ID=[['mean=%.3EADU/s - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)]],savename='/tmp/centers')
    if hasattr(r, 'w'):
        create_DS9regions([r.xc], [r.yc], radius=[2*r.w,2*r.h], form = ['box'],save=True,color = ['yellow'], ID=[['mean=%.3EADU/s - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)]],savename='/tmp/centers')

    d.set('region {}'.format('/tmp/centers.reg')) 
    return


def main():
    """Main function where the arguments are defined and the other functions called
    """
    #path = os.path.dirname(os.path.realpath(__file__))
    print(__file__)
    print(__package__)
    print('Python version = ', sys.version)
    CreateFolders()
    if len(sys.argv)==1:
        try:
            AnsDS9path= resource_filename('DS9FireBall','FireBall.ds9.ans')
        except:
            #sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
            print(__file__)
            pass
        else:
            print(bcolors.BLACK_RED + 'To use DS9Utils, add the following file in the DS9 Preferences->Analysis menu :  \n' + AnsDS9path + bcolors.END)
            print(bcolors.BLACK_RED + 'And switch on Autoreload' + bcolors.END)
            #sys.exit()

#    sys.argv.append('7f000001:65439')
#    sys.argv.append('photo_counting')   
    #sys.argv.append('f2-206')
    #sys.argv.append('10.49-0.25')#16.45-0.15

    
    
    print(datetime.datetime.now())
    start = timeit.default_timer()

    DictFunction = {
                    #GENERIC FUNCTIONS
                    'setup':DS9setup2,'Update':DS9Update, 
                    'DS9plot_spectra_Rest_Frame':DS9plot_spectra_Rest_Frame,'inverse': DS9inverse,
                    'next':DS9next, 'previous':DS9previous, 'ImageAnalysis': DS9AnalyzeImage,
                    'stack': DS9stack,'lock': DS9lock, 'open':DS9open, 'CreateHeaderCatalog':DS9CreateHeaderCatalog,
                    'StackAllImages': StackAllImages, 'DS9Catalog2Region':DS9Catalog2Region, 
                    'DS9Region2Catalog':DS9Region2Catalog, 'DS9MaskRegions':DS9MaskRegions,'BackgroundMeasurement':BackgroundMeasurement,

                    #AIT Functions
                    'centering':DS9center, 'radial_profile':DS9rp,
                    'throughfocus':DS9throughfocus, 'regions': Field_regions,
                    'throughfocus_visualisation':DS9visualisation_throughfocus, 
                    'snr': DS9snr, 'focus': DS9focus,
                    'throughslit': DS9throughslit, 'meanvar': DS9meanvar,
                    'xy_calib': DS9XYAnalysis,'SourcePhotometry':SourcePhotometry,
                    
                    #Flight Functions
                    'ReplaceWithNans': DS9replaceNaNs,'InterpolateNaNs': DS9InterpolateNaNs,
                    'OverscanCorrection': DS9OverscanCorrection, 'Trimming': DS9Trimming,
                    'SubstractImage': DS9RemoveImage, 'TotalReduction': DS9TotalReductionPipeline,
                    'photo_counting':DS9photo_counting, 'WCS':DS9guider, 'Remove_Cosmics': DS9removeCRtails2,
                    'BackgroundSubstraction': DS9SubstractBackground, #'plot_all_spectra':DS9plot_all_spectra,
                    'ColumnLineCorrelation': DS9CLcorrelation, 'OriginalSettings': DS9originalSettings,
                    'NoiseMinimization': DS9NoiseMinimization, 'SmearingProfile': SmearingProfile,
                    'DS9plot_spectra_Observed_Frame':DS9plot_spectra_Observed_Frame, 'lya_multi_image':create_multiImage,
                    'ComputeEmGain': DS9ComputeEmGain, '2D_autocorrelation': DS9_2D_autocorrelation,
                    'HistogramDifferences': DS9HistogramDifferences, 'ComputeStandardDeviation': DS9ComputeStandardDeviation,
                    'PlotSpatial': DS9PlotSpatial, 'PlotArea3D':PlotArea3D, 'ContinuumPhotometry':ContinuumPhotometry,
                    'HistogramSums': HistogramSums, 'DS9DetectCosmics':DS9DetectCosmics, 'DetectHotPixels':DetectHotPixels,
                    'DS9MultipleThreshold': DS9MultipleThreshold, 'DS9removeCRtails_CS':DS9removeCRtails_CS,
                    #Photutils
                    'LoadRegions': DS9LoadCSV, 'BackgroundEstimationPhot': DS9BackgroundEstimationPhot, 'ExtractSources':DS9ExtractSources,
                    'Centroiding': Centroiding,#'PSFphotometry': PSFphotometry,'PSFmatching':PSFmatching,'Datasets': Datasets
                    'GroupingAlgorithm': GroupingAlgorithm, 'AperturePhotometry': AperturePhotometry,'BuildingEPSF': BuildingEPSF,
                    'MorphologicalProperties': MorphologicalProperties, 'EllipticalIsophoteAnalysis': EllipticalIsophoteAnalysis,
                    
                    #Create test images
                    'DS9createImage': DS9createImage, 'FollowProbabilityLaw': FollowProbabilityLaw,'CreateImageFromCatalogObject':CreateImageFromCatalogObject,
                    'ApplyRealisation': ApplyRealisation,
                    
                     #Others
                    'test':DS9tsuite
             }

    xpapoint = sys.argv[1]
    function = sys.argv[2]
    Choose_backend(function)
    
    print(bcolors.BLACK_RED + 'DS9Utils ' + ' '.join(sys.argv[1:]) + bcolors.END)# %s %s '%(xpapoint, function) + ' '.join())
    
    print(bcolors.GREEN_WHITE + """
          ********************************************************************
                                     Function = %s
          ********************************************************************"""%(function)+ bcolors.END)
    a = DictFunction[function](xpapoint)             
    stop = timeit.default_timer()
    print(bcolors.BLACK_GREEN + """
        ********************************************************************
                            Exited OK, test duration = {}s      
        ******************************************************************** """.format(stop - start) + bcolors.END)
    return a



if __name__ == '__main__':
    a = main()

