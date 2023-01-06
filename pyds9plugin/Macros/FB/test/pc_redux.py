#!/usr/bin/env python

import numpy as np
import numpy.ma as ma
import os, sys, pyfits
from optparse import OptionParser
import scipy as sp
from time import time
from scipy import signal
import glob
import matplotlib.mlab as mlab
from pylab import *
from scipy.stats import norm
from scipy.optimize import curve_fit
from subprocess import call
from astropy.io import fits

##CHECK the order of the image cubes. The noise from each cube doesn't look like it's in the right order for some reason!!!!!!

data_dir = '/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220204_darks_T183_1MHz/6800/'
data_dir = data_dir+'redux/'

gain = 4.3

##########################################

##Markers for plotting
##http://matplotlib.org/api/markers_api.html
##Colours for plotting
##http://stackoverflow.com/questions/22408237/named-colors-in-matplotlib

img_area = [1100,1900,1300,1600]
stg_area = [100,900,1300,1600]
ovr_area = [500,1500,2600,3200]
pre_area = [500,1500,200,800]
#crop_area = [0,1500,2150,3000]

try:
    import multiprocessing as mp
except:
    print('\n Info : Multiprocessing module not present. Running on single core.')
    n_cpus = 1
else:
    n_cpus = mp.cpu_count()

plot_flag = True
try:
    import matplotlib.pylab as plt
except ImportError:
    plot_flag = False


def list_slice(S, step):
    return [S[i::step] for i in range(step)]

def gaussian(x, amp, x0, sigma):

    """
    One dimensional Gaussian function.

    Parameters
    ----------
    x : numpy array
        One dimensional vector

    amp : float
        Gaussian function amplitude

    x0 : float
        Mean value of the Gaussian function

    sigma : float
        Standard deviation of the Gaussian

    Returns
    -------
    Gaussian function
    """

    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussianFit(x, y, param):

    """
    Fit a Gaussian.

    Parameters
    ----------
    x : numpy array
        x-axis one dimensional vector

    y : numpy array
        y-axis one dimensional vector

    param : list
        List of guess parameter values

    Returns
    -------
    amp, x0, sigma : tuple of floats
        Fitted Gaussian curve parameter values
    """
    popt, pcov = curve_fit(gaussian, x, y, p0 = param,maxfev=5000)

    amp, x0, sigma = popt

    return (amp, x0, sigma)

def linefit(x, A, B):

    """
    Generate a line.

    Parameters
    ----------
    x : numpy array
        Linear vector

    a : float
        Line slope

    b : float
        Line intercept

    Returns
    -------
    Line function.
    """

    return A*x + B

def fitLine(x, y, param = None):
    """
    Fit a line.

    Parameters
    ----------
    x : numpy array
        One dimensional vector.

    y : numpy array
        One dimensional vector.

    param : list
       List of guess parameters
    """
    popt, pcov = curve_fit(linefit, x, y, p0 = param)

    a, b = popt

    return (a, b)

def ind2sub(array_shape, ind):
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)


def plot_hist(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,ygaussfit,n_bias,n_log,threshold0,threshold55):

    if plot_flag:

        plt.close("all")
        plt.clf()
        #plt.figure(figsize=(20,15))
        #plt.rc('text', usetex=True)
        plt.rcParams['mathtext.fontset']='stix'
        plt.rcParams['font.size']='40'
        #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times']})
        plt.xlabel("Pixel Value [ADU]",fontsize=20)
        plt.ylabel("# Pixels",fontsize=20)
        #plt.axis([0,np.max(n_log),0,bins[bins.size-1]])
        plt.plot(bin_center, n_log, "rx", label="Histogram")
        plt.plot(xgaussfit,np.log(ygaussfit), "b-", label="Gaussian")
        plt.plot(threshold0,n_log, "b--", label="Bias")
        plt.plot(threshold55, n_log, "k--", label="5.5 Sigma")
        plt.plot(xlinefit,ylinefit, "g--", label="EM gain fit")
        plt.figtext(.65, .15,'Bias value = %0.3f DN \nReadnoise = %0.3f e \n EM gain = %0.3f e/e' % (float(bias), float(sigma)*gain, float(emgain)), fontsize=10)
        plt.legend(loc="upper right",fontsize=15)

        plt.grid(b=True, which='major', color='0.75', linestyle='--')
        plt.grid(b=True, which='minor', color='0.75', linestyle='--')
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        axes = plt.gca()
        axes.set_ylim([0,np.log(n_bias)+0.1])
        axes.set_xlim([bias-500,bias+3000])

        fn = data_dir + 'figures/histograms/'
        imgname = image.replace(data_dir,'')
        fig_dir = os.path.dirname(fn)
        if not os.path.exists(fig_dir):# create data directory if needed
               os.makedirs(fig_dir)
        figname = fn + imgname.replace('.fits', '.hist.png')
        if os.path.isfile(figname):
            os.remove(figname)
        plt.savefig(fn + imgname.replace('.fits', '.hist.png'), dpi = 100, bbox_inches = 'tight')
        #plt.savefig(image.replace('.fits', '.hist.png'), dpi = 100, bbox_inches = 'tight')
        plt.close()

def calc_emgain(image, area):

        # Read data from FITS image
    try:
        img_data,header = fits.getdata(image,header=True)
    except IOError:
        raise IOError("Unable to open FITS image %s" %(image))


    if np.ndim(img_data) == 3:
        # Image dimension
        zsize, ysize, xsize = img_data.shape
        img_section = img_data[:,area[0]:area[1],area[2]:area[3]]
        stddev = np.std(img_data[:,area[0]:area[1],area[2]:area[3]])
        img_size = img_section.size

    else:
        # Image dimension
        ysize, xsize = img_data.shape
        img_section = img_data[area[0]:area[1],area[2]:area[3]]
        stddev = np.std(img_data[area[0]:area[1],area[2]:area[3]])
        img_size = img_section.size

    #print ysize, xsize
    nbins = 50
    readnoise = 50
        #gain = float(gain)

    # Histogram of the pixel values
    n, bins = np.histogram(np.array(img_section), bins = nbins)
    bin_center = 0.5 * (bins[:-1] + bins[1:])
    y0 = np.min(n)

    n_log = np.log(n)

    # What is the mean bias value?
    idx = np.where(n == n.max())
    bias = bin_center[idx][0]
    n_bias = n[idx][0]

    # Range of data in which to fit the Gaussian to calculate sigma
    bias_lower = bias - float(1.5) * readnoise
    bias_upper = bias + float(2.0) * readnoise

    idx_lower = np.where(bin_center >= bias_lower)[0][0]
    idx_upper = np.where(bin_center >= bias_upper)[0][0]

    gauss_range = np.where(bin_center >= bias_lower)[0][0]

    valid_idx = np.where(n[idx_lower:idx_upper] > 0)

    amp, x0, sigma = gaussianFit(bin_center[idx_lower:idx_upper][valid_idx], n[idx_lower:idx_upper][valid_idx], [n_bias, bias, readnoise])

    #plt.figure()
    #plt.plot(bin_center[idx_lower:idx_upper], n[idx_lower:idx_upper], 'r.')
    #plt.show()

    # Fitted frequency values
    xgaussfit = np.linspace(bin_center[idx_lower], bin_center[idx_upper], 1000)
    #print xgaussfit
    ygaussfit = gaussian(xgaussfit, amp, x0, sigma)
    #print ygaussfit

    max_sig = np.max(bin_center)

    # Define index of "linear" part of the curve
    threshold_min = bias + (float(6.0) * sigma) #Adjust lower limit for emgain line fit depending on level of gain
    threshold_max = max_sig - 5 #bias + (float(160.0) * sigma) #Adjust upper limit for emgain line fit depending on level of gain
    threshold_55 = bias + (float(5.5) * sigma)

    #print threshold_max

    # Lines for bias, 5.5*sigma line

    n_line = n_log.size
    zeroline = np.zeros([n_line], dtype = np.float32)
    threshold0 = zeroline + int(bias)
    threshold55 = zeroline + int(bias + 5.5*sigma)
    thresholdmin = zeroline + int(threshold_min)
    thresholdmax = zeroline + int(threshold_max)

    idx_threshmin = np.array(np.where(bin_center >= threshold_min))[0,0]
    idx_threshmax = np.array(np.where(bin_center >= threshold_max))[0,0]

    print(idx_threshmin)
    print(idx_threshmax)

    valid_idx = np.where(n[idx_threshmin:idx_threshmax] > 0)

    slope, intercept = fitLine(bin_center[idx_threshmin:idx_threshmax][valid_idx], n_log[idx_threshmin:idx_threshmax][valid_idx])

    # Fit line
    #print np.max(bin_center)
    xlinefit = np.linspace(bias-20*sigma, np.max(bin_center), 1000)
    #xlinefit = np.linspace(bias-20*sigma, bin_center[idx_threshmax], 1000)
    ylinefit = linefit(xlinefit, slope, intercept)


    emgain = (-1./slope) * (gain)
    print(emgain)

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
    #print frac_lost

    plot_hist(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,ygaussfit,n_bias,n_log,threshold0,threshold55)

    return (emgain,bias,sigma,frac_lost)


def calc_emgain_CR(image, area):

    print('We are starting' + image)


    # Read data from FITS image
    try:
        img_data,header = fits.getdata(image,header=True)
    except IOError:
        raise IOError("Unable to open FITS image %s" %(image))


    if np.ndim(img_data) == 3:
        # Image dimension
        zsize, ysize, xsize = img_data.shape
        img_section = img_data[:,area[0]:area[1],area[2]:area[3]]
        stddev = np.std(img_data[:,area[0]:area[1],area[2]:area[3]])
        img_size = img_section.size

    else:
        # Image dimension
        ysize, xsize = img_data.shape
        img_section = img_data[area[0]:area[1],area[2]:area[3]]
        stddev = np.std(img_data[area[0]:area[1],area[2]:area[3]])
        img_size = img_section.size

    #print ysize, xsize
    nbins = 50
    readnoise = 50
        #gain = float(gain)

    # Histogram of the pixel values
    n, bins = np.histogram(np.array(img_section), bins = nbins)
    bin_center = 0.5 * (bins[:-1] + bins[1:])
    y0 = np.min(n)

    n_log = np.log(n)

    # What is the mean bias value?
    idx = np.where(n == n.max())
    bias = bin_center[idx][0]
    n_bias = n[idx][0]

    # Range of data in which to fit the Gaussian to calculate sigma
    bias_lower = bias - float(1.5) * readnoise
    bias_upper = bias + float(2.0) * readnoise

    idx_lower = np.where(bin_center >= bias_lower)[0][0]
    idx_upper = np.where(bin_center >= bias_upper)[0][0]

    gauss_range = np.where(bin_center >= bias_lower)[0][0]

    valid_idx = np.where(n[idx_lower:idx_upper] > 0)

    amp, x0, sigma = gaussianFit(bin_center[idx_lower:idx_upper][valid_idx], n[idx_lower:idx_upper][valid_idx], [n_bias, bias, readnoise])

    #plt.figure()
    #plt.plot(bin_center[idx_lower:idx_upper], n[idx_lower:idx_upper], 'r.')
    #plt.show()

    # Fitted frequency values
    xgaussfit = np.linspace(bin_center[idx_lower], bin_center[idx_upper], 1000)
    #print xgaussfit
    ygaussfit = gaussian(xgaussfit, amp, x0, sigma)
    #print ygaussfit

    max_sig = np.max(bin_center)

    # Define index of "linear" part of the curve
    threshold_min = bias + (float(6.0) * sigma) #Adjust lower limit for emgain line fit depending on level of gain
    threshold_max = max_sig #bias + (float(180.0) * sigma) #Adjust upper limit for emgain line fit depending on level of gain
    threshold_55 = bias + (float(5.5) * sigma)

    #print threshold_max

        # Lines for bias, 5.5*sigma line

    n_line = n_log.size
    zeroline = np.zeros([n_line], dtype = np.float32)
    threshold0 = zeroline + int(bias)
    threshold55 = zeroline + int(bias + 5.5*sigma)
    thresholdmin = zeroline + int(threshold_min)
    thresholdmax = zeroline + int(threshold_max)

    print(np.array(np.where(bin_center >= threshold_min)))
    idx_threshmin = np.array(np.where(bin_center >= threshold_min))[0,0]
    idx_threshmax = np.array(np.where(bin_center >= threshold_max))[0,0]

    valid_idx = np.where(n[idx_threshmin:idx_threshmax] > 0)

    slope, intercept = fitLine(bin_center[idx_threshmin:idx_threshmax][valid_idx], n_log[idx_threshmin:idx_threshmax][valid_idx])

    # Fit line
    #print np.max(bin_center)
    xlinefit = np.linspace(bias-20*sigma, np.max(bin_center), 1000)
    #xlinefit = np.linspace(bias-20*sigma, bin_center[idx_threshmax], 1000)
    ylinefit = linefit(xlinefit, slope, intercept)


    emgain = (-1./slope) * (gain)
    print(emgain)

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
    #print frac_lost

    plot_hist(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,ygaussfit,n_bias,n_log,threshold0,threshold55)

    return (emgain,bias,sigma,frac_lost)



def make_datacubes():

    glob_pattern = data_dir + 'image0*.fits'
    extrapattern = data_dir + 'image0*.*.fits'
    images =  sorted(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern)))

    data,header = fits.getdata(images[0],header=True)
    n = header['IMBURST']

    j = 0

    for i in range(0, len(images), n):
        sublist = images[i:i + n]
        image_concat = []
        for image in sublist:
            data,header = fits.getdata(image,header=True)
            image_concat.append(data)

        imagecube = data_dir + 'image_cube_' + str(j) + '.fits'
        if os.path.exists(imagecube):
            os.remove(imagecube)

        fits.writeto(imagecube,np.array(image_concat),header=header,overwrite=True)
        print('Processed: image_cube_' + str(j) + '.fits')
        j = j + 1


def cropimages():

    glob_pattern = data_dir + 'image0*.fits'
    extrapattern = data_dir + 'image0*.*.fits'
    images =  sorted(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern)))

    print(images)

    data,header = fits.getdata(images[0],header=True)
    n = header['IMBURST']

    hist_output = calc_emgain(images[0],pre_area)
    prandom_array = np.random.normal(int(hist_output[1]),int(hist_output[2]), size=(data.shape[0],data.shape[1]))

    for img in images:
        data,header = fits.getdata(img,header=True)
        #data[1640:1720,:] = prandom_array[1640:1720,:]
        data[:,1450:1480] = prandom_array[:,1450:1480]
        data[:,1980:2150] = prandom_array[:,1980:2150]
        data[1750:,:] = prandom_array[1750:,:]
        new_img = img.replace(data_dir,rdata_dir)
        fits.writeto(new_img,data,header=header,overwrite=True)


def make_HP_mask():

    glob_pattern = data_dir + 'image0*.fits'
    extrapattern = data_dir + 'image0*.*.fits'
    images =  sorted(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern)))

    data,header = fits.getdata(images[0],header=True)
    n = header['IMBURST']

    j = 0

    for i in range(0, len(images), n):
        sublist = images[i:i + n]
        image_concat = []
        for image in sublist:
            data,header = fits.getdata(image,header=True)
            image_concat.append(data)

        z, y, x = np.array(image_concat).shape
        #print image_concat

        image_mean = np.median(np.array(image_concat),axis=0)
        imagesum = data_dir + 'image_median_' + str(j) + '.fits'
    if os.path.exists(imagesum):
        os.remove(imagesum)
    fits.writeto(imagesum,np.array(image_mean),header=header)
    print('Processed: image_median_' + str(j) + '.fits')
    j = j + 1


def make_CR_datacubes():

    glob_pattern = data_dir + 'image0*.CR.fits'
    extrapattern = data_dir + 'image0*mask*.CR.fits'
    images =  sorted(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern)))

    data,header = fits.getdata(images[0],header=True)
    n = header['IMBURST']

    j = 0

    for i in range(0, len(images), n):
        sublist = images[i:i + n]
        image_concat = []
        for image in sublist:
            data,header = fits.getdata(image,header=True)
            image_concat.append(data)

        imagecube = data_dir + 'image_CR_cube_' + str(j) + '.fits'
    if os.path.exists(imagecube):
        os.remove(imagecube)
    fits.writeto(imagecube,np.array(image_concat),header=header)
    print('Processed: image_CR_cube_' + str(j) + '.fits')
    j = j + 1


def make_CRmask_datacubes():

    glob_pattern = data_dir + 'image0*mask*.CR.fits'
    extrapattern = data_dir + 'image0*mask*.countCR.fits'
    images =  sorted(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern)))

    data,header = fits.getdata(images[0],header=True)
    n = header['IMBURST']

    j = 0

    for i in range(0, len(images), n):
        sublist = images[i:i + n]
        image_concat = []
        for image in sublist:
            data,header = fits.getdata(image,header=True)
            image_concat.append(data)

        imagecube = data_dir + 'image_CRmask_cube_' + str(j) + '.fits'
    if os.path.exists(imagecube):
        os.remove(imagecube)
    fits.writeto(imagecube,np.array(image_concat),header=header)
    print('Processed: image_CRmask_cube_' + str(j) + '.fits')
    j = j + 1

def make_corr_datacubes():

    glob_pattern = data_dir + 'image_CR_cube_*.corr.fits'
    extrapattern = data_dir + 'image_CR_cube_*mask.fits'
    images =  sorted(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern)))

    data,header = fits.getdata(images[0],header=True)
    n = header['IMBURST']

    j = 0


    for i in range(0, len(images), n):
        sublist = images[i:i + n]
        image_concat = []
        for image in sublist:
            data,header = fits.getdata(image,header=True)
            image_concat.append(data)

        imagecube = data_dir + 'image_corr_cube_' + str(j) + '.fits'
    if os.path.exists(imagecube):
        os.remove(imagecube)
    fits.writeto(imagecube,np.array(image_concat),header=header)
    print('Processed: image_corr_cube_' + str(j) + '.fits')
    j = j + 1


def countCR(indata):

    image,area = indata

    hist_output = calc_emgain_CR(image,area)
        #np.savetxt(data_dir + 'hist_output',hist_output,fmt='%d')

    print("Image #: ", image)

    img_data,header = fits.getdata(image,header=True)
    # Image dimension
        #zsize, ysize, xsize = img_data.shape
    # Image dimension
    ysize, xsize = img_data.shape

    cutoff = int(hist_output[1]) + 5000
    print(cutoff)

    cleaned_img_data = np.copy(img_data)

    img_data_mask = np.zeros([img_data.shape[0], img_data.shape[1]], dtype = np.int16)
    CR_data_mask = np.zeros([img_data.shape[0], img_data.shape[1]], dtype = np.int16)

    prandom_array = np.random.normal(int(hist_output[1]),int(hist_output[2]), size=(img_data.shape[0],img_data.shape[1]))
    #cleaned_img_data[:,3205:3217] = prandom_array[:,3205:3217]
        #print prandom_array[:,3205:3217]

    CR_check = 5
    CRcounter = 0
    zeroCR = 0
    HPcounter = 0

    zero_img_data = np.array(np.where(cleaned_img_data == 0))

    for i in range(0,zero_img_data.shape[1]):

       xzero = zero_img_data[1,i]
       yzero = zero_img_data[0,i]

       topcounter = 0
       botcounter = 0

       if xzero > 1 and xzero < img_data.shape[1] - 1 and yzero >= 0 and yzero < img_data.shape[0] - 1:
           #Top pixel CR
           yval = yzero
           while cleaned_img_data[yval,xzero+1] > cutoff:
           #right pixels
               xval = xzero+1
               counter = 0
               if xval == img_data_mask.shape[1]-1:
                   break

               while cleaned_img_data[yval,xval+1] > cutoff:

                   xval = xval + 1
                   counter = counter + 1
                   if xzero == img_data_mask.shape[1]-1:
                       break

               trpix = counter

           #left pixels
           counter = 0
           xval = xzero-1
           while cleaned_img_data[yval,xval] > cutoff:

                xval = xval - 1
                counter = counter + 1
                if xval < 1:
                       break
                tlpix = counter

                topcounter = topcounter + trpix + tlpix
                CR_data_mask[yval,xzero-tlpix:xzero+trpix+1] = -1
 
                yval = yval + 1

                if yval == img_data_mask.shape[0]-1:
                    break
    
            #Bottom pixel CR
       #right pixels
           yval = yzero-1
           while cleaned_img_data[yval,xzero+1] > cutoff and yzero >= 0:
               counter = 0
               xval = xzero+1
               while cleaned_img_data[yval,xval+1] > cutoff:

                   xval = xval + 1
                   counter = counter + 1
                   if xval == img_data_mask.shape[1]-1:
                       break

               brpix = counter

               #left pixels
               counter = 0
               xval = xzero-1
               while cleaned_img_data[yval,xval] > cutoff:

                   xval = xval - 1
                   counter = counter + 1
                   if xval < 1:
                       break

               blpix = counter

               botcounter = botcounter + brpix + blpix
               CR_data_mask[yval,xzero-blpix:xzero+brpix+1] = -1

               yval = yval - 1

               if yval <= 0:
                   break

       if (topcounter+botcounter) >= CR_check:
           zero_mask_idx = CR_data_mask < 0
           zeroCR = zeroCR + 1

       clean_mask_idx = CR_data_mask < 0
       cleaned_img_data[clean_mask_idx] = prandom_array[clean_mask_idx]

    maxvalue = np.max(cleaned_img_data)
    max_img_data = np.array(np.where(cleaned_img_data == maxvalue))

    xind = max_img_data[1,0]
    yind = max_img_data[0,0]

     #print "\t  ZeroCR counted in image : %s " %(zeroCR)

    while maxvalue > cutoff and img_data_mask[yind,xind] == 0 and CR_data_mask[yind,xind] == 0:

            #CR_data_mask[yind,xind] = -1
        img_data_mask[yind,xind] = -1
        topcounter = 0
        botcounter = 0

        if xind > 1 and xind < (img_data.shape[1] - 1) and yind >= 1 and yind < (img_data.shape[0] - 1):

    #Top pixel CR
            yval = yind
            while cleaned_img_data[yval,xind] > cutoff:

                #right pixels

                xval = xind
                counter = 0

                while cleaned_img_data[yval,xval] > cutoff:
            #print yval, xval
                    xval = xval + 1
                    counter = counter + 1
                    if xval == img_data_mask.shape[1]-1:
                         break

                trpix = counter
    
                #left pixels
                xval = xind-1
                counter = 0

                while cleaned_img_data[yval,xval] > cutoff:

                    xval = xval - 1
                    counter = counter + 1
                    if xval < 1:
                        break

                tlpix = counter

                topcounter = trpix + tlpix

                CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1

                yval = yval + 1

                if yval == img_data_mask.shape[0]-1:
                    break


            #Bottom pixel CR
            #right pixels
            yval = yind-1
            while cleaned_img_data[yval,xind] > cutoff and yind >= 0:
                counter = 0
                xval = xind + 1
                if xval == img_data_mask.shape[1]-1:
                    break

                while cleaned_img_data[yval,xval] > cutoff:
                    xval = xval + 1
                    counter = counter + 1
                    if xval == img_data_mask.shape[1]-1:
                        break

                brpix = counter

                #left pixels
                counter = 0
                xval = xind - 1

                while cleaned_img_data[yval,xval] > cutoff:

                    xval = xval - 1
                    counter = counter + 1
                    if xval < 1:
                        break

                blpix = counter

                botcounter = botcounter + brpix + blpix
                CR_data_mask[yval,xind-blpix:xind+brpix+1] = -1

                yval = yval - 1

                if yval <= 0:
                    break

        if (topcounter+botcounter) >= CR_check:
            CR_mask_idx = CR_data_mask < 0
            CRcounter = CRcounter + 1
        else:
            HPcounter = HPcounter + 1

        clean_mask_idx = CR_data_mask < 0
        cleaned_img_data[clean_mask_idx] = prandom_array[clean_mask_idx]

        maxvalue = np.max(cleaned_img_data)
    max_img_data = np.array(np.where(cleaned_img_data == maxvalue))
    xind = max_img_data[1,0]
    yind = max_img_data[0,0]

#new_image_fname = image.replace(".fits", ".CR.fits")
      #if os.path.exists(new_image_fname):
      #    os.remove(new_image_fname)
      #hdu = pyfits.PrimaryHDU(cleaned_img_data)
      #hdu.writeto(new_image_fname)
      #print "\t  CR corrected image : %s" %(new_image_fname)
    print("\t  CR counted in image : %s " %(zeroCR+CRcounter))
    print("\t  Hot pixels counted in image : %s " %(HPcounter))


    # Generate mask FITS image
    mask_fname = image.replace(".fits", ".mask.countCR.fits")
    if os.path.exists(mask_fname):
        os.remove(mask_fname)
    fits.writeto(mask_fname,CR_data_mask,header=header)
    print("\t  Corrected Mask : %s" %(mask_fname))

    imageout = image.replace(data_dir,'')
    imageout = imageout.replace('.fits','').replace('image','')

    output = np.array([imageout,image.replace(data_dir,''),(zeroCR+CRcounter),(HPcounter)])
    print(output)

    return(output)



def removeCRtails(indata):

    image,area = indata

    hist_output = calc_emgain_CR(image,area)
        #np.savetxt(data_dir + 'hist_output',hist_output,fmt='%d')

    print("Image #: ", image)

    img_data,header = fits.getdata(image,header=True)
    # Image dimension
        #zsize, ysize, xsize = img_data.shape
    # Image dimension
    ysize, xsize = img_data.shape

    cutoff = int(hist_output[1]) + 5000
    print(cutoff)

    cleaned_img_data = np.copy(img_data)

    img_data_mask = np.zeros([img_data.shape[0], img_data.shape[1]], dtype = np.int16)
    CR_data_mask = np.zeros([img_data.shape[0], img_data.shape[1]], dtype = np.int16)

    prandom_array = np.random.normal(int(hist_output[1]),int(hist_output[2]), size=(img_data.shape[0],img_data.shape[1]))
    #cleaned_img_data[:,3205:3217] = prandom_array[:,3205:3217]
        #print prandom_array[:,3205:3217]

    CR_check = 5
    CRcounter = 0
    zeroCR = 0
    HPcounter = 0

    zero_img_data = np.array(np.where(cleaned_img_data == 0))

    for i in range(0,zero_img_data.shape[1]):

       xzero = zero_img_data[1,i]
       yzero = zero_img_data[0,i]

       topcounter = 0
           botcounter = 0

           if xzero > 1 and xzero < img_data.shape[1] - 1 and yzero >= 0 and yzero < img_data.shape[0] - 1:
               #Top pixel CR
               yval = yzero
               while cleaned_img_data[yval,xzero+1] > cutoff:
               #right pixels
                   xval = xzero+1
                   counter = 0
                   if xval == img_data_mask.shape[1]-1:
                       break

                   while cleaned_img_data[yval,xval+1] > cutoff:

                       xval = xval + 1
                       counter = counter + 1
                       if xzero == img_data_mask.shape[1]-1:
                           break

                   trpix = counter

               #left pixels
               counter = 0
               xval = xzero-1
               while cleaned_img_data[yval,xval] > cutoff and xval > 1:

                       xval = xval - 1
                    counter = counter + 1
                    if xval < 1:
                           break

                   tlpix = counter

                   topcounter = topcounter + trpix + tlpix
                   CR_data_mask[yval,xzero-tlpix:xzero+trpix+1] = -1

                   yval = yval + 1

                   if yval == img_data_mask.shape[0]-1:
                       break

               #Bottom pixel CR
           #right pixels
               yval = yzero-1
               while cleaned_img_data[yval,xzero+1] > cutoff and yzero >= 0:
                   counter = 0
                   xval = xzero+1
                   if xval == img_data_mask.shape[1]-1:
                       break

                   while cleaned_img_data[yval,xval+1] > cutoff:

                       xval = xval + 1
                       counter = counter + 1
                       if xval == img_data_mask.shape[1]-1:
                           break

                   brpix = counter

                   #left pixels
                   counter = 0
                   xval = xzero-1
                   while cleaned_img_data[yval,xval] > cutoff and xval > 1:

                       xval = xval - 1
                       counter = counter + 1
                       if xval < 1:
                           break

                   blpix = counter

                   botcounter = botcounter + brpix + blpix
                   CR_data_mask[yval,xzero-blpix:xzero+brpix+1] = -1

                   yval = yval - 1

                   if yval <= 0:
                       break

           if (topcounter+botcounter) >= CR_check:
               zero_mask_idx = CR_data_mask < 0
               zeroCR = zeroCR + 1

           clean_mask_idx = CR_data_mask < 0
           cleaned_img_data[clean_mask_idx] = prandom_array[clean_mask_idx]

        maxvalue = np.max(cleaned_img_data)
        max_img_data = np.array(np.where(cleaned_img_data == maxvalue))

    xind = max_img_data[1,0]
    yind = max_img_data[0,0]

     #print "\t  ZeroCR counted in image : %s " %(zeroCR)

    while maxvalue > cutoff and img_data_mask[yind,xind] == 0 and CR_data_mask[yind,xind] == 0:

            #CR_data_mask[yind,xind] = -1
        img_data_mask[yind,xind] = -1
            topcounter = 0
            botcounter = 0

            if xind > 1 and xind < img_data.shape[1] - 1 and yind >= 0 and yind < img_data.shape[0] - 1:

        #Top pixel CR
                yval = yind
                while cleaned_img_data[yval,xind] > cutoff:

                    #right pixels

            xval = xind
                    counter = 0

                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval + 1
                     counter = counter + 1
                     if xval == img_data_mask.shape[1]-1:
                    break

                trpix = counter

                #left pixels
            xval = xind-1
                    counter = 0

                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval - 1
                        counter = counter + 1
                        if xval < 1:
                            break

                    tlpix = counter

                    topcounter = trpix + tlpix
                    #print 'first ' + str(topcounter)

                    #if topcounter >= CR_check:
                    #    CR_data_mask[yval,:] = -1
            if topcounter >= 3:
                        #print 'second ' + str(topcounter)
                        #CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1
                        #CR_data_mask[yval,:] = -1
            CR_data_mask[yval:yval+10,:] = -1
            if topcounter <= 2:
                        #print 'third ' + str(topcounter)
                        #CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1
                        CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1

                    yval = yval + 1

                    if yval == img_data_mask.shape[0]-1:
                        break


                #Bottom pixel CR
                #right pixels
                yval = yind-1
                while cleaned_img_data[yval,xind] > cutoff and yind >= 0:
                    counter = 0
                    xval = xind + 1
                    if xval == img_data_mask.shape[1]-1:
                        break

                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval + 1
                        counter = counter + 1
                        if xval == img_data_mask.shape[1]-1:
                            break

                    brpix = counter

                    #left pixels
                    counter = 0
                xval = xind - 1

                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval - 1
                        counter = counter + 1
                        if xval < 1:
                            break

                    blpix = counter

                    botcounter = botcounter + brpix + blpix
            #CR_data_mask[yval,xind-blpix:xind+brpix+1] = -1

                    #if topcounter >= CR_check:
                    #    CR_data_mask[yval,:] = -1
            if botcounter >= 3:
                        #print 'second ' + str(botcounter)
                        #CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1
                        CR_data_mask[yval-3:yval,:] = -1
            if botcounter <= 2:
                        #print 'third ' + str(botcounter)
                        #CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1
                        CR_data_mask[yval,xind-blpix:xind+brpix+1] = -1

                    yval = yval - 1

                    if yval <= 0:
                        break

            if (topcounter+botcounter) >= CR_check:
                CR_mask_idx = CR_data_mask < 0
                CRcounter = CRcounter + 1
        else:
                HPcounter = HPcounter + 1

            clean_mask_idx = CR_data_mask < 0
            cleaned_img_data[clean_mask_idx] = prandom_array[clean_mask_idx]

            maxvalue = np.max(cleaned_img_data)
        max_img_data = np.array(np.where(cleaned_img_data == maxvalue))
        xind = max_img_data[1,0]
        yind = max_img_data[0,0]

    new_image_fname = image.replace(".fits", ".CR.fits")
          if os.path.exists(new_image_fname):
              os.remove(new_image_fname)
          fits.writeto(new_image_fname,cleaned_img_data,header=header)
          print("\t  CR corrected image : %s" %(new_image_fname))
          print("\t  CR counted in image : %s " %(zeroCR+CRcounter))
    print("\t  Hot pixels counted in image : %s " %(HPcounter))


    # Generate mask FITS image
    mask_fname = image.replace(".fits", ".mask.CR.fits")
    if os.path.exists(mask_fname):
        os.remove(mask_fname)
    fits.writeto(mask_fname,CR_data_mask,header=header)
    print("\t  Corrected Mask : %s" %(mask_fname))

        imageout = image.replace(data_dir,'')
    imageout = imageout.replace('.fits','').replace('image','')

    output = np.array([imageout,image.replace(data_dir,''),(zeroCR+CRcounter),(HPcounter)])
    print(output)

    return(output)

def removeCR(indata):

        image, hist_output = indata

    print("Image #: ", image)

    img_data,header = fits.getdata(image,header=True)
    # Image dimension
        #zsize, ysize, xsize = img_data.shape
    # Image dimension
        ysize, xsize = img_data.shape

        checkval = int(hist_output[1]) + 5000
        #cutoff = int(hist_output[1]) + 5000
        cutoff = checkval

        cleaned_img_data = np.copy(img_data)

        img_data_mask = np.zeros([img_data.shape[0], img_data.shape[1]], dtype = np.int16)
        CR_data_mask = np.zeros([img_data.shape[0], img_data.shape[1]], dtype = np.int16)

    prandom_array = np.random.normal(int(hist_output[1]),int(hist_output[2]), size=(img_data.shape[0],img_data.shape[1]))

        CR_check = 5
        CRcounter = 0
        zeroCR = 0

    zero_img_data = np.array(np.where(img_data == 0))

    for i in range(0,zero_img_data.shape[1]):

       xzero = zero_img_data[1,i]
       yzero = zero_img_data[0,i]

       topcounter = 0
           botcounter = 0

           if xzero > 1 and xzero < img_data.shape[1] - 1 and yzero >= 0 and yzero < img_data.shape[0] - 1:
               #Top pixel CR
               yval = yzero
               while cleaned_img_data[yval,xzero+1] > cutoff:

                   #right pixels
                   xval = xzero
                   counter = 0
                   while cleaned_img_data[yval,xval+1] > cutoff:

                       xval = xval + 1
                    counter = counter + 1
                    if xzero == img_data_mask.shape[1]-1:
                        break

                   trpix = counter

               #left pixels
               counter = 0
              xval = xzero-1
               while cleaned_img_data[yval,xval] > cutoff:

                       xval = xval - 1
                    counter = counter + 1
                    if xval < 1:
                           break

                   tlpix = counter

                   topcounter = topcounter + trpix + tlpix
                   CR_data_mask[yval,xzero-tlpix:xzero+trpix+1] = -1

                   yval = yval + 1

                   if yval == img_data_mask.shape[0]-1:
                       break

               #Bottom pixel CR
               yval = yzero-1
               while cleaned_img_data[yval,xzero+1] > cutoff and yzero >= 0:
                   counter = 0
                   xval = xzero
                   while cleaned_img_data[yval,xval+1] > cutoff:

                       xval = xval + 1
                       counter = counter + 1
                       if xval == img_data_mask.shape[1]-1:
                           break

                   brpix = counter

                   #left pixels
                   counter = 0
                   xval = xzero-1
                   while cleaned_img_data[yval,xval] > cutoff:

                       xval = xval - 1
                       counter = counter + 1
                       if xval < 1:
                           break

                   blpix = counter

                   botcounter = botcounter + brpix + blpix
                   CR_data_mask[yval,xzero-blpix:xzero+brpix+1] = -1

                   yval = yval - 1

                   if yval <= 0:
                       break

           if (topcounter+botcounter) >= CR_check:
               #zero_mask_idx = CR_data_mask < 0
               zeroCR = zeroCR + 1

        clean_mask_idx = CR_data_mask < 0
        cleaned_img_data[clean_mask_idx] = prandom_array[clean_mask_idx]

        maxvalue = np.max(cleaned_img_data)
        max_img_data = np.array(np.where(cleaned_img_data == maxvalue))
    xind = max_img_data[1,0]
    yind = max_img_data[0,0]

        while maxvalue > checkval and img_data_mask[yind,xind] == 0:

            img_data_mask[yind,xind] = -1
            #print maxvalue
            #print xind, yind
            topcounter = 0
            botcounter = 0

            if xind > 1 and xind < img_data.shape[1] - 1 and yind >= 0 and yind < img_data.shape[0] - 1:

                #Top pixel CR
                yval = yind
                while cleaned_img_data[yval,xind] > cutoff:

                    #right pixels
                    xval = xind
                    counter = 0
                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval + 1
                     counter = counter + 1
                     if xval == img_data_mask.shape[1]-1:
                break

                trpix = counter

                #left pixels
                    counter = 0
                xval = xind-1
                while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval - 1
                     counter = counter + 1
                     if xval < 1:
                            break

                    tlpix = counter

                    CR_data_mask[yval,xind-tlpix:xind+trpix+1] = -1
                    topcounter = topcounter + trpix + tlpix

                    yval = yval + 1

                    if yval == img_data_mask.shape[0]-1:
                        break

                #Bottom pixel CR
                yval = yind-1
                while cleaned_img_data[yval,xind] > cutoff and yind >= 0:
                    counter = 0
                    xval = xind
                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval + 1
                        counter = counter + 1
                        if xval == img_data_mask.shape[1]-1:
                            break

                    brpix = counter

                    #left pixels
                    counter = 0
               xval = xind-1
                    while cleaned_img_data[yval,xval] > cutoff:

                        xval = xval - 1
                        counter = counter + 1
                        if xval < 1:
                            break

                    blpix = counter

                    botcounter = botcounter + brpix + blpix
                    CR_data_mask[yval,xind-blpix:xind+brpix+1] = -1

                    yval = yval - 1

                    if yval <= 0:
                        break

            if (topcounter+botcounter) >= CR_check:
                #CR_mask_idx = CR_data_mask < 0
                CRcounter = CRcounter + 1

            #cleaned_img_data[yind,xind] = prandom_array[yind,xind]
            clean_mask_idx = CR_data_mask < 0
            cleaned_img_data[clean_mask_idx] = prandom_array[clean_mask_idx]

            maxvalue = np.max(cleaned_img_data)
            #print maxvalue, xind, yind
            max_img_data = np.array(np.where(cleaned_img_data == maxvalue))
        xind = max_img_data[1,0]
        yind = max_img_data[0,0]

        #print np.max(CR_data_mask)

    new_image_fname = image.replace(".fits", ".CR.fits")
          if os.path.exists(new_image_fname):
              os.remove(new_image_fname)
          fits.writeto(new_image_fname,cleaned_img_data,header=header)
          print("\t  CR corrected image : %s" %(new_image_fname))
          print("\t  CR counted in image : %s " %(zeroCR+CRcounter))

    # Generate mask FITS image
    mask_fname = image.replace(".fits", ".mask.CR.fits")
    if os.path.exists(mask_fname):
        os.remove(mask_fname)
    fits.writeto(mask_fname,CR_data_mask,header=header)
    print("\t  Corrected Mask : %s" %(mask_fname))

    return(image.replace(data_dir,''),(zeroCR+CRcounter))


def run_CR_removal(nthreads):

    area = img_area

    # Get data cubes names
    glob_pattern = data_dir + 'image0*.fits'
    extrapattern = data_dir + 'image0*.*.fits'
    images =  set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern))

    # Create a python list as argument for mp pool
    imglist = []
    CRcount = []
    CRcorr = []

    for image in images:
    imglist.append((image,area))

    if nthreads != 12:
    n_cpus = int(nthreads)

    pool = mp.Pool(n_cpus)

    #results = pool.map(removeCR, imglist)
    results_count = pool.map(countCR, imglist)
    results_corr = pool.map(removeCRtails, imglist)

    arr = []
    CRcount.append(results_count)
    np.savetxt(data_dir + 'CRcount',CRcount[0],fmt='%s')
    print(np.array(CRcount)[0].shape)
    CRcorr.append(results_corr)
    np.savetxt(data_dir + 'CRcorr',CRcorr[0],fmt='%s')
    print(np.array(CRcorr)[0].shape)

    #imgnum = np.array(CRlist)[:,:,0]
    #images = np.array(CRlist)[:,:,1]
    #crnum = np.array(CRlist)[:,:,2]
    #imgnum_idx = np.argsort(imgnum)
    #print imgnum_idx
    #CRlist_sorted = np.array(CRlist)[np.argsort(np.array(CRlist)[:,:,0])]
    #np.savetxt(data_dir + 'CRlist_sorted',CRlist_sorted,fmt='%s')



    return()



def desmear(indata):

    image_cube, area, method, threshold = indata

    hist_output = calc_emgain_CR(image_cube,area)

    bias = hist_output[1]
    sigma = hist_output[2]

    # Read data from FITS image
    try:
    img_data,header = fits.getdata(image_cube,header=True)
    except IOError:
    raise IOError("Unable to open FITS image %s" %(image_cube))

    # Only 3D data cubes are accepted
    if np.ndim(img_data) != 3:
    raise ValueError("Routine only accepts 3D data cubes.")

    # Rejection method, threshold
    threshold = float(threshold)

    print("Method: ", method, "  Threshold: ", threshold, "Cube #: ", image_cube)

    # Image dimension
    zsize, ysize, xsize = img_data.shape

    if method == "mad":
    # Median and MAD values
    medianval = np.median(img_data, 0)
    madval = np.median(np.abs(medianval - img_data), 0)
    else:
      # Get median value for all the pixels between all the frames
      medianval = np.zeros([ysize, xsize], dtype = np.float32)
      for j in range(ysize):
      for i in range(xsize):
          if (i > 0 and i < xsize) and (j > 0 and j < ysize):
          medianval[j,i] = np.median(img_data[:,j-1:j+2,i-1:i+2])
          else:
          medianval[j,i] = np.median(img_data[:,j,i])
      # Calculate MAD value
      madval = np.zeros([ysize, xsize], dtype = np.float32)
      for j in range(ysize):
      for i in range(xsize):
          if (i > 0 and i < xsize) and (j > 0 and j < ysize):
          madval[j,i] = np.median(np.abs(medianval[j,i] - img_data[:,j-1:j+2,i-1:i+2]))
          else:
          madval[j,i] = np.median(np.abs(medianval[j,i] - img_data[:,j,i]))

    #Fill array with random numbers around the bias:
    prandom_array = np.random.normal(int(bias),int(sigma), size=(img_data.shape[1],img_data.shape[2]))

    pixel_length = []
    percent_flag_events = np.zeros(img_data.shape[0], dtype = np.float)
    for i in range(img_data.shape[0]):
    print("\tProcessing Frame: %d" %(i))
    idx = img_data[i,:,:] > medianval + threshold * madval
    #region = idx[1100:2000,1150:2100]
    region = idx[img_area[0]:img_area[1],img_area[2]:img_area[3]]
    percent_flag_events[i] = (np.count_nonzero(region)/(float(region.shape[0]) * region.shape[1]))*100
    print("\t  percent pixels flagged in region: %6.2f" %(percent_flag_events[i]))


    # Generate mask FITS image
    mask_data = np.zeros([img_data.shape[1], img_data.shape[2]], dtype = np.int16)
    mask_data[idx] = 1
    mask_fname = image_cube.replace(".fits", "." + str(i) +  ".mask.fits")
    if os.path.exists(mask_fname):
        os.remove(mask_fname)
    fits.writeto(mask_fname,mask_data,header=header)
    print("\t Generating mask : %s" %(mask_fname))

    #Generate masks for analysis

    new_img_data = np.copy(img_data[i,:,:])
    corr_mask_data = np.copy(mask_data)

        mask_data_ind = np.array(np.where(mask_data == 1))

    for j in range(0,mask_data_ind.shape[1]):

           xind = mask_data_ind[1,j]
           yind = mask_data_ind[0,j]

           if xind > 1 and xind < mask_data.shape[1] - 1:
               if mask_data[yind,xind] == 1 and mask_data[yind,xind-1] == 0:
                   #Left pixel smear V2 only
               #xval = xind
           #counter = 0

           #while mask_data[yind,xval-1] == 1:

                   #    xval = xval - 1
           #    counter = counter + 1
           #    if xval == mask_data.shape[1]-1:
           #        break
               #lpix = counter

               #Right pixel smear V3 only
           xval = xind
           counter = 0

           while mask_data[yind,xval] == 1:# and new_img_data[yind,(xval+1)] < bias - (5*sigma):
                   xval = xval + 1
               counter = counter + 1
               if xval == mask_data.shape[1]-1:
                   break
               rpix = counter

           #Making new mask with LHS pixel smear check
           corr_mask_data[yind,xind+1:xind+rpix+1] = -1
           fixed_smear = np.array(new_img_data[yind,xind:xind+rpix+1])
           fixed_smear_sum = np.sum(fixed_smear) - (rpix)*int(bias)
              #print 'image %i lpix %i orig %i summed %i' %(i, int(lpix), int(new_img_data[yind,xind]), int(fixed_smear_sum))
               new_img_data[yind,xind] = int(fixed_smear_sum)
           pixel_length.append(rpix)


           newmask_idx = corr_mask_data < 0
    new_img_data[newmask_idx] = prandom_array[newmask_idx]
    #new_img_data[yind,xind] = fixed_smear_sum

    #if fixed_smear_sum < 20000:
    #    new_img_data[yind,xind] = fixed_smear_sum
        #else:
        #    new_img_data[yind,xind] = np.random.normal(int(bias),int(sigma), size=1)
        #    #np.random.randint(r1,r2)

        percent_flag_pixels = np.zeros(img_data.shape[0], dtype = np.float)
        for k in range(img_data.shape[0]):
       region = newmask_idx[img_area[0]:img_area[1],img_area[2]:img_area[3]]
       percent_flag_pixels[k] = (np.count_nonzero(region)/(float(region.shape[0]) * region.shape[1]))*100

    new_image_fname = image_cube.replace(".fits", "." + str(i) + ".corr.fits")
    if os.path.exists(new_image_fname):
        os.remove(new_image_fname)
    fits.writeto(new_image_fname,new_img_data,header=header)
    print("\t  DESMEARED image : %s" %(new_image_fname))

    corr_mask = image_cube.replace(".fits", "." + str(i) + ".corrmask.fits")
    if os.path.exists(corr_mask):
        os.remove(corr_mask)
    fits.writeto(corr_mask,corr_mask_data,header=header)
    print("\t  CORRECTED image mask : %s" %(corr_mask))

        print("\t  Pixel smeared length : %s" %(np.sum(np.array(pixel_length))/len(pixel_length)))


    np.savetxt(image_cube.replace('.fits','.smeared'), percent_flag_events,fmt='%.2f')
    np.savetxt(image_cube.replace('.fits','.pixlen'), pixel_length)
    #np.savetxt(image_cube.replace('.fits','.darkevents'), percent_darkpix_events,fmt='%.2f')

    return True


def run_desmear_process(method, thresh, nthreads):

    thresh = np.array([float(thresh)])
    print(thresh.shape)
    np.savetxt(data_dir + 'desmear_thresh',thresh,fmt='%.2f')
    #hist_output = calc_emgain(data_dir + 'image_CR_cube_0.fits',shielded_area)
    #np.savetxt(data_dir + 'hist_output',hist_output,fmt='%d')

    #bias = int(hist_output[1])
    #sigma = int(hist_output[2])

    # Get data cubes names
    glob_pattern = data_dir + 'image_CR_*.fits'
    extrapattern = data_dir + 'image_CR_*.*.fits'
    images =  set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern))
#    images = [data_dir+'image_CR_cube_3.fits']

    # Create a python list as argument for mp pool
    imglist = []
    for image in images:
    imglist.append((image, img_area, method, thresh))

    if nthreads != 12:
    n_cpus = int(nthreads)

    pool = mp.Pool(n_cpus)
    results = pool.map(desmear, imglist)

    return (thresh)



def threshold(image,output,area):

    output = np.array(output)
    cutoff = int(output[1]) + int(output[2])*5.5
    #print cutoff
    img_data,header = fits.getdata(image,header=True)
    img_data = img_data[:,area[0]:area[1],area[2]:area[3]]

    idx = []
    idx = img_data >= cutoff

    # Generate mask FITS image
    thresh_data = np.zeros([img_data.shape[0],img_data.shape[1],img_data.shape[2]], dtype = np.int16)
    thresh_data[idx] = 1
    #print thresh_data.shape

    return(thresh_data)

def dark_rate(thresh):

    glob_pattern = data_dir + 'image_corr_cube*.fits'
    extrapattern = data_dir + 'image_*_cube_*.*.fits'
    images =  np.sort(glob.glob(glob_pattern))
    #images =  np.sort(set(glob.glob(glob_pattern))) - np.sort(set(glob.glob(extrapattern)))
    #print images
    mask_glob_pattern = data_dir + 'image_CRmask_cube*.fits'
    mask_images =  np.sort(glob.glob(mask_glob_pattern))

    #print mask_images

    #images = [images[0],images[1]]

    tot_img_output = []
    img_output = []
    stg_output = []
    pre_output = []
    ovr_output = []
    expt = []

    for image in images:

        print('Current image: %s' % image.replace(data_dir,''))
        data, header = fits.getdata(image,header=True)
        expt.append(int(header['EXPTIME']))

    img_output.append(calc_emgain(image,img_area))
    stg_output.append(calc_emgain(image,stg_area))
        pre_output.append(calc_emgain(image,pre_area))
        ovr_output.append(calc_emgain(image,ovr_area))

    np.savetxt(data_dir + 'img_output',img_output,fmt='%.4f')
    np.savetxt(data_dir + 'stg_output',stg_output,fmt='%.4f')
    np.savetxt(data_dir + 'pre_output',pre_output,fmt='%.4f')
    np.savetxt(data_dir + 'ovr_output',ovr_output,fmt='%.4f')
    #np.savetxt(data_dir + 'ovr_output',ovr_output,fmt='%.4f')

    #tot_noise = []
    #tot_img_noise = []
    img_noise = []
    stg_noise = []
    pre_noise = []
    ovr_noise = []

    #tot_img_noise_cube = []
    img_noise_cube = []
    stg_noise_cube = []
    pre_noise_cube = []
    ovr_noise_cube = []

    #tot_img_noise_lost = []
    img_noise_lost = []
    stg_noise_lost = []
    pre_noise_lost = []
    ovr_noise_lost = []

    #tot_img_noise_corr = []
    img_noise_corr = []
    stg_noise_corr = []
    pre_noise_corr = []
    ovr_noise_corr = []

    #tot_img_rate_corr = 0
    img_rate_corr = 0
    stg_rate_corr = 0
    pre_rate_corr = 0
    ovr_rate_corr = 0

    line_noise = []
    col_noise = []

    expt = np.array(expt)
    expt_idx = expt.argsort()
    expt_sort = expt[expt_idx]

    i = 0

    for image in images:

    mask_img = mask_images[i]
    mask_data = fits.getdata(mask_img)
    idx = mask_data == 0

    #tot_arr_idx = mask_data[:,img_array[0]:img_array[1],img_array[2]:img_array[3]] == 0
    #tot_img_idx = mask_data[:,tot_shielded_area[0]:tot_shielded_area[1],tot_shielded_area[2]:tot_shielded_area[3]] == 0
    img_idx = mask_data[:,img_area[0]:img_area[1],img_area[2]:img_area[3]] == 0
    stg_idx = mask_data[:,stg_area[0]:stg_area[1],stg_area[2]:stg_area[3]] == 0
    pre_idx = mask_data[:,pre_area[0]:pre_area[1],pre_area[2]:pre_area[3]] == 0
        ovr_idx = mask_data[:,ovr_area[0]:ovr_area[1],ovr_area[2]:ovr_area[3]] == 0

        #tot_noise = threshold(image,tot_img_output[i],tot_shielded_area)
    #tot_noise = tot_noise

    #tot_img_noise.append((np.mean(threshold(image,tot_img_output[i],tot_shielded_area)[tot_img_idx])))
    img_noise.append((np.mean(threshold(image,img_output[i],img_area)[img_idx])))
    stg_noise.append((np.mean(threshold(image,stg_output[i],stg_area)[stg_idx])))
        pre_noise.append((np.mean(threshold(image,pre_output[i],pre_area)[pre_idx])))
        ovr_noise.append((np.mean(threshold(image,ovr_output[i],ovr_area)[ovr_idx])))

        #line_noise.append(ma.mean(ma.mean(ma.array(threshold(image,tot_img_output[i],img_array),mask=list(~np.array(tot_arr_idx))),axis=1),axis=0))
    #col_noise.append(ma.mean(ma.mean(ma.array(threshold(image,tot_img_output[i],img_array),mask=list(~np.array(tot_arr_idx))),axis=2),axis=0))

        #tot_img_noise_cube.append(ma.mean(ma.mean(ma.array(threshold(image,tot_img_output[i],tot_shielded_area),mask=list(~np.array(tot_img_idx))),axis=2),axis=1))
    img_noise_cube.append(ma.mean(ma.mean(ma.array(threshold(image,img_output[i],img_area),mask=list(~np.array(img_idx))),axis=2),axis=1))
    stg_noise_cube.append(ma.mean(ma.mean(ma.array(threshold(image,stg_output[i],stg_area),mask=list(~np.array(stg_idx))),axis=2),axis=1))
    pre_noise_cube.append(ma.mean(ma.mean(ma.array(threshold(image,pre_output[i],pre_area),mask=list(~np.array(pre_idx))),axis=2),axis=1))
        ovr_noise_cube.append(ma.mean(ma.mean(ma.array(threshold(image,ovr_output[i],ovr_area),mask=list(~np.array(ovr_idx))),axis=2),axis=1))

        #tot_img_noise_lost.append(tot_img_noise[i]*tot_img_output[i][3])
        img_noise_lost.append(img_noise[i]*img_output[i][3])
        stg_noise_lost.append(stg_noise[i]*stg_output[i][3])
        pre_noise_lost.append(pre_noise[i]*pre_output[i][3])
        ovr_noise_lost.append(ovr_noise[i]*ovr_output[i][3])
        #tot_img_noise_corr.append((np.mean(threshold(image,tot_img_output[i],tot_shielded_area)[tot_img_idx]))+tot_img_noise_lost[i])
    img_noise_corr.append((np.mean(threshold(image,img_output[i],img_area)[img_idx]))+img_noise_lost[i])
    stg_noise_corr.append((np.mean(threshold(image,stg_output[i],stg_area)[stg_idx]))+stg_noise_lost[i])
        pre_noise_corr.append((np.mean(threshold(image,pre_output[i],pre_area)[pre_idx]))+pre_noise_lost[i])
    ovr_noise_corr.append((np.mean(threshold(image,ovr_output[i],ovr_area)[ovr_idx]))+ovr_noise_lost[i])

        #tot_img_rate_corr = tot_img_rate_corr + tot_img_output[i][3]
        img_rate_corr = img_rate_corr + img_output[i][3]
        stg_rate_corr = stg_rate_corr + stg_output[i][3]
        pre_rate_corr = pre_rate_corr + pre_output[i][3]
        ovr_rate_corr = ovr_rate_corr + ovr_output[i][3]

    i = i + 1

    #print np.array(tot_img_noise_cube).shape
    print(np.array(img_noise_cube).shape)
    print(np.array(stg_noise_cube).shape)
    print(np.array(pre_noise_cube).shape)
    print(np.array(ovr_noise_cube).shape)

    #print np.array(line_noise).shape
    #print np.array(col_noise).shape

    xfit = list(range(0,np.max(expt)*3,1))

    #print expt_sort
    cic_idx = np.where(expt_sort == 0)
    cic_val = np.mean(np.array(pre_noise)[cic_idx])

    #tot_img_noise = np.array(tot_img_noise)[expt_idx]
    img_noise = np.array(img_noise)[expt_idx]
    stg_noise = np.array(stg_noise)[expt_idx]
    pre_noise = np.array(pre_noise)[expt_idx]
    ovr_noise = np.array(ovr_noise)[expt_idx]

    #Image area noise and rate:
    img_noise_CICremoved = np.array(img_noise - cic_val)
    img_CICfit = cic_val
    img_CIClinefit = 0.00000001*np.array(xfit) + img_CICfit
    m,c = curve_fit(linefit, expt_sort,img_noise_CICremoved)[0]
    img_trendline = m*np.array(xfit)
    img_fit = img_CIClinefit + img_trendline
    img_rate = m*3600

    #Storage area noise and rate:
    stg_noise_CICremoved = np.array(stg_noise - cic_val)
    stg_CICfit = cic_val
    stg_CIClinefit = 0.00000001*np.array(xfit) + stg_CICfit
    m,c = curve_fit(linefit, expt_sort,stg_noise_CICremoved)[0]
    stg_trendline = m*np.array(xfit)
    stg_fit = stg_CIClinefit + stg_trendline
    stg_rate = m*3600

    #Prescan area noise and rate:
    pre_noise_CICremoved = np.array(pre_noise - cic_val)
    pre_CICfit = cic_val
    pre_CIClinefit = 0.00000001*np.array(xfit) + pre_CICfit
    m,c = curve_fit(linefit, expt_sort,pre_noise_CICremoved)[0]
    pre_trendline = m*np.array(xfit)
    pre_fit = pre_CIClinefit + pre_trendline
    pre_rate = m*3600

    #Ovrscan area noise and rate:
    ovr_noise_CICremoved = np.array(ovr_noise - cic_val)
    ovr_CICfit = cic_val
    ovr_CIClinefit = 0.00000001*np.array(xfit) + ovr_CICfit
    m,c = curve_fit(linefit, expt_sort,ovr_noise_CICremoved)[0]
    ovr_trendline = m*np.array(xfit)
    ovr_fit = ovr_CIClinefit + ovr_trendline
    ovr_rate = m*3600

    #tot_img_rate_corr = (tot_img_rate)/(1-(tot_img_rate_corr/len(img_noise)))
    img_rate_corr = (img_rate)/(1-(img_rate_corr/len(img_noise)))
    stg_rate_corr = (stg_rate)/(1-(stg_rate_corr/len(img_noise)))
    pre_rate_corr = (pre_rate)/(1-(pre_rate_corr/len(img_noise)))
    ovr_rate_corr = (ovr_rate)/(1-(ovr_rate_corr/len(img_noise)))

    #tot_img_CIC_corr = (tot_img_CIClinefit[0])/(1-(tot_img_rate_corr/len(img_noise)))
    img_CIC_corr = (img_CIClinefit[0])/(1-(img_rate_corr/len(img_noise)))
    stg_CIC_corr = (stg_CIClinefit[0])/(1-(stg_rate_corr/len(img_noise)))
    pre_CIC_corr = (pre_CIClinefit[0])/(1-(pre_rate_corr/len(img_noise)))
    ovr_CIC_corr = (ovr_CIClinefit[0])/(1-(ovr_rate_corr/len(img_noise)))

    np.savetxt(data_dir + 'dark_rate',[img_rate,stg_rate,pre_rate,ovr_rate],delimiter=',',header='img_rate,stg_rate,pre_rate,ovr_rate',fmt='%.8f')
    np.savetxt(data_dir + 'cic_noise',[img_noise[0],stg_noise[0],pre_noise[0],ovr_noise[0]],delimiter=',',header='img_CIC,stg_CIC,pre_CIC,ovr_CIC',fmt='%.8f')

    np.savetxt(data_dir + 'dark_rate_corr',[img_rate_corr,stg_rate_corr,pre_rate_corr,ovr_rate_corr],delimiter=',',header='img_rate,stg_rate,pre_rate,ovr_rate',fmt='%.8f')
    np.savetxt(data_dir + 'cic_noise_corr',[img_noise_corr[0],stg_noise_corr[0],pre_noise_corr[0],ovr_noise_corr[0]],delimiter=',',header='img_CIC,stg_CIC,pre_CIC,ovr_CIC',fmt='%.8f')
    #np.savetxt(data_dir + 'tot_noise',[tot_img_noise],fmt='%.8f')
    np.savetxt(data_dir + 'img_noise',[img_noise],fmt='%.8f')
    np.savetxt(data_dir + 'stg_noise',[stg_noise],fmt='%.8f')
    np.savetxt(data_dir + 'pre_noise',[pre_noise],fmt='%.8f')
    np.savetxt(data_dir + 'ovr_noise',[ovr_noise],fmt='%.8f')

    # Plot frame number versus number of flagged pixels
    if plot_flag:
        plt.close("all")
        plt.clf()
        #fig = plt.figure(figsize=(18,15))
        #plt.rc('text', usetex=True)
        #gca().set_position((.1, .3, .8, .65))
        plt.rcParams['mathtext.fontset']='stix'
        plt.rcParams['font.family']='STIXGeneral'
        plt.rcParams['font.size']='20'
        #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times'],'size': 50})
        plt.xlabel("Exposure Time [s}",fontsize=20)
        plt.ylabel("Noise [e/pix/fr]",fontsize=20)
        grid(b=True, which='major', color='0.75', linestyle='--')
        grid(b=True, which='minor', color='0.75', linestyle='--')

#colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        plt.loglog(np.array(expt_sort), np.array(img_noise),"m.",markersize=10,label='Dark noise')
        #plt.plot(np.array(expt_sort[2:]), np.array(img_noise[2:]),color="navy",linestyle='--',markersize=30,label='Dark noise - CIC')
        plt.plot(np.array(xfit), np.array(img_fit),"m-",markersize=10)
        plt.plot(np.array(xfit), np.array(img_CIClinefit),"--g",markersize=10,label='CIC')
        plt.loglog(np.array(xfit), np.array(img_trendline),"--k",markersize=10,label='Dark rate')

        #print xfit
        #print img_trendline

        #prescan area
        plt.plot(np.array(xfit), np.array(pre_CIClinefit),color="mediumpurple",linestyle="--",markersize=50,label='Prescan CIC')
        #overscan area
        #plt.plot(np.array(xfit), np.array(ovr_CIClinefit),color="violet",linestyle="--",markersize=50,label='Postscan CIC')

        plt.legend(loc=2,prop={'size':15})
        #plt.xlim(0,10**3.2)
        #plt.ylim(10**-3.0,10**-1.0)
        #figtext(.15, .15,'Image area dark rate = %0.4f e/pix/hr.\nImage area CIC = %0.4f e/pix/fr.\nStorage area dark rate = %0.4f e/pix/hr.\nStorage area CIC = %0.4f e/pix/fr.\nPrescan area dark rate = %0.4f e/pix/hr.\nPrescan area CIC = %0.4f e/pix/fr.\nOverscan area dark rate = %0.4f e/pix/hr.\nOverscan area CIC = %0.4f e/pix/fr.' % (img_rate_corr, img_CIC_corr, stg_rate_corr, stg_CIC_corr, pre_rate_corr, pre_CIC_corr, ovr_rate_corr, ovr_CIC_corr),fontsize=22)
        #figtext(.60, .15,'Image area dark rate = %0.4f e/pix/hr.\nImage area CIC = %0.4f e/pix/fr.\nStorage area dark rate = %0.4f e/pix/hr.\nStorage area CIC = %0.4f e/pix/fr.\nPrescan area dark rate = %0.4f e/pix/hr.\nPrescan area CIC = %0.4f e/pix/fr.\nOverscan area dark rate = %0.4f e/pix/hr.\nOverscan area CIC = %0.4f e/pix/fr.' % (img_rate, img_CIClinefit[0], stg_rate, stg_CIClinefit[0], pre_rate, pre_CIClinefit[0], ovr_rate, ovr_CIClinefit[0]),fontsize=22)
        #fig_name = data_dir.replace('/home/gillian/data/darks/','')
        fig_name = data_dir.replace('/home/cheng/data/NUVU/','')
        fig_name = fig_name[:-1].replace('/','_')
        fig_dir = data_dir + 'figures/'
        if not os.path.exists(fig_dir):# create data directory if needed
        os.makedirs(fig_dir)
    plt.savefig(fig_dir + fig_name + '.' + str(thresh) +'sigcorr.png',bbox_inches='tight')


    print('image statistics...')
    print(img_rate)
    print(stg_rate)

    return(img_noise_cube,stg_noise_cube,pre_noise_cube,ovr_noise_cube,expt,expt_idx,expt_sort,cic_val)


def dark_rate_CR():

    glob_pattern = data_dir + 'image_CR_cube*.fits'
    extrapattern0 = data_dir + 'image_*_cube_*.*.fits'
    extrapattern1 = data_dir + 'image_*_cube_*.*.*.fits'
    #images =  np.sort(glob.glob(glob_pattern))
    images =  np.sort(list(set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern0)) - set(glob.glob(extrapattern1))))
    mask_glob_pattern = data_dir + 'image_CRmask_cube*.fits'
    mask_images =  np.sort(glob.glob(mask_glob_pattern))

    #print mask_images

    tot_img_output = []
    img_output = []
    stg_output = []
    pre_output = []
    expt = []

    for image in images:

        print('Current image: %s' % image.replace(data_dir,''))
        data, header = fits.getdata(image,header=True)
        expt.append(int(header['EXPTIME']))

    tot_img_output.append(calc_emgain_CR(image,tot_shielded_area))
    img_output.append(calc_emgain_CR(image,shielded_area))
    stg_output.append(calc_emgain_CR(image,exposed_area))
        pre_output.append(calc_emgain_CR(image,prescan_area))

    np.savetxt(data_dir + 'tot_img_output_CR',tot_img_output,fmt='%.4f')
    np.savetxt(data_dir + 'img_output_CR',img_output,fmt='%.4f')
    np.savetxt(data_dir + 'stg_output_CR',stg_output,fmt='%.4f')
    np.savetxt(data_dir + 'pre_output_CR',pre_output,fmt='%.4f')
    #np.savetxt(data_dir + 'ovr_output',ovr_output,fmt='%.4f')

    tot_noise = []
    tot_img_noise = []
    img_noise = []
    stg_noise = []
    pre_noise = []

    tot_img_noise_lost = []
    img_noise_lost = []
    stg_noise_lost = []
    pre_noise_lost = []

    tot_img_noise_corr = []
    img_noise_corr = []
    stg_noise_corr = []
    pre_noise_corr = []

    tot_img_rate_corr = 0
    img_rate_corr = 0
    stg_rate_corr = 0
    pre_rate_corr = 0


    i = 0

    for image in images:

    mask_img = mask_images[i]
    mask_data = pyfits.getdata(mask_img)
    idx = mask_data == 0

    tot_img_idx = mask_data[:,tot_shielded_area[0]:tot_shielded_area[1],tot_shielded_area[2]:tot_shielded_area[3]] == 0
    img_idx = mask_data[:,shielded_area[0]:shielded_area[1],shielded_area[2]:shielded_area[3]] == 0
    stg_idx = mask_data[:,exposed_area[0]:exposed_area[1],exposed_area[2]:exposed_area[3]] == 0
    pre_idx = mask_data[:,prescan_area[0]:prescan_area[1],prescan_area[2]:prescan_area[3]] == 0

        tot_noise = threshold(image,tot_img_output[i],tot_shielded_area)
    tot_noise = tot_noise

    tot_img_noise.append((np.mean(threshold(image,tot_img_output[i],tot_shielded_area)[tot_img_idx])))
    img_noise.append((np.mean(threshold(image,img_output[i],shielded_area)[img_idx])))
    stg_noise.append((np.mean(threshold(image,stg_output[i],exposed_area)[stg_idx])))
        pre_noise.append((np.mean(threshold(image,pre_output[i],prescan_area)[pre_idx])))

        tot_img_noise_lost.append(tot_img_noise[i]*tot_img_output[i][3])
        img_noise_lost.append(img_noise[i]*img_output[i][3])
        stg_noise_lost.append(stg_noise[i]*stg_output[i][3])
        pre_noise_lost.append(pre_noise[i]*pre_output[i][3])

        tot_img_noise_corr.append((np.mean(threshold(image,tot_img_output[i],tot_shielded_area)[tot_img_idx]))+tot_img_noise_lost[i])
    img_noise_corr.append((np.mean(threshold(image,img_output[i],shielded_area)[img_idx]))+img_noise_lost[i])
    stg_noise_corr.append((np.mean(threshold(image,stg_output[i],exposed_area)[stg_idx]))+stg_noise_lost[i])
        pre_noise_corr.append((np.mean(threshold(image,pre_output[i],prescan_area)[pre_idx]))+pre_noise_lost[i])

        tot_img_rate_corr = tot_img_rate_corr + tot_img_output[i][3]
        img_rate_corr = img_rate_corr + img_output[i][3]
        stg_rate_corr = stg_rate_corr + stg_output[i][3]
        pre_rate_corr = pre_rate_corr + pre_output[i][3]


    i = i + 1


    xfit = list(range(0,1000,1))

    expt = np.array(expt)
    expt_idx = expt.argsort()

    xfit = list(range(0,np.max(expt)*3,1))

    expt_sort = expt[expt_idx]
    #print expt_sort
    cic_idx = np.where(expt_sort == 0)
    cic_val = np.mean(np.array(pre_noise)[cic_idx])

    tot_img_noise = np.array(tot_img_noise)[expt_idx]
    img_noise = np.array(img_noise)[expt_idx]
    stg_noise = np.array(stg_noise)[expt_idx]
    pre_noise = np.array(pre_noise)[expt_idx]

    #Total image area noise and rate:
    tot_img_noise_CICremoved = np.array(tot_img_noise - cic_val)
    tot_img_CICfit = cic_val
    tot_img_CIClinefit = 0.00000001*np.array(xfit) + tot_img_CICfit
    m,c = curve_fit(linefit, expt_sort,tot_img_noise_CICremoved)[0]
    tot_img_trendline = m*np.array(xfit)
    tot_img_fit = tot_img_CIClinefit + tot_img_trendline
    tot_img_rate = m*3600

    #Image area noise and rate:
    img_noise_CICremoved = np.array(img_noise - cic_val)
    img_CICfit = cic_val
    img_CIClinefit = 0.00000001*np.array(xfit) + img_CICfit
    m,c = curve_fit(linefit, expt_sort,img_noise_CICremoved)[0]
    img_trendline = m*np.array(xfit)
    img_fit = img_CIClinefit + img_trendline
    img_rate = m*3600

    #Storage area noise and rate:
    stg_noise_CICremoved = np.array(stg_noise - cic_val)
    stg_CICfit = cic_val
    stg_CIClinefit = 0.00000001*np.array(xfit) + stg_CICfit
    m,c = curve_fit(linefit, expt_sort,stg_noise_CICremoved)[0]
    stg_trendline = m*np.array(xfit)
    stg_fit = stg_CIClinefit + stg_trendline
    stg_rate = m*3600

    #Prescan area noise and rate:
    pre_noise_CICremoved = np.array(pre_noise - cic_val)
    pre_CICfit = cic_val
    pre_CIClinefit = 0.00000001*np.array(xfit) + pre_CICfit
    m,c = curve_fit(linefit, expt_sort,pre_noise_CICremoved)[0]
    pre_trendline = m*np.array(xfit)
    pre_fit = pre_CIClinefit + pre_trendline
    pre_rate = m*3600


    tot_img_rate_corr = (tot_img_rate)/(1-(tot_img_rate_corr/len(img_noise)))
    img_rate_corr = (img_rate)/(1-(img_rate_corr/len(img_noise)))
    stg_rate_corr = (stg_rate)/(1-(stg_rate_corr/len(img_noise)))
    pre_rate_corr = (pre_rate)/(1-(pre_rate_corr/len(img_noise)))

    tot_img_CIC_corr = (tot_img_CIClinefit[0])/(1-(tot_img_rate_corr/len(img_noise)))
    img_CIC_corr = (img_CIClinefit[0])/(1-(img_rate_corr/len(img_noise)))
    stg_CIC_corr = (stg_CIClinefit[0])/(1-(stg_rate_corr/len(img_noise)))
    pre_CIC_corr = (pre_CIClinefit[0])/(1-(pre_rate_corr/len(img_noise)))

    np.savetxt(data_dir + 'dark_rate_CR',[tot_img_rate,img_rate,stg_rate,pre_rate],delimiter=',',header='tot_img,img_rate,stg_rate,pre_rate',fmt='%.8f')
    np.savetxt(data_dir + 'cic_noise_CR',[tot_img_noise[0],img_noise[0],stg_noise[0],pre_noise[0]],delimiter=',',header='tot_img_CIC,img_CIC,stg_CIC,pre_CIC',fmt='%.8f')

    np.savetxt(data_dir + 'dark_rate_corr_CR',[tot_img_rate_corr,img_rate_corr,stg_rate_corr,pre_rate_corr],delimiter=',',header='tot_img,img_rate,stg_rate,pre_rate',fmt='%.8f')
    np.savetxt(data_dir + 'cic_noise_corr_CR',[tot_img_noise_corr[0],img_noise_corr[0],stg_noise_corr[0],pre_noise_corr[0]],delimiter=',',header='tot_img_CIC,img_CIC,stg_CIC,pre_CIC',fmt='%.8f')
    np.savetxt(data_dir + 'tot_noise_CR',[tot_img_noise],fmt='%.8f')
    np.savetxt(data_dir + 'img_noise_CR',[img_noise],fmt='%.8f')
    np.savetxt(data_dir + 'stg_noise_CR',[stg_noise],fmt='%.8f')
    np.savetxt(data_dir + 'pre_noise_CR',[pre_noise],fmt='%.8f')

    # Plot frame number versus number of flagged pixels
    if plot_flag:
        plt.close("all")
        plt.clf()
        fig = plt.figure(figsize=(18,15))
        #plt.rc('text', usetex=True)
        #gca().set_position((.1, .3, .8, .65))
        plt.rcParams['mathtext.fontset']='stix'
        plt.rcParams['font.family']='STIXGeneral'
        plt.rcParams['font.size']='40'
        #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times'],'size': 50})
        plt.xlabel("Exposure Time [s}",fontsize=35)
        plt.ylabel("Noise [e/pix/fr]",fontsize=35)
        grid(b=True, which='major', color='0.75', linestyle='--')
        grid(b=True, which='minor', color='0.75', linestyle='--')

#colors = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])

        #image area
        plt.loglog(np.array(expt_sort), np.array(img_noise),"m.",markersize=30,label='Dark noise')
        plt.plot(np.array(xfit), np.array(img_fit),"m-",markersize=50)
        plt.plot(np.array(xfit), np.array(img_CIClinefit),"--g",markersize=50,label='CIC')
        plt.plot(np.array(xfit), np.array(img_trendline),"--k",markersize=50,label='Dark rate')

        print(xfit)
        print(img_trendline)

        #prescan area
        plt.plot(np.array(xfit), np.array(pre_CIClinefit),color="mediumpurple",linestyle="--",markersize=50,label='Prescan CIC')
        #overscan area
        #plt.plot(np.array(xfit), np.array(ovr_CIClinefit),color="violet",linestyle="--",markersize=50,label='Postscan CIC')

        plt.legend(loc=2,prop={'size':30})
        #plt.xlim(0,10**3.2)
        #plt.ylim(10**-3.0,10**-1.5)
        #figtext(.15, .15,'Image area dark rate = %0.4f e/pix/hr.\nImage area CIC = %0.4f e/pix/fr.\nStorage area dark rate = %0.4f e/pix/hr.\nStorage area CIC = %0.4f e/pix/fr.\nPrescan area dark rate = %0.4f e/pix/hr.\nPrescan area CIC = %0.4f e/pix/fr.\nOverscan area dark rate = %0.4f e/pix/hr.\nOverscan area CIC = %0.4f e/pix/fr.' % (img_rate_corr, img_CIC_corr, stg_rate_corr, stg_CIC_corr, pre_rate_corr, pre_CIC_corr, ovr_rate_corr, ovr_CIC_corr),fontsize=22)
        #figtext(.60, .15,'Image area dark rate = %0.4f e/pix/hr.\nImage area CIC = %0.4f e/pix/fr.\nStorage area dark rate = %0.4f e/pix/hr.\nStorage area CIC = %0.4f e/pix/fr.\nPrescan area dark rate = %0.4f e/pix/hr.\nPrescan area CIC = %0.4f e/pix/fr.\nOverscan area dark rate = %0.4f e/pix/hr.\nOverscan area CIC = %0.4f e/pix/fr.' % (img_rate, img_CIClinefit[0], stg_rate, stg_CIClinefit[0], pre_rate, pre_CIClinefit[0], ovr_rate, ovr_CIClinefit[0]),fontsize=22)
        #fig_name = data_dir.replace('/home/gillian/data/darks/','')
        fig_name = data_dir.replace('/mnt/c/Users/gikyne/data/','')
        fig_name = fig_name[:-1].replace('/','_')
        fig_dir = data_dir + 'figures/'
        if not os.path.exists(fig_dir):# create data directory if needed
            os.makedirs(fig_dir)
        plt.savefig(fig_dir + fig_name + '.CR.png',bbox_inches='tight')
        plt.close()


    print('image statistics...')
    print(img_rate)
    print(stg_rate)

    return()

def cic_noise(indata):

    tot_img_noise_cube,img_noise_cube,stg_noise_cube,pre_noise_cube,line_noise,col_noise,expt,expt_idx,expt_sort,cic_val = indata

    y,x = np.array(tot_img_noise_cube).shape
    ximages = np.array(list(range(0,int(y*x))))

    y,x = np.array(line_noise).shape
    cols = np.arange(0,int(x))
    cols_xticks = np.arange(1,int(x),100)

    y,x = np.array(col_noise).shape
    rows = np.arange(0,int(x))

    tot_img_noise_sort = []
    img_noise_sort = []
    stg_noise_sort = []
    pre_noise_sort = []
    line_noise_sort = []
    col_noise_sort = []

    tot_img_noise_cube = list(ma.getdata(tot_img_noise_cube))
    img_noise_cube = list(ma.getdata(img_noise_cube))
    stg_noise_cube = list(ma.getdata(stg_noise_cube))
    pre_noise_cube = list(ma.getdata(pre_noise_cube))
    line_noise_sort = list(ma.getdata(line_noise))
    col_noise_sort = list(ma.getdata(col_noise))

    for i in range(0,y):
        tot_img_noise_sort.append(tot_img_noise_cube[i])
        img_noise_sort.append(img_noise_cube[i])
        stg_noise_sort.append(stg_noise_cube[i])
        pre_noise_sort.append(pre_noise_cube[i])
        line_noise_sort.append(line_noise[i])
        col_noise_sort.append(col_noise[i])

    tot_img_noise_sort = np.array(tot_img_noise_sort)[expt_idx]
    img_noise_sort = np.array(img_noise_sort)[expt_idx]
    stg_noise_sort = np.array(stg_noise_sort)[expt_idx]
    pre_noise_sort = np.array(pre_noise_sort)[expt_idx]
    line_noise_sort = np.array(line_noise_sort)[expt_idx]
    col_noise_sort = np.array(col_noise_sort)[expt_idx]

    # Plot frame number versus number of flagged pixels
    plt.close("all")
    plt.clf()
    fig = plt.figure(figsize=(30,15))
    #plt.rc('text', usetex=True)
    #gca().set_position((.1, .3, .8, .65))
    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']='40'
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times'],'size': 50})
    plt.xlabel("Frame #",fontsize=35)
    plt.ylabel("Noise [e/pix/fr]",fontsize=35)
    grid(b=True, which='major', color='0.75', linestyle='--')
    grid(b=True, which='minor', color='0.75', linestyle='--')
    #plt.ylim([0.004,0.016])

    #plt.plot(cols,line_noise[0],"k-",markersize=30,label='Serial register noise')
    plt.plot(ximages,np.reshape(np.array(tot_img_noise_sort),len(ximages)),color='darkgreen',marker=".",linestyle='None',markersize=30,label='Total image noise')
    plt.plot(ximages,np.reshape(np.array(img_noise_sort),len(ximages)),color='k',marker=".",linestyle='None',markersize=30,label='Image noise')
    plt.plot(ximages,np.reshape(np.array(stg_noise_sort),len(ximages)),color='m',marker=".",linestyle='None',markersize=30,label='Storage noise')
    plt.plot(ximages,np.reshape(np.array(pre_noise_sort),len(ximages)),color='navy',marker=".",linestyle='None',markersize=30,label='Prescan noise')
    plt.plot(ximages,(np.reshape(np.array(tot_img_noise_sort),len(ximages))-cic_val),color='black',linestyle='--',markersize=30,label='Prescan noise')
    plt.legend(loc="upper left",fontsize=40)

    fig_name = data_dir.replace('/mnt/c/Users/gikyne/Documents/data/radiation_tests/','')
    fig_name = fig_name[:-1].replace('/','_')
    fig_dir = data_dir + 'figures/'
    if not os.path.exists(fig_dir):# create data directory if needed
       os.makedirs(fig_dir)
    plt.savefig(fig_dir + fig_name + '.total-noise.png',bbox_inches='tight')
    #plt.show()


    # Plot frame number versus number of flagged pixels
    plt.close("all")
    plt.clf()
    fig = plt.figure(figsize=(30,15))
    #plt.rc('text', usetex=True)
    #gca().set_position((.1, .3, .8, .65))
    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']='40'
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times'],'size': 50})
    plt.xlabel("Column #",fontsize=35)
    plt.ylabel("Noise [e/pix/fr]",fontsize=35)
    grid(b=True, which='major', color='0.75', linestyle='--')
    grid(b=True, which='minor', color='0.75', linestyle='--')
    #plt.ylim([0.004,0.016])

    #plt.plot(cols,line_noise[0],"k-",markersize=30,label='Serial register noise')
    plt.plot(cols,line_noise_sort[0],color='navy',marker=".",linestyle='None',markersize=30,label='Column noise - 0 second exposure')
    plt.plot(cols,line_noise_sort[3],color='darkgreen',marker=".",linestyle='None',markersize=30,label='Column noise - 10 second exposure')
    plt.plot(cols,line_noise_sort[5],color='darkred',marker=".",linestyle='None',markersize=30,label='Column noise - 200 second exposure')
    plt.plot(cols,(np.median(line_noise_sort[0])*np.ones(cols.shape)),color='black',marker="None",linestyle='--',markersize=30,label='Median Column noise - 0 second exposure')
    plt.legend(loc="upper right",fontsize=40)
    figtext(.15, .8,'Median Column Noise = %0.4f e/pix/fr\n' % (np.median(line_noise_sort[0])),fontsize=40)
    figtext(.18, .4,'16 overscan pixels \n604 multiplication \nelements\n',fontsize=40)
    figtext(.30, .35,'Dump gate \nactivation\n',fontsize=40)
    figtext(.4, .35,'468 corner \nelements\n',fontsize=40)
    figtext(.5, .35,'1056 serial \nregister pixels\n',fontsize=40)
    fig_name = data_dir.replace('/mnt/c/Users/gikyne/Documents/data/radiation_tests/','')
    fig_name = fig_name[:-1].replace('/','_')
    fig_dir = data_dir + 'figures/'
    if not os.path.exists(fig_dir):# create data directory if needed
       os.makedirs(fig_dir)
    plt.savefig(fig_dir + fig_name + '.cols.png',bbox_inches='tight')
    #plt.show()

    # Plot frame number versus number of flagged pixels
    plt.close("all")
    plt.clf()
    fig = plt.figure(figsize=(30,15))
    #plt.rc('text', usetex=True)
    #gca().set_position((.1, .3, .8, .65))
    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']='40'
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times'],'size': 50})
    plt.xlabel("Row #",fontsize=35)
    plt.ylabel("Noise [e/pix/fr]",fontsize=35)
    grid(b=True, which='major', color='0.75', linestyle='--')
    grid(b=True, which='minor', color='0.75', linestyle='--')
    #plt.ylim([0.004,0.016])

    #plt.plot(cols,line_noise[0],"k-",markersize=30,label='Serial register noise')
    plt.plot(rows,col_noise_sort[0],color='navy',marker=".",linestyle='None',markersize=30,label='Row noise - 0 second exposure')
    plt.plot(rows,col_noise_sort[3],color='darkgreen',marker=".",linestyle='None',markersize=30,label='Row noise - 10 second exposure')
    plt.plot(rows,col_noise_sort[5],color='darkred',marker=".",linestyle='None',markersize=30,label='Row noise - 200 second exposure')
    plt.plot(rows,(np.median(col_noise_sort[0])*np.ones(rows.shape)),color='black',marker="None",linestyle='--',markersize=30,label='Median Row noise - 0 second exposure')
    plt.legend(loc="upper right",fontsize=40)
    figtext(.15, .8,'Median Row Noise = %0.4f e/pix/fr.\n' % (np.median(col_noise_sort[0])),fontsize=40)
    fig_name = data_dir.replace('/mnt/c/Users/gikyne/Documents/data/radiation_tests/','')
    fig_name = fig_name[:-1].replace('/','_')
    fig_dir = data_dir + 'figures/'
    if not os.path.exists(fig_dir):# create data directory if needed
       os.makedirs(fig_dir)
    plt.savefig(fig_dir + fig_name + '.rows.png',bbox_inches='tight')
    #plt.show()

    return()

def run_ALL(t0):

    #t0 = time()

    print('\n ***** Processing : ')
    print('\n ***** Making data cubes for processing : ')
    make_datacubes()

    t1 = time.time()

    print('\n ***** Removing cosmic rays : ')
    run_CR_removal(options.nthreads)

    t2 = time.time()
    print('\n ***** Total CR processing time %4.3f seconds : ' %(t2-t1))

    make_CR_datacubes()
    make_CRmask_datacubes()

    t3 = time.time()

    thresh = run_desmear_process(options.method, options.thresh, options.nthreads)

    t4 = time.time()
    print('\n ***** Total desmear processing time %4.3f seconds : ' %(t4-t3))

    make_corr_datacubes()

    try:
        dark_rate(thresh)
    except:
        fp = open(data_dir+ 'desmear_thresh', 'r')
        line = float(fp.readline())
        fp.close()
        dark_rate(line)
    #dark_rate_CR()

    t5 = time.time()

    print('\n ***** Total processing time %4.3f seconds : ' %(t5-t0))


if __name__ == "__main__":


    t0 = time.time()


    usage = "Usage: python %prog [options] image"
    description = "Description. Utility to reject cosmic rays and generate bad pixel masks and CR pixels flagged images."
    parser = OptionParser(usage = usage, version = "%prog 0.3", description = description)
    parser.add_option("-m", "--method", dest = "method", metavar="METHOD",
                    action="store", help = "Rejection method [default is nmad]",
                    choices = ["nmad", "mad"], default = "nmad"
        )
    parser.add_option("-t", "--thresh", dest = "thresh", metavar="THRESH",
            action="store", help = "Sigma threshold [default is 3.0]",
            default = 3.0
    )
    parser.add_option("-n", "--nthreads", dest = "nthreads", metavar="NTHREADS",
                    action="store", help = "Number of threads [default is 12]",
                    default = 12
     )

    (options, args) = parser.parse_args()


        run_ALL(t0)
        #cropimages()
        #indata = data_dir+'image_CR_cube_6.fits', img_area, 'nmad', '6'
        #desmear(indata)
        #make_datacubes()
        #indata = dark_rate(options.thresh)
        #dark_rate(options.thresh)
        #cic_noise(indata)
        #dark_rate_CR()
        #dark_rate_temp()
    #glob_pattern = data_dir + 'image0*.fits'
        #extrapattern = data_dir + 'image0*.*.fits'
        #images =  set(glob.glob(glob_pattern)) - set(glob.glob(extrapattern))
        #for image in images: #image = sys.argv[1]
        #    indata = image, shielded_area
        #    countCR(indata)
        #removeCRtails(indata)
        #calc_emgain(data_dir+sys.argv[1],img_area)

    t5 = time.time()

    print('\n ***** Total processing time %4.3f seconds : ' %(t5-t0))
