#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 16:13:44 2019

@author: dvibert
"""

import numpy as np
from scipy.special import hyp1f1, gamma

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from matplotlib import pyplot as plt
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from  pkg_resources  import resource_filename
import os

def normalgamma(nooe, g, r , lamb):
    lamb2 = lamb/2.
    rsq = r**2
    aux = np.square(-g*nooe + rsq)/(2*g**2*rsq)
    res = 2**(-lamb2) * np.exp(- np.square(nooe)/(2*rsq)) * g**(-lamb) * r**(lamb - 2)
    res *= r  * hyp1f1(lamb2, .5, aux) / np.sqrt(2)*gamma((lamb + 1)/2) + \
            (g*nooe - rsq)*hyp1f1((lamb+1)/2, 3/2, aux) / (g*gamma(lamb2))
    
    return res



def bias_from_right_overscan(ima, left=2135, right=2580):
   overscan = ima[left:right,:] 
   bias, median, stdev = sigma_clipped_stats(overscan, mask=np.isnan(overscan), axis=0)
   ima_bias_cor = ima - bias
   return ima_bias_cor, bias, stdev


def plot_histo_bias(data, mean, median, stdev):
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




def multiple_photon_counting(ima, prior=None):
    
    # tab_path = resource_filename('old', 'PhotoCountingTables')
    tab_prior_name = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FIREBall/old/PhotoCountingTables/tabwithprior-EMgain470-rnoise109.csv"#os.path.join(tab_path, 'tabwithprior-EMgain470-rnoise109.csv')
    tab_woprior_name = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FIREBall/old/PhotoCountingTables/tabnoprior-EMgain470-rnoise109.csv"#os.path.join(tab_path, 'tabnoprior-EMgain470-rnoise109.csv')
    
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
    # remove isolated bright pixels
    
    kernel  = Gaussian2DKernel(3)
    #bkg = convolve(ima, np.ones((11,11))/121)
    #bkg = gaussian_filter(ima, 3)
    bkg = convolve(ima, kernel)
    return bkg
    
    
def process_image(ima, with_prior=False):


    # if hdr[my_conf.gain[0]] == 0:
    #     EMgain = 1
    # if hdr[my_conf.gain[0]] == 9000:
    #     EMgain = 235
    # if hdr[my_conf.gain[0]] == 9200:
    #     EMgain = 470
    # if hdr[my_conf.gain[0]] == 9400:
    #     EMgain = 700

    EMgain = 470.
    ADU2e = .53
    
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



# if __name__ == '__main__':
    

#     path = '/home/dvibert/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CosmicRaysFree/'

#     im_nb = np.array([88, 92, 93, 94, 96, 97, 105, 108, 109, 111, 112, 113, 
#                       114, 118, 120, 124, 127, 128, 130, 131, 140])

#     fnames = ['image{:06d}.CRv.fits'.format(i) for i in im_nb]


    # imasum_noprior = np.zeros((3216, 2069))
    # imasum_prior = np.zeros((3216, 2069))
    # for fn in fnames:
        
    #     with fits.open(path + fn) as f:
    #         ima = f[0].data.T
    #         hdr = f[0].header
    
    #     ima_count_noprior = process_image(ima)
    #     ima_count_prior = process_image(ima, with_prior=True)

    #     imasum_noprior += ima_count_noprior
    #     imasum_prior += ima_count_prior

ds9 = process_image(ds9)
# ds9 = process_image(ds9, with_prior=True)


