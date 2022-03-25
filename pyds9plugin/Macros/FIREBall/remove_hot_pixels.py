from pyds9plugin.DS9Utils import *
from astropy.table import Column
from scipy import signal
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
import os

# TODO ask for mask

# if ('snr.py' in __file__) or (function=='execute_command'):
#     from astropy.table import Table
#     d=DS9n()
#     fitsfile=d.get_pyfits()
#     filename = get_filename(d)
#     table=create_table_from_header(filename, exts=[0], info='')#Table(data=[[1]],names=['test'])
#     filename=get_filename(d)
# else:
#     pass

    
# data = fitsfile[0].data.astype(float)
# header = fitsfile[0].header

lx, ly = ds9.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = 1120, 2100, 1300, 1900
# Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
physical_region = ds9[Yinf:Ysup, Xinf:Xsup]
pre_scan = ds9[:, 600:1000]
post_scan = ds9[:, 2500:3000]
range_=(np.nanpercentile(physical_region,0.1),np.nanpercentile(physical_region,99.9))
value, b = np.histogram(ds9[Yinf:Ysup, Xinf:Xsup].flatten(),bins=np.arange(1000,8000,1))
bins = (b[1:]+b[:-1])/2
bias = bins[np.argmax(value)]

limit_max = []
limit_max.append(bins[np.where((bins>bias) & ( np.convolve(value,np.ones(2),mode='same')==0))[0][0]])
limit_max.append(bins[np.where((bins>bias) & ( np.convolve(value,np.ones(1),mode='same')==int(value.max()/1e4)))[0][0]])
limit_max.append(bins[np.where((bins>bias) & ( np.convolve(value,np.ones(1),mode='same')==int(value.max()/5e2)))[0][0]])
for i in range(3):
    # limit_max = bins[np.where((bins>bias) & ( np.convolve(value,np.ones(2),mode='same')==0))[0][0]]-400*i
    mask = ds9>limit_max[i]
    mask2 =   np.hstack([mask[:,-1:], mask[:,:-1]])#.shape 
    mask3 =   np.hstack([mask[:,-2:], mask[:,:-2]])#.shape 
    ds9[mask | mask2 | mask3]=np.nan#np.median(ds9) #np.nan
    STD_DEV = 1
    while ~np.isfinite(ds9).all():
        print(limit_max, STD_DEV,np.mean(~np.isfinite(ds9)))
        kernel = Gaussian2DKernel(x_stddev=STD_DEV, y_stddev=STD_DEV)
        ds9 = interpolate_replace_nans(ds9, kernel)
        STD_DEV += 1    
    # filename = '/tmp/2022_hot_pixels_corrected_%i.fits'%(limit_max[i])
    # fitsfile.writeto(filename,overwrite=True)
