from pyds9plugin.DS9Utils import *
from astropy.table import Column
from scipy import signal
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
import os

# TODO ask for mask

if ('snr.py' in __file__) or (function=='execute_command'):
    from astropy.table import Table
    d=DS9n()
    fitsfile=d.get_pyfits()
    filename = get_filename(d)
    table=create_table_from_header(filename, exts=[0], info='')#Table(data=[[1]],names=['test'])
    filename=get_filename(d)
else:
    pass

    
data = fitsfile[0].data
header = fitsfile[0].header


lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = 1120, 2100, 1300, 1900
# Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
physical_region = data[Yinf:Ysup, Xinf:Xsup]
pre_scan = data[:, 600:1000]
post_scan = data[:, 2500:3000]
range_=(np.nanpercentile(physical_region,0.1),np.nanpercentile(physical_region,99.9))
value, b = np.histogram(data[Yinf:Ysup, Xinf:Xsup].flatten(),bins=np.arange(1000,8000,1))
bins = (b[1:]+b[:-1])/2
bias = bins[np.argmax(value)]




slits = fits.open("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220204_darks_T183_1MHz/7000/test_snr/1_image/image_220311-23H54M34.fits")[0].data
fitsfile[0].data = fitsfile[0].data.astype(float)
fitsfile[0].data += slits
# new_filename1 = '/tmp/hot_' + os.path.basename(filename.replace('.fits','_slits.fits'))
new_filename1 = '/tmp/2022_hot_pixels.fits'
fitsfile.writeto(new_filename1,overwrite=True)
fitsfile[0].data -= slits

files = glob.glob('/Users/Vincent/Nextcloud/LAM/FIREBALL/TestsFTS2018-Flight/E2E-AIT-Flight/all_diffuse_illumination/1/*im.fits')

data = fitsfile[0].data
for i in range(3):
    limit_max = bins[np.where((bins>bias) & ( np.convolve(value,np.ones(1),mode='same')==0))[0][0]]-400*i
    mask = data>limit_max
    mask2 =   np.hstack([mask[:,-1:], mask[:,:-1]])#.shape 
    mask3 =   np.hstack([mask[:,-2:], mask[:,:-2]])#.shape 
    data[mask | mask2 | mask3]=np.nan#np.median(data) #np.nan
    STD_DEV = 1
    while ~np.isfinite(data).all():
        print(limit_max, STD_DEV,np.mean(~np.isfinite(data)))
        kernel = Gaussian2DKernel(x_stddev=STD_DEV, y_stddev=STD_DEV)
        data = interpolate_replace_nans(data, kernel)
        STD_DEV += 1    
    fitsfile[0].data = data+slits
    filename = '/tmp/2022_hot_pixels_corrected_%i.fits'%(limit_max)
    fitsfile.writeto(filename,overwrite=True)
    files.append(filename)

# fitsfile[0].data = fitsfile[0].data.astype(int)
# fitsfile[0].header['BITPIX']=16 #does not work





file_2018 = "/Users/Vincent/Nextcloud/LAM/FIREBALL/TestsFTS2018-Flight/E2E-AIT-Flight/all_diffuse_illumination/1/180823_20s_20im.fits"
file_2018 = "/Users/Vincent/Nextcloud/LAM/FIREBALL/TestsFTS2018-Flight/E2E-AIT-Flight/all_diffuse_illumination/1/180904_60s_1im.fits"
cat_file_2018 =  file_2018.replace('.fits','_cat.fits')
os.system(' sex %s   -WRITE_XML Y  -CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -PARAMETERS_NAME /Users/Vincent/Github/pyds9plugin/pyds9plugin/Sextractor/sex_vignet.param -DETECT_TYPE CCD -DETECT_MINAREA 100 -DETECT_MAXAREA 0 -THRESH_TYPE RELATIVE -DETECT_THRESH 5 -ANALYSIS_THRESH 5 -FILTER Y -FILTER_NAME /Users/Vincent/Github/pyds9plugin/pyds9plugin/Sextractor/gauss_4.0_7x7.conv -DEBLEND_NTHRESH 64 -DEBLEND_MINCONT 0.0003 -CLEAN Y -CLEAN_PARAM 1.0 -MASK_TYPE CORRECT -WEIGHT_TYPE NONE -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y -FLAG_IMAGE NONE -FLAG_TYPE OR -PHOT_APERTURES 6,12,18 -PHOT_AUTOPARAMS 2.5,4.0 -PHOT_PETROPARAMS 2.0,4.0 -PHOT_FLUXFRAC 0.3,0.5,0.9 -SATUR_LEVEL 50000.0 -SATUR_KEY SATURATE -MAG_ZEROPOINT 0.0 -MAG_GAMMA 4.0 -GAIN GAIN -PIXEL_SCALE 0 -SEEING_FWHM 0.8 -STARNNW_NAME /Users/Vincent/Github/pyds9plugin/pyds9plugin/Sextractor/default.nnw -CHECKIMAGE_TYPE NONE -BACK_TYPE AUTO -BACK_VALUE 0.0 -BACK_SIZE 64 -BACK_FILTERSIZE 3 -BACKPHOTO_TYPE LOCAL -BACKPHOTO_THICK 24 -BACK_FILTTHRESH 0.0 -MEMORY_OBJSTACK 3000 -MEMORY_PIXSTACK 300000 -MEMORY_BUFSIZE 1024 -NTHREADS 10 -CHECKIMAGE_NAME /Users/Vincent/DS9QuickLookPlugIn/Users/Vincent/DS9QuickLookPlugIn/tmp/image_check_NONE.fits'%(file_2018, cat_file_2018))
cat_2018=Table.read(cat_file_2018)
# name=os.path.basename(file)[:-5]
fields = ['VIGNET',"SNR_WIN","FLUX_MAX"]
for field in fields:
    cat_2018.rename_column(field,field + os.path.basename(file_2018)[:-5])

files.remove(file_2018)
# files.append(new_filename1)
# files.append(new_filename2)
for filei in files:
    cat_file = filei.replace('.fits','_cat.fits')
    os.system(' sex %s,%s   -WRITE_XML Y  -CATALOG_NAME %s -CATALOG_TYPE FITS_1.0 -PARAMETERS_NAME /Users/Vincent/Github/pyds9plugin/pyds9plugin/Sextractor/sex_vignet.param -DETECT_TYPE CCD -DETECT_MINAREA 100 -DETECT_MAXAREA 0 -THRESH_TYPE RELATIVE -DETECT_THRESH 5 -ANALYSIS_THRESH 5 -FILTER Y -FILTER_NAME /Users/Vincent/Github/pyds9plugin/pyds9plugin/Sextractor/gauss_4.0_7x7.conv -DEBLEND_NTHRESH 64 -DEBLEND_MINCONT 0.0003 -CLEAN Y -CLEAN_PARAM 1.0 -MASK_TYPE CORRECT -WEIGHT_TYPE NONE -RESCALE_WEIGHTS Y -WEIGHT_GAIN Y -FLAG_IMAGE NONE -FLAG_TYPE OR -PHOT_APERTURES 6,12,18 -PHOT_AUTOPARAMS 2.5,4.0 -PHOT_PETROPARAMS 2.0,4.0 -PHOT_FLUXFRAC 0.3,0.5,0.9 -SATUR_LEVEL 50000.0 -SATUR_KEY SATURATE -MAG_ZEROPOINT 0.0 -MAG_GAMMA 4.0 -GAIN GAIN -PIXEL_SCALE 0 -SEEING_FWHM 0.8 -STARNNW_NAME /Users/Vincent/Github/pyds9plugin/pyds9plugin/Sextractor/default.nnw -CHECKIMAGE_TYPE NONE -BACK_TYPE AUTO -BACK_VALUE 0.0 -BACK_SIZE 64 -BACK_FILTERSIZE 3 -BACKPHOTO_TYPE LOCAL -BACKPHOTO_THICK 24 -BACK_FILTTHRESH 0.0 -MEMORY_OBJSTACK 3000 -MEMORY_PIXSTACK 300000 -MEMORY_BUFSIZE 1024 -NTHREADS 10 -CHECKIMAGE_NAME /Users/Vincent/DS9QuickLookPlugIn/Users/Vincent/DS9QuickLookPlugIn/tmp/image_check_NONE.fits'%(file_2018, filei, cat_file))
    cat=Table.read(cat_file)
    name=os.path.basename(filei)[:-5]
    for field in fields:#"SNR_WIN","BKGSIG"
        cat_2018[field + name] = cat[field]#cat['FLUX_MAX']colnames
# n=50
names = [name for name in cat_2018.colnames if len(cat_2018[name].shape) > 2] * 2
for field in names:
    for i in range(len(cat_2018)):
        cat_2018[field][i][cat_2018[field][i]<-1e29]=np.nan
cat_2018.write('/tmp/noise_analysis.fits',overwrite=True)


explore_throughfocus(argv="-p /tmp/noise_analysis.fits")

# names=[ 'VIGNET180823_20s_20im','VIGNET_SHIFT']
# for line in cat_2018:
#     fig, (ax1,ax2) = plt.subplots(1,2)
#     ax1.imshow(np.array(line[names[0]],dtype=float))
#     ax2.imshow(np.array(line[names[1]],dtype=float))
#     ax1.set_title(names[0])
#     ax2.set_title(names[1])
#     # ax3.imshow(np.array(line[names[2]],dtype=float))
#     # plt.colorbar()
#     plt.show()
