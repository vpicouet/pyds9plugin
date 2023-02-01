from tqdm.tk import tqdm
from astropy.io import fits
from pyds9plugin.DS9Utils import DS9n, get_filename, PlotFit1D, verboseprint, getregion
from pyds9plugin.DS9Utils import fitswrite, yesno, lims_from_region, globglob
import numpy as np
import os
from pyds9plugin.Macros.FB.FB_functions import emccd_model
from astropy.table import Table
d=DS9n()


def correct_smearing(new_img_data, mask_data):
    mask_data_ind = np.array(np.where(mask_data == 1))
    #for all masked values
    for j in range(0,mask_data_ind.shape[1]):
    # for j in tqdm(range(0,mask_data_ind.shape[1]), desc="Analysis: %i"%(i), ncols=99, unit=" count", unit_scale=True):
    
        xind = mask_data_ind[1,j]
        yind = mask_data_ind[0,j]
        #if in the area
        if xind > 1 and xind < mask_data.shape[1] - 1:
            if mask_data[yind,xind] == 1 and mask_data[yind,xind-1] == 0:
                xval = xind
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
                # print(xval)
        counter = 0
        while (xval < mask_data.shape[1]-1) & (mask_data[yind,xval] == 1):#  bug only with mad! and new_img_data[yind,(xval+1)] < bias - (5*sigma):
            xval = xval + 1
            counter = counter + 1
            # if xval == mask_data.shape[1]-1:
            #     break
            rpix = counter
    
        #Making new mask with LHS pixel smear check
        corr_mask_data[yind,xind+1:xind+rpix+1] = -1
        fixed_smear = np.array(new_img_data[yind,xind:xind+rpix+1])
        fixed_smear_sum = np.nansum(fixed_smear) - (rpix)*int(bias)
           #print 'image %i lpix %i orig %i summed %i' %(i, int(lpix), int(new_img_data[yind,xind]), int(fixed_smear_sum))
       # TODO should put it in the right or left of smearing depending on CTE direction
        new_img_data[yind,xind] = int(fixed_smear_sum)
        pixel_length.append(rpix)
    
    #TBC if indentation ok
    newmask_idx = corr_mask_data < 0
    # TODO takes 75% of the time!
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
    return new_img_data





# indata = data_dir+'image_CR_cube_6.fits', img_area, 'nmad', '6'
# if 1==0:
#     filename='/Users/Vincent/Downloads/Safari/image_cube_1.fits'
#     d=DS9n()
#     # filename=get_filename(d)
#     image_cube, img_area, method, threshold = filename, [1100,1900,1300,1600], 'nmad', '5.0'
#     # image_cube, img_area, method, threshold = filename, [0,1000,0,1000], 'mad', '5.0'
#     # Read data from FITS image
#     try:
#         img_data,header = fits.getdata(image_cube,header=True)
#     except IOError:
#         raise IOError("Unable to open FITS image %s" %(image_cube))
    
#     # Only 3D data cubes are accepted
#     if np.ndim(img_data) == 2:
#         # raise ValueError("Routine only accepts 3D data cubes.")
#         pre_scan = img_data[:, 600:1000]
#         pre_scan = img_data[:, :]
#     else:
#         pre_scan = img_data[:, :]
# else:
path = filename #"/Users/Vincent/Nextcloud/LAM/Work/EMCCD_reduction/smearing/field_test.fits"#get(d,"Give the path of the images you want to desmear (around 5)")
# path = "/Users/Vincent/Nextcloud/LAM/Work/FIREBall/FB_Images/2018_images/image000388_dark_modified.fits"
# %%
if 1==1:
    # pre_scan=region
    system = "Image"
    region = getregion(d, quick=True, selected=True, system=system, dtype=float)
    if region is None:
        img_area= [0, -1, 0, -1]
        Xinf, Xsup, Yinf, Ysup  = img_area
    else:
        Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region[0], dtype=float)
    
# if 1==1:
#     path = "/Users/Vincent/Nextcloud/LAM/Work/EMCCD_reduction/smearing/gillian_code/image??????.fits"
#     Xinf, Xsup, Yinf, Ysup = 1500,1800,1500,1800
    Xinf, Xsup, Yinf, Ysup = np.array([Xinf, Xsup, Yinf, Ysup],dtype=int)
    img_area= [Xinf, Xsup, Yinf, Ysup]
    print(img_area)
    image_cube = globglob(path)[0]
    header = fits.getheader(image_cube)
    method = 'mad'
    threshold =  '5.0'
    if yesno(d,"Do you confirm that smearing is in the left direction?"):
        direction = "left" 
    else:
        direction = "right" 
        
    for p in globglob(path):
        if direction == "left":
            fitswrite(fits.open(p)[0].data[Xinf: Xsup, Yinf: Ysup][:,::-1],p.replace(".fits","_sub.fits"))
        else:
            fitswrite(fits.open(p)[0].data[Xinf: Xsup, Yinf: Ysup],p.replace(".fits","_sub.fits"))
    if direction == "left":
        img_data =  np.dstack([fits.open(p)[0].data[Xinf: Xsup, Yinf: Ysup][:,::-1].T for p in globglob(path)]).T
    else:
        img_data =  np.dstack([fits.open(p)[0].data[Xinf: Xsup, Yinf: Ysup].T for p in globglob(path)]).T
    pre_scan = img_data
        
value_os, b_os = np.histogram(pre_scan.flatten(),bins=np.arange(np.nanmin(pre_scan),np.nanmax(pre_scan),1))
bins_os = (b_os[1:]+b_os[:-1])/2


bias_os = bins_os[np.nanargmax(value_os)]
RN=10
mask_RN_os = (bins_os>bias_os-1*RN) & (bins_os<bias_os+0.8*RN)  &(value_os>0)

# hist_output = calc_emgain_CR(image_cube,area)
popt = PlotFit1D(bins_os[mask_RN_os],value_os[mask_RN_os],deg='gaus', plot_=False,P0=[value_os[mask_RN_os].ptp(),bias_os,50,0])['popt']
# table['Amp'] =   popt[0]
# table['bias_fit'] =   popt[1]
bias =  popt[1]
sigma =  popt[2]#100#300#popt[2]
verboseprint(bias,sigma)
# sys.exit()


# Rejection method, threshold

# Rejection method, threshold
threshold = float(threshold)

print("Method: ", method, ",  Threshold: ", threshold, ", Cube : ", image_cube)

# Image dimension
zsize, ysize, xsize = img_data.shape
# print('compute mad')
if method == "mad":
# Median and MAD values
    medianval = np.nanmedian(img_data, axis=0)
    madval = np.nanmedian(np.abs(medianval - img_data), axis=0)
elif method == "nmad":
    # Get median value for all the pixels between all the frames
    # and taking -1 bottom and ledft and +2 top of right
    medianval = np.zeros([ysize, xsize], dtype = np.float32)
    for j in tqdm(range(ysize), desc="Analysis: ", ncols=99, unit=" lines", unit_scale=True):
        for i in range(xsize):
            if (i > 0 and i < xsize) and (j > 0 and j < ysize):
                medianval[j,i] = np.nanmedian(img_data[:,j-1:j+2,i-1:i+2])
            else:
                medianval[j,i] = np.nanmedian(img_data[:,j,i])
    # Calculate MAD value
    # check abs median(mad[i,j] - data[:,j-1:j+2,i-1:i+2])
    madval = np.zeros([ysize, xsize], dtype = np.float32)
    for j in tqdm(range(ysize), desc="Analysis: ", ncols=99, unit=" Columns", unit_scale=True):
        for i in range(xsize):
            if (i > 0 and i < xsize) and (j > 0 and j < ysize):
                madval[j,i] = np.nanmedian(np.abs(medianval[j,i] - img_data[:,j-1:j+2,i-1:i+2]))
            else:
                madval[j,i] = np.nanmedian(np.abs(medianval[j,i] - img_data[:,j,i]))

#Fill array with random numbers around the bias:
prandom_array = np.random.normal(int(bias),int(abs(sigma)), size=(img_data.shape[1],img_data.shape[2]))

pixel_length = []
percent_flag_events = np.zeros(img_data.shape[0], dtype = float)
for i in range(img_data.shape[0]):
    print("\tProcessing Frame: %s" %(i+1))
    # TODO here if only one image, madval=0 
    if img_data.shape[0]>1:
        idx = img_data[i,:,:] > medianval + threshold * madval
    else:
        idx = img_data[i,:,:] > bias + threshold * sigma
        
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
    
    #Generate masks for analysis
    
    new_img_data = np.copy(img_data[i,:,:])
    corr_mask_data = np.copy(mask_data)
    if direction == "left":
        ds9 = correct_smearing(new_img_data, mask_data)[:,::-1] 
        fits.writeto(mask_fname,mask_data[:,::-1],header=header)
    else:
        ds9 = correct_smearing(new_img_data, mask_data)
        fits.writeto(mask_fname,mask_data,header=header)
    new_image_fname = image_cube.replace(".fits", "." + str(i) + ".corr.fits")
    
    

filename = "/tmp/test_0g_0s.csv"
min_, max_ = (np.nanpercentile(ds9, 0.4), np.nanpercentile(ds9, 99.8))
val, bins = np.histogram(img_data.flatten(), bins=np.arange(min_, max_, 1))
val_new, _ = np.histogram(ds9.flatten(), bins=np.arange(min_, max_, 1))
bins = (bins[1:] + bins[:-1]) / 2
val = np.array(val, dtype=float)/ np.isfinite(img_data).mean()
val_new = np.array(val_new, dtype=float)/ np.isfinite(ds9).mean()
Table([bins,val,val_new]).write(filename)

emccd_model(xpapoint=None, path=filename, smearing=0, argv=[])

    # print("\t  DESMEARED image : %s" %(new_image_fname))
    # if os.path.exists(new_image_fname):
    #     os.remove(new_image_fname)
    # fits.writeto(new_image_fname,new_img_data,header=header)    
    # corr_mask = image_cube.replace(".fits", "." + str(i) + ".corrmask.fits")
    # if os.path.exists(corr_mask):
    #     os.remove(corr_mask)
    # fits.writeto(corr_mask,corr_mask_data,header=header)
    # print("\t  CORRECTED image mask : %s" %(corr_mask))    
    # print("\t  Pixel smeared length : %s" %(np.sum(np.array(pixel_length))/len(pixel_length)))
# np.savetxt(image_cube.replace('.fits','_smeared.txt'), percent_flag_events,fmt='%.2f')
# np.savetxt(image_cube.replace('.fits','_pixlen.txt'), pixel_length)
#np.savetxt(image_cube.replace('.fits','.darkevents'), percent_darkpix_events,fmt='%.2f')



# %load_ext line_profiler
# %lprun -f correct_smearing correct_smearing(new_img_data, mask_data)
