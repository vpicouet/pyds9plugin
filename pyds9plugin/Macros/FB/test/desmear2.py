from tqdm.tk import tqdm
from astropy.io import fits
from pyds9plugin.DS9Utils import DS9n, get_filename, PlotFit1D, verboseprint
import numpy as np
import os

# indata = data_dir+'image_CR_cube_6.fits', img_area, 'nmad', '6'
filename='/Users/Vincent/Downloads/Safari/image_cube_1.fits'
d=DS9n()
# filename=get_filename(d)
image_cube, img_area, method, threshold = filename, [1100,1900,1300,1600], 'nmad', '5.0'
# image_cube, img_area, method, threshold = filename, [0,1000,0,1000], 'mad', '5.0'
# Read data from FITS image
try:
    img_data,header = fits.getdata(image_cube,header=True)
except IOError:
    raise IOError("Unable to open FITS image %s" %(image_cube))

# Only 3D data cubes are accepted
if np.ndim(img_data) == 2:
    # raise ValueError("Routine only accepts 3D data cubes.")
    pre_scan = img_data[:, 600:1000]
    pre_scan = img_data[:, :]
else:
    pre_scan = img_data[:, :]

value_os, b_os = np.histogram(pre_scan.flatten(),bins=np.arange(np.nanmin(pre_scan),np.nanmax(pre_scan),1))
bins_os = (b_os[1:]+b_os[:-1])/2


bias_os = bins_os[np.argmax(value_os)]
RN=50
mask_RN_os = (bins_os>bias_os-1*RN) & (bins_os<bias_os+0.8*RN)  &(value_os>0)

# hist_output = calc_emgain_CR(image_cube,area)
popt = PlotFit1D(bins_os[mask_RN_os],value_os[mask_RN_os],deg='gaus', plot_=False,P0=[1,bias_os,50,0])['popt']
# table['Amp'] =   popt[0]
# table['bias_fit'] =   popt[1]
bias =  popt[1]
sigma =  popt[2]#100#300#popt[2]
verboseprint(bias,sigma)
# sys.exit()


# Rejection method, threshold

# Rejection method, threshold
threshold = float(threshold)

print("Method: ", method, "  Threshold: ", threshold, "Cube #: ", image_cube)

# Image dimension
zsize, ysize, xsize = img_data.shape
print('compute mad')
if method == "mad":
# Median and MAD values
    medianval = np.median(img_data, 0)
    madval = np.median(np.abs(medianval - img_data), 0)
else:
    # Get median value for all the pixels between all the frames
    # and taking -1 bottom and ledft and +2 top of right
    medianval = np.zeros([ysize, xsize], dtype = np.float32)
    for j in tqdm(range(ysize), desc="Analysis: ", ncols=99, unit=" lines", unit_scale=True):
        for i in range(xsize):
            if (i > 0 and i < xsize) and (j > 0 and j < ysize):
                medianval[j,i] = np.median(img_data[:,j-1:j+2,i-1:i+2])
            else:
                medianval[j,i] = np.median(img_data[:,j,i])
    # Calculate MAD value
    # check abs median(mad[i,j] - data[:,j-1:j+2,i-1:i+2])
    madval = np.zeros([ysize, xsize], dtype = np.float32)
    for j in tqdm(range(ysize), desc="Analysis: ", ncols=99, unit=" Columns", unit_scale=True):
        for i in range(xsize):
            if (i > 0 and i < xsize) and (j > 0 and j < ysize):
                madval[j,i] = np.median(np.abs(medianval[j,i] - img_data[:,j-1:j+2,i-1:i+2]))
            else:
                madval[j,i] = np.median(np.abs(medianval[j,i] - img_data[:,j,i]))

print('done')
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
    #for all masked values
    for j in tqdm(range(0,mask_data_ind.shape[1]), desc="Analysis: %i"%(i), ncols=99, unit=" count", unit_scale=True):
    
        xind = mask_data_ind[1,j]
        yind = mask_data_ind[0,j]
        #if in the area
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
        fixed_smear_sum = np.sum(fixed_smear) - (rpix)*int(bias)
           #print 'image %i lpix %i orig %i summed %i' %(i, int(lpix), int(new_img_data[yind,xind]), int(fixed_smear_sum))
        new_img_data[yind,xind] = int(fixed_smear_sum)
        pixel_length.append(rpix)
    
        #TBC if indentation ok
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
# d.set('file  '+ new_image_fname)

# plt.imshow((new_img_data == img_data)[-1,:,:]);plt.colorbar()

    print("\t  DESMEARED image : %s" %(new_image_fname))
    
    corr_mask = image_cube.replace(".fits", "." + str(i) + ".corrmask.fits")
    if os.path.exists(corr_mask):
        os.remove(corr_mask)
    fits.writeto(corr_mask,corr_mask_data,header=header)


# ds9 = new_img_data
# d.set('frame new ; file  '+ corr_mask)
    print("\t  CORRECTED image mask : %s" %(corr_mask))
    
    print("\t  Pixel smeared length : %s" %(np.sum(np.array(pixel_length))/len(pixel_length)))
    

np.savetxt(image_cube.replace('.fits','.smeared'), percent_flag_events,fmt='%.2f')
np.savetxt(image_cube.replace('.fits','.pixlen'), pixel_length)
#np.savetxt(image_cube.replace('.fits','.darkevents'), percent_darkpix_events,fmt='%.2f')
