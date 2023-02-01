#!/usr/bin/env python3


    # parser.add_option("-m", "--method", dest = "method", metavar="METHOD",
    #                 action="store", help = "Rejection method [default is nmad]",
    #                 choices = ["nmad", "mad"], default = "nmad"
    #     )
    # parser.add_option("-t", "--thresh", dest = "thresh", metavar="THRESH",
    #         action="store", help = "Sigma threshold [default is 3.0]",
    #         default = 3.0
    # )
    # parser.add_option("-n", "--nthreads", dest = "nthreads", metavar="NTHREADS",
    #                 action="store", help = "Number of threads [default is 12]",
    #                 default = 12
    #  )
img_area = [1100,1900,1300,1600]
data_dir = '/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220204_darks_T183_1MHz/6800/'
data_dir = data_dir+'redux/'
indata = data_dir+'image_CR_cube_6.fits', img_area, 'nmad', '6'

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
