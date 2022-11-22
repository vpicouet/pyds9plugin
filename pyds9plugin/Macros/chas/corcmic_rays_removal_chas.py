from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft as convolve

from scipy import ndimage

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

__all__ = ['create_stack', 'cross_correlation_shifts']

def create_stack(path, img_list, center, box, full, maxoff, outfile): 
    ''' 
        Calculate offset between images in a list and create a stack. 
	The match should be done using a cut of the full image to improve
        speed and accuracy (select region w/ good S/N).         

        Parameters
        ----------
        path (string): path to image directory (e.g. './dir/')  
        img_list (string list): list of image filenames to stack 
        center (np.array): the center of the cut image [c1, c2] 
        box (np.array): the shape of the cut image [len1, len2] 
        full (np.array): the shape of the full image around center cut

        Returns 
        -------
    	final_image (np.ndarray): final stacked image as an array
    '''

    # load first image
    hdul = fits.open(path+ img_list[0])
    img1 = hdul[0].data

    # cut first image 
    cut1= img1[center[1]-box[1]:center[1]+box[1]+1,\
	       center[0]-box[0]:center[0]+box[0]+1] 
    img1 = img1[center[1]-full[1]:center[1]+full[1]+1,\
		center[0]-full[0]:center[0]+full[0]+1]

    dy1, dx1 = [], []
    yoff, xoff = 0, 0
    for i in range(1, len(img_list)):    
        # load second image 
        hdul = fits.open(path+ img_list[i])
        img2 = hdul[0].data

        # cut small search area
        center = center + np.array([np.int(np.ceil(np.int(yoff))), \
                                    np.int(np.ceil(xoff))])
        cut2= img2[center[1]-box[1]:center[1]+box[1]+1, \
                   center[0]-box[0]:center[0]+box[0]+1]
        img2 = img2[center[1]-full[1]:center[1]+full[1]+1, \
                    center[0]-full[0]:center[0]+full[0]+1]

        # get shift 
        yoff, xoff = cross_correlation_shifts(cut1, cut2, maxoff = maxoff)
        
        # shift first image to second
        cut3 = np.roll(np.roll(cut1,int(yoff),1),int(xoff),0)
    
        img3 = np.roll(np.roll(img1,int(yoff),1),int(xoff),0)
    
        # stack them and make that the first image
        cut1 = (cut3 + cut2)
    
        # stack full image (no normalization)
        img1 = (img3 + img2)
    
        # save shifts for overall stacks 
        dy1.append(np.int(yoff))
        dx1.append(np.int(xoff)) 
    # save cut imagestack 
    final_cut = cut1/len(img_list)
    outfile_cut = path + 'whale_algorithm3_cut.fits' 

    hdu = fits.PrimaryHDU(final_cut)
    hdu.writeto(outfile_cut, overwrite=True)
        
    print('Saved Cut: ' + outfile_cut)
    # save full image stack 
    final_image = img1/len(img_list)

    outfile = path + outfile
    hdu = fits.PrimaryHDU(final_image)
    hdu.writeto(outfile, overwrite=True)
    print('Saved Stack: ' + outfile) 
    
    return final_image

def cross_correlation_shifts(image1, image2, maxoff=None):
    ''' 
	Use cross-correlation and a 2nd order taylor expansion to measure 
        the offset between two images. Given two images, calculate the 
        amount image2 is offset from image1 to sub-pixel accuracy using 
        2nd order taylor expansion. 
        
        This is a simplified version of the cross_correlation_shifts function in        the package: https://github.com/keflavich/image_registration
        
        It borrows HEAVILY from that code. 
        
        
        Parameters
        ----------
        image1 (np.ndarray): The reference image  
        image2 (np.ndarray): The offset image (with same shape)
        maxoff (int): Maximum allowed offset (in pixels). 
        Use for low s/n images that you know are reasonably well-aligned
        
        Returns 
        -------
        yoff (float): yoffset between image1 and image 2
        xoff (float): xoffset between image1 and image 2
        
        
        Example
        -------
        >>> yoff,xoff = image_registration.cross_correlation_shifts(im1,im2)
        >>> im1_aligned_to_im2 = np.roll(np.roll(im1,int(yoff),1),int(xoff),0)
    '''
    
    # check that the images inputs are the same shape 
    if not image1.shape == image2.shape:
        raise ValueError("Images must have same shape.")
        
    # fast fourier transform and inverse fast fourier transform functions
    # don't really need to specify...these are the astropy default
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    
    # astropy fft convolution 
    ccorr = convolve(np.conjugate(image1), image2[::-1, ::-1], \
	             normalize_kernel=False,fftn=fftn, ifftn=ifftn,\
                     boundary='wrap', nan_treatment='fill')
    
    # get center 
    ylen,xlen = image1.shape
    xcen = xlen/2-(1-xlen%2)
    ycen = ylen/2-(1-ylen%2)

    # limit the maximum offset 
    if maxoff is not None:
        subccorr = ccorr[ycen-maxoff:ycen+maxoff+1,xcen-maxoff:xcen+maxoff+1]
        ymax,xmax = np.unravel_index(subccorr.argmax(), subccorr.shape)
        xmax = xmax+xcen-maxoff
        ymax = ymax+ycen-maxoff
    
    # or not 
    else:
        ymax,xmax = np.unravel_index(ccorr.argmax(), ccorr.shape)
        subccorr = ccorr
   

    xshift_int = xmax-xcen
    yshift_int = ymax-ycen
    local_values = ccorr[ymax-1:ymax+2,xmax-1:xmax+2]

    d1y,d1x = np.gradient(local_values)
    d2y,d2x,dxy = second_derivative(local_values)
    fx,fy,fxx,fyy,fxy = d1x[1,1],d1y[1,1],d2x[1,1],d2y[1,1],dxy[1,1]

    shiftsubx=(fyy*fx-fy*fxy)/(fxy**2-fxx*fyy)
    shiftsuby=(fxx*fy-fx*fxy)/(fxy**2-fxx*fyy)
    xshift = -(xshift_int+shiftsubx)
    yshift = -(yshift_int+shiftsuby)
    
    
    return xshift, yshift  # reversed wrt image (see example)

def second_derivative(image):
    '''   
       Compute the second derivative of an image. The derivatives are set to 
       zero at the edges.Copied from image registration package!
    
       Parameters
       ----------
       image (np.ndarray): 
    
       Returns
       -------
       d/dx^2 (np.ndarray): 2nd derivative wrt x
       d/dy^2 (np.ndarray): 2nd derivative wrt y
       d/dxdy (np.ndarray): 2nd derivative wrt x and y
       All three have the same shape as image.
    ''' 
    
    shift_right = np.roll(image,1,1)
    shift_right[:,0] = 0
    shift_left = np.roll(image,-1,1)
    shift_left[:,-1] = 0
    shift_down = np.roll(image,1,0)
    shift_down[0,:] = 0
    shift_up = np.roll(image,-1,0)
    shift_up[-1,:] = 0

    shift_up_right = np.roll(shift_up,1,1)
    shift_up_right[:,0] = 0
    shift_down_left = np.roll(shift_down,-1,1)
    shift_down_left[:,-1] = 0
    shift_down_right = np.roll(shift_right,1,0)
    shift_down_right[0,:] = 0
    shift_up_left = np.roll(shift_left,-1,0)
    shift_up_left[-1,:] = 0
    dxx = shift_right+shift_left-2*image
    dyy = shift_up   +shift_down-2*image
    dxy=0.25*(shift_up_right+shift_down_left-shift_up_left-shift_down_right)

    return dxx,dyy,dxy


__all__ = ['fft_shift', 'roll_shift', 'cross_correlation_shifts']

def FITSlist_stack(path, img_list, track = True): 
    ''' 
    Stack a list of FITS images using either the sub-pixel fft_shift or 
        the integer roll_shift. When images are drifting consistently, 
        there is the option to track the cut region using the last 
    calculated offset. 

    Parameters
    ----------
     '''



def fft_shift(): 
    ''' need to add this function''' 
    return 0
 
def roll_shift(img1, img2, center, box, maxoff=None, DEBUG=False): 
    '''      
        Calculate offset between images and create a stack using 
        integer shifts (np.roll).

    Usually the match should be done using a cut of the full 
        image to improve speed and accuracy (select region w/ good S/N).         

        Parameters
        ----------
        img1 (np.ndarray): reference image (usually better signal)
    img2 (np.ndarray): comparison image 
        center (np.array): the center of the cut image [c1, c2] 
        box (np.array): the shape of the cut image [len1, len2] 
    maxoff (int): Optional maximum allowed offset (in pixels). 
    Use for low s/n images that you know are reasonably well-aligned 
    DEBUG (boolean): set this to True to make debugging plots

        Returns 
        -------
        xoff (double): x subpixel offset between images 
    yoff (double) y subpixel offset between images
        final_image (np.ndarray): final stacked image (not averaged)
    ''' 


    # cut the small comparison area
    cut1= img1[center[1]-box[1]:center[1]+box[1]+1,\
               center[0]-box[0]:center[0]+box[0]+1]

    cut2= img2[center[1]-box[1]:center[1]+box[1]+1, \
               center[0]-box[0]:center[0]+box[0]+1]


    # get shift 
    yoff, xoff = cross_correlation_shifts(cut1, cut2, maxoff = maxoff)
        
    # shift first image to second and add
    cut1 = np.roll(np.roll(cut1,int(yoff),1),int(xoff),0) + cut2
    img1 = np.roll(np.roll(img1,int(yoff),1),int(xoff),0) + img2
    

    if DEBUG == True:
        shape = np.shape(cut1)
        aspect = shape[0]/shape[1]
        width = np.float(shape[1])/np.float(np.shape(img1)[1])
        width = np.int(100*width)
        if width < 10: 
            width = 10
        if width > 30:
            width = 30

        fig = plt.figure(figsize = (width, width*aspect/3))
        gs = gridspec.GridSpec(ncols=3, nrows=1, figure=fig)
        ax1 = fig.add_subplot(gs[0])
        p1 = ax1.imshow(np.log10(cut1))
        ax1.set_title('cut1_to_cut2') 
        ax2 = fig.add_subplot(gs[1])
        p2 = ax2.imshow(np.log10(cut2))
        ax2.set_title('cut2')
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(np.log10(cut1-cut2))
        ax3.set_title('residual')
        
        print('DEBUG: Calculated offset (dx, dy) =  (' \
              + str(np.round(xoff,2)) + ',' + str(np.round(yoff,2)) +')')
        plt.tight_layout()
        plt.savefig('debug.png')
        plt.show()
    
    return xoff, yoff, img1
         

def cross_correlation_shifts(image1, image2, maxoff=None, mask = None):
    ''' 
    Use cross-correlation and a 2nd order taylor expansion to measure 
        the offset between two images. Given two images, calculate the 
        amount image2 is offset from image1 to sub-pixel accuracy using 
        2nd order taylor expansion. 
        
        This is a simplified version of the cross_correlation_shifts function in        the package: https://github.com/keflavich/image_registration
        
        It borrows HEAVILY from that code. 
        
        
        Parameters
        ----------
        image1 (np.ndarray): The reference image  
        image2 (np.ndarray): The offset image (with same shape)
        maxoff (int): Maximum allowed offset (in pixels). 
        Use for low s/n images that you know are reasonably well-aligned
        mask (np.ndarray): the mask if using cosmic ray rejection 

        Returns 
        -------
        yoff (float): yoffset between image1 and image 2
        xoff (float): xoffset between image1 and image 2
        
        
        Example
        -------
        >>> yoff,xoff = image_registration.cross_correlation_shifts(im1,im2)
        >>> im1_aligned_to_im2 = np.roll(np.roll(im1,int(yoff),1),int(xoff),0)
    '''
    
    # check that the images inputs are the same shape 
    if not image1.shape == image2.shape:
        raise ValueError("Images must have same shape.")
        
    # fast fourier transform and inverse fast fourier transform functions
    # don't really need to specify...these are the astropy default
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    
    # astropy fft convolution 
    ccorr = convolve(np.conjugate(image1), image2[::-1, ::-1], \
                 normalize_kernel=False,fftn=fftn, ifftn=ifftn,\
                     boundary='wrap', nan_treatment='fill', mask = mask)
    
    # get center 
    ylen,xlen = image1.shape
    xcen = int(xlen/2-(1-xlen%2))
    ycen = int(ylen/2-(1-ylen%2))

    # limit the maximum offset 
    if maxoff is not None:
        subccorr = ccorr[ycen-maxoff:ycen+maxoff+1,xcen-maxoff:xcen+maxoff+1]
        ymax,xmax = np.unravel_index(subccorr.argmax(), subccorr.shape)
        xmax = xmax+xcen-maxoff
        ymax = ymax+ycen-maxoff
    
    # or not 
    else:
        ymax,xmax = np.unravel_index(ccorr.argmax(), ccorr.shape)
        subccorr = ccorr
   

    xshift_int = xmax-xcen
    yshift_int = ymax-ycen

    local_values = ccorr[ymax-1:ymax+2,xmax-1:xmax+2]

    d1y,d1x = np.gradient(local_values)
    d2y,d2x,dxy = second_derivative(local_values)

    fx,fy,fxx,fyy,fxy = d1x[1,1],d1y[1,1],d2x[1,1],d2y[1,1],dxy[1,1]

    shiftsubx=(fyy*fx-fy*fxy)/(fxy**2-fxx*fyy)
    shiftsuby=(fxx*fy-fx*fxy)/(fxy**2-fxx*fyy)

    xshift = -(xshift_int+shiftsubx)
    yshift = -(yshift_int+shiftsuby)
    
    
    return xshift, yshift  # reversed wrt image (see example)

def second_derivative(image):
    '''   
       Compute the second derivative of an image. The derivatives are set to 
       zero at the edges.Copied from image registration package!
    
       Parameters
       ----------
       image (np.ndarray): 
    
       Returns
       -------
       d/dx^2 (np.ndarray): 2nd derivative wrt x
       d/dy^2 (np.ndarray): 2nd derivative wrt y
       d/dxdy (np.ndarray): 2nd derivative wrt x and y
       All three have the same shape as image.
    ''' 
    
    shift_right = np.roll(image,1,1)
    shift_right[:,0] = 0
    shift_left = np.roll(image,-1,1)
    shift_left[:,-1] = 0
    shift_down = np.roll(image,1,0)
    shift_down[0,:] = 0
    shift_up = np.roll(image,-1,0)
    shift_up[-1,:] = 0

    shift_up_right = np.roll(shift_up,1,1)
    shift_up_right[:,0] = 0
    shift_down_left = np.roll(shift_down,-1,1)
    shift_down_left[:,-1] = 0
    shift_down_right = np.roll(shift_right,1,0)
    shift_down_right[0,:] = 0
    shift_up_left = np.roll(shift_left,-1,0)
    shift_up_left[-1,:] = 0

    dxx = shift_right+shift_left-2*image
    dyy = shift_up   +shift_down-2*image
    dxy=0.25*(shift_up_right+shift_down_left-shift_up_left-shift_down_right)

    return dxx,dyy,dxy


def cr_rejection(image, filtering_type = 's',s_level = 2.5, lf_level = 2.,
                 gain = 2.2, read_noise = 5., second_pass = False, 
                 second_s_level = 2., second_lf_level = 1., DEBUG = False, 
                 plot_box = None):
    """
    Attempts to identify cosmic rays in image based on laplacian edges,
    using deviations from expected sampling noise and contrast with
    structures in the image

    Parameters:
    -----------
    image (np.ndarray): single image
    filtering_type (string): select which selection parameters to use
    s_level (float): filter level of S = Laplacian/ Noise for tagging cosmic rays
    lf_level(float): filter level of Laplacian/Fine Structure for tagging cosmic rays
    gain (float): detector gain value
    read_noise (float): detector read noise
    second_pass (bool): after a first pass, search for more cosmic rays in 
        surrounding pixels with a lower threshold values
    second_s_level(float):
    second_lf_level(float):
    DEBUG (bool): show 6 image panel
    plot_box (x0,xf,y0,yf): cut of image shown in DEBUG plots
    
    
    Returns:
    -------

    mask (np.ndarray): array where candidate pixels for cosmic rays are marked
    with value 0
    subtracted_img: original image, with pixels tagged in mask replaced by 5x5 
    median filter. 

    """



    s_filter = False
    lf_filter = False
    if filtering_type ==  "s" or filtering_type == "sf":
        s_filter = True
    if filtering_type ==  "f" or filtering_type == "sf":
        lf_filter = True


    (n,m) = image.shape
    if plot_box == None:
        plot_box = (0,n-1,0,m-1)
    #subsampling image
    img2 = np.ndarray(shape=(image.shape[0]*2,image.shape[1]*2), dtype=float)
    
    img2 = ndimage.zoom(image, 2, order=0)
    #for i in range(img2.shape[0]):
    #    for j in range(img2.shape[1]):
    #        img2[i][j]  = image[int((i)/2)][int((j)/2)]

    laplace_kernel =  [[0,-0.25,0], 
                        [-0.25,1,-0.25],
                        [0, -0.25, 0]]

    del2_img = convolve(img2, laplace_kernel, normalize_kernel = False, \
                     boundary = 'wrap', nan_treatment = 'fill')


    #setting negative values of L to 0

    idx_zero = np.where(del2_img < 0)
    del2_img[idx_zero] = 0



    #for i in range(del2_img.shape[0]):
    #    for j in range(del2_img.shape[1]):
    #        if del2_img[i][j] < 0:
    #            del2_img[i][j] = 0
    

    
    resampled_L  = np.ndarray(shape = (image.shape[0],image.shape[1]),
     dtype = float)

    #### Resampling back to original image size
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            resampled_L[i][j] = 0.25 * (del2_img[2*i][2*j] + 
                                del2_img[2*i][2*j+1] + del2_img[2*i+1][2*j] 
                                + del2_img[2*i+1][2*j+1])

    ## Noise profile 
    
    median_5 =  ndimage.median_filter(image,5)
    n_profile = ((gain *  median_5 + read_noise**2) ** (1/2)) / gain

    # Estimate deviations from expected poisson fluctuations
    S = resampled_L / (2 * n_profile)

    S_median = ndimage.median_filter(S,5)

    # Remove sampling flux
    S_prime = S - S_median

    structure_contrast = np.zeros(image.shape)
    #Fine structure image - looking for symmetric point sources
    if lf_filter == True:
        img_med3 = ndimage.median_filter(image,3)

        large_structure  = ndimage.median_filter(img_med3,7)

        F_structure = img_med3 - large_structure

        for i in range(F_structure.shape[0]):
            for j in range(F_structure.shape[1]):
                if F_structure[i][j] < 1:
                    F_structure[i][j] = 1

        structure_contrast  = resampled_L / F_structure

    mask_1st = np.zeros(image.shape)
    
    

    
    if s_filter: 
        idx_s = np.where(S_prime > s_level)
        mask_1st[idx_s] +=1.
        idx = idx_s

    if lf_filter:
        idx_lf = np.where(structure_contrast > lf_level)
        mask_1st[idx_lf] += 1.
        idx = idx_lf

    if s_filter and lf_filter:
        idx_both = np.where(mask_1st == 2.)
        mask_1st.fill(0)
        mask_1st[idx_both] = 1
        idx = idx_both


 
    
    print("%s cosmic rays found" % len(idx[0]))
    mask_2nd = mask_1st.copy()

    second_s_values = []
    second_lf_values = []

    

    if second_pass:
        x_indices = idx[0]
        y_indices = idx[1]
        for i in range(len(idx[0])):
            cr_x = idx[0][i]
            cr_y = idx[1][i]


            s_box = S_prime[cr_x-1:cr_x+1,cr_y-1:cr_y+1]
            lf_box = structure_contrast[cr_x-1:cr_x+1,cr_y-1:cr_y+1]
            if(DEBUG):
                second_s_values.extend(s_box.flatten())
                second_lf_values.extend(lf_box.flatten())

            local_mask = np.zeros((3,3))
            if s_filter: 
                idx_s2 = np.where(s_box > s_level)
                local_mask[idx_s2] += 1
                idx_local = idx_s2


            if lf_filter:
                idx_lf2 = np.where(lf_box > lf_level)
                local_mask[idx_lf2] += 1
                idx_local = idx_lf2

            if s_filter and lf_filter:
                idx_both2 = np.where(local_mask == 2.)
                local_mask.fill(0)
                local_mask[idx_both2] = 1
                idx_local = idx_both2

            x_indices = np.append(x_indices,idx_local[0] - 1 + cr_x)
            y_indices=  np.append(y_indices,idx_local[1] - 1 + cr_y)

        idx = (x_indices,y_indices)
        mask_2nd[idx] = 1
        print("%s cosmic rays found (second pass)" % len(idx[0]))

    



    
    subtracted_img  = image.copy()
    subtracted_img[idx] = median_5[idx]

    if DEBUG == True:

        x0,xf,y0,yf = plot_box
        space = 0
        
        width = 15

        if s_filter and lf_filter:

            fig = plt.figure(figsize = (width, (2*width)/3))
            gs = gridspec.GridSpec(ncols = 3, nrows = 2, figure = fig)
        else:
            fig = plt.figure(figsize = (width, width))
            gs = gridspec.GridSpec(ncols = 2, nrows = 2, figure = fig)


        ax1 = fig.add_subplot(gs[space])
        space += 1
        p1 = ax1.imshow(image[x0:xf,y0:yf],vmin = 0, vmax = 200)
        ax1.set_title('Original') 
        #ax2 = fig.add_subplot(gs[1])
        #p2 = ax2.imshow(resampled_L[x0:xf,y0:yf], vmin = 0,vmax = 300)
        #ax2.set_title('$L^+$')
        if s_filter:
            ax2 = fig.add_subplot(gs[space])
            space += 1
            ax2.imshow(S_prime[x0:xf,y0:yf], vmin = 0, vmax = 5)
            ax2.set_title('$S^\\prime$')

        if lf_filter:
            ax3 = fig.add_subplot(gs[space])
            space += 1
            ax3.imshow(F_structure[x0:xf,y0:yf],vmin = 0, vmax = 100)
            ax3.set_title('$F$')
            ax4 = fig.add_subplot(gs[space])
            space += 1
            ax4.imshow(structure_contrast[x0:xf,y0:yf], vmin = 0, vmax = 50)
            ax4.set_title('$L/F$')
        
        ax5 = fig.add_subplot(gs[space])
        space += 1
        ax5.imshow(mask_2nd[x0:xf,y0:yf],vmin = 0, vmax = 1)
        ax5.set_title('Cosmic Rays')
            



        if s_filter:
            ax6 = fig.add_subplot(gs[space])
            space += 1
            ax6.imshow(subtracted_img[x0:xf,y0:yf],vmin = 0 , vmax = 200 )
            ax6.set_title('Corrected Image')
        plt.tight_layout()
        plt.savefig('cr_debug.png')
        plt.show()

        fig2 = plt.figure(figsize = (width, width))
        space2 = 0
        if (second_pass):
            rows = 2
        else:
            rows = 1
            
        if s_filter and lf_filter:
            cols = 2
        else:
            cols = 1
        
        gs2 = gridspec.GridSpec(ncols = cols, nrows = rows, figure = fig2)
        if s_filter:
            ax21 = fig2.add_subplot(gs2[space2])
            space2 += 1
            hist1 = ax21.hist(S_prime.flatten(), range=(0,20), bins = 40, log = True)
            ax21.set_title('$S^\\prime$') 
        if lf_filter:
            ax22 = fig2.add_subplot(gs2[space2])
            space2 += 1
            hist2 = ax22.hist(structure_contrast.flatten(),range=(0,20), bins = 40, log=True)
            ax22.set_title('$L^+ / F$')
        if(second_pass):
            if s_filter:
                ax23 = fig2.add_subplot(gs2[space2])
                space2 += 1
                hist3 = ax23.hist(second_s_values,range=(0,20), bins = 40, log=True)
                ax23.set_title('$S ^\\prime$ (2nd Pass)')
            if lf_filter: 
                ax24 = fig2.add_subplot(gs2[space2])
                space2 += 1
                hist4 = ax23.hist(second_lf_values,range=(0,20), bins = 40, log=True)
                ax24.set_title('$L^+ / F$ (2nd Pass)') 
        plt.tight_layout()
        plt.savefig('2nd_pass_debug.png')
        plt.show()
        
    mask_2nd = 1 - mask_2nd           


    return mask_2nd , subtracted_img



# hdul = fits.open(directory + img_list[0])
#     img1_cr = hdul[0].data
mask1, subtracted1 = cr_rejection(ds9, s_level = 2.5 ,gain = 2.2, read_noise = 5)
ds9 *= mask1
