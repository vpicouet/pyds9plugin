from scipy import signal  # from scipy import misc
region = getregion(d, quick=True,message=False,selected=True)
if region is not None:
    Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
    area = [Yinf, Ysup, Xinf, Xsup]
else:
    area = [0, -1, 0, -1]

ds9 = ds9[area[0] : area[1], area[2] : area[3]]

image1 = np.copy(ds9 - ds9.mean()).astype('float64')
image2 = np.copy(ds9 - ds9.mean()).astype('float64')

image1 = np.copy(ds9 - ds9.min()).astype('float64')/100
image2 = np.copy(ds9 - ds9.min()).astype('float64')/100

# Type = "2d-xy"
Type = "x"
# Type = "y"
# Type="fft"
if Type.lower() == "2d-xy":
    ds9 = signal.correlate2d(image1,image2,boundary='symm',mode='same')
elif Type.lower() == "x":
    ds9 = np.zeros(image1.shape)
    for i in range(image1.shape[0]):
        ds9[i, :] = signal.correlate(image1[i, :], image2[i, :], mode="same")  # / 128
elif Type.lower() == "y":
    ds9 = np.zeros(image1.shape)
    for i in range(image1.shape[1]):
        ds9[:, i] = signal.correlate(image1[:, i], image2[:, i], mode="same")  # / 128
elif Type.lower() == "fft":
    # ds9 = np.abs(fftpack.fftshift(fftpack.fft2(ds9)))**2
    ds9 = np.fft.irfft2(np.fft.rfft2(ds9) * np.conj(np.fft.rfft2(ds9)))

#ds9=correlate2d(image.astype('float64'),image,boundary='symm',mode='same')



#from scipy import fftpack
#ds9=fftpack.fftshift(fftpack.fft2(ds9.astype('float64')))
#ds9=correlate2d(ds9.astype('float64'),ds9,boundary='symm',mode='same')
#ds9=np.abs(fftshift(fft2(ds9)))**2


# *    ds9=ds9[:100,:100]                           -> Trim the image
# *    ds9+=np.random.normal(0,0.1,size=ds9.shape)  -> Add noise to the image
# *    ds9[ds9>2]=np.nan                            -> Mask a part of the image
# *    ds9=1/ds9, ds9+=1, ds9+=1                    -> Different basic expressions
# *    ds9=convolve(ds9,np.ones((1,9)))[1:-1,9:-9]  -> Convolve unsymetrically"
# *    ds9+=np.linspace(0,1,ds9.size).reshape(ds9.shape) -> Add background
# *    ds9+=30*(ds9-gaussian_filter(ds9, 1))        -> Sharpening
# *    ds9=np.hypot(sobel(ds9,axis=0,mode='constant'),sobel(ds9,axis=1,mode='constant')) -> Edge Detection
# *    ds9=np.abs(fftshift(fft2(ds9)))**2           -> FFT
# *    ds9=correlate2d(ds9.astype('uint64'),ds9,boundary='symm',mode='same') -> Autocorr
# *    ds9=interpolate_replace_nans(ds9, Gaussian2DKernel(x_stddev=5, y_stddev=5)) -> Interpolate NaNs
