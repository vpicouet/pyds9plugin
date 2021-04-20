from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
stddev = 1
while ~np.isfinite(ds9).all():
    verboseprint(len(np.where(~np.isfinite(ds9))[0]), "NaN values found!")
    verboseprint("Infinite values in the image, inteprolating NaNs with 2D Gaussian kernel of standard deviation = ", stddev)
    kernel = Gaussian2DKernel(x_stddev=stddev, y_stddev=stddev)
    ds9 = interpolate_replace_nans(ds9, kernel)  # .astype('float16')
    stddev += 1
