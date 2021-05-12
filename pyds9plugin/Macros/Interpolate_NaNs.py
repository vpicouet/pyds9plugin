from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

STD_DEV = 1
while ~np.isfinite(ds9).all():
    kernel = Gaussian2DKernel(x_stddev=STD_DEV, y_stddev=STD_DEV)
    ds9 = interpolate_replace_nans(ds9, kernel)
    STD_DEV += 1
