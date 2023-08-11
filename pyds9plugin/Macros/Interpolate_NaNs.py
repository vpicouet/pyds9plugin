from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

def interpolate_nans(ds9, STD_DEV = 1,dtype=float,RN_std=0):
    mask = np.isfinite(ds9)
    while ~np.isfinite(ds9).all():
        print(STD_DEV)
        kernel = Gaussian2DKernel(x_stddev=STD_DEV, y_stddev=STD_DEV)
        ds9 = interpolate_replace_nans(ds9, kernel)
        STD_DEV += 2
    if RN_std>0:
        RN = np.zeros(ds9.shape)
        RN[~mask] = RN_std
        print(RN)
        print(RN,RN.min(),RN.max())
        ds9 = ds9 + np.random.normal(0, RN)
    return np.array(ds9, dtype=dtype) # no need for # header["BITPIX"] = 16


if __name__ == "__main__":
    # ds9 = interpolate_nans(ds9, STD_DEV=4, dtype=np.int16,RN_std=56)
    new_path = filename.replace(".fits","noNaN.fits")
    ds9 = interpolate_nans(ds9, STD_DEV=4,RN_std=0)
    