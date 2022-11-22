
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
ds9[:,283]=np.nan
ds9[ds9>40000]=np.nan
ds9=ds9[:,47:1068]
ds9-=np.nanmin(ds9[10:-10,10:-10])
ds9*=3.47/(header["EXPTIME"]+0.1) / 0.85 / 0.85

STD_DEV = 1
while ~np.isfinite(ds9).all():
    kernel = Gaussian2DKernel (x_stddev=STD_DEV, y_stddev=STD_DEV)
    ds9 = interpolate_replace_nans (ds9, kernel)
    STD_DEV += 1