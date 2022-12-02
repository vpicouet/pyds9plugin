
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
ds9[:,283]=np.nan
ds9[ds9>40000]=np.nan
x, y = np.indices(ds9.shape)

for xy, i in zip([[490,530,952,973],[490,530,952,973],[490,530,952,973]],[30,30,30][:1]):
    Xinf, Xsup, Yinf, Ysup = xy
    mask =  (x > Xinf) & (x < Xsup + 1) & (y > Yinf) & (y < Ysup + 1)
    masked_data = ds9[mask].flatten() 
    masked_data.sort()
    value = masked_data[i]
    ds9[(mask) & (ds9>value)] =  np.nan


# ds9 -= fits.open("/Users/Vincent/Pictures/chas13_nov22/Bias/stack_nanmean_162-172.fits")[0].data   #
ds9-=np.nanmin(ds9[10:-10,10:-10])
ds9=ds9[:,47:1068]
ds9*=3.47/(header["EXPTIME"]+0.1) / 0.85 / 0.85

STD_DEV = 1
while ~np.isfinite(ds9).all():
    kernel = Gaussian2DKernel (x_stddev=STD_DEV, y_stddev=STD_DEV)
    ds9 = interpolate_replace_nans (ds9, kernel)
    STD_DEV += 1