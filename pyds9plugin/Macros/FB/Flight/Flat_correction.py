from astropy.io import fits
from pyds9plugin.Macros.FB.Flight.OS_correction import ApplyOverscanCorrection


ds9, _ = ApplyOverscanCorrection(image=ds9,ColumnCorrection=False)


flat = fits.open("/Users/Vincent/Nextcloud/LAM/Work/EMCCD_reduction/factor_multiplication_big_large_regions.fits")[0].data
# if ds9.shape[1]>2000: # still need to modify this image using the same method as the small one 
ds9 = (6 * (flat[:,:ds9.shape[1]]-1)+1) * ds9
# else:
#     ds9 = 4 * ds9 * fits.open("/Users/Vincent/Nextcloud/LAM/Work/EMCCD_reduction/factor_multiplication.fits")[0].data
    
    