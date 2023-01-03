from astropy.io import fits

if ds9.shape[1]>2000: # still need to modify this image using the same method as the small one 
    ds9 *= fits.open("/Users/Vincent/Nextcloud/LAM/Work/EMCCD_reduction/factor_multiplication_full.fits")
else:
    ds9 *= fits.open("/Users/Vincent/Nextcloud/LAM/Work/EMCCD_reduction/factor_multiplication.fits")