from astropy.io import fits
import numpy as np
from astropy import wcs
from astropy.table import Table
path ="/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D11/20220504/image_00000163.fits"
hdulist=fits.open(path)
w = wcs.WCS(naxis=2)
w.wcs.crpix =[1, 1]              
w.wcs.cdelt = np.array([-0.0235, 0.0235])
w.wcs.crval = [0,0] #RA and dec values in hours and degrees    
w.wcs.ctype =["RA---TAN", "DEC--TAN"]
#w.wcs.set_pv([(180, NSGP, float(SCALE))])
hdulist[0].header.update(w.to_header())
hdulist.writeto(path.replace(".fits","_.fits"))


from astropy.wcs.utils import fit_wcs_from_points
https://stackoverflow.com/questions/69024765/using-astropy-fit-wcs-from-points-to-give-fits-file-a-new-wcs


slits = Table.read("/Users/Vincent/Github/pyds9fb/pyds9fb/Slits/F2_-161.csv")
detector_position = np.array([slits['X_IMAGE'], [slits['Y_IMAGE']])

mask_position = slits['xmask'],slits['ymask']

#  SkyCoord([(125.66419083, -42.96809252), (125.67730695, -42.98209958),
#                          (125.65082259, -42.9914015), (125.6611325, -43.01438513), (125.70471982, -43.01228167)],
#                         frame="icrs", unit="deg")

w = fit_wcs_from_points(xy = detector_position, world_coords = mask_position, projection='TAN')
hdulist[0].header.update(w.to_header())
hdulist.writeto('Swirl06p_1WCS.fits')
hdulist.close()