
def cropCLAUDS(path, position=[0, 0], size=[10, 10], all_ext=False):  # ,area=[0,100,0,100]
    """Cropping/Trimming function that keeps WCS header information
    """
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.io.fits import ImageHDU
    a = fits.open(path)
    b = a.copy()
    for i in range(1):
        try:
            di = Cutout2D(a[i].data, position=position, size=size, wcs=WCS(a[i].header))
            if i == 0:
                a[i] = fits.PrimaryHDU(data=di.data, header=di.wcs.to_header())
            else:
                a[i] = ImageHDU(data=di.data, header=di.wcs.to_header())
            # a[i].header["CD1_1"] = b[i].header["CD1_1"]
            # a[i].header["CD2_2"] = b[i].header["CD2_2"]
        except (ValueError, IndexError) as e:
            verboseprint(i, e)
            pass
    a.writeto(path[:-5] + "_trim.fits", overwrite=True)
    return a, path[:-5] + "_trim.fits"

system = 'Image'
region = getregion(d, quick=True, selected=True, system=system, dtype=float)
Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region[0], dtype=float)
if (len(region[0]) != 5) & (len(region[0]) != 4):
    message(d, "Trimming only works on box regions. Create and select a box region an d re-run the analysis.")
    sys.exit()
else:
    ds9_2, name = cropCLAUDS(path=getfilename(d), position=[region[0][0] - 1, region[0][1] - 1], size=np.array([region[0][3], region[0][2]], dtype=int), all_ext=False)
d.set("frame new ; tile yes ; file %s" % (name))
