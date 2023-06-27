#%%
def build_wcs_header(
    xpapoint=None,
    filename=None,
    CRPIX=None,
    CDELT=None,
    CTYPE="RA---AIR,DEC--AIR",
    CRVAL=None,
    argv=[],
):
    """Creates a WCS header"""
    # Set the WCS information manually by setting properties of the WCS
    # object.
    from astropy import wcs
    from astropy.io import fits
    import numpy as np

    # parser = CreateParser(get_name_doc(), path=True)
    # args = parser.parse_args_modif(argv, required=True)

    # if CRPIX is None:
    #     CRPIX = np.array(sys.argv[3].split(","), dtype=float)
    #     CDELT = np.array(sys.argv[4].split(","), dtype=float)
    #     # unit ?
    #     CTYPE = np.array(sys.argv[5].split(","), dtype=str)
    #     CRVAL = np.array(sys.argv[6].split(","), dtype=float)

    # Create a new WCS object.  The number of axes must be set
    # from the start
    w = wcs.WCS(naxis=2)
    # Set up an "Airy's zenithal" projection
    # Vector properties may be set with Python lists, or Numpy arrays
    w.wcs.crpix = CRPIX  # [-234.75, 8.3393]
    w.wcs.cdelt = CDELT  # np.array([-0.066667, 0.066667])
    w.wcs.crval = CRVAL  # [0, -90]
    w.wcs.ctype = [str(CTYPE[0]), str(CTYPE[1])]
    print([str(CTYPE[0]), str(CTYPE[1])])
    # ["RA---AIR", "DEC--AIR"], ["RA---TAN", "DEC--TAN"]#
    # w.wcs.set_pv([(2, 1, 45.0)])
    # Three pixel coordinates of interest.
    # The pixel coordinates are pairs of [X, Y].
    # The "origin" argument indicates whether the input coordinates
    # are 0-based (as in Numpy arrays) or
    # coming from DS9).
    pixcrd = np.array([[0, 0], [24, 38], [45, 98]], dtype=np.float64)

    # Convert pixel coordinates to world coordinates.
    # The second argument is "origin" -- in this case we're declaring we
    # have 0-based (Numpy-like) coordinates.
    world = w.wcs_pix2world(pixcrd, 0)
    print(world)
    # Convert the same coordinates back to pixel coordinates.
    pixcrd2 = w.wcs_world2pix(world, 0)
    print(pixcrd2)

    # These should be the same as the original pixel coordinates, modulo
    # some floating-point error.
    assert np.max(np.abs(pixcrd - pixcrd2)) < 1e-6
    # The example below illustrates the use of "origin" to convert between
    # 0- and 1- based coordinates when executing the forward and backward
    # WCS transform.
    x = 0
    y = 0
    origin = 0
    assert w.wcs_pix2world(x, y, origin) == w.wcs_pix2world(x + 1, y + 1, origin + 1)

    # Now, write out the WCS object as a FITS header
    header = w.to_header()
    # header is an astropy.io.fits.Header object.  We can use it to create a new
    # PrimaryHDU and write it to a file.
    #    hdu = fits.PrimaryHDU(header=header)
    if filename is None:
        d = DS9n(xpapoint)
        filename = get_filename(d)
    for line in header.cards:

        fits.setval(filename, line[0], value=line[1], comment=line[2])
    return header, w


# param Create-WCS
# CRPIX entry {CRPIX} 0,0 {Pixel coordinate of reference point}
# CDELT entry {CDELT} 1,1 {Coordinate increment at reference point [in degrees]}
# CTYPE menu {CTYPE} RA---AIR,DEC--AIR|RA---TAN,DEC--TAN {Supported projections}
# CRVAL entry {CRVAL} 1,1 {Coordinate value at reference point [in degrees]}
# endparam
# w.wcs.crpix = [centerX, centerY]
# w.wcs.crval = [centerRA, centerDEC]
# w.wcs.cdelt = np.array([-0.0235, 0.0235])
# w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

a = yesno(d,"Do you want to use the on sky WCS? No will lead to autocoll WCS.")
if a:
    factor=1
else:
    factor=2
# 1.271 1.106
# filename = "/Volumes/GoogleDrive-105248178238021002216/.shortcut-targets-by-id/1ZgB7kY-wf7meXrq8v-1vIzor75aRdLDn/FIREBall-2/FB2_2022/Detector_Data/220621/CIT_NUVU_platescale/image000049.fits"
header_det_sky, w = build_wcs_header(
    xpapoint=None,
    filename=filename,
    CRPIX=[0, 0],
    # CDELT=[1.106 / 3600, 1.271 / 3600], #2018
    # CDELT=[1.3745 / 3600, 1.09612142 / 3600],  # 2022
    # CDELT=[1.3745 / 3600/factor, 1.09612142 / 3600/factor],  # 2022
    CDELT=[1.26 / 3600/factor, 1.08 / 3600/factor],  # 2023
    # CDELT=[1, 1],
    # CTYPE=["RA---AIR", "DEC--AIR"],
    CTYPE=["RA---TAN", "DEC--TAN"],
    # CTYPE="RA---???,DEC--???",
    CRVAL=[1, 1],
    argv=[],
)

header_det_mask, w = build_wcs_header(
    xpapoint=None,
    filename=filename,
    CRPIX=[0, 0],
    CDELT=[1.3745 * 11.7 / 1000, 1.09612142 * 11.7 / 1000],  # 2022
    CTYPE=["LINEAR", "LINEAR"],
    CRVAL=[1, 1],
    argv=[],
)


header_dispersion, w = build_wcs_header(
    xpapoint=None,
    filename=filename,
    CRPIX=[0, 0],
    CDELT=[0.021, 1e-5],  # 2022
    CTYPE=["LINEAR", "LINEAR"],
    CRVAL=[1, 1],
    argv=[],
)


# %%

#%%
for line in header_det_sky.cards:
    header.append((line[0], line[1]), end=True)
    fits.setval(filename,line[0],value=line[1],comment="sky WCS",)


header["INHERIT"] = "T"
header["EXTTYPE"] = "PHYSICAL"
for line in header_det_mask.cards:
    header.append((line[0] + "A", line[1]), end=True)
    fits.setval(filename,line[0] + "A",value=line[1],comment="Mask mm WCS",)
    # header[line[0] + "_A"] = line[1]

fits.setval(filename,"CUNIT1A",value="mm",comment="",)
fits.setval(filename,"CUNIT2A",value="mm",comment="",)

for line in header_dispersion.cards:
    header.append((line[0] + "B", line[1]), end=True)
    fits.setval(filename,line[0] + "B",value=line[1],comment="disperison WCS",)

header["CUNIT1B"] = "nm"
header["CUNIT2B"] = "nm"
fits.setval(filename,"CUNIT1B",value="mm",comment="",)
fits.setval(filename,"CUNIT2B",value="mm",comment="",)





print(repr(header))
# fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(ds9, header_det_sky)]).writeto("/tmp/test.fits")
fits.HDUList([fits.PrimaryHDU(ds9, header=header)]).writeto(
    "/tmp/test.fits", overwrite=True
)
d.set("regions shape Ruler")
d.set("file " + filename)
#%%











#%% Way to measure 

# FB2_2023/instrument_alignment_focusing/XY_calibration/FireBallPipe/mapping_mask_det_2022.y
# fit_magnfication(X,-Y,3600*EL*2,3600*CE*2,Y, field="",x0=(0,0,0,1,1))
