import os
import numpy as np

def ComputeOSlevel1(
    image,
    OSR1=[20, -20, 20, 1053 - 20],
    OSR2=[20, -20, 2133 + 20, -20],
    lineCorrection=True,
):
    """Apply overscan correction line by line using overscan region
    """  # ds9 -=  np.nanmedian(image[:,2200:2400],axis=1)[..., np.newaxis]*np.ones((ds9.shape))
    import numpy as np

    if OSR2 is not None:
        OSregion = np.hstack((image[:, OSR1[2] : OSR1[3]], image[:, OSR2[2] : OSR2[3]]))
    else:
        OSregion = image[:, OSR1[2] : OSR1[3]]
    if lineCorrection:
        OScorrection = np.nanmedian(OSregion, axis=1)
        # reject_outliers(OSregion, stddev=3))
        OScorrection = OScorrection[..., np.newaxis] * np.ones((image.shape))
    else:
        OScorrection = np.nanmedian(OSregion)  # reject_outliers(OSregion, stddev=3))
    return OScorrection


def ApplyOverscanCorrection(
    path=None,
    fitsimage=None,
    image=None,
    stddev=3,
    OSR1=[20, -20, 20, 1053 - 20],
    OSR2=[20, -20, 2133 + 20, -20],
    save=False,
    lineCorrection=True,
    ColumnCorrection="ColumnByColumn",
):
    """Apply overscan correction line by line using overscan region1234
    """
    from astropy.io import fits

    if path is not None:
        fitsimage = fits.open(path)
        image = fitsimage[0].data.astype(float)  # .copy()
    elif fitsimage is not None:
        # path = fitsimage.filename()
        image = fitsimage[0].data.astype(float)  # .copy()
    else:
        path = "/tmp/test.fits"
    ###############################################

    OScorrection = ComputeOSlevel1(
        image, OSR1=OSR1, OSR2=OSR2, lineCorrection=lineCorrection
    )
    # fits.setval(path, "OVS_CORR", value=lineCorrection)
    new_im = image - OScorrection
    if ColumnCorrection == "ColumnByColumn":
        median = np.nanmedian(new_im, axis=0)
        colcorr = median[np.newaxis, ...] * np.ones((image.shape))
        colcorr2 = np.convolve(median, np.ones(2) / 2, mode="same")[
            np.newaxis, ...
        ] * np.ones((image.shape))
        new_im -= colcorr - 3000 - colcorr2
    name = os.path.join(
        os.path.dirname(path) + "/OS_corrected/%s" % (os.path.basename(path))
    )

    if save:

        fitsimage[0].data = new_im
        fitswrite(fitsimage[0], name)
    return new_im, name
if __name__ == "__main__":
    ds9, _ = ApplyOverscanCorrection(image=ds9, ColumnCorrection=False)