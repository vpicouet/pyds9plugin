from astropy.io import fits
import re
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from pyds9plugin.Macros.FB.Flight.CR_masking import *
from pyds9plugin.Macros.FB.Flight.OS_correction import *
from pyds9plugin.Macros.FB.Flight.photon_counting import *
from pyds9plugin.Macros.Interpolate_NaNs import *



if __name__ == "__main__":
    for name in ["/CosmicRayFree","/OS_subtracted","/PC_images"]:
        if os.path.exists(os.path.dirname(filename) + name) is False:
            os.mkdir(os.path.dirname(filename) +name)
    new_path = "/tmp/" + os.path.basename(filename)


    CR_removed = CR_masking(filename,ds9, n=3, area=[0,-1,0,-1],threshold = 40000)    
    fitsimage.data = CR_removed
    fitsimage.writeto(os.path.dirname(filename) + "/CosmicRayFree/" +     os.path.basename(filename).replace(".fits",  "_CR.fits"),overwrite=True)

    # CR_removed = interpolate_nans(CR_removed)
    # fitsimage.data = CR_removed
    # fitsimage.writeto(os.path.dirname(filename) + "/CosmicRayFree/" +     os.path.basename(filename).replace(".fits",  "_CR_int.fits"),overwrite=True)

    OS_subtracted, _ = ApplyOverscanCorrection(image=CR_removed, ColumnCorrection=False,save=True)
    fitsimage.data = OS_subtracted
    fitsimage.writeto(os.path.dirname(filename) + "/OS_subtracted/" + os.path.basename(filename).replace(".fits",  "_OS.fits"),overwrite=True)

    pc_image = DS9photo_counting(image=OS_subtracted, header=header, filename=filename, threshold=5.5)
    fitsimage.data = pc_image
    fitsimage.writeto(os.path.dirname(filename) + "/PC_images/" + os.path.basename(filename).replace(".fits",  "_PC.fits"),overwrite=True)


