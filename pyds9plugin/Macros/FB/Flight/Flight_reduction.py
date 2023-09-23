from astropy.io import fits
import re
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from pyds9plugin.Macros.FB.Flight.CR_masking import *
from pyds9plugin.Macros.FB.Flight.OS_correction import *
from pyds9plugin.Macros.FB.Flight.photon_counting import *
from pyds9plugin.Macros.FB.Flight.Visualization import *
from pyds9plugin.Macros.Interpolate_NaNs import *


CR = False
OS = True
PC = True



if __name__ == "__main__":

    name_CR = "/CosmicRayFree"
    name_OS = "/OS_subtracted"
    name_PC = "/PC_images"
    name_CR_image = os.path.dirname(filename) + name_CR + "/" +     os.path.basename(filename).replace(".fits",  "_CR.fits")
    name_OS_image = os.path.dirname(filename) + name_OS + "/" +  os.path.basename(filename).replace(".fits",  "_OS.fits")
    name_PC_image = os.path.dirname(filename)+ name_PC + "/" +  os.path.basename(filename).replace(".fits",  "_PC.fits")
    new_path = "/tmp/" + os.path.basename(filename)
    # if 1==0:
    if (os.path.exists(name_CR_image)) |   (os.path.exists(name_OS_image)) &  (os.path.exists(name_PC_image)):
        print(filename + " already processed.")
    else:
        for name, create in zip([name_CR, name_OS, name_PC],[CR,OS,PC]):
            if create:
                if os.path.exists(os.path.dirname(filename) + name) is False:
                    os.mkdir(os.path.dirname(filename) +name)
        if CR:
            CR_removed = CR_masking(filename,ds9, n=3, area=[0,-1,0,-1],threshold = 40000)    
            fitsimage.data = CR_removed
            fitsimage.writeto(name_CR_image,overwrite=True)

        VisualizationDetector3(fitsimage, filename, plot_flag=False, save=True, cbars=["fixed", "variable"], units=["counts"],emgain = 1500, conversion_gain = 1)
            # print("File saved : ", os.path.dirname(filename) + name_CR + "/" +     os.path.basename(filename).replace(".fits",  "_CR.fits"))
        # CR_removed = interpolate_nans(CR_removed)
        # fitsimage.data = CR_removed
        # fitsimage.writeto(os.path.dirname(filename) + "/CosmicRayFree/" +     os.path.basename(filename).replace(".fits",  "_CR_int.fits"),overwrite=True)
        if OS:
            # image = CR_removed if CR else ds9
            OS_subtracted, _ = ApplyOverscanCorrection(image=fitsimage.data, ColumnCorrection=False,save=True)
            fitsimage.data = OS_subtracted
            fitsimage.writeto(name_OS_image,overwrite=True)
            
        if PC:
            if OS:
                new_filename = os.path.dirname(filename) + name_OS + "/" + os.path.basename(filename).replace(".fits",  "_OS.fits") if OS else filename
            else:
                if CR:
                    new_filename = os.path.dirname(filename) + name_CR + "/" +     os.path.basename(filename).replace(".fits",  "_CR.fits")
                else:
                    new_filename = filename
            pc_image, fit_param = DS9photo_counting(image=fitsimage.data, header=header, filename=new_filename, threshold=5.5)
            fitsimage.data = pc_image
            fitsimage.writeto(name_PC_image,overwrite=True)

            fits.setval(filename, "hist_bias", value= fit_param["BIAS"])
            fits.setval(filename, "hist_ron", value= fit_param["RON"])
            fits.setval(filename, "hist_gain", value= fit_param["GAIN"])
            fits.setval(filename, "hist_flux", value= fit_param["FLUX"])
            fits.setval(filename, "FRAC5SIG", value= fit_param["FRAC5SIG"])  #/ float(table["EXPTIME"])

            if "EXPTIME" in list(dict.fromkeys(header.keys())):
                try:
                    # header["eperhour"] = fit_param["FLUX"] * 3600 / float(header["EXPTIME"])
                    fits.setval(filename, "eperhour", value= fit_param["FLUX"] * 3600 / float(header["EXPTIME"]))

                except ValueError:
                    # header["eperhour"] = np.nan
                    fits.setval(filename, "eperhour", value= np.nan)


                    


        # if "R_EXP" in table.colnames:
        #     table["e_per_hour"] = fit_param["FLUX"] * 3600 / float(table["R_EXP"]/1000)


