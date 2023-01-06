import matplotlib.pyplot as plt
from astropy.table import Table
from pyds9plugin.DS9Utils import *  # DS9n, plot_surface, getregion
from tqdm import trange, tqdm
from scipy.optimize import curve_fit
import os
from pyds9plugin.Macros.Fitting_Functions.functions import slit, smeared_slit
from pyds9plugin.Macros.FIREBall.stack_fit_slit import Measure_PSF_slits, change_val_list

slitm = slit



if __name__ == "__main__":

    d = DS9n()
    regs = getregion(d, selected=True)
    image = d.get_pyfits()[0].data
    cat, filename = Measure_PSF_slits(image, regs, filename=filename)

    d.set("regions delete all")
    create_ds9_regions(
        [cat["X_IMAGE"]],
        [cat["Y_IMAGE"]],
        # radius=[table_to_array(cat["h", "w"]).T],
        radius=[np.array(cat["lx_unsmear"]), np.array(cat["ly"])],
        save=True,
        savename= filename.replace(".fits","_c.reg"),
        form=["box"],
        color=cat["color"],
        ID=None,  # cat["name"],
    )
    d.set("regions %s" % (filename.replace(".fits","_c.reg")))

    print("center offset = ", ?)
    print("Then, master guide star must be moved in the guider by dx=%0.2f, dy=%0.2f "%(?,?))