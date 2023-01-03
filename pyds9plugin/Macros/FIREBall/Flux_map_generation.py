import re
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack, hstack

# normally it is already OS curbtracted
exp_time = header["EXPTIME"]
moninal_throughput = 0.13
telescope_area = 7854
LO_reduction = 0.9
altitude = 40 #km
atm = Table.read("/Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FIREBall/old/AtmTrans_wave_180-239-0.1000nm_Alt_25-45.csv")
altitude = 38
mask = (atm["Wavelength"]>199) & (atm["Wavelength"]<213)
atm_absorption = np.nanmean(atm["Trans_%skm"%(altitude)][mask])

Wavelength = 200 #nm
det_QE = 0.55 ##dependance on wavelength

EM_gain = 1800
conversion_gain = 0.2
dispersion  = 	4.66#pix/A
new_ds9 = ds9 * conversion_gain * dispersion / exp_time / EM_gain #photoelectrons/s/A
ds9 = new_ds9 / det_QE/ atm_absorption / telescope_area / LO_reduction / moninal_throughput # ph/s/cm2/A
