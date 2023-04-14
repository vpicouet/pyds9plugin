from astropy.table import Table, Column
import re
import numpy as np
from pyds9plugin.DS9Utils import *  # DS9n,PlotFit1D
from pyds9plugin.DS9Utils import blockshaped
from astropy.table import Column
from astropy.io import fits
from scipy import signal
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import matplotlib
import datetime


data = fitsfile[0].data
header = fitsfile[0].header
if data is not None:

    lx, ly = data.shape
    Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
    Xinf, Xsup, Yinf, Ysup = 1120, 2100, 0, 1900  # l1, l2, 1, -1
    # Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
    physical_region = data[Yinf:Ysup, Xinf:Xsup]
    pre_scan = data[100:-100, 600:1000]
    # post_scan = data[:, 2500:3000]
    column = np.nanmean(pre_scan, axis=1)
    line = np.nanmean(pre_scan, axis=0)

    # table["median_physical"] = np.nanmedian(physical_region)
    # table["mean_physical"] = np.nanmean(physical_region)
    # table["Col2ColDiff_pre_scan"] = np.nanmedian(line[::2]) - np.nanmedian(line[1::2])
    # table["Line2lineDiff_pre_scan"] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])
    # table["SaturatedPixels"] = (
    #     100 * float(np.sum(physical_region > 2 ** 16 - 10)) / np.sum(physical_region > 0)
    # )
    # table["median_pre_scan"] = np.nanmedian(pre_scan)
    # table["mean_pre_scan"] = np.nanmean(pre_scan)
    # # table['post_scan'] =  np.nanmedian(post_scan)
    # table["stdXY"] = np.nanstd(physical_region)
    # table["stdXY_Top"] = np.nanstd(physical_region[-30:-10, :])
    # table["stdXY_Bottom"] = np.nanstd(physical_region[10:480, :])
    # table["stdXY_pre_scan"] = np.nanstd(pre_scan)
    # table["BottomImage"] = np.nanmean(physical_region[10:480, :]) - table["median_pre_scan"]
    # table["TopImage"] = np.nanmean(physical_region[-30:-10, :]) - table["median_pre_scan"]


    table["RN_std_os"] = np.nanstd(pre_scan)
else:
    table["RN_std_os"] = -99.9
    