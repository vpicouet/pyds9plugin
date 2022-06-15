#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# table['median'] = np.nanmedian(fitsfile[0].data) if  fitsfile[0].data is not None else np.nan
from astropy.table import Column
from scipy.optimize import curve_fit
import datetime
from pyds9plugin.DS9Utils import *  # DS9n,PlotFit1D

# import re
# import numpy as np
# from pyds9plugin.DS9Utils import blockshaped
# from astropy.table import Column
# from astropy.io import fits
# from scipy import signal
# from skimage.measure import block_reduce
import matplotlib.pyplot as plt
from pyds9plugin.Macros.Fitting_Functions.functions import slit, smeared_slit
from pyds9plugin.Macros.FIREBall.old.merge_temp import give_value_from_time

# if ("Macros_Header_catalog/test_slit_thermal_drift.py" in __file__) or (
#     function == "execute_command"
# ):
#     from astropy.table import Table

d = DS9n()
#     fitsfile = d.get_pyfits()
#     filename = get_filename(d)
#     table = create_table_from_header(filename, exts=[0], info="")
#     filename = get_filename(d)
# else:
# pass

data = fitsfile[0].data
header = fitsfile[0].header


lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = 1120, 2100, 1300, 1900  # l1, l2, 1, -1
# Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
physical_region = data[Yinf:Ysup, Xinf:Xsup]
pre_scan = data[:, 600:1000]
# post_scan = data[:, 2500:3000]
column = np.nanmean(pre_scan, axis=1)
line = np.nanmean(pre_scan, axis=0)

table["physical"] = np.median(physical_region)
table["Col2ColDiff_pre_scan"] = np.nanmedian(line[::2]) - np.nanmedian(line[1::2])
table["Line2lineDiff_pre_scan"] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])
table["SaturatedPixels"] = (
    100 * float(np.sum(physical_region > 2 ** 16 - 10)) / np.sum(physical_region > 0)
)
table["pre_scan"] = np.nanmedian(pre_scan)
# table['post_scan'] =  np.nanmedian(post_scan)
table["stdXY"] = np.nanstd(physical_region)
table["stdXY_bottom"] = np.nanstd(physical_region)
table["stdXY_bottom"] = np.nanstd(physical_region)
table["stdXY_pre_scan"] = np.nanstd(pre_scan)
table["BottomImage"] = np.nanmean(physical_region[10:30, :]) - table["pre_scan"]
table["TopImage"] = np.nanmean(physical_region[-30:-10, :]) - table["pre_scan"]
table["BottomImage_median"] = (
    np.nanmedian(physical_region[10:30, :]) - table["pre_scan"]
)
table["TopImage_median"] = np.nanmedian(physical_region[-30:-10, :]) - table["pre_scan"]

table["flat"] = (np.nanmedian(physical_region) - table["pre_scan"]) / np.nanvar(
    physical_region
)


spos = [1475, 1495, 2045, 2058]
spos = [122, 145, 1802, 1814]  # 20220504
spos = [345, 345 + 70, 2075, 2075 + 55]  # drift 29 juin

spos1 = [1400, 1400 + 70, 1980, 1980 + 55]  # drift 29 juin
spos2 = [1863, 1863 + 70, 1874, 1874 + 55]  # drift 29 juin

table["flux"] = table["physical"] - table["pre_scan"]
table["F4_slit"] = np.mean(
    data[spos[0] : spos[1], spos[2] : spos[3]] - table["pre_scan"]
)
table["F4_smearing"] = np.mean(
    data[spos[0] : spos[1], spos[2] : spos[3]] - table["pre_scan"]
)
table["F4_background"] = np.mean(
    data[spos[0] : spos[1], spos[2] : spos[3]] - table["pre_scan"]
)
n = 10
# data = data[1475 - n : 1495 + n, 2045 - n : 2058 + n]
# dat = data[spos[0] : spos[1], spos[2] : spos[3]] - table["pre_scan"]
size = (40, 33)
# - table["pre_scan"]
# slit =   # - table["pre_scan"]
# print(slit)
# print(slit.shape)
image = data
table["slit0"] = Column([data[spos[0] : spos[1], spos[2] : spos[3]]], name="slit0")

table["slit1"] = Column([data[spos1[0] : spos1[1], spos1[2] : spos1[3]]], name="slit1")
table["slit2"] = Column([data[spos2[0] : spos2[1], spos2[2] : spos2[3]]], name="slit2")
# path =
regs = (
    open(
        "/Users/Vincent/Nextcloud/LAM/FIREBALL/TestsFTS2018-Flight/E2E-AIT-Flight/all_diffuse_illumination/FocusEvolution/tilted/old_tilted_2022_4_-70.reg",
        "r",
    )
    .read()
    .split("\n")[3:]
)
# regs = process_region(open(path, "r",))

for region in process_region(regs, d, message=False):
    x, y = int(region.xc), int(region.yc)
    if (x > 1060) & (y < 1950) & (y > 40) & (x < 2060):
        x_inf, x_sup, y_inf, y_sup = lims_from_region(region=region, coords=None)
        n = 20
        subim1 = image[y_inf - n : y_sup + n, x_inf:x_sup]
        subim2 = image[y_inf:y_sup, x_inf - n : x_sup + n]
        subim3 = image[y_inf - n : y_sup + n, x_inf - n : x_sup + n]
        # subim = image[y_inf:y_sup, x_inf:x_sup]
        # xx.append(x)
        # yy.append(y)
        # plt.figure()
        # plt.imshow(subim)
        # plt.show()
        y_spatial = np.nanmedian(subim1, axis=1)
        y_spectral = np.nanmedian(subim2, axis=0)
        x_spatial = np.arange(len(y_spatial))
        x_spectral = np.arange(len(y_spectral))
        table["slit_%s_spatial" % (region.id)] = Column([y_spatial], name="slit0")
        table["slit_%s_spectral" % (region.id)] = Column([y_spectral], name="slit0")


# if data.shape == size:
#     print(1)
#     table["slit_image"] = Column([data], name="slit_image")

#     y_spectral = np.mean(data, axis=0)
#     x_spectral = np.arange(len(y_spectral))

#     y_spatial = np.mean(data, axis=1)
#     x_spatial = np.arange(len(y_spatial))

#     P0 = [
#         y_spectral.ptp(),
#         3,
#         np.argmax(y_spectral[::-1]),
#         4,
#         np.median(y_spectral),
#         1.3,
#     ]
#     bounds = [
#         [0.7 * y_spectral.ptp(), 0.1, 0.1, 0.1, np.nanmin(y_spectral), 0.1],
#         [
#             y_spectral.ptp(),
#             len(y_spectral),
#             len(y_spectral),
#             10,
#             np.nanmax(y_spectral),
#             6,
#         ],
#     ]
#     try:
#         popt_spectral_deconvolved, pcov = curve_fit(
#             smeared_slit, x_spectral, y_spectral[::-1], p0=P0
#         )  # ,bounds=bounds)#,bounds=bounds
#     except RuntimeError:
#         popt_spectral_deconvolved = [0, 0, 0, 0, 0, 0, 0, 0, 0]
#     try:
#         popt_spatial, pcov = curve_fit(
#             slit,
#             x_spatial,
#             y_spatial[::-1],
#             p0=[
#                 y_spatial.ptp(),
#                 3,
#                 np.argmax(y_spatial[::-1]),
#                 4,
#                 np.median(y_spatial),
#             ],
#         )  # ,bounds=bounds
#     except RuntimeError:
#         popt_spatial = [0, 0, 0, 0, 0, 0, 0, 0, 0]

#     # popt_spectral_deconvolved = PlotFit1D(x_spectral,y_spectral[::-1],deg=smeared_slit, plot_=True,P0=P0,bounds=bounds)['popt']

#     print(popt_spectral_deconvolved[-1])
#     table["smearing"] = popt_spectral_deconvolved[-1]
#     table["x"] = popt_spectral_deconvolved[2]
#     table["y"] = popt_spatial[2]
#     # plt.figure()
#     # plt.plot(y_spectral[::-1],label='Data')
#     # plt.plot(smeared_slit(x_spectral,*P0),':',label='P0')#[y_spectral.ptp(),3,30,3,np.median(y_spectral),1.5]))
#     # plt.plot(smeared_slit(x_spectral,*popt_spectral_deconvolved),label='Fit')
#     # # plt.title( '%i  T=%0.1f  s=%0.1f c=%0.1f'%(float(os.path.basename(file)[6:-5]), float(header['EMCCDBAC']),abs( popt_spectral_deconvolved[-1]),popt_spectral_deconvolved[2] ))
#     # plt.legend()
#     # plt.show()

# else:
#     print(0)
#     table["slit_image"] = Column([np.ones(size)], name="slit_image")
#     table["smearing"] = 0
#     table["x"] = 0
#     table["y"] = 0

# table["date"] = datetime.datetime.strptime(
#     header["OBSDATE"] + header["OBSTIME"], "%Y-%m-%d%H:%M:%S.%f"
# ).strftime("%Y-%m-%dT%H:%M:%S")
# table["date_float"] = float(
#     datetime.datetime.strptime(
#         header["OBSDATE"] + header["OBSTIME"], "%Y-%m-%d%H:%M:%S.%f",
#     ).strftime("%y%m%d.%H%M")
# )

# print(filename)
# print(data.shape)
# table = give_value_from_time(
#     cat,
#     date_time=table,
#     timediff=0,
#     columns=None,
#     date_time_field="datime_time",
#     TimeFieldCat="time",
# )
# table["datime_time"] = str(header["OBSDATE"]) + str(header["OBSTIME"])
# cat_temp = Table.read()
# give_value_from_time(
#     date_time=table, date_time_field="datime_time", TimeFieldCat="time"
# )
