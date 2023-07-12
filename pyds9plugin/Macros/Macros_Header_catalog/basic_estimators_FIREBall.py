#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# table['median'] = np.nanmedian(fitsfile[0].data) if  fitsfile[0].data is not None else np.nan
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
# from pyds9plugin.Macros.FIREBall.old.merge_temp import give_value_from_time


# if (int(re.findall("[0-9]+", filename)[-1]) > 16) & (
#     int(re.findall("[0-9]+", filename)[-1]) < 23
# ):
#     table["CLEAN"] = False
# else:
#     table["CLEAN"] = True


# def FindTimeField(liste):
#     timestamps = [
#         "Datation GPS",
#         "CreationTime_ISO8601",
#         "date",
#         "Date",
#         "Time",
#         "Date_TU",
#         "UT Time",
#         "Date_Time",
#         "Time UT",
#         "DATETIME",
#     ]
#     timestamps_final = (
#         [field.upper() for field in timestamps]
#         + [field.lower() for field in timestamps]
#         + timestamps
#     )
#     try:
#         timestamps_final.remove("date")
#     except ValueError:
#         pass
#     for timename in timestamps_final:
#         if timename in liste:  # table.colnames:
#             timefield = timename
#     try:
#         print("Time field found : ", timefield)
#         return timefield
#     except UnboundLocalError as e:
#         print(e)
#         return "DATETIME"  # liste[0]  # table.colnames[0]


# def give_value_from_time(
#     cat,
#     date_time,
#     date_time_field=None,
#     timeformatImage=None,
#     TimeFieldCat=None,
#     timeformatCat=None,
#     columns=None,
#     timediff=0,
# ):
#     import numpy as np

#     def RetrieveTimeFormat(time):
#         formats = [
#             "%d/%m/%Y %H:%M:%S.%f",
#             "%Y-%m-%d %H:%M:%S.%f",
#             "%Y-%m-%d %H:%M:%S",
#             "%Y-%m-%dT%H:%M:%S",
#             "%m/%d/%Y %H:%M:%S",
#             "%m/%d/%y %H:%M:%S",
#         ]
#         form = []
#         for formati in formats:
#             try:
#                 datetime.datetime.strptime(time, formati)
#                 form.append(True)
#             except ValueError:
#                 form.append(False)
#         return formats[np.argmax(form)]

#     if TimeFieldCat is None:
#         TimeFieldCat = FindTimeField(cat.colnames)
#     if timeformatCat is None:
#         timeformatCat = RetrieveTimeFormat(cat[TimeFieldCat][0])
#     if date_time_field is None:
#         date_time_field = FindTimeField(date_time.colnames)
#     cat["timestamp"] = [
#         datetime.datetime.strptime(d, timeformatCat) for d in cat[TimeFieldCat]
#     ]  # .timestamp()
#     if timeformatImage is None:
#         timeformatImage = RetrieveTimeFormat(date_time[date_time_field][0])
#     # print("%s: %s is %s"%(date_time_field,date_time[date_time_field][0], timeformatImage))
#     # print("%s: %s is %s"%(TimeFieldCat,cat[TimeFieldCat][0], timeformatCat))

#     if columns is None:
#         columns = cat.colnames
#         columns.remove(TimeFieldCat)
#         columns.remove("timestamp")
#     for i, column in enumerate(columns):
#         date_time[column] = np.nan
#     for j, line in enumerate(date_time):
#         timestamp_image = datetime.datetime.strptime(
#             date_time[date_time_field][j], timeformatImage
#         )
#         for i, column in enumerate(columns):
#             mask = np.isfinite(cat[column])  # .mask
#             try:
#                 temp = cat[column][mask][
#                     np.argmin(
#                         abs(
#                             cat[mask]["timestamp"]
#                             + datetime.timedelta(hours=timediff)
#                             - timestamp_image
#                         )
#                     )
#                 ]
#                 # print(date_time[col][j])
#                 date_time[column][j] = temp
#             except ValueError:
#                 pass
#             # print(date_time[column])

#         # print(temp, type(temp))
#     return date_time


# temp_file = os.path.dirname(os.path.dirname(filename))+"/alltemps.csv"
# if os.path.isfile(temp_file):
#     cat = Table.read(temp_file)
#     table['datime_time'] = str(table['OBSDATE'][0]) + \
#         " " + str(table['OBSTIME'][0])

#     table = give_value_from_time(cat, date_time=table, timediff=0, columns=None, date_time_field="datime_time",TimeFieldCat='time')


SATURATION = 2 ** 16 - 1

# data = fitsfile[0].data
# if data is None:
#     data = np.nan * np.ones((2,2))
# columns = np.nanmean(data, axis=1)
# lines = np.nanmean(data, axis=0)
# ly,lx = data.shape
# table['median'] = np.nanmedian(data)
# table["Lines_difference"] = np.nanmedian(lines[::2]) - np.nanmedian(lines[1::2])
# table["Lines_difference"] = np.nanmedian(columns[::2]) - np.nanmedian(columns[1::2])
# table["SaturatedPixels"] = 100 * np.mean(data > SATURATION)
# table['image'] = Column([data[915:987,1176:1267]], name="twoD_std")


np.seterr(divide="ignore")
full_analysis = False
Plot = False
# if Plot:


if ("FIREBall.py" in __file__) or (function == "execute_command"):
    # print('should plot')
    # matplotlib.use('WX')#Agg #MacOSX
    #'madcosx': valid strings are ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
    # %load_ext autoreload
    # %autoreload 2
    from astropy.table import Table

    d = DS9n()
    fitsfile = d.get_pyfits()
    filename = get_filename(d)
    table = create_table_from_header(
        filename, exts=[0], info=""
    )  # Table(data=[[1]],names=['test'])
    filename = get_filename(d)
else:
    pass
    # matplotlib.use('MacOSX')#Agg #MacOSX
    # matplotlib.use('Agg')#Agg #MacOSX


analysis_path = os.path.join(os.path.dirname(filename), "analysis/")
if not os.path.isdir(analysis_path):
    os.mkdir(analysis_path)  # kills the jobs return none!!


data = fitsfile[0].data
header = fitsfile[0].header
try:
    date = float(header["DATE"][:4])
except KeyError:
    try:
        date = float(header["OBSDATE"][:4])
    except KeyError:
        date = 0

if date < 2020:
    conversion_gain = 0.53  # ADU/e-  0.53
    RN = 100
    l1, l2 = 1053, 2133
    smearing = 1.5
else:
    conversion_gain = 0.22  # 1 / 4.5# ADU/e-  0.53
    RN = 50
    l1, l2 = -2133, -1053
    smearing = 0.5


def create_cubsets(table, header):
    try:
        table["EXPOSURE==0"] = header["EXPOSURE"] == 0
        table["EXPOSURE>0"] = header["EXPOSURE"] > 0
    except KeyError:
        pass
    return table


lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = 1120, 2100, 0, 1900  # l1, l2, 1, -1
# Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
physical_region = data[Yinf:Ysup, Xinf:Xsup]
pre_scan = data[:, 600:1000]
# post_scan = data[:, 2500:3000]
column = np.nanmean(pre_scan, axis=1)
line = np.nanmean(pre_scan, axis=0)

table["median_physical"] = np.nanmedian(physical_region.flatten() + np.random.rand(physical_region.size))  
table["mean_physical"] = np.nanmean(physical_region)
table["Col2ColDiff_pre_scan"] = np.nanmedian(line[::2]) - np.nanmedian(line[1::2])
table["Line2lineDiff_pre_scan"] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])
table["SaturatedPixels"] = (
    100 * float(np.sum(physical_region > 2 ** 16 - 10)) / np.sum(physical_region > 0)
)
table["median_pre_scan"] = np.nanmedian(pre_scan)
table["mean_pre_scan"] = np.nanmean(pre_scan)
table["std_pre_scan"] = np.nanstd(pre_scan)
# table['post_scan'] =  np.nanmedian(post_scan)
table["stdXY"] = np.nanstd(physical_region)
table["stdXY_Top"] = np.nanstd(physical_region[-30:-10, :])
table["stdXY_Bottom"] = np.nanstd(physical_region[10:480, :])
table["stdXY_pre_scan"] = np.nanstd(pre_scan)
table["BottomImage"] = np.nanmean(physical_region[10:480, :]) - table["median_pre_scan"]
table["TopImage"] = np.nanmean(physical_region[-30:-10, :]) - table["median_pre_scan"]
table["BottomImage_median"] = (
    np.nanmedian(physical_region[10:480, :]) - table["median_pre_scan"]
)
table["TopImage_median"] = (
    np.nanmedian(physical_region[-30:-10, :]) - table["median_pre_scan"]
)

table["flat"] = (np.nanmedian(physical_region) - table["median_pre_scan"]) / np.nanvar(
    physical_region
)

from pyds9plugin.Macros.FB.FB_functions import emccd_model
#2018
# if header["EMGAIN"]==9200:
#     fit_param = emccd_model(xpapoint=None, path=filename, smearing=1.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.53,RN=40,mCIC=0.15,sCIC=0.02,gain=1400,RON=105*0.53)#,mCIC=0.005
# else:
#     fit_param = emccd_model(xpapoint=None, path=filename, smearing=1.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.53,RN=40,mCIC=0.15,sCIC=0.02,RON=105*0.53)#,mCIC=0.005
#2023
fit_param = emccd_model(xpapoint=None, path=filename, smearing=0.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.02,RON=2.2)#,mCIC=0.,sCIC=0.02,RON=105*0.53)#,mCIC=0.005

table["hist_bias"] = fit_param["BIAS"]
table["hist_ron"] = fit_param["RON"]
table["hist_gain"] = fit_param["GAIN"]
table["hist_flux"] = fit_param["FLUX"]
table["e_per_hour"] = fit_param["FLUX"] * 3600 / float(table["EXPTIME"])

# print(region)
# print(type(region))
if type(region) == tuple:
    Yinf, Ysup, Xinf, Xsup = region
    reg = data[Xinf:Xsup, Yinf:Ysup]
    reg_prescan = data[Xinf-1000:Xsup-1000, Yinf:Ysup]
    n = 20
    big_reg = data[Xinf - n : Xsup + n, Yinf - n : Ysup + n]
    column = np.mean(data[Xinf - n : Xsup + n, Yinf:Ysup], axis=1)
    line = np.mean(data[Xinf:Xsup, Yinf - n : Ysup + n], axis=0)
    table["mean_region"] = np.mean(reg.flatten()) - np.mean(reg_prescan.flatten())
    table["median_region"] = np.median(reg.flatten() + np.random.rand(reg.size)) - np.mean(reg_prescan.flatten() + np.random.rand(reg_prescan.size))
    table["min_region"] = np.min(reg) - table["median_pre_scan"]
    table["region"] = Column([big_reg - table["median_pre_scan"]])
    table["column"] = Column([column - table["median_pre_scan"]])
    table["line"] = Column([line - table["median_pre_scan"]])

# # table["flux"] = table["median_physical"] - table["median_pre_scan"]
# # table["tilted_slit_214"] = np.mean(
# #     data[1630:1650, 1907:1918] - table["median_pre_scan"]
# # )
# # table["tilted_slit_206"] = np.mean(
# #     data[1630:1650, 1551:1561] - table["median_pre_scan"]
# # )
# # table["tilted_background"] = np.mean(
# #     data[1630:1650, 1760:1778] - table["median_pre_scan"]
# # )
# # table["long_slit"] = np.mean(data[1908:2006, 1908:1926] - table["median_pre_scan"])
# # table["long_smearing"] = np.mean(data[1908:2006, 1926:1947] - table["median_pre_scan"])
# # table["long_background"] = np.mean(
# #     data[1777:1877, 1929:1947] - table["median_pre_scan"]
# # )

# # table["tilted_smearing"] = np.mean(data[1475:1495, 2058:2065] - table["median_pre_scan"])
# # table["flux"] = table["median_physical"] - table["median_pre_scan"]
# # table["F4_slit"] = np.mean(data[1475:1495, 2045:2058] - table["median_pre_scan"])
# # table["F4_smearing"] = np.mean(data[1475:1495, 2058:2065] - table["median_pre_scan"])
# # table["F4_background"] = np.mean(data[1575:1595, 2058:2065] - table["median_pre_scan"])


# # fig, (ax,ax1,ax2) = plt.subplots(1,3)
# # ax.imshow(data[1908:2006, 1908:1926] - table['pre_scan'],vmin=np.min(table['long_background']), vmax=np.max(table['long_slit']))
# # ax1.imshow(data[1908:2006,1926:1947] -  table['pre_scan'], vmin=np.min(table['long_background']), vmax=np.max(table['long_slit']))
# # ax2.imshow(data[1777:1877, 1929:1947] - table['pre_scan'],vmin=np.min(table['long_background']), vmax=np.max(table['long_slit']))
# # plt.show()


# table = create_cubsets(table, header)


# table["variance_intensity_slope"] = 0
# table["variance_intensity_slope_with_OS"] = 0

# try:
#     table["stdX"] = np.nanstd(data[int(Yinf + (Ysup - Yinf) / 2), Xinf:Xsup])
#     table["stdY"] = np.nanstd(data[Yinf:Ysup, int(Xinf + (Xsup - Xinf) / 2)])
# except IndexError:
#     table["stdX"] = np.nanstd(data[int(lx / 2), :])
#     table["stdY"] = np.nanstd(data[:, int(ly / 2)])

# range_ = (
#     np.nanpercentile(physical_region, 0.1),
#     np.nanpercentile(physical_region, 99.9),
# )

# value, b = np.histogram(
#     data[Yinf:Ysup, Xinf:Xsup].flatten(), bins=np.arange(1000, 8000, 1)
# )
# value_os, b_os = np.histogram(pre_scan.flatten(), bins=np.arange(1000, 8000, 1))
# bins = (b[1:] + b[:-1]) / 2
# bins_os = (b_os[1:] + b_os[:-1]) / 2


# bias = bins[np.argmax(value)]
# bias_os = bins_os[np.argmax(value_os)]
# table["bias_os"] = bias_os


# fits.setval(
#     filename,
#     "PRESCAN",
#     value=float(table["median_pre_scan"]),
#     comment="Median value of the pre scan region",
# )
# fits.setval(
#     filename,
#     "BIAS_OS",
#     value=float(table["bias_os"]),
#     comment="Bias estimated on overscan region",
# )

# table["bias"] = bias
# mask_RN = (bins > bias - 1 * RN) & (bins < bias + 0.8 * RN) & (value > 0)
# mask_RN_os = (
#     (bins_os > bias_os - 1 * RN) & (bins_os < bias_os + 0.8 * RN) & (value_os > 0)
# )
# if data.ptp() > 200:  # check if counting image
#     popt = PlotFit1D(
#         bins_os[mask_RN_os],
#         value_os[mask_RN_os],
#         deg="gaus",
#         plot_=False,
#         P0=[1, bias, 50, 0],
#     )["popt"]
#     table["Amp"] = popt[0]
#     table["bias_fit"] = popt[1]
#     ron = np.abs(
#         PlotFit1D(
#             bins_os[mask_RN_os],
#             value_os[mask_RN_os],
#             deg="gaus",
#             plot_=False,
#             P0=[1, bias, 50, 0],
#         )["popt"][2]
#         / conversion_gain
#     )
#     if ron == 0.0:
#         table["bias_fit"] = table["median_pre_scan"]  # bins[0]
#     table["RON"] = np.max([40, np.min([ron, 120])])
#     table["RON_os"] = np.abs(
#         PlotFit1D(
#             bins_os[mask_RN_os],
#             value_os[mask_RN_os],
#             deg="gaus",
#             plot_=False,
#             P0=[1, bias_os, RN, 0],
#         )["popt"][2]
#         / conversion_gain
#     )
# else:
#     ron, table["RON"], table["RON_os"] = -99, -99, -99

# ron_fixed = table["RON"]  # np.max([30,np.min([table['RON'],120])])
# try:
#     limit_max = bins[
#         np.where((bins > bias) & (np.convolve(value, np.ones(1), mode="same") == 0))[0][
#             0
#         ]
#     ]
# except IndexError:
#     limit_max = 1e6
# try:
#     limit_max_os = bins_os[
#         np.where(
#             (bins_os > bias) & (np.convolve(value_os, np.ones(1), mode="same") == 0)
#         )[0][0]
#     ]
# except IndexError:
#     limit_max_os = 1e6

# mask_gain1 = (bins > np.min([bias, 3200]) + 6 * ron_fixed) & (bins < limit_max)
# mask_gain2 = (bins > bias + 6 * ron_fixed) & (
#     bins < bias + 50 * ron_fixed
# )  # too dangerous, no values
# mask_gain3 = (bins > bias + 6 * ron_fixed) & (
#     bins < bias + 30 * ron_fixed
# )  # too dangerous, no values
# masks = [mask_gain1, mask_gain3, mask_gain2]  # mask_gain0,

# cst = 1  # 2 if float(table['EMGAIN'])>0 else 1


# appertures = blockshaped(physical_region - table["median_pre_scan"], 40, 40)
# vars_ = np.nanvar(appertures, axis=(1, 2))
# intensities = np.nanmean(appertures, axis=(1, 2))
# vars_masked, intensities_masked = SigmaClipBinned(vars_, intensities, sig=1, Plot=False)
# try:
#     popt = PlotFit1D(
#         vars_masked, intensities_masked / cst, deg=1, sigma_clip=[3, 10], plot_=False
#     )["popt"]
# except ValueError:
#     popt = [0, 0]
#     print("error!!!: ", filename, vars_, intensities, vars_masked, intensities_masked)
# table["var_intensity_slope"] = popt[1]
# table["var_intensity_"] = popt[0]

# if Plot:
#     fig, ax2 = plt.subplots(figsize=(10, 5))
#     ax2.plot(intensities, vars_ / cst, ".", label=popt[1])
#     limsx = ax2.get_xlim()
#     limsy = ax2.get_ylim()
#     popt = PlotFit1D(
#         intensities, vars_ / cst, deg=1, sigma_clip=[3, 10], plot_=True, ax=ax2
#     )["popt"]
#     ax2.set_ylim(limsy)
#     ax2.set_xlim(limsx)
#     ax2.legend()
#     ax2.set_xlabel("Intensity")
#     ax2.set_ylabel("Variance")
#     fig.tight_layout()
#     fig.savefig(
#         analysis_path + os.path.basename(filename).replace(".fits", "_varint.png")
#     )
#     np.savetxt("/tmp/varintens.dat", np.array([intensities, vars_]).T)


# # table['sCIC_OS'] = (table['post_scan'] - table['pre_scan'] )/ table['Gain0'] /conversion_gain

# n = 20
# image_center = data[
#     int(lx / 2 - n) : int(lx / 2 + n) + 1, int(ly / 2 - n) : int(ly / 2 + n) + 1
# ]
# image_center_01 = (image_center - image_center.min()) / (
#     image_center - image_center.min()
# ).max()
# fft = np.fft.irfft2(
#     np.fft.rfft2(image_center_01) * np.conj(np.fft.rfft2(image_center_01))
# )
# table["fft_sum"] = np.sum(fft)
# table["fft_var"] = np.var(fft)
# table["fft_mean"] = np.mean(fft)
# table["fft_median"] = np.median(fft)


# x_correlation = np.zeros(image_center_01.shape)
# for i in range(image_center_01.shape[0]):
#     x_correlation[i, :] = signal.correlate(
#         image_center_01[i, :], image_center_01[i, :], mode="same"
#     )  # / 128
# x_correlation /= x_correlation.min()
# size = 12
# lxa, lya = x_correlation.shape
# profile = np.mean(
#     x_correlation[:, int(lya / 2) - size - 1 : int(lya / 2) + size], axis=0
# )[: size + 2]
# try:
#     table["smearing_autocorr"] = PlotFit1D(
#         np.arange(len(profile)),
#         profile,
#         deg="exp",
#         P0=[5e-1, profile.max() - profile.min(), profile.min()],
#         plot_=False,
#     )["popt"][2]
# except ValueError:
#     table["smearing_autocorr"] = -99

# table["hot_pixel_value"] = data[330, 1332]
# table["hot_pixel_value_prior"] = data[330, 1331]
# table["hot_pixel_value_next"] = data[330, 1333]

# for i in range(3):
#     region = data[100 + 600 * i : 100 + 600 * (i + 1), 1120:2100]
#     table["hot_pixels_fraction_%i" % (i)] = (
#         100 * float(np.sum(region > limit_max)) / np.sum(region > 0)
#     )

# np.savetxt("/tmp/xy.txt", np.array([bins, np.log10(value)]).T)

# if full_analysis:
#     table["long_slit_image"] = Column(
#         [data[1908 - 20 : 2006 + 20, 1908 - 20 : 1926 + 20] - table["median_pre_scan"]],
#         name="long_slit_image",
#     )

#     table["var_analysis"] = Column([vars_], name="var_analysis")
#     table["intensity_analysis"] = Column([intensities], name="intensity_analysis")
#     table["percentiles"] = Column(
#         [np.nanpercentile(data[Yinf:Ysup, Xinf:Xsup], np.arange(100))],
#         name="percentiles",
#     )
#     value_to_save, b = np.histogram(
#         data[Yinf:Ysup, Xinf:Xsup].flatten() - bias_os, bins=np.arange(-500, 5000, 1)
#     )
#     value_to_save_os, b = np.histogram(
#         pre_scan.flatten() - bias_os, bins=np.arange(-500, 5000, 1)
#     )
#     table["bins"] = Column([(b[1:] + b[:-1]) / 2], name="bins")
#     table["hist"] = Column([value_to_save], name="hist")
#     table["hist_os"] = Column([value_to_save_os], name="hist_os")
#     table["overscan_decrease"] = Column(
#         [np.nanmean(data[:, 2143 : 2143 + 200], axis=0)], name="overscan_decrease"
#     )
#     table["hot_pixel_profile"] = Column([data[330, 1332 : 1332 + 5]], name="hot_pixel")
#     values = (
#         100
#         * np.sum(data[:, 1120:2100] > limit_max, axis=1)
#         / np.sum(data[:, 1120:2100] > 0, axis=1)
#     )
#     values_os = (
#         100
#         * np.sum(data[:, 1120:2100] > limit_max_os, axis=1)
#         / np.sum(data[:, 1120:2100] > 0, axis=1)
#     )
#     table["hot_pixels_fraction"] = Column(
#         [values[:-1].reshape(47, -1).mean(axis=1)], name="hot_pixels_fraction"
#     )
#     table["hot_pixels_fraction+"] = Column(
#         [values_os[:-1].reshape(47, -1).mean(axis=1)], name="hot_pixels_fraction"
#     )
#     n = 20
#     table["twoD_mean"] = Column(
#         [
#             np.array(
#                 block_reduce(
#                     data[:-9, 500:-516],
#                     block_size=(n, n),
#                     func=np.nanmean,
#                     cval=np.nanmean(data),
#                 ),
#                 dtype=int,
#             )
#         ],
#         name="twoD_mean",
#     )
#     table["twoD_median"] = Column(
#         [
#             np.array(
#                 block_reduce(
#                     data[:-9, 500:-516],
#                     block_size=(n, n),
#                     func=np.nanmedian,
#                     cval=np.nanmedian(data),
#                 ),
#                 dtype=int,
#             )
#         ],
#         name="twoD_median",
#     )
#     table["twoD_std"] = Column(
#         [
#             np.array(
#                 block_reduce(
#                     data[:-9, 500:-516],
#                     block_size=(n, n),
#                     func=np.nanstd,
#                     cval=np.nanstd(data),
#                 ),
#                 dtype=int,
#             )
#         ],
#         name="twoD_std",
#     )


# table.write("/tmp/test.fits", overwrite=True)

