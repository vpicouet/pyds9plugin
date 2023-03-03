from astropy.io import fits
from pyds9plugin.DS9Utils import stack_images_path, globglob, DS9n, verboseprint
import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

flux, dark, name = sys.argv[-3:]
# flux, dark = sys.argv[-2:]
# name = ""
# print(sys.argv)
# print(globglob(sys.argv[-2]))
# print(globglob(sys.argv[-1]))
def subtract_dark(flux_, dark_, name):
    if len(globglob(flux_)) > 1:
        flux, flux_name = stack_images_path(
            paths=globglob(flux_), Type="nanmean", clipping=3, dtype=float, std=False,
        )
    else:
        flux, flux_name = fits.open(flux_), flux_

    if len(globglob(dark_)) > 1:
        dark, dark_name = stack_images_path(
            paths=globglob(dark_), Type="nanmean", clipping=3, dtype=float, std=False,
        )
    else:
        dark, dark_name = fits.open(dark_), dark_
    # print(sys.argv[-2])
    # flux[0].header["FLUXPATH"] = flux
    # flux[0].header["DARKPATH"] = dark

    try:
        flux_numbers = np.array(flux[0].header["STK_NB"].split(" - "), dtype=int)
    except KeyError:
        if len(globglob(flux_))==1:
            flux_numbers=np.array(re.findall(r'\d+',os.path.basename(flux_)),dtype=int)
        else:
            flux_numbers=np.zeros(1)
    try:
        dark_numbers = np.array(dark[0].header["STK_NB"].split(" - "), dtype=int)
    except KeyError:
        if len(globglob(dark_))==1:
            dark_numbers=np.array(re.findall(r'\d+',os.path.basename(dark_)),dtype=int)
        else:
            dark_numbers=np.zeros(1)

    filename = os.path.dirname(flux_name) + "/%s_flux_%i-%i_dark_%i-%i.fits" % (
        name,
        flux_numbers.min(),
        flux_numbers.max(),
        dark_numbers.min(),
        dark_numbers.max(),
    )
    # print(dark_name, filename)
    # sys.exit()
    a = np.array(flux[0].data - dark[0].data,dtype=np.int16)
    
    fits.HDUList([fits.PrimaryHDU(a, header=flux[0].header)]).writeto(filename, overwrite=True)


    # texp = flux[0].header["EXPOSURE"]/1e3
    # fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 5))  # ,sharex=True)
    # ax0.plot(np.nanmean(a[10:, 10:100], axis=1),".",label="lambda ~ 213nm, f=%ie-/sec" % (np.nanmax(np.nanmean(a[10:, 10:100], axis=1))/texp))
    # ax0.plot(np.nanmean(a[10:, 500:600], axis=1), ".", label="lambda ~ 206nm, f=%ie-/sec" % (np.nanmax(np.nanmean(a[10:,  500:600], axis=1))/texp))
    # ax0.plot(np.nanmean(a[10:, -200:-100], axis=1), ".", label="lambda ~ 202nm, f=%ie-/sec" % (np.nanmax(np.nanmean(a[10:,-200:-100], axis=1))/texp))
    # ax0.legend()
    # ax0.set_xlabel("Spatial direction")
    # ax1.plot(np.nanmean(a[100:200, :-10], axis=0), ".", label="Bottom detector, f=%ie-/sec" % (np.nanmax(np.nanmean(a[100:200, :-10], axis=0))/texp))
    # ax1.plot(np.nanmean(a[1000:1100, :-10], axis=0), ".", label="Middle detector, f=%ie-/sec" % (np.nanmax(np.nanmean(a[1000:1100, :-10], axis=0))/texp))
    # ax1.plot(np.nanmean(a[-200:-100, :-10], axis=0), ".", label="Top detector, f=%ie-/sec" % (np.nanmax(np.nanmean(a[-200:-100, :-10], axis=0))/texp))
    # ax1.set_xlabel("Spectral direction")
    # ax1.legend()
    # try:
    #     ax0.set_title(os.path.basename(filename)[:-5] + " : Texp=%is" % (texp))
    # except Exception:
    #     pass
    # fig.tight_layout()
    # fig.savefig(filename.replace('.fits','.png'))
    # plt.show()
    return filename




# DS9Utils maxi_mask -x none -t 0 -b 8 -m 0 -n 0  -f 0-0-$F3-$F4-$F5-$F6-$F7-$F8-$F9-$F_10-$F_11-$F_12-$F_13-$F_14 -P $P1-$P2-$P3-$P4-$P5-$P6-$P7-$P8-$P9-$P_10-$P_11-$P_12-$P_13-$P_14  -T $T1-$T2-$T3-$T4-$T5-$T6-$T7-$T8-$T9-$T_10-$T_11-$T_12-$T_13-$T_14    | $text



d = DS9n()
filename = subtract_dark(flux, dark, name)
d.set("frame new ; file " + filename)
# corr_name = reduce_dark(filename)
# d.set("frame new ; file " + corr_name)



#
# import matplotlib.pyplot as plt
# from astropy.io import fits
# from astropy.table import Table, vstack, hstack
# number_zeros = 8
# log_file = '/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/Samtec_cables_D_lamp/log2.csv'
# # log_file = '/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/20220414/log2.csv'
# log = Table.read(log_file).to_pandas()
# path = os.path.dirname(log_file)
# sets = np.unique(log['set'])
# log['counts']=0.0
# log['counts_per_sec']=0.0
#
# log['exp_time']=0.0
# for i,set_ in enumerate(sets):
#     print(set_)
#     sublog=log.query('set == "%s"'%(set_))
#     print(sublog)
#     try:
#         dark_numbers = log.query('set == "%s" & name=="dark"'%(set_)).iloc[0]['numbers']
#     except IndexError:
#         pass
#     length_dark = len(str(dark_numbers.split('-')[-1]))
#     for name in np.unique(sublog['name']):
#         print('    ' + name)
#         if name!='dark':
#             numbers =  log.query('set == "%s" & name=="%s"'%(set_,name)).iloc[0]['numbers']
#             length = len(str(numbers.split('-')[-1]))
#
#             flux = path + '/image_%s[%s].fits'%('0'*(number_zeros-length),numbers)
#             dark = path + '/image_%s[%s].fits'%('0'*(number_zeros-length_dark),dark_numbers)
#             print(flux)
#             fitsfile, filename = subtract_dark(flux, dark, '%s_%s'%(name,set_))
#             a = fitsfile.data
#             log.loc[log.eval('set == "%s" & name=="%s"'%(set_,name)), 'counts'] = np.nanmedian(a[-100:,-100:])
#             log.loc[log.eval('set == "%s" & name=="%s"'%(set_,name)), 'counts_per_sec'] = np.nanmedian(a[-100:,-100:]) / ( fitsfile.header['EXPOSURE']/1000)
#             log.loc[log.eval('set == "%s" & name=="%s"'%(set_,name)), 'exp_time'] = fitsfile.header['EXPOSURE']/1000
#             log.loc[log.eval('set == "%s" & name=="%s"'%(set_,'dark')), 'exp_time'] = fitsfile.header['EXPOSURE']/1000
#             sys.exit()
# log.to_csv(log_file,index=False)
