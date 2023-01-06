from astropy.io import fits
from pyds9plugin.DS9Utils import stack_images_path, globglob, DS9n, verboseprint
import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

flux, dark, name = sys.argv[-3:]
# flux, dark = sys.argv[-2:]
# name = ""
# print(sys.argv)
# print(globglob(sys.argv[-2]))
# print(globglob(sys.argv[-1]))
def subtract_dark(flux, dark, name):
    if len(globglob(flux)) > 1:
        flux, flux_name = stack_images_path(
            paths=globglob(flux), Type="nanmean", clipping=3, dtype=float, std=False,
        )
    else:
        flux, flux_name = fits.open(flux), flux

    if len(globglob(dark)) > 1:
        dark, dark_name = stack_images_path(
            paths=globglob(dark), Type="nanmean", clipping=3, dtype=float, std=False,
        )
    else:
        dark, dark_name = fits.open(dark), dark
    # print(sys.argv[-2])
    # flux[0].header["FLUXPATH"] = flux
    # flux[0].header["DARKPATH"] = dark

    try:
        flux_numbers = np.array(flux[0].header["STK_NB"].split(" - "), dtype=int)
    except KeyError:
        flux_numbers=np.zeros(1)
    try:
        dark_numbers = np.array(dark[0].header["STK_NB"].split(" - "), dtype=int)
    except KeyError:
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
    a = flux[0].data - dark[0].data
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

filename = subtract_dark(flux, dark, name)
d = DS9n()

d.set("frame new ; file " + filename)



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
