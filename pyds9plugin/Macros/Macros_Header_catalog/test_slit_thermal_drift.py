#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# table['median'] = np.nanmedian(fitsfile[0].data) if  fitsfile[0].data is not None else np.nan
from astropy.table import Column
from scipy.optimize import curve_fit
import datetime
# import re
# import numpy as np
# from pyds9plugin.DS9Utils import *#DS9n,PlotFit1D
# from pyds9plugin.DS9Utils import blockshaped
# from astropy.table import Column
# from astropy.io import fits
# from scipy import signal
# from skimage.measure import block_reduce
import matplotlib.pyplot as plt
# import matplotl

def smeared_slit(x, amp, l, x0, FWHM, offset,Smearing):
    """Convolution of a box with a gaussian
    """
    # Smearing=0.8
    from scipy import special
    import numpy as np
    from scipy.sparse import dia_matrix

    def variable_smearing_kernels(
        image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
    ):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np

        n = 15
        smearing_length = Smearing * np.exp(-image / SmearExpDecrement)**np.ones(len(image))
        if type_ == "exp":
            # print(Smearing,smearing_length,image)
            smearing_kernels = np.exp(
                -np.arange(n)[:, np.newaxis, np.newaxis] / abs(smearing_length)
                # -np.arange(n)[::int(np.sign(smearing_length[-1])), np.newaxis, np.newaxis] / abs(smearing_length)
            )
        else:
            assert 0 <= Smearing <= 1
            smearing_kernels = np.power(Smearing, np.arange(n))[
                :, np.newaxis, np.newaxis
            ] / np.ones(smearing_length.shape)
        smearing_kernels /= smearing_kernels.sum(axis=0)
        return smearing_kernels
    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    function = amp * (a + b) / (a + b).ptp()#+1#4 * l
    # function = np.vstack((function,function)).T
    # print(FWHM,x0, l , function)
    smearing_kernels = variable_smearing_kernels(
        function, Smearing, SmearExpDecrement=50000)
    n = smearing_kernels.shape[0]
    # print(smearing_kernels.sum(axis=1))
    # print(smearing_kernels.sum(axis=1))
    A = dia_matrix(
        (smearing_kernels.reshape((n, -1)), np.arange(n)),
        shape=(function.size, function.size),
    )
    function = A.dot(function.ravel()).reshape(function.shape)
    # function = np.mean(function,axis=1)
    return  function + offset


def slit(x,  amp, l, x0, FWHM, offset):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np

    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    function = amp * (a + b) / (a + b).ptp()#4 * l
    return  function + offset


data = fitsfile[0].data
header = fitsfile[0].header



lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = 1120, 2100, 1300, 1900#l1, l2, 1, -1
# Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
physical_region = data[Yinf:Ysup, Xinf:Xsup]
pre_scan = data[:, 600:1000]
# post_scan = data[:, 2500:3000]
column = np.nanmean(pre_scan, axis=1)
line = np.nanmean(pre_scan, axis=0)

table['physical'] = np.median(physical_region)
table["Col2ColDiff_pre_scan"] = np.nanmedian(line[::2]) - np.nanmedian(line[1::2])
table["Line2lineDiff_pre_scan"] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])
table["SaturatedPixels"] = 100 * float(np.sum(physical_region> 2 ** 16 - 10)) / np.sum(physical_region > 0)
table['pre_scan'] = np.nanmedian(pre_scan)
# table['post_scan'] =  np.nanmedian(post_scan)
table["stdXY"] = np.nanstd(physical_region)
table["stdXY_bottom"] = np.nanstd(physical_region)
table["stdXY_bottom"] = np.nanstd(physical_region)
table["stdXY_pre_scan"] = np.nanstd(pre_scan)
table["BottomImage"] = np.nanmean(physical_region[10:30,:]) - table['pre_scan']
table["TopImage"] = np.nanmean(physical_region[-30:-10,:]) - table['pre_scan']
table["BottomImage_median"] = np.nanmedian(physical_region[10:30,:]) - table['pre_scan']
table["TopImage_median"] = np.nanmedian(physical_region[-30:-10,:]) - table['pre_scan']

table["flat"] = (np.nanmedian(physical_region) - table['pre_scan'] )/np.nanvar(physical_region)

table['flux'] = table['physical'] - table['pre_scan']
table['F4_slit'] = np.mean(data[1475:1495,2045:2058] -  table['pre_scan'])
table['F4_smearing'] = np.mean(data[1475:1495,2058:2065] -  table['pre_scan'])
table['F4_background'] = np.mean(data[1575:1595,2058:2065] -  table['pre_scan'])
n=10
data = data[1475-n:1495+n,2045-n:2058+n]
size = (40, 33)
if data.shape==size:
    print(1)
    table['slit_image'] = Column([data], name="slit_image")

    y_spectral = np.mean(data,axis=0)
    x_spectral = np.arange(len(y_spectral))

    y_spatial = np.mean(data,axis=1)
    x_spatial = np.arange(len(y_spatial))

    P0 = [y_spectral.ptp(),3,np.argmax(y_spectral[::-1]),4,np.median(y_spectral),1.3]
    bounds = [[0.7*y_spectral.ptp(),0.1,0.1,0.1,np.nanmin(y_spectral),0.1], [y_spectral.ptp(),len(y_spectral),len(y_spectral),10,np.nanmax(y_spectral),6]]
    try:
        popt_spectral_deconvolved,pcov = curve_fit(smeared_slit,x_spectral,y_spectral[::-1], p0=P0)#,bounds=bounds)#,bounds=bounds
    except RuntimeError:
        popt_spectral_deconvolved = [0,0,0,0,0,0,0,0,0]
    try:
        popt_spatial,pcov = curve_fit(slit,x_spatial,y_spatial[::-1], p0=[y_spatial.ptp(),3,np.argmax(y_spatial[::-1]),4,np.median(y_spatial)])#,bounds=bounds
    except RuntimeError:
        popt_spatial = [0,0,0,0,0,0,0,0,0]

    # popt_spectral_deconvolved = PlotFit1D(x_spectral,y_spectral[::-1],deg=smeared_slit, plot_=True,P0=P0,bounds=bounds)['popt']

    print(popt_spectral_deconvolved[-1])
    table['smearing'] = popt_spectral_deconvolved[-1]
    table['x'] = popt_spectral_deconvolved[2]
    table['y'] = popt_spatial[2]
    # plt.figure()
    # plt.plot(y_spectral[::-1],label='Data')
    # plt.plot(smeared_slit(x_spectral,*P0),':',label='P0')#[y_spectral.ptp(),3,30,3,np.median(y_spectral),1.5]))
    # plt.plot(smeared_slit(x_spectral,*popt_spectral_deconvolved),label='Fit')
    # # plt.title( '%i  T=%0.1f  s=%0.1f c=%0.1f'%(float(os.path.basename(file)[6:-5]), float(header['EMCCDBAC']),abs( popt_spectral_deconvolved[-1]),popt_spectral_deconvolved[2] ))
    # plt.legend()
    # plt.show()

else:
    print(0)
    table['slit_image'] = Column([np.ones(size)], name="slit_image")
    table['smearing'] = 0
    table['x'] = 0
    table['y'] = 0

table['date'] = datetime.datetime.strptime(header['OBSDATE'] + header['OBSTIME'], "%Y-%m-%d%H:%M:%S.%f").strftime("%Y-%m-%dT%H:%M:%S")
table['date_float'] = float(datetime.datetime.strptime(header['OBSDATE'] + header['OBSTIME'], "%Y-%m-%d%H:%M:%S.%f",).strftime("%y%m%d.%H%M"))

print(filename)
print(data.shape)
