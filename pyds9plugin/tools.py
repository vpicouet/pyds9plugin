#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:13 2018

@author: V. Picouet

Copyright Vincent Picouet (01/01/2019)

vincent@picouet.fr

This software is a computer program whose purpose is to perform quicklook
image processing and analysis. It can ionteract with SAOImage DS9 Software
when loaded as an extension.

This software is governed by the CeCILL-B license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-B
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-B license and that you accept its terms.
"""

# from pyds9plugin.DS9Utils import #verboseprint
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import sys
# from pyds9plugin.DS9Utils import create_ds9_regions, DS9n
def AddFieldAftermatching(
    FinalCat=None,
    ColumnCat=None,
    path1=None,
    path2=None,
    radec1=["RA", "DEC"],
    radec2=["RA", "DEC"],
    distance=0.5,
    field="Z_ML",
    new_field=None,
    query=None,
):
    """
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    if path1 is not None:
        try:
            FinalCat = Table.read(path1)
        except:
            FinalCat = Table.read(path1, format="ascii")

    if path2 is not None:
        try:
            ColumnCat = Table.read(path2)
        except:
            ColumnCat = Table.read(path2, format="ascii")
    # #verboseprint("cat 1 : %i lines" % (len(FinalCat)))
    # #verboseprint("cat 2 : %i lines" % (len(ColumnCat)))
    # print(ColumnCat['ZFLAG'])
    # #verboseprint(ColumnCat)
    if query is not None:
        ColumnCat = apply_query(
            cat=ColumnCat, query=query, path=None, new_path=None, delete=True
        )
        #verboseprint(ColumnCat)
        mask = np.isfinite(ColumnCat[radec2[0]]) & np.isfinite(ColumnCat[radec2[1]])
        ColumnCat = ColumnCat[mask]
    # print(ColumnCat['ZFLAG'])
    if len(radec1) == 2:
        try:
            c = SkyCoord(
                ra=ColumnCat[radec2[0]] * u.deg, dec=ColumnCat[radec2[1]] * u.deg
            )
        except Exception as e:
            print(e)
            c = SkyCoord(ra=ColumnCat[radec2[0]], dec=ColumnCat[radec2[1]])
        try:
            catalog = SkyCoord(
                ra=FinalCat[radec1[0]] * u.deg, dec=FinalCat[radec1[1]] * u.deg
            )
        except Exception:
            catalog = SkyCoord(ra=FinalCat[radec1[0]], dec=FinalCat[radec1[1]])
        #        idx, d2d, d3d = catalog.match_to_catalog_sky(c[mask])
        # #verboseprint(catalog)
        # #verboseprint(c)
        idx, d2d, d3d = catalog.match_to_catalog_sky(c)
        mask = 3600 * np.array(d2d) < distance
        # #verboseprint("Number of matches < %0.2f arcsec :  %i " % (distance, mask.sum()))

    elif len(radec1) == 1:
        import pandas as pd
        from pyds9plugin.DS9Utils import DeleteMultiDimCol

        ColumnCat = ColumnCat[radec2 + field]
        if new_field is not None:
            ColumnCat.rename_columns(field, new_field)
        ColumnCat.rename_column(radec2[0], "id_test")
        FinalCat = DeleteMultiDimCol(FinalCat)
        ColumnCat = DeleteMultiDimCol(ColumnCat)
        FinalCatp = FinalCat.to_pandas()
        ColumnCatp = ColumnCat.to_pandas()
        a = pd.merge(
            FinalCatp, ColumnCatp, left_on=radec1[0], right_on="id_test", how="left"
        ).drop("id_test", axis=1)
        return Table.from_pandas(a)  # .to_table()

    if new_field is None:
        new_field = field
    idx_ = idx[mask]
    for fieldi, new_field in zip(field, new_field):
        # #verboseprint("Adding field " + fieldi + " " + new_field)
        if new_field not in FinalCat.colnames:
            if type(ColumnCat[fieldi][0]) == np.ndarray:
                FinalCat[new_field] = (
                    np.ones((len(FinalCat), len(ColumnCat[fieldi][0]))) * -99.00
                )
            else:
                FinalCat[new_field] = -99.00
        # #verboseprint(FinalCat[new_field])
        FinalCat[new_field][mask] = ColumnCat[fieldi][idx_]
        # #verboseprint(FinalCat[new_field])
    return FinalCat


#%%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def init_values(bins, val, val_os, plot_=False):
    from pyds9plugin.DS9Utils import PlotFit1D
    bins_os = bins
    os_v = val_os
    bias = bins_os[np.nanargmax(val_os)] + 0.5  #  #ADDED xdata[np.nanargmax(ydata)]
    if bias > 1500:
        conversion_gain = 0.53
        smearing = 1.5
        gain_factor = 2
        RN = 60
    else:
        conversion_gain = 1 / 4.5
        smearing = 0.7
        RN = 10
        gain_factor = 1.2
    # mask_RN_os = (bins > bias - 1 * RN) & (bins < bias + 0.8 * RN) & (val_os > 0)
    mask_RN_os = (
        (bins > bias - 1 * RN / conversion_gain)
        & (bins < bias + 0.8 * RN / conversion_gain)
        & (val_os > 0)
    )
    if plot_:
        fig, ax = plt.subplots()
        ax.semilogy(bins, val)
        ax.semilogy(bins, val_os)
        ax.semilogy(bins[mask_RN_os], val_os[mask_RN_os], ":k")
        ax.set_xlim((bins.min(), bins.max()))
        ax.set_ylim((val.min(), val_os.max()))
        # plt.show()
    else:
        ax = None

    RON = np.abs(
        PlotFit1D(
            bins[mask_RN_os],
            val_os[mask_RN_os],
            deg="gaus",
            ax=ax,
            # ax=inset_axes(ax,width="30%", height="30%", loc=3),
            plot_=plot_,
            P0=[np.max(val_os[mask_RN_os]), bias, RN, 0],
        )["popt"][2]
        / conversion_gain
    )
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCDhist(
        x, bias=Bias, RN=RN, EmGain=EmGain, flux=flux, smearing=smearing, sCIC=sCIC
    )

    try:
        upper_limit = bins[
            np.where(
                (bins > bias)
                & (np.convolve(val, np.ones(1), mode="same") == np.nanmin(val))
            )[0][0]
        ]
    except (ValueError, IndexError) as e:
        upper_limit = np.max(bins)

    try:
        upper_limit_os = bins[
            np.where(
                (bins_os[np.isfinite(os_v)] > bias)
                & (
                    np.convolve(os_v[np.isfinite(os_v)], np.ones(1), mode="same")
                    == np.nanmin(os_v.min)
                )
            )[0][0]
        ]
    except (ValueError, IndexError) as e:
        upper_limit_os = np.max(bins_os)
    mask_gain1 = (bins > bias + 2 * RON) & (bins < upper_limit)
    try:
        gain = (
            PlotFit1D(
                bins[mask_gain1],
                val[mask_gain1],
                ax=ax,
                deg="exp",
                plot_=plot_,
                P0=[np.max(val[mask_gain1]), 600, 0],
            )["popt"][1]
            / conversion_gain
        )
        # (
        #     -1
        #     / np.log(10)
        #     / conversion_gain
        #     # / PlotFit1D(bins[mask_gain1], np.log10(val[mask_gain1]), ax=ax,deg=1, plot_=plot_,)[
        #     / PlotFit1D(bins[mask_gain1], val[mask_gain1], ax=ax,deg='exp', plot_=plot_,P0=[30,1000,0])[
        #         "popt"
        #     ][1]
        # )
    except ValueError:
        gain = 1200

    gain *= gain_factor
    flux = (np.average(bins, weights=val) - np.average(bins, weights=val_os)) / (
        gain * conversion_gain
    )
    sCIC = 0.005
    gain = np.max([np.min([gain, 2000]), 500])
    flux = np.max([np.min([flux, 1.5]), 0.005])
    RON = np.max([np.min([RON, 150]), 20])
    if plot_:
        n_pix = np.nansum(val_os[np.isfinite(val_os)])  # # 1e6#
        ax.semilogy(bins[mask_gain1], val[mask_gain1], ":k")
        stoch1 = EMCCDhist(
            bins,
            bias=bias,
            RN=RON,
            EmGain=gain,
            flux=0,
            smearing=smearing,
            sCIC=sCIC,
            n_pix=n_pix,
        )
        stoch2 = EMCCDhist(
            bins,
            bias=bias,
            RN=RON,
            EmGain=gain,
            flux=flux,
            smearing=smearing,
            sCIC=sCIC,
            n_pix=n_pix,
        )
        # ax.semilogy(bins,10**stoch1,':',alpha=0.7)
        ax.semilogy(bins, 10 ** stoch2, ":", alpha=0.7)
    # print(bias, RON, gain, flux, smearing, sCIC)
    return bias, RON, gain, flux, smearing, sCIC


from ipywidgets import (
    Button,
    Layout,
    jslink,
    IntText,
    IntSlider,
    interactive,
    interact,
    HBox,
    Layout,
    VBox,
)

# %matplotlib widget
import ipywidgets as widgets
import numpy as np
from scipy.sparse import dia_matrix
from astropy.table import Table
import glob
import os
from astropy.io import fits

n_conv = 11


def EMCCDhist(
    x,
    bias=[1e3, 4.5e3, 1194],
    RN=[0, 200, 53],
    EmGain=[100, 10000, 5000],
    flux=[0.001, 1, 0.04],
    smearing=[0, 1, 0.01],
    sCIC=[0, 1, 0],
    n_pix=10 ** 5.3,
):
    from scipy.sparse import dia_matrix
    import inspect
    from astropy.table import Table
    from matplotlib.widgets import Button
    import numpy as np

    if bias > 1500:
        ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
    else:
        ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018

    def variable_smearing_kernels(
        image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
    ):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np

        n = 15
        smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
        if type_ == "exp":
            smearing_kernels = np.exp(
                -np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length
            )
        else:
            assert 0 <= Smearing <= 1
            smearing_kernels = np.power(Smearing, np.arange(n))[
                :, np.newaxis, np.newaxis
            ] / np.ones(smearing_length.shape)
        smearing_kernels /= smearing_kernels.sum(axis=0)
        return smearing_kernels

    def simulate_fireball_emccd_hist(
        x,
        ConversionGain,
        EmGain,
        Bias,
        RN,
        Smearing,
        SmearExpDecrement,
        n_registers,
        flux,
        sCIC=0,
        n_pix=n_pix,
    ):
        """Silumate EMCCD histogram
        """
        import numpy as np

        # try:
        #     y = globals()["y"]
        #     # print('len y=',len(y))
        #     # print(' y=',y[y>1])
        #     # print('sum',np.nansum(y[np.isfinite(y)]))
        #     n_pix = np.nansum(10 ** y[np.isfinite(10 ** y)])  # 1e6#
        #     # print("global", np.sum(10 ** y), n_pix)
        # except TypeError:
        #     n_pix = 10 ** 6.3
        #     # print("fixed", np.sum(10 ** y), n_pix)
        n = 1
        im = np.zeros(int(n_pix))  #
        im = np.zeros((1000, int(n_pix / 1000)))
        imaADU = np.random.gamma(
            np.random.poisson(np.nanmax([flux, 0]), size=im.shape), abs(EmGain)
        )
        # changing total sCIC (e-) into the percentage of pixels experiencing spurious electrons
        p_sCIC = sCIC  # / np.mean(
        #     1 / np.power(EmGain * ConversionGain, np.arange(604) / 604)
        # pixels in which sCIC electron might appear
        id_scic = np.random.rand(im.shape[0], im.shape[1]) < p_sCIC
        # stage of the EM register at which each sCIC e- appear
        register = np.random.randint(1, n_registers, size=id_scic.shape)
        # Compute and add the partial amplification for each sCIC pixel
        # when f=1e- gamma is equivalent to expoential law
        # should we add poisson here?
        imaADU += np.random.gamma(
            np.random.poisson(1 * id_scic), np.power(EmGain, register / n_registers)
        )
        # imaADU[id_scic] += np.random.gamma(1, np.power(EmGain, register / n_registers))
        imaADU *= ConversionGain
        # smearing data
        if Smearing > 0:
            n_smearing = 15
            smearing_kernels = variable_smearing_kernels(
                imaADU, Smearing, SmearExpDecrement
            )
            offsets = np.arange(n_smearing)
            A = dia_matrix(
                (smearing_kernels.reshape((n_smearing, -1)), offsets),
                shape=(imaADU.size, imaADU.size),
            )
            imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
        # adding read noise and bias
        read_noise = np.random.normal(0, abs(RN * ConversionGain), size=im.shape)
        imaADU += Bias
        imaADU += read_noise
        range = [np.nanmin(x), np.nanmax(x)]
        n, bins = np.histogram(imaADU.flatten(), bins=[x[0] - 1] + list(x))
        return n

    y = simulate_fireball_emccd_hist(
        x=x,
        ConversionGain=ConversionGain,  # 0.53,
        EmGain=EmGain,
        Bias=bias,
        RN=RN,
        Smearing=smearing,
        SmearExpDecrement=1e4,  # 1e4,  # 1e5 #2022=1e5, 2018=1e4...
        n_registers=604,
        flux=flux,
        sCIC=sCIC,
    )
    y[y == 0] = 1.0
    # y = y / (x[1] - x[0])
    return np.log10(y)


import io
import pandas as pd

from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")
import inspect
from functools import wraps


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    # names, varargs, keywords, defaults = inspect.getargspec(func)
    names, varargs, keywords, defaults, _, _, _ = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


class HistogramFitter(widgets.HBox):
    @initializer
    def __init__(self, file=None, flux=None, smearing=None, sCIC=None, emgain=None):
        super().__init__()
        self.output = widgets.Output()
        self.pre_path = (
            "/Users/Vincent/Github/fireball2-etc/notebooks/histograms/Histogram_"
        )
        self.params = Table.read(
            "/Users/Vincent/Github/fireball2-etc/notebooks/parameters.csv"
        )  # ,type=['S20']*20)

        if file is not None:
            Xinf, Xsup, Yinf, Ysup = [1130, 1430, 1300, 1900]
            fitsfile = fits.open(file)[0]
            data = fitsfile.data
            header = fitsfile.header
            ly, lx = data.shape
            gain, exptime, temp = (
                float(header["EMGAIN"]),
                float(header["EXPTIME"]),
                float(header["TEMPB"]),
            )
            date = header["DATE"]
            if lx > 2500:
                Xinf_os, Xsup_os = Xinf + 1000, Xsup + 1000
            else:
                Xinf_os, Xsup_os = Xinf - 1000, Xsup - 1000
            im = data[Yinf:Ysup, Xinf:Xsup]
            os = data[Yinf:Ysup, Xinf_os:Xsup_os]
            # print(os.shape, im.shape)
            median_im = np.nanmedian(im)
            min_, max_ = (np.nanpercentile(os, 0.4), np.nanpercentile(im, 99.8))
            val, bins = np.histogram(im.flatten(), bins=np.arange(min_, max_, 1))
            val_os, bins_os = np.histogram(os.flatten(), bins=np.arange(min_, max_, 1))

            self.x = (bins[1:] + bins[:-1]) / 2
            self.y = np.array(val, dtype=float)
            self.y_os = np.array(
                val_os, dtype=float
            )  # * os.size / len(os[np.isfinite(os)])
            # )  # TODO take care of this factor
            mask = np.isfinite(np.log10(self.y_os)) & np.isfinite(np.log10(self.y))
            self.x, self.y, self.y_os = self.x[mask], self.y[mask], self.y_os[mask]
            self.n_pix = np.nansum(self.y[np.isfinite(self.y)])
            path_name = "%s_%iT_%iG_%is.csv" % (date[:10], temp, gain, exptime)
            Table([self.x, self.y, self.y_os]).write(
                self.pre_path + path_name, overwrite=True
            )

        self.files = glob.glob(self.pre_path + "*20??*_*G_*.csv")
        self.files.sort()
        self.options = [file.replace(self.pre_path, "") for file in self.files][::-1]
        if file is None:
            a = Table.read(self.pre_path + self.options[0])
            self.x = a["col0"]  # - a['col0'].min()
            self.y = a["col1"]
            self.y_os = a["col2"]
            self.n_pix = np.nansum(self.y[np.isfinite(self.y)])  # # 1e6#
            path_name = self.options[0]

        self.file_w = widgets.Dropdown(
            options=self.options,
            value=path_name,
            description="Histogram",
            layout=Layout(width="430px"),
            continuous_update=False,
        )  # Combobox

        self.file = ""
        if file is None:
            a = Table.read(self.pre_path + self.file_w.value)
            self.x = a["col0"]  # - a['col0'].min()
            self.y = a["col1"]
            self.y_os = a["col2"]
            self.n_pix = np.nansum(self.y[np.isfinite(self.y)])  # # 1e6#
            # bins_os, os_v = bins[np.isfinite(os_v)], os_v[np.isfinite(os_v)]
            # try:
            #     header_exptime, header_gain = header["EXPTIME"], header["EMGAIN"]
            # except:
            #     pass

        # print(bias, RN, emgain, flux, smearing, sCIC)
        self.fig, self.ax = plt.subplots(figsize=(9, 6))
        self.ploty = self.ax.semilogy(
            self.x, self.y, c="k", alpha=0.2
        )  # ,label='Physical region')
        # self.y_conv = np.convolve(self.y, np.ones(n_conv) / n_conv, mode="same")
        self.y_conv = 10 ** np.convolve(
            np.log10(self.y), np.ones(n_conv) / n_conv, mode="same"
        )
        self.ploty_conv = self.ax.semilogy(
            self.x[np.isfinite(self.x)],
            self.y_conv[np.isfinite(self.x)],
            "k",
            alpha=0.8,
            label="Physical region",
        )
        # self.ax.semilogy(self.x,10**self.y_conv_log,'k:',alpha=0.5,label='Physical region')
        self.ploty_os = self.ax.semilogy(
            self.x, self.y_os, c="r", alpha=0.2
        )  # ,label='OS region')
        self.y_os_conv = 10 ** np.convolve(
            np.log10(self.y_os), np.ones(n_conv) / n_conv, mode="same"
        )
        self.ploty_os_conv = self.ax.semilogy(
            self.x, self.y_os_conv, "r--", alpha=0.8, label="OS region"
        )
        bias, RN, emgain, flux, smearing, sCIC = init_values(self.x, self.y, self.y_os)
        if self.emgain is None:
            self.emgain = emgain
        if self.smearing is None:
            self.smearing = smearing
        if self.flux is None:
            self.flux = flux
        if self.sCIC is None:
            self.sCIC = sCIC
        print(self.emgain, emgain)
        # sys.exit()
        self.stock_os = self.ax.semilogy(
            1, 1, ":", c="r", label="Stochastical model OS"
        )
        self.stock_phys = self.ax.semilogy(
            1, 1, ":", c="k", label="Stochastical model Phys"
        )
        # self.ax.semilogy(self.x,10**EMCCDhist(self.x, bias=bias, RN=RN, EmGain=emgain, flux=0, smearing=smearing, sCIC=sCIC,n_pix=self.n_pix),':',c='r',label='Stochastical model OS')
        # self.ax.semilogy(self.x,10**EMCCDhist(self.x, bias=bias, RN=RN, EmGain=emgain, flux=flux, smearing=smearing, sCIC=sCIC,n_pix=self.n_pix),':',c='k',label='Stochastical model Phys')
        self.ax.legend(loc="upper right", fontsize=12)
        self.ax.set_xlabel("ADU", fontsize=12)
        self.ax.set_ylabel("Number of occurence", fontsize=12)
        self.ax.set_xlim((self.x.min(), self.x.max()))
        self.ax.set_ylim((5e-1, 1.1 * self.y_os.max()))
        self.ax.set_title("Histogram fitting with zoom")
        self.fig.tight_layout()
        width = "400px"
        # file_w = widgets.Combobox(options=options,value=path,description='Histogram', layout=Layout(width= '500px'),continuous_update=False)#Combobox

        self.bias_w = widgets.FloatSlider(
            min=0,
            max=4000,
            value=bias,
            layout=Layout(width=width),
            step=0.1,
            readout_format=".1f",
        )
        self.rn_w = widgets.FloatSlider(
            min=0,
            max=200,
            value=RN,
            layout=Layout(width=width),
            step=0.1,
            readout_format=".1f",
        )
        self.gain_w = widgets.FloatSlider(
            min=500,
            max=4000,
            value=self.emgain,
            layout=Layout(width=width),
            step=0.1,
            readout_format=".1f",
        )

        self.flux_w = widgets.FloatRangeSlider(
            value=[0, self.flux],
            min=0.00,
            max=1.5,
            step=0.0001,
            readout_format=".3f",
            layout=Layout(width=width),
        )
        self.smearing_w = widgets.FloatSlider(
            min=0.0,
            max=1.8,
            step=0.01,
            value=self.smearing,
            layout=Layout(width=width),
            readout_format=".2f",
        )
        self.sCIC_w = widgets.FloatSlider(
            min=0.0,
            max=0.2,
            value=self.sCIC,
            step=0.0001,
            layout=Layout(width=width),
            readout_format=".4f",
        )
        # exp_w = widgets.FloatRangeSlider( min=3, max=5,value=[4,4], layout=Layout(width='500px'))

        self.fit_w = widgets.Button(
            value=False,
            description="Fit least square",
            disabled=False,
            button_style="",
            tooltip="Description",
        )  # ,icon='check')
        self.save_w = widgets.Button(
            value=True,
            description="Save parameters & next",
            disabled=False,
            button_style="",
            tooltip="Description",
        )  # ,icon='check')
        self.upload_w = widgets.FileUpload(accept="*.fits", multiple=True)

        wids = widgets.interactive(
            self.update,
            file=self.file_w,
            bias=self.bias_w,
            RN=self.rn_w,
            EmGain=self.gain_w,
            flux=self.flux_w,
            smearing=self.smearing_w,
            sCIC=self.sCIC_w,
            upload=self.upload_w,
        )  # ,fit=self.fit_w)#,SmearExpDecrement=exp)
        controls = VBox(
            [
                HBox([self.file_w, self.fit_w, self.save_w, self.upload_w]),
                HBox([self.bias_w, self.rn_w]),
                HBox([self.gain_w, self.flux_w]),
                HBox([self.smearing_w, self.sCIC_w]),
            ]
        )
        display(HBox([self.output, controls]))
        # self.update(self.file_w.value, 0, 0, self.emgain, [0, self.flux], self.smearing, self.smearing)
        self.update(
            self.file_w.value,
            self.rn_w.value,
            self.bias_w.value,
            self.gain_w.value,
            self.flux_w.value,
            self.smearing_w.value,
            self.sCIC_w.value,
        )

        # self.update(self.file_w.value, 0, 0, 0, [0, 0], 0, 0)
        # a = interact(view_image, file=file_w,bias=bias_w,RN=rn_w, EmGain=gain_w, flux_w=flux_w, smearing=smearing_w, sCIC=sCIC_w)#,SmearExpDecrement=exp)

        def save(event):  # , self):
            self.temperature = "-99"
            # self.date, self.DAQ, self.exposure = self.file.replace('-','').replace('G','').replace('s.csv','').split('/')[1].split('_')[1:]
            self.date, self.temperature, self.DAQ, self.exposure = (
                os.path.basename(self.file)
                .replace("-", "")
                .replace("G", "")
                .replace("s.csv", "")
                .split("_")[1:]
            )
            mask = self.params["name"] == self.file.replace(self.pre_path, "")
            if len(self.params[mask]) > 0:
                self.params.remove_rows(np.arange(len(self.params))[mask])
            self.params.add_row(
                [
                    self.file.replace(self.pre_path, ""),
                    self.date,
                    -int(self.temperature[:-1]),
                    self.DAQ,
                    self.exposure,
                    np.round(self.bias_w.value, 1),
                    np.round(self.rn_w.value, 1),
                    np.round(self.gain_w.value, 1),
                    np.round(self.flux_w.value[1], 4),
                    np.round(self.smearing_w.value, 2),
                    np.round(self.sCIC_w.value, 4),
                ]
            )
            # params[mask][params.colnames[2:]] = self.date, self.temperature, self.DAQ, self.exposure, self.bias_w.value,self.rn_w.value, self.gain_w.value, self.flux_w.value[1], self.smearing_w.value, self.sCIC_w.value      (self.self.flux_w.value[1]self.smearing_w.valueself.sCIC_w.value,4) ]),3) , np.round(,4),np.round
            self.params.write("parameters.csv", overwrite=True)
            index = self.options.index(self.file_w.value) + 1

            if index < len(self.options[index]):
                self.file_w.value = self.options[index + 1]
                self.file = ""
                self.update(self.file_w.value, 0, 0, 0, [0, 0], 0, 0)
            return

        def fit(event):  # , self):
            # bias, RN, emgain, flux, smearing, sCIC
            (
                self.bias_w.value,
                self.rn_w.value,
                self.gain_w.value,
                flux,
                self.smearing_w.value,
                self.sCIC_w.value,
            ) = init_values(self.x, self.y, self.y_os)
            self.flux_w.value = [0, flux]
            # print(bias, RN, emgain, flux, smearing, sCIC)
            # self.update(self.file.replace(self.pre_path,''),bias, RN, emgain, [flux,flux], smearing, sCIC )
            # np.convolve(n_log, np.ones(n_conv) / n_conv, mode="same")
            function = lambda bins, RN, EmGain, flux1, sCIC: np.convolve(
                EMCCDhist(
                    bins,
                    bias=self.bias_w.value,
                    RN=RN,
                    EmGain=EmGain,
                    flux=flux1,
                    smearing=self.smearing_w.value,
                    sCIC=sCIC,
                    n_pix=self.n_pix,
                ),
                np.ones(n_conv) / n_conv,
                mode="same",
            )
            p0 = [self.rn_w.value, self.gain_w.value, flux, self.sCIC_w.value]
            print("p0Rn %0.1f, gain %i flux %0.2f, scic %0.3f" % (*p0,))
            # popt,pcov = curve_fit(function,self.x,self.y,p0=p0,epsfcn=1)#[xdata < upper_limit]
            val_max = np.percentile(self.x, 90)
            print(val_max)
            bounds = [[0, 1000, 0.001, 0.002], [150, 2300, 1, 0.005]]
            mask = (self.x < val_max) & np.isfinite(np.log10(self.y_conv))
            popt, pcov = curve_fit(
                function, self.x[mask], np.log10(self.y_conv[mask]), p0=p0, epsfcn=1
            )  # [xdata < upper_limit]
            # popt,pcov = curve_fit(function,self.x[mask],np.log10(self.y_conv[mask]),p0=p0,bounds=bounds)#[xdata < upper_limit]
            print("popt Rn %0.1f, gain %i flux %0.2f, scic %0.3f" % (*popt,))
            print(
                "diff Rn %0.1f, gain %i flux %0.2f, scic %0.3f"
                % (*list(np.array(p0) - np.array(popt)),)
            )
            self.rn_w.value, self.gain_w.value, flux, self.sCIC_w.value = popt

            # function = lambda bins, RN, EmGain, flux1, smearing, sCIC:np.convolve(EMCCDhist(bins, bias=self.bias_w.value, RN=RN, EmGain=EmGain, flux=flux1, smearing=smearing, sCIC=sCIC,n_pix=self.n_pix), np.ones(n_conv) / n_conv, mode="same")
            # p0 = [self.rn_w.value, self.gain_w.value, flux, self.smearing_w.value, self.sCIC_w.value]
            # popt,pcov = curve_fit(function,self.x[mask],np.log10(self.y_conv[mask]),p0=p0,epsfcn=1)#[xdata < upper_limit]            print('popt',popt)
            # print('popt Rn %0.1f, gain %i flux %0.2f, smearing %0.2f, scic %0.3f'%(*popt,))
            # self.rn_w.value, self.gain_w.value, flux, self.smearing_w.value, self.sCIC_w.value = popt

            function = lambda bins, RN, EmGain, flux1, smearing: np.convolve(
                EMCCDhist(
                    bins,
                    bias=self.bias_w.value,
                    RN=RN,
                    EmGain=EmGain,
                    flux=flux1,
                    smearing=smearing,
                    sCIC=0.005,
                    n_pix=self.n_pix,
                ),
                np.ones(n_conv) / n_conv,
                mode="same",
            )
            p0 = [self.rn_w.value, self.gain_w.value, flux, self.smearing_w.value]
            popt, pcov = curve_fit(
                function, self.x[mask], np.log10(self.y_conv[mask]), p0=p0, epsfcn=1
            )  # [xdata < upper_limit]            print('popt',popt)
            print("popt Rn %0.1f, gain %i flux %0.2f, smearing %0.2f" % (*popt,))
            self.rn_w.value, self.gain_w.value, flux, self.smearing_w.value = popt
            self.sCIC_w.value = 0.005

            self.flux_w.value = [0, flux]
            self.update(
                self.file.replace(self.pre_path, ""),
                self.rn_w.value,
                self.bias_w.value,
                self.gain_w.value,
                self.flux_w.value,
                self.smearing_w.value,
                self.sCIC_w.value,
            )
            print("fit worked",)

        self.fit_w.on_click(fit)  # ,self)
        self.save_w.on_click(save)  # ,self)

    def update(
        self, file, RN, bias, EmGain, flux, smearing, sCIC
    ):  # ,upload):#,fit=self.fit_w)#,SmearExpDecrement=exp)
        with self.output:
            uploaded_file = self.upload_w.value
            for name in uploaded_file.keys():
                if ".fits" in name:
                    print(name, uploaded_file[name])

                if name not in self.options:
                    cat = pd.read_csv(io.BytesIO(uploaded_file[name]["content"]))
                    cat.to_csv(self.pre_path + name, index=False)
                    self.options.append(name)
            self.file_w.options = self.options

            if self.file != self.pre_path + file:
                a = Table.read(self.pre_path + file)
                self.x = a["col0"]  # - a['col0'].min()
                self.y = a["col1"]
                self.y_conv = 10 ** np.convolve(
                    np.log10(self.y), np.ones(n_conv) / n_conv, mode="same"
                )
                self.y_os = a["col2"]
                self.y_os_conv = 10 ** np.convolve(
                    np.log10(self.y_os), np.ones(n_conv) / n_conv, mode="same"
                )
                self.n_pix = np.nansum(self.y[np.isfinite(self.y)])  # # 1e6#
                mask = self.params["name"] == file
                if len(self.params[mask]) > 0:
                    bias, RN, EmGain, flux, smearing, sCIC = (
                        float(self.params[mask]["bias"]),
                        float(self.params[mask]["RN"]),
                        float(self.params[mask]["EmGain"]),
                        float(self.params[mask]["flux"]),
                        float(self.params[mask]["smearing"]),
                        float(self.params[mask]["sCIC"]),
                    )
                else:
                    pass
                    flux = flux[1]
                    # bias, RN, EmGain, flux, smearing, sCIC = init_values(
                    #     self.x, self.y, self.y_os
                    # )
                # print('bias, RN, EmGain, flux, smearing, sCIC',bias, RN, EmGain, flux, smearing, sCIC)
                flux1, flux2 = 0, flux
                try:
                    self.bias_w.value, self.bias_w.min, self.bias_w.max = (
                        bias,
                        self.x.min(),
                        self.x.max(),
                    )
                except Exception:
                    self.bias_w.value, self.bias_w.max, self.bias_w.min = (
                        bias,
                        self.x.max(),
                        self.x.min(),
                    )
                    pass
                (
                    self.rn_w.value,
                    self.gain_w.value,
                    self.flux_w.value,
                    self.smearing_w.value,
                    self.sCIC_w.value,
                ) = (RN, EmGain, [0, flux], smearing, sCIC)
                self.ax.set_title("Histogram fitting with zoom: " + file)
            else:
                flux1, flux2 = flux
            stoch1 = EMCCDhist(
                self.x,
                bias=bias,
                RN=RN,
                EmGain=EmGain,
                flux=flux1,
                smearing=smearing,
                sCIC=sCIC,
                n_pix=self.n_pix,
            )
            stoch2 = EMCCDhist(
                self.x,
                bias=bias,
                RN=RN,
                EmGain=EmGain,
                flux=flux2,
                smearing=smearing,
                sCIC=sCIC,
                n_pix=self.n_pix,
            )
            # self.ax.lines[0].set_data(self.x,self.y)
            self.ploty[0].set_data(self.x, self.y)
            self.ploty_conv[0].set_data(
                self.x[self.y_conv > 0], self.y_conv[self.y_conv > 0]
            )
            self.ploty_os[0].set_data(self.x, self.y_os)
            self.ploty_os_conv[0].set_data(
                self.x[self.y_os_conv > 0], self.y_os_conv[self.y_os_conv > 0]
            )

            # self.stock_os[0].set_data(self.x,10**stoch1)
            self.stock_os[0].set_data(
                self.x, 10 ** np.convolve(stoch1, np.ones(n_conv) / n_conv, mode="same")
            )
            self.stock_phys[0].set_data(
                self.x, 10 ** np.convolve(stoch2, np.ones(n_conv) / n_conv, mode="same")
            )
            # self.stock_phys[0].set_data(self.x,10**stoch2)
            if self.file != self.pre_path + file:
                if self.x.max() != self.ax.get_xlim()[1]:
                    self.ax.set_xlim((self.x.min(), self.x.max()))
                    self.ax.set_ylim(ymax=1.1 * self.y_os.max())
            self.file = self.pre_path + file

def addAtPos(M1, M2, center):
    """Add a matrix in a higher dimension matric at a given position
    """
    size_x, size_y = np.shape(M2)
    coor_x, coor_y = center 
    coor_x, coor_y = coor_x - int(size_x/2), coor_y - int(size_y/2)
    end_x, end_y = (coor_x + size_x), (coor_y + size_y)
    # try:
        
    sx, sy =  np.shape(M1[coor_x:end_x, coor_y:end_y])
    a = M1[coor_x:end_x, coor_y:end_y] + M2[:sx,:sy]
    M1[coor_x:end_x, coor_y:end_y] = a
    # except ValueError:
    #     pass
    return M1

def variable_smearing_kernels(
        image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
    ):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np

        n = 15
        smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
        if type_ == "exp":
            smearing_kernels = np.exp(
                -np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length
            )
        else:
            assert 0 <= Smearing <= 1
            smearing_kernels = np.power(Smearing, np.arange(n))[
                :, np.newaxis, np.newaxis
            ] / np.ones(smearing_length.shape)
        smearing_kernels /= smearing_kernels.sum(axis=0)
        return smearing_kernels


def SimulateFIREBallemCCDImage(
    conv_gain=0.53, EmGain=1500, Bias="Auto", RN=80, p_pCIC=0.0005, p_sCIC=0, Dark=5e-4, Smearing=0.7, SmearExpDecrement=50000, exposure=50, flux=1e-3, source="Slit", Rx=8, Ry=8, size=[100, 100], OSregions=[0, -1], name="Auto", spectra="-", cube="-", n_registers=604, sky=0,save=False,stack=1,readout_time=1.5, cosmic_ray_loss=None, counting=True, QE=0.45, field="targets_F2.csv",QElambda=True,atmlambda=True,fraction_lya=0.05):
    #%%
    # ConversionGain=0.53
    # EmGain=1500
    # Bias="Auto"
    # RN=80 
    # p_pCIC=0.0005
    # p_sCIC=0
    # Dark=0.5/3600
    # Smearing=0.7
    # SmearExpDecrement=50000
    # exposure=50
    # flux=1e-3
    # source="Field"
    # Rx=4
    # Ry=4
    # name="Auto"
    # n_registers=604
    # sky=0
    # save=False
    # stack=stack=1#int(3600*1/50)
    # readout_time=1.5
    # cosmic_ray_loss=None
    # size=[1058, 2069]
    # OSregions=[0, 1058]

    # size=[3216, 2069]
    # OSregions=[1066, 2124]
    # EmGain=1500; Bias=0; RN=80; p_pCIC=1; p_sCIC=0; Dark=1/3600; Smearing=1; SmearExpDecrement=50000; exposure=50; flux=1; sky=4; source="Spectra m=17"; Rx=8; Ry=8;  size=[100, 100]; OSregions=[0, 120]; name="Auto"; spectra="Spectra m=17"; cube="-"; n_registers=604; save=False;readout_time=5;stack=100;QE=0.5
    from astropy.modeling.functional_models import Gaussian2D, Gaussian1D
    from scipy.sparse import dia_matrix
    from scipy.interpolate import interp1d


    OS1, OS2 = OSregions
    # ConversionGain=1
    ConversionGain = conv_gain
    Bias=0
    image = np.zeros((size[1], size[0]), dtype="float64")
    image_stack = np.zeros((size[1], size[0]), dtype="float64")

    # dark & flux
    source_im = 0 * image[:, OSregions[0] : OSregions[1]]
    source_im_wo_atm = 0 * image[:, OSregions[0] : OSregions[1]]
    lx, ly = source_im.shape
    y = np.linspace(0, lx - 1, lx)
    x = np.linspace(0, ly - 1, ly)
    x, y = np.meshgrid(x, y)

    # Source definition. For now the flux is not normalized at all, need to fix this
    # Cubes still needs to be implememted, link to detector model or putting it here?
    # if os.path.isfile(cube):
    throughput = 0.13*0.9
    atm = 0.45
    area = 7854
    dispersion = 46.6/10
    wavelength=2000
            #%%
    if source == "Flat-field":
        source_im += flux
    elif source == "Dirac":
        source_im += Gaussian2D.evaluate(x, y,  flux, ly / 2, lx / 2, Ry, Rx, 0)
    elif "Spectra" in source:
        if "m=" not in source:
            # for file in glob.glob("/Users/Vincent/Downloads/FOS_spectra/FOS_spectra_for_FB/CIV/*.fits"):
            try:
                a = Table.read("Spectra/h_%sfos_spc.fits"%(source.split(" ")[-1]))
                slits = None#Table.read("Targets/2022/" + field).to_pandas()
                trans = Table.read("transmission_pix_resolution.csv")
                QE = Table.read("QE_2022.csv")
            except FileNotFoundError: 
                a = Table.read("/Users/Vincent/Github/notebooks/Spectra/h_%sfos_spc.fits"%(source.split(" ")[-1]))
                slits = Table.read("/Users/Vincent/Github/FireBallPipe/Calibration/Targets/2022/" + field).to_pandas()
                trans = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/transmission_pix_resolution.csv")
                QE = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/PSFDetector/efficiencies/QE_2022.csv")
            QE = interp1d(QE["wave"]*10,QE["QE_corr"])#
            trans["trans_conv"] = np.convolve(trans["col2"],np.ones(5)/5,mode="same")
            trans = trans[:-5]
            atm_trans =  interp1d([1500,2500]+list(trans["col1"]*10),[0,0] + list(trans["trans_conv"]))#

            a["photons"] = a["FLUX"]/9.93E-12   
            a["e_pix_sec"]  = a["photons"] * throughput * atm  * area /dispersion
            nsize,nsize2 = 100,500
            source_im=np.zeros((nsize,nsize2))
            source_im_wo_atm=np.zeros((nsize2,nsize))
            mask = (a["WAVELENGTH"]>1960) & (a["WAVELENGTH"]<2280)
            lmax = a["WAVELENGTH"][mask][np.argmax( a["e_pix_sec"][mask])]
            # plt.plot( a["WAVELENGTH"],a["e_pix_sec"])
            # plt.plot( a["WAVELENGTH"][mask],a["e_pix_sec"][mask])
            f = interp1d(a["WAVELENGTH"],a["e_pix_sec"])#
            profile =   Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx) /Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx).sum()
            subim = np.zeros((nsize2,nsize))
            wavelengths = np.linspace(lmax-nsize2/2/dispersion,lmax+nsize2/2/dispersion,nsize2)

            if 1==0:
                # source_im=np.zeros((100,100))
                # # plt.plot(a["WAVELENGTH"][mask],a["e_pix_exp"][mask])
                # profile =   Gaussian1D.evaluate(np.arange(100),  1,  50, Rx) /Gaussian1D.evaluate(np.arange(100),  1,  50, Rx).sum()
                # i = np.argmin(abs(a["WAVELENGTH"]-1960))
                # source_im[:,:] +=   profile
                # source_im = source_im.T*a["e_pix_sec"][i:i+100]
                
                
                fig,(ax0,ax1,ax2) = plt.subplots(3,1)
                ax0.fill_between(wavelengths, profile.max()*f(wavelengths),profile.max()* f(wavelengths) * atm_trans(wavelengths),label="Atmosphere impact",alpha=0.3)
                ax0.fill_between(wavelengths, profile.max()*f(wavelengths)* atm_trans(wavelengths)*QE(wavelengths),profile.max()* f(wavelengths) * atm_trans(wavelengths),label="QE impact",alpha=0.3)
                ax1.plot(wavelengths,f(wavelengths)/f(wavelengths).ptp(),label="Spectra")
                ax1.plot(wavelengths, f(wavelengths)* atm_trans(wavelengths)/(f(wavelengths)* atm_trans(wavelengths)).ptp(),label="Spectra * Atm")
                ax1.plot(wavelengths, f(wavelengths)* atm_trans(wavelengths)*QE(wavelengths)/( f(wavelengths)* atm_trans(wavelengths)*QE(wavelengths)).ptp(),label="Spectra * Atm * QE")
                ax2.plot(wavelengths,atm_trans(wavelengths) ,label="Atmosphere")
                ax2.plot(wavelengths,QE(wavelengths) ,label="QE")
                ax0.legend()
                ax1.legend()
                ax2.legend()
                ax0.set_ylabel("e/pix/sec")
                ax1.set_ylabel("Mornalized prof")
                ax2.set_ylabel("%")
                ax2.set_xlabel("wavelength")
                ax0.set_title(source.split(" ")[-1])
                fig.savefig("/Users/Vincent/Github/notebooks/Spectra/h_%sfos_spc.png"%(source.split(" ")[-1]))
                plt.show()
            QE = QE(wavelengths) if QElambda else QE(lmax) 
            atm_trans = atm_trans(wavelengths) if atmlambda else atm_trans(lmax) 
            source_im[:,:] +=  (subim+profile).T*f(wavelengths) * atm_trans * QE
            # source_im_wo_atm[:,:] +=  (subim+profile).T*f(wavelengths) #* atm_trans(wavelengths)
        else:
            #%%
            mag=float(source.split("m=")[-1])
            factor_lya = fraction_lya
            flux = 10**(-(mag-20.08)/2.5)*2.06*1E-16/((6.62E-34*300000000/(wavelength*0.0000000001)/0.0000001))
            elec_pix = flux * throughput * atm * QE * area /dispersion# should not be multiplied by exposure time here
            with_line = elec_pix*(1-factor_lya) + factor_lya * (3700/1)*elec_pix* Gaussian1D.evaluate(np.arange(size[0]),  1,  size[0]/2, Ry)/ Gaussian1D.evaluate(np.arange(size[0]),  1,  size[0]/2, Ry).sum()
            # source_im[50:55,:] += elec_pix #Gaussian2D.evaluate(x, y, flux, ly / 2, lx / 2, 100 * Ry, Rx, 0)
            profile =  np.outer(with_line,Gaussian1D.evaluate(np.arange(size[1]),  1,  50, Rx) /Gaussian1D.evaluate(np.arange(size[1]),  1,  50, Rx).sum())
            source_im = source_im.T
            source_im[:,:] += profile
            source_im = source_im.T

            # a = Table(data=([np.linspace(1500,2500,nsize2),np.zeros(nsize2)]),names=("WAVELENGTH","e_pix_sec"))
            # a["e_pix_sec"] = elec_pix*(1-factor_lya) + factor_lya * (3700/1)*elec_pix* Gaussian1D.evaluate(a["WAVELENGTH"],  1,  line["wave"], 8) 
            # f = interp1d(a["WAVELENGTH"],a["e_pix_sec"])
            # profile =   Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx) /Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx).sum()
            # subim = np.zeros((nsize2,nsize))
            # wavelengths = np.linspace(2060-yi/dispersion,2060+(1000-yi)/dispersion,nsize2)
            # source_im[int(xi-nsize/2):int(xi+nsize/2), OSregions[0] : OSregions[1]] +=  (subim+profile).T*f(wavelengths) * atm_trans(wavelengths) * QE(wavelengths)
            # source_im_wo_atm[int(xi-nsize/2):int(xi+nsize/2), OSregions[0] : OSregions[1]] +=  (subim+profile).T*f(wavelengths) #* atm_trans(wavelengths)

#%%
            # print(exposure*profile.max(), exposure*profile.sum())
    elif source == "Slit":
        ConvolveSlit2D_PSF_75muWidth = lambda xy, amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
        source_im += ConvolveSlit2D_PSF_75muWidth((x, y), flux , 9, ly / 2, lx / 2, Rx, Ry).reshape(lx, ly)
    elif source == "Fibre":
        fibre = convolvePSF(radius_hole=10, fwhmsPSF=[2.353 * Rx, 2.353 * Ry], unit=1, size=(201, 201), Plot=False)  # [:,OSregions[0]:OSregions[1]]
        source_im = addAtPos(source_im, fibre, (int(lx / 2), int(ly / 2)))
 
    elif source[:5] == "Field":
       #%%
        ConvolveSlit2D_PSF_75muWidth = lambda xy, amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
        ws = [2025, 2062, 2139]
        file = '/Users/Vincent/Github/fireball2-etc/notebooks/10pc/cube_204nm_guidance0.5arcsec_slit100um_total_fc_rb_detected.fits'#%(pc,wave,slit)
        gal=fits.open(file)[0].data * 0.7 #cf athmosphere was computed at 45km

        slits = Table.read("/Users/Vincent/Github/FireBallPipe/Calibration/Targets/2022/" + field).to_pandas()
        trans = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/transmission_pix_resolution.csv")
        QE = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/PSFDetector/efficiencies/QE_2022.csv")
        QE = interp1d(QE["wave"]*10,QE["QE_corr"])#
        trans["trans_conv"] = np.convolve(trans["col2"],np.ones(5)/5,mode="same")
        trans = trans[:-5]
        atm_trans =  interp1d([1500,2500]+list(trans["col1"]*10),[0,0] + list(trans["trans_conv"]))#
        # plt.plot( trans["col1"], trans["trans_conv"])
        
        #passer en pandas
        #couper ce qui sort du detecteur
        #puis tranformer direct xmm y_lin en detecteur sans prendre en compte le redshift
        # if "yline_mm" not in slits.columns:
        #     slits["yline_mm"] = slits["y_mm"]
        # print(len(slits))
        try:
            slits.loc[ ~np.isfinite(slits["NUV_ned"]), 'NUV_ned'] = slits.loc[ ~np.isfinite(slits["NUV_ned"])]["FLUX_g"]+2#29.9
        except KeyError:
            slits.loc[ ~np.isfinite(slits["NUV_ned"]), 'NUV_ned']  = 29.9
        slits["yline_mm"] = 0
        slits["em_line"] = 0
        slits["wave"] = 0
        slits["X_IMAGE"] = (slits["y_mm"]+6.5) / 0.013
        slits["Y_IMAGE"] =( -slits["x_mm"]+13) / 0.013
        queries = ["Z<0.01","(Z>0.044 & Z<0.072) | (Z>0.081 & Z<0.117)","(Z>0.285 & Z<0.320) | (Z>0.331 & Z<0.375)","(Z>0.59 & Z<0.682) | (Z>0.696 & Z<0.78) "," (Z>0.926 & Z<0.98)| (Z>0.996 & Z<1.062) ","(Z>1.184 & Z<1.245) | (Z>1.263 & Z<1.338)"]
        for q,line in zip(queries,[2060,1908.7,1549.5,1215.67,1033.8,911.8]):
            if len(slits.query(q))>0:
                slits.loc[slits.eval(q), 'em_line'] =  line
                slits.loc[slits.eval(q), 'wave'] = (slits.query(q)['Z']+1)* line
                slits.loc[slits.eval(q), 'yline_mm'] =  slits.query(q)['y_mm']  + ((slits.query(q)['Z']+1)* line-2060)*dispersion*0.013
                slits.loc[slits.eval(q), 'X_IMAGE_line'] =  slits.query(q)['X_IMAGE']  + ((slits.query(q)['Z']+1)* line-2060)*dispersion
        Table.from_pandas(slits).write("/Users/Vincent/Github/FireBallPipe/Calibration/Targets/2022/test/" + field,overwrite=True)        
        slits = slits.query("x_mm>-13  & x_mm<13 & y_mm>-6.55 & y_mm<6.55 & yline_mm>-6.55 & yline_mm<6.55 & X_IMAGE_line>-1000")

        xs = slits["Y_IMAGE"] 
        ys = slits["X_IMAGE"]  + OSregions[0]
        # ys = slits["X_IMAGE"] - 1066 + OSregions[0]
        nsize =50
        nsize2=len(source_im[0, OSregions[0] : OSregions[1]])

        for i, line in slits.iterrows():
            yi, xi, centre,mag = np.array(line["X_IMAGE"]  + OSregions[0]) - OS1, line["Y_IMAGE"] ,line["wave"],line["NUV_ned"]
            z = line["Z"]
            factor_lya = 0.05 if z>0.001 else 0
            # if ~np.isfinite(mag):
            #     mag = 26 
            wavelength=2000
            flux = 10**(-(mag-20.08)/2.5)*2.06*1E-16/((6.62E-34*300000000/(wavelength*0.0000000001)/0.0000001))
            elec_pix = flux * throughput * atm * area /dispersion# should not be multiplied by exposure time here
            if "MAIN_ID" in slits.columns:
                if line["MAIN_ID"].replace(" ", "") in ["7C1821+6419","87GB004432.0+030343","PG1538+47"]:
                    # if line["spectra"]!="None":
                    a = Table.read("/Users/Vincent/Github/FireBallPipe/Calibration/Targets/2022/h_%sfos_spc.fits"%(line["MAIN_ID"].replace(" ", "")))
                    a["photons"] = a["FLUX"]/9.93E-12   
                    a["e_pix_sec"]  = a["photons"]  * throughput * atm * area /dispersion
                    f = interp1d(a["WAVELENGTH"],a["e_pix_sec"])#
                else:
                    a = Table(data=([np.linspace(1500,2500,nsize2),np.zeros(nsize2)]),names=("WAVELENGTH","e_pix_sec"))
                    a["e_pix_sec"] = elec_pix*(1-factor_lya) + factor_lya * (3700/1)*elec_pix* Gaussian1D.evaluate(a["WAVELENGTH"],  1,  line["wave"], 8) 
                    f = interp1d(a["WAVELENGTH"],a["e_pix_sec"])
                    print(xi,line["X_IMAGE_line"])
            else:
                a = Table(data=([np.linspace(1500,2500,nsize2),np.zeros(nsize2)]),names=("WAVELENGTH","e_pix_sec"))
                a["e_pix_sec"] = elec_pix*(1-factor_lya) + factor_lya * (3700/1)*elec_pix* Gaussian1D.evaluate(a["WAVELENGTH"],  1,  line["wave"], 8) 
                f = interp1d(a["WAVELENGTH"],a["e_pix_sec"])
                print(xi,line["X_IMAGE_line"])
            
            profile =   Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx) /Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx).sum()
            subim = np.zeros((nsize2,nsize))
            wavelengths = np.linspace(2060-yi/dispersion,2060+(1000-yi)/dispersion,nsize2)
            source_im[int(xi-nsize/2):int(xi+nsize/2), OSregions[0] : OSregions[1]] +=  (subim+profile).T*f(wavelengths) * atm_trans(wavelengths) * QE(wavelengths)
            source_im_wo_atm[int(xi-nsize/2):int(xi+nsize/2), OSregions[0] : OSregions[1]] +=  (subim+profile).T*f(wavelengths) #* atm_trans(wavelengths)
            # source_im_wo_atm[-int(xi+nsize/2):-int(xi-nsize/2), OSregions[0] : OSregions[1]] +=  (subim+profile).T*f(wavelengths) #* atm_trans(wavelengths)
            if ~np.isfinite(f(wavelengths).max()):
                sys.exit()
            # source_im=source_im.T
            if "spectra" in slits.columns:
                if line["spectra"]!="None":
                    fig,(ax0,ax1,ax2) = plt.subplots(3,1)
                    ax0.fill_between(wavelengths, profile.max()*f(wavelengths),profile.max()* f(wavelengths) * atm_trans(wavelengths),label="Atmosphere impact",alpha=0.3)
                    ax0.fill_between(wavelengths, profile.max()*f(wavelengths)* atm_trans(wavelengths)*QE(wavelengths),profile.max()* f(wavelengths) * atm_trans(wavelengths),label="QE impact",alpha=0.3)
                    ax1.plot(wavelengths, f(wavelengths)/f(wavelengths).ptp(),label="Spectra")
                    ax1.plot(wavelengths, f(wavelengths)* atm_trans(wavelengths)/(f(wavelengths)* atm_trans(wavelengths)).ptp(),label="Spectra * Atm")
                    ax1.plot(wavelengths, f(wavelengths)* atm_trans(wavelengths)*QE(wavelengths)/( f(wavelengths)* atm_trans(wavelengths)*QE(wavelengths)).ptp(),label="Spectra * Atm * QE")
                    ax2.plot(wavelengths,atm_trans(wavelengths) ,label="Atmosphere")
                    ax2.plot(wavelengths,QE(wavelengths) ,label="QE")
                    ax0.legend()
                    ax1.legend()
                    ax2.legend()
                    ax0.set_ylabel("e/pix/sec")
                    ax1.set_ylabel("Mornalized prof")
                    ax2.set_ylabel("%")
                    ax2.set_xlabel("wavelength")
                    ax0.set_title(line["spectra"])
                    plt.show()

                    
                    #%%
        if 1==0:
                if 1==1:
                    wavelength=2000
                    flux = 10**(-(mag-20.08)/2.5)*2.06*1E-16/((6.62E-34*300000000/(wavelength*0.0000000001)/0.0000001))
                    elec_pix = flux * throughput * atm * QE * area /dispersion# should not be multiplied by exposure time here
                    # source_im[50:55,:] += elec_pix #Gaussian2D.evaluate(x, y, flux, ly / 2, lx / 2, 100 * Ry, Rx, 0)
                    n = 300
                    gal = np.zeros((n,n))
                    cont = Gaussian1D.evaluate(np.arange(n),  1,  int(n/2), Rx) 
                    new_cont = cont/cont.sum()
                    profile_cont =  (1-factor_lya) * elec_pix * new_cont
                    line = Gaussian2D.evaluate(np.meshgrid(np.arange(n),np.arange(n))[0],np.meshgrid(np.arange(n),np.arange(n))[1],  1,  int(n/2),int(n/2), Rx, 2*Ry,0) 
                    line /= line.sum()
                    profile_line =  factor_lya * (3700/1)*elec_pix* line * cont.sum()
                    gal[:,:] += profile_cont+profile_line
                    j = np.argmin(abs(centre-trans["col1"]))
                    gal_absorbed = gal.T*trans["trans_conv"][j-int(n/2):j+int(n/2)]
                    source_im = addAtPos(source_im, 1*gal_absorbed, [int(xi), int(yi)])
                    # imshow(gal.T)
                    # source_im = addAtPos(source_im, 1*profile_line, [int(xi), int(yi)])
                else:
                    #verboseprint(xi, yi)
                    i = np.argmin(abs(centre-trans["col1"]))
                    print(i)
                    gal2 = gal*trans["trans_conv"][i-50:i+50]
                    source_im = addAtPos(source_im, 1*gal2, [int(xi), int(yi)])

    else:

        pc = int(float(source.split('=')[1].split('%')[0]))
        wave = int(float(source.split('=')[3]))
        slit = int(float(source.split('=')[2].split('mu')[0]))
        file = '%spc/cube_%snm_guidance0.5arcsec_slit%sum_total_fc_rb_detected.fits'%(pc,wave,slit)
        fitsim = fits.open(file)[0].data * 0.7 #cf athmosphere was computed at 45km
        source_im[:fitsim.shape[0],:fitsim.shape[1]]+=fitsim

        
    source_im = (Dark + source_im + sky) * int(exposure)
    source_im_wo_atm = (Dark + source_im_wo_atm + sky) * int(exposure)
    y_pix=1000
    # print(len(source_im),source_im.shape)
    if readout_time > 10:
        cube = np.array([(readout_time/exposure/y_pix)*np.vstack((np.zeros((i,len(source_im))),source_im[::-1,:][:-i,:]))[::-1,:] for i in np.arange(1,len(source_im))],dtype=float)
        source_im = source_im+np.sum(cube,axis=0)
    if cosmic_ray_loss is None:
        cosmic_ray_loss = np.minimum(0.005*(exposure+readout_time/2),1)#+readout_time/2
    stack = np.max([int(stack * (1-cosmic_ray_loss)),1])
    cube_stack = -np.ones((int(stack),size[1], size[0]), dtype="int32")

    # print(cosmic_ray_loss)
    n_smearing=6
    # image[:, OSregions[0] : OSregions[1]] += source_im
    # print(image[:, OSregions[0] : OSregions[1]].shape,source_im.shape)
    image[:, OSregions[0] : OSregions[1]] += np.random.gamma( np.random.poisson(source_im) + np.array(np.random.rand(size[1], OSregions[1]-OSregions[0])<p_pCIC,dtype=int) , EmGain)
    # take into acount CR losses
    #18%
    # image_stack[:, OSregions[0] : OSregions[1]] = np.nanmean([np.where(np.random.rand(size[1], OSregions[1]-OSregions[0]) < cosmic_ray_loss/n_smearing,np.nan,1) * (np.random.gamma(np.random.poisson(source_im)  + np.array(np.random.rand(size[1], OSregions[1]-OSregions[0])<p_pCIC,dtype=int) , EmGain)) for i in range(int(stack))],axis=0)
    image_stack[:, OSregions[0] : OSregions[1]] = np.mean([(np.random.gamma(np.random.poisson(source_im)  + np.array(np.random.rand(size[1], OSregions[1]-OSregions[0])<p_pCIC,dtype=int) , EmGain)) for i in range(int(stack))],axis=0)
    
    # a = (np.where(np.random.rand(int(stack), size[1],OSregions[1]-OSregions[0]) < cosmic_ray_loss/n_smearing,np.nan,1) * np.array([ (np.random.gamma(np.random.poisson(source_im)  + np.array(np.random.rand( OSregions[1]-OSregions[0],size[1]).T<p_pCIC,dtype=int) , EmGain))  for i in range(int(stack))]))
    # Addition of the phyical image on the 2 overscan regions
#     image += source_im2
    image +=  np.random.gamma( np.array(np.random.rand(size[1], size[0])<p_sCIC,dtype=int) , np.random.randint(1, n_registers, size=image.shape))
    #30%
    image_stack += np.random.gamma( np.array(np.random.rand(size[1], size[0])<int(stack)*p_sCIC,dtype=int) , np.random.randint(1, n_registers, size=image.shape))
    if counting:
        a = np.array([ (np.random.gamma(np.random.poisson(source_im)  + np.array(np.random.rand( OSregions[1]-OSregions[0],size[1]).T<p_pCIC,dtype="int32") , EmGain))  for i in range(int(stack))])
        cube_stack[:,:, OSregions[0] : OSregions[1]] = a
        cube_stack += np.random.gamma( np.array(np.random.rand(int(stack),size[1], size[0])<int(stack)*p_sCIC,dtype=int) , np.random.randint(1, n_registers, size=image.shape)).astype("int32")

    #         # addition of pCIC (stil need to add sCIC before EM registers)
    #         prob_pCIC = np.random.rand(size[1], size[0])  # Draw a number prob in [0,1]
    #         image[prob_pCIC < p_pCIC] += 1
    #         source_im2_stack[prob_pCIC < p_pCIC*stack] += 1

    #         # EM amp (of source + dark + pCIC)
    #         id_nnul = image != 0
    #         image[id_nnul] = np.random.gamma(image[id_nnul], EmGain)

            # Addition of sCIC inside EM registers (ie partially amplified)
    #         prob_sCIC = np.random.rand(size[1], size[0])  # Draw a number prob in [0,1]
    #         id_scic = prob_sCIC < p_sCIC  # sCIC positions
    #         # partial amplification of sCIC
    #         register = np.random.randint(1, n_registers, size=id_scic.sum())  # Draw at which stage of the EM register the electorn is created
    #         image[id_scic] += np.random.exponential(np.power(EmGain, register / n_registers))

        # semaring post EM amp (sgest noise reduction)
        #TODO must add smearing for cube!
    if Smearing > 0:
        # smearing dependant on flux
        #2%
        smearing_kernels = variable_smearing_kernels(image, Smearing, SmearExpDecrement)
        offsets = np.arange(n_smearing)
        A = dia_matrix((smearing_kernels.reshape((n_smearing, -1)), offsets), shape=(image.size, image.size))

        image = A.dot(image.ravel()).reshape(image.shape)
        image_stack = A.dot(image_stack.ravel()).reshape(image_stack.shape)

    #     if readout_time > 0:
    #         # smearing dependant on flux
    #         smearing_kernels = variable_smearing_kernels(image.T, readout_time, SmearExpDecrement)#.swapaxes(1,2)
    #         offsets = np.arange(n_smearing)
    #         A = dia_matrix((smearing_kernels.reshape((n_smearing, -1)), offsets), shape=(image.size, image.size))#.swapaxes(0,1)

    #         image = A.dot(image.ravel()).reshape(image.shape)#.T
    #         image_stack = A.dot(image_stack.ravel()).reshape(image_stack.shape)#.T
            
            
        # read noise
    #14%
    type_ = "int32"
    type_ = "float64"
    readout = np.random.normal(Bias, RN, (size[1], size[0]))
    readout_stack = np.random.normal(Bias, RN/np.sqrt(int(stack)), (size[1], size[0]))
    if counting:
        readout_cube = np.random.normal(Bias, RN, (int(stack),size[1], size[0])).astype("int32")
        # print((np.random.rand(source_im.shape[0], source_im.shape[1]) < cosmic_ray_loss).mean())
        #TOKEEP  for cosmic ray masking readout[np.random.rand(source_im.shape[0], source_im.shape[1]) < cosmic_ray_loss]=np.nan
        #print(np.max(((image + readout) * ConversionGain).round()))
    #     if np.max(((image + readout) * ConversionGain).round()) > 2 ** 15:
    imaADU_wo_RN = (image * ConversionGain).round().astype(type_)
    imaADU_RN = (readout * ConversionGain).round().astype(type_)
    imaADU = ((image + 1*readout) * ConversionGain).round().astype(type_)
    imaADU_stack = ((image_stack + 1*readout_stack) * ConversionGain).round().astype(type_)
    if counting:
        imaADU_cube = ((cube_stack + 1*readout_cube) * ConversionGain).round().astype("int32")
    else:
        imaADU_cube = imaADU_stack
    return imaADU, imaADU_stack, imaADU_cube, source_im, source_im_wo_atm#imaADU_wo_RN, imaADU_RN

