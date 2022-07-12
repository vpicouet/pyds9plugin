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

from pyds9plugin.DS9Utils import verboseprint
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt


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
    verboseprint("cat 1 : %i lines" % (len(FinalCat)))
    verboseprint("cat 2 : %i lines" % (len(ColumnCat)))
    # print(ColumnCat['ZFLAG'])
    verboseprint(ColumnCat)
    if query is not None:
        ColumnCat = apply_query(
            cat=ColumnCat, query=query, path=None, new_path=None, delete=True
        )
        verboseprint(ColumnCat)
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
        verboseprint(catalog)
        verboseprint(c)
        idx, d2d, d3d = catalog.match_to_catalog_sky(c)
        mask = 3600 * np.array(d2d) < distance
        verboseprint("Number of matches < %0.2f arcsec :  %i " % (distance, mask.sum()))

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
        verboseprint("Adding field " + fieldi + " " + new_field)
        if new_field not in FinalCat.colnames:
            if type(ColumnCat[fieldi][0]) == np.ndarray:
                FinalCat[new_field] = (
                    np.ones((len(FinalCat), len(ColumnCat[fieldi][0]))) * -99.00
                )
            else:
                FinalCat[new_field] = -99.00
        verboseprint(FinalCat[new_field])
        FinalCat[new_field][mask] = ColumnCat[fieldi][idx_]
        # verboseprint(FinalCat[new_field])
    return FinalCat


#%%

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pyds9plugin.DS9Utils import PlotFit1D


def init_values(bins, val, val_os, plot_=False):
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


# %%
