import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
import inspect
from astropy.table import Table
from matplotlib.widgets import Button
import numpy as np

# from dataphile.graphics.widgets import Slider
from matplotlib.widgets import Slider, RangeSlider

from astropy.io import fits
from scipy.optimize import curve_fit
from pyds9plugin.DS9Utils import *


def emccd_model(xpapoint=None, path=None, smearing=1, argv=[]):
    """Plot EMCCD simulation
    """
    if path is None:
        d = DS9n()
        xpapoint = d.get("xpa").split("\t")[-1]
        name = get_filename(d)
        region = getregion(d, quick=True, message=False, selected=True)
        if region is not None:
            Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
        else:
            Xinf, Xsup, Yinf, Ysup = [1130, 1430, 1300, 1900]
        # os = data[Yinf:Ysup, Xinf + 1000 : Xsup + 1000]

        fitsim = fits.open(name)[0]
        header = fitsim.header
        data = fitsim.data
        if len(data.shape)==3:
            im = data[0,Yinf:Ysup, Xinf:Xsup]
            os = data[0,Yinf:Ysup, Xinf + 1000 : Xsup + 1000]
        else:
            im = data[Yinf:Ysup, Xinf:Xsup]
            os = data[Yinf:Ysup, Xinf + 1000 : Xsup + 1000]
        median_im = np.nanmedian(im)
        min_, max_ = (np.nanpercentile(os, 0.1), np.nanpercentile(im, 99.8))
        print(os, min_, max_)
        val, bins = np.histogram(im.flatten(), bins=np.arange(min_, max_, 1))
        val_os, bins_os = np.histogram(os.flatten(), bins=np.arange(min_, max_, 1))
        bins = (bins[1:] + bins[:-1]) / 2
        val = np.array(val, dtype=float)
        os_v = np.log10(
            np.array(val_os, dtype=float) * os.size / len(os[np.isfinite(os)])
        )  # TODO thake care of this factor
        bins_os, os_v = bins[np.isfinite(os_v)], os_v[np.isfinite(os_v)]
        # header_exptime, header_gain = header["EXPTIME"], header["EMGAIN"]

        # val *= im.size / len(im[np.isfinite(im)])
    else:
        tab = Table.read(path)
        name = path
        bins, val, val_os = (
            tab["col0"][tab["col0"] < 10000],
            tab["col1"][tab["col0"] < 10000],
            tab["col2"][tab["col0"] < 10000],
        )
        bins_os = bins
        os_v = np.log10(np.array(val_os, dtype=float))
        median_im = np.average(bins, weights=val)
        header = None
        header_exptime, header_gain = -99, -99

    xdata, ydata = (
        bins[np.isfinite(np.log10(val))],
        np.log10(val)[np.isfinite(np.log10(val))],
    )
    np.savetxt("/tmp/xy.txt", np.array([xdata, ydata]).T)

    import sys

    sys.path.append("../../../../pyds9plugin")
    from pyds9plugin.Macros.Fitting_Functions.functions import EMCCDhist, EMCCD

    n = np.log10(np.sum([10 ** yi for yi in ydata]))
    lims = np.array([0, 2])

    x = np.linspace(
        np.nanmin(bins[np.isfinite(np.log10(val_os))]),
        np.nanmax(bins[np.isfinite(np.log10(val))]),
        len(ydata),
    )  # ADDED
    bias = bins_os[np.nanargmax(val_os)] + 0.5  #  #ADDED xdata[np.nanargmax(ydata)]
    # PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.nanargmax(value)],50,0])['popt'][1]
    if bias > 1500:
        conversion_gain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
        smearing = 0.8  # 1.5  # ADDED
        RN = 60  # 45 #ADDED
    else:
        conversion_gain = 1 / 4.5  # ADU/e-  0.53 in 2018
        smearing = 0.7  # ADDED
        RN = 10
    mask_RN_os = (bins > bias - 1 * RN) & (bins < bias + 0.8 * RN) & (val_os > 0)
    RON = np.abs(
        PlotFit1D(
            bins[mask_RN_os],
            val_os[mask_RN_os],
            deg="gaus",
            plot_=False,
            P0=[1, bias, RN, 0],
        )["popt"][2]
        / conversion_gain
    )
    RON = np.nanmax([RON] + [47])
    # print("bias = ", bias, xdata[np.nanargmax(ydata)])

    # centers = [xdata[np.nanargmax(ydata)], 50, 1200, 0.01, 0, 0.01, 1.5e4]

    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCDhist(
        x, bias=Bias, RN=RN, EmGain=EmGain, flux=flux, smearing=smearing, sCIC=sCIC
    )
    args_number = len(inspect.getargspec(function).args) - 1

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.05 + 0.08 + args_number * 0.03)
    try:
        upper_limit = bins[
            np.where(
                (xdata > bias)
                & (
                    np.convolve(10 ** ydata, np.ones(1), mode="same")
                    == np.nanmin(ydata)
                )
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
    mask_gain1 = (xdata > bias + 6 * RON) & (xdata < upper_limit)
    try:
        gain = (
            -1
            / np.log(10)
            / conversion_gain
            / PlotFit1D(xdata[mask_gain1], ydata[mask_gain1], deg=1, plot_=False,)[
                "popt"
            ][1]
        )
    except ValueError:
        gain = 1200
    # flux = np.nanmax([0.01] + [(np.nanmean(im) - bias) / (gain * conversion_gain)])
    flux = np.nanmax(
        [0.01]
        + [
            (np.average(bins, weights=val) - np.average(bins, weights=val_os))
            / (gain * conversion_gain)
        ]
    )

    dict_values = {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "x": x,
        "xdata": xdata,
        "ydata": ydata,
    }
    if (median_im - bias) > 1e3:
        flux_max = 25
    elif (median_im - bias) > 2e2:
        flux_max = 3  # ADDED
    elif (median_im - bias) > 1e2:
        flux_max = 1  # ADDED
    else:
        flux_max = 0.3  # ADDED
    lims = [
        (bins.min(), bins.min() + bins.ptp() / 2),
        (0, 300),
        (100, 3200),
        (0, flux_max),
        (0, 3),
        (0, 0.1),
    ]
    # (1.5e3,1.5e5),
    centers = [
        bias,  # ADDED
        RON,
        # RN / conversion_gain,
        gain,
        0,  # ADDED
        smearing,  # ADDED
        0.01,  # ADDED
    ]

    names = inspect.getargspec(function).args[1:]
    # print(names)
    names = ["Bias", "Read Noise", "EM gain", "sCIC - Flux", "Smearing", "mCIC"]
    plt.plot(
        xdata, ydata, "k", alpha=0.2,
    )
    n_conv = 11
    y_conv = np.convolve(ydata, np.ones(n_conv) / n_conv, mode="same")
    y_os_conv = np.convolve(os_v, np.ones(n_conv) / n_conv, mode="same")
    y = function(x, *centers)
    (datal,) = plt.plot(
        xdata[xdata < upper_limit],
        # ydata[xdata < upper_limit],
        y_conv[xdata < upper_limit],
        "-",
        c="black",
        # label="Data: Gconv=%0.2f\ntexp=%i\nG=%i"
        # % (conversion_gain, header_exptime, header_gain),
    )
    # plt.plot(
    #     xdata[xdata > upper_limit], ydata[xdata > upper_limit], ":", c="black",
    # )
    # print("y=", np.sum(10 ** ydata))

    plt.plot(
        bins_os[bins_os < upper_limit_os],
        # os_v[bins_os < upper_limit_os],
        y_os_conv[bins_os < upper_limit_os],
        ":",
        c="black",
        label="OS",
        alpha=1,
    )
    # plt.plot(
    #     bins_os[bins_os > upper_limit_os],
    #     os_v[bins_os > upper_limit_os],
    #     ":",
    #     c="black",
    #     alpha=0.4,
    # )

    plt.plot(
        bins_os, os_v, "k", alpha=0.2,
    )
    f1 = np.convolve(function(x, *centers), np.ones(n_conv) / n_conv, mode="same")
    (l1,) = plt.plot(x, f1, "-", lw=1, label="EMCCD OS model", alpha=0.7)
    (l2,) = plt.plot(x, f1, "-", c=l1.get_color(), lw=1, label="EMCCD model")
    ax.set_ylim((0.9 * np.nanmin(ydata), 1.1 * np.nanmax(os_v)))
    ax.set_ylabel("Log (Frequency of occurence)", fontsize=15)

    ax.margins(x=0)

    c = "white"
    hc = "0.975"
    button = Button(
        plt.axes([0.77, 0.025, 0.1, 0.04]), "Fit+smearing", color=c, hovercolor=hc,
    )
    button0 = Button(
        plt.axes([0.67 - 0.02 * 1, 0.025, 0.1, 0.04]), "Reset", color=c, hovercolor=hc,
    )
    simplified_least_square = Button(
        plt.axes([0.57 - 0.02 * 2, 0.025, 0.1, 0.04]),
        "Fit (B,RN,G,F)",
        color=c,
        hovercolor=hc,
    )

    fit_os_button = Button(
        plt.axes([0.47 - 0.02 * 3, 0.025, 0.1, 0.04]), "Fit OS", color=c, hovercolor=hc,
    )
    # Button(plt.axes([0.37 - 0.02 * 4, 0.025, 0.1, 0.04]),"Save").on_clicked(fig.savefig(name.replace('.fits','.png')))

    def update(val):
        vals1 = []
        vals2 = []
        for slid in sliders:
            vals1.append(slid.val)
            dict_values[slid.label.get_text()] = slid.val

        x = dict_values["x"]
        v1 = [v if type(v) != np.ndarray else v[0] for v in vals1]
        v2 = [v if type(v) != np.ndarray else v[1] for v in vals1]
        l1.set_ydata(
            np.convolve(function(x, *v1), np.ones(n_conv) / n_conv, mode="same")
        )
        l2.set_ydata(
            np.convolve(function(x, *v2), np.ones(n_conv) / n_conv, mode="same")
        )
        # l1.set_ydata(function(x, *v1))
        # l2.set_ydata(function(x, *v2))
        fig.canvas.draw_idle()
        return

    sliders = []
    for i, (lim, center, t) in enumerate(
        zip(lims[::-1], centers[::-1], [0, 0, 1, 0, 0, 0])
    ):
        if t == 1:
            slid = RangeSlider(
                figure=fig,
                ax=plt.axes([0.25, 0.08 + i * 0.03, 0.55, 0.03], facecolor="None"),
                label=names[::-1][i],
                valmin=lim[0],
                valmax=lim[1],
                valinit=(center, center + 0.1),
            )
        else:
            slid = Slider(
                figure=fig,
                ax=plt.axes([0.25, 0.08 + i * 0.03, 0.55, 0.03], facecolor="None"),
                label=names[::-1][i],
                valmin=lim[0],
                valmax=lim[1],
                valinit=center,
            )
        sliders.append(slid)
        dict_values[slid.label.get_text()] = slid.val  # ADDED

    sliders = sliders[::-1]
    for slider in sliders:
        slider.on_changed(update)

    def fit(event):
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
        vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]
        function = lambda x, RN, EmGain, flux, smearing: EMCCDhist(
            x,
            bias=vals1[0],
            RN=RN,
            EmGain=EmGain,
            flux=flux,
            smearing=smearing,
            sCIC=vals1[5],
        )
        p0 = [vals2[1], vals2[2], vals2[3], vals2[4]]
        # bounds = [[], []]
        # bounds = np.array([(0, 300), (100, 2200), (0.001, 6), (0, 3)]).T
        popt, pcov = curve_fit(
            function,
            xdata[xdata < upper_limit],
            ydata[xdata < upper_limit],
            p0=p0,
            epsfcn=1,
        )
        print(p0)
        print(popt)
        vals2[1], vals2[2], vals2[3], vals2[4] = popt  # , vals[4]
        # We do not want to add the flux
        vals1[1], vals1[2] = popt[:-2]  # , vals[4]
        vals1[4] = popt[-1]  # , vals[4]
        print(popt)

        l2.set_ydata(EMCCDhist(x, *vals2[:args_number]))
        l1.set_ydata(EMCCDhist(x, *vals1[:args_number]))
        plt.draw()
        for slid, val1i, val2i in zip(sliders, vals1, vals2):
            try:
                slid.set_val(val2i)
            except np.AxisError:
                slid.set_val([val1i, val2i])
            dict_values[slid.label.get_text()] = slid.val
        return

    def fit0(event):
        vals1 = centers
        vals2 = np.array(centers) + 0.1
        for slid, val1i, val2i in zip(sliders, vals1, vals2):
            try:
                slid.set_val(val1i)
            except np.AxisError:
                slid.set_val([val1i, val2i])
            dict_values[slid.label.get_text()] = slid.val

        return

    def simplified_least_square_fit(event):
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
        vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]
        function = lambda x, RN, EmGain, flux: np.convolve(
            EMCCDhist(
                x,
                bias=vals1[0],
                RN=RN,
                EmGain=EmGain,
                flux=flux,
                smearing=vals1[4],
                sCIC=vals1[5],
            ),
            np.ones(n_conv) / n_conv,
            mode="same",
        )
        # f1 = np.convolve(ydata, np.ones(n_conv) / n_conv, mode="same")

        p0 = [vals2[1], vals2[2], vals2[3]]  # , vals[4]
        # bounds = [[], []]
        # bounds = np.array([(0, 300), (100, 2200), (0.001, 6), (0, 3)]).T
        popt, pcov = curve_fit(
            function,
            xdata[xdata < upper_limit],
            y_conv[xdata < upper_limit],
            p0=p0,
            epsfcn=1,
        )
        print(p0)
        print(popt)
        vals2[1], vals2[2], vals2[3] = popt  # , vals[4]
        vals2[1], vals2[2] = popt[:-1]  # , vals[4]
        print(popt)

        l2.set_ydata(
            np.convolve(
                EMCCDhist(x, *vals2[:args_number]),
                np.ones(n_conv) / n_conv,
                mode="same",
            )
        )
        l1.set_ydata(
            np.convolve(
                EMCCDhist(x, *vals1[:args_number]),
                np.ones(n_conv) / n_conv,
                mode="same",
            )
        )

        # l2.set_ydata(function(x, *vals2[:args_number]))
        # l1.set_ydata(function(x, *vals1[:args_number]))
        plt.draw()
        for slid, val1i, val2i in zip(sliders, vals1, vals2):
            try:
                slid.set_val(val2i)
            except np.AxisError:
                slid.set_val([val1i, val2i])
            dict_values[slid.label.get_text()] = slid.val
        return

    def fit_os(event):
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
        vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]

        vals1[3] = 0
        function = lambda x, bias, RN, cic: EMCCDhist(
            x,
            bias=bias,
            RN=RN,
            EmGain=vals1[2],
            flux=vals1[3],
            smearing=vals1[4],
            sCIC=cic,
        )
        p0 = [vals1[0], vals1[1], vals1[-1]]

        popt, pcov = curve_fit(
            function,
            bins_os[(bins_os < upper_limit_os)],
            os_v[(bins_os < upper_limit_os)],
            p0=p0,
            epsfcn=1,
        )
        print("vals1 = ", vals1)
        vals1[0], vals1[1], vals1[-1] = popt
        vals2[0], vals2[1], vals2[-1] = popt
        l1.set_ydata(EMCCDhist(x, *vals1[:args_number]))
        l2.set_ydata(EMCCDhist(x, *vals2[:args_number]))
        print("p0 = ", p0)
        print("vals1 = ", vals1)
        plt.draw()
        for i, (slid, vali) in enumerate(zip(sliders, vals1)):
            try:
                slid.set_val(vals1[i])
            except np.AxisError:
                slid.set_val([vals1[i], vals2[i]])
            dict_values[slid.label.get_text()] = slid.val
        return

    button0.on_clicked(fit0)
    button.on_clicked(fit)
    fit_os_button.on_clicked(fit_os)
    simplified_least_square.on_clicked(simplified_least_square_fit)

    plt.draw()
    ax.legend(loc="upper right", fontsize=12, ncol=2)
    ax.set_title(name, fontsize=11)
    plt.show()
    return 1, 1