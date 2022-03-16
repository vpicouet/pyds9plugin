import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
import inspect
from astropy.table import Table
from matplotlib.widgets import Button
import numpy as np
from dataphile.graphics.widgets import Slider
from astropy.io import fits
from scipy.optimize import curve_fit

# from pyds9fb.DS9FB import calc_emccdParameters
# from pyds9plugin.DS9Utils import variable_smearing_kernels  # , EMCCD


# def variable_smearing_kernels(image, Smearing=1.5, SmearExpDecrement=50000):
#     """Creates variable smearing kernels for inversion
#     """
#     import numpy as np

#     smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
#     smearing_kernels = np.exp(
#         -np.arange(6)[:, np.newaxis, np.newaxis] / smearing_length
#     )
#     smearing_kernels /= smearing_kernels.sum(axis=0)
#     return smearing_kernels


def emccd_model(xpapoint=None, path=None, smearing=1, argv=[]):
    """Plot EMCCD simulation
    """

    xpapoint = d.get("xpa").split("\t")[-1]
    region = getregion(d, quick=True, message=False, selected=True)
    if region is not None:
        Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
    else:
        Xinf, Xsup, Yinf, Ysup = [1120, 2100, 1300, 1900]

    # if len(d.get("regions selected").split("\n")) > 3:
    #     im = getdata(xpapoint)
    # else:
    data = d.get_pyfits()[0].data
    im = data[Yinf:Ysup, Xinf:Xsup]
    os = data[Yinf:Ysup, Xinf - 1000 : Xsup - 1000]
    # val, bins = np.histogram(im.flatten(), bins=np.linspace(2000, 7000, 500))
    bias = np.nanmedian(im)
    # min_, max_ = bias - 500
    # max = bias + 2000
    min_, max_ = (np.nanpercentile(im, 0.1), np.nanpercentile(im, 99.9))
    # n=700
    val, bins = np.histogram(im.flatten(), bins=np.arange(min_, max_, 1))
    val_os, bins_os = np.histogram(os.flatten(), bins=np.arange(min_, max_, 1))
    bins = (bins[1:] + bins[:-1]) / 2
    val = np.array(val, dtype=float)
    val *= im.size / len(im[np.isfinite(im)])
    if path is not None:
        tab = Table.read(path)
        bins, val = (
            tab["col0"][tab["col0"] < 10000],
            tab["col1"][tab["col0"] < 10000],
        )

    val[(val == 0) & (bins > 3000)] = 1
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

    x = np.linspace(np.nanmin(xdata), np.nanmax(xdata), len(ydata))
    bias = xdata[
        np.argmax(ydata)
    ]  # PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][1]
    if bias > 1500:
        ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
        RN = 45
    else:
        ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018
        RN = 10

    dict_values = {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "x": x,
        "xdata": xdata,
        "ydata": ydata,
    }
    # EMCCD_new = lambda x, biais, RN, EmGain, flux: EMCCD(
    #     x, biais, RN, EmGain, flux, bright_surf=ydata
    # )  # -2
    lims = [
        (-1e3, 4.5e3),
        (0, 300),
        (100, 2200),
        (0.001, 6),
        (0, 3),
        (0, 0.3),
        # (1.5e3,1.5e5),
    ]  # ,(0,1)]
    centers = [
        xdata[np.argmax(ydata)],
        RN / ConversionGain,
        1200,
        0.01,
        0,
        0.01,
    ]  # , 1.5e4
    # centers = [xdata[np.argmax(ydata)], 50, 1200, 0.01, 0, 0.01, 1.5e4]

    f0 = EMCCDhist(x, *centers)
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCDhist(
        x, bias=Bias, RN=RN, EmGain=EmGain, flux=flux, smearing=smearing, sCIC=sCIC
    )
    args_number = len(inspect.getargspec(function).args) - 1

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.05 + 0.08 + args_number * 0.03)
    upper_limit = bins[
        np.where(
            (xdata[np.isfinite(ydata)] > centers[0])
            & (np.convolve(ydata[np.isfinite(ydata)], np.ones(1), mode="same") == 0)
        )[0][0]
    ]

    os_v = np.log10(np.array(val_os, dtype=float) * os.size / len(os[np.isfinite(os)]))
    bins_os, os_v = bins[np.isfinite(os_v)], os_v[np.isfinite(os_v)]
    upper_limit_os = bins[
        np.where(
            (bins_os[np.isfinite(os_v)] > centers[0])
            & (np.convolve(os_v[np.isfinite(os_v)], np.ones(1), mode="same") == 0)
        )[0][0]
    ]

    names = inspect.getargspec(function).args[1:]
    y = function(x, *centers)
    (datal,) = plt.plot(
        xdata[xdata < upper_limit],
        ydata[xdata < upper_limit],
        "-",
        c="black",
        label="Data",
    )
    plt.plot(
        xdata[xdata > upper_limit], ydata[xdata > upper_limit], ":", c="black",
    )
    plt.plot(
        bins_os[bins_os < upper_limit_os],
        os_v[bins_os < upper_limit_os],
        "-",
        c="black",
        label="OS",
        alpha=0.4,
    )
    plt.plot(
        bins_os[bins_os > upper_limit_os],
        os_v[bins_os > upper_limit_os],
        ":",
        c="black",
        alpha=0.4,
    )
    # (datal,) = plt.plot(xdata, ydata, "-", c="black", label="Data")
    # os_v = np.log10(np.array(val_os, dtype=float) * os.size / len(os[np.isfinite(os)]))
    # plt.plot(bins, os_v, "-", c="black", label="OS", alpha=0.4)

    (l,) = plt.plot(x, function(x, *centers), "-", lw=1, label="EMCCD model")
    ax.set_ylim((0.9 * np.nanmin(ydata), 1.1 * np.nanmax(ydata)))
    ax.set_ylabel("Log (Frequency of occurence)", fontsize=15)

    ax.margins(x=0)

    bounds_box = plt.axes([0.87, -0.029, 0.15, 0.15], facecolor="None")
    c = "white"
    hc = "0.975"
    button = Button(
        plt.axes([0.77, 0.025, 0.1, 0.04]), "Fit slope", color=c, hovercolor=hc,
    )
    button0 = Button(
        plt.axes([0.67 - 0.02 * 1, 0.025, 0.1, 0.04]),
        "Least square",
        color=c,
        hovercolor=hc,
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

    for edge in "left", "right", "top", "bottom":
        bounds_box.spines[edge].set_visible(False)

    def update(val):
        vals = []
        try:
            for slid in sliders:
                vals.append(slid.value)
                dict_values[slid.label] = slid.value
        except AttributeError:
            for slid in sliders:
                vals.append(slid.val)
                dict_values[slid.label] = slid.val

        x = dict_values["x"]
        l.set_ydata(function(x, *vals))

        fig.canvas.draw_idle()

        return

    sliders = []
    for i, (lim, center) in enumerate(zip(lims[::-1], centers[::-1])):
        # if names is None:
        #     slid = Slider(
        #         figure=fig,
        #         location=[0.3, 0.08 + i * 0.03, 0.6, 0.03],
        #         label="param %i" % (i),
        #         bounds=lim,
        #         init_value=np.array(lim).mean(),
        #     )
        # else:
        slid = Slider(
            figure=fig,
            location=[0.3, 0.08 + i * 0.03, 0.6, 0.03],
            label=names[::-1][i],
            bounds=lim,
            init_value=center,  # np.array(lim).mean(),
        )
        sliders.append(slid)
    sliders = sliders[::-1]
    for slider in sliders:
        slider.on_changed(update)

    def fit(event):
        bins, value = xdata, ydata
        mask_RN = (
            (bins > bins[np.argmax(value)] - 1 * RN)
            & (bins < bins[np.argmax(value)] + 0.8 * RN)
            & (value > 0)
        )
        ron = np.abs(
            PlotFit1D(
                bins[mask_RN],
                10 ** value[mask_RN],
                deg="gaus",
                plot_=False,
                P0=[1, bins[np.argmax(value)], 50, 0],
            )["popt"][2]
            / ConversionGain
        )
        mask_gain1 = (bins > bins[np.argmax(value)] + 4 * RN) & (
            bins < bins[np.argmax(value)] + 10 * RN
        )
        gain = (
            -1
            / np.log(10)
            / ConversionGain
            / PlotFit1D(
                bins[mask_gain1 & (value > 0)],
                value[mask_gain1 & (value > 0)],
                deg=1,
                plot_=False,
            )["popt"][1]
        )
        flux = np.nanmax([0.01] + [(np.nanmean(im) - bias) / (gain * ConversionGain)])
        ron = np.nanmax([RN / ConversionGain] + [ron])
        vals = [
            bias,
            ron,
            gain,
            flux,
            0.01,
            0.01,
        ]
        bounds = (
            [bias - 10, ron / 10, gain / 10, flux / 10, 0, 0],
            [bias + 10, 10 * ron, gain * 10, flux * 10, 0.2, 0.1],
        )
        bounds = (
            [bias - 10, ron / 10, 0, flux / 10, 0, 0],
            [bias + 10, 200, gain * 10, flux * 10, 0.1, 0.1],
        )
        # print(vals)
        # print(bounds[0])
        # print(bounds[1])
        upper_limit = bias + 20 * RN / ConversionGain
        # print(upper_limit)

        vals, pcov = curve_fit(
            EMCCD,
            xdata[xdata < upper_limit],
            ydata[xdata < upper_limit],
            p0=vals,
            bounds=bounds,
        )  # np.array(lims).T
        # print(vals)
        n = 6
        try:
            for slid in sliders[n:]:
                vals.append(slid.value)
        except AttributeError:
            for slid in sliders[n:]:
                vals.append(slid.val)
        l.set_ydata(function(x, *vals[:args_number]))
        plt.draw()

        for slid, vali in zip(sliders, vals):
            slid.widget.set_val(vali)

    button.on_clicked(fit)
    name = get_filename(d)
    plt.draw()
    ax.legend(loc="upper right", fontsize=15)
    ax.set_title(name, fontsize=11)
    plt.show()
    return

    def fit0(event):
        bins, value = xdata, ydata
        vals = [dict_values[slid.label] for slid in sliders]
        bounds = ([0.1 * val for val in vals], [10 * val for fal in vals])
        bounds = (
            [
                vals[0] - 100,
                vals[1] - 10,
                vals[2] / 2,
                vals[3] / 3,
                vals[4] - 0.01,
                vals[5] - 0.01,
            ],
            [
                vals[0] + 100,
                vals[1] + 10,
                vals[2] * 2,
                vals[3] * 3,
                vals[4] + 0.01,
                vals[5] + 0.1,
            ],
        )
        bounds = (-np.inf * vals, np.inf * vals)
        # print(vals)
        # print(bounds)
        # vals = [bias,ron ,gain,flux,0.01,0.01,]
        # bounds = ([bias-10,ron/10,gain/10,flux/10,0,0],[bias+10,10*ron,gain*10,flux*10,0.2,0.1])
        # upper_limit = bias+20*RN/ConversionGain
        # vals, pcov = curve_fit(EMCCD, xdata[xdata<upper_limit], ydata[xdata<upper_limit],p0=vals,bounds=bounds)#np.array(lims).T
        vals, pcov = curve_fit(
            EMCCD,
            xdata[xdata < upper_limit],
            ydata[xdata < upper_limit],
            p0=vals,
            bounds=bounds,
        )  # np.array(lims).T
        n = 6
        try:
            for slid in sliders[n:]:
                vals.append(slid.value)
        except AttributeError:
            for slid in sliders[n:]:
                vals.append(slid.val)
        l.set_ydata(function(x, *vals[:args_number]))
        plt.draw()

        for slid, vali in zip(sliders, vals):
            slid.widget.set_val(vali)

    def simplified_least_square_fit(event):
        bins, value = xdata, ydata
        vals = [dict_values[slid.label] for slid in sliders[:4]]
        bounds = ([0.1 * val for val in vals], [10 * val for fal in vals])
        bounds = (
            [vals[0] - 100, vals[1] - 10, vals[2] / 2, vals[3] / 3],
            [vals[0] + 100, vals[1] + 10, vals[2] * 2, vals[3] * 3],
        )
        # bounds = np.array(lims).T#(-np.inf*np.array(vals),np.inf*np.array(vals))
        # print(vals)
        # print(bounds)
        # vals = [bias,ron ,gain,flux,0.01,0.01,]
        # bounds = ([bias-10,ron/10,gain/10,flux/10,0,0],[bias+10,10*ron,gain*10,flux*10,0.2,0.1])
        # upper_limit = bias+20*RN/ConversionGain
        # vals, pcov = curve_fit(EMCCD, xdata[xdata<upper_limit], ydata[xdata<upper_limit],p0=vals,bounds=bounds)#np.array(lims).T
        model_to_fit = lambda bin_center, bias, ron, EmGain, flux: EMCCD(
            bin_center, bias, ron, EmGain, flux, 0, 0
        )
        model_to_fit_stoch = lambda bin_center, bias, ron, EmGain, flux: function(
            bin_center, bias, ron, EmGain, flux, 0, 0
        )
        print("ok")
        # vals, pcov = curve_fit(model_to_fit, xdata,ydata,p0=vals,bounds=bounds)#np.array(lims).T
        vals, pcov = curve_fit(
            model_to_fit,
            xdata[xdata < upper_limit],
            ydata[xdata < upper_limit],
            p0=vals,
        )
        # ,bounds=bounds)#np.array(lims).T
        print("ok")
        n = 6
        try:
            for slid in sliders[n:]:
                vals.append(slid.value)
        except AttributeError:
            for slid in sliders[n:]:
                vals.append(slid.val)
        l.set_ydata(model_to_fit_stoch(x, *vals[:args_number]))
        plt.draw()

        for slid, vali in zip(sliders, vals):
            slid.widget.set_val(vali)

    def fit_os(event):
        vals = [dict_values[slid.label] for slid in sliders]
        vals[vals.index(dict_values[sliders[3].label])] = 1e-5
        vals[-1] = 2e-2
        bounds = (
            [vals[0] - 100, vals[1] - 10, vals[2] / 2, 0, vals[4] - 0.01, 0,],
            [vals[0] + 100, vals[1] + 10, vals[2] * 2, 0.01, vals[4] + 0.01, 0.03,],
        )
        vals, pcov = curve_fit(
            EMCCD,
            bins_os[(bins_os < upper_limit_os)],
            os_v[(bins_os < upper_limit_os)],
            p0=vals,
        )
        n = 6
        try:
            for slid in sliders[n:]:
                vals.append(slid.value)
        except AttributeError:
            for slid in sliders[n:]:
                vals.append(slid.val)
        l.set_ydata(function(x, *vals[:args_number]))
        plt.draw()

        for slid, vali in zip(sliders, vals):
            slid.widget.set_val(vali)

    button0.on_clicked(fit0)
    button.on_clicked(fit)
    fit_os_button.on_clicked(fit_os)
    simplified_least_square.on_clicked(simplified_least_square_fit)

    name = get_filename(d)
    plt.draw()
    ax.legend(loc="upper right", fontsize=15)
    ax.set_title(name, fontsize=11)
    plt.show()
    return 1, 1


# if __name__ == "__main__":
emccd_model(xpapoint=None, path=None, smearing=1, argv=[])
