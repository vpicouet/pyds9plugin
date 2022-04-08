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


def emccd_model(xpapoint=None, path=None, smearing=1, argv=[]):
    """Plot EMCCD simulation
    """
    xpapoint = d.get("xpa").split("\t")[-1]
    region = getregion(d, quick=True, message=False, selected=True)
    if region is not None:
        Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
    else:
        Xinf, Xsup, Yinf, Ysup = [1120, 2100, 1300, 1900]
    data = d.get_pyfits()[0].data
    # if len(d.get("regions selected").split("\n")) > 3:
    #     im = getdata(xpapoint)
    # else:
    im = data[Yinf:Ysup, Xinf:Xsup]
    os = data[Yinf:Ysup, Xinf - 1000 : Xsup - 1000]

    # val, bins = np.histogram(im.flatten(), bins=np.linspace(2000, 7000, 500))
    # min = np.nanpercentile(im,0.01)
    # max = np.nanpercentile(im,99.999)
    bias = np.nanmedian(im)
    # min_, max_ = bias - 500
    # max = bias + 2000
    min_, max_ = (np.nanpercentile(os, 0.1), np.nanpercentile(im, 99.9))

    n = 800  # 500
    val, bins = np.histogram(im.flatten(), bins=np.arange(min_, max_, 1))
    val_os, bins_os = np.histogram(os.flatten(), bins=np.arange(min_, max_, 1))
    # print(bins)
    bins = (bins[1:] + bins[:-1]) / 2
    val = np.array(val, dtype=float)  # * im.size / len(im[np.isfinite(im)])
    if path is not None:
        tab = Table.read(path)
        bins, val = (
            tab["col0"][tab["col0"] < 10000],
            tab["col1"][tab["col0"] < 10000],
        )

    # val[(val == 0) & (bins > 3000)] = 1
    xdata, ydata = (
        bins[np.isfinite(np.log10(val))],
        np.log10(val)[np.isfinite(np.log10(val))],
    )
    # print("y", bins, val, ydata)
    np.savetxt("/tmp/xy.txt", np.array([xdata, ydata]).T)
    import sys

    sys.path.append("../../../../pyds9plugin")
    from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD

    n = np.log10(np.sum([10 ** yi for yi in ydata]))
    lims = np.array([0, 2])
    x = np.linspace(np.nanmin(xdata), np.nanmax(xdata), len(ydata))

    bias = bins[np.nanargmax(val_os)]
    if bias > 1500:
        ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
        RN = 45
    else:
        ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018
        RN = 10

    lims = [
        (bins.min(), bins.max()),
        (0, 300),
        (100, 2200),
        (1e-5, 0.1),
        (0, 1.5),
        (0, 0.3),
    ]
    centers = [bias, RN / ConversionGain, 1200, 0.001, 0, 0.0]
    # centers2 = centers+0.01
    # , 1.5e4
    # from pyds9plugin.Macros.Fitting_Functions import functions
    # from inspect import signature
    # function_ = getattr(functions, "EMCCD")
    # names = list(signature(function_).parameters.keys())[1:]
    # p_ = signature(function_).parameters
    # centers = [
    #     np.mean(p_[p].default)
    #     if len(p_[p].default) < 3
    #     else p_[p].default[2]
    #     for p in names
    # ]
    # lims = [(p_[p].default[0], p_[p].default[1]) for p, v in zip(names, values)]

    f0 = EMCCD(x, *centers)
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCD(
        x, Bias, RN, EmGain, flux, smearing, sCIC
    )  # +(ydata.max()-f0.max())

    args_number = len(inspect.getargspec(function).args) - 1
    # print(ydata)
    # print(centers[0], xdata, 10 ** ydata)
    upper_limit = bins[np.where((xdata > centers[0]) & (10 ** ydata == 1.0))[0][0]]

    os_v = np.log10(np.array(val_os, dtype=float) * os.size / len(os[np.isfinite(os)]))
    bins_os, os_v = bins[np.isfinite(os_v)], os_v[np.isfinite(os_v)]
    # print(centers[0], bins_os, 10 ** os_v)
    upper_limit_os = bins[np.where((bins_os > centers[0]) & (10 ** os_v == 1.0))[0][0]]

    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.05 + 0.08 + args_number * 0.03)

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
    (l1,) = plt.plot(x, function(x, *centers), "-", c="k", lw=1, label="EMCCD model")
    (l2,) = plt.plot(x, function(x, *centers), "-", c="k", lw=1, label="EMCCD model")
    # ax.set_ylim((0.9 * np.nanmin(ydata), 1.1 * np.nanmax(ydata)))
    ax.set_ylim((0.9 * np.nanmin(ydata), 1.1 * np.nanmax(os_v)))

    ax.set_ylabel("Log (Frequency of occurence)", fontsize=15)

    ax.margins(x=0)

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

    def update(val):
        vals1 = []
        vals2 = []
        try:
            for slid in sliders:
                vals1.append(slid.value)
                # vals2.append(slid.value[1])
                dict_values[slid.label] = slid.value
        except AttributeError:
            for slid in sliders:
                vals1.append(slid.val)
                # vals2.append(slid.val)
                dict_values[slid.label] = slid.val

        x = dict_values["x"]
        # print([type(v)for v in vals1])
        v1 = [v if type(v) != np.ndarray else v[0] for v in vals1]
        v2 = [v if type(v) != np.ndarray else v[1] for v in vals1]
        # print(v1, v2)
        l1.set_ydata(function(x, *v1))
        l2.set_ydata(function(x, *v2))
        fig.canvas.draw_idle()
        return

    sliders = []
    for i, (lim, center, t) in enumerate(
        zip(lims[::-1], centers[::-1], [0, 0, 1, 0, 0, 0])
    ):
        if t == 1:
            slid = RangeSlider(
                figure=fig,
                ax=plt.axes([0.3, 0.08 + i * 0.03, 0.6, 0.03], facecolor="None"),
                label=names[::-1][i],
                valmin=lim[0],
                valmax=lim[1],
                valinit=(center, center + 0.1),
            )
        else:
            slid = Slider(
                figure=fig,
                ax=plt.axes([0.3, 0.08 + i * 0.03, 0.6, 0.03], facecolor="None"),
                label=names[::-1][i],
                valmin=lim[0],
                valmax=lim[1],
                valinit=center,
            )

        # slid = Slider(
        #     figure=fig,
        #     location=[0.3, 0.08 + i * 0.03, 0.6, 0.03],
        #     label=names[::-1][i],
        #     bounds=lim,
        #     init_value=center,
        # )
        sliders.append(slid)
    sliders = sliders[::-1]

    dict_values = {
        "x": xdata,
        "y": xdata,
        "xdata": xdata,
        "ydata": ydata,
    }
    for slid in sliders:
        dict_values[slid.label] = slid.val

    for slider in sliders:
        slider.on_changed(update)

    def fit(event):
        bins, value = xdata, ydata
        bias = xdata[
            np.nanargmax(ydata)
        ]  # PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.nanargmax(value)],50,0])['popt'][1]
        if bias > 1500:
            ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
            RN = 45
        else:
            ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018
            RN = 10

        mask_RN = (
            (bins > bins[np.nanargmax(value)] - 1 * RN)
            & (bins < bins[np.nanargmax(value)] + 0.8 * RN)
            & (value > 0)
        )
        ron = np.abs(
            PlotFit1D(
                bins[mask_RN],
                10 ** value[mask_RN],
                deg="gaus",
                plot_=False,
                P0=[1, bins[np.nanargmax(value)], 50, 0],
            )["popt"][2]
            / ConversionGain
        )
        mask_gain1 = (bins > bins[np.nanargmax(value)] + 4 * RN) & (
            bins < bins[np.nanargmax(value)] + 10 * RN
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
        # upper_limit = bias + 20 * RN / ConversionGain
        # upper_limit = bins[
        #     np.where(
        #         (xdata[np.isfinite(ydata)] > vals[0])
        #         & (np.convolve(ydata[np.isfinite(ydata)], np.ones(1), mode="same") == 0)
        #     )[0][0]
        # ]
        # print(upper_limit)
        # TODO create first optimization and second if optimization if the values are equal to first optimization
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
        # bounds = np.array(lims).T#(-np.inf*np.array(vals),np.inf*np.array(vals))
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
            [vals[0] + 100, vals[1] + 10, vals[2] * 2, 1],
        )
        # bounds = np.array(lims).T#(-np.inf*np.array(vals),np.inf*np.array(vals))
        # print(vals)
        # print(bounds)
        # vals = [bias,ron ,gain,flux,0.01,0.01,]
        # bounds = ([bias-10,ron/10,gain/10,flux/10,0,0],[bias+10,10*ron,gain*10,flux*10,0.2,0.1])
        # upper_limit = bias+20*RN/ConversionGain
        # vals, pcov = curve_fit(EMCCD, xdata[xdata<upper_limit], ydata[xdata<upper_limit],p0=vals,bounds=bounds)#np.array(lims).T

        model_to_fit = lambda bin_center, bias, ron, EmGain, flux: function(
            bin_center,
            bias,
            ron,
            EmGain,
            flux,
            dict_values[sliders[-2].label],
            dict_values[sliders[-1].label],
        )
        # vals, pcov = curve_fit(model_to_fit, xdata,ydata,p0=vals,bounds=bounds)#np.array(lims).T
        vals, pcov = curve_fit(
            model_to_fit,
            xdata[xdata < upper_limit],
            ydata[xdata < upper_limit],
            p0=vals,
        )  # ,bounds=bounds)#np.array(lims).T
        n = 6
        try:
            for slid in sliders[n:]:
                vals.append(slid.value)
        except AttributeError:
            for slid in sliders[n:]:
                vals.append(slid.val)
        l.set_ydata(model_to_fit(x, *vals[:args_number]))
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
    ax.set_title("/".join(name.split("/")[5:]), fontsize=10)
    plt.show()
    return 1, 1


# if __name__ == "__main__":
emccd_model(xpapoint=None, path=None, smearing=1, argv=[])
