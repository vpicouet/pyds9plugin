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
# from Fitting_Functions import EMCCD
# from pyds9 import DS9
# d=DS9('7f000001:53722')
# sys.path.append("../../../pyds9plugin")
# sys.path.append("../../pyds9plugin")
import sys
sys.path.append("../../../../pyds9plugin")
from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD
def emccd_model(xpapoint=None, path=None, smearing=1, argv=[]):
    """Plot EMCCD simulation
    """
    xpapoint = d.get("xpa").split("\t")[-1]
    if len(d.get("regions selected").split("\n")) > 3:
        im = getdata(xpapoint)
    else:
        im = d.get_pyfits()[0].data[1300:2000, 1172:2145]  # ,:800]#
    # val, bins = np.histogram(im.flatten(), bins=np.linspace(2000, 7000, 500))
    val, bins = np.histogram(im.flatten(), bins=np.linspace(1000, 3000, 500))
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
    n = np.log10(np.sum([10 ** yi for yi in ydata]))
    lims = np.array([0, 2])
    x = np.linspace(np.nanmin(xdata), np.nanmax(xdata), len(ydata))
    dict_values = {
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "x": x,
        "xdata": xdata,
        "ydata": ydata,
    }
    lims = [
        (1e3, 4.5e3),
        (0, 200),
        (100, 2200),
        (0.001, 1),
        (0, 3),
        # (1e4, 9e4),
        (0, 1),
    ]  # ,(0,1)]
    centers = [1191, 50, 800, 0.01, 0, 0,]#, 1.5e4
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


    f0 = EMCCD(x,*centers)
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCD(x,Bias,RN,EmGain,flux,smearing,sCIC) +(ydata.max()-f0.max())
        

        # [np.mean(np.array(lim)) for lim in lims]
    # else:
    #     function = EMCCD_new
    #     lims = [(2e3, 4.5e3), (80, 140), (100, 2200), (0.001, 1.5)]
    #     centers = [1191, 17, 400, 0.03, 0, 1.5e4, 0]
        # [np.mean(np.array(lim)) for lim in lims]

    args_number = len(inspect.getargspec(function).args) - 1

    fig, ax = plt.subplots(figsize=(10, 7))
    # plt.subplots_adjust(bottom=0.25)
    plt.subplots_adjust(bottom=0.05 + 0.08 + args_number * 0.03)

    names = inspect.getargspec(function).args[1:]
    y = function(x, *centers)
    (datal,) = plt.plot(xdata, ydata, "-", c="black", label="Data")
    (l,) = plt.plot(x, function(x, *centers), "-", lw=1, label="EMCCD model")
    ax.set_ylim((0.9 * np.nanmin(ydata), 1.1 * np.nanmax(ydata)))
    ax.set_ylabel("Log (Frequency of occurence)", fontsize=15)

    ax.margins(x=0)

    bounds_box = plt.axes([0.87, -0.029, 0.15, 0.15], facecolor="None")
    button = Button(
        plt.axes([0.77, 0.025, 0.1, 0.04]), "Fit", color="white", hovercolor="0.975",
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
        slid = Slider(
            figure=fig,
            location=[0.3, 0.08 + i * 0.03, 0.6, 0.03],
            label=names[::-1][i],
            bounds=lim,
            init_value=center)
        sliders.append(slid)
    sliders = sliders[::-1]
    for slider in sliders:
        slider.on_changed(update)

    def reset(event):
        for slider in sliders:
            slider.reset()

    def fit(event):
        vals = [bins[np.nanargmax(val)]]
        bias, sigma, emgain = list(calc_emccdParameters(xdata, ydata))
        vals = [bias, sigma, emgain]
        # print(args_number)
        if args_number == 4:
            new_function = lambda x, a: function(x, bias, sigma, emgain, a)
        else:
            new_function = lambda x, a: function(
                x, bias, sigma, emgain, a, smearing=0, SmearExpDecrement=50000
            )
        popt, pcov = curve_fit(
            new_function,
            xdata[(xdata < 5000) & (ydata > 1)],
            ydata[(xdata < 5000) & (ydata > 1)],
            p0=0.1,
        )  #
        # print(popt, pcov)
        vals.append(popt)
        vals.append(0)
        vals.append(50000)
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

    def onclick(event):
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, n)
        a = dict_values["a"]
        b = dict_values["b"]
        c = dict_values["c"]
        d = dict_values["d"]
        dict_values["x"] = x
        y = EMCCD_new(x, a, b, c, d)
        dict_values["y"] = y
        l.set_xdata(x)
        l.set_ydata(y)
        return

    name = get_filename(d)
    plt.draw()
    ax.legend(loc="upper right", fontsize=15)
    ax.set_title(name)
    plt.show()
    return 1, 1


emccd_model(xpapoint=None, path=None, smearing=1, argv=[])
