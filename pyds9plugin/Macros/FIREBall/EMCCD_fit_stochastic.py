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

import sys
sys.path.append("../../../../pyds9plugin")
from pyds9plugin.Macros.Fitting_Functions.functions import EMCCDhist

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
    if len(d.get("regions selected").split("\n")) > 3:
        verboseprint("Taking region")
        im = getdata(xpapoint)
    else:
        verboseprint("Taking nominal center region")
        im = d.get_pyfits()[0].data[1300:2000, 1172:2145]  # ,:800]#
    # val, bins = np.histogram(im.flatten(), bins=np.linspace(2000, 7000, 500))
    bias = np.median(im)
    min = bias - 500
    max = bias + 2000
    n=700
    val, bins = np.histogram(im.flatten(), bins=np.linspace(min, max, n))
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
    # EMCCD_new = lambda x, biais, RN, EmGain, flux: EMCCD(
    #     x, biais, RN, EmGain, flux, bright_surf=ydata
    # )  # -2
    lims = [
        (1e3, 4.5e3),
        (0, 200),
        (100, 2200),
        (0.001, 1),
        (0, 3),
        (0, 0.3),
        # (1.5e3,1.5e5),
    ]  # ,(0,1)]
    centers = [xdata[np.argmax(ydata)], 50, 1200, 0.01, 0, 0.01,]#, 1.5e4
    # centers = [xdata[np.argmax(ydata)], 50, 1200, 0.01, 0, 0.01, 1.5e4]

    f0 = EMCCDhist(x,*centers)
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCDhist(x, bias=Bias, RN=RN, EmGain=EmGain, flux=flux, smearing=smearing, sCIC=sCIC) #+(ydata.max()-f0.max())
    # function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC, smear_dec: EMCCDhist(x, bias=Bias, RN=RN, EmGain=EmGain, flux=flux, smearing=smearing, sCIC=sCIC,SmearExpDecrement=smear_dec) #+(ydata.max()-f0.max())

    # function = lambda x, Bias, RN, EmGain, flux, smearing, SmearExpDecrement, sCIC: np.log10(
    #     simulate_fireball_emccd_hist(
    #         x=x,
    #         data=im,
    #         ConversionGain=1 / 4.5,  # 0.53,
    #         EmGain=EmGain,
    #         Bias=Bias,
    #         RN=RN,
    #         p_pCIC=0,
    #         p_sCIC=0,
    #         Dark=0,
    #         Smearing=smearing,
    #         SmearExpDecrement=SmearExpDecrement,
    #         exposure=50,
    #         n_registers=604,
    #         flux=flux,
    #         sCIC=sCIC,
    #     )
    # )


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

    def reset(event):
        for slider in sliders:
            slider.reset()

    def fit(event):
        conversion_gain=1/4.5
        bins,value = xdata, ydata
        RN=50
        mask_RN = (bins>bins[np.argmax(value)]-1*RN) & (bins<bins[np.argmax(value)]+0.8*RN)  &(value>0)
        bias =   xdata[np.argmax(ydata)]#PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][1]
        ron =   np.abs(PlotFit1D(bins[mask_RN],10**value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][2]/conversion_gain)
        mask_gain1 = (bins>bins[np.argmax(value)]+4*RN) & (bins<bins[np.argmax(value)]+10*RN)  
        gain =   -1 /np.log(10)/conversion_gain/ PlotFit1D(bins[mask_gain1 & (value>0)],value[mask_gain1 & (value>0)],deg=1, plot_=False)['popt'][1]
        flux =  (np.mean(im)-bias)/ (gain *conversion_gain)
        vals = [bias,ron ,gain,flux,0,0]
        print(vals)
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
    return


emccd_model(xpapoint=None, path=None, smearing=1, argv=[])
