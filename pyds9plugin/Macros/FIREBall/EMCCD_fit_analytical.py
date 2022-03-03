import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
import inspect
from astropy.table import Table
from matplotlib.widgets import Button
import numpy as np
from dataphile.graphics.widgets import Slider
from astropy.io import fits
from scipy.optimize import curve_fit



def emccd_model(xpapoint=None, path=None, smearing=1, argv=[]):
    """Plot EMCCD simulation
    """
    xpapoint = d.get("xpa").split("\t")[-1]
    if len(d.get("regions selected").split("\n")) > 3:
        im = getdata(xpapoint)
    else:
        im = d.get_pyfits()[0].data[1300:2000, 1172:2145]  # ,:800]#
    # val, bins = np.histogram(im.flatten(), bins=np.linspace(2000, 7000, 500))
    # min = np.nanpercentile(im,0.01)
    # max = np.nanpercentile(im,99.999)
    bias = np.nanmedian(im)
    min = bias - 500
    max = bias + 2000
    n=800#500
    val, bins = np.histogram(im.flatten(), bins=np.linspace(min, max, n))
    print(bins)
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
    from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD

    n = np.log10(np.sum([10 ** yi for yi in ydata]))
    lims = np.array([0, 2])
    x = np.linspace(np.nanmin(xdata), np.nanmax(xdata), len(ydata))

    bias =   xdata[np.argmax(ydata)]#PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][1]
    if bias>1500:
        ConversionGain = 0.53#1/4.5 #ADU/e-  0.53 in 2018 
        RN=45
    else: 
        ConversionGain = 1/4.5 #ADU/e-  0.53 in 2018 
        RN=10


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
        (-1e3, 4.5e3),
        (0, 300),
        (100, 2200),
        (0.001, 3),
        (0, 1),#Can not go higher because we only take the first 6 pixels
        # (1e4, 9e4),
        (0, 0.3),
    ]  # ,(0,1)]
    centers = [xdata[np.argmax(ydata)], 50, 1200, 0.01, 0, 0.01,]#, 1.5e4
    centers = [xdata[np.argmax(ydata)], RN/ConversionGain, 1200, 0.01, 0, 0.,]#, 1.5e4
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
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCD(x,Bias,RN,EmGain,flux,smearing,sCIC) #+(ydata.max()-f0.max())
        

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


    def fit(event):
        bins,value = xdata, ydata
        bias =   xdata[np.argmax(ydata)]#PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][1]
        if bias>1500:
            ConversionGain = 0.53#1/4.5 #ADU/e-  0.53 in 2018 
            RN=45
        else: 
            ConversionGain = 1/4.5 #ADU/e-  0.53 in 2018 
            RN=10

        mask_RN = (bins>bins[np.argmax(value)]-1*RN) & (bins<bins[np.argmax(value)]+0.8*RN)  &(value>0)
        ron =   np.abs(PlotFit1D(bins[mask_RN],10**value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][2]/ConversionGain)
        mask_gain1 = (bins>bins[np.argmax(value)]+4*RN) & (bins<bins[np.argmax(value)]+10*RN)  
        gain =   -1 /np.log(10)/ConversionGain/ PlotFit1D(bins[mask_gain1 & (value>0)],value[mask_gain1 & (value>0)],deg=1, plot_=False)['popt'][1]
        flux =  np.nanmax([0.01]+[(np.nanmean(im)-bias)/ (gain *ConversionGain)])
        ron =  np.nanmax([RN/ConversionGain]+[ron])
        vals = [bias,ron ,gain,flux,0.01,0.01,]
        bounds = ([bias-10,ron/10,gain/10,flux/10,0,0],[bias+10,10*ron,gain*10,flux*10,0.2,0.1])
        bounds = ([bias-10,ron/10,0,flux/10,0,0],[bias+10,200,gain*10,flux*10,0.1,0.1])
        print(vals)
        print(bounds[0])
        print(bounds[1])
        upper_limit = bias+20*RN/ConversionGain
        print(upper_limit)
        vals, pcov = curve_fit(EMCCD, xdata[xdata<upper_limit], ydata[xdata<upper_limit],p0=vals,bounds=bounds)#np.array(lims).T
        # vals, pcov = curve_fit(EMCCD, xdata[(xdata<upper_limit)][np.isfinite(y)], ydata[(xdata<upper_limit)][np.isfinite(y)],p0=vals,bounds=bounds)#np.array(lims).T
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
    ax.set_title(name)
    plt.show()
    return 1, 1


emccd_model(xpapoint=None, path=None, smearing=1, argv=[])
