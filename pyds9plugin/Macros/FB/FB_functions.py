import matplotlib.pyplot as plt
from scipy.sparse import dia_matrix
import inspect
from astropy.table import Table
from matplotlib.widgets import Button
import numpy as np
from matplotlib.widgets import CheckButtons
# from dataphile.graphics.widgets import Slider
from matplotlib.widgets import Slider, RangeSlider

from astropy.io import fits
from scipy.optimize import curve_fit
from pyds9plugin.DS9Utils import *
color="k"
# if check_appearance():
#     plt.style.use('dark_background')
#     color="white"
# import warnings
# warnings.filterwarnings("ignore")


def emccd_model(
    xpapoint=None, path=None, smearing=0.5, gain=None, argv=[], stack=False, save=False, conversion_gain=1, fit="EMCCDhist",gui=True,RN=3,mCIC=0.02,sCIC=0.005,RON=None ):#RN in ADU
    """Plot EMCCD simulation
    """
    import os
    print("path = ", path)
    if ".fit" not in path:
        n_conv=1
        try:
            tab = Table.read(path)
        except Exception:
            tab = Table.read(path,format="csv")

        print("openning table ", path)
        name = path
        cols = tab.colnames
        bins, val, val_os = (
            tab[cols[0]][tab[cols[0]] < 10000],
            tab[cols[1]][tab[cols[0]] < 10000],
            tab[cols[2]][tab[cols[0]] < 10000],
        )
        bins_os = bins
        os_v = np.log10(np.array(val_os, dtype=float))
        median_im = np.average(bins, weights=val)
        header = None
        header_exptime, header_gain = -99, -99
        # header_exptime, header_gain = (
        #     float(path.split("_")[-1].split(".csv")[0][:-1]),
        #     float(path.split("_")[-2][:-1]),
        # )

    else:
        if path is None:
            d = DS9n()
            xpapoint = d.get("xpa").split("\t")[-1]
            name = get_filename(d)
        else:
            d = DS9n()
            name = path
            save = True
        fitsim = fits.open(name)
        if len(fitsim)>1:
            fitsim = fitsim[2]
        else:
            fitsim = fitsim[0]

        header = fitsim.header
        data = np.array(fitsim.data, dtype=float)
        data[(data==0) | (data<-1000)] = np.nan
        if len(data.shape) == 3:
            data = data[0,:,:]
        ly, lx = data.shape

        region = getregion(d, quick=True, message=False, selected=True)
        if region is not None:
            Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
        elif lx > 1500:
            # Xinf, Xsup, Yinf, Ysup = [1130, 1430, 1300, 1900]
            Xinf, Xsup, Yinf, Ysup = [1130, 1600, 300, 1900]
        else:
            Xinf, Xsup, Yinf, Ysup = [0, -1, 0, -1]

        # os = data[Yinf:Ysup, Xinf + 1000 : Xsup + 1000]

        if lx > 2500:
            Xinf_os, Xsup_os = Xinf + 1000, Xsup + 1000
        elif lx > 1500:
            Xinf_os, Xsup_os = Xinf - 1000, Xsup - 1000
        else:
            Xinf_os, Xsup_os = Xinf, Xsup

        if stack:
            paths = get(
                d, "Provide the path of images you want to analyze the histogram"
            )
            im = np.hstack(
                fits.open(file)[0].data[Yinf:Ysup, Xinf:Xsup]
                for file in globglob(paths)
            )
            osv = np.hstack(
                fits.open(file)[0].data[Yinf:Ysup, Xinf_os:Xsup_os]
                for file in globglob(paths)
            )
        else:
            im = data[Yinf:Ysup, Xinf:Xsup]
            osv = data[Yinf:Ysup, Xinf_os:Xsup_os]

        try:
            date = float(header["DATE"][:4])
        except KeyError:
            try:
                date = float(header["OBSDATE"][:4])
            except KeyError:
                date = 2023
        except TypeError:
            date = 2023
            print("No date keeping 2022 conversion gain")

        if conversion_gain<0.3:
            n_conv = 1#21
        else:
            n_conv = 11
            if RON is None:
                RON=35
        

        # if bias > 1500:
        # if date < 2020:
        #     # conversion_gain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
        #     # smearing = 0.8  # 1.5  # ADDED
        #     RN = 60  # 45 #ADDED
        # else:
        #     # conversion_gain = 1 / 4.5  # ADU/e-  0.53 in 2018
        #     # conversion_gain = 0.5  # ADU/e-  0.53 in 2018
        #     # smearing = 1.3  # 0.7  # ADDED
        #     RN = 20

        conv=1#.1
        # conv = conversion_gain
        # im = im/conversion_gain
        # osv = osv/conversion_gain
        # conversion_gain=1
        im_nan_fraction = np.isfinite(im).mean()
        os_nan_fraction = np.isfinite(osv).mean()
        # print(osv.shape, im.shape)
        median_im = np.nanmedian(im)
        min_, max_ = (int(np.nanpercentile(osv, 0.4)-5), int(np.nanpercentile(im, 99.8)))
        # print(im.shape,os.shape)
        # print(min_, max_,np.nanpercentile(im[np.isfinite(im)], 99.8),np.nanpercentile(im[np.isfinite(im)], 0.4),np.nanpercentile(im[np.isfinite(im)], 40))
        val, bins = np.histogram(im.flatten(), bins=0.5+np.arange(min_, max_, 1*conversion_gain))
        val_os, bins_os = np.histogram(osv.flatten(), bins=0.5+np.arange(min_, max_, 1*conversion_gain))
        # print(val_os,min_, max_,osv)
        bins = (bins[1:] + bins[:-1]) / 2
        t=1#0#1#2.5
        val = np.array(val, dtype=float) *t/ im_nan_fraction 
        os_v = np.log10(np.array(val_os *t/ os_nan_fraction, dtype=float)) 
        #* os.size / len(os[np.isfinite(os)])) 
        # TODO take care of this factor
        try:
            header_exptime, header_gain = header["EXPTIME"], header["EMGAIN"]
        except:
            header_exptime, header_gain = -99, -99
            pass
        try:
            temp_device = header["TEMPD"]
        except:
            temp_device = -99
            pass

            # try:
            # header_exptime, header_gain = (
            #     path.split("_")[-1].split(".csv")[0],
            #     path.split("_")[-2],
            # )
        # val_os = val_os / t  / os_nan_fraction
        # val *= im.size / len(im[np.isfinite(im)])

    xdata, ydata, ydata_os = (
        bins[np.isfinite(np.log10(val))],
        np.log10(val)[np.isfinite(np.log10(val))],os_v[np.isfinite(np.log10(val))]
    )
    bins_os, os_v = bins[np.isfinite(os_v)], os_v[np.isfinite(os_v)]
    np.savetxt("/tmp/xy.txt", np.array([xdata, ydata,ydata_os]).T)
    # np.savetxt("/tmp/"+os.path.basename(name).replace(".fits",".txt"), np.array([xdata, ydata,ydata_os]).T)
    # np.savetxt("/tmp/xy.txt", np.array([xdata, ydata,ydata_os]).T)

    import sys

    sys.path.append("../../../../pyds9plugin")
    if fit!="EMCCDhist":
        from pyds9plugin.Macros.Fitting_Functions.functions import EMCCDhist as EMCCD
        from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD as EMCCD2
    else:
        from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD
        from pyds9plugin.Macros.Fitting_Functions.functions import EMCCDhist as EMCCD2

    n = np.log10(np.sum([10 ** yi for yi in ydata]))
    lims = np.array([0, 2])

    x = np.linspace(
        np.nanmin(bins[np.isfinite(np.log10(val_os))]),
        np.nanmax(bins[np.isfinite(np.log10(val))]),
        len(ydata),
    )  # ADDED
    bias = bins_os[np.nanargmax(os_v)]  # + 0.5  #  #ADDED xdata[np.nanargmax(ydata)]


    # conversion_gain=1
    # print("conversion_gain = ", conversion_gain)
    if RON is None:
        lim_rn1, lim_rn2 = (
            bias - 1 * RN / conversion_gain,
            bias + 0.8 * RN / conversion_gain,
        )
        mask_RN_os = (bins > lim_rn1) & (bins < lim_rn2) & (val_os > 0)
        RON = np.abs(
            PlotFit1D(
                bins[mask_RN_os],
                val_os[mask_RN_os],
                deg="gaus",
                plot_=False,
                P0=[1, bias, RN, 0],
            )["popt"][2]
            # / conversion_gain
        )
        RON = np.nanmax([abs(RON)] + [1])
    else:
        lim_rn1, lim_rn2 = bias - 1 * RON,bias + 0.8 * RON
        # lim_rn1, lim_rn2 = (bias - 0.5 * RON,bias + 0.5 * RON,)
        lim_rn1, lim_rn2 = bias - 3 * RON,bias + 3 * RON
        mask_RN_os = (bins > lim_rn1) & (bins < lim_rn2) & (val_os > 0)
        lim_rn1, lim_rn2 = bias - 0.5 + 5.5 * RON,  bias - 0.5 + 5.5 * RON

        RON = np.abs(
            PlotFit1D(
                bins[mask_RN_os],
                val_os[mask_RN_os],
                deg="gaus",
                plot_=False,
                P0=[1, bias, RON/conversion_gain, 0],
            )["popt"][2]
            # / conversion_gain
        )
        # RON = np.nanmax([abs(RON)] + [1])
    # print("bias = ", bias, xdata[np.nanargmax(ydata)])

    # centers = [xdata[np.nanargmax(ydata)], 50, 1200, 0.01, 0, 0.01, 1.5e4]
    # print("RN",RN,RN * conversion_gain,RON,RON/conversion_gain)
    function = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCD(
        x,
        bias=Bias,
        RN=RN * conversion_gain,
        EmGain=EmGain * conversion_gain,
        flux=flux,
        smearing=smearing,
        sCIC=sCIC,
    )
    function2 = lambda x, Bias, RN, EmGain, flux, smearing, sCIC: EMCCD2(
        x,
        bias=Bias,
        RN=RN * conversion_gain,
        EmGain=EmGain * conversion_gain,
        flux=flux,
        smearing=smearing,
        sCIC=sCIC,
    )
    args_number = len(inspect.getargspec(function).args) - 1

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim((min_, max_))
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
    lim_gain_1 = bias + 4 * RON
    mask_gain1 = (xdata > lim_gain_1) & (xdata < upper_limit)
    if gain is None:
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
    # if (median_im - bias) > 1e3:
    #     flux_max = 25
    # elif (median_im - bias) > 2e2:
    #     flux_max = 3  # ADDED
    # else:# (median_im - bias) > 1e2:
    #     flux_max = 1#0.1#1  # ADDED



    # else:
    #     flux_max = 0.8  # ADDED
    lims = [
        (bins.min(), bins.min() + bins.ptp() / 2),
        (0, 300),
        (10, 8000),  # 3200
        (0, flux*10),
        (0, 3),
        (0, 0.2),
    ]
    # (1.5e3,1.5e5),
    if os.path.isfile("/tmp/emccd_fit_values.npy"):
        centers = np.genfromtxt("/tmp/emccd_fit_values.npy")
        
    else:
        centers = [
            bias-0.5,  # ADDED
            RON/ conversion_gain,
            # RN / conversion_gain,
            gain,
            sCIC,  # ADDED
            smearing,  # ADDED
            mCIC,  # ADDED0.01
        ]
    #chaaange
    # if gui is False:
    #     return {"BIAS":bias,"RON":RON/ conversion_gain,"GAIN":gain,"FLUX":flux}
    names = inspect.getargspec(function).args[1:]
    # print(names)
    names = [
        "Bias (ADU)",
        "Read Noise (e-)",
        "EM gain (e-/e-)",
        "sCIC - Flux (e-/exp)",
        "Smearing (pix)",
        "mCIC (e-/exp)",
    ]
    plt.plot(
        xdata, ydata, "grey", alpha=0.2,
    )
    plt.plot([lim_rn1, lim_rn1], [0, 5], ":", c="grey", alpha=0.7)
    plt.plot([lim_rn2, lim_rn2], [0, 5], ":", c="grey", alpha=0.7)
    # plt.plot([lim_gain_1, lim_gain_1], [0, 3], ":", c="grey", alpha=0.7)
    # plt.plot([upper_limit, upper_limit], [0, 3], ":", c="grey", alpha=0.7)
    y_conv = np.convolve(ydata, np.ones(n_conv) / n_conv, mode="same")
    y_os_conv = np.convolve(os_v, np.ones(n_conv) / n_conv, mode="same")
    y = function(x, *centers)
    (datal,) = ax.plot(
        xdata[xdata < upper_limit],
        # ydata[xdata < upper_limit],
        y_conv[xdata < upper_limit],
        "-",
        c=color,
        label="Data: Gconv=%0.2fADU/e-\ntexp=%ss\nG=%sDAQ"
        % (conversion_gain, header_exptime, header_gain),
        # label="Data: Gconv=%0.2fADU/e-\ntexp=%ss\n"#G=%sDAQ\nF=%0.1fe-/h"
        # % (conversion_gain, header_exptime)#, header_gain,flux*3600/header_exptime),


    )
    ax.plot(
        bins_os[bins_os < upper_limit_os],
        y_os_conv[bins_os < upper_limit_os],
        ":",
        c=color,
        label="OS",
        alpha=1,
    )


    ax.plot(bins_os, os_v, color, alpha=0.2)

    f1 = np.convolve(function(x, *centers), np.ones(n_conv) / n_conv, mode="same")
    centers_ = np.array(centers) + [0, 0, 0, flux, 0, 0]
    # centers_[-3] = flux
    f2 = np.convolve(function(x, *centers_), np.ones(n_conv) / n_conv, mode="same")
    fraction_thresholded = 100*np.sum(10**y_conv[xdata>bias-0.5 + 5.5 *RON ])/np.sum(10**y_conv)
    # (l1,) = ax.plot(x, f1, "-", lw=1, alpha=0.7, label="EMCCD OS model")     #\nFraction>5.5σ (RN=%0.1f)=%0.1f%%"%(RON,fraction_thresholded))
    (l1,) = ax.plot(x, f1, "-", lw=1, alpha=0.7, label="EMCCD OS model\nFraction>5.5σ (RN=%0.1f)=%0.1f%%"%(RON,fraction_thresholded))
    (l2,) = ax.plot(x, f2, "-", c=l1.get_color(), lw=1)

    if 1==0:
        (l3,) = ax.plot(x, np.convolve(function2(x, *centers), np.ones(n_conv) / n_conv, mode="same"), "-", c=l1.get_color(), lw=1, alpha=0.7)
        (l4,) = ax.plot(x, np.convolve(function2(x, *centers_), np.ones(n_conv) / n_conv, mode="same"), "-", c=l1.get_color(), lw=1)


      # , label="EMCCD model (Gconv=%0.2f)"%(conversion_gain))
    ax.set_ylim((0.9 * np.nanmin(ydata), 1.1 * np.nanmax(os_v)))
    ax.set_ylabel("Log (Frequency of occurence)", fontsize=15)

    ax.margins(x=0)

    c = "white"
    c =  '#FF000000'
    hc = "0.975"
    # button = Button(
    #     plt.axes([0.77, 0.025, 0.1, 0.04]), "Fit+smearing", color=c, hovercolor=hc,
    # )
    # button0 = Button(
    #     plt.axes([0.67 - 0.02 * 1, 0.025, 0.1, 0.04]),
    #     "Reset smear",
    #     color=c,
    #     hovercolor=hc,
    # )
    button0 = Button(
        plt.axes([0.67 - 0.02 * 1, 0.025, 0.1, 0.04]),
        "Save values",
        color=c,
        hovercolor=hc,
    )
    simplified_least_square = Button(
        plt.axes([0.57 - 0.02 * 2, 0.025, 0.1, 0.04]),
        "Fit (B,RN,G,F)",
        color=c,
        hovercolor=hc,
    )


    write_model = Button(
        plt.axes([0.47 - 0.02 * 3, 0.025, 0.1, 0.04]), "Gen image", color=c, hovercolor=hc,
    )

    # fit_os_button = Button(
    #     plt.axes([0.47 - 0.02 * 3, 0.025, 0.1, 0.04]), "Fit OS", color=c, hovercolor=hc,
    # )
    # Button(plt.axes([0.37 - 0.02 * 4, 0.025, 0.1, 0.04]),"Save").on_clicked(fig.savefig(name.replace('.fits','.png')))

    def update(val):
        vals1 = []
        vals2 = []
        for slid in sliders:
            vals1.append(slid.val)
            dict_values[slid.label.get_text()] = slid.val

        x = dict_values["x"]
        v1 = [v if ((type(v) != np.ndarray)&(type(v) != tuple)) else v[0] for v in vals1]
        v2 = [v if ((type(v) != np.ndarray)&(type(v) != tuple)) else v[1] for v in vals1]
        # print(vals1)

        # print("xdata[0]: ", xdata[0])
        # if xdata[0]%1<0.5:
        #     xdata_ = xdata #+ 1
        # else:
        #     xdata_ = xdata #- 1
        l2.set_ydata(
            np.convolve(
                function(xdata, *v2), #+1 else bias is shifted by 1 ADU
                np.ones(n_conv) / n_conv,
                mode="same",
            )
        )
        l1.set_ydata(
            np.convolve(
                function(xdata, *v1),
                np.ones(n_conv) / n_conv,
                mode="same",
            )
        )
        if 1==0:

            l3.set_ydata(
                np.convolve(
                    function2(xdata, *v2), #+1 else bias is shifted by 1 ADU
                    np.ones(n_conv) / n_conv,
                    mode="same",
                )
            )
            l4.set_ydata(
                np.convolve(
                    function2(xdata, *v1),
                    np.ones(n_conv) / n_conv,
                    mode="same",
                )
            )

        # l1.set_ydata(
        #     np.convolve(function(x, *v1), np.ones(n_conv) / n_conv, mode="same")
        # )
        # l2.set_ydata(
        #     np.convolve(function(x, *v2), np.ones(n_conv) / n_conv, mode="same")
        # )
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
                valinit=(center, flux),  # center + 0.1
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


    rax = plt.axes([0, 0.08 - 0.035 * 0, 0.1, 0.2])
    # rax = plt.axes([0.5, 0.05-0.035*0, 0.1, 0.15])
    for edge in "left", "right", "top", "bottom":
        rax.spines[edge].set_visible(False)
    check = CheckButtons(rax, ["Bias","RN","Gain","Flux","Smearing","mCIC" ],actives=[True,True,False,False,True,True]) # ,"sCIC"
    check = CheckButtons(rax, ["Bias","RN","Gain","Flux","Smearing","mCIC" ],actives=[True,True,True,True,True,True]) # ,"sCIC"
    def func(label):
        fig.canvas.draw_idle()
    check.on_clicked(func)



    def curve_fit_with_bounds(function, x, y, p0,checks=check,epsfcn=None,check=check):
        bounds1 = []
        bounds2 = []
        # if hasattr(self, "check"):
        for i, p in enumerate(checks.get_status()):
            if p:
                bounds1.append(p0[i] - 1e-5)
                bounds2.append(p0[i] + 1e-5)
            else:
                bounds1.append(-np.inf)
                bounds2.append(np.inf)
        # print("bounds = ",bounds1,bounds2)
        try:
            popt, pcov = curve_fit(
                function, x, y, p0, bounds=[bounds1, bounds2]
            )
        except ValueError:
            for i in range(len(p0)-len(bounds1)):
                bounds1.append(-np.inf)
                bounds2.append(np.inf)
            # print("bounds = ",bounds1,bounds2)
            # print("x = ",x,y)

            popt, pcov = curve_fit(
                function, x[np.isfinite(x) & np.isfinite(y)], y[np.isfinite(x) & np.isfinite(y)], p0, bounds=[bounds1, bounds2],
            # epsfcn=epsfcn
            )
            # popt, pcov = p0, np.zeros((len(p0),len(p0)))
        with open('/tmp/test.csv', 'a') as file:
          file.write("\n%s, %s, %s, %s, %s, %s"%(popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],))   
        
        corr_matrix = pcov / np.outer(np.sqrt(np.diag(pcov)), np.sqrt(np.diag(pcov)))
        # # Exemple de sélection de deux paramètres à analyser
        # param_index_1 = 0
        # param_index_2 = 1
        # correlation = corr_matrix[param_index_1, param_index_2]
        print("correlation matrix =  ", corr_matrix)
        print("corr gain-flux = ",corr_matrix[2,3])
        print("corr gain-smearing = ",corr_matrix[3,4])
        return popt, pcov
    
    def generate_image(event):
        from pyds9plugin.Macros.Fitting_Functions.functions import simulate_emccd_image
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals2 = [v if type(v) != tuple else v[0] for v in vals_tot]
        Bias, RN, EmGain, sCIC, Smearing, mCIC = vals2
        flux = vals_tot[3][1]
        im = simulate_emccd_image(ConversionGain=1,EmGain=EmGain,Bias=Bias,RN=RN,Smearing=Smearing,SmearExpDecrement=1e10,flux=flux,sCIC=sCIC,n_registers=604)
        os = simulate_emccd_image(ConversionGain=1,EmGain=EmGain,Bias=Bias,RN=RN,Smearing=Smearing,SmearExpDecrement=1e10,flux=0,sCIC=sCIC,n_registers=604)
        # header["DATE"] = 
        fitswrite(np.hstack([os,im[:, ::-1]]), name.replace(".fits","_model.fits"),header=header)

        return


    def save_values(event):
        # print(1)
        # import pickle
        # pickle.dump(fig, open("/tmp/test.pkl", 'wb'))  
        # print(2)
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals2 = [v if type(v) != tuple else v[0] for v in vals_tot]
        for val, n in zip(vals2, ["BIAS_ADU","RN_e-","GAINEM","sCIC","SMEARING","mCIC"]):
            fits.setval(name, n, value=str(val)[:7], comment="Histogram fitting")
            print(name, "  OK")
        fits.setval(name, "FLUX", value=str(vals_tot[3][1])[:7], comment="Histogram fitting")
        import pickle
        pickle.dump(fig, file(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting/" + os.path.basename(path).replace(".fits", ".pkl"), 'wb'))  
        if (".fit" not in path) & (".csv" in path):
            plt.savefig(path.replace(".csv", ".svg"))
        if gui is False:

            # plt.savefig(path.replace(".fits", ".png"))
            if os.path.exists(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting") is False:
                os.mkdir(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting")
            plt.savefig(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting/" + os.path.basename(path).replace(".fits", ".svg"))

        # print("vals2", vals2)
        # np.savetxt("/tmp/emccd_fit_values.npy", vals2, fmt='%s')
        # print(np.fromfile("/tmp/emccd_fit_values.npy", fmt='%s'))



        return


    def simplified_least_square_fit_function():
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals1 = [v if type(v) != tuple else v[0] for v in vals_tot]
        vals2 = [v if type(v) != tuple else v[1] for v in vals_tot]
        p0 = vals2#[vals2[1], vals2[2], vals2[3]]  # , vals[4]
        # print("p0=",p0)
        xmin,xmax = ax.get_xlim()
        popt, pcov = curve_fit_with_bounds(
            function,
            # xdata[xdata < upper_limit],
            # y_conv[xdata < upper_limit],
            xdata[(xdata > xmin) & (xdata < xmax) & (xdata < upper_limit)],
            y_conv[(xdata > xmin) & (xdata < xmax) & (xdata < upper_limit)],
            p0=p0,
            # bounds=[[p -10 for p in p0],[p +10 for p in p0]]
            # epsfcn=1#,optimizer=curve_fit_with_bounds
        )
        # print("popt=",popt)
        # print("p0 - popt=",np.array(p0) - np.array(popt))
        vals2 = popt  # , vals[4]
        vals1[:3]= popt[:3]
        vals1[-2:]= popt[-2:]

        plt.draw()
        for slid, val1i, val2i in zip(sliders, vals1, vals2):
            try:
                slid.set_val(val2i)
            except np.AxisError as e:
                # print(e)
                slid.set_val([val1i, val2i])
            dict_values[slid.label.get_text()] = slid.val
        print(os.path.basename(path), n,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val[0],sliders[3].val[1],sliders[-2].val,sliders[-1].val,header_exptime, header_gain,sliders[3].val[1]*3600/header_exptime,fraction_thresholded/100)
        print("file","number","bias","RN","emgain","sCIC","flux","smearing","mCIC","exptime","header_gain","flux_e-/hour","fraction_threshold")
        a = Table(names=["file","number","bias","RN","emgain","sCIC","flux","smearing","mCIC","exptime","header_gain","flux_e-/hour","fraction_threshold"],dtype=["S20"]+[float]*12)
        a.add_row((os.path.basename(path), n,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val[0],sliders[3].val[1],sliders[-2].val,sliders[-1].val,header_exptime, header_gain,sliders[3].val[1]*3600/header_exptime,fraction_thresholded/100))
        a.write(os.path.dirname(os.path.dirname(path))+"/"+os.path.basename(path).replace(".fits",".csv"),overwrite=True)
        fig.savefig(path.replace(".fits",".png"))
        return


    def simplified_least_square_fit(event):
        simplified_least_square_fit_function()
        return
   
    simplified_least_square_fit_function()

    button0.on_clicked(save_values)
    write_model.on_clicked(generate_image)

    simplified_least_square.on_clicked(simplified_least_square_fit)

    plt.draw()
    ax.legend(loc="upper right", fontsize=12, ncol=2)
    ax.set_title("/".join(name.split("/")[-4:]), fontsize=9)
    if (".fit" not in path) & (".csv" in path):
        plt.savefig(path.replace(".csv", ".png"))
    if gui is False:
        # plt.savefig(path.replace(".fits", ".png"))
        if os.path.exists(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting") is False:
            os.mkdir(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting")
        plt.savefig(os.path.dirname(os.path.dirname(path)) + "/histogram_fitting/" + os.path.basename(path).replace(".fits", ".png"))
        return {"BIAS":bias,"RON":RON/ conversion_gain,"GAIN":gain,"FLUX":flux,"FRAC5SIG":fraction_thresholded/100}

    else:
        plt.show()

    import pandas as pd
    import os
    n = int(os.path.basename(name).split("image")[-1].split(".")[0])
    try:
        ncr = int(header["N_CR"])
        mask = float(header["MASK"])
    except KeyError:
        ncr = np.nan
        mask = np.nan

    try:                              
        df = pd.DataFrame([np.array([n,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val[0],sliders[3].val[1],sliders[-2].val,sliders[-1].val,header_exptime, header_gain,ncr,sliders[3].val[1]*3600/header_exptime,fraction_thresholded/100,temp_device])])#mask
    except Exception as e:
        print(e)
        df = pd.DataFrame([np.array([sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val[0],sliders[3].val[1],sliders[-2].val,sliders[-1].val,temp_device,fraction_thresholded/100])])
    df.to_clipboard(index=False,header=False)
    return 1, fig
