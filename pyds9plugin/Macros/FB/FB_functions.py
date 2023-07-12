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


def emccd_model(
    xpapoint=None, path=None, smearing=0.5, gain=None, argv=[], stack=False, save=False, conversion_gain=1, fit="EMCCDhist",gui=True,RN=3,mCIC=0.005,sCIC=0.005,RON=None ):#RN in ADU
    """Plot EMCCD simulation
    """
    import os
    print("path = ", path)
    if ".fit" not in path:
        tab = Table.read(path)
        print("openning table ", path)
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
        # header_exptime, header_gain = -99, -99
        header_exptime, header_gain = (
            float(path.split("_")[-1].split(".csv")[0][:-1]),
            float(path.split("_")[-2][:-1]),
        )

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
        data = fitsim.data
        if len(data.shape) == 3:
            data = data[0,:,:]
        ly, lx = data.shape

        region = getregion(d, quick=True, message=False, selected=True)
        if region is not None:
            Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
        elif lx > 1500:
            Xinf, Xsup, Yinf, Ysup = [1130, 1430, 1300, 1900]
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

        if conversion_gain<0.5:
            n_conv = 1#21
        else:
            n_conv = 21
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
        print(osv.shape, im.shape)
        median_im = np.nanmedian(im)
        min_, max_ = (int(np.nanpercentile(osv, 0.4)-5), int(np.nanpercentile(im, 99.8)))
        # print(im.shape,os.shape)
        # print(min_, max_,np.nanpercentile(im[np.isfinite(im)], 99.8),np.nanpercentile(im[np.isfinite(im)], 0.4),np.nanpercentile(im[np.isfinite(im)], 40))
        val, bins = np.histogram(im.flatten(), bins=0.5+np.arange(min_, max_, 1*conversion_gain))
        val_os, bins_os = np.histogram(osv.flatten(), bins=0.5+np.arange(min_, max_, 1*conversion_gain))
        print(val_os,min_, max_,osv)
        bins = (bins[1:] + bins[:-1]) / 2
        t=1#0#1#2.5
        val = np.array(val, dtype=float) *t/ im_nan_fraction 
        os_v = np.log10(np.array(val_os *t/ os_nan_fraction, dtype=float)) #* os.size / len(os[np.isfinite(os)])) 
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
    if fit=="EMCCDhist":
        from pyds9plugin.Macros.Fitting_Functions.functions import EMCCDhist as EMCCD
    else:
        from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD

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
    lim_rn1, lim_rn2 = (
        bias - 1 * RN / conversion_gain,
        bias + 0.8 * RN / conversion_gain,
    )
    mask_RN_os = (bins > lim_rn1) & (bins < lim_rn2) & (val_os > 0)
    if RON is None:
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
    ) #+np.log10(1/conv) 
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
    # plt.plot([lim_rn1, lim_rn1], [0, 5], ":", c="grey", alpha=0.7)
    # plt.plot([lim_rn2, lim_rn2], [0, 5], ":", c="grey", alpha=0.7)
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
        label="Data: Gconv=%0.2fADU/e-\ntexp=%is\nG=%iDAQ"
        % (conversion_gain, header_exptime, header_gain),
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
    (l1,) = ax.plot(x, f1, "-", lw=1, label="EMCCD OS model\nFraction>5.5σ (RN=%0.1f)=%0.1f%%"%(RON,100*np.sum(10**y_conv[xdata>bias-0.5 + 5.5 *RON ])/np.sum(10**y_conv)), alpha=0.7)
    (l2,) = ax.plot(x, f2, "-", c=l1.get_color(), lw=1)
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
        print(vals1)

        print("xdata[0]: ", xdata[0])
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
        print("bounds = ",bounds1,bounds2)
        try:
            popt, pcov = curve_fit(
                function, x, y, p0, bounds=[bounds1, bounds2]
            )
        except ValueError:
            for i in range(len(p0)-len(bounds1)):
                bounds1.append(-np.inf)
                bounds2.append(np.inf)
            print("bounds = ",bounds1,bounds2)
            popt, pcov = curve_fit(
                function, x, y, p0, bounds=[bounds1, bounds2],
            # epsfcn=epsfcn
            )

        # else:
        #     popt, pcov = curve_fit(function, x, y, p0)
        return popt, pcov
        # model = CompositeModel(
    #         *Models, label="General fit", optimizer=curve_fit_with_bounds
    #     )



    # def fit(event):
    #     vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
    #     vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
    #     vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]
    #     function = lambda x, RN, EmGain, flux, smearing: EMCCD(
    #         x,
    #         bias=vals1[0],
    #         RN=RN,
    #         EmGain=EmGain,
    #         flux=flux,
    #         smearing=smearing,
    #         sCIC=vals1[5],
    #     )
    #     p0 = [vals2[1], vals2[2], vals2[3], vals2[4]]
    #     # bounds = [[], []]
    #     # bounds = np.array([(0, 300), (100, 2200), (0.001, 6), (0, 3)]).T
    #     popt, pcov = curve_fit(
    #         function,
    #         xdata[xdata < upper_limit],
    #         ydata[xdata < upper_limit],
    #         p0=p0,
    #         epsfcn=1,
    #     )
    #     print(p0)
    #     print(popt)
    #     vals2[1], vals2[2], vals2[3], vals2[4] = popt  # , vals[4]
    #     # We do not want to add the flux
    #     vals1[1], vals1[2] = popt[:-2]  # , vals[4]
    #     vals1[4] = popt[-1]  # , vals[4]
    #     print(popt)

    #     l2.set_ydata(EMCCD(x, *vals2[:args_number]))
    #     l1.set_ydata(EMCCD(x, *vals1[:args_number]))
    #     plt.draw()
    #     for slid, val1i, val2i in zip(sliders, vals1, vals2):
    #         try:
    #             slid.set_val(val2i)
    #         except np.AxisError:
    #             slid.set_val([val1i, val2i])
    #         dict_values[slid.label.get_text()] = slid.val
    #     return
    def save_values(event):
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals2 = [v if type(v) != tuple else v[0] for v in vals_tot]
        for val, n in zip(vals2, ["BIAS_ADU","RN_e-","GAINEM","sCIC","SMEARING","mCIC"]):
            fits.setval(name, n, value=str(val)[:7], comment="Histogram fitting")
            print(name, "  OK")
        fits.setval(name, "FLUX", value=str(vals_tot[3][1])[:7], comment="Histogram fitting")

        # print("vals2", vals2)
        # np.savetxt("/tmp/emccd_fit_values.npy", vals2, fmt='%s')
        # print(np.fromfile("/tmp/emccd_fit_values.npy", fmt='%s'))



        return
    # def fit0(event):
    #     vals1 = centers
    #     vals2 = np.array(centers) + 0.1
    #     for slid, val1i, val2i in zip(sliders, vals1, vals2):
    #         try:
    #             slid.set_val(val1i)
    #         except np.AxisError:
    #             slid.set_val([val1i, val2i])
    #         dict_values[slid.label.get_text()] = slid.val

    #     return

    # def fit0(event):
    #     vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
    #     vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
    #     vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]
    #     if vals1[-2] > 0.5:
    #         vals1[-2] = 0
    #     else:
    #         vals1[-2] = 0.8
    #     for slid, val1i, val2i in zip(sliders, vals1, vals2):
    #         try:
    #             slid.set_val(val1i)
    #         except np.AxisError:
    #             slid.set_val([val1i, val2i])
    #         dict_values[slid.label.get_text()] = slid.val

    #     return

    # def simplified_least_square_fit(event):
    #     vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
    #     vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
    #     vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]
    #     function = lambda x, RN, EmGain, flux: np.convolve(
    #         EMCCD(
    #             x,
    #             bias=vals1[0],
    #             RN=RN,
    #             EmGain=EmGain,
    #             flux=flux,
    #             smearing=vals1[4],
    #             sCIC=vals1[5],
    #         ),
    #         np.ones(n_conv) / n_conv,
    #         mode="same",
    #     )
    #     # f1 = np.convolve(ydata, np.ones(n_conv) / n_conv, mode="same")

    #     p0 = [vals2[1], vals2[2], vals2[3]]  # , vals[4]
    #     # bounds = [[], []]
    #     # bounds = np.array([(0, 300), (100, 2200), (0.001, 6), (0, 3)]).T
    #     popt, pcov = curve_fit(
    #         function,
    #         xdata[xdata < upper_limit],
    #         y_conv[xdata < upper_limit],
    #         p0=p0,
    #         epsfcn=1,
    #     )
    #     print(p0)
    #     print(popt)
    #     vals2[1], vals2[2], vals2[3] = popt  # , vals[4]
    #     vals2[1], vals2[2] = popt[:-1]  # , vals[4]
    #     print(popt)

    #     l2.set_ydata(
    #         np.convolve(
    #             EMCCD(x, *vals2[:args_number]),
    #             np.ones(n_conv) / n_conv,
    #             mode="same",
    #         )
    #     )
    #     l1.set_ydata(
    #         np.convolve(
    #             EMCCD(x, *vals1[:args_number]),
    #             np.ones(n_conv) / n_conv,
    #             mode="same",
    #         )
    #     )

    #     # l2.set_ydata(function(x, *vals2[:args_number]))
    #     # l1.set_ydata(function(x, *vals1[:args_number]))
    #     plt.draw()
    #     for slid, val1i, val2i in zip(sliders, vals1, vals2):
    #         try:
    #             slid.set_val(val2i)
    #         except np.AxisError:
    #             slid.set_val([val1i, val2i])
    #         dict_values[slid.label.get_text()] = slid.val
    #     return

    def simplified_least_square_fit(event):
        vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
        vals1 = [v if type(v) != tuple else v[0] for v in vals_tot]
        vals2 = [v if type(v) != tuple else v[1] for v in vals_tot]
        # function = lambda x, bias,RN, EmGain, flux, smearing, sCIC: np.convolve(
        #     EMCCD(
        #         x,
        #         bias=bias,
        #         RN=RN,
        #         EmGain=EmGain,
        #         flux=flux,
        #         smearing=smearing,
        #         sCIC=sCIC,
        #     ),
        #     np.ones(n_conv) / n_conv,
        #     mode="same",
        # )

        p0 = vals2#[vals2[1], vals2[2], vals2[3]]  # , vals[4]
        print("p0=",p0)
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
        
        print("popt=",popt)
        print("p0 - popt=",np.array(p0) - np.array(popt))
        # vals2[1], vals2[2], vals2[3] = popt  # , vals[4]
        print("vals1=",vals1)
        print("vals2=",vals2)

        vals2 = popt  # , vals[4]
        # new_vals1 = popt
        # new_vals1[4] = vals1[4]
        vals1[:3]= popt[:3]
        vals1[-2:]= popt[-2:]
        # vals2[1], vals2[2] = popt[:-1]  # , vals[4]
        # print("x-xdata=",x-xdata)
        # print("y-xdata=",x-xdata)
        # l2.set_data(xdata,ydata)
        # ax.plot(xdata,ydata,":r")
        # ax.plot(xdata,y_conv,"--r")
        # ax.plot(xdata,np.convolve(
        #         function(x, *vals2[:args_number]),
        #         np.ones(n_conv) / n_conv,
        #         mode="same"),"or")
        # l2.set_ydata(
        #     np.convolve(
        #         function(x, *vals2[:args_number]),
        #         np.ones(n_conv) / n_conv,
        #         mode="same")
        # )
        # l1.set_ydata(
        #     np.convolve(
        #         function(x, *vals1[:args_number]),
        #         np.ones(n_conv) / n_conv,
        #         mode="same")
        # )

        # l2.set_ydata(function(x, *vals2[:args_number]))
        # l1.set_ydata(function(x, *vals1[:args_number]))
        plt.draw()
        for slid, val1i, val2i in zip(sliders, vals1, vals2):
            try:
                slid.set_val(val2i)
            except np.AxisError as e:
                # print(e)
                slid.set_val([val1i, val2i])
            dict_values[slid.label.get_text()] = slid.val
        return
    # def fit_os(event):
    #     vals_tot = [dict_values[slid.label.get_text()] for slid in sliders]
    #     vals1 = [v if type(v) != np.ndarray else v[0] for v in vals_tot]
    #     vals2 = [v if type(v) != np.ndarray else v[1] for v in vals_tot]

    #     vals1[3] = 0
    #     function = lambda x, bias, RN, cic: EMCCD(
    #         x,
    #         bias=bias,
    #         RN=RN,
    #         EmGain=vals1[2],
    #         flux=vals1[3],
    #         smearing=vals1[4],
    #         sCIC=cic,
    #     )
    #     p0 = [vals1[0], vals1[1], vals1[-1]]

    #     popt, pcov = curve_fit(
    #         function,
    #         bins_os[(bins_os < upper_limit_os)],
    #         os_v[(bins_os < upper_limit_os)],
    #         p0=p0,
    #         epsfcn=1,
    #     )
    #     print("vals1 = ", vals1)
    #     vals1[0], vals1[1], vals1[-1] = popt
    #     vals2[0], vals2[1], vals2[-1] = popt
    #     l1.set_ydata(EMCCD(x, *vals1[:args_number]))
    #     l2.set_ydata(EMCCD(x, *vals2[:args_number]))
    #     print("p0 = ", p0)
    #     print("vals1 = ", vals1)
    #     plt.draw()
    #     for i, (slid, vali) in enumerate(zip(sliders, vals1)):
    #         try:
    #             slid.set_val(vals1[i])
    #         except np.AxisError:
    #             slid.set_val([vals1[i], vals2[i]])
    #         dict_values[slid.label.get_text()] = slid.val
    #     return

    button0.on_clicked(save_values)
    # button0.on_clicked(fit0)
    # button.on_clicked(fit)
    # fit_os_button.on_clicked(fit_os)
    simplified_least_square.on_clicked(simplified_least_square_fit)

    plt.draw()
    ax.legend(loc="upper right", fontsize=12, ncol=2)
    ax.set_title("/".join(name.split("/")[-4:]), fontsize=9)
    if (".fit" not in path) & (".csv" in path):
        plt.savefig(path.replace(".csv", ".png"))
    if gui is False:
        plt.savefig(path.replace(".fits", ".png"))
        return {"BIAS":bias,"RON":RON/ conversion_gain,"GAIN":gain,"FLUX":flux}

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
        df = pd.DataFrame([np.array([n,sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val[0],sliders[3].val[1],sliders[-2].val,sliders[-1].val,header_exptime, header_gain,ncr,sliders[3].val[1]*3600/header_exptime,temp_device])])#mask
    except Exception as e:
        print(e)
        df = pd.DataFrame([np.array([sliders[0].val,sliders[1].val,sliders[2].val,sliders[3].val[0],sliders[3].val[1],sliders[-2].val,sliders[-1].val,temp_device])])
    df.to_clipboard(index=False,header=False)
    return 1, fig
