def TwoD_autocorrelation(
    filename,
    save=True,
    area=None,
    plot_flag=True,
    DS9backUp=DS9_BackUp_path,
    ds9=None,
    Type="x",
    verbose=False,
):
    """Return 2D_autocorrelation plot
    """
    from scipy import signal  # from scipy import misc
    from astropy.io import fits
    import numpy as np

    fitsimage = fits.open(filename)
    fitsimage = fitsimage[fits_ext(fitsimage)]
    data = fitsimage.data

    if area is None:
        lx, ly = data.shape
        area = [int(lx / 3), 2 * int(lx / 3), int(ly / 3), 2 * int(ly / 3)]
    w = abs((area[1] - area[0]))
    h = abs((area[3] - area[2]))
    new_area = [area[0] - w, area[1] + w, area[2] - h, area[3] + h]
    finite = np.isfinite(np.mean(data[:, new_area[2] : new_area[3]], axis=1))
    data = data[finite, :]
    # gain, temp = fitsimage.header.get(my_conf.gain[0], default=0.0), fitsimage.header.get(my_conf.temperature[0], default=0.0)
    template = np.copy(data[area[0] : area[1], area[2] : area[3]]).astype("uint64")

    if Type == "2d-xy":
        image = np.copy(
            data[new_area[0] : new_area[1], new_area[2] : new_area[3]]
        ).astype("uint64")
        image = image - np.nanmin(data) + 100.0
        template = template - np.nanmin(data) + 100.0
        corr = signal.correlate2d(image, template, boundary="symm", mode="same")

    if Type == "x":
        image = np.copy(data[area[0] : area[1], new_area[2] : new_area[3]]).astype(
            "uint64"
        )
        corr = np.zeros(image.shape)
        for i in range(template.shape[0]):
            corr[i, :] = signal.correlate(
                image[i, :], template[i, :], mode="same"
            )  # / 128
    if Type == "y":
        image = np.copy(data[new_area[0] : new_area[1], area[2] : area[3]]).astype(
            "uint64"
        )
        corr = np.zeros(image.shape)
        for i in range(template.shape[1]):
            corr[:, i] = signal.correlate(
                image[:, i], template[:, i], mode="same"
            )  # / 128
    fitsimage = fits.HDUList([fits.PrimaryHDU(corr)])[0]
    fitsimage.header["PATH"] = filename
    name = DS9_BackUp_path + "Images/%s_%sAutocorr.fits" % (
        datetime.datetime.now().strftime("%y%m%d-%HH%M"),
        os.path.basename(filename)[:-5],
    )
    fitswrite(fitsimage, name)
    D = {"corr": corr, "name": name}  # , "gain": gain, "temp": temp}
    return D


def SmearingProfileAutocorr(
    filename=None,
    area=None,
    DS9backUp=DS9_BackUp_path,
    name="",
    Plot=True,
    Type="x",
    verbose=False,
):
    """
    plot a stack of the cosmic rays
    """
    # from astropy.io import fits
    # from astropy.table import Table
    from scipy.optimize import curve_fit
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if not verbose:
        np.warnings.filterwarnings("ignore")
    verboseprint("Type  = ", Type)
    if area is None:
        if Type == "2d-xy":
            area = [1500, 1550, 1500, 1550]
        else:
            area = None  # [890-470,890+470,1565-260,1565+260]#[1200,1800,1200,1800]
    D = TwoD_autocorrelation(
        filename=filename,
        save=True,
        area=area,
        plot_flag=Plot,
        DS9backUp=DS9_BackUp_path,
        ds9=False,
        Type=Type,
    )
    verboseprint("Autocorr = ", D["corr"])
    try:
        autocorr = D["corr"] / D["corr"].min()
    except ValueError:
        return {
            "Exp_coeff": 1 / -100,
            # "NoiseReductionFactor": Smearing2Noise(exp_coeff=1 / -100),
        }
    # temp, gain = D["temp"], D["gain"]
    # imshow(autocorr)#d.set_np2arr(autocorr)
    lx, ly = autocorr.shape
    verboseprint(lx, ly)
    size = 6  #
    # y = autocorr[int(lx/2)-size:int(lx/2)+size,int(ly/2)-1];x = np.arange(len(y));plot(x,y,'--o')
    if Type == "2d-xy":
        y = autocorr[int(lx / 2) - 1, int(ly / 2) - size - 1 : int(ly / 2) + size]
        x = np.arange(len(y))
        # plot(x,y,'--o')
        y1 = autocorr[int(lx / 2) - 1, int(ly / 2) - size - 1 : int(ly / 2)]
        x1 = np.arange(len(y1))
        y2 = autocorr[int(lx / 2) - 1, int(ly / 2) - 1 : int(ly / 2) + size]
        yy = (y1[::-1] + y2) / 2
    if Type == "x":
        y = np.mean(autocorr[:, int(ly / 2) - size - 1 : int(ly / 2) + size], axis=0)
        x = np.arange(len(y))
        # plot(x,y,'--o')
        # print(y.shape)
        y1 = np.mean(autocorr[:, int(ly / 2) - size : int(ly / 2) + 1], axis=0)
        x1 = np.arange(len(y1))
        # print(y1,x1)
        y2 = np.mean(autocorr[:, int(ly / 2) : int(ly / 2) + size + 1], axis=0)
        yy = (y1[::-1] + y2) / 2
    #    plt.plot(x1,y1[::-1],'--')
    #    plt.plot(x1,y2,'--')
    #    plt.plt.plot(x1,yy,'--o')
    # plt.figure(figsize=(10,6))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    exp = lambda x, a, b, offset: offset + b * np.exp(-a * x)
    # exp2 = lambda x, a, b, offset, a1, b1: offset + b*np.exp(-a*x) + b1*np.exp(-a1*x)
    # cr_image = autocorr
    x, y, end = x1, yy, size
    verboseprint(x)
    x100 = np.linspace(x[0], x[size], 100)
    # plt.vlines(median(x),y.min(),y.max())
    # plt.text(median(x)+50,mean(y),'Amplitude = %i'%(max(y)-min(y)))
    try:
        p0 = [5e-1, y.max() - y.min(), y.min()]
        # p1=[0,5e-1,y.max()-y.min(),y.min(),0,5e-1,y.max()-y.min()]
        popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0)
        # popt1, pcov1 = curve_fit(exp2, x[:end], y[:end], p0=np.hstack((popt,[5e-1,y.max()-y.min()])))
    except (RuntimeError or TypeError) as e:  # ( or ValueError) as e :
        verboseprint(e, verbose=True)
        offset = np.min(y)
        a = -0.01
        offsetn = y.min()
    except ValueError as e:
        verboseprint(e, verbose=True)
        a = -0.01
        offset = 0
        offsetn = 0
    else:
        a, b, offset = popt
        # a0, b0, offset0, a1, b1 = popt1
        offsetn = y.min()
        ax2.plot(
            x100,
            np.log10(exp(x100, *popt) - offsetn),
            linestyle="dotted",
            label="Exp Fit: %i*exp(-x/%0.2f)+%i" % (b, 1 / a, offset),
        )
        # ax2.plot(x[:end], np.log10(exp2(x[:end],*popt1)-offsetn), label='Exp Fit: %i*exp(-x/%0.2f)+ %i + %i*exp(-x/%0.2f)'%(b0, 1/a0, offset0,b1, 1/a1))

        ax1.plot(
            x100,
            exp(x100, *popt),
            linestyle="dotted",
            label="Exp Fit: %i*exp(-x/%0.2f)+%i" % (b, 1 / a, offset),
        )
        # ax1.plot(x[:end], exp2(x[:end],*popt1), label='Exp Fit: %i*exp(-x/%0.2f)+ %i + %i*exp(-x/%0.2f)'%(b0, 1/a0, offset0,b1, 1/a1))

    ax2.plot(
        x,
        np.log10(y - offsetn),
        "-o",
        c="black",
        label="DS9 values - %0.2f Expfactor "  # \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f"
        % (
            1 / a,
            # Smearing2Noise(exp_coeff=1 / a)["Var_smear"],
            # Smearing2Noise(exp_coeff=1 / a)["Hist_smear"],
        ),
    )
    #    ax2.plot(x1,np.log10(y1[::-1]- offsetn),linestyle='dotted',c='grey')
    #   ax2.plot(x1,np.log10(y2- offsetn) ,linestyle='dotted',c='grey')
    ax2.fill_between(
        x1,
        np.log10(y1[::-1] - offsetn),
        np.log10(y2 - offsetn),
        alpha=0.2,
        label="Left-right difference in autocorr profile",
    )
    ax1.plot(
        x,
        y,
        "-o",
        c="black",
        label="DS9 values - %0.2f Expfactor "  # \n -> Noise reduc = %0.2f, Slope reduc =  %0.2f"
        % (
            1 / a,
            # Smearing2Noise(exp_coeff=1 / a)["Var_smear"],
            # Smearing2Noise(exp_coeff=1 / a)["Hist_smear"],
        ),
    )
    # ax1.plot(x1,y1[::-1],linestyle='dotted',c='grey')
    # ax1.plot(x1,y2,linestyle='dotted',c='grey')
    ax1.fill_between(
        x1, y1[::-1], y2, alpha=0.2, label="Left-right difference in autocorr profile"
    )
    ax1.grid(True, linestyle="dotted")
    ax2.grid(True, linestyle="dotted")
    fig.suptitle(
        "%s Smearing analysis: Autocorrelation ", y=1
    )  # - T=%s, gain=%i, area=%s" % (Type.upper(), temp, float(gain), area), y=1)
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel("Pixels - %s" % (os.path.basename(filename)))
    ax1.set_ylabel("ADU value")
    ax2.set_ylabel("Log ADU value")
    fig.tight_layout()
    #    plt.savefig(DS9backUp + 'Plots/%s_CR_HP_profile%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss"), name) )
    if not os.path.exists(os.path.dirname(filename) + "/SmearingAutocorr"):
        os.makedirs(os.path.dirname(filename) + "/SmearingAutocorr")
    plt.savefig(
        os.path.dirname(filename)
        + "/SmearingAutocorr/%s_%s.png"
        % (os.path.basename(filename)[:-5], Type.upper())
    )
    if Plot:
        plt.show()
    else:
        plt.close()
    csvwrite(
        np.vstack((x, y)).T,
        DS9backUp
        + "CSVs/%s_CR_HP_profile%s.csv"
        % (datetime.datetime.now().strftime("%y%m%d-%HH%M"), name),
    )
    return {
        "Autocorr": y,
        "Exp_coeff": 1 / a,
        # "NoiseReductionFactor": Smearing2Noise(exp_coeff=1 / a),
    }


Type = "x"  # sys.argv[3].lower()  #'2D-xy'.lower()
verboseprint("Type =", Type)
path = globglob(filename)
region = getregion(d, quick=True)

Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
area = [Yinf, Ysup, Xinf, Xsup]
verboseprint(Yinf, Ysup, Xinf, Xsup)

if len(path) > 1:
    Plot = False
else:
    Plot = True
SmearingProfileAutocorr(
    filename=filename,
    area=area,
    DS9backUp=DS9_BackUp_path,
    name="",
    Plot=Plot,
    Type=Type,
)

