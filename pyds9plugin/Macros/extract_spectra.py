
def returnXY(field, w=2060, frame="observed", keyword="Lya_Gal", mag_min=None, mag_max=None):
    """Return redshift, position of the slit, wavelength used for each mask
    given the DS9 entry
    """
    from astropy.table import Table

    #    try:
    from .mapping import Mapping

    #    except ValueError:
    #        from Calibration.mapping import Mapping
    field = field.lower()
    if (w == "lya") or (w == "Lya") or (w == "Lya"):
        w = 1215.67
    w = float(w)
    w *= 1e-4
    verboseprint("Selected Line is : %0.4f microns" % (w))

    try:
        # slit_dir = resource_filename('pyds9plugin', 'Slits')
        Target_dir = resource_filename("pyds9plugin", "Targets")
        Mapping_dir = resource_filename("pyds9plugin", "Mappings")
    except:
        # slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        Target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Targets")
        Mapping_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Mappings")

    if ("f1" in field) or ("119" in field):
        # csvfile = os.path.join(slit_dir,'F1_119.csv')
        targetfile = os.path.join(Target_dir, "targets_F1.txt")
        mappingfile = os.path.join(Mapping_dir, "mapping-mask-det-w-1806012-F1.pkl")  # mapping-mask-det-180612-F1.pkl
    if ("f2" in field) or ("161" in field):
        # csvfile = os.path.join(slit_dir,'F2_-161.csv')
        targetfile = os.path.join(Target_dir, "targets_F2.csv")
        mappingfile = os.path.join(Mapping_dir, "mapping-mask-det-w-1806012-F2.pkl")
    if ("f3" in field) or ("121" in field):
        # csvfile = os.path.join(slit_dir,'F3_-121.csv')
        targetfile = os.path.join(Target_dir, "targets_F3.txt")
        mappingfile = os.path.join(Mapping_dir, "mapping-mask-det-w-1806012-F3.pkl")
    if ("f4" in field) or ("159" in field):
        # csvfile = os.path.join(slit_dir,'F4_159.csv')
        targetfile = os.path.join(Target_dir, "targets_F4.txt")
        mappingfile = os.path.join(Mapping_dir, "mapping-mask-det-w-1806012-F4.pkl")
    verboseprint(field)
    if "none" in field.lower():
        return [], [], [], [], [], []
    # print('Selected field in : ', csvfile)
    verboseprint("targetfile = ", targetfile)
    mapping = Mapping(filename=mappingfile)

    try:
        target_table = Table.read(targetfile)  # , format='ascii')
    except:
        target_table = Table.read(targetfile, format="ascii", delimiter="\t")

    if "f1" in field:
        idok = (target_table["slit_length_right"] != 0) & (target_table["slit_length_left"] != 0)
        target_table = target_table[idok]
        xmask = target_table["xmask"] + (target_table["slit_length_right"] - target_table["slit_length_left"]) / 2.0
        ymask = target_table["ymask"] + target_table["offset"]
        z = target_table["z"]
        internalCount = target_table["Internal-count"]

    else:
        xmask = target_table["xmm"]
        ymask = target_table["ymm"]
        internalCount = target_table["Internal-count"]
        z = target_table["Z"]
    redshift = z
    slit = internalCount

    if frame.lower() == "restframe":
        verboseprint("Working in rest frame wevelength")
        # w = 1215.67
        wavelength = (1 + redshift) * w  # * 1e-4
        y, x = mapping.map(wavelength, xmask, ymask, inverse=False)
    if frame.lower() == "observedframe":
        verboseprint("Working in observed frame wevelength")
        y, x = mapping.map(w, xmask, ymask, inverse=False)
    w *= 1e4

    # print(x[0],y[0])
    if keyword is None:
        return x, y, redshift, slit, np.zeros(len(slit)), w
    elif "all" in keyword.lower():
        return x, y, redshift, slit, np.zeros(len(slit)), w
    elif "lya" in keyword.lower():
        index = [len(sliti) < 3 for sliti in slit]
        if ("f2" in field) or ("161" in field):
            mag = target_table["mgNUVMAG_L"]
            if (mag_max is not None) and (mag_min is not None):
                index = [((magi > mag_min) and (magi < mag_max)) for magi in mag]
            x, y, redshift, slit, mag, w = x[index], y[index], redshift[index], slit[index], mag[index], w
        else:
            x, y, redshift, slit, mag, w = x[index], y[index], redshift[index], slit[index], w
        return x, y, redshift, slit, mag, w  # x[index], y[index], redshift[index], slit[index], w#[index]
    elif "qso" in keyword.lower():
        index = ["qso" in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], np.zeros(len(slit[index])), w  # [index]
    elif "lyc" in keyword.lower():
        index = ["lyc" in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], np.zeros(len(slit[index])), w  # [index]
        index = ["ovi" in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], np.zeros(len(slit[index])), w  # [index]


def DS9plot_spectra(xpapoint, w=None, n=4, rapport=1.8, continuum=False, DS9backUp=DS9_BackUp_path, Lya=False):
    """Plot spectra in local frame
    """
    import matplotlib.pyplot as plt

    field, Frame, w, kernel, threshold, ObjType, magnitude = sys.argv[3:]  #'f3 names'#sys.argv[3]
    kernel, threshold = int(kernel), int(threshold)
    mag_min, mag_max = np.array(magnitude.split(","), dtype=float)
    verboseprint("Field, Frame, w, kernel = ", field, Frame, w, kernel)

    d = DS9n(xpapoint)
    filename = d.get("file")
    fitsfile = d.get_pyfits()  # fits.open(filename)
    image = fitsfile[0].data
    x, y, redshift, slit, mag, w = returnXY(field, w=w, frame=Frame, keyword=ObjType, mag_min=mag_min, mag_max=mag_max)

    create_ds9_regions(y, x, radius=10, form="box", save=True, color="yellow", savename="/tmp/centers")
    verboseprint("x = ", x)
    verboseprint("y = ", y)
    # print('redshift = ', redshift)
    # print('slit = ', slit)
    try:
        region = getregion(d)
    except:
        pass
    else:
        Xinf, Xsup, Yinf, Ysup = lims_from_region(region)
        mask = (y > Xinf) & (y < Xsup) & (x > Yinf) & (x < Ysup)
        verboseprint(mask)
        x, y, redshift, slit = x[mask], y[mask], redshift[mask], slit[mask]
        verboseprint("Selected objects are : ", slit)

    n1 = n
    n2 = 700
    verboseprint("n1,n2 = ", n1, n2)
    redshift = redshift.tolist()

    lambdasup = []
    lambdainf = []
    sup = 2100  # 2133
    inf = 1100  # 1053
    x2w = 10.0 / 46
    flag_visible = (y > inf) & (y < sup) & (x > 0) & (x < 2070)
    if Frame == "RestFrame":
        lya_z = 0.7
        Lya = True
        nb_gaussian = 1
    if Frame == "ObservedFrame":
        lya_z = 0
        nb_gaussian = 3
        redshift = np.zeros(len(redshift))
    redshifti = np.array(redshift)[flag_visible]
    sliti = np.array(slit)[flag_visible]
    xi = np.array(x)[flag_visible]
    yi = np.array(y)[flag_visible]
    imagettes = []
    for i in range(len(xi)):
        imagettes.append(image[int(xi[i]) - n1 : int(xi[i]) + n1, int(yi[i]) - n2 : int(yi[i]) + n2])
        lambdainf.append(-(sup - yi[i]) * x2w / (1 * redshift[i] + 1) + w)
        lambdasup.append(-(inf - yi[i]) * x2w / (1 * redshift[i] + 1) + w)
    v1, v2 = int(len(xi) / 2 + 1), 2
    verboseprint("v1,v2=", v1, v2)
    # fig, axes = plt.subplots(v1, v2, figsize=(18,50),sharex=True)
    fig, axes = plt.subplots(v1, v2, figsize=(12.8, 70 / 40 * v1), sharex=True, sharey=True)
    xaxis = np.linspace(w - n2 * x2w / (1 + lya_z), w + n2 * x2w / (1 + lya_z), 2 * n2)
    xinf = np.searchsorted(xaxis, lambdainf)
    xsup = np.searchsorted(xaxis, lambdasup)
    fig.suptitle("Spectra, lambda in %s" % (Frame), y=1)

    spectras = []
    for i, ax in enumerate(axes.ravel()[1 : len(imagettes) + 1]):
        spectra = imagettes[i][:, ::-1].mean(axis=0)
        spectra[: xinf[i]] = np.nan
        spectra[xsup[i] :] = np.nan
        spectras.append(spectra)
        ax.step(xaxis, spectra, label="Slit: %s - Mag: %0.1f" % (sliti[i], mag[i]) + "\nz = %0.2f" % (redshifti[i]) + "\nx,y = %i - %i" % (yi[i], xi[i]))
        ax.axvline(x=lambdainf[i], color="black", linestyle="dotted")
        ax.axvline(x=lambdasup[i], color="black", linestyle="dotted")
        ax.legend()
        ax.set_xlim(xaxis[[0, -1]])
        ax.tick_params(labelbottom=True)
    ax = axes.ravel()[0]
    verboseprint(np.array(imagettes).shape)
    #    stack = np.nanmean(np.array(spectras),axis=0)
    stack = np.convolve(np.nanmean(np.array(spectras), axis=0), np.ones(kernel) / kernel, mode="same")  # [kernel:-kernel]
    #    verboseprint(np.nanmean(np.array(spectras),axis=0),stack.shape)
    ax.step(xaxis[kernel:-kernel], stack[kernel:-kernel], label="Stack", c="orange")
    ax.legend()
    ax.set_xlim(xaxis[[0, -1]])
    ax.tick_params(labelbottom=True)
    for ax in axes[-1, :]:
        ax.set_xlabel("Wavelength [A] rest frame")
    fig.tight_layout()
    fig.savefig(filename[:-5] + "_Spectras.png")
    if v1 > 12:
        ScrollableWindow(fig)
    else:
        plt.show()
    detectLine2(xaxis[kernel:-kernel], stack[kernel:-kernel], clipping=threshold, window=20, savename=filename[:-5] + "_StackedSpectra.png", Lya=Lya, nb_gaussian=nb_gaussian)
    plt.show()

    csvwrite(np.vstack((xaxis, stack)).T, DS9backUp + "CSVs/%s_SpectraBigRange.csv" % (datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss")))
    csvwrite(np.vstack((xaxis, stack)).T, filename[:-5] + "_SpectraBigRange.csv")

    return imagettes


def detectLine(x, y, clipping=10, window=20, savename="/tmp/stacked_spectrum.png"):
    """Fit a gaussian
    interger the flux on 1/e(max-min) and then add the few percent calculated by the gaussian at 1/e
    """
    # x, y = Table.read('/Users/Vincent/DS9BackUp/CSVs/190510-11H14_SpectraBigRange.csv')['col0'][8:-8],Table.read('/Users/Vincent/DS9BackUp/CSVs/190508-16H24_SpectraBigRange.csv')['col1'][8:-8]
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    n = 0.3
    fit = np.polyfit(x[y < y.mean() + n * np.std(y)], y[y < y.mean() + n * np.std(y)], 1)
    # xmax = x[np.argmax(y)]
    fit_fn = np.poly1d(fit)
    # texp=1
    limfit = fit_fn(x)  # +gaussian(x, *popt1).max()/np.exp(1)
    std_signal = np.nanstd((y - limfit)[:100])
    image = np.array([np.zeros(len(y)), y - limfit, np.zeros(len(y))])
    CS = plt.contour(image, levels=np.nanmean(y - limfit) + clipping * std_signal)
    plt.close()
    # size = [cs[:,0].max() - cs[:,0].min()     for cs in CS.allsegs[0] ]
    if (y - limfit > np.nanmean(y - limfit) + clipping * std_signal).any():
        maxx = np.array([int(cs[:, 0].max()) for cs in CS.allsegs[0]])
        minx = np.array([int(cs[:, 0].min()) for cs in CS.allsegs[0]])
        maxx.sort()
        minx.sort()
        meanx = np.array((maxx + minx) / 2, dtype=int)
    else:
        maxx, minx, meanx = [], [], []
    #    index = np.where(maxx>popt1[1])[0][0]#np.argmax(size)
    #    verboseprint(minx[index],maxx[index])

    y0 = y - fit_fn(x)
    plt.figure()
    plt.xlabel("Spatial direction")
    plt.ylabel("ADU mean value")
    p = plt.step(x, y, label="Data")  # , label='Data, F=%0.2fADU - SNR=%0.2f'%(np.nansum(y0)/texp,gaussian(x, *popt1).max()/np.std(y0[~mask])))
    # plt.fill_between(x[mask],y[mask],y2=fit_fn(x)[mask],alpha=0.2, label='Best SNR flux calculation: F=%0.2f'%(1.155*np.sum(y0[mask])/texp))
    plt.plot(x, fit_fn(x), label="Fitted background", linestyle="dotted", c=p[0].get_color())

    for i, xm in enumerate(meanx):
        popt1, pcov1 = curve_fit(gaussian, x[xm - window : xm + window], y0[xm - window : xm + window], p0=[y0[xm], x[xm], 5])
        plt.plot(x[xm - window : xm + window], gaussian(x[xm - window : xm + window], *popt1) + fit_fn(x[xm - window : xm + window]), label="%i Gaussian Fit, F = %0.2f - sgm=%0.1f" % (i, np.sum(gaussian(x, *popt1)), popt1[-1]))
    # mask = (x>minx[index]) & (x<maxx[index])
    plt.plot(x[meanx], y[meanx], "o", label="Detections")
    plt.plot(x, np.ones(len(x)) * np.nanmean(y - limfit) + clipping * std_signal + fit_fn(x), label="Detection level: %i sigma" % (clipping))
    plt.legend()
    plt.title("Stacked spectrum")
    plt.xlabel("Wavelength [Angstrom]")
    plt.ylabel("ADU Value")
    plt.savefig(savename)
    plt.show()
    return


def detectLine2(x, y, clipping=10, window=20, savename="/tmp/stacked_spectrum.png", Lya=False, nb_gaussian=1):
    """Fit a gaussian
    interger the flux on 1/e(max-min) and then add the few percent calculated by the gaussian at 1/e
    """
    # x, y = Table.read('/Users/Vincent/DS9BackUp/CSVs/190510-11H14_SpectraBigRange.csv')['col0'][8:-8],Table.read('/Users/Vincent/DS9BackUp/CSVs/190508-16H24_SpectraBigRange.csv')['col1'][8:-8]
    from .dataphile.demos import auto_gui

    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # from scipy.optimize import curve_fit
    # plt.figure()
    plt.close()
    # y /= 1500 * 0.55 * 50 * 208
    index = np.isfinite(y) & np.isfinite(x)
    obj = auto_gui.GeneralFit(x[index], y[index], nb_gaussians=nb_gaussian)
    limit = np.nanmean(y[30:-30]) + 3 * np.nanstd(y[30:-30])
    obj.ax.plot(x, np.ones(len(y)) * limit, label="3 sigma limit", linestyle="dashed", color="g")
    if Lya:
        obj.ax.fill_betweenx(np.linspace(np.nanmin(y), np.nanmax(y), 100), 1215.6 - 3, 1215.6 + 3, color="red", alpha=0.3, label="Lyman alpha line +/- 1000km/s")
    obj.ax.legend()


    return obj
