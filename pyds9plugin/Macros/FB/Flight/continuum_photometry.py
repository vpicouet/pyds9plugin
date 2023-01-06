

def ContinuumPhotometry(xpapoint=None, x=None, y=None, DS9backUp=DS9_BackUp_path, config=my_conf, axis="y", kernel=1, fwhm="", center=False, Type=None, n=200):
    """Fit a gaussian
    interger the flux on 1/e(max-min) and then add the few percent calculated by the gaussian at 1/e
    """
    # from astropy.io import fits
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from scipy.optimize import curve_fit
    from astropy.io import fits
    from scipy import ndimage

    if xpapoint is not None:
        d = DS9n(xpapoint)
        filename = get_filename(d)
        Type, axis, kernel, fwhm, center, test = sys.argv[-11:-5]
        verboseprint("Type, axis, kernel, fwhm, center, test = ", Type, axis, kernel, fwhm, center, test)
        path = Charge_path_new(filename) if len(sys.argv) > 10 else [filename]  # and verboseprint('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
        if len(path) == 1:
            plot_flag = True
        else:
            plot_flag = False
        verboseprint("plot_flag = ", plot_flag, verbose=config.verbose)

    else:
        filename = None
    L = []
    for filename in path:
        verboseprint(filename)
        texp = 1  # d.get_pyfits()[0].header[my_conf.exptime[0]]

        verboseprint("Type, axis, kernel =", Type, axis, kernel, verbose=config.verbose)
        if Type == "ProjectionRegion":
            if y is None:
                try:
                    path = sys.argv[3]
                    a = Table.read(path, format="ascii")
                except IndexError:
                    a = Table.read(DS9backUp + "/DS9Curves/ds9.dat", format="ascii")
                x, ys = a["col1"], [a["col2"]]
        elif Type == "BoxRegion":
            region = getregion(d, quick=True)
            Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
            data = fits.open(filename)[0].data

            if axis == "y":
                ys = [np.nanmean(data[Yinf:Ysup, Xinf:Xsup], axis=1)]
                yerrs = [np.nanstd(data[Yinf:Ysup, Xinf:Xsup], axis=1) / np.sqrt(data.shape[1])]
            else:
                ys = [np.nanmean(data[Yinf:Ysup, Xinf:Xsup], axis=0)]
                yerrs = [np.nanstd(data[Yinf:Ysup, Xinf:Xsup], axis=0) / np.sqrt(data.shape[0])]

        if bool(int(test)):
            plot_flag = False
            sizex = Xsup - Xinf
            sizey = Ysup - Yinf
            verboseprint("\n\n\n", 1000 + sizex / 2, 2000 - sizex / 2)
            yinfs, xinfs = np.random.randint(10, 2000 - sizey, size=n), np.random.randint(1100, 2000 - sizex, size=n)
            verboseprint(yinfs)
            verboseprint(xinfs)
            images = [data[Yinf : Yinf + sizey, Xinf : Xinf + sizex] for Xinf, Yinf in zip(xinfs, yinfs)]
            if axis == "y":
                ys = [np.nanmean(image, axis=1) for image in images]
                yerrs = [np.nanstd(image, axis=1) / np.sqrt(data.shape[1]) for image in images]
            else:
                ys = [np.nanmean(image, axis=1) for image in images]
                yerrs = [np.nanstd(image, axis=0) / np.sqrt(data.shape[0]) for image in images]
            verboseprint("Test: number of images = %s" % (len(images)))

        #            kernel = int(kernel)
        #            y = np.convolve(y,np.ones(kernel)/kernel,mode='same')[kernel:-kernel]
        #            yerr = yerr[kernel:-kernel]
        #        else:
        #            kernel = int(kernel)
        #            y = np.convolve(y,np.ones(kernel)/kernel,mode='same')[kernel:-kernel]
        #            yerr = np.zeros(len(y))

        for y, yerr in zip(ys, yerrs):
            if x is None:
                x = np.arange(len(y))
            n = 1
            mask = (y < y.mean() + n * np.std(y)) & (y > y.mean() - n * np.std(y))
            verboseprint(x, y, yerr, mask)
            fit = np.polyfit(x[mask], y[mask], 1)
            # xmax = x[np.argmax(y)]
            fit_fn = np.poly1d(fit)
            try:
                p0 = [y.max() - y.min(), x[y.argmax()], 5]
                if fwhm.split("-")[0] == "":
                    if bool(int(center)):
                        popt1, pcov1 = curve_fit(gaussian, x, y - fit_fn(x), p0=[y.max() - y.min(), len(y) / 2, 5], bounds=([-np.inf, len(y) / 2 - 1, -np.inf], [np.inf, len(y) / 2 + 1, np.inf]))
                    else:
                        popt1, pcov1 = curve_fit(gaussian, x, y - fit_fn(x), p0=[y.max() - y.min(), x[y.argmax()], 5])

                else:
                    stdmin, stdmax = np.array(fwhm.split("-"), dtype=float) / 2.35
                    if bool(int(center)):
                        verboseprint(bool(int(center)))
                        popt1, pcov1 = curve_fit(gaussian, x, y - fit_fn(x), p0=[y.max() - y.min(), len(y) / 2, (stdmin + stdmin) / 2], bounds=([-np.inf, len(y) / 2 - 1, stdmin], [np.inf, len(y) / 2 + 1, stdmax]))
                    else:
                        verboseprint(bool(int(center)))
                        popt1, pcov1 = curve_fit(gaussian, x, y - fit_fn(x), p0=[y.max() - y.min(), x[y.argmax()], (stdmin + stdmin) / 2], bounds=([-np.inf, -np.inf, stdmin], [np.inf, np.inf, stdmax]))
            except RuntimeError as e:
                verboseprint(e)
                popt1 = p0

            verboseprint(popt1)
            limfit = fit_fn(x) + gaussian(x, *popt1).max() / np.exp(1)

            #    import matplotlib._cntr as cntr
            #    c = cntr.Cntr(x, y, z)
            #    nlist = c.trace(level, level, 0)
            #    CS = nlist[:len(nlist)//2]
            image = np.array([np.zeros(len(y)), y > limfit, np.zeros(len(y))])
            # CS = plt.contour(image,levels=1);plt.close()
            ax = plt.gca()
            CS = ax.contour(image, levels=1)

            # size = [cs[:,0].max() - cs[:,0].min()     for cs in CS.allsegs[0] ]
            maxx = [int(cs[:, 0].max()) for cs in CS.allsegs[0]]
            minx = [int(cs[:, 0].min()) for cs in CS.allsegs[0]]
            maxx.sort()
            minx.sort()
            index = np.where(maxx > popt1[1])[0][0]  # np.argmax(size)
            verboseprint(minx[index], maxx[index])
            y0 = y - fit_fn(x)
            # L.append({'Flux':1.155*np.sum(y0[mask])})
            mask = (x > minx[index]) & (x < maxx[index])
            L.append(1.155 * np.sum(y0[mask]))
    verboseprint(y0)
    verboseprint(L)
    plt.close()
    if plot_flag:
        plt.figure()  # figsize=(10,6))
        plt.xlabel("$\parallel$ pix", fontsize=18)
        plt.ylabel("$\widehat{Val_{\perp pix}}$", fontsize=18)
        plt.plot(np.linspace(x.min(), x.max(), 10 * len(x)), gaussian(np.linspace(x.min(), x.max(), 10 * len(x)), *popt1) + fit_fn(np.linspace(x.min(), x.max(), 10 * len(x))), label="Gaussian Fit, F = %0.2f - sgm=%0.1f" % (np.sum(gaussian(x, *popt1)) / texp, popt1[-1]))
        #    plt.plot(x[y>limfit], y[y>limfit], 'o', color=p[0].get_color(), label='Best SNR flux calculation: F=%i'%(1.155*np.sum(y0[y>limfit])/texp))
        # p = plt.errorbar(x, y , yerr = yerr, elinewidth=1, alpha=0.4, fmt='o', linestyle='dotted', label='Data, F=%0.2fADU - SNR=%0.2f'%(np.nansum(y0)/texp,gaussian(x, *popt1).max()/np.std(y0[~mask])))
        filtered = ndimage.filters.gaussian_filter1d(y0[~mask], sigma=7 / 2.35)[7:-7]
        std = np.std(filtered) * 7 / 2.35 * np.sqrt(2 * np.pi)
        # std = np.std(np.convolve(y0[~mask],np.ones(7),mode='valid'))#scipy.ndimage.filters.gaussian_filter1d
        # plt.plot(filtered)
        # plt.plot(np.convolve(y0[~mask],np.ones(7)/7,mode='valid'))
        p = plt.errorbar(x, y, yerr=yerr, elinewidth=1, alpha=0.4, fmt="o", linestyle="dotted", label="Data, F=%0.2fADU - std = %0.2fADU - SNR=%0.2f" % (np.nansum(y0) / texp, std, 1.155 * np.sum(y0[mask]) / std))
        plt.plot(x, y, linestyle="dotted", c=p[0].get_color())

        #    plt.plot(x[mask], y[mask], 'o', color=p[0].get_color(), label='Best SNR flux calculation: F=%i'%(1.155*np.sum(y0[mask])/texp))
        # plt.plot(x,  limfit,'--', c='black', label='1/e limit')
        plt.fill_between(x[mask], y[mask], y2=fit_fn(x)[mask], alpha=0.2, label="Best SNR flux calculation: F=%0.4f" % (1.155 * np.sum(y0[mask])))
        plt.plot(x, fit_fn(x), label="Fitted background", linestyle="dotted", c=p[0].get_color())
        plt.legend()
        plt.title(os.path.basename("%s" % (filename)))
        plt.savefig(DS9backUp + "Plots/%s_Continuumphotometry_%s.jpg" % (datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss"), os.path.basename(filename)))
        if plot_flag:
            plt.show()

    else:
        L = np.array(L)
        median, std = np.median(L), np.std(L)
        limit = median + 3 * std
        mask = L > limit
        create_ds9_regions([xinfs[mask] + sizex / 2], [yinfs[mask] + sizey / 2], radius=[sizex, sizey], form=["box"] * len(xinfs[mask]), save=True, savename="/tmp/centers", ID=[L[mask].astype(int)])
        create_ds9_regions([xinfs[~mask] + sizex / 2], [yinfs[~mask] + sizey / 2], radius=[sizex, sizey], color=["red"] * len(xinfs[~mask]), form=["box"] * len(xinfs[~mask]), save=True, savename="/tmp/centers2", ID=[L[~mask].astype(int)])
        d.set("regions /tmp/centers.reg")
        # d.set('regions /tmp/centers2.reg')

        fig = plt.figure()  # figsize=(10,6))
        ax = fig.add_subplot(111)
        ax.hist(L[~mask], label="mean = %0.1f, std = %0.1f" % (np.mean(L[~mask]), np.std(L[~mask])), alpha=0.3, bins=50)
        ax.hist(L[mask], label="Detections", alpha=0.3)  # ,bins=50)
        ax.vlines(limit, 0, 10, label="5 sigma limit")
        # plt.plot(np.array([-1.772662017174298e-12, 1.2638423640964902e-12, -2.462030579408747e-14, -1.148947603724082e-13, 4.267519670975162e-13, 6.565414878423326e-13, 1.9039703147427643e-12, -6.401279506462743e-13, -9.519851573713822e-13, 1.1817746781161986e-12, 1.8383161659585314e-12, -2.33892905043831e-13, 3.446842811172246e-13, -6.073008762541576e-13, -1.0832934549398488e-12, 1.3787371244688983e-12, -1.0340528433516738e-12, -8.86331008587149e-13, -6.565414878423326e-13, 3.282707439211663e-14, -2.2158275214678726e-13, 2.462030579408747e-13, 1.1653611409201403e-12, 3.3155345136037795e-12, 3.282707439211663e-12, 1.2802559012925486e-12, -1.8465229345565604e-13, 6.23714413450216e-13, 1.0504663805477321e-12, -2.9544366952904965e-13, -1.3787371244688983e-12, 1.9039703147427643e-12, -2.9544366952904965e-13, -4.103384299014579e-14, -2.6261659513693303e-13, 1.5510792650275108e-12, 4.267519670975162e-13, 1.2720491326945193e-13, 4.924061158817494e-14, 1.0340528433516738e-12, -1.148947603724082e-13, -1.8054890915664146e-13, -4.5137227289160365e-14, 1.2146017525083152e-12, 1.3130829756846652e-12, -2.6261659513693303e-13, 6.770584093374055e-13, 1.723421405586123e-13, 2.7903013233299137e-13, 3.1513991416431962e-12]))
        # plt.plot(np.array([-1.772662017174298e-12, 1.2638423640964902e-12, -2.462030579408747e-14, -1.148947603724082e-13, 4.267519670975162e-13, 6.565414878423326e-13, 1.9039703147427643e-12, -6.401279506462743e-13, -9.519851573713822e-13, 1.1817746781161986e-12, 1.8383161659585314e-12, -2.33892905043831e-13, 3.446842811172246e-13, -6.073008762541576e-13, -1.0832934549398488e-12, 1.3787371244688983e-12, -1.0340528433516738e-12, -8.86331008587149e-13, -6.565414878423326e-13, 3.282707439211663e-14, -2.2158275214678726e-13, 2.462030579408747e-13, 1.1653611409201403e-12, 3.3155345136037795e-12, 3.282707439211663e-12, 1.2802559012925486e-12, -1.8465229345565604e-13, 6.23714413450216e-13, 1.0504663805477321e-12, -2.9544366952904965e-13, -1.3787371244688983e-12, 1.9039703147427643e-12, -2.9544366952904965e-13, -4.103384299014579e-14, -2.6261659513693303e-13, 1.5510792650275108e-12, 4.267519670975162e-13, 1.2720491326945193e-13, 4.924061158817494e-14, 1.0340528433516738e-12, -1.148947603724082e-13, -1.8054890915664146e-13, -4.5137227289160365e-14, 1.2146017525083152e-12, 1.3130829756846652e-12, -2.6261659513693303e-13, 6.770584093374055e-13, 1.723421405586123e-13, 2.7903013233299137e-13, 3.1513991416431962e-12]))
        plt.legend()
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Best SNR flux estimator [ADU]")
        plt.show()
        # print(4)
        verboseprint(L)

    return L