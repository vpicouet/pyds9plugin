#%%
from pyds9plugin.testing.startup_spyder import *

DS9backUp = "/Users/Vincent/DS9QuickLookPlugIn/"


def SmearingProfile(
    filename=None,
    path=None,
    xy=None,
    area=None,
    DS9backUp=DS9_BackUp_path,
    name="",
    Plot=True,
):
    """
    plot a stack of the cosmic rays
    """
    from astropy.io import fits
    from astropy.table import Table
    from scipy.optimize import curve_fit
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # if filename is None:
    if path is not None:
        if "merged" in os.path.basename(path).lower():

            cat = Table.read(path)
            lx, ly, offset = 5, 100, 20
            cr_image = np.nan * np.zeros((2 * lx, 2 * ly + offset, len(cat)))
            n = len(cat)
            for image_number in np.unique(cat["number"]):
                image = ReturnPath(filename, number=image_number, All=False)
                fitsimage = fits.open(image)[0]
                data = fitsimage.data
                header = fitsimage.header
                cat["doublons"] = 0
                cat["distance"] = 0
                cr2 = delete_doublons_CR(cat[cat["number"] == image_number], dist=30)
                x, y = cr2["xcentroid"], cr2["ycentroid"]
                #        x, y = cr2['xcentroid'][cr2['distance']>10], cr2['ycentroid'][cr2['distance']>10]
                index = (
                    np.sum(np.isnan(cr_image), axis=(0, 1))
                    == 2 * lx * (2 * ly + offset)
                ).argmax()

                for k in range(len(x)):
                    j = x[k]
                    i = y[k]
                    verboseprint("k+index = ", k + index)
                    try:
                        cr_image[:, :, k + index] = data[
                            i - lx : i + lx, j - 2 * ly : j + offset
                        ]
                    except ValueError as e:
                        verboseprint(e)
                        n -= 1
                        pass
            cr_im = np.nansum(cr_image, axis=2)
            cr_im /= n

    else:
        try:
            verboseprint(xy)
            x, y = xy
        except TypeError:
            D = DetectHotPixels(
                filename, area=area, DS9backUp=DS9_BackUp_path, T1=None, T2=None
            )
            table = D["table"]
            x, y = table["xcentroid"], table["ycentroid"]
            x, y = x + 1, y + 1
        fitsimage = fits.open(filename)[0]
        image = fitsimage.data
        header = fitsimage.header
        lx, ly, offset = 5, 50, 20
        cr_image = np.nan * np.zeros((2 * lx, 2 * ly + offset, len(x)))
        for k in range(len(x)):
            j = x[k]
            i = y[k]
            try:
                cr_image[:, :, k] = image[i - lx : i + lx, j - 2 * ly : j + offset]
            except ValueError:
                pass
        cr_im = np.nanmean(cr_image, axis=2)

    y = cr_im[4, :]
    y = y[: np.argmax(y)]
    y = y[::-1]
    x = np.arange(len(y))

    exp = lambda x, a, b, offset: offset + b * np.exp(-a * x)
    #    exp2 = lambda x, a, b, offset, a1, b1: offset + b*np.exp(-a*x) + b1*np.exp(-a1*x)
    endd = 50
    x, y = x[:endd], y[:endd]
    end = 8

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    try:
        p0 = [5e-1, y.max() - y.min(), y.min()]
        popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0)
    except (RuntimeError or TypeError) as e:  # ( or ValueError) as e :
        verboseprint(e)
        offset = np.min(y)
        a = -0.01
        offsetn = y.min()
        popt = p0
    except ValueError:
        a = -0.01
        offset = 0
        offsetn = 0
    else:
        a, b, offset = popt
        offsetn = offset
        ax2.plot(
            x[:end],
            np.arcsinh(exp(x[:end], *popt) - offsetn),
            color="#1f77b4",
            label="Exp Fit: %i*exp(-x/%0.2f)+%i" % (b, 1 / a, offset),
        )
        ax2.plot(
            x, np.arcsinh(exp(x, *popt) - offsetn), color="#1f77b4", linestyle="dotted"
        )

        ax1.plot(
            x[:end],
            exp(x[:end], *popt),
            color="#1f77b4",
            label="Exp Fit: %i*exp(-x/%0.2f)+%i" % (b, 1 / a, offset),
        )
        ax1.plot(x, exp(x, *popt), color="#1f77b4", linestyle="dotted")

    ax2.plot(
        x, np.arcsinh(y - offsetn), "o", color="#1f77b4"
    )  # , label="DS9 values - %i images \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f" % (cr_image.shape[-1], Smearing2Noise(exp_coeff=1 / a)["Var_smear"], Smearing2Noise(exp_coeff=1 / a)["Hist_smear"]))
    ax1.plot(
        x, y, "o"
    )  # , label="DS9 values - %i images \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f" % (cr_image.shape[-1], Smearing2Noise(exp_coeff=1 / a)["Var_smear"], Smearing2Noise(exp_coeff=1 / a)["Hist_smear"]))

    # plt.plot(x, exp(x,*p0), label='p0')
    temp, gain = header.get("TEMPB", default=0.0), header.get("EMGAIN", default=0.0)
    ax1.grid(True, linestyle="dotted")
    ax2.grid(True, linestyle="dotted")
    fig.suptitle(
        "Smearing analysis: Hot pixel / Cosmic rays profile - T=%s, gain=%i"
        % (temp, gain),
        y=1,
    )
    ax1.legend()
    ax2.legend()
    ax2.set_xlabel("Pixels - %s" % (os.path.basename(filename)))
    ax1.set_ylabel("ADU value")
    ax2.set_ylabel("~Log [Arcsinh] ADU value")
    fig.tight_layout()
    # plt.savefig(DS9backUp + "Plots/%s_CR_HP_profile%s.png" % (datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss"), name))
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
        "Exp_coeff": 1 / a
    }  # , "NoiseReductionFactor": Smearing2Noise(exp_coeff=1 / a)}


def delete_doublons_CR(sources, dist=4, delete_both=False):
    """Function that delete doublons detected in a table,
    the initial table and the minimal distance must be specifies
    """
    from tqdm import tqdm

    if delete_both:
        for i in tqdm(range(len(sources))):
            distances = distance(
                sources[sources["doublons"] == 0]["xcentroid"],
                sources[sources["doublons"] == 0]["ycentroid"],
                sources["xcentroid"][i],
                sources["ycentroid"][i],
            )
            a = distances >= dist
            a = list(1 * a)
            a.remove(0)
            if np.nanmean(a) < 1:
                sources["doublons"][i] = 1
                sources["distance"][i] = np.nanmin(distances[distances > 0])
    else:
        for i in range(len(sources)):  # tqdm(range(len(sources))):
            distances = distance(
                sources["xcentroid"],
                sources["ycentroid"],
                sources["xcentroid"][i],
                sources["ycentroid"][i],
            )
            a = distances >= dist
            a = list(1 * a)
            a.remove(0)
            if (
                np.nanmean(a) < 1
            ):  # if there is still a 0, means if there is a neighboor closer to min distance
                sources["doublons"][i] = 1
                sources["distance"][i] = np.nanmin(distances[distances > 0])
    verboseprint(
        len(sources[sources["doublons"] == 0]), " Comsic rays detected, youpi!"
    )
    return sources


def detectCosmics_new(image, T=6 * 1e4, T2=None, area=[0, 2069, 1053, 2133], n=3):
    """Detect cosmic rays, for FB-2 specfic case it is a simplistic case
    where only thresholding is enough
    """
    from astropy.table import Table, Column
    import matplotlib.dates as mdates
    from astropy.io import fits

    locator = mdates.HourLocator(interval=1)
    ax = plt.gca()
    CS1 = (
        ax.contour(
            np.array((image > T), dtype=int), levels=[1], colors="white", alpha=0.5
        ).allsegs[0]
        if (image > T).any()
        else []
    )
    if T2 is not None:
        CS2 = (
            ax.contour(
                np.array((image > T2), dtype=int), levels=[1], colors="white", alpha=0.5
            ).allsegs[0]
            if (image > T2).any()
            else []
        )
        contours = CS1 + CS2
        verboseprint(
            "%i hot pixels above %i, %i hot pixels above %i"
            % (len(CS1), T, len(CS2), T2)
        )
    else:
        contours = CS1
    print(len(contours))
    names = ("id", "sizex", "sizey", "len_contour", "max_x", "min_y", "max_y")
    cosmics = Table(np.zeros((len(contours), len(names))), names=names)
    cosmics["id"] = np.arange(len(contours))
    cosmics["sizex"] = [cs[:, 0].max() - cs[:, 0].min() for cs in contours]
    cosmics["sizey"] = [cs[:, 1].max() - cs[:, 1].min() for cs in contours]

    cosmics["len_contour"] = [len(cs[:, 1]) for cs in contours]
    cosmics["max_x"] = [int(cs[:, 0].max() + n * 1) for cs in contours]
    cosmics["min_y"] = [int(cs[:, 1].min() - n * 1) for cs in contours]
    cosmics["max_y"] = [int(cs[:, 1].max() + n * 2) for cs in contours]
    cosmics["mean_y"] = [int((cs[:, 1].max() + cs[:, 1].max()) / 2) for cs in contours]
    cosmics["size"] = [n * 50 for cs in contours]
    cosmics["size_opp"] = [n * 1 for cs in contours]
    imagettes = [
        image[
            int(cs[:, 1].min()) : int(cs[:, 1].max()) + 1,
            int(cs[:, 0].min()) : int(cs[:, 0].max()) + 1,
        ]
        for cs in contours
    ]
    cosmics["cx"] = [np.where(ima == np.nanmax(ima))[1][0] for ima in imagettes]
    cosmics["cy"] = [np.where(ima == np.nanmax(ima))[0][0] for ima in imagettes]
    cosmics["c0x"] = [int(cs[:, 0].min()) for cs in contours]
    cosmics["c0y"] = [int(cs[:, 1].min()) for cs in contours]
    cosmics["xcentroid"] = cosmics["c0x"] + cosmics["cx"]  # + area[2]
    cosmics["ycentroid"] = cosmics["c0y"] + cosmics["cy"]  # + area[0]
    cosmics["value"] = [
        image[y, x] for x, y in zip(cosmics["xcentroid"], cosmics["ycentroid"])
    ]
    ly, lx = image.shape
    if area[1] != -1:
        index = (
            (cosmics["ycentroid"] > area[0])
            & (cosmics["ycentroid"] < area[1])
            & (cosmics["xcentroid"] > area[2])
            & (cosmics["xcentroid"] < area[3])
        )
    else:
        index = (
            (cosmics["ycentroid"] > 10)
            & (cosmics["ycentroid"] < ly - 10)
            & (cosmics["xcentroid"] > 10)
            & (cosmics["xcentroid"] < lx - 10)
        )
    cosmics = cosmics[index]
    n_ = 5
    imagettes_bigger = [
        image[int(y) - n_ + 1 : int(y) + n_, int(x) - n_ + 1 : int(x) + n_]
        for x, y in zip(cosmics["xcentroid"], cosmics["ycentroid"])
    ]
    cosmics["VIGNET"] = Column(
        data=np.array(imagettes_bigger, dtype=int)
    )  # , format='%sI'%(len(cosmics)), dim='(9,9)',name='VIGNET')
    cosmics["VIGNET_log"] = Column(
        data=np.array(np.log10(imagettes_bigger), dtype=float)
    )  # , format='%sI'%(len(cosmics)), dim='(9,9)',name='VIGNET')
    print(len(cosmics))

    if T2 is not None:
        index = (
            cosmics["value"] < T2
        )  # &(cosmics['xcentroid']>area[2]) & (cosmics['xcentroid']<area[3]) & (cosmics['ycentroid']>area[0]) & (cosmics['ycentroid']<area[1])
        cosmics = cosmics[index]
    verboseprint("%s cosmic detected" % (len(cosmics)))
    print(len(cosmics))
    if len(cosmics) == 0:
        verboseprint("No cosmic rays detected... Please verify the detection threshold")
        sys.exit()
        cosmics.add_columns(
            [
                Column(name="doublons"),
                Column(name="dark"),
                Column(name="id"),
                Column(name="distance"),
            ]
        )
        return cosmics
    else:
        cosmics["doublons"] = 0
        cosmics["dark"] = -1
        cosmics["id"] = -1
        cosmics["distance"] = -1
        cosmics["sum_10"] = -1
        cosmics["contour"] = -1
        cosmics["Nb_saturated"] = -1
        verboseprint(len(cosmics), " detections, youpi!")
        cosmics_n = delete_doublons_CR(cosmics, dist=1, delete_both=False)
        verboseprint(len(cosmics_n[cosmics_n["doublons"] == 0]))
        verboseprint(cosmics_n["xcentroid", "ycentroid", "value", "doublons"])
    return cosmics_n


def GetThreshold(data, nb=100):
    """
    """
    n = np.dot(*data.shape)
    verboseprint(nb, n)
    pc = 1 - nb / n
    verboseprint("Percentile = ", 100 * pc)
    T = np.nanpercentile(data, 100 * pc)
    return T


def DetectHotPixels(image, T1=None, T2=None, nb=None):
    """
    """
    from astropy.io import fits

    if T1 is None:
        T1, T2 = GetThreshold(image, nb=nb), 7e4

    cosmicRays = detectCosmics_new(image, T=T1, T2=T2, area=[0, -1, 0, -1])
    cosmicRays = cosmicRays[cosmicRays["doublons"] == 0]
    if nb is None:
        nb = 0
    cosmicRays = cosmicRays[-nb:]
    cosmicRays.write(
        DS9backUp + "CSVs/HotPixels%s-%s.fits" % (T1, T2), overwrite=True
    )  # , hdu="LDAC_OBJECTS")
    # cosmicRays.write(DS9backUp + "CSVs/%s_HotPixels%s-%s.fits" % (os.path.basename(filename)[:-5], T1, T2))#, hdu="LDAC_OBJECTS")
    name = DS9backUp + "HotPixels%s-%s.reg" % (T1, T2)
    create_ds9_regions(
        [list(cosmicRays["xcentroid"])],
        [list(cosmicRays["ycentroid"])],
        form=["circle"],
        radius=[5],
        save=True,
        savename=name,
        color=["yellow"],
        ID=None,
    )
    return {"table": cosmicRays, "region": name}


#%%
n = 8
# filename = get_filename(d)

# fitsimage = fits.open(filename)[0]
# image = fitsimage.data

# header = fitsimage.header

image=ds9

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
if date < 2020:
    image = image[:, ::-1]


Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=getregion(d, quick=True))
area = [Xinf, Xsup, Yinf, Ysup]
print(area)
region = image[Yinf:Ysup, Xinf:Xsup]
print(region.shape)
T1, T2 = GetThreshold(region, nb=200), 7e4
n_test = 1


smearings = []
intens = []
pcovs=[]
# for T1 in np.linspace((np.median(region)+T1)/7 ,T1,12):
#     T2 = T1+500#np.nanmax(region) - 50
for T1,T2 in zip([GetThreshold(region, nb=50),GetThreshold(region, nb=100),GetThreshold(region, nb=200),GetThreshold(region, nb=400),GetThreshold(region, nb=800)],[7e4,7e4,7e4,7e4,7e4,7e4,7e4,7e4]):
# for T1,T2 in zip([GetThreshold(region, nb=50),GetThreshold(region, nb=100),GetThreshold(region, nb=200),GetThreshold(region, nb=400),GetThreshold(region, nb=800)],[7e4,7e4,7e4,7e4,7e4,7e4,7e4,7e4]):
    x, y = np.where((region[:,:] > T1) & (region[:,:] < T2)           )  # & (region[:,1:-1]>region[:,:-2])     & (region[:,1:-1]>region[:,2:])  
    # x, y = np.where((region[:,n_test:-n_test] > T1) & (region[:,n_test:-n_test] < np.nanmax(region) - 50)           )  # & (region[:,1:-1]>region[:,:-2])     & (region[:,1:-1]>region[:,2:])  
    # x, y = np.where((region[1:-1,:] > T1) & (region[1:-1,:] < np.nanmax(region) - 50)      & (region[1:-1,:]>region[2:,:])   & (region[1:-1,:]>region[:-2,:])        )  # & (region[:,:-1]>region[:,1:])
    print(len(x))
    ly, lx = region.shape


    if (len(x)>20) & (len(x)<1e4):

        n0 = n
        i = 0
        xx = np.arange(n + n0) - n0

        mask = (x < lx - n) & (y < ly - n) & (x > n0) & (y > n0)
        x, y = x[mask], y[mask]
        a = Table([x, y, region[x, y]], names=["x", "y", "value"])
        a.sort("value", reverse=True)
        a.write("/tmp/test.csv", overwrite=True)

        b = Table([x, y, region[x, y],[region[xi : xi + 1, yi - n0 : yi + n].T  - region[xi : xi + 1, yi - n0 : yi + n].min() for xi,yi in zip(x,y) ]], names=["x", "y", "value","profile"])
        b.sort("value", reverse=True)
        b.write("/tmp/test2.fits", overwrite=True)


        lines=[]
        first_pix = []
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5 ), sharex=True)
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5 ))#, sharex=True
        for i in range(len(x)):
            xi, yi = x[i], y[i]
            line = (
                region[xi : xi + 1, yi - n0 : yi + n].T
                - region[xi : xi + 1, yi - n0 : yi + n].min()
            )
            if line[n0] >= np.nanmax(line) :
                ax1.plot(xx, line / line.ptp(), alpha=0.1,c="grey")
                ax2.semilogy(xx, line / line.ptp(), alpha=0.1)
                lines.append(line)
                first_pix.append(line[n0])
        stack = np.hstack([line/line.ptp() for line in lines])
        # ax1.plot(xx, np.nanmedian(stack, axis=1), "r:", lw=3)
        popt = PlotFit1D(
            xx[xx >= 0],
            np.nanmedian(stack, axis=1)[xx >= 0],
            ax=ax1,
            color="k",
            deg=lambda x, a,b: b+ np.exp(-x/a),#"exp",
            P0=[1,0],
            lw=0.5,
        )["popt"]
        pcov = PlotFit1D(
            xx[xx >= 0],
            np.nanmedian(stack, axis=1)[xx >= 0],
            ax=ax1,
            color="k",
            deg=lambda x, a,b: b+ np.exp(-x/a),#"exp",
            P0=[1,0],
            lw=0.5,
        )["pcov"]
        ax1.errorbar(
            xx[xx >= 0],
            np.nanmedian(stack, axis=1).T[xx >= 0],
            fmt=":o",
            yerr=np.nanstd(stack, axis=1).T[xx >= 0],
            c="k",
            label="Exp length =%0.2fpix"#\nFirst pixel keep %i%% energy"
            # label="Exp length =%spix\nFirst pixel keep %i%% energy\nMin,Max,mean (first pix)=%i, %i, %i"
            % (
                popt[0],
                ), lw=3)
                #100 * np.nanmean(stack, axis=1)[n0] / (np.nanmean(stack, axis=1)[n0:].sum())
        # ,yerr=np.std(stack,axis=0).T[0]
        # ax2.semilogy(xx, np.nanmean(stack, axis=1), "k")
        ax2.semilogy(xx, np.nanmedian(stack, axis=1), "r:",label="Exp length =%0.2fpix\nFirst pixel keep %i%% energy\nMin,Max,mean (first pix)=%i, %i, %i"
            # label="Exp length =%spix\nFirst pixel keep %i%% energy\nMin,Max,mean (first pix)=%i, %i, %i"
            % (
                popt[0],
                100 * np.nanmean(stack, axis=1)[n0] / (np.nanmean(stack, axis=1)[n0:].sum()),
                np.nanmin(first_pix),
                np.nanmax(first_pix),
                np.nanmean(first_pix)))
        ax1.set_xlim((-2.5, n - 1))
        ax1.set_xlim((-1, n - 3))
        ax1.set_ylim((-0.1, 1.1))
        ax2.set_ylim(ymin=1e-2)
        ax1.legend(fontsize=12)
        ax1.grid()
        ax2.set_xlabel("pixels")
        ax1.set_ylabel("ADU decay")
        ax2.set_ylabel("ADU decay (log)")
        
        fig.tight_layout()
        fig.savefig(filename.replace(".fits","") + "smearing_profile_%i_%i_%i_%0.2f.png"%(np.nanmin(first_pix),np.nanmax(first_pix),np.nanmean(first_pix),popt[0]))
        # plt.show()
        smearings.append(popt[0])
        intens.append(np.nanmean(first_pix))
        pcovs.append(pcov[0])


# plt.figure()
# plt.errorbar(intens,smearings,yerr=pcovs,fmt="o")
# for exp_decr in [1e3,5e3, 1e4, 5e4,1e5,5e5]:
#     plt.plot(np.linspace(0,10000), np.max(smearings) * np.exp(-np.linspace(0,10000) / exp_decr),label="exp decrease = %i"%(int(exp_decr)))
# # plt.axvline([1400],c="k",ls=":",label="Gain")
# plt.fill_between([0,np.max(intens)],[0.2,0.2],[np.max(smearings),np.max(smearings)],alpha=0.1)
# plt.xlabel("Pixel value")
# plt.ylabel("Smearing length")
# plt.legend()
# plt.grid()
# plt.savefig(filename.replace(".fits","") + "smearing_profile.png")
# %%

# files = glob.glob("/Volumes/ExtremePro/LAM/FIREBALL/2022/DetectorData/220609/diffusefocus/*14?.fits")
# files.sort()
# for file in files[:10]:
#     fitsim = fits.open(file)[0]
#     im = fitsim.data[:,:1000]
#     plt.figure()
#     plt.hist(im.flatten(),bins=np.linspace(1000,2000,1000),alpha=0.3,log=True)
#     plt.title(os.path.basename(file) + " %s"%(fitsim.header["TEMPB"]))
#     plt.show()


# #%%

# # files = glob.glob("/Volumes/ExtremePro/LAM/FIREBALL/2022/DetectorData/220609/diffusefocus/*[123]?5.fits")
# files = glob.glob("/Volumes/ExtremePro/LAM/FIREBALL/2022/DetectorData/220609/diffusefocus/*[456]?5.fits")
# files.sort()
# plt.figure(figsize=(10,10))
# for i,file in enumerate(files):
#     fitsim = fits.open(file)[0]
#     im = fitsim.data
#     # plt.figure()
#     slit=im[-10+1630:1650+10, 1551:1561].mean(axis=1) - np.median(im[1630:1650, 1760:1778])
#     exptime = fitsim.header["EXPTIME"]
#     plt.plot(slit/exptime,label=os.path.basename(file)[8:11] + ":, T=%i, f=%i"%(float(fitsim.header["TEMPB"]),(slit/exptime).ptp()),c="k",alpha=(i+1)/len(files))
# # plt.title(os.path.basename(file) + " %s"%(fitsim.header["TEMPB"]))
# plt.legend()
# plt.show()

# # %%
