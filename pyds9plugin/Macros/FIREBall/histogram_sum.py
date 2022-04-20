from pyds9plugin.DS9Utils import *
import sys, glob
import matplotlib.pyplot as plt


def HistogramSums(path=[]):
    """
    """
    from astropy.io import fits
    import matplotlib
    import numpy as np

    # matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from pyds9plugin.Macros.FIREBall.FB_functions import emccd_model

    # d = DS9n()
    # filename = get_filename(d)
    # # log = bool(int(sys.argv[3]))
    # try:
    #     hrange = np.array(sys.argv[4].split("-"), dtype=float)
    # except ValueError:
    #     hrange = None
    # if len(sys.argv) > 5:
    #     path = Charge_path_new(filename, entry_point=5)

    region = None  #
    # d = DS9n()
    # region = getregion(d, quick=True, message=False)
    if region is None:
        Yinf, Ysup, Xinf, Xsup = [1130, 1430, 1300, 1900]
        Yinf, Ysup, Xinf, Xsup = [1300, 1900, 1130, 1430]
        Yinf, Ysup, Xinf, Xsup = [30, 530, 1500, 2100]
    else:
        Yinf, Ysup, Xinf, Xsup = lims_from_region(None, coords=region)
        # image_area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup, Xinf, Xsup)

    bins = np.arange(0, 10000, 1)
    fitsfile = fits.open(path[0])[0]
    exptime, daq, date = (
        fitsfile.header["EXPTIME"],
        fitsfile.header["EMGAIN"],
        fitsfile.header["DATE"][:10],
    )
    print(date)
    if int(date[:4]) > 2020:
        fitsfile.data = fitsfile.data[:, ::-1]
    data = fitsfile.data[Yinf:Ysup, Xinf:Xsup]  # avant [Xinf:Xsup,Yinf:Ysup]
    os_ = fitsfile.data[Yinf:Ysup, Xinf + 1000 : Xsup + 1000]
    # value, bins = np.histogram(data.flatten(), range=[0, 2 ** 16], bins=int(2 ** 16 / 2 ** 2))
    value, bins = np.histogram(data.flatten(), bins=bins)
    value_os, bins = np.histogram(os_.flatten(), bins=bins)
    value_os = [value_os]
    value = [value]
    for filename in path[1:]:
        verboseprint(filename)
        fitsfile = fits.open(filename)[0]
        data = fitsfile.data[Xinf:Xsup, Yinf:Ysup]
        os_ = fitsfile.data[Yinf:Ysup, Xinf + 1000 : Xsup + 1000]
        value1, bins = np.histogram(data, bins=bins)
        os1, bins = np.histogram(os_, bins=bins)
        # value += value1
        # value_os += os1
        value_os.append(os1)
        value.append(value1)
    value = 2 * np.nanmedian(np.array(value), axis=0)
    value_os = 2 * np.nanmedian(np.array(value_os), axis=0)
    # value = np.nansum(np.array(value),axis=0)
    # value_os = np.nansum(np.array(value_os),axis=0)
    bins_c = (bins[1:] + bins[:-1]) / 2
    bias = bins_c[np.nanargmax(value_os)]
    try:
        upper_limit = bins_c[
            np.where(
                (bins_c > bias)
                & (np.convolve(value, np.ones(3), mode="same") == np.nanmin(value))
            )[0][0]
        ]

        lower_limit_os = bins_c[
            np.where(
                (bins_c < bias)
                & (
                    np.convolve(value_os, np.ones(3), mode="same")
                    == np.nanmin(value_os)
                )
            )[0][-1]
        ]
    except IndexError:
        upper_limit, lower_limit_os = bins_c.max(), bins_c.min()
    mask = (bins_c > lower_limit_os) & (bins_c < upper_limit)  # & (value>0)
    bins_c, value, value_os = bins_c[mask], value[mask], value_os[mask]
    # plt.figure()  # figsize=(10,6))
    # if log:
    #     plt.step(bins_c, np.log10(value))
    #     plt.step(bins_c, np.log10(value_os))
    #     plt.ylabel("Frequency Log(#)")
    # else:
    #     plt.step(bins_c, value)
    #     plt.ylabel("Frequency # ")
    # plt.grid(True, which="both", linestyle="dotted")
    # plt.xlabel("ADU value")
    # plt.title("Histogram's sum : " + "%s_HistogramSum.csv" % (datetime.datetime.now().strftime("%y%m%d-%HH%M")))
    # plt.ylabel("Log(#)")
    # plt.grid()
    # print(path)
    # plt.savefig(os.path.dirname(path[0]) + "/HistogramSum.png" )#% (datetime.datetime.now().strftime("%y%m%d-%HH%M")))  # piki
    # plt.show()
    csvwrite(
        Table(np.vstack((bins_c, value, value_os)).T),
        os.path.dirname(path[0]) + "/Histogram_%s_%sG_%is.csv" % (date, daq, exptime),
    )
    csvwrite(
        Table(np.vstack((bins_c, value, value_os)).T),
        "/tmp" + "/Histogram_%s_%sG_%is.csv" % (date, daq, exptime),
    )

    emccd_model(
        xpapoint=None,
        path=os.path.dirname(path[0])
        + "/Histogram_%s_%sG_%is.csv" % (date, daq, exptime),
        smearing=1,
        argv=[],
    )
    return


if "" in sys.argv:
    sys.argv.remove("")
path = sys.argv[1:]
HistogramSums(path=path)
# for folder in glob.glob('/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220204_darks_T183_1MHz/7000/220329_18H54m27/EMGAIN_7000.0/*'):
# for folder in glob.glob('/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220204_darks_T183_1MHz/6800/220330_09H27m13/EMGAIN_6800/*'):
# for folder in glob.glob('/Volumes/Vincent/FIREBall_Data/190223/lowsignalT_113/220329_17H18m24/EMGAIN_9200/*'):
# for fold in glob.glob('/Users/Vincent/DS9QuickLookPlugIn/subsets/220328_12H41m59/Directory_darks_T95/*'):
#     for folder in glob.glob(fold+'/*'):
#         if os.path.isdir(folder):
#             print(folder)
#             path = glob.glob(folder + '/image*.fits')
#             print(path)
#             HistogramSums(path=path[1:10])
# sys.exit()
#%%


def EMCCDhist(
    x,
    bias=[1e3, 4.5e3, 1194],
    RN=[0, 200, 53],
    EmGain=[100, 10000, 5000],
    flux=[0.001, 1, 0.04],
    smearing=[0, 1, 0.01],
    sCIC=[0, 1, 0],
):
    # def EMCCDhist(x, bias=[1e3, 4.5e3,1194], RN=[0,200,53], EmGain=[100, 10000,5000], flux=[0.001, 1,0.04], smearing=[0, 3,0.31], sCIC=[0,1,0],SmearExpDecrement=[1.5e3,1.5e5,15e4]):
    from scipy.sparse import dia_matrix
    import inspect
    from astropy.table import Table
    from matplotlib.widgets import Button
    import numpy as np

    if bias > 1500:
        ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
    else:
        ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018

    # def variable_smearing_kernels(image, Smearing=1.5, SmearExpDecrement=50000):
    #     """Creates variable smearing kernels for inversion
    #     """
    #     import numpy as np
    #     n=6
    #     smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    #     smearing_kernels = np.exp(-np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length)
    #     smearing_kernels /= smearing_kernels.sum(axis=0)
    #     return smearing_kernels
    def variable_smearing_kernels(
        image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
    ):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np

        n = 15
        smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
        if type_ == "exp":
            smearing_kernels = np.exp(
                -np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length
            )
        else:
            assert 0 <= Smearing <= 1
            smearing_kernels = np.power(Smearing, np.arange(n))[
                :, np.newaxis, np.newaxis
            ] / np.ones(smearing_length.shape)
        smearing_kernels /= smearing_kernels.sum(axis=0)
        return smearing_kernels

    def simulate_fireball_emccd_hist(
        x,
        ConversionGain,
        EmGain,
        Bias,
        RN,
        Smearing,
        SmearExpDecrement,
        n_registers,
        flux,
        sCIC=0,
    ):
        """Silumate EMCCD histogram
        """
        import numpy as np

        try:
            y = globals()["y"]
            # print('len y=',len(y))
            # print(' y=',y[y>1])
            # print('sum',np.nansum(y[np.isfinite(y)]))
            n_pix = np.nansum(10 ** y[np.isfinite(10 ** y)])  # 1e6#
            # print("global", np.sum(10 ** y), n_pix)
        except TypeError:
            n_pix = 10 ** 6.3
            # print("fixed", np.sum(10 ** y), n_pix)
        n = 1
        # print('npix', n_pix)
        im = np.zeros(int(n_pix))  #
        im = np.zeros((1000, int(n_pix / 1000)))
        # im = np.zeros((1000,10+1))
        # factor = 1#np.log(2)
        # EmGain *= factor
        # imaADU = np.random.gamma(flux, EmGain, size=im.shape)
        # print(np.max([flux,0]),flux,EmGain)
        imaADU = np.random.gamma(
            np.random.poisson(np.nanmax([flux, 0]), size=im.shape), abs(EmGain)
        )
        # Add pCIC (no interest, as flux)
        # imaADU[np.random.rand(size[1],size[0]) <  p_pCIC] += 1

        # pixels in which sCIC electron might appear
        p_sCIC = sCIC  # / np.mean(
        #     1 / np.power(EmGain * ConversionGain, np.arange(604) / 604)
        # )
        # / np.mean(1 / np.power(EmGain * ConversionGain, np.arange(604) / 604))

        id_scic = np.random.rand(im.shape[0], im.shape[1]) < p_sCIC
        print(id_scic.sum() / id_scic.size)
        # sCIC  # sCIC positions
        # np.random.rand(im.shape[0])< p_sCIC
        # stage of the EM register at which each sCIC e- appear
        register = np.random.randint(1, n_registers, size=id_scic.sum())
        # Compute and add the partial amplification for each sCIC pixel
        imaADU[id_scic] += np.random.exponential(
            np.power(EmGain, register / n_registers)
        )
        imaADU *= ConversionGain

        if Smearing > 0:
            smearing_kernels = variable_smearing_kernels(
                imaADU, Smearing, SmearExpDecrement
            )
            offsets = np.arange(6)
            A = dia_matrix(
                (smearing_kernels.reshape((6, -1)), offsets),
                shape=(imaADU.size, imaADU.size),
            )
            imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
        read_noise = np.random.normal(0, abs(RN * ConversionGain), size=im.shape)
        imaADU += Bias
        imaADU += read_noise
        range = [np.nanmin(x), np.nanmax(x)]
        # n, bins = np.histogram(imaADU.flatten(), range=range, bins=len(x))
        # print(x)
        n, bins = np.histogram(imaADU.flatten(), bins=[x[0] - 1] + list(x))
        n_conv = 1
        return np.convolve(n, np.ones(n_conv) / n_conv, mode="same")

    y = simulate_fireball_emccd_hist(
        x=x,
        ConversionGain=ConversionGain,  # 0.53,
        EmGain=EmGain,
        Bias=bias,
        RN=RN,
        Smearing=smearing,
        SmearExpDecrement=1e4,  # 1e4,  # 1e5 #2022=1e5, 2018=1e4...
        n_registers=604,
        flux=flux,
        sCIC=sCIC,
    )
    y[y == 0] = 1
    y /= x[1] - x[0]
    # print("len(y)", np.sum(y))
    return np.log10(y)


plot(
    EMCCDhist(
        np.arange(3000),
        bias=1000,
        RN=100,
        EmGain=1000,
        flux=0.1,
        smearing=0.7,
        sCIC=0.01,
    )
)
