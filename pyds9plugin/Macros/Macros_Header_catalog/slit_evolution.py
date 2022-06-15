#%%
print(0)
from pyds9plugin.testing.startup_spyder import *
from scipy.optimize import curve_fit
from pyds9plugin.Macros.Fitting_Functions.functions import slit, smeared_slit

print(1)
d = DS9n()
# path = get(d, "Path of the files to analyzed")
path = "/Volumes/SSDPICOUET/LAM/FIREBALL/2022/DetectorData/220610/DIFFUSE_FOCUS_W_TEMP/*[05].fits"

files = glob.glob(path)
regs = getregion(d, selected=True)
files.sort()
n = 15


temp = header["TEMPB"]
for i, reg in enumerate(regs):
    x_inf, x_sup, y_inf, y_sup = lims_from_region(region=regs[i], coords=None)
    subim = fitsfile[0].data[y_inf - n : y_sup + n, x_inf - n : x_sup + n][::-1, :]
    stack = np.nanmean(subim, axis=0)
    stack = (stack - stack.min()) / (stack - stack.min()).ptp()

    x_spectral = np.arange(len(stack))
    y_spectral = stack[::-1]
    P0 = [
        y_spectral.ptp(),
        3.3,
        len(y_spectral) / 2,
        1.7,
        np.median(y_spectral),
        1.2,
    ]
    bounds = [
        [0.7 * y_spectral.ptp(), 0, 0, 0, np.nanmin(y_spectral), 0.1],
        [
            y_spectral.ptp(),
            len(y_spectral),
            len(y_spectral),
            10,
            np.nanmax(y_spectral),
            6,
        ],
    ]
    # PlotFit1D(x_spectral, y_spectral, deg=smeared_slit, P0,)
    try:
        popt_spectral_deconvolved, pcov = curve_fit(
            smeared_slit, x_spectral, y_spectral, p0=P0
        )
    except RuntimeError:
        pass
    else:
        table["slit_%i_amp" % i] = popt_spectral_deconvolved[0]
        table["slit_%i_l" % i] = popt_spectral_deconvolved[1]
        table["slit_%i_x0" % i] = popt_spectral_deconvolved[2]
        table["slit_%i_FWHM" % i] = popt_spectral_deconvolved[3]
        table["slit_%i_offset" % i] = popt_spectral_deconvolved[4]
        table["slit_%i_smearing" % i] = popt_spectral_deconvolved[5]

    # plt.plot(x_spectral, y_spectral)
    # plt.plot(x_spectral, smeared_slit(x_spectral, *popt_spectral_deconvolved))
    # plt.plot(x_spectral, smeared_slit(x_spectral, *P0))
    # plt.show()
    # print(popt_spectral_deconvolved)

# sys.exit()
