#%%
from pyds9plugin.DS9Utils import *  # DS9n,PlotFit1D
from pyds9plugin.DS9Utils import blockshaped

# from pyds9plugin.Macros import functions
# TODO  create histogram for every image
from astropy.table import Column
from astropy.io import fits
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

#%%
all_intensities = []
all_vars = []
n = 20
# im = fits.open("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/ptc_T183_1MHz_405nm/image000027.fits")
# for file in globglob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/ptc_T183_1MHz_405nm/image00004?.fits"):

# ds9 = np.hstack(
#     [
#         fits.open(f)[0].data[:, 1000:-1000]
#         for f in glob.glob(
#             "/Volumes/ExtremePro/LAM/FIREBALL/2022/DetectorData/220618/CIT_NUVU_m100_ptc/image??????.fits"
#         )
#     ]
# )
path = "/Volumes/ExtremePro/LAM/FIREBALL/2019-01+MarseilleTestsImages/DetectorAnalysis/TestVincent/190225/highsignalT_113/image0000??.fits"
# "/Volumes/ExtremePro/LAM/FIREBALL/2022/DetectorData/220618/CIT_NUVU_m100_ptc/image??????.fits"
for file in globglob(path):
    im = fits.open(file)
    region = im[0].data[:1900, 1100:2100] - im[0].data[:, :100].mean()
    appertures = blockshaped(region, 20, 20)

    vars_ = list(np.nanvar(appertures, axis=(1, 2)))
    intensities = list(np.nanmean(appertures, axis=(1, 2)))
    all_intensities += intensities
    all_vars += vars_
    vars_masked, intensities_masked = SigmaClipBinned(
        vars_, intensities, sig=1, Plot=False
    )
    fig, ax = plt.subplots()
    ax.loglog(intensities, vars_, ".")  # , label=p)
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    mask = np.array(intensities) < 4000
    p = PlotFit1D(
        np.array(intensities)[mask], np.array(vars_)[mask], deg=1, ax=ax, plot_=True
    )[
        "popt"
    ]  # ,P0=[1,bias,50,0]
    ax.loglog(intensities, vars_, ".", label=p)
    # ax.plot(intensities, vars_, ".", label=p)
    # ax.plot(all_intensities,all_vars,'.')#,label=p)
    ax.legend()
    ax.set_title(os.path.basename(file))
    plt.show()

# %%


fig, ax = plt.subplots()
ax.loglog(all_intensities, all_vars, ".", label=p, alpha=0.005)
ax.legend()
ax.set_title(os.path.basename(file))
plt.show()
# %%

np.savetxt("/tmp/test.dat", np.array([all_intensities, all_vars]).T)

# %%
