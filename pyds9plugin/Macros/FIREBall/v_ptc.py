from pyds9plugin.DS9Utils import *#DS9n,PlotFit1D
from pyds9plugin.DS9Utils import blockshaped
# from pyds9plugin.Macros import functions
#TODO  create histogram for every image
from astropy.table import Column
from astropy.io import fits
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib


all_intensities = []
all_vars = []
n=20
# im = fits.open("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/ptc_T183_1MHz_405nm/image000027.fits")
# for file in globglob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/ptc_T183_1MHz_405nm/image00004?.fits"):
for file in globglob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/20220508/PTC/image_000000[61-61].fits"):
    im = fits.open(file)
    region = im[0].data[:1900,1100:2100] -  im[0].data[:,:100].mean()
    appertures = blockshaped(region, 20, 20)
    
    vars_ = list(np.nanvar(appertures,axis=(1,2)))
    intensities = list(np.nanmean(appertures,axis=(1,2)))
    all_intensities+=intensities
    all_vars+=vars_
    vars_masked, intensities_masked= SigmaClipBinned(vars_, intensities, sig=1,Plot=False)
    fig, ax = plt.subplots()
    ax.plot(intensities,vars_,'.')
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    mask = np.array(intensities)<4000
    p=PlotFit1D(np.array(intensities)[mask],np.array(vars_)[mask],deg=1,ax=ax, plot_=True)['popt']#,P0=[1,bias,50,0]
    ax.plot(intensities,vars_,'.',label=p)
    # ax.plot(all_intensities,all_vars,'.')#,label=p)
    ax.legend()
    ax.set_title(os.path.basename(file))
    plt.show()