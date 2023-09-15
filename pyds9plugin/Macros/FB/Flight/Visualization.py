from pyds9plugin.Macros.FB.Flight.OS_correction import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian2DKernel, convolve

def VisualizationDetector3(fitsimage,path, plot_flag=False, save=True,  cbars=["fixed", "variable"], units=["counts", "counts_per_second"],emgain = 1500, conversion_gain = 1.78):
    # from astropy.table import Table
    from astropy.io import fits
    from scipy import stats, ndimage

    offset = 30
    # fitsimage = fits.open(path)[0]
    header = fitsimage.header
    data = fitsimage.data
    for cbar in cbars:
        for unit in units:
            # verboseprint(unit, cbar)
            data_biasremoved, _ = ApplyOverscanCorrection(image=data, ColumnCorrection=False,save=False)
            data_biasremoved = data_biasremoved[:, 1053 : 2133]

            texp = header["EXPTIME"]  # PrincipalCatalog['EXPTIME'][PrincipalCatalog['DETNAME'] == number]

            if unit == "counts":
                texp = 1

            if (emgain > 0) & (np.isfinite(emgain)):
                data_flux = data_biasremoved * conversion_gain / (emgain * texp)
            else:
                data_flux = data_biasremoved * conversion_gain / (texp)

            masked_array = np.ma.array(data_flux, mask=np.isnan(data_flux))
            # smoothed_image = ndimage.gaussian_filter(masked_array, sigma=(1.3, 1.3), order=0)
            kernel = Gaussian2DKernel(x_stddev=1.3,y_stddev=1.3)
            smoothed_image = convolve(masked_array, kernel)

            exposure, gain = header["EXPTIME"], header["EMGAIN"]
            try:
                temp = header["TEMP"]
            except KeyError:
                temp = header["TEMPD"]

            try:
                egain, bias, sigma = header["C_EGAIN"], header["C_BIAS"], header["C_SIGR0"]
            except KeyError:
                egain, bias, sigma = -1, -1, -1
            current_cmap = cmocean.cm.deep# current_cmap = cmocean.cm.solar# self.current_cmap = cmocean.cm.thermal
            current_cmap = plt.cm.cubehelix
            current_cmap.set_bad("black", 0.9)
            fig, (ax1,ax2) = plt.subplots(2,1, figsize=(9,6.5), gridspec_kw={'height_ratios': [0.3, 1]})
            hist_im = fitsimage.data[offset:-offset, 1053+offset : 2133-offset].flatten()
            min_, max_ = (int(np.nanpercentile(hist_im, 0.4)-5), int(np.nanpercentile(hist_im, 99)))
            hist_im1 = fitsimage.data[offset:1000, 1053+offset : 2133-offset].flatten()
            hist_im2 = fitsimage.data[1000:-offset, 1053+offset : 2133-offset].flatten()
            vals1, b = np.histogram(hist_im1[np.isfinite(hist_im1)],bins=300)
            vals2, b = np.histogram(hist_im2[np.isfinite(hist_im2)],bins=300)

            params =  "Emgain calc = %i ADU\nbias = %i ADU\n" "sigma = %0.2f ADU" % (egain, bias, sigma)#, fontsize=15, bbox={"facecolor": "blue", "alpha": 0.5, "pad": 10})
            params2 =  "Exposure = %i sec\nGain = %i \n" "T det = %0.2f C" % (exposure, gain, float(temp))#, fontsize=15, bbox={"facecolor": "blue", "alpha": 0.5, "pad": 10})
            params3 =  "Mean = %0.5f %s\nStd = %0.3f %s\n" "SKew = %0.3f %s" % (np.nanmean(data_flux), unit, np.nanstd(data_flux), unit, stats.skew(data_flux, axis=None, nan_policy="omit"), unit)#, fontsize=15, bbox={"facecolor": "blue", "alpha": 0.5, "pad": 10})


            ax1.semilogy((b[1:]+b[:-1])/2,vals1,c="k",label=params2)
            ax1.semilogy((b[1:]+b[:-1])/2,vals2,c="gray")
            if cbar == "variable":
                #                    plt.imshow(smoothed_image[:,:].T, cmap=cmap,vmin=np.nanpercentile(smoothed_image,50),vmax=np.nanpercentile(smoothed_image,99.95))#, interpolation='nearest')
                im= ax2.imshow(smoothed_image[:, :].T, cmap=current_cmap, vmin=np.nanpercentile(smoothed_image, 10), vmax=np.nanpercentile(smoothed_image, 99.99),label=params3)
            if cbar == "fixed":
                if unit == "counts":
                    im= ax2.imshow(smoothed_image[:, :].T, cmap=current_cmap, vmin=0, vmax=2,label=params3)
                if unit == "counts_per_second":
                    im= ax2.imshow(smoothed_image[:, :].T, cmap=current_cmap, vmin=0, vmax=0.03,label=params3)

            ax1.legend()
            ax2.text(50, 170, params3, bbox={'facecolor': 'white', 'pad': 10,"alpha":0.5})
            cax0 = make_axes_locatable(ax2).append_axes('right', size='2%', pad=0.05)
            fig.colorbar(im, cax=cax0)#, orientation='vertical')
            ax1.set_title(os.path.basename(path) + " - " + header["DATE"] + " - map [e-] " + unit)
            name = os.path.dirname(path) + "/DetVisualization/Maps_" + unit + "_" + cbar + "scale/"
            # verboseprint(name)
            try:
                os.makedirs(name)
            except OSError:
                pass
            fig.tight_layout()
            fig.savefig(name + os.path.basename(path)[:-5] + ".png", dpi=100, bbox_inches="tight")
            if plot_flag:
                plt.show()
            else:
                plt.close(fig)
    return

if __name__ == "__main__":
    VisualizationDetector3(fitsimage, filename, plot_flag=False, save=True, cbars=["fixed", "variable"], units=["counts"],emgain = 1500, conversion_gain = 1)








		
		
		
