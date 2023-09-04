from astropy.io import fits
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pyds9plugin.DS9Utils import verboseprint
from pyds9plugin.Macros.FB.FB_functions import emccd_model


def apply_pc(image, bias, sigma, threshold=5.5):
    """Put image pixels to 1 if superior to threshold and 0 else
    """
    cutoff = int(bias + sigma * threshold)  # )5.5)
    idx = image > cutoff - 1
    image[idx] = np.ones(1, dtype=np.uint8)[0]
    image[~idx] = np.zeros(1, dtype=np.uint8)[0]

    mask = np.isfinite(image)
    image[~mask] = np.zeros(1, dtype=np.uint8)[0]
    return np.array(image,dtype=np.uint8)


def DS9photo_counting(image, header, filename, threshold=5.5,plot_flag=False):
    """Calculate threshold of the image and apply phot counting
    """
    verboseprint("Threshold = %0.2f" % (threshold))
    verboseprint("plot_flag = ", plot_flag)
    Xinf, Xsup, Yinf, Ysup = [0,2069,1172,2145]
    image_area = [Yinf, Ysup, Xinf, Xsup]
    verboseprint(Yinf, Ysup, Xinf, Xsup)
    # D = calc_emgainGillian(filename, area=image_area, plot_flag=plot_flag)
    # emgain, bias, sigma, frac_lost = [D[x] for x in ["emgain", "bias", "sigma", "frac_lost"]]  # D[my_conf.gain[0],'bias','sigma','frac_lost']
    # b = [D[x] for x in ["image", "emgain", "bias", "sigma", "bin_center", "n", "xlinefit", "ylinefit", "xgaussfit", "ygaussfit", "n_bias", "n_log", "threshold0", "threshold55", "exposure", "gain", "temp"]]


    if "hist_bias" not in list(dict.fromkeys(header.keys())):
        
        if header["EMGAIN"]==9200:
            # 2018
            fit_param = emccd_model(xpapoint=None, path=filename, smearing=1.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.53,RN=40,mCIC=0.15,sCIC=0.02,gain=1400,RON=105*0.53)#,mCIC=0.005

        elif  header["ROS"]==2:
            #2023 s2_hdr
            fit_param = emccd_model(xpapoint=None, path=filename, smearing=0.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.97,RON=42)#,mCIC=0.,sCIC=0.02,RON=105*0.53)#,mCIC=0.005
        # else:
        elif  header["ROS"]==1:
            #2023 s2
            fit_param = emccd_model(xpapoint=None, path=filename, smearing=0.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.053,RON=2.6)#,mCIC=0.,sCIC=0.02,RON=105*0.53)#,mCIC=0.005
        elif  header["ROS"]==5:
            #2022
            fit_param = emccd_model(xpapoint=None, path=filename, smearing=0.5,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.22,RON=10)#,mCIC=0.,sCIC=0.02,RON=105*0.53)#,mCIC=0.005
        else:
            fit_param = emccd_model(xpapoint=None, path=filename, smearing=0.2,fit="EMCCDhist", argv=[],gui=False,conversion_gain=0.72,RON=17)#,mCIC=0.,sCIC=0.02,RON=105*0.53)#,mCIC=0.005
        new_image = apply_pc(image=image, bias=fit_param["BIAS"], sigma=fit_param["RON"], threshold=threshold)
    else:
        new_image = apply_pc(image=image, bias=float(header["hist_bias"]), sigma=float(header["hist_ron"]), threshold=threshold)

    return new_image

if __name__ == "__main__":
    new_path = filename.replace(".fits","_pc.fits")
    ds9 = DS9photo_counting(image=ds9, header=header,filename=filename, threshold=5.5)

