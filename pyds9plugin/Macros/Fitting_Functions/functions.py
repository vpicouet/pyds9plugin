#%%

# Function used to fit some DS9 plots
# Please be sure to define a list as default parameters for each arguments
# as they will be used to define the lower and upper bounds of each widget.
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
try:
    x, y, z = np.loadtxt("/tmp/xy.txt").T
    # x, y =  np.array(x), np.array(y)
    # print(np.log10(np.sum(10**y)))
except (OSError, ValueError) as e:
    try:
        x, y = np.loadtxt("/tmp/xy.txt").T

    except (OSError, ValueError) as e:
        x, y = np.array([0, 1]), np.array([0, 1])



dispersion = 4.6
vf = dispersion /2060
 

def variable_smearing_kernels(
    image, Smearing=0.7, SmearExpDecrement=50000, ratio=1, type_="exp"
):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np

    n = 15
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    # smearing_length = Smearing * np.ones(image.shape)#np.exp(-image / SmearExpDecrement)
    if type_ == "exp":
        try:
            x = np.arange(n)[
                :: int(np.sign(smearing_length[0])), np.newaxis, np.newaxis
            ]
        except ValueError:
            x = np.arange(n)[:, np.newaxis, np.newaxis]
        smearing_kernels = ratio * np.exp(-x / abs(smearing_length))
        if ratio != 1:
            smearing_kernels[0, :, :] = 1
        # smearing_kernels = amp*np.exp(-np.arange(n)[:, np.newaxis, np.newaxis] / abs(smearing_length))
        # smearing_kernels[0,:,:] = 1

    else:
        assert 0 <= Smearing <= 1
        smearing_kernels = np.power(Smearing, np.arange(n))[
            :, np.newaxis, np.newaxis
        ] / np.ones(smearing_length.shape)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels



def convolve_diskgaus_2d(x, amp=y.ptp() * np.array([0, 1.3, 1]), R=x.ptp() * np.array([0, 1, 0.1]), σ=x.ptp() * np.array([0, 1, 0.1]), offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)])):
    """Convolution of a disk with a gaussian to simulate the image of a fiber
    """
    from scipy.integrate import quad
    from scipy import special
    import numpy as np

    integrand = (
        lambda eta, r_: special.iv(0, r_ * eta / np.square(σ))
        * eta
        * np.exp(-np.square(eta) / (2 * np.square(σ)))
    )
    integ = [
        quad(integrand, 0, R, args=(r_,))[0]
        * np.exp(-np.square(r_) / (2 * np.square(σ)))
        / (np.pi * np.square(R * σ))
        for r_ in x
    ]
    # print(integ)
    integ = np.array(integ)
    integ /= integ.ptp()
    # print(integ)
    return offset + amp * integ

def slit(
    x,
    amp=y.ptp() * np.array([0, 1.3, 1]),
    l=len(y) * np.array([0, 1, 0.2]),
    x0=len(y) * np.array([0, 1, 0.5]),
    FWHM=len(y) * np.array([0, 1/2, 2/len(y)]),
    offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)]),
):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np

    l /= 2
    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    function = amp * (a + b) / (a + b).ptp()  # 4 * l
    return function + offset


#defocused_gaussian
def line_analysis(
    x, amp=y.ptp() * np.array([-1.3, 1.3, 1]), amp_2=y.ptp() * np.array([-1.3, 1.3, -0.5]), sigma=[0, 100,50], sigma_2=[0, 100,3], v1=[x.min()/vf,x.max()/vf,x[np.argmax(y)],0],v2=[x.min()/vf,x.max()/vf,10]
):
    """Defines a gaussian function with offset
    """
    import numpy as np
    # xo = float(xo)
    g = amp * np.exp(-0.5 * (np.square(x - v1*vf) / sigma ** 2))
    g2 = amp_2 * np.exp(-0.5 * (np.square(x - v2*vf) / sigma_2 ** 2))
    return (g + g2).ravel()


#TODO Throughput should be the real throughput when we know exposure time and all
#TODO add an offset in wavelegth for the QE and atm?
def fit_spectra(x,lmax=[1900,2130,2060],dispersion=[0.8,1.2,1],throughput=[0,y.ptp(),y.ptp()/100], qe=[0,1.5,0.5],atm=[0,25,0.5], noise=[0,1,1]): #spectral_res=[1,15,5]
    from astropy.modeling.functional_models import Gaussian2D, Gaussian1D
    from scipy.interpolate import interp1d
    from astropy.table import Table
    from scipy.ndimage import gaussian_filter1d
    # spectral_res = 5
    # throughput=1
    area=7854
    dispersion *= 46.6/10
    Rx=5
    spectral_res=5
    wavelengths = np.linspace(lmax-len(x)/2/dispersion,lmax+len(x)/2/dispersion,len(x))
    # a=Table.read("/Users/Vincent/Nextcloud/LAM/Work/FIREBall/Simulation_fields/h_1821p643fos_spc.fits")
    a=Table.read("/Users/Vincent/Nextcloud/LAM/Work/FIREBall/Simulation_fields/h_1538p477fos_spc.fits")
    trans = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/transmission_pix_resolution.csv")
    QE = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/PSFDetector/efficiencies/QE_2022.csv")
    # QE = Table.read("/Users/Vincent/Github/FIREBall_IMO/Python Package/FireBallIMO-1.0/FireBallIMO/PSFDetector/efficiencies/5LayerModel_refl.txt")
    QE = interp1d(QE["wave"]*10,QE["QE_corr"])#
    trans["trans_conv"] = gaussian_filter1d(trans["col2"], spectral_res)# np.convolve(trans["col2"],np.ones(5)/5,mode="same")
    trans = trans[5:-5]
    atm_trans =  interp1d([1500,2500]+list(trans["col1"]*10),[0,0] + list(trans["trans_conv"]))#

    a["photons"] = a["FLUX"]/9.93E-12   
    a["e_pix_sec"]  = a["photons"] * throughput  * area /dispersion #* atm 
    nsize,nsize2 = 100,500
    source_im=np.zeros((nsize,nsize2))
    source_im_wo_atm=np.zeros((nsize2,nsize))
    mask = (a["WAVELENGTH"]>1960) & (a["WAVELENGTH"]<2280)
    lmax = a["WAVELENGTH"][mask][np.argmax( a["e_pix_sec"][mask])]
    # plt.plot( a["WAVELENGTH"],a["e_pix_sec"])
    # plt.plot( a["WAVELENGTH"][mask],a["e_pix_sec"][mask])
    f = interp1d(a["WAVELENGTH"],a["e_pix_sec"])#
    profile =   Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx) /Gaussian1D.evaluate(np.arange(nsize),  1,  nsize/2, Rx).sum()
    # subim = np.zeros((nsize2,nsize))
    # print(f(wavelengths))
    final_QE = qe*QE(wavelengths) + (1-qe)*np.ones(len(QE(wavelengths)))
    final_atm = atm*atm_trans(wavelengths) + (1-atm)*np.ones(len(atm_trans(wavelengths)))
    f_final = noise*f(wavelengths) + (1-noise)*np.ones(len(f(wavelengths)))
    final = f_final * final_atm * final_QE
    return throughput*final-np.nanmin(final)/np.ptp(final-np.nanmin(final))
    





#%%


def slit_astigmatism(
    x,
    amp=y.ptp() * np.array([0, 1.3, 1]),
    l=len(y) * np.array([0, 1, 0.2]),
    x0=len(y) * np.array([0, 1, 0.5]),
    FWHM=[0.1, 35, 2],
    astigm=[0, 50, 2],
    offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)]),
):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np
    from scipy import signal

    l /= 2
    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    function = amp * (a + b) / (a + b).ptp()  # 4 * l
    astigm = int(astigm)
    if astigm > 1:
        triang = signal.triang(astigm) / signal.triang(astigm).sum()
        function = np.convolve(function, triang, mode="same")
    return function + offset


def moffat_profile(
    x,
    amp=y.ptp() * np.array([0, 1.3, 1]),
    σ=[0.01, 50, 4],
    alpha=[0, 10, 2.5],
    offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)]),
):
    """1D moffat function
    """
    import numpy as np

    x0 = 0
    return amp * np.power((1 + np.square((x - x0) / σ)), -alpha) + offset



def atm(
    lambda_,
    Flux_ADU_per_A=y.ptp() * np.array([0, 1.3, 1]),
    Altitude_km=[25, 40, 30],
    # offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)]),
    Dist_from_center_pixel=np.array([-500,500,0]), 
):
    atm = Table.read("/Users/Vincent/Github/FIREBallIMO/FireBallIMO/Atmosphere/AtmTrans_wave_180-239-0.1000nm_Alt_25-45.csv")
    det_efficiency = Table.read("/Users/Vincent/Github/FIREBallIMO/FireBallIMO/PSFDetector/efficiencies/QE_detector_2022_D10.csv")[::-1]
    det_efficiency = Table.read("/Users/Vincent/Github/FIREBallIMO/FireBallIMO/PSFDetector/efficiencies/DetectorQE-20170808.csv")#[::-1]
    

    absorption = atm["Trans_%ikm"%(Altitude_km)]
    absorption = np.convolve(absorption,np.ones(10)/10,mode="same")

    # x=np.linspace(190,210,100)
    dispersion = 0.2137 #anstrom per pixels
    if lambda_[0]<1700:
        lambda_ = np.linspace(2060-dispersion*len(lambda_)/2 - Dist_from_center_pixel*dispersion,2060+dispersion*len(lambda_)/2 - Dist_from_center_pixel*dispersion,len(lambda_))
    atm_interp = np.interp(lambda_,10*atm[atm.colnames[0]],absorption)
    det_interp = np.interp(lambda_,10*det_efficiency[det_efficiency.colnames[0]],det_efficiency[det_efficiency.colnames[1]])


    return Flux_ADU_per_A * dispersion * atm_interp * det_interp #+ offset

# def gaussian_profile(
#     x,
#     amp=y.ptp() * np.array([0, 1.3, 1]),
#     FWHM=[0.01, 50, 2],
#     offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)]),
# ):
#     FWHM /= 2.35
#     return offset + amp * np.exp(-np.square(x / FWHM) / 2)

def fiber_radial_profile(
    r,
    amp=y.ptp() * np.array([0, 1.3, 1]),
    rad_fiber=[0, 30, 5],
    FWHM=[0.01, 50, 2],
    offset=np.array([np.nanmin(y), np.nanmax(y), np.nanmin(y)]),
):
    """Convolution of a disk with a gaussian to simulate the image of a fiber
    """
    from scipy.integrate import quad
    from scipy import special
    import numpy as np

    FWHM /= 2.35
    integrand = (
        lambda eta, r_: special.iv(0, r_ * eta / np.square(FWHM))
        * eta
        * np.exp(-np.square(eta) / (2 * np.square(FWHM)))
    )
    integ = [
        quad(integrand, 0, rad_fiber, args=(r_,))[0]
        * np.exp(-np.square(r_) / (2 * np.square(FWHM)))
        / (np.pi * np.square(rad_fiber * FWHM))
        for r_ in r
    ]
    integ = np.array(integ) / np.max(integ)
    return offset + amp * integ


def madau(z,rho=[0.001,0.01],n=[2,3],n2=[2,3],pow=[2,7]):
    return np.log10((rho * (1 + z/10) ** n / (1 + ((1 + z/10) / n2) ** pow)))


# def gaussian_flux(x, Flux=[0, np.nansum(y)], xo=[0, 1.5*len(x)], sigma=[0, 10]):
#     """Defines a gaussian function with offset
#     """
#     import numpy as np
#     xo = float(xo)
#     g = np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
#     g *= Flux / g.sum()
#     return g.ravel()


def gaussian_flux(x, Flux=len(y)*y.ptp() * np.array([-2, 2, 1/5]), xo=[-0.5*len(x), 1.5*len(x),0.5*len(x)], sigma=[0, x.ptp(),5]):#off=[np.nanmin(y)-y.ptp(),np.nanmax(y)+ y.ptp(),np.nanmean(y)]#, sigma=[0, x.ptp(),x.ptp()/2]
    """Defines a gaussian function with offset
    """
    import numpy as np
    xo = float(xo)
    g = np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
    g *= Flux / g.sum()
    return g.ravel()#+off




def schechter(x, phi=[1e-3, 1e-2], m=[16, 22], alpha=[-2, -1]):
    """ Schecter function for luminosity type function
    """
    import numpy as np

    y = np.log10(
        0.4
        * np.log(10)
        * phi
        * 10 ** (0.4 * (m - x) * (alpha + 1))
        * (np.e ** (-pow(10, 0.4 * (m - x))))
    )
    return y[::-1]


def double_schechter(
    x, phi=[1e-3, 1e-2], alpha=[-2, -1], M=[16, 22], phi2=[1e-3, 1e-2], alpha2=[-2, -1],
):
    return np.log10(
        10 ** schechter(x, phi, M, alpha) + +(10 ** schechter(x, phi2, M, alpha2))
    )


test = True
test = False


# def EMCCD(
#     x,
#     bias=[x.min(), x.max(), x[np.argmax(y)]],
#     RN=[5, 350, 12],
#     EmGain=[10, 2000, 1900],
#     flux=[0, 1, 0.01],
#     smearing=[0, 3, 0.01],
#     sCIC=[0, 2, 0],
# ):
#     """EMCCD model based on convolution of distributions: Gamma(poison)xNormal
#     First attempt to add smearing
#     RN : Read noise in e-/pix
#     EmGain : amplificaiton gain in e-/e-
#     flux : incoming charges in e-/pix
#     smearing : exponential length in pixel of the charge decay due to poor CTE
#     sCIC : fraction of semi-amplified spurious charges that appear in the amplifier register
#     """
#     from astropy.convolution import Gaussian1DKernel, convolve
#     import scipy.special as sps
#     import numpy as np
#     from scipy.stats import poisson
#     # recover number of pixels to generate distributions
#     try:
#         y = globals()["y"]
#         n_pix = np.sum(10 ** y)
#     except TypeError:
#         n_pix = 10 ** 6.3
#     n_registers = 604  # number of amplification registers
#     ConversionGain = 1 
#     bin_size = np.median((x[1:] - x[:-1]))
#     bins = x - np.nanmin(x)
#     distributions = []
#     pixs_sup_0 = 0  # fraction of pixels that have ADU higher than distribution limit
#     n = 0
#     i=0
#     flux_=flux
#     gamma_distribution = 0
#     smearing_kernel = np.exp(-bins / smearing)
#     smearing_kernel /= smearing_kernel.sum()    

#     # We sum up distributions from from the different poisson output
#     for f in np.arange(1, 20):  # careful about the 20 limit which needs to be enough at high flux
#         v = poisson.pmf(k=f, mu=np.nanmax([flux_, 0]))
#         # gamma distribution : https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
#         denominator = sps.gamma(f) * (EmGain * ConversionGain) ** f
#         distribution = (
#             bin_size
#             * bins ** (f - 1)
#             * (np.exp(-bins / (EmGain * ConversionGain)) / denominator)
#         )
#         distribution_up = (3* bin_size
#             * (3 * bins + bins.ptp()) ** (f - 1)
#             * (np.exp(-(3 * bins + bins.ptp()) / (EmGain * ConversionGain))
#                 / denominator))

#         distribution = np.convolve(distribution, smearing_kernel, mode='same')
#         distribution_up = np.convolve(distribution_up, smearing_kernel, mode='same')


#         # disminush the number of pixels by the fraction above distribution range
#         factor = np.sum(distribution[np.isfinite(distribution_up)]) / (
#             np.sum(distribution_up[np.isfinite(distribution_up)])
#             + np.sum(distribution[np.isfinite(distribution)])
#         )
#         gamma_distribution += distribution * v
#         pixs_sup_0 += (1 - factor) * v
#     gamma_distribution[0] = (
#         1
#         - np.nansum(gamma_distribution[1:][np.isfinite(gamma_distribution[1:])])
#         - pixs_sup_0
#     )
#     # adding sCIC and comvolving distributions as independant draws
#     if sCIC > 0:
#         # changing total sCIC (e-) into the percentage of pixels experiencing spurious electrons
#         p_sCIC = sCIC  # / np.mean(1 / np.power(EmGain * ConversionGain, np.arange(604) / 604))
#         # Estimation of average gain for semi-amplified CIC
#         gain_ = np.power(EmGain * ConversionGain, np.linspace(1, n_registers, 100) / n_registers)
#         cic_disdribution = np.sum([(1 / gaini) * np.exp(-bins / gaini) for gaini in gain_], axis=0)  # *n_pix/len(gain_)
#         cic_disdribution /= cic_disdribution.sum()
#         cic_disdribution *= p_sCIC
#         cic_disdribution[0] = 1 - np.sum(cic_disdribution[1:])
#         cic_disdribution = np.hstack(
#             (np.zeros(len(cic_disdribution) - 1), cic_disdribution)
#         )
#         gamma_distribution = np.convolve(
#             gamma_distribution, cic_disdribution, mode="valid"
#         )


#     # Addition of the bias
#     if bias > x[0]:
#         gamma_distribution[(x > bias)] = gamma_distribution[: -np.sum(x <= bias)]
#         gamma_distribution[x < bias] = 0
#     read_noise = Gaussian1DKernel(
#         stddev=RN * ConversionGain / bin_size, x_size=int(301.1 * 10)
#     )
#     # Convolution with read noise
#     y = convolve(gamma_distribution, read_noise) * n_pix  #
#     # y /= x[1] - x[0]
#     return np.log10(y)






def simulate_emccd_image(
    ConversionGain,
    EmGain,
    Bias,
    RN,
    Smearing,
    SmearExpDecrement,
    flux,
    sCIC=0,
    n_registers=604,
):
    """Silumate EMCCD histogram
    flux = flux + dark + CIC 
    im = ConversionGain * gamma( poisson(np.nanmax([flux, 0]), abs(EmGain) )
    gamma( poisson(1 * id_scic), np.power(EmGain, np.random.randint(1, n_registers, size=id_scic.shape)  / n_registers)
    smearing(im) + normal(Bias, abs(RN * ConversionGain))

    """
    import numpy as np
    from scipy.sparse import dia_matrix

    # recover number of pixels to generate distributions
    try:
        y = globals()["y"]
        n_pix = np.nansum(10 ** y[np.isfinite(y)])  # 1e6#
        # print(y, n_pix)
        # print(1, n_pix)
    except TypeError:
        n_pix = 10 ** 6.3
        # print(2)
    n = 1
    im = np.zeros(int(n_pix))  #
    im = np.zeros((1000, int(n_pix / 1000)))
    # print(flux)
    imaADU = np.random.gamma(np.random.poisson(np.nanmax([flux, 0]), size=im.shape), abs(EmGain))
    # imaADU = np.random.gamma(np.nanmax([flux, 0])*np.ones(im.shape), abs(EmGain))
    # changing total sCIC (e-) into the percentage of pixels experiencing spurious electrons
    p_sCIC = sCIC  # / np.mean(
    #     1 / np.power(EmGain * ConversionGain, np.arange(604) / 604)
    # pixels in which sCIC electron might appear
    id_scic = np.random.rand(im.shape[0], im.shape[1]) < p_sCIC
    # stage of the EM register at which each sCIC e- appear
    register = np.random.randint(1, n_registers, size=id_scic.shape)
    # Compute and add the partial amplification for each sCIC pixel
    # when f=1e- gamma is equivalent to expoential law
    # should we add poisson here?
    imaADU += np.random.gamma(np.random.poisson(1 * id_scic), np.power(EmGain, register / n_registers))
    # imaADU += np.random.gamma(1 * id_scic, np.power(EmGain, register / n_registers))
    # imaADU[id_scic] += np.random.gamma(1, np.power(EmGain, register / n_registers))
    imaADU *= ConversionGain
    # smearing data
    def variable_smearing_kernels(
        image, Smearing=0.7, SmearExpDecrement=50000/2, type_="exp"
    ):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np
        n = 30
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

    if Smearing > 0:
        smearing_kernels = variable_smearing_kernels(
            imaADU, Smearing, SmearExpDecrement
        )
        n_smearing = smearing_kernels.shape[0]
        offsets = np.arange(n_smearing)
        A = dia_matrix(
            (smearing_kernels.reshape((n_smearing, -1)), offsets),
            shape=(imaADU.size, imaADU.size),
        )
        imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
    # adding read noise and bias
    read_noise = np.random.normal(0, abs(RN * ConversionGain), size=im.shape)
    imaADU += Bias
    imaADU += read_noise
    return imaADU


def EMCCDhist(
    x,
    bias=[1e3, 4.5e3, 1194],
    RN=[0, 350, 53],
    EmGain=[100, 10000, 5000],
    flux=[0.0001, 1, 0.04],
    smearing=[0, 1.8, 0.01],
    sCIC=[0, 1, 0],
):
    """
    Stochastic model for EMCCD histogram.
    RN : Read noise in e-/pix
    EmGain : amplificaiton gain in e-/e-
    flux : incoming charges in e-/pix
    smearing : exponential length in pixel of the charge decay due to poor CTE
    sCIC : fraction of semi-amplified spurious charges that appear in the amplifier register
    """
    # def EMCCDhist(x, bias=[1e3, 4.5e3,1194], RN=[0,200,53], EmGain=[100, 10000,5000], flux=[0.001, 1,0.04], smearing=[0, 3,0.31], sCIC=[0,1,0],SmearExpDecrement=[1.5e3,1.5e5,15e4]):
    from scipy.sparse import dia_matrix
    import inspect
    from astropy.table import Table
    # from matplotlib.widgets import Button
    import numpy as np

    # if bias > 1500:
    #     ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
    # else:
    #     ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018
    ConversionGain = 1  # /4.5


    # print("flux=",flux,"Smearing=", smearing)
    imaADU = simulate_emccd_image(
        ConversionGain=ConversionGain, 
        EmGain=EmGain,
        Bias=bias,
        RN=RN,
        Smearing=smearing,
        SmearExpDecrement=1e10,  # 1e4,  # 1e5 #2022=1e5, 2018=1e4...
        n_registers=604,
        flux=flux,
        sCIC=sCIC,
    )

    range = [np.nanmin(x), np.nanmax(x)]
    y, bins = np.histogram(imaADU.flatten(), bins=[x[0] - 1] + list(x))
    b = (bins[1:] + bins[:-1])/2
    flux=np.nansum(y[b>bias+5.5*RN]) / np.nansum(y)
    # print(flux)
    y[y == 0] = 1.0

    # y = y / (x[1] - x[0])
    # y = y / (bins[1] - bins[0])
    # print(np.nansum(y))
    return np.log10(y)



def smeared_slit_ratio(
    x,
    amp=y.ptp() * np.array([0.7, 1.3, 1]),
    l=[0, len(y), 4],
    x0=len(y) * np.array([0, 1, 0.5]),
    FWHM=[0.1, 10, 2],
    offset=np.nanmin(y) * np.array([0.5, 3, 1]),
    Smearing=[-5, 5, 0.8],
    ratio=[0.01, 1, 0.9],
):  # ,SmearExpDecrement=[1,500000,40000]):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np
    from scipy.sparse import dia_matrix

    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    function = amp * (a + b) / (a + b).ptp()  # +1#4 * l
    # function = np.vstack((function,function)).T
    smearing_kernels = variable_smearing_kernels(function, Smearing, ratio=ratio)
    n = smearing_kernels.shape[0]
    # print(smearing_kernels.sum(axis=1))
    # print(smearing_kernels.sum(axis=1))
    A = dia_matrix(
        (smearing_kernels.reshape((n, -1)), np.arange(n)),
        shape=(function.size, function.size),
    )
    function = A.dot(function.ravel()).reshape(function.shape)
    # function = np.mean(function,axis=1)
    return function + offset


def smeared_slit(
    x,
    amp=y.ptp() * np.array([0.7, 1.3, 1]),
    l=[0.1, len(y), 4],
    # x0=len(y) * np.array([0, 1, 0.5]),
    x0=[x.min(),x.max(),(x.min()+x.max())/2],

    FWHM=[0.1, 10, 2],
    offset=np.nanmin(y) * np.array([0.5, 3, 1]),
    Smearing=[-10, 10, 0.8],
):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np
    from scipy.sparse import dia_matrix

    if Smearing < 0:
        x0 += 14
    l /= 2
    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    function = amp * (a + b) / (a + b).ptp()  # +1#4 * l
    # function = np.vstack((function,function)).T
    smearing_kernels = variable_smearing_kernels(
        function, Smearing, SmearExpDecrement=50000
    )
    n = smearing_kernels.shape[0]
    # print(smearing_kernels.sum(axis=1))
    # print(smearing_kernels.sum(axis=1))
    A = dia_matrix(
        (smearing_kernels.reshape((n, -1)), np.arange(n)),
        shape=(function.size, function.size),
    )
    function = A.dot(function.ravel()).reshape(function.shape)
    # function = np.mean(function,axis=1)
    return function + offset


def smeared_slit_astigm(
    x,
    amp=y.ptp() * np.array([0.7, 1.3, 1]),
    l=[0.1, len(y), 4],
    # x0=len(y) * np.array([0, 1, 0.5]),
    x0=[x.min(),x.max(),(x.min()+x.max())/2],
    FWHM=[0.1, 10, 2],
    offset=np.nanmin(y) * np.array([0.5, 3, 1]),
    astigm=[0.1, 30, 2],
    Smearing=[-5, 5, 0.8],
):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np
    from scipy.sparse import dia_matrix
    from scipy import signal

    l /= 2
    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM / 2.35) ** 2))
    function = amp * (a + b) / (a + b).ptp()  # +1#4 * l
    astigm = int(astigm)
    if astigm > 1:
        triang = signal.triang(astigm) / signal.triang(astigm).sum()
        function = np.convolve(function, triang, mode="same")

    # function = np.vstack((function,function)).T
    smearing_kernels = variable_smearing_kernels(
        function, Smearing, SmearExpDecrement=50000
    )
    n = smearing_kernels.shape[0]
    # print(smearing_kernels.sum(axis=1))
    # print(smearing_kernels.sum(axis=1))
    A = dia_matrix(
        (smearing_kernels.reshape((n, -1)), np.arange(n)),
        shape=(function.size, function.size),
    )
    function = A.dot(function.ravel()).reshape(function.shape)
    # function = np.mean(function,axis=1)

    return function + offset





def EMCCD_no_smearing(
    x,
    bias=[x.min(), x.max(), x[np.argmax(y)]],
    RN=[1, 350, 12],
    EmGain=[10, 2000, 1900],
    flux=[0, 1, 0.01],
    # smearing=[0, 3, 0.01],
    sCIC=[0, 2, 0],
):
    """EMCCD model based on convolution of distributions: Gamma(poison)xNormal
    First attempt to add smearing
    RN : Read noise in e-/pix
    EmGain : amplificaiton gain in e-/e-
    flux : incoming charges in e-/pix
    smearing : exponential length in pixel of the charge decay due to poor CTE
    sCIC : fraction of semi-amplified spurious charges that appear in the amplifier register
    """
    from astropy.convolution import Gaussian1DKernel, convolve
    import scipy.special as sps
    import numpy as np
    from scipy.stats import poisson
    from scipy.special import hyp0f1
    from scipy.interpolate import interp1d

    # recover number of pixels to generate distributions
    try:
        y = globals()["y"]
        n_pix = np.sum(10 ** y)
    except TypeError:
        n_pix = 10 ** 6.3
    n_registers = 604  # number of amplification registers
    ConversionGain = 1 
    bin_size = np.median((x[1:] - x[:-1]))
    bins = x - np.nanmin(x)
    distributions = []
    gamma_distribution = flux*np.exp(-flux)/EmGain * np.exp(- bins/EmGain)  *hyp0f1(2,bins*flux/EmGain) *bin_size
    gamma_distribution[0] = np.exp(-flux)

    # adding sCIC and comvolving distributions as independant draws.
    if sCIC>0:
        gain_ = np.power(EmGain *np.sqrt(2)  , np.linspace(1, n_registers, 100) / n_registers)
        cic_disdribution = np.sum([(1 / gaini) * np.exp(-bins / gaini) for gaini in gain_], axis=0)  # *n_pix/len(gain_)
        cic_disdribution *= np.exp(-flux)*(1-np.exp(-sCIC)) / cic_disdribution.sum() / 2
        gamma_distribution[1:]  += cic_disdribution[1:]
        gamma_distribution[0] -= np.sum(cic_disdribution[1:]) 


    if (bias > x[0]) & (bias!=0):
        gamma_distribution[(x > bias)] = gamma_distribution[: -np.sum(x <= bias)]
        gamma_distribution[x < bias] = 0


    read_noise = Gaussian1DKernel(
        stddev=RN * ConversionGain / bin_size, x_size=int(301.1 * 10)
    )
    # Convolution with read noise
    if RN>0:
        y = convolve(gamma_distribution, read_noise) * n_pix  #
    else:
        y = gamma_distribution #* n_pix

    # y[y<=0] =np.nan
    # return y#np.log10(y) 
    interpolated_function = interp1d(x[x<np.nanmax(x)-3*RN],np.log10(y[x<np.nanmax(x)-3*RN]), kind='linear',fill_value="extrapolate",bounds_error=False)

    return interpolated_function(x)

    

def EMCCD(
    x,
    bias=[x.min(), x.max(), x[np.argmax(y)]],
    RN=[1, 350, 12],
    EmGain=[10, 2000, 1900],
    flux=[0, 1, 0.01],
    smearing=[0, 3, 0.01],
    sCIC=[0, 2, 0],
):
    """EMCCD model based on convolution of distributions: Gamma(poison)xNormal
    First attempt to add smearing
    RN : Read noise in e-/pix
    EmGain : amplificaiton gain in e-/e-
    flux : incoming charges in e-/pix
    smearing : exponential length in pixel of the charge decay due to poor CTE
    sCIC : fraction of semi-amplified spurious charges that appear in the amplifier register
    """
    from astropy.convolution import Gaussian1DKernel, convolve
    import scipy.special as sps
    import numpy as np
    from scipy.stats import poisson,gamma
    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    try:
        _, y, _ = np.loadtxt("/tmp/xy.txt").T
        # x, y =  np.array(x), np.array(y)
    except (OSError, ValueError) as e:
        try:
            _, y = np.loadtxt("/tmp/xy.txt").T

        except (OSError, ValueError) as e:
            _, y = np.array([0, 1]), np.array([0, 1])

    # recover number of pixels to generate distributions
    try:
        # y = globals()["y"]
        n_pix = np.sum(10 ** y)
    except TypeError:
        n_pix = 10 ** 6.3
    # print(np.log10(n_pix))
    minx = np.nanmin(x) 
    x -=  minx


    n_registers = 604  # number of amplification registers
    ConversionGain = 1 
    bin_size = np.median((x[1:] - x[:-1]))

    # energy_fraction_kept = (1-np.exp(-1/smearing)*(np.exp(-1/smearing)+np.exp(-2/smearing)+np.exp(-3/smearing))) if smearing>0 else 1

    val=np.nan#1e-500
    normalization = np.sum(np.exp(-np.arange(1,100)/smearing))
    energy_fraction_kept = 1 - np.sum(np.exp(-np.arange(1,6)/smearing))/normalization  if smearing>0 else 1  # -np.exp(-2/smearing)-np.exp(-3/smearing)
    # energy_fraction_kept = (1-np.exp(-flux/smearing)*(np.exp(-1/smearing)+np.exp(-2/smearing)+np.exp(-3/smearing))) if smearing>0 else 1
    # gamma_distribution = EMCCD_no_smearing(x=x,bias=0,RN=0,EmGain=EmGain, flux=flux,sCIC=sCIC)
    y= EMCCD_no_smearing(x=x,bias=0,RN=0,EmGain=EmGain, flux=flux,sCIC=sCIC)
    n = np.sum(10**y)
    first_function = interp1d(x*energy_fraction_kept,y, kind='linear',fill_value="extrapolate",bounds_error=False)#
    gamma_distribution = 10**first_function(x)/energy_fraction_kept
    gamma_distribution[0] = np.exp(-flux)
#     plt.figure()
    # plt.semilogy(x,gamma_distribution,label="0")
    if smearing>0. :
        for i in np.arange(1,6):
            # normalization=1
            # interpolated_function = interp1d(x*np.exp(-i/smearing),EMCCD_no_smearing(x=x,bias=0,RN=0,EmGain=EmGain, flux=flux,sCIC=sCIC)/np.exp(-i/smearing), kind='linear',fill_value=val,bounds_error=False)# * np.exp(-i*flux)
        # if np.exp(-i/smearing)>0.00001:
            # x2 = np.array(list(x) + [1e5])#np.arange(0,np.nanmax(x)/np.exp(-i/smearing), x[1]-x[0])
            x2=x
            interpolated_function = interp1d(x2*np.exp(-i/smearing)/normalization,EMCCD_no_smearing(x=x2,bias=0,RN=0,EmGain=EmGain, flux=flux,sCIC=sCIC), kind='linear',fill_value="extrapolate",bounds_error=False)#   #
            smeared_distri = 10**interpolated_function(x)/(np.exp(-i/smearing)/normalization)* np.exp(-i*flux) # doit diminuer car on ajoute trop
            # smeared_distri[smeared_distri==1]=1e500
            gamma_distribution[1:] = np.nansum([gamma_distribution[1:],smeared_distri[1:]],axis=0)
            gamma_distribution[0] -= np.nansum(smeared_distri[1:])
    # gamma_distribution[0] = np.exp(-flux)
    gamma_distribution[1:] = n *  gamma_distribution[1:]/np.sum(gamma_distribution)

#     plt.legend()
#     plt.show()
    # Addition of the bias
    x +=  minx
    if bias > x[0]:
        gamma_distribution[(x > bias)] = gamma_distribution[: -np.sum(x <= bias)]
        gamma_distribution[x < bias] = 0
    if RN>0:
        read_noise = Gaussian1DKernel(stddev=RN * ConversionGain / bin_size, x_size=int(301.1 * 10))
        y = convolve(gamma_distribution, read_noise) * n_pix  #
    else:
        y = gamma_distribution #* n_pix
    # y = np.convolve(gamma_distribution, Gaussian1DKernel(stddev=RN * ConversionGain / bin_size, x_size=len(gamma_distribution)), 'same') * n_pix  #

    # y /= x[1] - x[0]
    # y[y<1]=1
    # return np.log10(y)
    interpolated_function = interp1d(x[x<np.nanmax(x)-3*RN],np.log10(y[x<np.nanmax(x)-3*RN]), kind='linear',fill_value="extrapolate",bounds_error=False)

    return interpolated_function(x)



# def EMCCD_dev(
#     x,
#     bias=[x.min(), x.max(), x[np.argmax(y)]],
#     RN=[5, 350, 12],
#     EmGain=[10, 2000, 1900],
#     flux=[0, 1, 0.01],
#     smearing=[0, 3, 0.01],
#     sCIC=[0, 2, 0],
#     test=True
# ):
#     """EMCCD model based on convolution of distributions: Gamma(poison)xNormal
#     First attempt to add smearing
#     RN : Read noise in e-/pix
#     EmGain : amplificaiton gain in e-/e-
#     flux : incoming charges in e-/pix
#     smearing : exponential length in pixel of the charge decay due to poor CTE
#     sCIC : fraction of semi-amplified spurious charges that appear in the amplifier register
#     """
#     from astropy.convolution import Gaussian1DKernel, convolve
#     import scipy.special as sps
#     import numpy as np
#     from scipy.stats import poisson
#     from scipy.stats import poisson,gamma

#     # recover number of pixels to generate distributions
#     try:
#         y = globals()["y"]
#         n_pix = np.sum(10 ** y)
#     except TypeError:
#         n_pix = 10 ** 6.3
#     ConversionGain = 1 
#     bin_size = np.median((x[1:] - x[:-1]))

#     energy_fraction_kept = (1-np.exp(-flux/smearing)*(np.exp(-1/smearing)+np.exp(-2/smearing)+np.exp(-3/smearing))) if smearing>0 else 1
   
#     gamma_distribution =  EMCCD_no_smearing(x=x,bias=0,RN=0,EmGain=EmGain*energy_fraction_kept, flux=flux,sCIC=sCIC)

#     if (smearing>0.0)  :
#         for i in range(1,4):
#             smeared_distri = EMCCD_no_smearing(x=x,bias=0,RN=0,EmGain=EmGain*np.exp(-i/smearing), flux=flux,sCIC=0)   * np.exp(-i*flux) #/smearing
#             gamma_distribution[1:] +=  smeared_distri[1:]
#             gamma_distribution[0] -= np.sum(smeared_distri[1:])



#     # Addition of the bias
#     if bias > x[0]:
#         gamma_distribution[(x > bias)] = gamma_distribution[: -np.sum(x <= bias)]
#         gamma_distribution[x < bias] = 0
#     read_noise = Gaussian1DKernel(
#         stddev=RN * ConversionGain / bin_size, x_size=int(301.1 * 10)
#     )
#     # Convolution with read noise
#     y = convolve(gamma_distribution, read_noise) * n_pix  #
#     y[y<=0] = np.nan
#     return np.log10(y)



