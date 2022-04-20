# Function used to fit some DS9 plots
# Please be sure to define a list as default parameters for each arguments
# as they will be used to define the lower and upper bounds of each widget.
import matplotlib.pyplot as plt
import numpy as np

try:
    x, y = np.loadtxt("/tmp/xy.txt").T
    # print(np.log10(np.sum(10**y)))
except OSError:
    x, y = [0, 1], [0, 1]


def slit(x, amp=y.ptp() * np.array([0.7,1.3,1]), l=len(y) * np.array([0,1,0.2]), x0=len(y) * np.array([0,1,0.5]), FWHM=[0.1,10,2], offset=np.nanmin(y)*np.array([0.5,3,1])):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np

    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    function = amp * (a + b) / (a + b).ptp()#4 * l
    return  function + offset


def gaussian(x, a=[0, 100], xo=[0, 100], sigma=[0, 10]):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = a * np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
    return g.ravel()


def defocused_gaussian(
    x, amp=[0, 1000], amp_2=[0, 1000], xo=[0, 100], sigma=[0, 10], sigma_2=[0, 10],
):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = amp * np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
    g2 = amp_2 * np.exp(-0.5 * (np.square(x - xo) / sigma_2 ** 2))
    return (g - g2).ravel()


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


def EMCCD(
    x,
    bias=[1000, 4000, x[np.argmax(y)]],
    RN=[20, 150, 44],
    EmGain=[300, 10000, 1900],
    flux=[0, 0.2, 0.01],
    smearing=[0, 1, 0.01],
    sCIC=[0, 1, 0],
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

    # recover number of pixels to generate distributions
    try:
        y = globals()["y"]
        n_pix = np.sum(10 ** y)
    except TypeError:
        n_pix = 10 ** 6.3
    n_registers = 604  # number of amplification registers
    if bias > 1500:  # 2018
        ConversionGain = 0.53  # Conversiongain in ADU/e-
    else:
        ConversionGain = 1 / 4.5  # 2022
    bin_size = np.median((x[1:] - x[:-1]))
    bins = x - np.nanmin(x)
    distributions = []
    pixs_sup_0 = 0  # fraction of pixels that have ADU higher than distribution limit
    n = np.arange(2) if smearing > 0 else [0]
    fluxes = flux * np.power(smearing, n) / np.sum(np.power(smearing, n))
    emgains = [EmGain] + [EmGain / (1.7 * i) for i in n[1:]]
    # for smearing we consider each pixel recieve x% of their flux and Then
    # they receive the rest with a lower gain. This approximation is # NOTE:
    # true and need to be revised. The interest is that it is then independant
    # draws and that distributions can be convoluted.
    for i, (flux_, EmGain) in enumerate(zip(fluxes, emgains)):
        gamma_distribution = 0
        # We sum up distributions from from the different poisson output
        for f in np.arange(
            1, 20
        ):  # careful about the 20 limit which needs to be enough at high flux
            v = poisson.pmf(k=f, mu=np.nanmax([flux_, 0]))
            # gamma distribution : https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
            denominator = sps.gamma(f) * (EmGain * ConversionGain) ** f
            distribution = (
                bin_size
                * bins ** (f - 1)
                * (np.exp(-bins / (EmGain * ConversionGain)) / denominator)
            )
            distribution_up = (
                3
                * bin_size
                * (3 * bins + bins.ptp()) ** (f - 1)
                * (
                    np.exp(-(3 * bins + bins.ptp()) / (EmGain * ConversionGain))
                    / denominator
                )
            )
            # disminush the number of pixels by the fraction above distribution range
            factor = np.sum(distribution[np.isfinite(distribution_up)]) / (
                np.sum(distribution_up[np.isfinite(distribution_up)])
                + np.sum(distribution[np.isfinite(distribution)])
            )
            gamma_distribution += distribution * v
            pixs_sup_0 += (1 - factor) * v
        gamma_distribution[0] = (
            1
            - np.nansum(gamma_distribution[1:][np.isfinite(gamma_distribution[1:])])
            - pixs_sup_0
        )
        # adding sCIC and comvolving distributions as independant draws.
        if sCIC > 0:
            # changing total sCIC (e-) into the percentage of pixels experiencing spurious electrons
            p_sCIC = sCIC  # / np.mean(1 / np.power(EmGain * ConversionGain, np.arange(604) / 604))
            # Estimation of average gain for semi-amplified CIC
            gain_ = np.power(
                EmGain * ConversionGain, np.linspace(1, n_registers, 100) / n_registers
            )
            cic_disdribution = np.sum(
                [(1 / gaini) * np.exp(-bins / gaini) for gaini in gain_], axis=0
            )  # *n_pix/len(gain_)
            cic_disdribution /= cic_disdribution.sum()
            cic_disdribution *= p_sCIC
            cic_disdribution[0] = 1 - np.sum(cic_disdribution[1:])
            cic_disdribution = np.hstack(
                (np.zeros(len(cic_disdribution) - 1), cic_disdribution)
            )
            gamma_distribution = np.convolve(
                gamma_distribution, cic_disdribution, mode="valid"
            )
        distributions.append(gamma_distribution)
        if i > 0:
            smeared_distribution = np.hstack(
                (np.zeros(len(distributions[i]) - 1), distributions[i])
            )
            gamma_distribution = np.convolve(
                distributions[i - 1], smeared_distribution, mode="valid"
            )
            distributions[i] = gamma_distribution

    # Addition of the bias
    if bias > x[0]:
        gamma_distribution[(x > bias)] = gamma_distribution[: -np.sum(x <= bias)]
        gamma_distribution[x < bias] = 0
    read_noise = Gaussian1DKernel(
        stddev=RN * ConversionGain / bin_size, x_size=int(301.1 * 10)
    )
    # Convolution with read noise
    y = convolve(gamma_distribution, read_noise) * n_pix  #
    y /= x[1] - x[0]
    return np.log10(y)

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

def EMCCDhist(
    x,
    bias=[1e3, 4.5e3, 1194],
    RN=[0, 200, 53],
    EmGain=[100, 10000, 5000],
    flux=[0.001, 1, 0.04],
    smearing=[0, 1, 0.01],
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
    from matplotlib.widgets import Button
    import numpy as np

    if bias > 1500:
        ConversionGain = 0.53  # 1/4.5 #ADU/e-  0.53 in 2018
    else:
        ConversionGain = 1 / 4.5  # ADU/e-  0.53 in 2018


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

        # recover number of pixels to generate distributions
        try:
            y = globals()["y"]
            n_pix = np.nansum(10 ** y[np.isfinite(10 ** y)])  # 1e6#
        except TypeError:
            n_pix = 10 ** 6.3
        n = 1
        im = np.zeros(int(n_pix))  #
        im = np.zeros((1000, int(n_pix / 1000)))
        imaADU = np.random.gamma(
            np.random.poisson(np.nanmax([flux, 0]), size=im.shape), abs(EmGain)
        )
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
        imaADU += np.random.gamma(
            np.random.poisson(1 * id_scic), np.power(EmGain, register / n_registers)
        )
        # imaADU[id_scic] += np.random.gamma(1, np.power(EmGain, register / n_registers))
        imaADU *= ConversionGain
        # smearing data
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
        # adding read noise and bias
        read_noise = np.random.normal(0, abs(RN * ConversionGain), size=im.shape)
        imaADU += Bias
        imaADU += read_noise
        range = [np.nanmin(x), np.nanmax(x)]
        n, bins = np.histogram(imaADU.flatten(), bins=[x[0] - 1] + list(x))
        return n

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
    y[y == 0] = 1.0
    y = y / (x[1] - x[0])
    return np.log10(y)


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
            -np.arange(n)[::int(np.sign(smearing_length[0])), np.newaxis, np.newaxis] / abs(smearing_length)
        )
    else:
        assert 0 <= Smearing <= 1
        smearing_kernels = np.power(Smearing, np.arange(n))[
            :, np.newaxis, np.newaxis
        ] / np.ones(smearing_length.shape)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels



def smeared_slit(x, amp=y.ptp() * np.array([0.7,1.3,1]), l=[0,len(y),4], x0=len(y) * np.array([0,1,0.5]), FWHM=[0.1,10,2], offset=np.nanmin(y)*np.array([0.5,3,1]),Smearing=[-2,2,0.8]):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np
    from scipy.sparse import dia_matrix


    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    function = amp * (a + b) / (a + b).ptp()#+1#4 * l
    # function = np.vstack((function,function)).T
    smearing_kernels = variable_smearing_kernels(
        function, Smearing, SmearExpDecrement=50000)
    n = smearing_kernels.shape[0]
    # print(smearing_kernels.sum(axis=1))
    # print(smearing_kernels.sum(axis=1))
    A = dia_matrix(
        (smearing_kernels.reshape((n, -1)), np.arange(n)),
        shape=(function.size, function.size),
    )
    function = A.dot(function.ravel()).reshape(function.shape)
    # function = np.mean(function,axis=1)
    return  function + offset
