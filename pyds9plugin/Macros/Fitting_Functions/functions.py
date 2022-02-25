# Function used to fit some DS9 plots
# Please be sure to define a list as default parameters for each arguments
# as they will be used to define the lower and upper bounds of each widget.

import numpy as np
try:
    x,y = np.loadtxt("/tmp/xy.txt").T
except OSError:
    x,y = [0,1],[0,1]


def gaussian(x, a=[0, 100], xo=[0, 100], sigma=[0, 10]):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = a * np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
    return g.ravel()


def defocused_gaussian(
    x,
    amp=[0, 1000],
    amp_2=[0, 1000],
    xo=[0, 100],
    sigma=[0, 10],
    sigma_2=[0, 10],
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
    x,
    phi=[1e-3, 1e-2],
    alpha=[-2, -1],
    M=[16, 22],
    phi2=[1e-3, 1e-2],
    alpha2=[-2, -1],
):
    return np.log10(
        10 ** schechter(x, phi, M, alpha)
        + +(10 ** schechter(x, phi2, M, alpha2))
    )


def EMCCD(
    x,
    bias=[1000,4000,x[np.argmax(y)]],
    RN=[20,150,44],#107
    EmGain=[300,10000,1900],#600
    flux=[0,0.2,0.01],
    # bright_surf=[5,9,8.3],#8.3
    smearing=[0,1.5,0],#0.7
    p_sCIC=[0,0.1,0],#0
):
    """EMCCD model based on convolution of distributions: Gamma(poison)xNormal
    TODO: add smearing and CIC!
    TODO and remove too high value at 0
    """
    from astropy.convolution import Gaussian1DKernel, convolve
    import scipy.special as sps
    import numpy as np
    try:
        bright_surf = np.log10(np.sum(10**globals()['y']))#6.3
    except TypeError:
        bright_surf=6.3
    n_registers = 604
    ConversionGain = 1/4.5# ADU/e-  0.53  
    #    ycounts = x**(shape-1)*(np.exp(-x/EmGain) /(sps.gamma(shape)*EmGain**shape))
    bin_size = np.median((x[1:] - x[:-1]))
    ycounts = (bin_size*
        (x - np.nanmin(x) + 0) ** (flux - 1)
        * (np.exp(-(x - np.nanmin(x)) / (EmGain*ConversionGain))
            / (sps.gamma(flux) * (EmGain*ConversionGain) ** flux)
        )
    )
    ycounts[0] = 1 - np.nansum(ycounts[np.isfinite(ycounts)])
    yscic = [
        np.exp(-x / np.power(EmGain, register / n_registers))
        / np.power(EmGain, register / n_registers)
        for register in np.arange(n_registers)
    ]
    yscic = np.sum(yscic, axis=0)
    if bias > x[0]:
        ycounts[(x > bias)] = ycounts[: -np.sum(x <= bias)]
        ycounts[x < bias] = 0
    #    yscic[x>biais] = yscic[:-np.sum(x<=biais)]
    #    yscic[x<biais] = 0
    # y = ycounts
    n = 1
    kernel = Gaussian1DKernel(
        stddev= RN * ConversionGain / bin_size, x_size=int(301.1 * 10 ** n)
    )

    return np.log10(convolve(ycounts, kernel)*    10**bright_surf)  #+ np.log10(np.sum([10 ** bright_surf]))


# 
# Bias = x[np.argmax(y)];RN=44;EmGain=1900;flux=0.05;smearing=0;sCIC=0
#x[np.argmax(y)]
def EMCCDhist(x, bias=[1e3, 4.5e3,1194], RN=[0,200,53], EmGain=[100, 10000,5000], flux=[0.001, 1,0.04], smearing=[0, 3,0.31], sCIC=[0,1,0]):
    from scipy.sparse import dia_matrix
    import inspect
    from astropy.table import Table
    from matplotlib.widgets import Button
    import numpy as np  

    def variable_smearing_kernels(image, Smearing=1.5, SmearExpDecrement=50000):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np
    
        smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
        smearing_kernels = np.exp(
            -np.arange(6)[:, np.newaxis, np.newaxis] / smearing_length
        )
        smearing_kernels /= smearing_kernels.sum(axis=0)
        return smearing_kernels


    def simulate_fireball_emccd_hist(
        x,
        ConversionGain,
        EmGain,
        Bias,
        RN,
        p_pCIC,
        p_sCIC,
        Smearing,
        SmearExpDecrement,
        n_registers,
        flux,
        sCIC=0,
    ):
        """Silumate EMCCD histogram
        """
        import numpy as np
        im= np.zeros((1000,2000))
        imaADU = np.random.gamma(flux, EmGain, size=im.shape)

        # prob_pCIC = np.random.rand(size[1],size[0])    #Draw a number prob in [0,1]
        # image[prob_pCIC <  p_pCIC] += 1
        # np.ones((im.shape[0], im.shape[1]))
        prob_sCIC = np.random.rand(im.shape[0], im.shape[1])
        # Draw a number prob in [0,1]
        id_scic = prob_sCIC < sCIC  # sCIC positions
        # partial amplification of sCIC
        register = np.random.randint(1, n_registers, size=id_scic.sum())
        # Draw at which stage of the EM register the electorn is created
        imaADU[id_scic] += np.random.exponential(np.power(EmGain, register / n_registers))
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
            # print(imaADU==A.dot(imaADU.ravel()).reshape(imaADU.shape))
            imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
        read_noise = np.random.normal(0, RN * ConversionGain, size=im.shape)        
        imaADU += Bias
        imaADU += read_noise
        range = [np.nanmin(x), np.nanmax(x)]
        n, bins = np.histogram(imaADU.flatten(), range=range, bins=len(x))
        return n
        # fig, ax1=plt.subplots(1,1,sharex=True)
        # sig,_,_= ax1.hist(imaADU.flatten(),log=False,bins=10000,range=(0,1e4),histtype='step')
        # noise,b,_=ax1.hist(read_noise.flatten(),log=False,bins=10000,range=(0,1e4),histtype='step')
        # th = 5.5*RN* ConversionGain
        # ax1.plot([th,th],[0,1e5])
        # bins = (b[1:]+b[:-1])/2
        # ax1.set_title('fraction kept = %0.1f'%(np.sum(sig[bins>th])/np.sum(sig)))
        # ax1.set_xlim((0,500))
        # ax1.set_ylim((0,1e5))
        # ax2.plot(bins,np.array([np.sum(sig[bins>xi]) for xi in bins])/np.array([np.sum(noise[bins>xi]) for xi in bins]))
        # ax2.plot(bins,np.array([np.sum(sig[bins>xi]) for xi in bins])/noise)
    y = simulate_fireball_emccd_hist(
        x=x,
        ConversionGain=1 / 4.5,  # 0.53,
        EmGain=EmGain,
        Bias=bias,
        RN=RN,
        p_pCIC=0,
        p_sCIC=0,   
        Smearing=smearing,
        SmearExpDecrement=1e4,
        n_registers=604,
        flux=flux,
        sCIC=sCIC)
    y[y==0]=1
    return np.log10(y)






