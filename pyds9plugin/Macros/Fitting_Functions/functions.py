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
    RN=[20,150,44],
    EmGain=[300,10000,1900],
    flux=[0,0.2,0.01],
    smearing=[0,1,0.01],
    sCIC=[0,1,0],
):
    """EMCCD model based on convolution of distributions: Gamma(poison)xNormal
    TODO: add smearing!
    """
    from astropy.convolution import Gaussian1DKernel, convolve
    import scipy.special as sps
    import numpy as np
    from scipy.stats import poisson

    try:
        y = globals()['y']
        n_pix = np.sum(10**y)
        # print('global',np.sum(10**y),n_pix)
    except TypeError:
        n_pix=10**6.3   
        # print('fixed',np.sum(10**y),n_pix)
    n_registers = 604
    if bias>1500:
        ConversionGain = 0.53#1/4.5 #ADU/e-  0.53 in 2018 
    else: 
        ConversionGain = 1/4.5 #ADU/e-  0.53 in 2018 
    bin_size = np.median((x[1:] - x[:-1]))
    bins = x - np.nanmin(x)
    # factor = 1#np.log(2)
    # gamma_distribution = bins**(shape-1)*(np.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
    

    # denominator = (sps.gamma(flux) * (EmGain*ConversionGain) ** flux)
    # gamma_distribution = (bin_size* bins ** (flux - 1) * (np.exp(-bins / (EmGain*ConversionGain)) / denominator))

    distributions=[]
    n=np.arange(10)
    fluxes = flux *np.power(smearing,n)/ np.sum(np.power(smearing,n)) #[flux*smearing**i for i in range(n)] / np.sum([flux*smearing**i for i in range(n)])
    emgains = [EmGain] + [EmGain/(1.7*i) for i in n[1:]]
    for i,(flux_,EmGain) in enumerate(zip(fluxes,emgains)):
    # for i,(flux,EmGain) in enumerate(zip([(1-smearing)*flux, smearing*flux],[EmGain,EmGain/2])):
        gamma_distribution = 0
        # gamma_distribution_up = 0
        for f in np.arange(1,10) :
            v = poisson.pmf(k=f, mu=np.max([flux_,0]))
            denominator = (sps.gamma(f) * (EmGain*ConversionGain) ** f)
            # print(1, denominator)
            # denominator[~np.isfinite(denominator)]=1
            # print(bin_size* bins ** (f - 1) * (np.exp(-bins / (EmGain*ConversionGain))))
            distribution = (bin_size* bins ** (f - 1) * (np.exp(-bins / (EmGain*ConversionGain)) / denominator))
            # distribution_up = (bin_size* (bins+bins.ptp()) ** (f - 1) * (np.exp(-(bins+bins.ptp()) / (EmGain*ConversionGain)) / denominator))
            # disminush the number of pixels by the number of ones that are above the threshold:
            # factor = np.sum(distribution[np.isfinite(distribution)]) / np.sum(distribution_up[np.isfinite(distribution_up)])
            #no need for the factor, just recale here
            distribution[0] = (1 - np.nansum(distribution[np.isfinite(distribution)]))
            gamma_distribution += distribution*v


        gamma_distribution[0] = (1 - np.nansum(gamma_distribution[np.isfinite(gamma_distribution)]))
        if sCIC>0:
            #changing total sCIC (e-) into the percentage of pixels experiencing spurious electrons
            p_sCIC =  sCIC/4#/ np.mean(1/ np.power(EmGain * ConversionGain, np.arange(604) / 604))
            gain_ = np.power(EmGain*ConversionGain, np.linspace(1,n_registers,100) / n_registers)
            cic_disdribution = np.sum([(1/gaini) * np.exp(-bins/gaini) for gaini in gain_],axis=0)#*n_pix/len(gain_)
            cic_disdribution /= cic_disdribution.sum()

            cic_disdribution *= p_sCIC
            cic_disdribution[0] = 1 -np.sum(cic_disdribution[1:])
            cic_disdribution = np.hstack((np.zeros(len(cic_disdribution)-1),cic_disdribution))
            gamma_distribution = np.convolve(gamma_distribution,cic_disdribution,mode='valid')
        distributions.append(gamma_distribution)
        if i>0:
            smeared_distribution = np.hstack((np.zeros(len(distributions[i])-1),distributions[i]))
            gamma_distribution  = np.convolve(distributions[i-1],smeared_distribution,mode='valid')
            distributions[i] = gamma_distribution
    # gamma_distribution = 0
    # for f in np.arange(1,10) :
    #     v = poisson.pmf(k=f, mu=np.max([flux,0]))
    #     denominator = (sps.gamma(f) * (EmGain*ConversionGain) ** f)
    #     # print(1, denominator)
    #     # denominator[~np.isfinite(denominator)]=1
    #     # print(bin_size* bins ** (f - 1) * (np.exp(-bins / (EmGain*ConversionGain))))
    #     distribution = (bin_size* bins ** (f - 1) * (np.exp(-bins / (EmGain*ConversionGain)) / denominator))
    #     gamma_distribution += distribution*v
    # gamma_distribution[0] = 1 - np.nansum(gamma_distribution[np.isfinite(gamma_distribution)])
    # if sCIC>0:
    #     #changing total sCIC (e-) into the percentage of pixels experiencing spurious electrons
    #     p_sCIC =  sCIC/ np.mean([(1/ np.power(EmGain * ConversionGain, reg / 604))  for reg in np.arange(604)])
    #     gain_ = np.power(EmGain*ConversionGain, np.linspace(1,n_registers,100) / n_registers)
    #     cic_disdribution = n_pix*np.sum([(1/gaini) * np.exp(-bins/gaini) for gaini in gain_],axis=0)/len(gain_)
    #     cic_disdribution /= cic_disdribution.sum()

    #     cic_disdribution *= p_sCIC
    #     cic_disdribution[0] = 1 -np.sum(cic_disdribution[1:])
    #     cic_disdribution = np.hstack((np.zeros(len(cic_disdribution)-1),cic_disdribution))
    #     gamma_distribution = np.convolve(gamma_distribution,cic_disdribution,mode='valid')
    #Renormalization of the gamma distribution to one #I might doing it wrong here!
    # print(factor)
    # gamma_distribution[0]*/(1+factor)

    # Addition of the bias
    if bias > x[0]:
        gamma_distribution[(x > bias)] = gamma_distribution[: -np.sum(x <= bias)]
        gamma_distribution[x < bias] = 0
    read_noise = Gaussian1DKernel(stddev= RN * ConversionGain / bin_size, x_size=int(301.1*10))
    y = convolve(gamma_distribution, read_noise)*n_pix#
    return np.log10(y) 


def EMCCDhist(x, bias=[1e3, 4.5e3,1194], RN=[0,200,53], EmGain=[100, 10000,5000], flux=[0.001, 1,0.04], smearing=[0, 1,0.01], sCIC=[0,1,0]):
# def EMCCDhist(x, bias=[1e3, 4.5e3,1194], RN=[0,200,53], EmGain=[100, 10000,5000], flux=[0.001, 1,0.04], smearing=[0, 3,0.31], sCIC=[0,1,0],SmearExpDecrement=[1.5e3,1.5e5,15e4]):
    from scipy.sparse import dia_matrix
    import inspect
    from astropy.table import Table
    from matplotlib.widgets import Button
    import numpy as np  
    if bias>1500:
        ConversionGain = 0.53#1/4.5 #ADU/e-  0.53 in 2018 
    else: 
        ConversionGain = 1/4.5 #ADU/e-  0.53 in 2018 

    # def variable_smearing_kernels(image, Smearing=1.5, SmearExpDecrement=50000):
    #     """Creates variable smearing kernels for inversion
    #     """
    #     import numpy as np
    #     n=6
    #     smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    #     smearing_kernels = np.exp(-np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length)
    #     smearing_kernels /= smearing_kernels.sum(axis=0)
    #     return smearing_kernels
    def variable_smearing_kernels(image, Smearing=0.7, SmearExpDecrement=50000,type_='expw'):
        """Creates variable smearing kernels for inversion
        """
        import numpy as np
        n=15
        smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
        if type_=='exp':       
            smearing_kernels = np.exp(-np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length)
        else:
            assert 0<=Smearing<=1
            smearing_kernels = np.power(Smearing,np.arange(n))[:, np.newaxis, np.newaxis] / np.ones(smearing_length.shape)
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
        try:
            y = globals()['y']
            n_pix = np.sum(10**y)
            # print('global',np.sum(10**y),n_pix)
        except TypeError:
            n_pix=10**6.3   
            # print('fixed',np.sum(10**y),n_pix)
        n=1
        im = np.zeros((1000,int(n_pix/1000)+1))
        # factor = 1#np.log(2)
        # EmGain *= factor
        # imaADU = np.random.gamma(flux, EmGain, size=im.shape)
        imaADU = np.random.gamma(np.random.poisson(np.max([flux,0]), size=im.shape), EmGain)
        #Add pCIC (no interest, as flux)
        # imaADU[np.random.rand(size[1],size[0]) <  p_pCIC] += 1 

        # pixels in which sCIC electron might appear
        id_scic = np.random.rand(im.shape[0], im.shape[1]) < sCIC  # sCIC positions
        # stage of the EM register at which each sCIC e- appear
        register = np.random.randint(1, n_registers, size=id_scic.sum())
        # Compute and add the partial amplification for each sCIC pixel
        imaADU[id_scic] += np.random.exponential(np.power(EmGain, register / n_registers))
        imaADU *= ConversionGain

        if Smearing > 0:
            smearing_kernels = variable_smearing_kernels(imaADU, Smearing, SmearExpDecrement)
            offsets = np.arange(6)
            A = dia_matrix((smearing_kernels.reshape((6, -1)), offsets),shape=(imaADU.size, imaADU.size),)
            imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
        read_noise = np.random.normal(0, RN * ConversionGain, size=im.shape)        
        imaADU += Bias
        imaADU += read_noise
        range = [np.nanmin(x), np.nanmax(x)]
        n, bins = np.histogram(imaADU.flatten(), range=range, bins=len(x))
        return n



    y = simulate_fireball_emccd_hist(
        x=x,
        ConversionGain=ConversionGain,  # 0.53,
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






