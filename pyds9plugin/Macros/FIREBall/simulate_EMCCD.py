import numpy as np
import sys, os
from astropy.io import fits
import matplotlib.pyplot as plt
from decimal import Decimal
from astropy.table import Table
from pyds9plugin.DS9Utils import verboseprint, DS9n


def ConvolveSlit2D_PSF(xy=np.meshgrid(np.linspace(-20, 20, 100), np.linspace(-20, 20, 100)), amp=1, l=3, L=20, xo=0, yo=0, sigmax2=20, sigmay2=20):
    """Convolve a 2D slit with a gaussina
    """
    from scipy import special

    x, y = xy
    A1 = special.erf((l - (x - xo)) / np.sqrt(2 * sigmax2))
    A2 = special.erf((l + (x - xo)) / np.sqrt(2 * sigmax2))
    B1 = special.erf((L - (y - yo)) / np.sqrt(2 * sigmay2))
    B2 = special.erf((L + (y - yo)) / np.sqrt(2 * sigmay2))
    function = amp * (1 / (16 * l * L)) * (A1 + A2) * (B1 + B2)
    return function.ravel()


def Smearing2Noise(exp_coeff=1.5):
    """
    """
    # try:
    #     noisePath = resource_filename("pyds9fb", "CSVs")
    # except:
    noisePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CSVs")
    coeff1 = np.loadtxt(os.path.join(noisePath, "GainHistVSsmearing.csv"), delimiter=",")
    coeff2 = np.loadtxt(os.path.join(noisePath, "GainVarVSsmearing.csv"), delimiter=",")
    x1, y1 = coeff1.T
    x2, y2 = coeff2.T
    Hist_smear = PlotFit1D(x1, y1, deg=6, plot_=False)["popt"]
    # y = [1/Smearing2Noise(a) for a in np.linspace(0,2,100)]
    # plot(np.linspace(0,2,100),y)
    #    n=100
    #    x = np.linspace(0,n,n+1)
    #    y = np.exp(-x/exp_coeff) *(1-np.exp(-1/exp_coeff))
    #    return np.sqrt(np.square(y).sum())**2
    Var_smear = PlotFit1D(x2, y2, deg=6, plot_=False)["popt"]
    return {"Hist_smear": 1 / Hist_smear(exp_coeff), "Var_smear": 1 / Var_smear(exp_coeff)}


def SimulateFIREBallemCCDHist(ConversionGain, EmGain, Bias, RN, p_pCIC, p_sCIC, Dark, Smearing, SmearExpDecrement, exposure, n_registers):
    # bins, ConversionGain, EmGain, Bias, RN, p_pCIC, p_sCIC, Dark, Smearing, SmearExpDecrement, exposure,  n_registersbins, ConversionGain, EmGain, Bias, RN, p_pCIC, p_sCIC, Dark, Smearing, SmearExpDecrement, exposure,  n_registers = args
    # ConversionGain=0.53, EmGain=1500, Bias=3000, RN=80, p_pCIC=1, p_sCIC=1, Dark=5e-4, Smearing=0.7, SmearExpDecrement=50000, exposure=50,  n_registers=604
    imaADU, imaADU_wo_RN, imaADU_RN = SimulateFIREBallemCCDImage(ConversionGain=ConversionGain, EmGain=EmGain, Bias=Bias, RN=RN, p_pCIC=p_pCIC, p_sCIC=p_sCIC, Dark=Dark, Smearing=Smearing, SmearExpDecrement=SmearExpDecrement, exposure=exposure, n_registers=n_registers, save=False)
    # n, bins = np.histogram(imaADU[:,1066:2124].flatten(),range=[0,2**16], bins = int(2**16/2**2))#, range=(-200,11800))
    return imaADU[:, 1066:2124]  # n#, (bins[:-1]+bins[1:])/2


def SimulateFIREBallemCCDImage(
    ConversionGain=0.53, EmGain=1500, Bias="Auto", RN=80, p_pCIC=1, p_sCIC=1, Dark=5e-4, Smearing=0.7, SmearExpDecrement=50000, exposure=50, flux=1e-3, source="Spectra", Rx=8, Ry=8, size=[3216, 2069], OSregions=[1066, 2124], name="Auto", spectra="-", cube="-", n_registers=604, save=True
):
    # print('conversion gain = ', ConversionGain)
    from astropy.modeling.functional_models import Gaussian2D
    from scipy.sparse import dia_matrix

    OS1, OS2 = OSregions
    if Bias == "Auto":
        if EmGain > 1:
            Bias = 3000 / ConversionGain
        else:
            Bias = 6000 / ConversionGain
    else:
        Bias = float(Bias) / ConversionGain
    image = np.zeros((size[1], size[0]), dtype="float32")

    # dark & flux
    source_im = 0 * image[:, OSregions[0] : OSregions[1]]
    lx, ly = source_im.shape
    y = np.linspace(0, lx - 1, lx)
    x = np.linspace(0, ly - 1, ly)
    x, y = np.meshgrid(x, y)

    # Source definition. For now the flux is not normalized at all, need to fix this
    # Cubes still needs to be implememted, link to detector model or putting it here?
    if os.path.isfile(cube):
        from .FitsCube import FitsCube

        source_im = FitsCube(filename=cube)
    elif source == "Flat-field":
        source_im += flux
    elif source == "Dirac":
        source_im += Gaussian2D.evaluate(x, y, 100 * flux, ly / 2, lx / 2, Ry, Rx, 0)
    elif source == "Spectra":
        source_im += Gaussian2D.evaluate(x, y, 100 * flux, ly / 2, lx / 2, 100 * Ry, Rx, 0)
    elif source == "Slit":
        ConvolveSlit2D_PSF_75muWidth = lambda xy, amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
        source_im += ConvolveSlit2D_PSF_75muWidth((x, y), 30000 * flux, 9, ly / 2, lx / 2, Rx, Ry).reshape(lx, ly)
    elif source == "Fibre":
        verboseprint("Create fibre source, FWHM: ", 2.353 * Rx, 2.353 * Ry)
        fibre = convolvePSF(radius_hole=10, fwhmsPSF=[2.353 * Rx, 2.353 * Ry], unit=1, size=(201, 201), Plot=False)  # [:,OSregions[0]:OSregions[1]]
        source_im = addAtPos(source_im, fibre, (int(lx / 2), int(ly / 2)))
        verboseprint("Done")
    elif source[:5] == "Field":
        ConvolveSlit2D_PSF_75muWidth = lambda xy, amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
        ws = [2025, 2062, 2139]
        # x, y = [], []
        for i, w in enumerate(ws):
            slits = returnXY(source[0].lower() + source[-1], w=w, frame="observedframe")
            xs = slits[0]
            ys = slits[1]
            index = (ys > OS1) & (ys < OS2)
            verboseprint(xs, ys)
            for yi, xi in zip(np.array(ys[index]) - OS1, xs[index]):
                verboseprint(xi, yi)
                source_im += ConvolveSlit2D_PSF_75muWidth((x, y), 40000 * flux, 9, yi, xi, Rx, Ry).reshape(lx, ly)

    # Poisson realisation
    source_im2 = np.random.poisson((Dark + source_im) * exposure)

    # Addition of the phyical image on the 2 overscan regions
    image[:, OSregions[0] : OSregions[1]] += source_im2

    if save:
        #        test = image.copy()
        #        test[:,OSregions[0]:OSregions[1]] += (Dark + source_im) * exposure
        # fitswrite(image.astype("int32"), name[:-5] + "_beforeAmp.fits")
        fits.HDUList(fits.PrimaryHDU(image.astype("int32"))).writeto(name[:-5] + "_beforeAmp.fits",overwrite=True)
    if EmGain > 1:

        # addition of pCIC (stil need to add sCIC before EM registers)
        prob_pCIC = np.random.rand(size[1], size[0])  # Draw a number prob in [0,1]
        image[prob_pCIC < p_pCIC] += 1

        # EM amp (of source + dark + pCIC)
        id_nnul = image != 0
        image[id_nnul] = np.random.gamma(image[id_nnul], EmGain)

        # Addition of sCIC inside EM registers (ie partially amplified)
        prob_sCIC = np.random.rand(size[1], size[0])  # Draw a number prob in [0,1]
        id_scic = prob_sCIC < p_sCIC  # sCIC positions
        # partial amplification of sCIC
        register = np.random.randint(1, n_registers, size=id_scic.sum())  # Draw at which stage of the EM register the electorn is created
        image[id_scic] += np.random.exponential(np.power(EmGain, register / n_registers))

    if save:
        # fitswrite((image * ConversionGain).round().astype("int32"), name[:-5] + "_before_smearing_and_RN.fits")
        fits.HDUList(fits.PrimaryHDU((image * ConversionGain).round().astype("int32"))).writeto(name[:-5] + "_before_smearing_and_RN.fits",overwrite=True)

    # semaring post EM amp (sgest noise reduction)
    if Smearing > 0:
        # exp = lambda x, a : np.exp(-x/a)
        # smearingKernelx = exp(np.arange(4,-1,-1),Smearing) #plt.plot(smearingKernel/np.sum(smearingKernel))
        # smearingKernely = exp(np.arange(2,-1,-1),Smearing/2) #plt.plot(smearingKernel/np.sum(smearingKernel))
        # image = np.apply_along_axis(lambda m: np.convolve(m, smearingKernely/smearingKernely.sum(), mode='same'), axis=0, arr=image)
        # image = np.apply_along_axis(lambda m: np.convolve(m, smearingKernelx/smearingKernelx.sum(), mode='same'), axis=1, arr=image)

        # smearing dependant on flux
        smearing_kernels = VariableSmearingKernels(image, Smearing, SmearExpDecrement)
        offsets = np.arange(6)
        A = dia_matrix((smearing_kernels.reshape((6, -1)), offsets), shape=(image.size, image.size))
        verboseprint("TEST")
        verboseprint(A)
        A
        verboseprint(image.ravel().shape)
        image = A.dot(image.ravel()).reshape(image.shape)

    # read noise
    readout = np.random.normal(Bias, RN, (size[1], size[0]))

    verboseprint(np.max(((image + readout) * ConversionGain).round()))
    if np.max(((image + readout) * ConversionGain).round()) > 2 ** 15:
        type_ = "int32"
    else:
        type_ = "int16"
    verboseprint("Flux = %0.3f, gamma scale = %0.1f, RN = %0.1f" % (Dark * exposure, EmGain, RN))
    verboseprint("Saving data in type " + type_)

    imaADU_wo_RN = (image * ConversionGain).round().astype(type_)
    imaADU_RN = (readout * ConversionGain).round().astype(type_)
    imaADU = ((image + readout) * ConversionGain).round().astype(type_)

    if save:
        # fitswrite(imaADU_wo_RN, name[:-5] + "_before_RN.fits")
        # fitswrite(imaADU_RN, name[:-5] + "_RN.fits")
        # fitswrite(imaADU, name)
        fits.HDUList(fits.PrimaryHDU(imaADU_wo_RN)).writeto( name[:-5] + "_before_RN.fits",overwrite=True)
        fits.HDUList(fits.PrimaryHDU(imaADU_RN)).writeto(name[:-5] + "_RN.fits",overwrite=True)
        fits.HDUList(fits.PrimaryHDU(imaADU)).writeto(name,overwrite=True)


    # Not sure why now, but several events much higher than 2**16 -> put then to 0 for now...
    id_null = np.array((image + readout) * ConversionGain, dtype="int16") < 0
    image[id_null] = 0

    return imaADU, imaADU_wo_RN, imaADU_RN


def VariableSmearingKernels(image, Smearing=1.5, SmearExpDecrement=50000):
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    smearing_kernels = np.exp(-np.arange(6)[:, np.newaxis, np.newaxis] / smearing_length)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels





# def SimulateFIREBallemCCD(xpapoint, DS9backUp=DS9_BackUp_path):
    # from astropy.io import fits
    # from decimal import Decimal
    # from astropy.table import Table

# d = DS9n(xpapoint)

print(sys.argv[3])
lx, ly = np.array(sys.argv[3].split(","), dtype=int)
if sys.argv[4] == "1":
    try:
        if len(sys.argv[5].split(",")) == 1:
            OS1, OS2 = int(sys.argv[5]), -1
        else:
            OS1, OS2 = np.array(sys.argv[5].split(","), dtype=int)
    except ValueError:
        OS1, OS2 = 0, -1
else:
    OS1, OS2 = 0, -1

ConversionGain, EmGain, n_registers, Bias, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, source, Rx, Ry, name, spectra, cube, sigma, hist_path = sys.argv[6:]
ConversionGain, EmGain, n_registers, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, Rx, Ry, sigma = np.array([ConversionGain, EmGain, n_registers, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, Rx, Ry, sigma], dtype=float)
if SmearExp == 0:
    SmearExpDecrement = np.inf
else:
    SmearExpDecrement = SmearExp
# if name == "Auto":
name =  "/tmp/" + "image000001_emgain%i_RON%i_CIC%0.0E_Dark%0.0E_Exp%i_Smearing%0.1E_ExpSmear%0.1E_.fits" % (EmGain, RN, Decimal(pCIC), Decimal(Dark), exposure, Decimal(Smearing), Decimal(SmearExp))
# else:
    # name = DS9backUp + "CreatedImages/" + name
image, image_woRN, ReadNOise = SimulateFIREBallemCCDImage(
    ConversionGain=ConversionGain,
    EmGain=EmGain,
    Bias=Bias,
    RN=RN,
    p_pCIC=pCIC,
    p_sCIC=sCIC,
    Dark=Dark,
    Smearing=Smearing,
    SmearExpDecrement=SmearExpDecrement,
    exposure=exposure,
    flux=flux,
    source=source,
    Rx=Rx / 2.335,
    n_registers=n_registers,
    Ry=Ry / 2.353,
    size=(lx, ly),
    spectra=spectra,
    cube=cube,
    OSregions=(OS1, OS2),
    name=name,
)
fitsimage = fits.HDUList([fits.PrimaryHDU(image)])[0]
fitsimage.header["CONVGAIN"] = (float(ConversionGain), "Conversion Gain")
fitsimage.header["EMGAIN"] = (float(EmGain), "Amplification gain")
fitsimage.header["BIAS"] = (Bias, "Detector bias")
fitsimage.header["READNOIS"] = (float(RN), "Read noise in ADU")
fitsimage.header["CIC"] = (float(pCIC), "Charge induced current")
fitsimage.header["sCIC"] = (float(sCIC), "Charge induced current")
fitsimage.header["DARK"] = (float(Dark), "Dark current")
fitsimage.header["SMEARING"] = (float(Smearing), "Smearing length")
fitsimage.header["SMEARDEC"] = (float(SmearExp), "Smearing length exponential decrement")
fitsimage.header["EXPTIME"] = (float(exposure), "Exposure time in second")
fitsimage.header["FLUX"] = (float(flux), "Flux un e'-/pix/sec")
fitsimage.header["SOURCE"] = (source, "Type of source")
fitsimage.header["Rx"] = (float(Rx), "Spatial resolution in pixel FWHM")
fitsimage.header["Ry"] = (float(Ry), "Spectral resolution in pixel FWHM")
fitsimage.header["OS-1"] = (float(OS1), "Over scann left position")
fitsimage.header["OS-2"] = (float(OS2), "Over scann right position")
fitsimage.header["SPECTRA"] = (spectra, "path of the spectra in entry")
fitsimage.header["MOCKCUBE"] = (cube, "path of the spatio-spectral mock cube in entry")
# fitswrite(fitsimage, name)
fits.HDUList(fitsimage).writeto(name,overwrite=True)

d=DS9n()
d.set("frame new ; file " + name)
# verboseprint(lx, ly)
# verboseprint(" ConversionGain, EmGain, Bias, RN, CIC, Dark, Smearing =", ConversionGain, EmGain, Bias, RN, pCIC, Dark, Smearing)
# if Bias == "Auto":
#     if EmGain > 1:
#         Bias = 3000  # / ConversionGain
#     else:
#         Bias = 6000  # / ConversionGain
# else:
#     Bias = float(Bias)
#
# value, bins = np.histogram(image[:, OS1:OS2], range=[0, 2 ** 16], bins=int(2 ** 16 / 2 ** 2))  # ,
# bins_c = (bins[1:] + bins[:-1]) / 2
# detections = 100 * np.exp(-RN * sigma * ConversionGain / (EmGain * Smearing2Noise(exp_coeff=Smearing)["Hist_smear"]))
# Fake_detections = 100 * FakeDetectionGaussian(sigma=1, threshold=sigma * ConversionGain)
# plt.figure()
# if os.path.isfile(hist_path):
#     cat = Table.read(hist_path)
#     x, y = cat["col0"], cat["col1"]
#     plt.step(x, np.log10(y) - (np.max(np.log10(y)) - np.max(np.log10(value))), c="grey", label="Input histogram: " + os.path.basename(hist_path), alpha=0.5)
#
# plt.step(bins_c, np.log10(value))#, label="%0.1f%% of True Detections\n%0.1f%% of Fake detections RN" % (detections, Fake_detections))
# plt.fill_between(bins_c[bins_c <= (Bias) + sigma * RN * ConversionGain], np.log10(value[bins_c <= (Bias) + sigma * RN * ConversionGain]), alpha=0.2, step="pre", label="0 in pc mode: %0.1f sigma" % (sigma))
# plt.fill_between(bins_c[bins_c >= (Bias) + sigma * RN * ConversionGain], np.log10(value[bins_c > (Bias) + sigma * RN * ConversionGain]), alpha=0.2, step="pre", label="1 in pc mode: %0.1f sigma" % (sigma))
# plt.grid(True, which="both", linestyle="dotted")
# plt.legend()
# plt.xlim((2500, 10000))
# # plt.gca().set_xlim(right=10000)
# plt.xlabel("ADU value")
# plt.title("Histogram")
# plt.ylabel("Log(#)")
# plt.grid(True)
# plt.show()
# return
