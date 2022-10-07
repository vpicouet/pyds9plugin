import numpy as np
import sys, os
from astropy.io import fits
import matplotlib.pyplot as plt
from decimal import Decimal
from astropy.table import Table
from pyds9plugin.DS9Utils import verboseprint, DS9n


import astropy.io.fits as pyfits
from astropy.wcs import WCS
from astropy import units

import numpy as np

class FitsCube(object):
    
    def __init__(self, *args, **kwargs):
        filename = kwargs.get('filename')
        if not filename is None:
            self.read(filename)
        else:
            self.build(*args, **kwargs)
         
        self.default_cunit = self.wcs.wcs.cunit

            
    def read(self, fname):
        """
        Read a FITS cube from file name *fname*.
        """
        self.filename = fname
        
        with pyfits.open(fname) as hdu:
    
            self.hdr = hdu[0].header
            
            self.wcs = WCS(self.hdr)
                            
            self.cube = np.transpose(hdu[0].data)
        
        
        return  self.cube, self.wcs


    def build(self, cube, steps, center, crpix=None, 
              ctype=["LINEAR", "LINEAR", "WAVE"] , cunit=['deg', 'deg', 'um']):
        """
        steps = arr(dx,dy,dw)
        
        dw: step along wavelength axis
        dx: step along field of view x
        dy: step along field of view y
        
        center = arr(x,y,w)
        x:  field of view of reference (fov slice center)
        y:  field of view of reference (fov slice center)
        w:  wavelength of reference    (bandwidth center)

        """
        
        self.cube = cube
        
        # The data is stored in row-major format
        # NAXIS3 is the cube w (slices)
        # NAXIS2 is the cube y (rows)
        # NAXIS1 is the cube x (columns)
        
        # create a primary header 
        naxis = cube.ndim
        self.wcs = WCS(naxis=naxis)
        
        naxis1 = self.cube.shape[0]
        naxis2 = self.cube.shape[1]
        if naxis == 3:
            naxis3 = self.cube.shape[2]
        
        # compute the pixel of reference associated to        
        # 1) if the cube has an odd number of pixels
        #   the central pixel has a defined integral value 
        #   given center is associated to it.
        #
        # 2) if the cube has an even number of pixel
        #   then crpix is is not integer
        if crpix is None:
            crpix = np.zeros(naxis)
        
            if naxis1 % 2:
                crpix[0] = naxis1 / 2 + 1
            else:
                # even number of pixels
                crpix[0] = (naxis1 - 1) / 2 + 1 
            
            if naxis2 % 2:
                # odd number of pixels
                crpix[1] = naxis1 / 2 + 1
            else:
                # even number of pixels
                crpix[1] = (naxis2 - 1) / 2 + 1 
            
            if naxis ==3:
                if naxis3 % 2:
                    # odd number of pixels
                    crpix[2] = naxis3 / 2 + 1
                else:
                    # even number of pixels
                    crpix[2] = (naxis3 - 1) / 2 + 1 

        if naxis == 2:
            ctype = ctype[0:2]    
            cunit = cunit[0:2]
                   
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.cdelt = steps
        self.wcs.wcs.crval = center
        self.wcs.wcs.ctype = ctype
        self.wcs.wcs.cunit = cunit

        self.hdr = self.wcs.to_header()
          

   
    def write(self, filename=None):
        """
        Write a fits *cube* to file name *fname*.
        """
        
        hdu = pyfits.PrimaryHDU(data=np.transpose(self.cube), header=self.hdr) 

        if not filename is None:
            self.filename = filename
        hdu.writeto(self.filename, clobber=True)        


    def get_to_unit(self, attr, unit):
        """
        Return the given attribute in the given units
        """
        wcs_unit = map(units.Unit, self.wcs.wcs.cunit)
        final_unit = map(units.Unit, unit)
        res = getattr(self.wcs.wcs, attr) * wcs_unit
        res = np.array([r.to(u) for r,u in zip(res, final_unit)] ,dtype=object)
        return res

    @property
    def crpix(self):
        return self.wcs.wcs.crpix        

    @property
    def cdelt(self):
        """
        Return the cube resolution in default_cunit
        """       
        return self.get_cdelt(unit=self.default_cunit)

        
    def get_cdelt(self, unit=["deg", "deg", "um"]):
        """
        Return the cube resolution in the given units
        """
        return self.get_to_unit("cdelt", unit)
        
    @property
    def crval(self):
        """
        Return the reference pixel world coordinates in deg x deg x um
        """       
        return self.get_crval(unit=self.default_cunit)

        
    def get_crval(self, unit=["deg", "deg", "um"]):
        """
        Return the reference pixel world coordinates in the given units
        """
        return self.get_to_unit("crval", unit)

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

def SimulateFIREBallemCCDHist(ConversionGain, EmGain, Bias, RN, p_pCIC, p_sCIC, Dark, Smearing, SmearExpDecrement, exposure, n_registers):
    # bins, ConversionGain, EmGain, Bias, RN, p_pCIC, p_sCIC, Dark, Smearing, SmearExpDecrement, exposure,  n_registersbins, ConversionGain, EmGain, Bias, RN, p_pCIC, p_sCIC, Dark, Smearing, SmearExpDecrement, exposure,  n_registers = args
    # ConversionGain=0.53, EmGain=1500, Bias=3000, RN=80, p_pCIC=1, p_sCIC=1, Dark=5e-4, Smearing=0.7, SmearExpDecrement=50000, exposure=50,  n_registers=604
    imaADU, imaADU_wo_RN, imaADU_RN = SimulateFIREBallemCCDImage(ConversionGain=ConversionGain, EmGain=EmGain, Bias=Bias, RN=RN, p_pCIC=p_pCIC, p_sCIC=p_sCIC, Dark=Dark, Smearing=Smearing, SmearExpDecrement=SmearExpDecrement, exposure=exposure, n_registers=n_registers, save=False)
    # n, bins = np.histogram(imaADU[:,1066:2124].flatten(),range=[0,2**16], bins = int(2**16/2**2))#, range=(-200,11800))
    return imaADU[:, 1066:2124]  # n#, (bins[:-1]+bins[1:])/2

#+ "image000001_emgain%i_RON%i_CIC%0.0E_Dark%0.0E_Exp%i_Smearing%0.1E_ExpSmear%0.1E_.fits" % (EmGain, RN, Decimal(pCIC), Decimal(Dark), exposure, Decimal(Smearing), Decimal(SmearExp))
from pyds9plugin.tools import SimulateFIREBallemCCDImage
name_source =  "/tmp/source.fits" 
name_single =  "/tmp/single.fits" 
name_stack =  "/tmp/stack.fits" 
name_counting =  "/tmp/counting.fits" 
#%%
# %load_ext line_profiler
RN=60
size=[1058, 2069]
OSregions=[0, 1058]
#69 sec without OS, 1.5 sec without stack, 37 sec sithout OS, 20% due to CR, 16% due to cube RN, 40% due to counting
# 23 sec withotu OS and CR, 19 % cube + 20% cube + 30% cube, 30% stack
# 7 sec without counting: 84 ims atck, 7% smearing
#%lprun -f SimulateFIREBallemCCDImage 
imaADU, imaADU_stack, cube_stack, source_im = SimulateFIREBallemCCDImage(source="Field", size=size, OSregions=OSregions,p_pCIC=0.0005,exposure=50,Dark=1/3600,cosmic_ray_loss=None,Smearing=0.3,stack=int(200*1/50),RN=RN,Rx=5,Ry=5,readout_time=5,counting=True)

print(cube_stack.min(),cube_stack.max())
#%%

fits.HDUList(fits.HDUList([fits.PrimaryHDU(imaADU_stack)])[0]).writeto(name_stack,overwrite=True)
fits.HDUList(fits.HDUList([fits.PrimaryHDU(imaADU)])[0]).writeto(name_single,overwrite=True)
fits.HDUList(fits.HDUList([fits.PrimaryHDU(source_im)])[0]).writeto(name_source,overwrite=True)
threshold=5.5

stacked_image = np.nansum(cube_stack>threshold*RN,axis=0)
fits.HDUList(fits.HDUList([fits.PrimaryHDU(stacked_image)])[0]).writeto(name_counting,overwrite=True)
# im0 = self.ax0.imshow(stacked_image, aspect="auto",cmap=self.current_cmap)


d=DS9n()
d.set("frame new ; file " + name_source)
d.set("frame new ; file " + name_single)
d.set("frame new ; file " + name_stack)
d.set("frame new ; file " + name_counting)


#TODO take into account the redshift and type of the source
#TODO take into account magnitude
#TODO add the atmosphere absorption/emission features
#TODO add the stacking 


# def SimulateFIREBallemCCDImage(
#     ConversionGain=0.53, EmGain=1500, Bias="Auto", RN=80, p_pCIC=1, p_sCIC=1, Dark=5e-4, Smearing=0.7, SmearExpDecrement=50000, exposure=50, flux=1e-3, source="Spectra", Rx=8, Ry=8, size=[3216, 2069], OSregions=[1066, 2124], name="Auto", spectra="-", cube="-", n_registers=604, save=True
# ):
#     # print('conversion gain = ', ConversionGain)
#     from astropy.modeling.functional_models import Gaussian2D
#     from scipy.sparse import dia_matrix

#     OS1, OS2 = OSregions
#     if Bias == "Auto":
#         if EmGain > 1:
#             Bias = 3000 / ConversionGain
#         else:
#             Bias = 6000 / ConversionGain
#     else:
#         Bias = float(Bias) / ConversionGain
#     image = np.zeros((size[1], size[0]), dtype="float32")

#     # dark & flux
#     source_im = 0 * image[:, OSregions[0] : OSregions[1]]
#     lx, ly = source_im.shape
#     y = np.linspace(0, lx - 1, lx)
#     x = np.linspace(0, ly - 1, ly)
#     x, y = np.meshgrid(x, y)

#     # Source definition. For now the flux is not normalized at all, need to fix this
#     # Cubes still needs to be implememted, link to detector model or putting it here?
#     if os.path.isfile(cube):
#         file = '/Users/Vincent/Github/fireball2-etc/notebooks/10pc/cube_204nm_guidance0.5arcsec_slit100um_total_fc_rb_detected.fits'#%(pc,wave,slit)
#         source_im+=fits.open(file)[0].data * 0.7 #cf athmosphere was computed at 45km

#         source_im = FitsCube(filename=cube)
#     elif source == "Flat-field":
#         source_im += flux
#     elif source == "Dirac":
#         source_im += Gaussian2D.evaluate(x, y, 100 * flux, ly / 2, lx / 2, Ry, Rx, 0)
#     elif source == "Spectra":
#         source_im += Gaussian2D.evaluate(x, y, 100 * flux, ly / 2, lx / 2, 100 * Ry, Rx, 0)
#     elif source == "Slit":
#         ConvolveSlit2D_PSF_75muWidth = lambda xy, amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
#         source_im += ConvolveSlit2D_PSF_75muWidth((x, y), 30000 * flux, 9, ly / 2, lx / 2, Rx, Ry).reshape(lx, ly)
#     elif source == "Fibre":
#         verboseprint("Create fibre source, FWHM: ", 2.353 * Rx, 2.353 * Ry)
#         fibre = convolvePSF(radius_hole=10, fwhmsPSF=[2.353 * Rx, 2.353 * Ry], unit=1, size=(201, 201), Plot=False)  # [:,OSregions[0]:OSregions[1]]
#         source_im = addAtPos(source_im, fibre, (int(lx / 2), int(ly / 2)))
#         verboseprint("Done")
#     elif source[:5] == "Field":
#         ConvolveSlit2D_PSF_75muWidth = lambda xy, amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
#         ws = [2025, 2062, 2139]
#         # x, y = [], []
#         # for i, w in enumerate(ws):
#             # slits = returnXY(source[0].lower() + source[-1], w=w, frame="observedframe")
#         file = '/Users/Vincent/Github/fireball2-etc/notebooks/10pc/cube_204nm_guidance0.5arcsec_slit100um_total_fc_rb_detected.fits'#%(pc,wave,slit)
#         gal=fits.open(file)[0].data * 0.7 #cf athmosphere was computed at 45km

#         slits = Table.read("/Users/Vincent/Github/FireBallPipe/Calibration/Targets/2022/targets_F1.csv")
#         xs = slits["Y_IMAGE"]
#         ys = slits["X_IMAGE"]
#         index = (ys > OS1) & (ys < OS2)
#         verboseprint(xs, ys)
#         for yi, xi in zip(np.array(ys[index]) - OS1, xs[index]):
#             verboseprint(xi, yi)
#             source_im = addAtPos(source_im, 10*gal, [int(xi), int(yi)])
#             # source_im += ConvolveSlit2D_PSF_75muWidth((x, y), 40000 * flux, 9, yi, xi, Rx, Ry).reshape(lx, ly)

#     # Poisson realisation
#     source_im2 = np.random.poisson((Dark + source_im) * exposure)

#     # Addition of the phyical image on the 2 overscan regions
#     image[:, OSregions[0] : OSregions[1]] += source_im2

#     if save:
#         #        test = image.copy()
#         #        test[:,OSregions[0]:OSregions[1]] += (Dark + source_im) * exposure
#         # fitswrite(image.astype("int32"), name[:-5] + "_beforeAmp.fits")
#         fits.HDUList(fits.PrimaryHDU(image.astype("int32"))).writeto(name[:-5] + "_beforeAmp.fits",overwrite=True)
#     if EmGain > 1:

#         # addition of pCIC (stil need to add sCIC before EM registers)
#         prob_pCIC = np.random.rand(size[1], size[0])  # Draw a number prob in [0,1]
#         image[prob_pCIC < p_pCIC] += 1

#         # EM amp (of source + dark + pCIC)
#         id_nnul = image != 0
#         image[id_nnul] = np.random.gamma(image[id_nnul], EmGain)

#         # Addition of sCIC inside EM registers (ie partially amplified)
#         prob_sCIC = np.random.rand(size[1], size[0])  # Draw a number prob in [0,1]
#         id_scic = prob_sCIC < p_sCIC  # sCIC positions
#         # partial amplification of sCIC
#         register = np.random.randint(1, n_registers, size=id_scic.sum())  # Draw at which stage of the EM register the electorn is created
#         image[id_scic] += np.random.exponential(np.power(EmGain, register / n_registers))

#     if save:
#         # fitswrite((image * ConversionGain).round().astype("int32"), name[:-5] + "_before_smearing_and_RN.fits")
#         fits.HDUList(fits.PrimaryHDU((image * ConversionGain).round().astype("int32"))).writeto(name[:-5] + "_before_smearing_and_RN.fits",overwrite=True)

#     # semaring post EM amp (sgest noise reduction)
#     if Smearing > 0:
#         # exp = lambda x, a : np.exp(-x/a)
#         # smearingKernelx = exp(np.arange(4,-1,-1),Smearing) #plt.plot(smearingKernel/np.sum(smearingKernel))
#         # smearingKernely = exp(np.arange(2,-1,-1),Smearing/2) #plt.plot(smearingKernel/np.sum(smearingKernel))
#         # image = np.apply_along_axis(lambda m: np.convolve(m, smearingKernely/smearingKernely.sum(), mode='same'), axis=0, arr=image)
#         # image = np.apply_along_axis(lambda m: np.convolve(m, smearingKernelx/smearingKernelx.sum(), mode='same'), axis=1, arr=image)

#         # smearing dependant on flux
#         smearing_kernels = VariableSmearingKernels(image, Smearing, SmearExpDecrement)
#         offsets = np.arange(6)
#         A = dia_matrix((smearing_kernels.reshape((6, -1)), offsets), shape=(image.size, image.size))
#         verboseprint("TEST")
#         verboseprint(A)
#         A
#         verboseprint(image.ravel().shape)
#         image = A.dot(image.ravel()).reshape(image.shape)

#     # read noise
#     readout = np.random.normal(Bias, RN, (size[1], size[0]))

#     verboseprint(np.max(((image + readout) * ConversionGain).round()))
#     if np.max(((image + readout) * ConversionGain).round()) > 2 ** 15:
#         type_ = "int32"
#     else:
#         type_ = "int16"
#     verboseprint("Flux = %0.3f, gamma scale = %0.1f, RN = %0.1f" % (Dark * exposure, EmGain, RN))
#     verboseprint("Saving data in type " + type_)

#     imaADU_wo_RN = (image * ConversionGain).round().astype(type_)
#     imaADU_RN = (readout * ConversionGain).round().astype(type_)
#     imaADU = ((image + readout) * ConversionGain).round().astype(type_)

#     if save:
#         # fitswrite(imaADU_wo_RN, name[:-5] + "_before_RN.fits")
#         # fitswrite(imaADU_RN, name[:-5] + "_RN.fits")
#         # fitswrite(imaADU, name)
#         fits.HDUList(fits.PrimaryHDU(imaADU_wo_RN)).writeto( name[:-5] + "_before_RN.fits",overwrite=True)
#         fits.HDUList(fits.PrimaryHDU(imaADU_RN)).writeto(name[:-5] + "_RN.fits",overwrite=True)
#         fits.HDUList(fits.PrimaryHDU(imaADU)).writeto(name,overwrite=True)


#     # Not sure why now, but several events much higher than 2**16 -> put then to 0 for now...
#     id_null = np.array((image + readout) * ConversionGain, dtype="int16") < 0
#     image[id_null] = 0

#     return imaADU, imaADU_wo_RN, imaADU_RN





# print(sys.argv[3])
# lx, ly = np.array(sys.argv[3].split(","), dtype=int)
# if sys.argv[4] == "1":
#     try:
#         if len(sys.argv[5].split(",")) == 1:
#             OS1, OS2 = int(sys.argv[5]), -1
#         else:
#             OS1, OS2 = np.array(sys.argv[5].split(","), dtype=int)
#     except ValueError:
#         OS1, OS2 = 0, -1
# else:
#     OS1, OS2 = 0, -1

# ConversionGain, EmGain, n_registers, Bias, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, source, Rx, Ry, name, spectra, cube, sigma, hist_path = sys.argv[6:]
# ConversionGain, EmGain, n_registers, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, Rx, Ry, sigma = np.array([ConversionGain, EmGain, n_registers, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, Rx, Ry, sigma], dtype=float)
# ConversionGain, EmGain, n_registers, RN, pCIC, sCIC, Dark, Smearing, SmearExp, exposure, flux, Rx, Ry, sigma = np.array([1, 2000, 506, 60, 0.005, 0.005, 1/3600, 0.5, 50000, 50, 10, Rx, Ry, sigma], dtype=float)
# if SmearExp == 0:
#     SmearExpDecrement = np.inf
# else:
#     SmearExpDecrement = SmearExp
# if name == "Auto":
# else:
    # name = DS9backUp + "CreatedImages/" + name
# image, image_woRN, ReadNOise = SimulateFIREBallemCCDImage(ConversionGain=ConversionGain,EmGain=EmGain,Bias=Bias,RN=RN,p_pCIC=pCIC,p_sCIC=sCIC,Dark=Dark,Smearing=Smearing,SmearExpDecrement=SmearExpDecrement,exposure=exposure,flux=flux,source=source,Rx=Rx / 2.335,n_registers=n_registers,Ry=Ry / 2.353,size=(lx, ly),spectra=spectra,cube=cube,OSregions=(OS1, OS2),name=name,)
# if __name__ == "__main__":



# fitsimage.header["CONVGAIN"] = (float(ConversionGain), "Conversion Gain")
# fitsimage.header["EMGAIN"] = (float(EmGain), "Amplification gain")
# fitsimage.header["BIAS"] = (Bias, "Detector bias")
# fitsimage.header["READNOIS"] = (float(RN), "Read noise in ADU")
# fitsimage.header["CIC"] = (float(pCIC), "Charge induced current")
# fitsimage.header["sCIC"] = (float(sCIC), "Charge induced current")
# fitsimage.header["DARK"] = (float(Dark), "Dark current")
# fitsimage.header["SMEARING"] = (float(Smearing), "Smearing length")
# fitsimage.header["SMEARDEC"] = (float(SmearExp), "Smearing length exponential decrement")
# fitsimage.header["EXPTIME"] = (float(exposure), "Exposure time in second")
# fitsimage.header["FLUX"] = (float(flux), "Flux un e'-/pix/sec")
# fitsimage.header["SOURCE"] = (source, "Type of source")
# fitsimage.header["Rx"] = (float(Rx), "Spatial resolution in pixel FWHM")
# fitsimage.header["Ry"] = (float(Ry), "Spectral resolution in pixel FWHM")
# fitsimage.header["OS-1"] = (float(OS1), "Over scann left position")
# fitsimage.header["OS-2"] = (float(OS2), "Over scann right position")
# fitsimage.header["SPECTRA"] = (spectra, "path of the spectra in entry")
# fitsimage.header["MOCKCUBE"] = (cube, "path of the spatio-spectral mock cube in entry")
# fitswrite(fitsimage, name)
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
