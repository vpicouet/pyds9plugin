# from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD, EMCCDhist
from pyds9plugin.DS9Utils import *#DS9n,PlotFit1D
# from pyds9plugin.Macros import functions
#TODO  create histogram for every image
from astropy.table import Column
from scipy import signal
import matplotlib

np.seterr(divide = 'ignore') 

matplotlib.use('Agg')

#TODO define fucntions
#3216*2069
#OS 1053 ... 2133
#Backside: -2133: -1053

# from astropy.table import Table
# d=DS9n()
# fitsfile=d.get_pyfits()
# table=Table(data=[[1]],names=['test'])
# filename=get_filename(d)

full_analysis = True
Plot = True
data = fitsfile[0].data
header = fitsfile[0].header
conversion_gain =  1 / 4.5# ADU/e-  0.53  



analysis_path = os.path.join(os.path.dirname(filename),'analysis/')
# os.mkdir(analysis_path) #kills the jobs return none!!



def plot_hist(bin_center, n,filename, header, table,masks, conversion_gain=conversion_gain,analysis_path=analysis_path):
    from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD, EMCCDhist
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    limit = 2000 #After 2000 it is getting very bad
    f=1.1
    n_conv = 4
    # table['Gain1'] *=1.2
    bias, ron, gain, flux = table['bias_fit'],  table['RON'], table['Gain1'],table["TopImage"]/table['Gain1']/conversion_gain
    fig, ax = plt.subplots(figsize=(9,5))  
    # ax = fig.add_subplot(111)
    ax.set_xlabel("Pixel Value [ADU]", fontsize=12)
    ax.set_ylabel("Log(\# Pixels)", fontsize=12)
    l_data = "%s-%s\nExposure = %i sec\nGain = %i \n" "T det = %0.1f C" % (header['TEMPDATE'],header['TEMPTIME'],header['EXPTIME'], header['EMGAIN'], float(header['TEMPA']))
    l_model = "Bias = %0.1f DN\n$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10" % (bias,ron, gain, flux )
    threshold = table['bias_fit']+ 5.5* (table['RON']*conversion_gain)
    # ax.semilogy([threshold,threshold],[0,1e6],'k:')
    # model_to_fit = lambda bin_center,biais\,RN,EmGain,flux,p_sCIC,Smearing
    ax.semilogy(bin_center, np.convolve(n,np.ones(n_conv)/n_conv,mode='same'),  label="Data:\n"+l_data,c='k')
    model =            10**EMCCD(    bin_center,bias, ron, gain, flux, sCIC=0,smearing=0)
    model_stochastic = 10**EMCCDhist(bin_center,bias, ron, gain, flux,sCIC=0,smearing=0)
    model_low = 10**EMCCD(bin_center,bias, ron, f*gain, flux/f,sCIC=0,smearing=0)
    model_high = 10**EMCCD(bin_center,bias, ron, gain/f, flux*f,sCIC=0,smearing=0)
    constant = np.nanmax(n)/np.nanmax(model)
    constant_stochastic  = np.nanmax(n)/np.nanmax(model_stochastic)
    model_to_fit =            lambda bin_center,EmGain,flux : EMCCD(    bin_center,bias,ron,EmGain,flux,0,0)+np.log10(constant)
    model_to_fit_cic =            lambda bin_center,EmGain,flux, cic : EMCCD(    bin_center,bias,ron,EmGain,flux,0,cic)+np.log10(constant)
    model_to_fit_stochastic = lambda bin_center,EmGain,flux : EMCCDhist(bin_center,bias,ron,EmGain,flux,0,0)+np.log10(constant_stochastic)
    ax.semilogy(bin_center, model * constant, ":", label="Model (from slope):\n"+l_model,color='k')
    ax.semilogy(bin_center[masks[0]], 10*np.ones(len(bin_center[masks[0]])), "k-")#, label="Mask %i"%(i+1))
    mask = (bin_center>1100)&(bin_center<limit)&(n>0)
    p0=np.array([float(table['Gain1']),float(table["TopImage"]/table['Gain1']/conversion_gain),0.005])
    popt, pcov = curve_fit(model_to_fit_cic,bin_center[mask],np.log10(n[mask]),p0=p0)
    l_fit = "$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10%%\nCIC=%0.4fe-/pix" % (ron,popt[0],popt[1],popt[2])
    # ax.semilogy(bin_center[mask], 10**model_to_fit(bin_center[mask],*popt), "b", label="Fit:\n"+l_fit)

    def change_gain_flux(popt,factor):
        popt1 = list(map(lambda item: popt[1]*factor if item==popt[1] else item, popt))
        popt2 = list(map(lambda item: popt1[0]/factor if item==popt1[0] else item, popt1))
        return popt2

    # fit_stochastic_high = 10**model_to_fit_stochastic(bin_center[mask],*change_gain_flux(popt,1/f))
    # ax.semilogy(bin_center[mask], 10**model_to_fit_stochastic(bin_center[mask],*popt), "blue", label="Fit:\n"+l_fit,lw=1)
    # fit_stochastic_low = 10**model_to_fit_stochastic(bin_center[mask],*change_gain_flux(popt,f))
    fit_stochastic_low = 10**model_to_fit_cic(bin_center[mask],*change_gain_flux(popt,f))
    fit_stochastic_high = 10**model_to_fit_cic(bin_center[mask],*change_gain_flux(popt,1/f))
    ax.semilogy(bin_center[mask], 10**model_to_fit_cic(bin_center[mask],*popt), "blue", label="Least square fit:\n"+l_fit,lw=1)
    # ax.semilogy(bin_center[mask], 10**model_to_fit_stochastic(bin_center[mask],*popt), "blue", label="Least square fit:\n"+l_fit,lw=1)
    # ax.semilogy(bin_center[mask], 10**model_to_fit_stochastic(bin_center[mask],*popt), "blue", label="Least square fit:\n"+l_fit,lw=1)
    ax.fill_between(bin_center[mask],fit_stochastic_low,fit_stochastic_high,alpha=0.15,color='blue')
    table['gain_ls']=popt[0]
    table['flux_ls']=popt[1]
    table['sCIC_ls']=popt[1]
    # def model_to_fit_stochastic_smearing_cic(bin_center, smearing,cic):
    #     return EMCCDhist(bin_center,bias,ron,table['gain_ls'],table['flux_ls'],smearing,cic )+np.log10(constant_stochastic)
    # # model_to_fit_stochastic_smearing_cic = lambda bin_center, smearing,cic : EMCCDhist(bin_center,bias,ron,table['gain_ls'],table['flux_ls'],smearing,cic )+np.log10(constant_stochastic)
    # # ax.semilogy(bin_center[mask], 10**model_to_fit_stochastic_smearing_cic(bin_center[mask],0.01,0.01), "r", label=None,lw=1)
    # # ax.semilogy(bin_center[mask], 10**model_to_fit_stochastic_smearing_cic(bin_center[mask],0.01,0.01), "blue", label=None,lw=1)
    # mask_smearing_CIC = (bin_center>1100)&(bin_center<1350)&(np.log10(n)>0)
    # p0=[0.01,0.01,]
    # # popt2, pcov2 = curve_fit(model_to_fit_stochastic_smearing_cic,bin_center[mask_smearing_CIC],np.log10(n[mask_smearing_CIC]),p0=p0)
    # plt.plot(bin_center[mask_smearing_CIC],10**np.log10(n[mask_smearing_CIC]))
    # plt.plot(bin_center[mask_smearing_CIC],10**model_to_fit_stochastic_smearing_cic(bin_center[mask_smearing_CIC],*p0))
    # popt2, pcov2 = curve_fit( model_to_fit_stochastic_smearing_cic,bin_center[mask_smearing_CIC],np.log10(n[mask_smearing_CIC]),p0=p0)
    # print(popt2)
    # plt.plot(bin_center[mask_smearing_CIC],10**model_to_fit_stochastic_smearing_cic(bin_center[mask_smearing_CIC],*popt2))

    ax.legend(loc="upper right", fontsize=10,ncol=3)
    ax.set_ylim(ymin=1e0,ymax=2.1*n.max())
    ax.set_xlim(xmin=bias-ron,xmax=limit+500)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(os.path.basename(filename).replace(".fits",""))
    fig.savefig(analysis_path + os.path.basename(filename).replace(".fits","_hist.png"))
    plt.show()
    plt.close(fig)
    # return



if data is None:
    data = np.nan * np.ones((10,10))
lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = -2133, -1053, 1, -1
physical_region = data[Yinf:Ysup, Xinf:Xsup]
pre_scan = data[:, 600:1000]
post_scan = data[:, 2500:3000]
column = np.nanmean(pre_scan, axis=1)
line = np.nanmean(pre_scan, axis=0)
table["Col2ColDiff_pre_scan"] = np.nanmedian(line[::2]) - np.nanmedian(line[1::2])
table["Line2lineDiff_pre_scan"] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])
table["SaturatedPixels"] = 100 * float(np.sum(physical_region> 2 ** 16 - 10)) / np.sum(physical_region > 0)
table['pre_scan'] = np.nanmedian(pre_scan)
table['post_scan'] =  np.nanmedian(post_scan)
table["stdXY_pre_scan"] = np.nanstd(pre_scan)
table["BottomImage"] = np.nanmean(physical_region[10:30,:]) - table['pre_scan']
table["TopImage"] = np.nanmean(physical_region[-30:-10,:]) - table['pre_scan']


##create subset
table['EMGAIN>0'] = header['EMGAIN']>0 
table['EMGAIN==0'] = header['EMGAIN']==0 
table['EXPTIME==0'] = header['EXPTIME']==0 
table['EXPTIME>0'] = header['EXPTIME']>0 

try:
    table["stdX"] = np.nanstd(data[int(Yinf + (Ysup - Yinf) / 2), Xinf:Xsup])
    table["stdY"] = np.nanstd(data[Yinf:Ysup, int(Xinf + (Xsup - Xinf) / 2)])
except IndexError:
    table["stdX"] = np.nanstd(data[int(lx / 2), :])
    table["stdY"] = np.nanstd(data[:, int(ly / 2)])
value, b = np.histogram(data[Yinf+1000:Ysup, Xinf:Xsup].flatten(),range=(1000,5000),bins=1000)
bins = (b[1:]+b[:-1])/2

if full_analysis:
    table['bins'] = Column([bins], name="bins")   
    table['hist'] = Column([ value], name="hist")   

# bins,value = bins[value>0],value[value>0]
table['bias'] = bins[np.argmax(value)]
RN=50
mask_RN = (bins>bins[np.argmax(value)]-1*50) & (bins<bins[np.argmax(value)]+0.8*50)  &(value>0)
table['Amp'] =   PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][0]
table['bias_fit'] =   PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][1]
table['RON'] =   np.abs(PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', plot_=False,P0=[1,bins[np.argmax(value)],50,0])['popt'][2]/conversion_gain)
mask_gain1 = (bins>bins[np.argmax(value)]+4*RN) & (bins<bins[np.argmax(value)]+10*RN)  
mask_gain2 = (bins>bins[np.argmax(value)]+10*RN) & (bins<bins[np.argmax(value)]+30*RN)
try:
    table['Gain1'] =   -1/np.log(10) / conversion_gain / PlotFit1D(bins[mask_gain1 & (value>0)],np.log10(value[mask_gain1 & (value>0)]),deg=1, plot_=False)['popt'][1]
    table['Gain2'] =   -1 / conversion_gain / PlotFit1D(bins[mask_gain2 & (value>0)],np.log(value[mask_gain2 & (value>0)]),deg=1, plot_=False)['popt'][1]
    table['Flux1'] =  table["TopImage"]/ table['Gain1'] /conversion_gain
    table['Flux2'] =   table["TopImage"]/ table['Gain2'] /conversion_gain
except (ValueError,RuntimeWarning):
    table['Gain1'] = -99
    table['Gain2'] = -99
    table['Flux1'] = -99
    table['Flux2'] = -99

if Plot :
    if (header['EMGAIN']>0):
        plot_hist(bins,value,filename, header, table,masks=[mask_gain1,mask_gain2])
    else:
        table['gain_ls']=0
        table['flux_ls']=0
        table['sCIC_ls']=0

table['sCIC_OS'] = (table['post_scan'] - table['pre_scan'] )/ table['gain_ls'] /conversion_gain

n=20
image_center = data[int(lx/2-n):int(lx/2+n)+1,int(ly/2-n):int(ly/2+n)+1]
image_center_01 =   (image_center - image_center.min()) / (image_center - image_center.min()).max()
fft =  np.fft.irfft2(np.fft.rfft2(image_center_01) * np.conj(np.fft.rfft2(image_center_01)))
table['fft_sum'] = np.sum(fft)
table['fft_var'] = np.var(fft)
table['fft_mean'] = np.mean(fft)
table['fft_median'] = np.median(fft)




x_correlation = np.zeros(image_center_01.shape)
for i in range(image_center_01.shape[0]):
    x_correlation[i, :] = signal.correlate(image_center_01[i, :], image_center_01[i, :], mode="same")  # / 128
x_correlation /= x_correlation.min()
size = 12
lxa, lya = x_correlation.shape
# plt.plot(np.mean(x_correlation[:, int(lya / 2) - size  : int(lya / 2) + size+1], axis=0))
# plt.plot(np.mean(x_correlation[:, int(lya / 2) - size  : int(lya / 2) + size+1], axis=0)[:size])
# # plt.imshow(np.log10(x_correlation[:, int(lya / 2) - size - 1 : int(lya / 2) + size]))
# plt.show()
profile = np.mean(x_correlation[:, int(lya / 2) - size - 1 : int(lya / 2) + size], axis=0)[:size+2]
if full_analysis:
    table['x_correlation'] =  Column([profile], name="x_correlation")
# plt.plot(np.arange(len(profile)), profile[::-1])
# PlotFit1D(np.arange(len(profile)), profile[::-1], deg='exp',P0=[5e-1, profile.max() - profile.min(), profile.min()], plot_=True,ax=ax)
# # plt.xlim((np.arange(len(profile).min(),np.arange(len(profile).max()))
# plt.show()
try:
    table['smearing_autocorr'] = PlotFit1D(np.arange(len(profile)), profile, deg='exp',P0=[5e-1, profile.max() - profile.min(), profile.min()], plot_=False)['popt'][2]
except ValueError:
    table['smearing_autocorr'] = -99

if full_analysis:
    table['overscan_decrease'] = Column([np.nanmedian(data[:, 2143:2143+200],axis=0)], name="overscan_decrease")    







# np.log10(5)

# np.log(5)/ np.log()

# y,bins = np.histogram(data[1053:2133,:].flatten())
# x = (bins[1:]+bins[:-1])/2

# fig, ax = plt.subplots()
# ax.plot(bins[mask_RN],value[mask_RN])
# l=PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', P0=[1e6,1160,50,0],plot_=True,ax=ax)
# ax.set_title(l['popt'])
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(bins[mask_RN],np.log10(value[mask_RN]))
# l=PlotFit1D(bins[mask_RN],np.log10(value[mask_RN]),deg=2, plot_=True)
# ax.set_title(l['popt'])
# plt.show()

# fig, ax = plt.subplots()
# ax.plot(bins[mask_gain],np.log10(value[mask_gain]))
# l=PlotFit1D(bins[mask_gain],np.log10(value[mask_gain]),deg=1, plot_=True,ax=ax)
# ax.set_title(l['popt'])
# plt.show()


# fig, ax = plt.subplots()
# ax.plot(bins[mask_RN],value[mask_RN])
# l=PlotFit1D(bins[mask_RN],value[mask_RN],deg='gaus', P0=[1e6,1160,50,0],plot_=True,ax=ax)
# # ax.plot(bins[mask_RN],np.log10(value[mask_RN]))
# # l=PlotFit1D(bins[mask_RN],np.log10(value[mask_RN]),deg=2, plot_=True)
# ax.set_title(l['popt'])
# # ax.plot(x[x<2500],np.log10(y[x<2500]))
# # PlotFit1D(x[x<2500],np.log10(y[x<2500]),deg=EMCCD,P0=[x[np.argmax(y)],44,1900,0.05,0],ax=ax)
# # PlotFit1D(x[x<2500],np.log10(y[x<2500]),deg=EMCCDhist,P0=[x[np.argmax(y)],44,1900,0.05,0,0],ax=ax)
# # ax.set_ylim((0,5.5))
# # ax.set_xlim((x.min(),x.max()))
# # plt.savefig('/tmp/test.png')
# plt.show()

# def DS9AddToCatalog(xpapoint):
#     path = sys.argv[3]
#     functions = np.array(np.array(sys.argv[4:], dtype=int), dtype=bool)
#     dict_functions = {
#         "ComputeEmGain": [ComputeEmGain, "EMG_var_int_w_OS"],
#         "calc_emgainGillian": [calc_emgainGillian, "EMG_hist"],
#         "SmearingProfileAutocorr": [SmearingProfileAutocorr, "Exp_coeff"],
#         "SmearingProfile": [SmearingProfile, "Exp_coeff"],
#         "CountHotPixels": [CountHotPixels, "#_HotPixels"],
#         "AnalyzeOSSmearing": [AnalyzeOSSmearing, "Exp1", "Exp2"],
#         "FluxDeffect": [FluxDeffect, "Flux"],
#     }
#     for function, f in zip(functions, dict_functions.keys()):
#         if function:
#             AddToCatalog(path, f, f)
#     return


# def AddToCatalog(path, function, field, Plot=False):
#     from astropy.table import Table

#     dict_functions = {
#         "ComputeEmGain": [ComputeEmGain, "EMG_var_int_w_OS"],
#         "calc_emgainGillian": [calc_emgainGillian, "EMG_hist"],
#         "SmearingProfileAutocorr": [SmearingProfileAutocorr, "Exp_coeff"],
#         "SmearingProfile": [SmearingProfile, "Exp_coeff"],
#         "CountHotPixels": [CountHotPixels, "#_HotPixels"],
#         "AnalyzeOSSmearing": [AnalyzeOSSmearing, "Exp1", "Exp2"],
#         "FluxDeffect": [FluxDeffect, "Flux"],
#     }
#     cat = Table.read(path)
#     output = np.zeros((len(cat["PATH"]), len(dict_functions[function][1:])))

#     for j, key in enumerate(dict_functions[function][1:]):
#         for i, filename in enumerate(cat["PATH"]):
#             verboseprint(filename)
#             output[i, j] = dict_functions[function][0](filename, Plot=Plot)[key]

#         cat[field + "_%i" % (j)] = output[:, j]
#     csvwrite(cat, path)
#     return cat

