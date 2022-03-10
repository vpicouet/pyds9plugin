# from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD, EMCCDhist
from pyds9plugin.DS9Utils import *#DS9n,PlotFit1D
# from pyds9plugin.DS9Utils import blockshaped
# from pyds9plugin.Macros import functions
#TODO  create histogram for every image
from astropy.table import Column
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
np.seterr(divide = 'ignore') 



if ('FIREBall.py' in __file__) or (function=='execute_command'):
    # print('should plot')
    # matplotlib.use('WX')#Agg #MacOSX
    #'madcosx': valid strings are ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
    # %load_ext autoreload
    # %autoreload 2
    from astropy.table import Table
    d=DS9n()
    fitsfile=d.get_pyfits()
    filename = get_filename(d)
    table=create_table_from_header(filename, exts=[0], info='')#Table(data=[[1]],names=['test'])
    filename=get_filename(d)
else:
    pass
    # matplotlib.use('MacOSX')#Agg #MacOSX
    matplotlib.use('Agg')#Agg #MacOSX
    


# colors= ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8'] + ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8']
full_analysis = True
Plot = True
data = fitsfile[0].data
header = fitsfile[0].header
try:
    date = float(header['DATE'][:4])
except KeyError:
    date = float(header['OBSDATE'][:4])
    
if date<2020:
    conversion_gain = 0.53# ADU/e-  0.53  
    RN=100
    l1,l2 = 1053, 2133
else:
    conversion_gain =  0.22#1 / 4.5# ADU/e-  0.53 
    RN=50
    l1,l2 = -2133, -1053


# header['EMGAIN']=9200
# table['EMGAIN']=9200


analysis_path = os.path.join(os.path.dirname(filename),'analysis/')
if not os.path.isdir(analysis_path):
    os.mkdir(analysis_path) #kills the jobs return none!!


def create_cubsets(table, header):
    try:
        table['EMGAIN>0'] = header['EMGAIN']>0 
        table['EMGAIN==0'] = header['EMGAIN']==0 
        table['EXPTIME==0'] = header['EXPTIME']==0 
        table['EXPTIME>0'] = header['EXPTIME']>0 
    except KeyError:
        pass
    return table

def Create_label_header(header,L):
    if L is not None:
        text = [l + ' = ' + cat[l].astype(str) + '<br>'  for l in L if l in cat.columns]
        return np.sum(text,axis=0)
    else:
        return None
    
def plot_hist(bin_center, n,filename, header, table,masks, conversion_gain=conversion_gain,analysis_path=analysis_path,ax=None,im=None):
    from pyds9plugin.Macros.Fitting_Functions.functions import EMCCD, EMCCDhist
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
    colors= ['r','g','b','orange','#FBC15E','#8EBA42','#FFB5B8'] + ['#E24A33','#348ABD','#988ED5','#777777','#FBC15E','#8EBA42','#FFB5B8']
    f=1.1
    n_conv = 1
    # table['Gain0'] *=1.2
    bias, ron, gain, flux = table['bias_os'],  table['RON_os'], table['Gain0'],table['Flux1'] #table["TopImage"]/table['Gain0']/conversion_gain
    if bias ==0.0:
        bias = table['bias']
    # limit = bias+1000 #After 2000 it is getting very bad
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))  
        save=True
    else:        
        save=False
    # ax = fig.add_subplot(111)
    ax.set_xlabel("Pixel Value [ADU]", fontsize=12)
    ax.set_ylabel("Log(\# Pixels)", fontsize=12)
    try:
        l_data = "%s-%s\nExposure = %i sec\nGain = %i \n" "T det = %0.1f C" % (header['TEMPDATE'],header['TEMPTIME'],header['EXPTIME'], header['EMGAIN'], float(header['TEMPA']))
    except KeyError:
        l_data = ""#"%s-%s\nExposure = %i sec\nGain = %i \n" "T det = %0.1f C" % ("","","", "","")
    # l_model = "Bias = %0.1f DN\n$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10" % (bias,ron, gain, flux )
    # l_model2 = "Bias = %0.1f DN\n$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10" % (bias,ron, table['Gain2'], flux )
    threshold = table['bias_fit']+ 5.5* (table['RON']*conversion_gain)
    # ax.semilogy([threshold,threshold],[0,1e6],'k:')
    # model_to_fit = lambda bin_center,biais\,RN,EmGain,flux,p_sCIC,Smearing
    ax.semilogy(bin_center, np.convolve(n,np.ones(n_conv)/n_conv,mode='same'),'k:',alpha=0.3,  label="Data:\n"+l_data)
    model =            10**EMCCD(    bin_center,bias, ron, gain, flux, sCIC=0,smearing=0)
    # model2 =            10**EMCCD(    bin_center,bias, ron, table['Gain2'], table['Flux2'], sCIC=0,smearing=0)
    model_stochastic = 10**EMCCDhist(bin_center,bias, ron, gain, flux,sCIC=0,smearing=0)
    model_low = 10**EMCCD(bin_center,bias, ron, f*gain, flux/f,sCIC=0,smearing=0)
    model_high = 10**EMCCD(bin_center,bias, ron, gain/f, flux*f,sCIC=0,smearing=0)
    constant = 1#np.nanmax(n)/np.nanmax(model)
    constant_stochastic  = 1#np.nanmax(n)/np.nanmax(model_stochastic)
    model_to_fit =            lambda bin_center, EmGain,flux : EMCCD(    bin_center,bias,ron,EmGain,flux,0,0)+np.log10(constant)
    model_to_fit_cic =            lambda bin_center,EmGain,flux, cic : EMCCD(    bin_center,bias,ron,EmGain,flux,0,cic)+np.log10(constant)
    model_to_fit_all =            lambda  bin_center,  EmGain,flux, cic : EMCCD(    bin_center,bias,ron,EmGain,flux,0,cic)+np.log10(constant)
    model_to_fit_stochastic = lambda bin_center,EmGain,flux : EMCCDhist(bin_center,bias,ron,EmGain,flux,0,0)+np.log10(constant_stochastic)
    for i, (mask,c) in enumerate(zip(masks,colors)):
        model =  10**EMCCD(    bin_center,bias, ron,  table['Gain%i'%(i)],table['Flux%i'%(i)], sCIC=0,smearing=0)
        l_model = "Bias = %0.1f DN\n$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10" % (bias,ron,  table['Gain%i'%(i)],table['Flux%i'%(i)] )
        # a =  PlotFit1D(bin_center[mask & (n>0)],np.log10(value[mask & (n>0)]),deg=1, plot_=False)
        # ax.semilogy(bin_center, 10**a['function'](bin_center), "-",c=c, label="Model (from slope):\n"+l_model)
        # a =  PlotFit1D(bin_center[(bin_center>1500) ],value[(bin_center>1500)],deg='exp', plot_=False,P0=[30,1000,0])#lambda x,a,b:b*np.exp(-a*x)
        # ax.semilogy(bin_center, a['function'](bin_center,*a['popt']), "-",c='k',label=a['popt'][1]/conversion_gain)
        # ax.semilogy(bin_center[masks[i]], 10+5*(i+1)*np.ones(len(bin_center[masks[i]])), '-',c=c)#, label="Mask %i"%(i+1))

    # ax.semilogy(bin_center[masks[1]], 5*np.ones(len(bin_center[masks[1]])), "k-")#, label="Mask %i"%(i+1))
    mask = (bin_center>bias-ron)&(bin_center<np.nanpercentile(im,99.9))&(n>0)
    
    if len(bin_center[mask])==0:
        print(bin_center,n,bin_center[mask],bias-ron,np.nanpercentile(im,99.5))
        print('size of the mask is null')
        mask = (n>0) #np.ones(len(bin_center),dtype=bool)
    
    p0=np.array([float(table['Gain0']),3*float(table["TopImage"]/table['Gain0']/conversion_gain)])#
    p0_all=[float(table['Gain0']),float(table["TopImage"]/table['Gain0']/conversion_gain),0.01]
    popt, pcov = curve_fit(model_to_fit,bin_center[mask],np.log10(n[mask]),p0=p0)
    popt_all, pcov = curve_fit(model_to_fit_all,bin_center[mask],np.log10(n[mask]),p0=p0_all)
    l_fit = "bias = %0.1f e-\n$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10%%" % (bias,ron,popt[0],popt[1])
    l_fit_all = "$\sigma$ = %0.1f e-\nEMg = %0.1f e/e\nF=%0.3fe-$\pm$10%%\nCIC=%0.4fe-/pix" % (ron,popt_all[0],popt_all[1],popt_all[2])

    def change_gain_flux(popt,factor):
        popt1 = list(map(lambda item: popt[1]*factor if item==popt[1] else item, popt))
        popt2 = list(map(lambda item: popt1[0]/factor if item==popt1[0] else item, popt1))
        return popt2 

    # # fit_stochastic_low = 10**model_to_fit_all(bin_center[mask],*change_gain_flux(popt_all,f))
    # # fit_stochastic_high = 10**model_to_fit_all(bin_center[mask],*change_gain_flux(popt_all,1/f))
    ax.semilogy(bin_center[mask], 10**model_to_fit(bin_center[mask],*popt), "b-", label="Least square:\n"+l_fit,lw=1)
    ax.semilogy(bin_center[mask], 10**model_to_fit_all(bin_center[mask],*popt_all), "r:", label="Least square all:\n"+l_fit_all,lw=1)
    # ax.fill_between(bin_center[mask],fit_stochastic_low,fit_stochastic_high,alpha=0.15,color='blue')
    table['gain_ls']=popt[0]
    table['flux_ls']=popt[1]
    table['sCIC_ls']=popt[1]

    ax.legend(loc="upper right", fontsize=10,ncol=2)
    ax.set_ylim(ymin=1e-1,ymax=2.1*n.max())
    ax.set_xlim(xmin=bin_center[n>1].min(),xmax=np.nanpercentile(im,99.999))#limit+500)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_title(os.path.basename(filename).replace(".fits","") )#+ 'Counts image:%0.2f'%())
    plt.show()
    if save:
        fig.savefig(analysis_path + os.path.basename(filename).replace(".fits","_hist.png"))
        fig.savefig('/tmp/' + os.path.basename(filename).replace(".fits","_hist.png"))
        plt.show()
        plt.close()




# if data is None:
#     data = np.nan * np.ones((10,10))
lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
Xinf, Xsup, Yinf, Ysup = 1120, 2100, 1300, 1900#l1, l2, 1, -1
# Xinf, Xsup, Yinf, Ysup = l1, l2, 1, -1
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
table = create_cubsets(table, header)
# table['EMGAIN>0'] = header['EMGAIN']>0 
# table['EMGAIN==0'] = header['EMGAIN']==0 
# table['EXPTIME==0'] = header['EXPTIME']==0 
# table['EXPTIME>0'] = header['EXPTIME']>0 

table['variance_intensity_slope']=0
table['variance_intensity_slope_with_OS']=0

try:
    table["stdX"] = np.nanstd(data[int(Yinf + (Ysup - Yinf) / 2), Xinf:Xsup])
    table["stdY"] = np.nanstd(data[Yinf:Ysup, int(Xinf + (Xsup - Xinf) / 2)])
except IndexError:
    table["stdX"] = np.nanstd(data[int(lx / 2), :])
    table["stdY"] = np.nanstd(data[:, int(ly / 2)])
# value, b = np.histogram(data[Yinf+1000:Ysup, Xinf:Xsup].flatten(),range=(1000,5000),bins=1000)
# range_=(np.nanmedian(physical_region)-500,np.nanmedian(physical_region)+2000)
range_=(np.nanpercentile(physical_region,0.1),np.nanpercentile(physical_region,99.9))
# range_=(1000,5000)
# value, b = np.histogram(data[Yinf+1000:Ysup, Xinf:Xsup].flatten(),range=range_,bins=501)
# value, b = np.histogram(data[Yinf+1000:Ysup, Xinf:Xsup].flatten(),range=(1000,5000),bins=1000)
# value, b = np.histogram(data[Yinf+1000:Ysup, Xinf:Xsup].flatten(),bins=np.arange(range_[0],range_[1],1))
value, b = np.histogram(data[Yinf:Ysup, Xinf:Xsup].flatten(),bins=np.arange(1000,8000,1))
value_os, b_os = np.histogram(pre_scan.flatten(),bins=np.arange(np.nanpercentile(pre_scan,0.1),np.nanpercentile(pre_scan,99.9),1))
bins = (b[1:]+b[:-1])/2
bins_os = (b_os[1:]+b_os[:-1])/2


bias = bins[np.argmax(value)]
bias_os = bins_os[np.argmax(value_os)]
table['bias_os'] = bias_os


np.savetxt("/tmp/xy.txt", np.array([bins, np.log10(value)]).T)
if full_analysis:
    value_to_save, b = np.histogram(data[Yinf:Ysup, Xinf:Xsup].flatten()-bias_os,bins=np.arange(-500,5000,1))
    table['bins'] = Column([(b[1:]+b[:-1])/2], name="bins")   
    table['hist'] = Column([ value_to_save], name="hist")   
    # plt.figure();plt.plot(table['bins'],table['hist'] ,'.');plt.show()
    # table['bins'] = Column([bins], name="bins")   
    # table['hist'] = Column([ value], name="hist")   


# bins,value = bins[value>0],value[value>0]

table['bias'] = bias
mask_RN = (bins>bias-1*RN) & (bins<bias+0.8*RN)  &(value>0)
mask_RN_os = (bins_os>bias_os-1*RN) & (bins_os<bias_os+0.8*RN)  &(value_os>0)
popt = PlotFit1D(bins_os[mask_RN_os],value_os[mask_RN_os],deg='gaus', plot_=False,P0=[1,bias,50,0])['popt']
table['Amp'] =   popt[0]
table['bias_fit'] =   popt[1]
ron = np.abs(PlotFit1D(bins_os[mask_RN_os],value_os[mask_RN_os],deg='gaus', plot_=False,P0=[1,bias,50,0])['popt'][2]/conversion_gain)
if ron == 0.0:
    table['bias_fit'] = table['pre_scan']#bins[0]
table['RON'] =   np.max([40,np.min([ron,120])])
table['RON_os'] =   np.abs(PlotFit1D(bins_os[mask_RN_os],value_os[mask_RN_os],deg='gaus', plot_=False,P0=[1,bias,50,0])['popt'][2]/conversion_gain)
# if table['RON'] == 0.0:
#     table['RON'] =   np.abs(PlotFit1D(bins[mask_RN][:-1],(value[mask_RN][1:]+value[mask_RN][:-1])/2,deg='gaus', plot_=False,P0=[1,bias,50,0])['popt'][2]/conversion_gain)
# if table['bias_fit'] == 0.0:
#     table['bias_fit'] =   PlotFit1D(bins[mask_RN][:-1],(value[mask_RN][1:]+value[mask_RN][:-1])/2,deg='gaus', plot_=False,P0=[1,bias,50,0])['popt'][1]

# np.min([bias,3200])
ron_fixed = table['RON']#np.max([30,np.min([table['RON'],120])])
# mask_gain1 = (bins>np.min([bias,3200])+6*ron_fixed) & (bins<np.min([bias,3200])+30*ron_fixed)
mask_gain0 = (bins>np.min([bias,3200])+6*ron_fixed) & (bins<bins[np.where((bins>bias) & ( np.convolve(value,np.ones(1),mode='same')==0))[0][0]])
mask_gain1 = (bins>np.min([bias,3200])+6*ron_fixed) & (bins<bins[np.where((bins>bias) & ( np.convolve(value,np.ones(2),mode='same')==0))[0][0]])
mask_gain2 = (bins>bias+6*ron_fixed) & (bins<bias+50*ron_fixed) #too dangerous, no values
mask_gain3 = (bins>bias+6*ron_fixed) & (bins<bias+30*ron_fixed) #too dangerous, no values
masks = [mask_gain1,mask_gain3,mask_gain2]#mask_gain0,
for i, mask in enumerate(masks):
    try:
        if header['EMGAIN']==0:
            table['Type'] = 0
            table['Gain%i'%(i)] = 1
            # table['Gain2'] = 1
            table['Flux%i'%(i)] = table["TopImage"] / conversion_gain
            # table['Flux2'] = table["BottomImage"] / conversion_gain
        else:
            if table["TopImage"]>30:
                table['Type'] = 2
                table['Gain%i'%(i)] =   1000
                # table['Gain2'] =   1000
                table['Flux%i'%(i)] =  4*table["TopImage"]/ table['Gain%i'%(i)] /conversion_gain
                # table['Flux2'] =  4* table["TopImage"]/ table['Gain2'] /conversion_gain
            else:
                table['Type'] = 1
                # table['Gain0'] =   -1/np.log(10) / conversion_gain / PlotFit1D(bins[mask & (value>0)],np.log10(value[mask & (value>0)]),deg=1, plot_=False)['popt'][1]
                # table['Gain%i'%(i)] =   -1 / conversion_gain / PlotFit1D(bins[mask & (value>0)],np.log(value[mask & (value>0)]),deg=1, plot_=False)['popt'][1]
                table['Gain%i'%(i)] =  PlotFit1D(bins[ (bins>bias+6*ron_fixed)],value[(bins>bias+6*ron_fixed)],deg='exp', plot_=False,P0=[30,1000,0])['popt'][1]/ conversion_gain
                # print(table['Gain0'], table['Gain2'])
                table['Flux%i'%(i)] =  table["TopImage"]/ table['Gain%i'%(i)] /conversion_gain
                # table['Flux2'] =   table["TopImage"]/ table['Gain2'] /conversion_gain
    except IndexError as e:#(ValueError,RuntimeWarning)
        print(e)
        table['Gain%i'%(i)] = -99
        table['Gain%i'%(i)] = -99
        # table['Flux1'] = -99
        # table['Flux2'] = -99
        a =  PlotFit1D(bin_center[(bin_center>1500) ],value[(bin_center>1500)],deg='exp', plot_=False,P0=[30,1000,0])#lambda x,a,b:b*np.exp(-a*x)
        # ax.semilogy(bin_center, a['function'](bin_center,*a['popt']), "-",c='k',label=None)

cst = 2 if float(table['EMGAIN'])>0 else 1

# fig, ax = plt.subplots()
# ax.plot(intensities,vars_/cst,'.')
# limsx =ax.get_xlim() 
# limsy =ax.get_ylim() 
# popt =  PlotFit1D(intensities,vars_/cst,deg=1,sigma_clip=[3,10], plot_=True,ax=ax)['popt']
# ax.set_ylim(limsy)
# ax.set_xlim(limsx)
# fit.tight_layout()
# fig.savefig(analysis_path + os.path.basename(filename).replace(".fits","_varint.png"))

appertures = blockshaped(physical_region-table['pre_scan'] , 40, 40)
vars_ = np.nanvar(appertures,axis=(1,2))
intensities = np.nanmean(appertures,axis=(1,2))
vars_masked, intensities_masked= SigmaClipBinned(vars_, intensities, sig=1,Plot=False)
popt =  PlotFit1D(vars_masked, intensities_masked/cst,deg=1,sigma_clip=[3,10], plot_=False)['popt']
table['var_intensity_slope'] = popt[1]
table['var_intensity_'] =  popt[0]
if full_analysis:
    table['var_analysis'] = Column([vars_], name="var_analysis")   
    table['intensity_analysis'] = Column([ intensities], name="intensity_analysis")   

if Plot :
    # fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,10))
    # ax2.plot(intensities,vars_/cst,'.',label=popt[1])
    # limsx =ax2.get_xlim() 
    # limsy =ax2.get_ylim() 
    # popt =  PlotFit1D(intensities,vars_/cst,deg=1,sigma_clip=[3,10], plot_=True,ax=ax2)['popt']
    # ax2.set_ylim(limsy)
    # ax2.set_xlim(limsx)
    # ax2.legend()
    # ax2.set_xlabel('Intensity')
    # ax2.set_ylabel('Variance')
    if (header['EMGAIN']>0):
        plot_hist(bins,value,filename, header, table,masks=masks,ax=None,im=physical_region)
    else:
        table['gain_ls']=0
        table['flux_ls']=0
        table['sCIC_ls']=0
    # fig.tight_layout()
    # fig.savefig(analysis_path + os.path.basename(filename).replace(".fits","_varint.png"))


table['sCIC_OS'] = (table['post_scan'] - table['pre_scan'] )/ table['Gain0'] /conversion_gain

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




if ('FIREBall.py' in __file__) or (function=='execute_command'):
    print(table)


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

