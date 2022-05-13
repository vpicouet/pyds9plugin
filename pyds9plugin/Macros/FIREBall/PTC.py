#!/usr/bin/python
import os.path
import glob
from glob import iglob, glob
import pyfits
import numpy as np
from pylab import load, save
from subprocess import call
import shutil
import glob
import os
import matplotlib.mlab as mlab
from pylab import *
import matplotlib.pylab as plt
from subprocess import call
#from scipy.stats import nanmean
from astropy.io import fits
from scipy.optimize import curve_fit

# flatlist = glob.glob('/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/ptc_T183_1MHz_405nm/image0000[345]*.fits')
# biaslist = glob.glob('/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/bias_T183_1MHz/image000*.fits')
# flatlist.sort()
# biaslist.sort()


def linefit(x, A, B):

    """
    Generate a line.

    Parameters
    ----------
    x : numpy array
        Linear vector

    a : float
        Line slope

    b : float
        Line intercept

    Returns
    -------
    Line function.
    """

    return A*x + B

def fitLine(x, y, param = None):
    """
    Fit a line.

    Parameters
    ----------
    x : numpy array
        One dimensional vector.

    y : numpy array
        One dimensional vector.

    param : list
       List of guess parameters
    """
    popt, pcov = curve_fit(linefit, x[np.isfinite(y)], y[np.isfinite(y)], p0 = param)

    a, b = popt

    return (a, b)



def gain_calc():
    # bias_dir = '/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/bias_T183_1MHz/'
    # # dark_dir = '/home/cheng/data/NUVU/211109/w18d7/darks_T300/'
    # data_dir = '/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/ptc_T183_1MHz_405nm/'

    # data_pattern = data_dir + 'image0*.fits'
    # data_pattern = data_dir + 'image0*[345]?.fits'
 
    # images = sorted(set(globglob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/ptc_T183_1MHz_405nm/image0000[21-60].fits")))
    # bias = sorted(set(glob.glob('/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/W17D13/bias_T183_1MHz/image0*.fits')))


    # images = sorted(set(globglob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/20220508/PTC/to_keep/image_000000[57-81].fits")))

    
    # bias = sorted(set(globglob('/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/20220508/PTC/image_000000[6-25].fits')))
    images = sorted(set(glob.glob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220509/PTC_phonelight/image000*.fits")))
    bias = sorted(set(glob.glob("/Users/Vincent/Nextcloud/LAM/FIREBALL/2022/DetectorData/220509/PTC_phonelight/image000*.fits")))

    flatlist = []
    #darklist = []
    biaslist = []

    print(len(images))

    for image in images:
        flatlist.append(image)
        
    for img in bias:
        biaslist.append(img)

    data_dir = os.path.commonpath(flatlist)
      
    gainlist = []
    readnoiselist = []
    signallist = []
    variance = []
    signal = []
    #print biaslist
    #print flatlist

    #exit()
    n = 100
    yregions = np.arange(200,1400,n)
    xregions = [1200,1900]
    xregions_b = [1200,1900]
    xregions_b = [100,1000]
    x1 = 1300
    x2 = 1500
    j = 0
   
    print(len(biaslist))

    for i in range(0,(len(flatlist)),5):
        #print i,j
        if i >= len(flatlist)-1:
            break
        for l in range(0,1):
            for m in range(len(yregions)):       
                if j >= len(biaslist)-2:
                    j = 0
                imageb1 = np.array(fits.getdata(biaslist[j+0])) ##modified from j to i index for one data set
                imageb2 = np.array(fits.getdata(biaslist[j+1]))
                imageb1 = imageb1*1.0
                imageb2 = imageb2*1.0

                imagef1 = np.array(fits.getdata(flatlist[i+0]))
                imagef2 = np.array(fits.getdata(flatlist[i+1]))
                imagef1 = imagef1*1.0
                imagef2 = imagef2*1.0
                #print np.mean(imagef1) 


                imagef12 = np.subtract(imagef1,imagef2)
                imageb12 = np.subtract(imageb1,imageb2)
            
                #print np.mean(imagef12)
                #print np.mean(imageb12)
                #print np.max(imaged12)

                imagef1_sec = imagef1[yregions[m]:yregions[m]+n,xregions[l]:xregions[l+1]]
                imagef2_sec = imagef2[yregions[m]:yregions[m]+n,xregions[l]:xregions[l+1]]
                imageb1_sec = imageb1[yregions[m]:yregions[m]+n,xregions_b[l]:xregions_b[l+1]]
                imageb2_sec = imageb2[yregions[m]:yregions[m]+n,xregions_b[l]:xregions_b[l+1]]

                imagef12_sec = imagef12[yregions[m]:yregions[m]+n,xregions[l]:xregions[l+1]]
                imageb12_sec = imageb12[yregions[m]:yregions[m]+n,xregions_b[l]:xregions_b[l+1]]
        
                sigf1 = np.std(imagef1_sec)
                sigb1 = np.std(imageb1_sec)
                sigf12 = np.std(imagef12_sec)
                sigb12 = np.std(imageb12_sec)
                #print sigb1
                #print sigf1
                #print imagef1_sec[1,0]
                #print 'space1'
                #print imagef2_sec[1,0]
                #print 'space2'
                #print imagef12_sec[1,0]

                mf1 = np.mean(imagef1_sec) 
                mf2 = np.mean(imagef2_sec)
                mb1 = np.mean(imageb1_sec)
                mb2 = np.mean(imageb2_sec)

                Np = imagef1_sec.size
                average_signal = ((mf1-mb1) + (mf2-mb2))/2
                signal.append(average_signal)
                var1 = (imagef1_sec-mb1)-(imagef2_sec-mb2)
                var2 = np.power(var1,2)
                variance.append(var2.sum()/(2*Np))


                #print mf1,mf2,mb1,mb2,sigf12,sigb12

                gain = ((mf1+mf2-mb1-mb2)/(sigf12**2))# - sigb12**2))
                readnoise = gain*sigb1
                gainlist.append(gain)
                readnoiselist.append(readnoise)
                signallist.append((mf1-mb1))
        j = j + 2

    np.savetxt(data_dir + '/gainlist.txt',[gainlist],delimiter=',',fmt='%.8f')
    np.savetxt(data_dir + '/signallist.txt',[signallist],delimiter=',',fmt='%.8f')
    np.savetxt(data_dir + '/readnoiselist.txt',[readnoiselist],delimiter=',',fmt='%.8f')
    sig_idx = np.where(np.logical_and(np.array(signallist)>=30000, np.array(signallist)<=55000))
    print(np.mean(np.array(gainlist)[sig_idx]))
    print(np.mean(np.array(readnoiselist)[sig_idx]))


    ptc_plot(variance,signal,sigb1)


def gain_plot(signallist,gainlist,sigb1):        

    plt.close("all")
    plt.clf()
    #figure(figsize=(30,15))
    #plt.rc('text', usetex=True)
    #gca().set_position((.1, .3, .8, .65))
    plt.rcParams['mathtext.fontset']='stix'
    plt.rcParams['font.family']='STIXGeneral'
    plt.rcParams['font.size']='40'
    #plt.rc('font',**{'family':'sans-serif','sans-serif':['Times'],'size': 50})
    plt.plot(np.array(signallist),np.array(gainlist),color="black",linestyle="None",marker='.',markersize=20,label='Conversion Gain')
    plt.xlabel("Frame #",fontsize=35)
    plt.ylabel("Conversion Gain [e/ADU]",fontsize=35)
    grid(b=True, which='major', color='0.75', linestyle='--')
    grid(b=True, which='minor', color='0.75', linestyle='--')
    plt.ylim([-5,25])
    plt.show()
    fn = data_dir + '/figures/'
    fig_dir = os.path.dirname(fn)
    if not os.path.exists(fig_dir):# create data directory if needed
        os.makedirs(fig_dir)
    plt.savefig(fn + 'w9d3_gain.png')


def ptc_plot(variance,signal,sigb1):
#%%
    #print np.max(variance)
    rms_noise = np.sqrt(np.array(variance))

    idx = np.array(np.argsort([signal]))
    signal_sort = np.sort(signal)

    rms_noise_sort = np.array(rms_noise)[idx]
    print(rms_noise_sort)
    print(signal_sort)

    log_rms_noise = np.log10(rms_noise_sort)
    log_signal = np.log10(signal_sort)
    rn_idx = np.where((rms_noise_sort > 0) & (rms_noise_sort < 12))
    rn_idx = rn_idx[1][:]
    max_sig = np.where(log_signal == np.nanmax(log_signal))
         
    rnfit = np.linspace(0,np.nanmax(np.array(log_signal)), num=np.array(max_sig[0])[0])
    RN_raw = (np.array(rms_noise_sort)[0,rn_idx]).mean()
    
    RNoffset = 0
    RN = RN_raw + RNoffset
    print(RN)
    val_max =3300
    readnoise_trendline = 0.00000001*rnfit + RN_raw

    log_shot_noise = np.real(np.log10(np.sqrt((rms_noise_sort)**2 - (RN)**2)))
    shot_idx_low = np.where(signal_sort>int(val_max))
    shot_idx_high = np.where(signal_sort>int(40000))
    shot_idx_low1 = (np.array(shot_idx_low)[0,0])
    shot_idx_high1 = -1#(np.array(shot_idx_high)[0,0])
    #print shot_idx_low1
    #print shot_idx_high1
 
    maxsig = np.nanmax(log_signal)
    xfit = np.arange(log10(0.05),maxsig+(0.5*maxsig),0.005)
    b = np.array(log_shot_noise)[0,shot_idx_low1:shot_idx_high1]-0.5*np.array(log_signal)[shot_idx_low1:shot_idx_high1]
    print('Start printing parameters..')
    print(np.mean(b))
   
    print(np.array(log_signal)[shot_idx_low1:shot_idx_high1])
    print(np.array(log_shot_noise)[0,shot_idx_low1:shot_idx_high1])

    slope, intercept = fitLine(np.array(log_signal)[shot_idx_low1:shot_idx_high1], np.array(log_shot_noise)[0,shot_idx_low1:shot_idx_high1])


    print('Shot noise slope = %0.6f.' % (slope))
    print('Intercept = %0.6f.' % (intercept))
            
    slope_trendline = 0.5*xfit + np.nanmean(b[np.isfinite(b)])
    preamp_gain = (10**(np.nanmean(b[np.isfinite(b)])*(-1)))
    readnoise = preamp_gain*RN_raw
    print('Bias noise is = %0.6f.' % (sigb1))
    print('Readnoise from bias frame is = %0.6f.' % (preamp_gain*sigb1))
    print('Conversion gain = %0.6f.' % (preamp_gain))
    print('Readnoise from plot = %0.6f.' % (readnoise))

    xmin = np.log10(np.nanmin(10**log_signal)+1)
    xmax = np.log10(np.nanmax(10**log_signal)+1)
    ymin = np.log10(np.nanmin(10**log_shot_noise[0,:])+1)
    ymax = np.log10(np.nanmax(10**log_shot_noise[0,:])+1)

    #print xmin,xmax,ymin,ymax
   
    #plt.close("all")
    #plt.clf()
    #plt.figure(figsize=(20,15))
    #plt.rc('text', usetex=True)
    # plt.rcParams['mathtext.fontset']='stix'
    # plt.rcParams['font.family']='STIXGeneral'
    # plt.rcParams['font.size']='40'
    plt.title('Conversion gain = %0.1f, RN=%0.1f'%(preamp_gain,preamp_gain*sigb1))
    plt.xlabel("Signal mean [ADU]",fontsize=12)
    plt.ylabel("Noise[ADU]",fontsize=12)
    grid(b=True, which='major', color='0.75', linestyle='--')
    grid(b=True, which='minor', color='0.75', linestyle='--')
    plt.tick_params(axis='x')#, labelsize=25)
    plt.tick_params(axis='y')#, labelsize=25)
    #plt.axis([0,np.max(n_log),0,bins[bins.size-1]])

    #axes = plt.gca()
    #axes.set_ylim([10**0.1,10**3])
    #axes.set_xlim([10**0,10**5])
    #plt.ylim([10**0,10**1.3])
    #plt.xlim([10**xmin,10**ymax])
    plt.loglog(10**log_signal, 10**(log_shot_noise[0,:]), color='red',marker='.',markersize=8,label='Shot Noise')
    plt.loglog(10**xfit, 10**(slope_trendline), color='black',linestyle='--',linewidth=2,label='Slope 0.5')
    plt.loglog(10**log_signal, 10**(np.log10(rms_noise_sort[0,:])),'.',linewidth=0,color='blue',label='Total Noise')
    plt.loglog(10**rnfit,readnoise_trendline, color='green',linestyle='-',linewidth=1,label='Readnoise')
    plt.legend(loc=2,fontsize=10)#,prop={'size':20})
    #figtext(.65, .15,'Pre-amp = %0.4f e-/ADU.\nReadnoise = %0.4f e-' % (preamp_gain, readnoise),fontsize=20)
    # fig_name = data_dir#.replace('/home/cheng/data/NUVU/','')
    # fig_name = fig_name[:-1].replace('/','_')
    # fn = data_dir+'/figures/'
    # fig_dir = os.path.dirname(fn)
    # if not os.path.exists(fig_dir):# create data directory if needed
    #     os.makedirs(fig_dir)

    plt.savefig('/tmp/ptc.png')
    plt.show()
#%%
    return



## ----- Entry point for script -----

if __name__ == "__main__":
    gain_calc()

