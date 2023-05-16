#!/usr/bin/python
import os.path
import glob
from glob import iglob, glob
import numpy as np
#from subprocess import call
import subprocess
from subprocess import Popen, PIPE, STDOUT
import shutil
import glob
import os
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from subprocess import call
from astropy.io import fits
from scipy.optimize import curve_fit

bias_dir = '/'
#dark_dir = '/home/cheng/data/NUVU/211109/w18d7/darks_T300/'
data_dir = '/'


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
    popt, pcov = curve_fit(linefit, x, y, p0 = param)

    a, b = popt

    return (a, b)


#def img_diff(img):

    

def gain_calc():


        data_pattern = data_dir + '*.fits'
        images = sorted(set(glob.glob(data_pattern)))
             
        bias_pattern = bias_dir + '*.fits'
        bias = sorted(set(glob.glob(bias_pattern)))

 

        flatlist = []
        #darklist = []
        biaslist = []
        
        for image in images:
            flatlist.append(image)
            
        
        for img in bias:
            biaslist.append(img)

                  
        gainlist = []
        readnoiselist = []
        signallist = []
        variance = []
        signal = []

        n= 100
        yregions = np.arange(0,1000,n)
        xregions = [1100,1900]
        xregions_b = [1100,1900]
        yregions_b = [500,700]
        x1 = 650
        x2 = 1100
        j = 0
       
        print 'Bias list length is %s.' % (len(biaslist))
 
        for i in range(0,(len(flatlist)),2):
            #print flatlist[i]
            #print flatlist[i+1]
            if i >= len(flatlist)-1:
                break
            for m in np.arange(0,1000,n): #range(len(yregions)):       
                imageb1 = np.array(fits.getdata(biaslist[0])) ##you can change which bias frames to use, decided to use the same frames, it shouldn't make a big diffence
                imageb2 = np.array(fits.getdata(biaslist[1]))
                imageb1 = imageb1*1.0
                imageb2 = imageb2*1.0

                imagef1 = np.array(fits.getdata(flatlist[i]))
                imagef2 = np.array(fits.getdata(flatlist[i+1]))
                imagef1 = imagef1*1.0
                imagef2 = imagef2*1.0
                #print np.mean(imagef1) 


                imagef12 = np.subtract(imagef1,imagef2)
                imageb12 = np.subtract(imageb1,imageb2)
                    
                imagef1_sec = imagef1[m:m+n,xregions[0]:xregions[1]]
                imagef2_sec = imagef2[m:m+n,xregions[0]:xregions[1]]
                imageb1_sec = imageb1[yregions_b[0]:yregions_b[1],xregions_b[0]:xregions_b[1]]
                imageb2_sec = imageb2[yregions_b[0]:yregions_b[1],xregions_b[0]:xregions_b[1]]
 
                imagef12_sec = imagef12[m:m+n,xregions[0]:xregions[1]]
                imageb12_sec = imageb12[yregions_b[0]:yregions_b[1],xregions_b[0]:xregions_b[1]]
            
                sigf1 = np.std(imagef1_sec)
                sigb1 = np.std(imageb1_sec)
                sigf12 = np.std(imagef12_sec)
                sigb12 = np.std(imageb12_sec)
                

                mf1 = np.mean(imagef1_sec) 
                mf2 = np.mean(imagef2_sec)
                mb1 = np.mean(imageb1_sec)
                mb2 = np.mean(imageb2_sec)

                Np = imagef1_sec.size
                average_signal = ((mf1-mb1) + (mf2-mb2))/2
                signal.append(average_signal)
                img_diff = (imagef1_sec-mb1)-(imagef2_sec-mb2)
                var = np.std(img_diff)**2
                variance.append(var/2)

                #printing out the data from flats and biases required for ptc, good idea to check these numbers to make sure frames are matched. Comment out if too much printout
                print mf1,mf2,mb1,mb2,sigb1,sigf1,sigf12,sigb12

                gain = ((mf1+mf2-mb1-mb2)/(sigf12**2))# - sigb12**2))
                #print gain
                readnoise = gain*sigb1
                gainlist.append(gain)
                readnoiselist.append(readnoise)
                signallist.append((mf1-mb1))

        #print signallist
        #print gainlist
        #print readnoiselist

        np.savetxt(data_dir + 'gainlist.txt',[gainlist],delimiter=',',fmt='%.8f')
        np.savetxt(data_dir + 'signallist.txt',[signallist],delimiter=',',fmt='%.8f')
        np.savetxt(data_dir + 'readnoiselist.txt',[readnoiselist],delimiter=',',fmt='%.8f')
        np.savetxt(data_dir + 'variance.txt',[variance],delimiter=',',fmt='%.8f')
        #sig_idx = np.where(np.logical_and(np.array(signallist)>=500, np.array(signallist)<=8000))
        #print 'Mean gainlist indexed %s' % (np.mean(np.array(gainlist)[sig_idx]))
        #print 'Mean readnoise list indexed %s' % (np.mean(np.array(readnoiselist)[sig_idx]))
        #print variance,signal,sigb1
        

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
        fn = data_dir + 'figures/'
        fig_dir = os.path.dirname(fn)
        if not os.path.exists(fig_dir):# create data directory if needed
            os.makedirs(fig_dir)
        plt.savefig(fn + 'ptc_gain.png')


def ptc_plot(variance,signal,sigb1):


        #print np.max(variance)
        rms_noise = np.sqrt(np.array(variance))
       
        idx = np.array(np.argsort([signal]))
        signal_sort = np.sort(signal)
     
        rms_noise_sort = (np.array(rms_noise)[idx])
        rms_noise_sort = rms_noise_sort[0]
        print rms_noise_sort.shape


        #signal_sort = np.array(signal) #np.sort(signal)
        #rms_noise_sort = np.array(rms_noise) #np.array(rms_noise)[idx]
        
        print 'RMS noise array'
        print rms_noise_sort

        log_rms_noise = np.log10(rms_noise_sort)
        log_signal = np.log10(signal_sort)
        print log_rms_noise.shape, log_signal.shape        

        #rn_idx = np.where((rms_noise_sort > 0) & (rms_noise_sort < int(sigb1)))
        #rn_idx = rn_idx[1][:]
        print 'Readnoise in ADU %s' % (int(sigb1))
        print 'Readnoise index for values below the above'
        #print rn_idx
        max_sig = np.where(log_signal == np.nanmax(log_signal)) #for the trendline limits along x axis
        print max_sig
        rnfit = np.linspace(0,np.nanmax(np.array(log_signal)), num=np.array(max_sig[0])[0])
        #RN = (np.array(rms_noise_sort)[0,rn_idx]).mean() - 1
        RN = int(sigb1)
        print 'RN is %s ' % (RN)
        
        readnoise_trendline = 0.00000001*rnfit + RN
        #zeroline = np.ones(1000)*(-1)
        #log_signal_full = log_signal
        log_shot_noise = np.real(np.log10(np.sqrt((rms_noise_sort)**2 - (RN)**2))) #estimated readnoise subtraction
        #filter_shotnoise = np.where(log_shot_noise_full < 1.5)
        #print filter_shotnoise
        log_shot_noise = log_shot_noise #[filter_shotnoise]
        log_signal = log_signal#[filter_shotnoise]
        print log_shot_noise.shape
        print signal_sort.shape

        shot_idx_low = np.where(signal_sort>int(1000)) #have to adjust the limits for the trendline for every ptc
        shot_idx_high = np.where(signal_sort>int(5000))
        shot_idx_low1 = (np.array(shot_idx_low)[0,0])
        shot_idx_high1 = (np.array(shot_idx_high)[0,0])
        print 'Shot noise low value %s' % (shot_idx_low1)
        print 'Shot noise high value %s' % (shot_idx_high1)
        print 'Log of shot noise'
        print log_shot_noise

        maxsig = np.nanmax(log_signal)
        xfit = np.arange(np.log10(0.05),maxsig+(0.5*maxsig),0.005)
        #b = np.array(log_shot_noise)[0,shot_idx_low1:shot_idx_high1]-0.5*np.array(log_signal)[shot_idx_low1:shot_idx_high1]
        b = np.array(log_shot_noise)[shot_idx_low1:shot_idx_high1]-0.5*np.array(log_signal)[shot_idx_low1:shot_idx_high1]
        print b # b is the offset from a linear log plot (y=bx**m), see comments below.
        print 'Start printing parameters..'
        print 'Mean b value %s' % (np.mean(b))
       
        log_signal_fit = np.array(log_signal)[shot_idx_low1:shot_idx_high1]
        #log_shot_noise_fit = np.array(log_shot_noise)[0,shot_idx_low1:shot_idx_high1]
        log_shot_noise_fit = np.array(log_shot_noise)[shot_idx_low1:shot_idx_high1]

        mask = ~np.isnan(log_signal_fit) & ~np.isnan(log_shot_noise_fit)
        nan_idx = np.argwhere(~np.isnan(log_shot_noise_fit))
                
        slope, intercept = fitLine(log_signal_fit[mask],log_shot_noise_fit[mask])
        fit = np.polyfit(log_signal_fit[mask],log_shot_noise_fit[mask],1)

        print 'Shot noise slope = %0.6f.' % (slope)
        print 'Intercept = %0.6f.' % (intercept)
        
        slope_trendline = 0.5*xfit + np.nanmean(b)

        #preamp_gain = ((1-intercept)/0.5)
        #print 10**((1-np.nanmean(b))/0.5)
        #print 10**(intercept)
        #exit()
        preamp_gain = (1.0/10**(np.nanmean(b)))**2 ##this is k gain ##linear equation of a line in log space (base 10). y = b(x**m) which is log10(y) = m(log10(x)) + log10(b)
        readnoise = preamp_gain*RN
        print 'Bias noise is = %0.6f.' % (sigb1)
        print 'Readnoise from bias frame is = %0.6f.' % (preamp_gain*sigb1)
        print 'Conversion gain = %0.6f.' % (preamp_gain)
        print 'Readnoise from plot = %0.6f.' % (readnoise)

        
        plt.rcParams['mathtext.fontset']='stix'
        plt.rcParams['font.family']='STIXGeneral'
        plt.rcParams['font.size']='40'
        plt.xlabel("Signal mean [ADU]",fontsize=25)
        plt.ylabel("Noise[ADU]",fontsize=25)
        plt.grid(b=True, which='major', color='0.75', linestyle='--')
        plt.grid(b=True, which='minor', color='0.75', linestyle='--')
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        
        #plt.ylim([10**(-1),10**2.0])
        #plt.xlim([0,11000])
        #plt.loglog(10**zeroline,10**zeroline,color='black',marker='None',linestyle='--',linewidth=5,label='Zero line')
        plt.loglog(10**log_signal, 10**(log_shot_noise), color='red',marker='x',linestyle='None',markersize=6,label='Shot Noise')
        plt.loglog(10**xfit, 10**(slope_trendline), color='black',linestyle='--',linewidth=2,label='Slope 0.5')
        plt.loglog(10**log_signal, rms_noise_sort,linestyle='-',marker='.',markersize=6,linewidth=1,color='blue',label='Total Noise')
        plt.plot(10**rnfit,readnoise_trendline, color='green',linestyle='-',linewidth=1,label='Readnoise')
        plt.legend(loc=2,prop={'size':12})
        #plt.yscale('log')
        #figtext(.65, .15,'Pre-amp = %0.4f e-/ADU.\nReadnoise = %0.4f e-' % (preamp_gain, readnoise),fontsize=20)
        fig_name = data_dir.replace('/home/','')
        fig_name = fig_name[:-1].replace('/','_')
        fn = data_dir+'figures/'
        fig_dir = os.path.dirname(fn)
        if not os.path.exists(fig_dir):# create data directory if needed
            os.makedirs(fig_dir)
        plt.savefig(fn + fig_name + '.png')
        plt.show()

        return()


def plot_var(variance,signal):

        #variance,signal,sigb1 = gain_calc()
        #print np.max(variance)
        #rms_noise = np.sqrt(np.array(variance))

        idx = np.array(np.argsort([signal]))
        signal_sort = np.sort(signal)

        var_sort = (np.array(variance)[idx])[0]
        rms_sort = np.sqrt(var_sort)

        #plt.close("all")
        #plt.clf()
        #plt.figure(figsize=(20,15))
        #plt.rc('text', usetex=True)
        plt.rcParams['mathtext.fontset']='stix'
        plt.rcParams['font.family']='STIXGeneral'
        plt.rcParams['font.size']='40'
        plt.xlabel("Signal mean [ADU]",fontsize=25)
        plt.ylabel("Noise[ADU]",fontsize=25)
        #grid(b=True, which='major', color='0.75', linestyle='--')
        #grid(b=True, which='minor', color='0.75', linestyle='--')
        plt.tick_params(axis='x', labelsize=25)
        plt.tick_params(axis='y', labelsize=25)
        #plt.axis([0,np.max(n_log),0,bins[bins.size-1]])

        #axes = plt.gca()
        #axes.set_ylim([10**0.1,10**3])
        #axes.set_xlim([0,3000])
        #plt.ylim([10**(0),10**2])
        #plt.xlim([10**0,10**3.5])
        plt.loglog(signal_sort,rms_sort, color='red',marker='x',linestyle='None',markersize=6,label='Shot Noise')
        #plt.loglog(10**xfit, 10**(slope_trendline), color='black',linestyle='--',linewidth=2,label='Slope 0.5')
        #plt.loglog(10**log_signal, 10**(np.log10(rms_noise_sort[0,:])),linestyle='-',marker='.',markersize=6,linewidth=1,color='blue',label='Total Noise')
        #plt.loglog(10**rnfit,readnoise_trendline, color='green',linestyle='-',linewidth=1,label='Readnoise')
        plt.legend(loc=2,prop={'size':12})
        #figtext(.65, .15,'Pre-amp = %0.4f e-/ADU.\nReadnoise = %0.4f e-' % (preamp_gain, readnoise),fontsize=20)
        fig_name = data_dir.replace('/home/','')
        fig_name = fig_name[:-1].replace('/','_')
        fn = data_dir+'figures/'
        fig_dir = os.path.dirname(fn)
        if not os.path.exists(fig_dir):# create data directory if needed
            os.makedirs(fig_dir)
        plt.savefig(fn + fig_name + '.var.png')
        plt.show()

        return()



## ----- Entry point for script -----

if __name__ == "__main__":
    
       
       gain_calc()

