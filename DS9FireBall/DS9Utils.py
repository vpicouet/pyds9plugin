#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:13 2018

@author: Vincent
"""

from __future__ import print_function, division
import timeit
import glob
import os
import sys
import numpy as np
from pyds9 import DS9
import datetime
from  pkg_resources  import resource_filename
from astropy.table import Table
try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()

import matplotlib; matplotlib.use('TkAgg')  
import matplotlib.pyplot as plt

#import tkinter as tk
#root = tk.Tk()
width = 23#root.winfo_screenmmwidth() / 25.4
height = 14#root.winfo_screenmmheight() / 25.4

IPython_default = plt.rcParams.copy()
plt.rcParams['figure.figsize'] = (2*width/3, 3*height/5)
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['axes.grid'] = True
plt.rcParams['image.interpolation'] = None
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'x-large'





DS9_BackUp_path = os.environ['HOME'] + '/DS9BackUp/'
def CreateFolders(DS9_BackUp_path=DS9_BackUp_path):
    if not os.path.exists(os.path.dirname(DS9_BackUp_path)):
        os.makedirs(os.path.dirname(DS9_BackUp_path))
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/Plots'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/Plots')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/DS9Regions'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/DS9Regions')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/DS9Curves'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/DS9Curves')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/CSVs'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/CSVs')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/Images'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/Images')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/config'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/config')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/CreatedImages'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/CreatedImages')
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + '/subsets'):
        os.makedirs(os.path.dirname(DS9_BackUp_path)+ '/subsets')
    return

class config(object):
    """
    """
    def __init__(self,path):
        """
        """
        config = Table.read(path,format='csv')
        self.exptime = np.array(config[config['param']=='exptime']['value'].data[0].split('-'),dtype=str)
        self.temperature = np.array(config[config['param']=='temperature']['value'].data[0].split('-'),dtype=str)
        self.gain = np.array(config[config['param']=='gain']['value'].data[0].split('-'),dtype=str)
        self.physical_region = np.array(config[config['param']=='physical_region']['value'].data[0].split('-'), dtype=int)
        self.extension = config[config['param']=='extension']['value'].data[0]
        self.verbose = config[config['param']=='verbose']['value'].data[0]
        self.format_date = config[config['param']=='format_date']['value'].data[0]
        return 
    
try:
    conf_dir = resource_filename('DS9FireBall', 'config')
except:
    conf_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
try:
    my_conf = config(path=conf_dir+'/config.csv')
except (IOError or FileNotFoundError) as e:
    print(e)
    pass

    

def ChangeConfig(xpapoint):
    from astropy.table import Table
    exptime, temperature, gain, physical_region, extension, date, format_date, verbose = sys.argv[3:]
    try:
        conf_dir = resource_filename('DS9FireBall', 'config')
    except:
        conf_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
    my_conf = Table.read(conf_dir+'/config.csv', format='csv')
    my_conf['value'][my_conf['value']=='exptime'] = exptime
    my_conf['value'][my_conf['param']=='temperature'] = temperature
    my_conf['value'][my_conf['param']=='gain'] = gain
    my_conf['value'][my_conf['param']=='physical_region']  = physical_region
    my_conf['value'][my_conf['param']=='extension']  = extension
    my_conf['value'][my_conf['param']=='date']   = date
    my_conf['value'][my_conf['param']=='format_date'] = format_date
    my_conf['value'][my_conf['param']=='verbose'] = verbose
    csvwrite(my_conf,conf_dir+'/config.csv')
    return my_conf
    
    
    
def DS9createSubset(xpapoint, cat=None, number=2,dpath=DS9_BackUp_path+'subsets/', config=my_conf):
    """
    """
    from astropy.table import Table
    try:
        cat, number, gain, folder, exp, temp = sys.argv[3:3+6]
        gain, folder, exp, temp = np.array(np.array([gain, folder, exp, temp],dtype=int),dtype=bool)
        print('gain, folder, exp, temp = ',gain, folder, exp, temp)
    except:
        pass
    
    vgain, vdate, vexp, vtemp = sys.argv[-4:]
    print('vgain, vdate, vexp, vtemp =',vgain, vdate, vexp, vtemp)

    try:
        cat = Table.read(cat)
    except:
        print('Impossible to open table, verify your path.')
    print(cat)
    print(cat['FPATH'])
    
    fieldTemp = 'EMCCDBAC'
    fieldExp = my_conf.exptime[0]
    fieldGain = my_conf.gain[0]
    fieldDate = 'date'

    EMGAIN = Transform(vgain)
    EXPTIME = Transform(vexp)
    TEMP = Transform(vtemp)
    date = Transform(vdate)
    t=cat.copy()
    index_hist = ReturnIndex(t, fields=[fieldGain,fieldExp, fieldTemp, fieldDate],values=[EMGAIN,EXPTIME,TEMP, date])
    cat = cat[index_hist]

    gains = cat[fieldGain]
    fpath = cat['FPATH']
    exps = cat[fieldExp]
    temps = np.round(cat[fieldTemp].astype(float),1)
    if gain:
        gains = np.unique(gains)
    if folder:
        fpath = np.unique(fpath)
        if type(fpath)==str:
            fpath = [fpath] 
    if exp:
        exps = np.unique(cat[fieldExp])
    if temp:
        temps = np.unique(temps)
    path_date = dpath+datetime.datetime.now().strftime("%y%m%d_%HH%Mm%S")
    
    if not os.path.exists(path_date):
        os.makedirs(path_date)       
        
    print(fpath)
    print(gains)
    print(exps)
    print(temps)
    for path in np.unique(fpath):
        print(path)
        gainsi = np.unique(cat[([path in cati['PATH'] for cati in cat]) ][fieldGain])
        print(gainsi)
        for gain in np.unique(gainsi):
            expsi = np.unique(cat[([path in cati['PATH'] for cati in cat])  & (cat[fieldGain] == gain) ][fieldExp])
            print(expsi)
            for exp in np.unique(expsi):
                tempsi = np.unique(cat[([path in cati['PATH'] for cati in cat])  & (cat[fieldGain] == gain) & (cat[fieldExp] == exp)][fieldTemp])
                print(tempsi)
                tempsi = np.round(tempsi,1)
                for temp in np.unique(tempsi):
                    print('GAIN = %i, exposure  = %i, temp = %0.2f and path=%s'%(float(gain), float(exp), float(temp), path))
                    files = cat[([path in cati['PATH'] for cati in cat])  & (cat[fieldGain] == gain) & (cat[fieldExp] == exp) & (np.round(cat[fieldTemp].astype(float),1) == temp) ]['PATH']
                    folders = cat[([path in cati['PATH'] for cati in cat])  & (cat[fieldGain] == gain) & (cat[fieldExp] == exp) & (np.round(cat[fieldTemp].astype(float),1) == temp) ]['FPATH']
                    #print(files)
                    if len(files) > 0:
                        for folder,file in zip(folders[-int(number):], files[-int(number):]):
                            try:
                                os.makedirs(path_date+'/%s/%s'%(os.path.basename(os.path.dirname(folder)),os.path.basename(folder)))  
                            except FileExistsError:
                                pass
                            print('Copying file',os.path.basename(file))
                            symlink_force(file,path_date+'/%s/%s/%s'%(os.path.basename(os.path.dirname(folder)),os.path.basename(folder),os.path.basename(file)))
    return

def Transform(value):
    if value == '-':
        result = None
    elif len(value.split(',')) == 2:
        n1, n2 = value.split(',')
        n1, n2 = int(float(n1)), int(float(n2))
        n1, n2 = np.min([n1,n2]), np.max([n1,n2])
        result = n1, n2
    elif value.isdigit():
        result = int(float(value))
    elif len(value.split(',')) > 2:
        result = list(np.array(np.array(value.split(','),dtype=float), dtype=int))
    return result
    

def CreateSubCatalog(table0, field, value):
    """
    """
    #print(globals())
    table = table0.copy()
    #print(globals())
    value = value#globals()[field]
    print(field, ' = ', value)
    if field in table.colnames:
        if value is None:
            return table, True      
        if type(value) == tuple:
            index = (table[field]>=min(value)) & (table[field]<=max(value))
        elif type(value) == list:
            indexes = [table[field] == val for val in value]
            index = False
            for ind in indexes:
                index = np.ma.mask_or(index, ind)#index.data or ind.data
        elif (type(value) == int) or (type(value) == float)or (type(value) == str):
            index = table[field] == value
        return table[index], index
    else:
        return table, True


def ReturnIndex(table, fields, values):
    index = np.full(len(table), True, dtype=bool)
    for field, value in zip(fields, values):
        #print(t[fields])
        t1, ind = CreateSubCatalog(table0=table, field=field, value=value)
        if type(ind) is not bool:
            #index = np.ma.mask_or(index, ind)
            index = [all(tup) for tup in zip(index, ind)]
            print('%0.1f %% of the table kept: [%i / %i]'%(100*np.sum(index*1)/len(index), np.sum(index*1),len(index)))
    #print(t[index][my_conf.gain[0],my_conf.exptime[0],'TEMP'])   
    for field in fields:
        print(np.unique(table[index][field]))
    return index 

    

def PlotFit1D(x=None,y=[709, 1206, 1330],deg=1, Plot=True, sigma_clip=None, title=None, xlabel=None, ylabel=None):
    """ PlotFit1D(np.arange(100),np.arange(100)**2 + 1000*np.random.poisson(1,size=100),2)
    """
    #ajouter exp, sqrt, power low, gaussian, 
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    if x is None:
        x = np.arange(len(y))
    x = np.array(x)
    y = np.array(y)


    if sigma_clip is not None:
        index = (x > np.nanmean(x) - sigma_clip[0] * np.nanstd(x)) & (x < np.nanmean(x) + sigma_clip[0] * np.nanstd(x)) & (y > np.nanmean(y) - sigma_clip[1] * np.nanstd(y)) & (y < np.nanmean(y) + sigma_clip[1] * np.nanstd(y))
        x, y = x[index], y[index]
    if Plot:
        fig = plt.figure()#figsize=(10,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=(4,1))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        #fig, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(10,6),height_ratios=[4,1])
        ax1.plot(x, y, '.',label='Data')
    xp = np.linspace(x.min(),x.max(), 1000)

    if type(deg)==int:
        z = np.polyfit(x, y, deg)
        popt = np.poly1d(z)
        zp = popt(xp)
        zz = popt(x)
        degs = [' %0.2f * x^%i'%(a,i) for i,a in enumerate(popt.coef[::-1])]
#        name = 'Fit: ' + '+'.join(popt.coef.astype(int).astype(str))
        name = 'Fit: ' + '+'.join(degs)
    else:
        from scipy.optimize import curve_fit
        if deg=='exp':
            law = lambda x, b, a, offset:  b*np.exp(-x/a) + offset
            P0 = [np.nanmax(y)-np.nanmin(y),1,np.nanmin(y)]
        elif deg=='gaus':
            law = lambda x, a, xo, sigma, offset: a**2 * np.exp(-np.square((x-xo) / sigma) / 2) + offset
            P0 = [np.nanmax(y)-np.nanmin(y),x[np.argmax(y)],np.std(y),np.nanmin(y)]
        elif deg=='power':
            law = lambda x, amp, index, offset: amp * (x**index) + offset
            P0 = None
        popt, pcov = curve_fit(law, x, y, p0=P0)
        zp = law(xp,*popt)
        zz = law(x,*popt)
        name = 'Fit %s'%(popt)
        #plt.plot(xp, , '--', label='Fit: ')
    if Plot:
        if title:
            fig.suptitle(title,y=1)
        if xlabel:
            ax2.set_xlabel(xlabel)
        if ylabel:
            ax1.set_ylabel(ylabel)
        ax1.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
        ax2.set_ylabel('Error')
        ax1.plot(xp, zp, '--', label=name)
        ax2.plot(x, y-zz, 'x', label=name)
        ax1.grid(linestyle='dotted');ax2.grid(linestyle='dotted')
        ax1.legend()
        plt.tight_layout()
        plt.show()
    return popt

#PlotFit1D(a[(a[my_conf.gain[0]]==gain)&(a[my_conf.exptime[0]]>5)][my_conf.exptime[0]],a[(a[my_conf.gain[0]]==gain)&(a[my_conf.exptime[0]]>5)]['MeanADUValue']/a[(a[my_conf.gain[0]]==gain)&(a[my_conf.exptime[0]]>5)][my_conf.exptime[0]],sigma_clip=[10,4]) 
#PlotFit1D(b[(b[my_conf.gain[0]]==gain)&(b[my_conf.exptime[0]]>5)][my_conf.exptime[0]],b[(b[my_conf.gain[0]]==gain)&(b[my_conf.exptime[0]]>5)]['MeanADUValue']/b[(b[my_conf.gain[0]]==gain)&(b[my_conf.exptime[0]]>5)][my_conf.exptime[0]]) 


def TimerSMS(start, hour=1, message='Hello my friend, have a beer your super long code just finished... Now I hope it did not crash'):
    import urllib, timeit
    stop = timeit.default_timer()
    if stop - start > hour * 3600:
        urllib.request.urlopen('https://smsapi.free-mobile.fr/sendmsg?user=26257004&pass=AHepwpHliwFHET&msg=%s'%(message))  
        return 'SMS sent'
    else:
        return 'Code ran fast, SMS not sent'


def create_DS9regions(xim, yim, radius=20, more=None, save=True, savename="test", form=['circle'], color=['green'], ID=None, verbose=False):#of boxe
    """Returns and possibly save DS9 region (circles) around sources with a given radius
    """
    
    regions = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image
    """
    if type(radius) == int:
        r, r1 = radius, radius
    else:
        r , r1 = radius
    #r = radius   
    #print(range(len(xim)))
    for i in range(len(xim)):
        print(form[i])
        if form[i] == 'box':
            #print('Box')
            rest = '{:.2f},{:.2f})'.format(r, r1)
            rest += ' # color={}'.format(color[i])
        elif form[i] =='circle':
            #print('Circle')
            rest = '{:.2f})'.format(r)
            rest += ' # color={}'.format(color[i])
        elif form[i] == 'bpanda':
            #print('Bpanda')
            rest = '0,360,4,0.1,0.1,{:.2f},{:.2f},1,0)'.format(r, 2*r)
            rest += ' # color={}'.format(color[i])
        elif form[i] == 'annulus':
            #print('Annulus')
            rtab = np.linspace(0, r, 10)
            rest = '{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f})'.format(*list(rtab))  
            rest += ' # color={}'.format(color[i])
        try:
            for j, (x, y) in enumerate(np.nditer([xim[i], yim[i]])):
                if form[0] == 'ellipse':
                    rest = '{:.2f},{:.2f},{:.2f})'.format(more[0][j],more[1][j],more[2][j])
                    rest += ' # color={}'.format(color[i])
                regions += '{}({:f},{:f},'.format(form[i], x+1, y+1) + rest
                if ID is not None:
                    regions += ' text={{{}}}'.format(ID[i][j])
                regions +=  '\n'  
        except ValueError:
            pass

    if save:
        with open(savename+'.reg', "w") as text_file:
            text_file.write(regions)        
        verboseprint(('region file saved at: ' +  savename + '.reg'),verbose=verbose)
        return 



def create_DS9regions2(xim, yim, radius=20, more=None, save=True, savename="test",text=0, form='circle', color='green', verbose=False):#of boxe
    """Returns and possibly save DS9 region (circles) around sources with a given radius
    """
    regions = """
        # Region file format: DS9 version 4.1
        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        image
        """
    #print ('form = ' + form )
    if form != 'ellipse':
        if type(radius) == int:
            r1, r2 = radius, radius
        else:
            r1 , r2 = radius
        if form == 'box':
            rest = ',%.2f,%.2f,%.2f) # color=%s'%(r1,r2,0,color)
        if form == 'circle':
            rest = ',%.2f) # color=%s'%(r1,color)
        if form == 'bpanda':
            rest = ',0,360,4,0.1,0.1,%.2f,%.2f,1,0) # color=%s'%(r1,2*r1,color)
        if form == 'annulus':
            n=10
            radius = np.linspace(0,r1,n)
            rest = ',%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) # color=%s'%(radius[0],radius[1],radius[2],radius[3],radius[4],radius[5],radius[6],radius[7],radius[8],radius[9],color)  
         
        for i, x, y in zip(np.arange(len(xim)),xim, yim):
            if form == '# text':
                rest = ') color=%s text={%s}'%(color,text[i])          #regions += """\ncircle({},{},{})""".format(posx, posy, radius)
            regions +="""\n%s(%.2f,%.2f"""%(form,x+1,y+1) + rest
        
        
        
    if form == 'ellipse':
        for i in range(len(more[0])):
            rest = ',%.2f,%.2f,%.2f) # color=%s'%(more[0][i],more[1][i],more[2][i],color)                
            regions += """\n%s(%.2f,%.2f"""%(form,xim[i]+1,yim[i]+1) + rest
#    try: 
#        r = pyregion.parse(regions)
#    except ValueError or SyntaxError:
#        print(("Problem with the region" )  )
#        return regions
#    print regions
    if save:
        verboseprint('Saving region file',verbose=verbose)
        with open(savename+'.reg', "w") as text_file:
            text_file.write("{}".format(regions))        
#        r.write(savename+'.reg') 
#        np.savetxt(savename+'.reg',regions)
        verboseprint(('Predicted region file saved at: ' +  savename + '.reg'),verbose=verbose)
        return 

    
def DS9MultipleThreshold(xpapoint, config=my_conf):
    """
    """
    from astropy.io import fits
    #import matplotlib; matplotlib.use('TkAgg')  
    #import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    filename = d.get("file")
    try:
        region = getregion(d)
    except ValueError:
        Xinf, Xsup, Yinf, Ysup = my_conf.physical_region#[0,2069,1172,2145]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    image_area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)

    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    d = DS9(xpapoint)
    fitsimage = d.get_fits()[0]
    imasum_noprior = np.zeros(fitsimage.data.shape[:])
    imasum_prior = np.zeros(fitsimage.data.shape[:])
    
    for fn in path:
        print(fn)
        with fits.open(fn) as f:
            ima = f[0].data#.T
            hdr = f[0].header
    
        ima_count_noprior = process_image(ima, hdr)
        ima_count_prior = process_image(ima, hdr, with_prior=True)

        imasum_noprior += ima_count_noprior
        imasum_prior += ima_count_prior
        fitsimage.data = imasum_noprior
        fitswrite(fitsimage,fn[:-5] + '_sum_no_prior.fits')
        fitsimage.data = imasum_noprior
        fitswrite(fitsimage,fn[:-5] + '_sum_with_prior.fits')
    if len(path) == 1:
        d.set('frame new')
        d.set('file ' + fn[:-5] + '_sum_no_prior.fits')  
        d.set('frame new')
        d.set('file ' + fn[:-5] + '_sum_with_prior.fits')  
    return


def normalgamma(nooe, g, r , lamb):
    """
    """
    from scipy.special import hyp1f1, gamma
    lamb2 = lamb/2.
    rsq = r**2
    aux = np.square(-g*nooe + rsq)/(2*g**2*rsq)
    res = 2**(-lamb2) * np.exp(- np.square(nooe)/(2*rsq)) * g**(-lamb) * r**(lamb - 2)
    res *= r  * hyp1f1(lamb2, .5, aux) / np.sqrt(2)*gamma((lamb + 1)/2) + \
            (g*nooe - rsq)*hyp1f1((lamb+1)/2, 3/2, aux) / (g*gamma(lamb2))
    
    return res



def bias_from_right_overscan(ima, left=2135, right=2580):
    """
    """
    from astropy.stats import sigma_clipped_stats
    overscan = ima[left:right,:] 
    bias, median, stdev = sigma_clipped_stats(overscan, mask=np.isnan(overscan), axis=0)
    ima_bias_cor = ima - bias
    return ima_bias_cor, bias, stdev


def plot_histo_bias(data, mean, median, stdev):
    """
    """
    from matplotlib import pyplot as plt    
    # full stat
    fmean = np.nanmean(data)
    fmed = np.nanmedian(data)
    # fitted gaussian
    plt.figure()
    hits, bins, _ = plt.hist(data, bins=100, range=[300,3700])
    binsize = bins[1] - bins[0]
    x = np.linspace(3000, 3500)
    gaussian = 1./(np.sqrt(2*np.pi)*stdev)*np.exp(-.5*np.square(x-mean)/np.square(stdev))
    count = np.count_nonzero((data>=(mean-3*stdev)) & (data<=(mean+3*stdev)))
    amp = count*binsize
    plt.plot(x, amp*gaussian, 'r')
    plt.plot([mean, mean],plt.ylim(), 'r', label='clipped mean')
    plt.plot([fmean, fmean], plt.ylim(), ':r', label='full mean')
    plt.plot([median, median],plt.ylim(), 'g', label='clipped median')
    plt.plot([fmed, fmed], plt.ylim(), ':g', label='full median')
    plt.legend()
    return




def multiple_photon_counting(ima, prior=None):
    """
    """
    tab_path = resource_filename('Calibration', 'PhotoCountingTables')
    tab_prior_name = os.path.join(tab_path, 'tabwithprior-EMgain470-rnoise109.csv')
    tab_woprior_name = os.path.join(tab_path, 'tabnoprior-EMgain470-rnoise109.csv')
    
    if prior is None:
        tab = np.loadtxt(tab_woprior_name, delimiter=',')
    else:
        tab = np.loadtxt(tab_prior_name, delimiter=',')
        
    # thresholding
    if prior is None:
        imac = np.searchsorted(tab, ima)
    else:
        prior_edge_values = tab[:-1,0] + (tab[1:,0] - tab[:-1,0])/2
        prior_index = np.searchsorted(prior_edge_values, prior)
        imac = np.zeros_like(ima, dtype=int)
        for i in range(tab.shape[0]):
            thresholds = tab[i,1:]
            # clean the table at the end (repeated max value or 0)
            last = np.where(thresholds == thresholds.max())[0].min()
            thresholds = thresholds[:last+1]
            mask = (prior_index == i)
            imac[mask] = np.searchsorted(thresholds, ima[mask])
            
    return imac
    

def compute_local_background(ima):
    """
    """
    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve
    # remove isolated bright pixels
    kernel  = Gaussian2DKernel(3)
    #bkg = convolve(ima, np.ones((11,11))/121)
    #bkg = gaussian_filter(ima, 3)
    bkg = convolve(ima, kernel)
    return bkg
    
    
def process_image(ima, hdr, with_prior=False, config=my_conf):
    """
    """
    if hdr[my_conf.gain[0]]==0:
        EMgain = 1
    if hdr[my_conf.gain[0]]==9000:
        EMgain = 235
    if hdr[my_conf.gain[0]]==9200:
        EMgain = 470
    if hdr[my_conf.gain[0]]==9400:
        EMgain = 700
    EMgain /= .32 #Accounting for smearing
    ADU2e = .53    
    print('EM gain = %i'%(EMgain))
    print('Conversion gain = %0.2f'%(ADU2e))

    # sub bias line/line
    ima0, _, _ = bias_from_right_overscan(ima)
    # remove hot pixels
    # compute local bkg (for prior estimation)
    if with_prior:
        bkg = compute_local_background(ima0)
        prior_val = bkg/ADU2e/EMgain
    ima0 /= ADU2e
    # thresholding & saturation at 10xprior
    if with_prior:
        ima_count = multiple_photon_counting(ima0, prior=prior_val)
    else:
        ima_count = multiple_photon_counting(ima0)  
    return ima_count

def StackAllImages(xpapoint, cat=None):
    """
    """
    from astropy.table import Table
    if cat is None:
        try:
            cat = Table.read(sys.argv[3])
        except:
            print('Impossible to open table, verify your path.')
    
    #################FIELD YOU WANT PRESERVE
    path_field = 'PATH'
    Field1 = 'FPATH'
    Field2 = my_conf.gain[0]
    Field3 = my_conf.exptime[0]
    Field4 = 'EMCCDBAC'
    
    
    print(cat)
    print(cat[Field1])
    fpath = np.unique(cat[Field1])
    if type(fpath)==str:
        fpath = [fpath]   
    for path in fpath:
        print(path)
        for gain in np.unique(cat[(cat[Field1]==path)][Field2]):
            for exp in np.unique(cat[(cat[Field1]==path)&(cat[Field2]==gain)][Field3]):
                for temp in np.unique(cat[(cat[Field1]==path)&(cat[Field2]==gain)&(cat[Field3]==exp)][Field4].astype(float)).astype(int):
                    files = cat[([path in cati[Field1] for cati in cat])  & (cat[Field2] == gain) & (cat[Field3] == exp) & (cat[Field4].astype(float).astype(int) == temp) ][path_field]
                    #print(files)
                    if len(files) > 0:
                        print('Stacking images of GAIN = %i, exposure  = %i, temp = %i and path=%s'%(float(gain), float(exp), float(temp), path))
                        print(files)
                        StackImagesPath(files, fname='-Gain%i-Texp%i-Temp%i'%(float(gain), float(exp), float(temp)))
    return
    
    


def Normalization2D(amplitude, xo, yo, sigma_x, sigma_y , offset):
    """Given the parameters of a gaussian function, returns the amplitude so that it is normalized in volume 
    """
    B = dblquad(lambda x, y: twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, 0), -float('inf'), +float('inf'), lambda x: -float('inf'), lambda x: float('inf'))       
    return (amplitude/B[0], xo, yo, sigma_x, sigma_y , 0)


def DS9guider(xpapoint):
    """Display on DS9 image from SDSS at the same location if a WCS header is 
    present in the image. If not, it send the image on the astrometry.net server
    and run a lost in space algorithm to have this header. Processing might take a few minutes
    """
    from astropy.io import fits
    d = DS9(xpapoint)
    filename = getfilename(d)#d.get("file")
    header = fits.getheader(filename)#d.get_fits()[0].header
    if ('WCSAXES' in header):
        print('WCS header existing, checking Image servers')
        d.set("grid")
        d.set("scale mode 99.5")
        try:# if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
            d.set("dsssao")
        except:
            pass
        d.set("lock scalelimits no")
        d.set("lock frame wcs")
        d.set("frame last")
        d.set("scale squared")
        
    else:
        print ('Nop header WCS - Applying lost in space algorithm: Internet needed!')
        print ('Processing might take a few minutes ~5-10')
        PathExec = os.path.dirname(os.path.realpath(__file__)) + '/astrometry-net.py'
        Newfilename = filename[:-5] + '_wcs.fits'
        CreateWCS(PathExec, filename, Newfilename)
        filename = d.set("file {}".format(Newfilename))
        filename = getfilename(d)#d.get("file")
        header = d.get_fits()[0].header
        if header['WCSAXES'] == 2:
            print('WCS header existing, checking Image servers')
            d.set("grid")
            d.set("scale mode 99.5")#vincent
            try:# if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
                d.set("dsssao")
            except:
                pass
            d.set("lock scalelimits no")
            d.set("lock frame wcs")        
    return

#    parser.add_option('--scale-units', dest='scale_units',
#                      choices=('arcsecperpix', 'arcminwidth', 'degwidth', 'focalmm'), help='Units for scale estimate')
#    #parser.add_option('--scale-type', dest='scale_type',
#    #                  choices=('ul', 'ev'), help='Scale bounds: lower/upper or estimate/error')
#    parser.add_option('--scale-lower', dest='scale_lower', type=float, help='Scale lower-bound')
#    parser.add_option('--scale-upper', dest='scale_upper', type=float, help='Scale upper-bound')
#    parser.add_option('--scale-est', dest='scale_est', type=float, help='Scale estimate')
#    parser.add_option('--scale-err', dest='scale_err', type=float, help='Scale estimate error (in PERCENT), eg "10" if you estimate can be off by 10%')
#    parser.add_option('--ra', dest='center_ra', type=float, help='RA center')
#    parser.add_option('--dec', dest='center_dec', type=float, help='Dec center')
#    parser.add_option('--radius', dest='radius', type=float, help='Search radius around RA,Dec center')
#    parser.add_option('--downsample', dest='downsample_factor', type=int, help='Downsample image by this factor')
#    parser.add_option('--parity', dest='parity', choices=('0','1'), help='Parity (flip) of image')
#    parser.add_option('--tweak-order', dest='tweak_order', type=int, help='SIP distortion order (default: 2)')
#    parser.add_option('--crpix-center', dest='crpix_center', action='store_true', default=None, help='Set reference point to center of image?')
#    parser.add_option('--sdss', dest='sdss_wcs', nargs=2, help='Plot SDSS image for the given WCS file; write plot to given PNG filename')

def CreateWCS(PathExec, filename, Newfilename):
    """Sends the image on the astrometry.net server
    and run a lost in space algorithm to have this header. 
    Processing might take a few minutes
    """
    import subprocess
    print(filename, Newfilename)
    print(os.path.dirname(filename) + "/--wait.fits")
    start = timeit.default_timer()
    print('''\n\n\n\n      Start lost in space algorithm - might take a few minutes \n\n\n\n''')
    subprocess.check_output("python " + PathExec + " --apikey apfqmasixxbqxngm --newfits " + Newfilename + " --upload " + filename,shell=True)
#    try:
#        os.rename(os.path.dirname(PathExec) + "/--wait", Newfilename)
#    except OSError:
#        os.rename(os.path.dirname(PathExec) + "/--wait.fits", Newfilename)
    stop = timeit.default_timer()
    print('File created')
    print('Lost in space duration = {} seconds'.format(stop-start))
    return


def DS9setup2(xpapoint, config=my_conf):
    """This function aims at giving a quick and general visualisation of the image by applying specific thresholding
        and smoothing parameters. This allows to detect easily:
        •Different spot that the image contains
        •The background low frequency/medium variation
        •If a spot saturates
        •If the image contains some ghost/second pass spots. . .
    """
    #from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
#    try:
#        region = getregion(d)
#    except ValueError:
#        image_area = [0,2069,1172,2145]
#        Yinf, Ysup,Xinf, Xsup = image_area
#    else:
#        Yinf, Ysup, Xinf, Xsup = Lims_from_region(region)#[131,1973,2212,2562]
#        print(Yinf, Ysup,Xinf, Xsup)
    #print(type(d.get("fits height")))
    if d.get("fits height") == '2069':
        print('Detector image')
        image_area = my_conf.physical_region#[10,2000,1172,2100]
    else:
        image_area = [0,-1,0,-1]
    Yinf, Ysup,Xinf, Xsup = image_area
    print(getfilename(d))
    fitsimage = d.get_pyfits()[0].data#d.get_fits()[0].data#d.get_arr2np()
    #print(fitsimage)
    image = fitsimage[Yinf: Ysup,Xinf: Xsup]#[Xinf: Xsup, Yinf: Ysup]
    #print(image)
    
#    if d.get("lock bin") == 'no':
    d.set("grid no") 
#        d.set("scale limits {} {} ".format(np.percentile(fitsimage[0].data,9),
#              np.percentile(fitsimage[0].data,99.6)))
    d.set("scale limits {} {} ".format(np.nanpercentile(image,50),np.nanpercentile(image,99.95)))
    d.set("scale asinh")
    d.set("cmap bb")
#        d.set("smooth yes")
#        d.set("cmap Cubehelix0")
#        d.set("smooth radius {}".format(2))
#        d.set("smooth yes")
    d.set("lock bin yes")
#    elif d.get("lock bin") == 'yes':
#       
#        d.set("cmap grey") #d.set("regions delete all")
#        d.set("scale linear")
#        d.set("scale mode minmax")
#        d.set("grid no")
#        d.set("smooth no")
#        d.set("lock bin no")
    return 


def DS9originalSettings(xpapoint):
    """DS9 original settings
    """
    d = DS9(xpapoint)
    d.set("cmap grey") #d.set("regions delete all")
    d.set("scale linear")
    d.set("scale mode minmax")
    d.set("grid no")
    d.set("smooth no")
    d.set("lock bin no")
    return d


def parse_data(data, nan=np.nan, map=map, float=float):
    """Parse DS9 defined region to give pythonic regions
    """
    vals = []
    xy = []
    for s in data.split('\n'):
        coord,val = s.split('=')
        val = val.strip() or nan
        xy.append(list(map(float, coord.split(','))))
        vals.append(float(val))
    vals = np.array(vals)
    xy = np.floor(np.array(xy)).astype(int)
    x,y = xy[:,0], xy[:,1]
    w = x.ptp() + 1
    h = y.ptp() + 1
    arr = np.empty((w, h))
    X = np.empty((w, h))
    Y = np.empty((w, h))
    indices = x - x.min(), y - y.min()
    arr[indices] = vals
    X[indices] = x
    Y[indices] = y
    return X.T, Y.T, arr.T

def process_region(regions, win):
    """Process DS9 regions to return pythonic regions
    """
    from collections import namedtuple
    processed_regions = []
    for region in regions:
        name, info = region.split('(')
        coords = [float(c) for c in info.split(')')[0].split(',')]
        print(coords)
        if name == 'box':
            xc,yc,w,h,angle = coords
            dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
            X,Y,arr = parse_data(dat)
            box = namedtuple('Box', 'data xc yc w h angle')
            processed_regions.append(box(arr, xc, yc, w, h, angle))
            #return box(arr, xc, yc, w, h, angle)
        elif name == 'bpanda':
            xc, yc, a1, a2, a3, a4,a5, w, h,a6,a7 = coords
            dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
            X,Y,arr = parse_data(dat)
            box = namedtuple('Box', 'data xc yc w h angle')
            processed_regions.append(box(arr, xc, yc, w, h, 0))
            #return box(arr, xc, yc, w, h, 0)
        elif name == 'circle':
            xc,yc,r = coords
            dat = win.get("data physical %s %s %s %s no" % (xc - r, yc - r, 2*r, 2*r))
            X,Y,arr = parse_data(dat)
            Xc,Yc = np.floor(xc), np.floor(yc)
            inside = (X - Xc)**2 + (Y - Yc)**2 <= r**2
            circle = namedtuple('Circle', 'data databox inside xc yc r')
            processed_regions.append(circle(arr[inside], arr, inside, xc, yc, r))
        elif name == '# vector':
            xc, yc, xl, yl = coords
            #dat = win.get("data physical %s %s %s %s no" % (xc - r, yc - r, 2*r, 2*r))
            #X,Y,arr = parse_data(dat)
            #Xc,Yc = np.floor(xc), np.floor(yc)
            #inside = (X - Xc)**2 + (Y - Yc)**2 <= r**2
            vector = namedtuple('Vector', 'data databox inside xc yc r')
            processed_regions.append(vector(xc, yc, xl, yl,0,0))

#return circle(arr[inside], arr, inside, xc, yc, r)
    #    elif name == 'annulus':
    #        xc,yc,r1,r2 = coords
    #        w = 2*r2
    #        h = 2*r2
    #        dat = win.get("data physical %s %s %s %s no" % (xc-r2,yc-r2,w,h))
    #        X,Y,arr = parse_data(dat)
    #        Xc,Yc = np.floor(xc), np.floor(yc)
    #        inside = between((X - Xc)**2 + (Y - Yc)**2, r1**2, r2**2)
    #        annulus = namedtuple('Annulus', 'data databox inside xc yc r1 r2')
    #        return annulus(arr[inside], arr, inside, xc, yc, r1, r2)
        elif name == 'ellipse':
            if len(coords) == 5:
                xc, yc, a2, b2, angle = coords
            else:
                xc, yc, a1, b1, a2, b2, angle = coords
            w = 2*a2
            h = 2*b2
            dat = win.get("data physical %s %s %s %s no" % (xc - a2,yc - b2,w,h))
            X,Y,arr = parse_data(dat)
            Xc,Yc = np.floor(xc), np.floor(yc)
            inside = ((X - Xc)/a2)**2 + ((Y - Yc)/b2)**2 <= 1
            if len(coords) == 5:
                ellipse = namedtuple('Ellipse',
                                     'data databox inside xc yc a b angle')
                return ellipse(arr[inside], arr, inside, xc, yc, a2, b2, angle)
    
            inside &= ((X - Xc)/a1)**2 + ((Y - Yc)/b1)**2 >= 1
            annulus = namedtuple('EllipticalAnnulus',
                                 'data databox inside xc yc a1 b1 a2 b2 angle')
            processed_regions.append(annulus(arr[inside], arr, inside, xc, yc, a1, b1, a2, b2, angle))
            #return annulus(arr[inside], arr, inside, xc, yc, a1, b1, a2, b2, angle)
        else:
            raise ValueError("Can't process region %s" % name)
    if len(processed_regions) == 1:
        return processed_regions[0]
    else:
        return processed_regions

def DS9realignImage(xpapoint):
    """
    """
    #import matplotlib; matplotlib.use('TkAgg')  
    #import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    filename = d.get("file")
    try:
        vector = getregion(d)
    except ValueError:
        pass    
    #fitsimage = fits.open(filename)[0]
    fitsimage = d.get_fits()[0]
    image = fitsimage.data
    xs, ys, angle = FindPixelsPositionsOnVector(vector)
    new_im = RecrateArrayFromIndexList(xs, ys, angle, image)
    fitsimage.data = new_im
    name = filename[:-5] + '_reAlighned.fits'
    fitswrite(fitsimage,name)
    d.set('frame new')
    d.set("file {}".format(name))#a = OpenFile(xpaname,filename = filename)

    return
    
    
def FindPixelsPositionsOnVector(vector):
    """
    """
    xc, yc, length, angle = vector.data, vector.databox, vector.inside, vector.xc
    xs = xc + np.linspace(0,length,length) * np.cos(np.pi*angle/180)
    ys = yc + np.linspace(0,length,length) * np.sin(np.pi*angle/180)
    #plt.plot(xs,ys,'x')
    #plt.axis('equal')
    #plt.show()plo
    return xs, ys, angle

def RecrateArrayFromIndexList(xs, ys, angle, image):
    """
    """
    xs, ys = xs.astype(int), ys.astype(int)
    n1, n2 = xs.min() - (xs.max()-xs.min())/2, 2936 - xs.max() - (xs.max()-xs.min())/2 #int(1500 - (xs.max()-xs.min())/2)
    #a = array[ys[0]-20:ys[0]+20,xs[0]-n:xs[0]+n]
    n1,n2=700,700
    a = image[ys[0],xs[0]-n1:xs[0]+n2]

    for i in range(len(xs)-1):
        a = np.vstack((a,image[ys[i+1],xs[i+1]-n1:xs[i+1]+n2]))
    #imshow(a)
    if (angle > 270.):
        a = a[::-1,:]
        print(angle)
        up = image[ys[0]:,xs[-1]-n1:xs[-1]+n2][::-1,:]
        down = image[:ys[-1],xs[0]-n1:xs[0]+n2][::-1,:]
        new_image = np.vstack((up,a,down))
    elif (angle > 180.) & (angle < 270.):
        print(angle)
        up = image[ys[0]:,xs[0]-n1:xs[0]+n2][::-1,:]
        down = image[:ys[-1],xs[-1]-n1:xs[-1]+n2][::-1,:]
        new_image = np.vstack((up,a,down))  
    #imshow(new_image)
    return new_image
    
    
    


def getregion(win, debug=False, all=False):
    """ Read a region from a ds9 instance.

    Returns a tuple with the data in the region.
    """
    win.set("regions format ds9")
    win.set("regions system physical")
    #rows = win.get("regions list")
    rows = win.get("regions all")
    rows = [row for row in rows.split('\n') if row]
    if len(rows) < 3:
        print( "No regions found")

    #units = rows[2]
    #assert units == 'physical'
    if debug:
        print (rows[4])
        if rows[5:]:
            print('discarding %i regions' % len(rows[5:]) )
    #print(rows[5:])
    if all:
        region = process_region(rows[3:], win)
        if type(region) == list:
            return region
        else:
            return [region]
    else:
        return process_region([rows[-1]], win)
#getregion(d, all=True)
        
def create_PA(A=15.45,B=13.75,C=14.95,pas=0.15,nombre=11):
    """Return encoder steps of FB2 tip-tilds focus
    """
    a = np.linspace(A-int(nombre/2)*pas, A+int(nombre/2)*pas, nombre)
    b = np.linspace(B-int(nombre/2)*pas, B+int(nombre/2)*pas, nombre)
    c = np.linspace(C-int(nombre/2)*pas, C+int(nombre/2)*pas, nombre)
    return a[::-1],b[::-1],c[::-1]
#ENCa, ENCb, ENCc = create_PA()

def ENC(x,ENCa):
    """Return encoder step of FB2 tip-tilds focus A
    """
    a = (ENCa[-1]-ENCa[0])/(len(ENCa)-1) * x + ENCa[0]
    #b = (ENCb[10]-ENCb[0])/(10) * x + ENCb[0]
    #c = (ENCc[10]-ENCc[0])/(10) * x + ENCc[0]
    return a#, b, c



def throughfocus(center, files,x=None, 
                 fibersize=0, center_type='barycentre', SigmaMax= 4,Plot=True,
                 Type=None, ENCa_center=None, pas=None, WCS=False, DS9backUp = DS9_BackUp_path, config=my_conf):
    """How to use: Open an image of the through focus which is close to the focus.  Click on region. Then click
    precisely on what you think is the centre of the PSF (the size does not matter). Select the region you created
    and press t (throughfocus) or go in analysis menu: throughfocus. This will open a dialog box that asks what
    is the number of the images of the through focus. You can either put the numbers (eg: "10-21") or only press
    enter if the folder in which is the image contains only the images from the throughfocus.
    This will pick up the center of the region you put, compute the barycenter of the image after removing the
    background. Then it keeps this center and for each image of the throughfocus the code computes the radial
    profile (+encircled energy) and fit it by the radial profile of the 2D convolution of a disk with a gaussian. It
    returns the characteristics of the spots and plot their evolution throughout the though focus (fwhm, EE50,
    EE80
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import AnalyzeSpot
    from .focustest import estimateBackground
    from scipy.optimize import curve_fit
    fwhm = []
    EE50 = []
    EE80 = [] 
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    sec=[]
    images=[]
    ENCa = []
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[0]
            image = fitsfile.data
        time = fitsfile.header['DATE']
        if Type == 'guider':    
            ENCa.append(fitsfile.header['LINAENC'])
        else:
            nombre = 5
            if ENCa_center is not None:
                print('Actuator given: Center = {} , PAS = {}'.format(ENCa_center,pas))
                ENCa = np.linspace(ENCa_center-nombre*pas, ENCa_center+nombre*pas, 2*nombre+1)[::-1]
#            else:
#                ENCa = np.linspace(100,100, 2*nombre+1)[::-1]
        day,h, m, s = float(time[-11:-9]),float(time[-8:-6]), float(time[-5:-3]), float(time[-2:])
        sec.append(t2s(h=h,m=m,s=s,d=day))

        background = 1*estimateBackground(image,center)
        n = 25
        subimage = (image-background)[int(center[1]) - n:int(center[1]) + n, int(center[0]) - n:int(center[0]) + n]
        images.append(subimage)
        d = AnalyzeSpot(image,center=center,fibersize=fibersize,
                        center_type=center_type, SigmaMax = SigmaMax)
        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d['Sigma'])
        EE50.append(d['EE50'])
        EE80.append(d['EE80'])
        xo.append(d['Center'][0])
        yo.append(d['Center'][1])        
        maxpix.append(max20[-20:].mean())
        sumpix.append(d['Flux'])
        varpix.append(subimage.var())
    f = lambda x,a,b,c: a * (x-b)**2 + c#a * np.square(x) + b * x + c
    x = np.arange(len(files))
    fig, axes = plt.subplots(4, 2,sharex=True)#, figsize=(10,6)
    xtot = np.linspace(x.min(),x.max(),200)
    print(ENCa)
    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted')
        bestx1 = xtot[np.argmin(f(xtot,*opt1))]
        axes[0,0].plot(np.ones(2)*bestx1,[min(fwhm),max(fwhm)])
        if len(ENCa) > 0:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx1,ENC(bestx1,ENCa)))
        else:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx1))
            
    except RuntimeError:
        opt1 = [0,0,0]
        bestx1 = np.nan
        pass 
    try:
        opt2,cov2 = curve_fit(f,x,EE50)
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted')
        bestx2 = xtot[np.argmin(f(xtot,*opt2))]
        if len(ENCa) > 0:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx2,ENC(bestx2,ENCa)))
        else:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx2))
        axes[1,0].plot(np.ones(2)*bestx2,[min(EE50),max(EE50)])
    except RuntimeError:
        opt2 = [0,0,0]
        bestx2 = np.nan
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[2,0].plot(xtot,f(xtot,*opt3),linestyle='dotted')
        bestx3 = xtot[np.argmin(f(xtot,*opt3))]
        if len(ENCa) > 0:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx3,ENC(bestx3,ENCa)))
        else:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx3))
        axes[2,0].plot(np.ones(2)*bestx3,[min(EE80),max(EE80)])
    except RuntimeError:
        opt3 = [0,0,0]
        bestx3 = np.nan
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted')
        bestx4 = xtot[np.argmax(f(xtot,*opt4))]
        axes[0,1].plot(np.ones(2)*bestx4,[min(maxpix),max(maxpix)])
        if len(ENCa) > 0:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx4,ENC(bestx4,ENCa)))
        else:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx4))
    except RuntimeError:
        opt4 = [0,0,0]
        bestx4 = np.nan
        pass            
    try:
        opt5,cov5 = curve_fit(f,x,sumpix)
        axes[1,1].plot(xtot,f(xtot,*opt5),linestyle='dotted')
        bestx5 = xtot[np.argmax(f(xtot,*opt5))]
        if len(ENCa) > 0:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx5,ENC(bestx5,ENCa)))
        else:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx5))
        axes[1,1].plot(np.ones(2)*bestx5,[min(sumpix),max(sumpix)])
    except RuntimeError:
        opt5 = [0,0,0]
        bestx5 = np.nan
        pass
    try:
        opt6,cov6 = curve_fit(f,x,varpix)
        axes[2,1].plot(xtot,f(xtot,*opt6),linestyle='dotted')
        bestx6 = xtot[np.argmax(f(xtot,*opt6))]
        if len(ENCa) > 0:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx6,ENC(bestx6,ENCa)))
        else:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx6))
        axes[2,1].plot(np.ones(2)*bestx6,[min(varpix),max(varpix)])
    except RuntimeError:
        opt6 = [0,0,0]
        bestx6 = np.nan
        pass

   

    name = '{} - {} - {}'.format(os.path.basename(filename),[int(center[0]),int(center[1])],fitsfile.header['DATE'])
    fig.tight_layout()
    fig.suptitle(name, y=1.01)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
    print(name) 
    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix','Best sigma','Best EE50','Best EE80','Best Maxpix','Best Varpix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    t.add_row((os.path.basename(filename),float(os.path.basename(filename)[-11:-4]),
               t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],min(fwhm),
               min(EE50),min(EE80),min(maxpix),min(sumpix),max(varpix),
               ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)))

    if Plot:
        axes[0,0].plot(x,fwhm, '-o')
        axes[0,0].set_ylabel('Sigma')
    
        axes[1,0].plot(x,EE50, '-o')
        axes[1,0].set_ylabel('EE50')
    
        axes[2,0].plot(x,EE80, '-o')
        axes[2,0].set_ylabel('EE80')
    
        axes[3,0].plot(x,xo, '-o')
        axes[3,0].set_ylabel('y center')
        axes[3,0].grid()
    
        axes[0,1].plot(x,maxpix, '-o')
        axes[0,1].set_ylabel('Max pix')
    
        axes[1,1].plot(x,sumpix, '-o')
        axes[1,1].set_ylabel('Flux')
        
        axes[2,1].plot(x,varpix, '-o')
        axes[2,1].set_ylabel('Var pix (d=50)')
        axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
        axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
        #axes[3,1].grid()
        axes[3,1].set_ylabel('y center')        
        plt.show()
        
        
        fig, axes = plt.subplots(1, 11,sharey=True)#,figsize=(24,3))
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('equal')
            subname = os.path.basename(files[i])[6:-5] 
            try:
                axes[i].set_xlabel('%i - %0.2f'%(int(subname),float(ENCa[i])))
                #axes[i].set_title(float(ENCa[i]))
            except:
                axes[i].set_xlabel(int(subname))
                pass
        fig.suptitle(name)
        fig.subplots_adjust(top=0.88)   
        fig.tight_layout()
        #plt.axis('equal')
        fig.savefig(os.path.dirname(filename) + '/ThroughfocusImage-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
        #fig.show()


    try:
        print(os.path.dirname(filename) + '/Throughfocus.csv')
        OldTable = Table.read(os.path.dirname(filename) + '/Throughfocus.csv')
        print ('old',OldTable)
    except IOError:
        t.write(os.path.dirname(filename) + '/Throughfocus.csv')
    else:
        t = vstack((OldTable,t))
        #print ('new',newTable)
        t.write(os.path.dirname(filename) + '/Throughfocus.csv',overwrite=True)
    return fwhm, EE50, EE80


def throughfocusWCS(center, files,x=None, 
                 fibersize=0, center_type='barycentre', SigmaMax= 4,Plot=True,
                 Type=None, ENCa_center=None, pas=None, WCS=False, DS9backUp = DS9_BackUp_path):
    """Same algorithm than throughfocus except it works on WCS coordinate 
    and not on pixels. Then the throughfocus can be run on stars even
    with a sky drift
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import AnalyzeSpot
    from .focustest import estimateBackground
    from scipy.optimize import curve_fit
    fwhm = []
    EE50 = []
    EE80 = [] 
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    sec=[]
    images=[]
    ENCa = []
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[0]
            image = fitsfile.data
        header = fitsfile.header
        time = header['DATE']
        if Type == 'guider':    
            ENCa.append(header['LINAENC'])
        else:
            nombre = 5
            if ENCa_center is not None:
                print('Actuator given: Center = {} , PAS = {}'.format(ENCa_center,pas))
                ENCa = np.linspace(ENCa_center-nombre*pas, ENCa_center+nombre*pas, 2*nombre+1)[::-1]
#            else:
#                ENCa = np.linspace(100,100, 2*nombre+1)[::-1]
        day,h, m, s = float(time[-11:-9]),float(time[-8:-6]), float(time[-5:-3]), float(time[-2:])
        sec.append(t2s(h=h,m=m,s=s,d=day))
        if WCS:
            from astropy import units as u
            from astropy import wcs

            print ('Using WCS')
            w = wcs.WCS(header)
            center_wcs = center
            center_pix = w.all_world2pix(center_wcs[0]*u.deg, center_wcs[1]*u.deg,0, )
            center_pix = [int(center_pix[0]), int(center_pix[1])]
            print('CENTER PIX= ' , center_pix)
        else:
            center_pix = center
        d = AnalyzeSpot(image,center=center_pix,fibersize=fibersize,
                        center_type=center_type, SigmaMax = SigmaMax)
#        else:
#            d = AnalyzeSpot(image,center=center,fibersize=fibersize,
#                            center_type=center_type, SigmaMax = SigmaMax)
        background = 1*estimateBackground(image,center)
        n = 25
        subimage = (image-background)[int(center_pix[1]) - n:int(center_pix[1]) + n, int(center_pix[0]) - n:int(center_pix[0]) + n]
        images.append(subimage)

        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d['Sigma'])
        EE50.append(d['EE50'])
        EE80.append(d['EE80'])
        xo.append(d['Center'][0])
        yo.append(d['Center'][1])        
        maxpix.append(max20[-20:].mean())
        sumpix.append(d['Flux'])
        varpix.append(subimage.var())
    f = lambda x,a,b,c: a * (x-b)**2 + c     #a * np.square(x) + b * x + c
    if Type == 'guider':
        x = np.array(ENCa)
        xtot = np.linspace(x.min(),x.max(),200)
        ENC = lambda x,a : x
    if Type == 'detector':
        x = np.arange(len(files))
        xtot = np.linspace(x.min(),x.max(),200)
        if len(ENCa)==0:
            ENC = lambda x,a : 0
        else :
            ENC = lambda x,a : (ENCa[-1]-ENCa[0])/(len(ENCa)-1) * x + ENCa[0]

   
    #x = np.array(ENCa)
    fig, axes = plt.subplots(4, 2,sharex=True)#,figsize=(24,3)
    
#    if ENCa_center is not None:
#        axes2 = axes[0,0].twinx()
#        X2tick_location= axes[0,0].xaxis.get_ticklocs() #Get the tick locations in data coordinates as a numpy array
#        axes2.set_xticks(X2tick_location)
#        axes2.set_xticklabels(ENCa)
    #print(ENCa)
    #print(ENC(,ENCa))
    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted')
        bestx1 = xtot[np.argmin(f(xtot,*opt1))]
        axes[0,0].plot(np.ones(2)*bestx1,[min(fwhm),max(fwhm)])
        if len(ENCa) > 0:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx1,ENC(bestx1,ENCa)))
        else:
            axes[0,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx1))
            
    except RuntimeError:
        opt1 = [0,0,0]
        bestx1 = np.nan
        pass 
    try:
        opt2,cov2 = curve_fit(f,x,EE50)
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted')
        bestx2 = xtot[np.argmin(f(xtot,*opt2))]
        if len(ENCa) > 0:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx2,ENC(bestx2,ENCa)))
        else:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx2))
        axes[1,0].plot(np.ones(2)*bestx2,[min(EE50),max(EE50)])
    except RuntimeError:
        opt2 = [0,0,0]
        bestx2 = np.nan
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[2,0].plot(xtot,f(xtot,*opt3),linestyle='dotted')
        bestx3 = xtot[np.argmin(f(xtot,*opt3))]
        if len(ENCa) > 0:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx3,ENC(bestx3,ENCa)))
        else:
            axes[2,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx3))
        axes[2,0].plot(np.ones(2)*bestx3,[min(EE80),max(EE80)])
    except RuntimeError:
        opt3 = [0,0,0]
        bestx3 = np.nan
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted')
        bestx4 = xtot[np.argmax(f(xtot,*opt4))]
        axes[0,1].plot(np.ones(2)*bestx4,[min(maxpix),max(maxpix)])
        if len(ENCa) > 0:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx4,ENC(bestx4,ENCa)))
        else:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx4))
    except RuntimeError:
        opt4 = [0,0,0]
        bestx4 = np.nan
        pass            
    try:
        opt5,cov5 = curve_fit(f,x,sumpix)
        axes[1,1].plot(xtot,f(xtot,*opt5),linestyle='dotted')
        bestx5 = xtot[np.argmax(f(xtot,*opt5))]
        if len(ENCa) > 0:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx5,ENC(bestx5,ENCa)))
        else:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx5))
        axes[1,1].plot(np.ones(2)*bestx5,[min(sumpix),max(sumpix)])
    except RuntimeError:
        opt5 = [0,0,0]
        bestx5 = np.nan
        pass
    try:
        opt6,cov6 = curve_fit(f,x,varpix)
        axes[2,1].plot(xtot,f(xtot,*opt6),linestyle='dotted')
        bestx6 = xtot[np.argmax(f(xtot,*opt6))]
        if len(ENCa) > 0:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx6,ENC(bestx6,ENCa)))
        else:
            axes[2,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx6))
        axes[2,1].plot(np.ones(2)*bestx6,[min(varpix),max(varpix)])
    except RuntimeError:
        opt6 = [0,0,0]
        bestx6 = np.nan
        pass
    axes[0,0].plot(x,fwhm, '-o')
    axes[0,0].set_ylabel('Sigma')

    axes[1,0].plot(x,EE50, '-o')
    axes[1,0].set_ylabel('EE50')

    axes[2,0].plot(x,EE80, '-o')
    axes[2,0].set_ylabel('EE80')

    axes[3,0].plot(x,xo, '-o')
    axes[3,0].set_ylabel('y center')

    axes[0,1].plot(x,maxpix, '-o')
    axes[0,1].set_ylabel('Max pix')

    axes[1,1].plot(x,sumpix, '-o')
    axes[1,1].set_ylabel('Flux')
    
    axes[2,1].plot(x,varpix, '-o')
    axes[2,1].set_ylabel('Var pix (d=50)')
    axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
    axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
    axes[3,1].set_ylabel('y center')
   

    fig.tight_layout()

    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix','Best sigma','Best EE50','Best EE80','Best Maxpix','Best Varpix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    t.add_row((os.path.basename(filename),os.path.basename(filename)[5:11],
               t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],min(fwhm),
               min(EE50),min(EE80),min(maxpix),min(sumpix),max(varpix),
               ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)))
    mean = np.nanmean(np.array([ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)]))
    print(mean)
    name = '%s - %i - %i - %s - %0.3f'%(os.path.basename(filename),int(center_pix[0]),int(center_pix[1]),fitsfile.header['DATE'],mean)
    #name = '%s - %i - %i'%(os.path.basename(filename),int(center_pix[0]),int(center_pix[1]),fitsfile.header['DATE'],mean)
    print(name) 
    fig.suptitle(name, y=0.99)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center_pix[0]) ,int(center_pix[1]), fitsfile.header['DATE']))
    if Plot:        
        plt.show()
        fig, axes = plt.subplots(1, 11,figsize=(24,3),sharey=True)
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('equal')
            subname = os.path.basename(files[i])[6:11] 
            try:
                axes[i].set_xlabel('%i - %0.2f'%(int(subname),float(ENCa[i])))
                #axes[i].set_title(float(ENCa[i]))
            except:
                axes[i].set_xlabel(int(subname))
                pass
        fig.suptitle(name)
        fig.subplots_adjust(top=0.88)   
        fig.tight_layout()
        #plt.axis('equal')
        fig.savefig(os.path.dirname(filename) + '/ThroughfocusImage-{}-{}-{}.png'.format( int(center_pix[0]) ,int(center_pix[1]), fitsfile.header['DATE']))
        #fig.show()
    try:
        print(os.path.dirname(filename) + '/Throughfocus.csv')
        OldTable = Table.read(os.path.dirname(filename) + '/Throughfocus.csv')
        print ('old',OldTable)
    except IOError:
        t.write(os.path.dirname(filename) + '/Throughfocus.csv')
    else:
        t = vstack((OldTable,t))
        #print ('new',newTable)
        t.write(os.path.dirname(filename) + '/Throughfocus.csv',overwrite=True)

    return fwhm, EE50, EE80

def throughfocus2(center, files,x=np.linspace(11.95,14.45,11)[::-1][3:8], 
                 fibersize=100, center_type='barycentre', SigmaMax= 4, box=25):
    """How to use: Open an image of the through focus which is close to the focus.  Click on region. Then click
    precisely on what you think is the centre of the PSF (the size does not matter). Select the region you created
    and press t (throughfocus) or go in analysis menu: throughfocus. This will open a dialog box that asks what
    is the number of the images of the through focus. You can either put the numbers (eg: "10-21") or only press
    enter if the folder in which is the image contains only the images from the throughfocus.
    This will pick up the center of the region you put, compute the barycenter of the image after removing the
    background. Then it keeps this center and for each image of the throughfocus the code computes the radial
    profile (+encircled energy) and fit it by the radial profile of the 2D convolution of a disk with a gaussian. It
    returns the characteristics of the spots and plot their evolution throughout the though focus (fwhm, EE50,
    EE80
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from .focustest import AnalyzeSpot
    from .focustest import estimateBackground
    from scipy.optimize import curve_fit
    from astropy.table import Table, vstack
    fwhm = []
    EE50 = []
    EE80 = [] 
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    sec = []
    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    n = box
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[0]
            image = fitsfile.data
        time = fitsfile.header['DATE']
        day,h, m, s = float(time[-11:-9]),float(time[-8:-6]), float(time[-5:-3]), float(time[-2:])
        sec.append(t2s(h=h,m=m,s=s,d=day))
        #image = fitsfile.data
        background = 1*estimateBackground(image,center)
        subimage = (image-background)[int(center[1]) - n:int(center[1]) + n, int(center[0]) - n:int(center[0]) + n]
        d = AnalyzeSpot(image,center=center,fibersize=fibersize,
                        center_type=center_type, SigmaMax = SigmaMax)
        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d['Sigma'])
        EE50.append(d['EE50'])
        EE80.append(d['EE80'])
        xo.append(d['Center'][0])
        yo.append(d['Center'][1])        
        maxpix.append(max20[-20:].mean())
        sumpix.append(d['Flux'])
        varpix.append(subimage.var())
        t.add_row((os.path.basename(filename),float(os.path.basename(filename)[-11:-4]), t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],d['Sigma'],d['EE50'],d['EE80'],max20[-20:].mean(),subimage.sum(),subimage.var() ))
    try:
        print(os.path.dirname(filename) + '/Analysis.csv')
        OldTable = Table.read(os.path.dirname(filename) + '/Analysis.csv')
        print ('old',OldTable)
    except IOError:
        t.write(os.path.dirname(filename) + '/Analysis.csv')
    else:
        t = vstack((OldTable,t))
        #print ('new',newTable)
        t.write(os.path.dirname(filename) + '/Analysis.csv',overwrite=True)
    sec = np.array(sec)
    sec = (sec - sec.min())/60
    xtot = np.linspace(sec.min(),sec.max(),200)
    x = sec
    f = lambda x,a,b,c: a * (x-b)**2 + c#a * np.square(x) + b * x + c

    fig, axes = plt.subplots(4, 2)#, figsize=(10,6))


    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted')
        axes[0,0].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt1))],[min(fwhm),max(fwhm)])
        axes[0,0].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt1))]))
    except RuntimeError:
        pass 
    try:
        opt2,cov2 = curve_fit(f,x,EE50)
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted')
        axes[1,0].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt2))]))
        axes[1,0].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt2))],[min(EE50),max(EE50)])
    except RuntimeError:
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[2,0].plot(xtot,f(xtot,*opt3),linestyle='dotted')
        axes[2,0].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt3))]))
        axes[2,0].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt3))],[min(EE80),max(EE80)])
    except RuntimeError:
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted')
        axes[0,1].plot(np.ones(2)*xtot[np.argmax(f(xtot,*opt4))],[min(maxpix),max(maxpix)])
        axes[0,1].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmax(f(xtot,*opt4))]))
    except RuntimeError:
        pass            
    try:
        opt5,cov5 = curve_fit(f,x,sumpix)
        axes[1,1].plot(xtot,f(xtot,*opt5),linestyle='dotted')
        axes[1,1].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmin(f(xtot,*opt5))]))
        axes[1,1].plot(np.ones(2)*xtot[np.argmin(f(xtot,*opt5))],[min(sumpix),max(sumpix)])
    except RuntimeError:
        pass
    try:
        opt6,cov6 = curve_fit(f,x,varpix)
        axes[2,1].plot(xtot,f(xtot,*opt6),linestyle='dotted')
        axes[2,1].set_xlabel('Best Image index = %0.3f' % (xtot[np.argmax(f(xtot,*opt6))]))
        axes[2,1].plot(np.ones(2)*xtot[np.argmax(f(xtot,*opt6))],[min(varpix),max(varpix)])
    except RuntimeError:
        pass
    axes[0,0].plot(x,fwhm, '-o')
    axes[0,0].set_ylabel('Sigma')

    axes[1,0].plot(x,EE50, '-o')
    axes[1,0].set_ylabel('EE50')

    axes[2,0].plot(x,EE80, '-o')
    axes[2,0].set_ylabel('EE80')

    axes[3,0].plot(x,xo, '-o')
    axes[3,0].set_ylabel('y center')
    axes[3,0].grid()

    axes[0,1].plot(x,maxpix, '-o')
    axes[0,1].set_ylabel('Max pix')

    axes[1,1].plot(x,sumpix, '-o')
    axes[1,1].set_ylabel('Flux')
    
    axes[2,1].plot(x,varpix, '-o')
    axes[2,1].set_ylabel('Var pix (d=50)')
    axes[3,1].plot(x,yo - np.array(yo).mean(), '-o')
    axes[3,1].plot(x,xo - np.array(xo).mean(), '-o')
    axes[3,1].grid()
    axes[3,1].set_ylabel('y center')
   

    name = '{} - {} - {}'.format(os.path.basename(filename),[int(center[0]),int(center[1])],fitsfile.header['DATE'])
    fig.tight_layout()
    fig.suptitle(name, y=1.)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
    plt.show()
    print(name) 
    return t#fwhm, EE50, EE80



def DS9throughfocus(xpapoint, Plot=True):
    """How to use: Open an image of the through focus which is close to the focus.  Click on region. Then click
    precisely on what you think is the centre of the PSF (the size does not matter). Select the region you created
    and press t (throughfocus) or go in analysis menu: throughfocus. This will open a dialog box that asks what
    is the number of the images of the through focus. You can either put the numbers (eg: "10-21") or only press
    enter if the folder in which is the image contains only the images from the throughfocus.
    This will pick up the center of the region you put, compute the barycenter of the image after removing the
    background. Then it keeps this center and for each image of the throughfocus the code computes the radial
    profile (+encircled energy) and fit it by the radial profile of the 2D convolution of a disk with a gaussian. It
    returns the characteristics of the spots and plot their evolution throughout the though focus (fwhm, EE50,
    EE80
    """
    from astropy.io import fits
    from .focustest import AnalyzeSpot

    print('''\n\n\n\n      START THROUGHFOCUS \n\n\n\n''')
    d = DS9(xpapoint)
    filename = getfilename(d)#d.get("file ")
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    try:
        ENCa_center, pas = sys.argv[4].split('-')
        ENCa_center, pas = float(ENCa_center), float(pas)
    except ValueError:
        print('No actuator given, taking header ones for guider images, none for detector images')
        ENCa_center, pas = None, None
    except IndexError :
        print('No actuator given, taking header ones for guider images, none for detector images')
        ENCa_center, pas = None, None        
    x = np.arange(len(path))
    
    a = getregion(d)
    image = fits.open(filename)[0]
    rp = AnalyzeSpot(image.data,center = [np.int(a.xc),
                     np.int(a.yc)],fibersize=0)
    x,y = rp['Center']    
    d.set('regions system image')
    print('\n\n\n\n     Centring on barycentre of the DS9 image '
          '(need to be close to best focus) : %0.1f, %0.1f'
          '--> %0.1f, %0.1f \n\n\n\n' % (a.xc,a.yc,rp['Center'][0],rp['Center'][1]))
    print('Applying throughfocus')
    if image.header['BITPIX'] == -32:
        Type = 'guider'
    else:
        Type = 'detector'        
    if d.get('wcs lock frame') == 'wcs':
        from astropy import wcs
        print ('Using WCS')
        w = wcs.WCS(image.header)
        center_wcs = w.all_pix2world(x, y,0)
                #d.set('crosshair {} {} physical'.format(x,y))
        alpha, delta = float(center_wcs[0]), float(center_wcs[1])
        print('alpha, delta = ',alpha, delta)
        
        #alpha, delta = float(alpha), float(delta)
        throughfocusWCS(center = [alpha,delta], files=path,x = x,fibersize=0,
                     center_type='user',SigmaMax=6, Plot=Plot, Type=Type,ENCa_center=ENCa_center, pas=pas,WCS=True)
        
    else:
        throughfocus(center = rp['Center'], files=path,x = x,fibersize=0,
                     center_type='user',SigmaMax=6, Plot=Plot, Type=Type,ENCa_center=ENCa_center, pas=pas)

    return 

def hasNumbers(inputString):
    """Tcheck if number in the string
    """
    return any(char.isdigit() for char in inputString)

def DS9rp(xpapoint, Plot=True, config=my_conf):
    """How to use: Click on region and select Circle shape (default one). Then click precisely on what you think is
    the centre of the PSF. Select the region you created and press p or go in analysis menu: radial profile.
    The code will:
    •Pick up the center of the region you put
    •Subtract the background by evaluating it in an annulus around the spot
    •Compute the barycenter of the image
    •Compute the radial profile and encircled energy using this new center
    •Return the characteristics of the spot (see image: EE50, FWHM, source size. . . )  and plot the radial
    profile of the spot 
    """
    #from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
#    import re
    d = DS9(xpapoint)#DS9(xpapoint)
#    try:
    print(sys.argv[3])
    center_type = sys.argv[3]
#    except IndexError:
#        entry = ''
    
    fibersize = sys.argv[4] if sys.argv[4].replace('.','',1).isdigit() else 0 and print('Fiber size not understood, setting to 0')
#    if hasNumbers(entry):
#        fibersize = float(re.findall(r'\d+','456')[0])
#    else:
#        fibersize = 0

#    try:
#        fibersize = entry
#    except IndexError:
#        print('No fibersize, Using point source object')
#        fibersize = 0
#    if fibersize == '':
#        fibersize = 0
    #fibersize = 1
    filename =  getfilename(d)#d.get("file ")
    a = getregion(d)
    #fitsfile = fits.open(filename)[0]
    fitsfile = d.get_fits()[0]


    print(center_type)
    spot = DS9plot_rp_convolved(data=fitsfile.data,
                                center = [np.int(a.xc),np.int(a.yc)],
                                fibersize=fibersize, center_type=center_type)    
    try:
        plt.title('{} - {} - {}'.format(os.path.basename(filename),[np.int(a.xc),np.int(a.yc)],fitsfile.header['DATE']),y=0.99)
    except KeyError:
        print('No date in header')
    if Plot:
        plt.show()
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=red"' % (spot['Center'][0]+1,spot['Center'][1]+1,40))
    return


def DS9plot_rp_convolved(data, center, size=40, n=1.5, anisotrope=False,
                         angle=30, radius=40, ptype='linear', fit=True, 
                         center_type='barycentre', maxplot=0.013, minplot=-1e-5, 
                         radius_ext=12, platescale=None,fibersize = 100,SigmaMax=4, DS9backUp = DS9_BackUp_path, config=my_conf):
  """Function used to plot the radial profile and the encircled energy of a spot,
  Latex is not necessary
  """
  from .focustest import  radial_profile_normalized
  import matplotlib; matplotlib.use('TkAgg')  
  import matplotlib.pyplot as plt
  from .focustest import ConvolveDiskGaus2D
  from .focustest import gausexp
  from scipy.optimize import curve_fit
  from scipy import interpolate
  if anisotrope == True:
      spectral, spatial, EE_spectral, EE_spatial = radial_profile_normalized(data, center, anisotrope=anisotrope, angle=angle, radius=radius, n=n, center_type=center_type)
      spectral = spectral[~np.isnan(spectral)]
      spatial = spatial[~np.isnan(spatial)]

      norm_spatial = spatial[:size]#(spatial[:n] - min(min1,min2)) / np.nansum((spatial[:n] - min(min1,min2) ))
      norm_spectral = spectral[:size]#(spectral[:n] - min(min1,min2)) / np.nansum((spectral[:n] - min(min1,min2) ))              
      if ptype == 'linear':
          popt1, pcov1 = curve_fit(gausexp, np.arange(size), norm_spatial)       
          popt2, pcov2 = curve_fit(gausexp, np.arange(size), norm_spectral) 
          plt.plot(np.arange(size), norm_spectral, label='spectral direction')   
          plt.plot(np.arange(size), norm_spatial, label='spatial direction') 
          if fit==True:
              plt.plot(np.linspace(0,size,10*size), gausexp(np.linspace(0,size,10*size), *popt1), label='Spatial fit')            
              plt.plot(np.linspace(0,size,10*size), gausexp(np.linspace(0,size,10*size), *popt2), label='Spectral fit')                       
              plt.figtext(0.5,0.5,'Sigma = %0.3f-%0.3f pix \nLambda = %0.3f-%0.3f pix \npcGaus = %0.0f-%0.3fpc' % (popt1[2],popt2[2],popt1[3],popt2[3],100*popt1[0]/(popt1[1] + popt1[0]),100*popt2[0]/(popt2[1] + popt2[0])), fontsize=11,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
      else:
          plt.semilogy(np.arange(size),norm_spectral, label='spectral direction')   
          plt.semilogy(np.arange(size),norm_spatial, label='spatial direction')           
      return popt1, popt2
  else: 
      rsurf, rmean, profile, EE, NewCenter = radial_profile_normalized(data, center, anisotrope=anisotrope, angle=angle, radius=radius, n=n, center_type=center_type)
      profile = profile[:size]#(a[:n] - min(a[:n]) ) / np.nansum((a[:n] - min(a[:n]) ))
      fig, ax1 = plt.subplots()#figsize=(12, 6))
          #popt, pcov = curve_fit(ConvolveDiskGaus2D, np.linspace(0,size,size), profile, p0=[2,2,2])#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):  3.85  
      fiber = float(fibersize) #/ (2*1.08*(1/0.083))
      if fiber == 0:
          gaus = lambda x, a, sigma: a**2 * np.exp(-np.square(x / sigma) / 2)
          popt, pcov = curve_fit(gaus, rmean[:size], profile, p0=[1, 2])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
          ax1.plot(np.linspace(0,size,10*size), gaus(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
          ax1.fill_between(rmean[:size], profile - 1.5*np.abs(profile - gaus(rmean[:size], *popt)), 
                           profile + 1.5*np.abs(profile - gaus(rmean[:size], *popt)), alpha=0.3, label=r"3*Residuals")
          
      else:
          #profile /= profile.max()
          #SigmaMax = 10
          #popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[profile.max(),fiber,2, np.nanmean(profile)],bounds=([0,0.95*fiber,1,-1],[2,1.05*fiber,SigmaMax,1]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
          popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[profile.max(),fiber,2, np.nanmean(profile)],bounds=([1e-3*profile.max(),0.95*fiber,1,1e-1*profile.mean()],[1e3*profile.max(),1.05*fiber,SigmaMax,1e1*profile.mean()]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
          ax1.plot(np.linspace(0,size,10*size), ConvolveDiskGaus2D(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
          ax1.fill_between(rmean[:size], profile - 1.5*np.abs(profile - ConvolveDiskGaus2D(rmean[:size], *popt)), 
                           profile + 1.5*np.abs(profile - ConvolveDiskGaus2D(rmean[:size], *popt)), alpha=0.3, label=r"3*Residuals")
      ax1.set_xlabel('Distance to center [pix]', fontsize=12)                      
      #ax1.plot(rmean[:size], profile, '+', c='black', label='Normalized isotropic profile')
      ax1.plot(rmean[:size], profile, '+', c='black', label='Normalized isotropic profile')
      #ax1.plot(np.linspace(0, size, size), profile, '+', c='black', label='Normalized isotropic profile')
      #ax1.plot(np.linspace(0, size, 10*size), exp(np.linspace(0, size, 10*size), *popt), c='navy')
      #ax1.plot(np.linspace(0, size, 10*size), gaus(np.linspace(0, size, 10*size), *popt), c='blue')
      ax1.set_ylabel('Radial Profile', color='b', fontsize=12)
      ax1.tick_params('y', colors='b')
      ax1.set_ylim((minplot, np.nanmax([np.nanmax(1.1*(profile)), maxplot])))
      ax2 = ax1.twinx()
      #ax2.plot(np.linspace(0, size, len(norm)), EE[:len(norm)], 'r--x')
      EE_interp = interpolate.interp1d(rsurf[:size], EE[:size],kind='cubic')
      ninterp = 10
      xnew = np.linspace(rsurf[:size].min(),rsurf[:size].max(),ninterp*len(rsurf[:size]))
      ax2.plot(xnew,EE_interp(xnew),linestyle='dotted',c='r')

      
      ax2.plot(rsurf[:size], EE[:size], 'rx')

      mina = min(xnew[EE_interp(xnew)[:ninterp*size]>79])
      minb = min(xnew[EE_interp(xnew)[:ninterp*size]>49])

      ax2.plot(np.linspace(minb, minb, 2), np.linspace(0, 50, 2), 'r-o')                    
      ax2.plot(np.linspace(minb, size, 2), np.linspace(50, 50, 2), 'r-o')
      ax2.plot(np.linspace(mina, mina, 2), np.linspace(0, 80, 2), 'r-o')                    
      ax2.plot(np.linspace(mina, size, 2), np.linspace(80, 80, 2), 'r-o')
      #EE_gaus = np.cumsum(gaus(np.linspace(0,size,100*size),*popt) *2 * np.pi * np.linspace(0,size,100*size)**1)
      #EE_exp = np.cumsum(exp(np.linspace(0,size,100*size),*popt) * 2 * np.pi * np.linspace(0,size,100*size)**1)
      ax2.set_ylim((0, 110))
      ax2.set_ylabel('Encircled Energy', color='r', fontsize=12)
      ax2.tick_params('y', colors='r')
      fig.tight_layout()
      ax1.xaxis.grid(True)
      ax1.tick_params(axis='x', labelsize=12)
      ax1.tick_params(axis='y', labelsize=12)
      ax2.tick_params(axis='y', labelsize=12)                    
#                  e_gaus = np.nansum(gaus(np.linspace(0,size,100*size),*popt) *2 * np.pi * np.linspace(0,size,100*size)**1)
#                  e_exp = np.nansum(exp(np.linspace(0,size,100*size),*popt) * 2 * np.pi * np.linspace(0,size,100*size)**1)
      ax1.legend(loc = (0.54,0.05),fontsize=12)
      if fiber == 0:
          flux = 2*np.pi*np.square(popt[1])*np.square(popt[0])
          plt.figtext(0.53,0.53,"Flux = %0.0f\nRadius = %0.3f pix \nSigmaPSF = %0.3f pix \nEE50-80 = %0.2f - %0.2f p" % (flux,0,abs(popt[1]),minb,mina), 
                      fontsize=14,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})#    norm_gaus = np.pi*sigma    norm_exp = 2*np.pi * lam**2 * gamma(2/alpha)/alpha
          d = {"Flux":flux,"SizeSource":0,"Sigma":popt[1],"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
          print("Flux = {}\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(flux,0,popt[1],minb,mina,platescale,NewCenter))
      else:
          plt.figtext(0.53,0.53,"Amp = %0.3f\nRadius = %0.3f pix \nSigmaPSF = %0.3f pix \nEE50-80 = %0.2f - %0.2f p" % (popt[0],popt[1],popt[2],minb,mina), 
                      fontsize=14,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})#    norm_gaus = np.pi*sigma    norm_exp = 2*np.pi * lam**2 * gamma(2/alpha)/alpha
          d = {"Flux":0,"SizeSource":popt[1],"Sigma":popt[2],"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
          print("Flux = 0\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(popt[1],popt[2],minb,mina,platescale,NewCenter))
      csvwrite(np.vstack((rmean[:size], profile,ConvolveDiskGaus2D(rmean[:size], *popt))).T, DS9backUp + 'CSVs/%s_EnergyProfile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
      csvwrite(np.vstack((rsurf, EE)).T, DS9backUp + 'CSVs/%s_RadialProfile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
      return d


def DS9open(xpapoint, filename=None):
    """As the OSX version of DS9 does not allow to enter the path of an image when we want to access some data
    I added this possibility. Then you only need to press o (open) so that DS9 opens a dialog box where you can
    enter the path. Then click OK.
    """
    from PIL import Image
    
    if filename is None:
        filename = sys.argv[3]
    if (filename[-4:].lower() == '.jpg') or (filename[-4:].lower()  == 'jpeg') or (filename[-4:].lower()  == '.png') :
        im = Image.open(filename, 'r')
        lx, ly = im.size
        pix = np.array(im.getdata())
        #n = pix.shape[-1]
        #new_pix = np.zeros((ly,lx,n))
        #for i in range(n):
        #    new_pix[:,:,i] = pix[:,i].reshape(ly,lx)[:,::-1]
        pix = np.mean(pix, axis=1).reshape(ly,lx)[:,::-1]
        filename = '/tmp/test.fits'
        fitswrite(pix,filename)

    if os.path.isfile(filename):
        print('Opening = ',filename)
        d = DS9(xpapoint)#DS9(xpapoint)
        d.set('grid no')
        d.set('frame new')
        d.set("file {}".format(filename))#a = OpenFile(xpaname,filename = filename)
    else:
        print(bcolors.BLACK_RED + 'File not found, please verify your path' + bcolors.END)
        sys.exit()
    return


    
  

def Charge_path_new(filename, entry_point = 3, entry=None, All=0, begin='-', end='-', liste='-', patht='-', config=my_conf):
    """From the entry gave in DS9 (either nothing numbers or beginning-end),
    reuturns the path of the images to take into account in the asked analysis
    """
    #from astropy.io import fits
    import re
#    if len(sys.argv[entry_point:]) == 5:    
#        All, begin, end, liste, patht = sys.argv[entry_point:]
    try:
        a, b, e, l, p = sys.argv[-5:]
    except ValueError:
        pass
    else:
        if a.isdigit() & (b=='-' or b.isdigit()) & (e=='-' or e.isdigit()) & (p=='-' or os.path.isdir(p)):
            All, begin, end, liste, patht = sys.argv[-5:]
        else:
            print('Taking function argument not sys.argv')
    print('All, begin, end, liste, path =', All, begin, end, liste, patht)
    #fitsimage = fits.open(filename)
    number = re.findall(r'\d+',os.path.basename(filename))[-1]
    n = len(number)
    filen1, filen2 = filename.split(number)
    print(filen1, filen2)
    
#    if fitsimage[0].header['BITPIX'] == -32:
#        Type = 'guider'
#    else:
#        Type = 'detector'
#    print ('Type = {}'.format(Type))
#    print(int(float(All)), type(int(float(All))))


    
    if os.path.isdir(patht):
        print('Folder given, going though all the repositories of %s'%(patht))
        path = glob.glob(os.path.join(patht , '**/*.fits'), recursive=True)
        path += glob.glob(os.path.join(patht , '**/*.FIT'), recursive=True)
        path += glob.glob(os.path.join(patht , '**/*.fit'), recursive=True)

    elif int(float(All)) == 1:
        print('Not numbers, taking all the .fits images from the current repository')
        path = glob.glob('%s%s%s'%(filen1,'?'*n,filen2))

    
    elif liste.split('-')[0]!='':
        path = []
        print('Specified numbers are integers, opening corresponding files ...')
        numbers = np.array( liste.split('-'),dtype=int)
        print('Numbers used: {}'.format(numbers))               
        for number in numbers:
            path.append(filen1 + '%0{}d'.format(n)%(number) + filen2) #'%0{}d'.format(n)%(1) + filen2
    
    elif (begin == '-') & (end == '-')  :
        path = [filename]
    
    else:
        path = []
        print('Two numbers given, opening files in range ...')
        n1, n2 = int(float(begin)), int(float(end))
        numbers = [n1,n2]
        files = np.sort(glob.glob('%s%s%s'%(filen1,'?'*n,filen2)));#print(files)
        #print(files)
        im_numbers = []
        for file in files:
            n = int(file.split(filen1)[-1].split(filen2)[0])
            im_numbers.append(n)
        im_numbers = np.array(im_numbers);im_numbers.sort()
        print('Files in folder : ', im_numbers)
        for i, im_number in enumerate(im_numbers):
            #print('ok',n1,n2,im_number)
            if (im_number >= n1) & (im_number <= n2):
                #print('yes')
                path.append(files[i])   
    path = np.sort(path)
    print("\n".join(path))
    return path     
                        

def Charge_path(filename, entry_point = 3, entry=None):
    """From the entry gave in DS9 (either nothing numbers or beginning-end),
    reuturns the path of the images to take into account in the asked analysis
    """
    #print('salut')
    #sys.quit()
    from astropy.io import fits
    import re
    if entry is None:
        try:
            entry = sys.argv[entry_point]#'325-334'# sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# 
            print('Entry 1 = ', entry)
        except IndexError:
            n1=''
            n2='' 
            numbers = ['']
            entry = ''
    if os.path.isdir(entry):
        print('Folder given, going though all the repositories')
        path = entry 
#        size = get_size(path)*1e-9
#        var = input("The total file of the folder is %0.2fGo this would lead to %0.1f minute analysis with info (%0.1fmin without)... Are you sure you want to proceed [y]/n : "%(size, size*1e3*0.050/60, size*1e3*0.001/60))
#        if var.lower() != 'n':
        files = glob.glob(os.path.join(path , '**/*.fits'), recursive=True)
        files += glob.glob(os.path.join(path , '*.fits'))
        np.sort(files)
        return files
        
    if entry == 'all':
        path = glob.glob(os.path.dirname(filename) + '/*.fits')
        print(path)
        path = np.sort(path)
        return path[::-1]
    numbers = entry.split('-')
#    if numbers in None:
#        pass
    if len(numbers) == 1:
        try:
            numbers = int(numbers[0])#None
        except ValueError:
            numbers = None
    elif len(numbers) == 2:
        n1, n2 = entry.split('-')
        n1, n2 = int(n1), int(n2)
        #if Type == 'detector':            
        numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
    #elif len(numbers) > 2:
        
    print('Numbers used: {}'.format(numbers))               
    fitsimage = fits.open(filename)#d.get_fits()
    number = re.findall(r'\d+',os.path.basename(filename))[-1]
    filen1, filen2 = filename.split(number)
    print(filen1, filen2 )
    
    if fitsimage[0].header['BITPIX'] == -32:
        Type = 'guider'
    else:
        Type = 'detector'
    print ('Type = {}'.format(Type))
    
    path = []
    if Type == 'detector':
        if numbers is None:
            print('Not numbers, taking all the .fits images from the current repository')
            path = glob.glob(os.path.dirname(filename) + '/image*.fits')
            #path = [filename]
        elif type(numbers) == int:
            print('Single number given, using this image.')
            #path = [os.path.dirname(filename) + '/%s%06d%s' % (os.path.basename(filename)[:5],int(numbers),os.path.basename(filename)[11:])]
            path = [filen1 + '%06d'%(numbers) + filen2]
        elif numbers is not None:
            print('Specified numbers are integers, opening corresponding files ...')
            for number in numbers:
#                path.append(os.path.dirname(filename) + '/%s%06d%s' % (os.path.basename(filename)[:5],int(number),os.path.basename(filename)[11:]))
                path.append(filen1 + '%06d'%(int(number)) + filen2)

    
    if Type == 'guider':
        files = glob.glob(os.path.dirname(filename) + '/stack*.fits')
        im_numbers = []
        try:
            for file in files:
                name = os.path.basename(file)
                im_numbers.append(int(name[5:13]))
            im_numbers = np.array(im_numbers)
            print('Files in folder : ', im_numbers)
        except:
            pass
        map
        if numbers is None:
            print('Not numbers, taking all the .fits images from the current repository')
            path = glob.glob(os.path.dirname(filename) + '/stack*.fits')
            #path = [os.path.dirname(filename) + '/%s%06d%s' % (os.path.basename(filename)[:5],int(number),os.path.basename(filename)[11:])]
        elif type(numbers) == int:
            print('Only one number given, running analysis on this image')
            #path = glob.glob(os.path.dirname(filename) + '/stack*.fits')
            path = [filename]
        elif len(numbers) == 2:
            print('Two numbers given, opening files in range ...')
            path = []
            for i, im_number in enumerate(im_numbers):
                print('ok',n1,n2,im_number)
                if (im_number >= n1) & (im_number <= n2):
                    print('yes')
                    path.append(files[i])
                    print(files[i])
        elif len(numbers) > 2:
            print('More than 2 numbers given, opening the corresponding files ...')
            path = []
            numbers = [int(number) for number in numbers]
            for i, im_number in enumerate(im_numbers):
                if im_number in numbers:
                    path.append(files[i])
    for file in path:
        if 'table' in file:
            path.remove(file)
    path = np.sort(path)
    print(path)   
    return path
    
def DS9visualisation_throughfocus(xpapoint):
    """
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    d.set('tile yes')
    try:
        a = getregion(d)
        print('Region found')
    except:
        print('Region not found')
        pass
    d.set("frame delete")
    d.set("smooth no")
    for filen in path[:]:
        d.set('frame new')
        d.set("fits {}".format(filen))        
    d.set("lock frame physical")
    d.set("lock scalelimits yes") 
    d.set("lock smooth yes") 
    d.set("lock colorbar yes") 

    try:
        d.set('pan to %0.3f %0.3f physical' % (a.xc,a.yc))
    except:
        pass
    return


def plot_hist2(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,
               ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain,temp, 
               plot_flag=False, ax=None, save=True, DS9backUp = DS9_BackUp_path):
    """Plot the log histogram of the image used to apply thresholding photocounting
    process
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from  scipy import stats
    from astropy.table import Table
    if ax is None:
        fig = plt.figure()#figsize=(12,4.5))
        ax = fig.add_subplot(111)
    else:
        save = False
    ax.set_xlabel("Pixel Value [ADU]",fontsize=12)
    ax.set_ylabel("Log(\# Pixels)",fontsize=12)
    ax.plot(bin_center, n_log, "rx", label="Histogram")
    ax.plot(xgaussfit,np.log(ygaussfit), "b-", label="Gaussian")
    ax.plot(np.ones(len(n_log))*threshold0,n_log, "b--", label="Bias")
    ax.plot(np.ones(2) * xlinefit.min(), [0.8 * ylinefit.max(), 1.2 * ylinefit.max()], "k--", label="5.5 Sigma")
    ax.plot(np.ones(2) * xlinefit.max(), [0.8 * ylinefit.min(), 1.2 * ylinefit.min()], "k--")
    for k in [1,10]:
        ax.plot(np.ones(2) * (bias + k * sigma), [0,1], "k--")
        flux = np.nansum(10**(n_log[(bin_center > bias + k * sigma) & (np.isfinite(n_log))])) / np.nansum(10**(n_log[np.isfinite(n_log)])) 
        ax.text(bias + k * sigma,1.1, 'k=%i \nRONpc=%0.1f \nF=%0.3fe-/p '%(k, 200*(1-stats.norm(0, 1).cdf(k)),flux),
                fontsize=8)
        
    #ax.plot(np.ones(len(n_log))*threshold55, n_log, "k--", label="5.5 Sigma")
    ax.plot(xlinefit,ylinefit, "g--", label="EM gain fit")
    ax.text(.55, .75, 'Bias value = %0.3f DN\nSigma = %0.3f DN\n '
                'EM gain = %0.1f e/e \nEMg smearg corr (0.32)=%0.1f e/e' % (bias, sigma, emgain, emgain/0.32),transform=ax.transAxes,
                fontsize=10,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    ax.text(.83, .25, 'Exposure = %i sec\nGain = %i \n '
                'T det = %0.2f C' % (exposure, gain, float(temp)),transform=ax.transAxes,
                fontsize=10,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
    ax.legend(loc="upper right",fontsize=12)   
    ax.grid(b=True, which='major', color='0.75', linestyle='--')
    ax.grid(b=True, which='minor', color='0.75', linestyle='--')
    ax.tick_params(axis='x', labelsize=13)
    ax.tick_params(axis='y', labelsize=13)
    ax.set_ylim([0,np.log(n_bias) + 0.2 ])
#    ax.set_xlim((-1000,10000))
    #ax.set_xlim((bias - 10 * sigma, bias + 100 * sigma))
    #a = np.isfinite(n_log)
    #ax.set_xlim((np.nanpercentile(bin_center[a],0.1),np.nanpercentile(bin_center[a],80)))#([10**4,10**4.3])    
    #ax.set_xlim((bias - 10 * sigma, bias + 100 * sigma))#([10**4,10**4.3]) 
    
    if save:
        if not os.path.exists(os.path.dirname(image) +'/Histograms'):
            os.makedirs(os.path.dirname(image) +'/Histograms')
        plt.savefig(os.path.dirname(image) +'/Histograms/'+ os.path.basename(image).replace('.fits', '.hist.png'), dpi = 100, bbox_inches = 'tight')
    csvwrite(Table(np.vstack((bin_center,n_log)).T), DS9backUp + 'CSVs/%s_Histogramcsv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    #a = Table(np.vstack((bin_center,n_log)).T)
    #print(a)
    #a.write(DS9backUp + 'CSVs/%s_Histogramcsv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")),format='csv')
    return n_log, bin_center 


def apply_pc(image,bias, sigma,area=0, threshold=5.5):
    """Put image pixels to 1 if superior to threshold and 0 else
    """
    cutoff = int(bias + sigma*threshold)#)5.5)
    idx = image > cutoff - 1        
    image[idx] = np.ones(1, dtype = np.uint16)[0]
    image[~idx] = np.zeros(1, dtype = np.uint16)[0]
    return image

def CountCRevent(paths='/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CosmicRaysFree/image000???.CRv.fits', config=my_conf):
    """
    """
    import glob
    from astropy.ito import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt    
    files = glob.glob(paths)
    files.sort()
    Xinf, Xsup, Yinf, Ysup = my_conf.physical_region
    for path in files[::-1]:
        print(path)
        with fits.open(path) as f:
            fitsimage = f[0]
            header = f[0].header
            image = f[0].data.copy()
            image[np.isfinite(image)]=0
            image[~np.isfinite(image)]=1
            CS = plt.contour(image, levels=1, colors='white', alpha=0.5)
            try:
                header['N_CR'] = len(CS.allsegs[0]) - 1 
            except IndexError:
                header['N_CR'] = 0
            header['%NaNs'] = 100*np.nansum(image[:,Xinf:Xsup]) / (Ysup*(Xsup-Xinf))
            fitsimage.header = header
            fitsimage.writeto(path,overwrite=True)
    return




def getImage(xpapoint):
    """
    """
    #from astropy.io import fits
    d = DS9(xpapoint)
    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    #fitsimage = fits.open(filename)[0]
    fitsimage = d.get_fits()[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]#picouet
    header = fitsimage.header
    return image, header, area
    

def PlotArea3D(xpapoint):
    """
    Plot the image area defined in DS9 in 3D, should add some kind of 2D
    polynomial fit
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #Needed even if not used
    from .polynomial import polyfit2d
    image, header, area = getImage(xpapoint)
    X,Y = np.indices(image.shape)
    x, y = image.shape
    x,y = np.meshgrid(np.arange(x),np.arange(y),indexing='ij')
    x, y = x.ravel(), y.ravel()
#    pol = polyfit2d(np.arange(image.shape[1]), np.arange(image.shape[0]), image, [5,5])
    imager = image.ravel()
    coeff = polyfit2d(x, y, imager, [4,4])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, image, c='r', s=0.1)#, cstride=1, alpha=0.2)
    ax.plot_surface(X, Y, popol2D(x, y , coeff).reshape(X.shape) ,rstride=1, cstride=1, shade=True)#,  linewidth=0)
    plt.title('3D plot, area = %s'%(area))
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlim((0.9 * image.min(), 1.1 * image.max()))
    ax.set_zlabel('Pixels ADU value')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    return
    
def popol2D(x, y , coeff):
    fit = 0
    for i in range(coeff.shape[0]):
        for j in range(coeff.shape[1]):
            fit += np.power(x,i) * np.power(y,j) * coeff[i,j] 
    return fit

def ContinuumPhotometry(xpapoint, x=None, y=None, DS9backUp = DS9_BackUp_path, config=my_conf):
    """Fit a gaussian
    interger the flux on 1/e(max-min) and then add the few percent calculated by the gaussian at 1/e 
    """
    #from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.table import Table
    from scipy.optimize import curve_fit
#    from .focustest import Gaussian
#    if xpapoint is not None:
#        d = DS9(xpapoint)
    texp = 1#d.get_fits()[0].header[my_conf.exptime[0]]
        
    if y is None:
        try:
            path = sys.argv[3]
            a = Table.read(path,format='ascii')
        except IndexError:
            a = Table.read(DS9backUp + '/DS9Curves/ds9.dat',format='ascii')
        
        x, y = a['col1'], a['col2']
    if x is None:
        x = np.arange(len(y))
    n=0.3
    fit = np.polyfit(x[y<y.mean()+n*np.std(y)], y[y<y.mean()+n*np.std(y)] , 1)
    #xmax = x[np.argmax(y)]
    fit_fn = np.poly1d(fit) 

    popt1, pcov1 = curve_fit(gaussian, x,y-fit_fn(x), p0=[y.max()-y.min(), x[y.argmax()], 5])
  
    
    limfit =  fit_fn(x)+gaussian(x, *popt1).max()/np.exp(1)

#    import matplotlib._cntr as cntr
#    c = cntr.Cntr(x, y, z)
#    nlist = c.trace(level, level, 0)
#    CS = nlist[:len(nlist)//2]
    image = np.array([np.zeros(len(y)),y>limfit,np.zeros(len(y))])
    CS = plt.contour(image,levels=1);plt.close()
    #size = [cs[:,0].max() - cs[:,0].min()     for cs in CS.allsegs[0] ]
    maxx = [ int(cs[:,0].max())   for cs in CS.allsegs[0] ]
    minx = [ int(cs[:,0].min())   for cs in CS.allsegs[0] ]
    maxx.sort();minx.sort()
    index = np.where(maxx>popt1[1])[0][0]#np.argmax(size)
    print(minx[index],maxx[index])

    y0 = y-fit_fn(x)
    plt.figure()#figsize=(10,6))
    plt.xlabel('Spatial direction')
    plt.ylabel('ADU mean value')
    plt.plot(x, gaussian(x, *popt1) + fit_fn(x), label='Gaussian Fit, F = %0.2f - sgm=%0.1f'%(np.sum(gaussian(x, *popt1))/texp,popt1[-1]))
#    plt.plot(x[y>limfit], y[y>limfit], 'o', color=p[0].get_color(), label='Best SNR flux calculation: F=%i'%(1.155*np.sum(y0[y>limfit])/texp))
    mask = (x>minx[index]) & (x<maxx[index])
    p = plt.plot(x, y , linestyle='dotted', label='Data, F=%0.2fADU - SNR=%0.2f'%(np.nansum(y0)/texp,gaussian(x, *popt1).max()/np.std(y0[~mask])))
#    plt.plot(x[mask], y[mask], 'o', color=p[0].get_color(), label='Best SNR flux calculation: F=%i'%(1.155*np.sum(y0[mask])/texp))
    #plt.plot(x,  limfit,'--', c='black', label='1/e limit')
    plt.fill_between(x[mask],y[mask],y2=fit_fn(x)[mask],alpha=0.2, label='Best SNR flux calculation: F=%0.2f'%(1.155*np.sum(y0[mask])/texp))
    plt.plot(x, fit_fn(x),label='Fitted background',linestyle='dotted',c=p[0].get_color())
    plt.legend()
    plt.show()

#plt.imshow(np.array([np.zeros(len(y)),y>limfit,np.zeros(len(y))]))


    #### la limit en x pour 1/e est sigma*sqrt(2) donc 
#erf2 = lambda x : (1 + special.erf(x))/2
#gauss = lambda x: np.exp(-np.power(x, 2.) / (2 * np.power(1, 2.))) / np.sqrt(2*np.pi)
#x = np.linspace(-10,10,1000)
#g = gauss(x)/gauss(x).sum()
#a = 2*(np.sum(g[x>0])/np.sum(g) - np.sum(g[x>np.sqrt(2)])/np.sum(g))
#print('Need to correct for ',1-a)
    
    
#    popt, pcov = curve_fit(Gaussian, x,y, p0=[y.max()-y.min(), x[y.argmax()], 5, y.min()])
#    plt.figure(figsize=(10,6))
#    plt.xlabel('Spatial direction')
#    plt.ylabel('ADU mean value')
#    plt.plot(x, y ,'--o', label='Data, F=%0.2fADU/sec/col'%(np.nansum(y-fit_fn(x))/texp))
#    plt.plot(x, Gaussian(x, *popt) ,'--', label='Gaussian Fit, F = %0.3f'%(np.sum(Gaussian(x, *popt)-popt[-1])))
#    limfit =   Gaussian(x, *popt).min() + (Gaussian(x, *popt).max() -  Gaussian(x, *popt).min())/np.exp(1)
#    plt.plot(x[y>limfit], y[y>limfit], 'P', color='red', label='Point for flux calculation: F=%i'%(1.1579*np.sum(y[y>limfit]-popt[-1])))
#    plt.plot([x.min(), x.max()], np.ones(2) * limfit,'--', c='black', label='1/e limit')
#    plt.plot(x, fit_fn(x),label='Fitted background')
#    plt.legend()
#    plt.show()
    return popt1


def DS9photo_counting(xpapoint, save=True, config=my_conf):
    """Calculate threshold of the image and apply phot counting
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    filename = d.get("file")
    try:
        region = getregion(d)
    except ValueError:
        Xinf, Xsup, Yinf, Ysup = my_conf.physical_region#[0,2069,1172,2145]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    image_area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)
    try:
        threshold = float(sys.argv[3])
    except IndexError:
        threshold = 5.5
    print('Threshold = %0.2f'%(threshold))
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    if len(path) == 1:
        plot_flag = True
    else:
        plot_flag = False
    print('plot_flag = ', plot_flag)
        
    for filename in path:
        print(filename)
        try:
            fitsimage = fits.open(filename)
        except IOError as e:
            print(bcolors.BLACK_RED + 'FILE NOT FOUND: ' + bcolors.END, e)
        else:
            image = fitsimage[0].data
            #emgain,bias,sigma,amp,slope,intercept = calc_emgain(image,area=image_area,plot_flag=True)
            print(type(filename))
#            a, b = calc_emgainGillian(filename,area=image_area,plot_flag=plot_flag)
#            print(len(a),len(b))
#            (emgain,bias,sigma,frac_lost) = a
            
            D = calc_emgainGillian(filename,area=image_area,plot_flag=plot_flag)
            emgain,bias,sigma,frac_lost =  [D[x] for x in ["emgain",'bias','sigma','frac_lost']]# D[my_conf.gain[0],'bias','sigma','frac_lost']
            b = [D[x] for x in ["image","emgain","bias","sigma","bin_center","n","xlinefit","ylinefit","xgaussfit", "ygaussfit","n_bias","n_log","threshold0","threshold55","exposure", "gain", "temp"]]
            
            
            #image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp = b
            try:    
                a = plot_hist2(*b, ax=None,plot_flag=plot_flag)
            except TypeError:
                pass
            else:
                if plot_flag:
                    plt.show()
                if save: 
                    new_image = apply_pc(image=image, bias=bias, sigma=sigma ,area=0, threshold=threshold)
                    print (new_image.shape)
                    fitsimage[0].data = new_image
#                    if not os.path.exists(os.path.dirname(filename) +'/Thresholded_images'):
#                        os.makedirs(os.path.dirname(filename) +'/Thresholded_images')
                    name = os.path.dirname(filename) +'/Thresholded_images/'+ os.path.basename(filename)[:-5] + '_THRES.fits'
                    #name = '/tmp/test_pc.fits'
                    fitswrite(fitsimage[0], name)
                    if len(path) == 0:
                        d.set('frame new')
                        d.set('file ' + name)  
    return 

def fitswrite(fitsimage, filename, verbose=True, config=my_conf):
    """
    """
    from astropy.io import fits
    if type(fitsimage) == np.ndarray:
        fitsimage = fits.HDUList([fits.PrimaryHDU(fitsimage)])[0]
    if len(filename)==0:
        print('Impossible to save image in filename %s, saving it to /tmp/image.fits'%(filename))
        filename = '/tmp/image.fits'
        fitsimage.header['NAXIS3'], fitsimage.header['NAXIS1'] = fitsimage.header['NAXIS1'], fitsimage.header['NAXIS3']
        fitsimage.writeto(filename,overwrite=True) 
    if 'NAXIS3' in fitsimage.header:
        verboseprint('2D array: Removing NAXIS3 from header...',verbose=verbose)
        fitsimage.header.remove('NAXIS3')
    elif not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if not os.path.exists(os.path.dirname(filename)):
        verboseprint('%s not existing: creating folde...'%(os.path.dirname(filename)),verbose=verbose)
        os.makedirs(os.path.dirname(filename))
    try:
        fitsimage.writeto(filename, overwrite=True)
    except IOError:
        print(bcolors.BLACK_RED + 'Can not write in this repository : ' + filename + bcolors.END)
        filename = '/tmp/' + os.path.basename(filename)
        print(bcolors.BLACK_RED + 'Instead writing new file in : ' + filename + bcolors.END)
        fitsimage.writeto(filename,overwrite=True) 
    verboseprint('Image saved: %s'%(filename),verbose=verbose)
    return filename


def csvwrite(table, filename, verbose=True, config=my_conf):
    """
    """
    from astropy.table import Table
    if type(table) == np.ndarray:
        table = Table(table)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    try:                                          
        table.write(filename, overwrite=True, format='csv')
    except UnicodeDecodeError:
        print('UnicodeDecodeError: you should consider change the name of the file/folder...')
        reload(sys); sys.setdefaultencoding('utf8')
    try:
        table.write(filename, overwrite=True, format='csv')
    except IOError:
        print(bcolors.BLACK_RED + 'Can not write in this repository : ' + filename + bcolors.END)
        filename = '/tmp/' + os.path.basename(filename)
        print(bcolors.BLACK_RED + 'Instead writing new file in : ' + filename + bcolors.END)
        table.write(filename, overwrite=True, format='csv')
    verboseprint('Table saved: %s'%(filename),verbose=verbose)
    return table

def gaussian(x, amp, x0, sigma):
    """Gaussian funtion
    """
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))

def gaussianFit(x, y, param):
    """Fit Gaussian funtion
    """
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(gaussian, x, y, p0=param)   
    amp, x0, sigma = popt   
    return (amp, x0, sigma) 


def linefit(x, A, B):
    """
    """
    return A*x + B


def fitLine(x, y, param=None):
    """Fit line
    """
    from scipy.optimize import curve_fit
    try:
        popt, pcov = curve_fit(linefit, x, y, p0 = param)
    except TypeError as e:
        print(e)
        return (1,1)
    else:
        a, b = popt
        return (a, b)

def ind2sub(array_shape, ind):
    """?
    """
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def DS9next(xpapoint):
    """Load the next alphanumerical image in DS9
    """
    d = DS9(xpapoint)
    filename = d.get("file")
    files = glob.glob(os.path.dirname(filename) + '/*.fits')
    files.sort()
    index = files.index(filename)
    print(files,filename,index)
    d.set('tile no')
    try:
        d.set('frame clear')
        d.set("file {}".format(files[index+1]))
    except IndexError:
        print('No more files')
        sys.exit()
    return
                         
def DS9previous(xpapoint):
    """Still to be written, load previous image in DS9
    """
    return                         
                         

def create_multiImage(xpapoint, w=None, n=30, rapport=1.8, continuum=False):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """
    field, Frame, w = sys.argv[3:]#'f3 names'#sys.argv[3]
    print('Field, Frame, w = ', field, Frame, w)
    d = DS9(xpapoint)
    #filename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    
    x, y, redshift, slit, w = returnXY(field, w=w, frame=Frame)  
    
    n1, n2 = n, n 
    print('n1,n2 = ',n1,n2)
    redshift = redshift.tolist()
    sliti=[]
    redshifti=[]
    imagettes=[]
    xi=[]
    yi=[]
    for i in range(len(x)):
        if (y[i]>1053) & (y[i]<2133) & (x[i]>0+n1) & (x[i]<2070-n1):
            imagettes.append(image[int(x[i])-n1:int(x[i]) +n1,int(y[i])-n2:int(y[i]) +n2])
            redshifti.append(redshift[i])
            sliti.append(slit[i])
            xi.append(x[i]);yi.append(y[i])
            print(y[i],x[i])
    v1,v2 = 6,14
    try:
        new_image = np.ones((v1*(2*n) + v1,v2*(2*n) + v2))*np.nanmin(imagettes[0])
    except ValueError:
        print ('No matching in the catalog, please run focustest before using this function')
        #sys.exit()
    for index,imagette in enumerate(imagettes):
        j,i = index%v2,index//v2
        centrei, centrej = 1 + (2*i+1) * n,1 + (2*j+1) * n
        #print (i,j)
        #print (centrei,centrej)
        try:
            new_image[centrei-n:centrei+n,centrej-n:centrej+n] = imagette
        except:
            pass
    new_image[1:-1:2*n, :] = np.nanmax(np.array(imagettes[0]))
    new_image[:,1:-1:2*n] = np.nanmax(np.array(imagettes[0]))
    if continuum:
        new_image[0:-2:2*n, :] = np.nanmax(np.array(imagettes))
        new_image[:,0:-2:4*n] = np.nanmax(np.array(imagettes))
    fitsfile[0].data = new_image[::-1, :]   
    fitswrite(fitsfile[0],'/tmp/imagettes.fits')
    
    d.set('frame new')
    d.set("file /tmp/imagettes.fits")
    d.set('scale mode minmax')
    return




def check_if_module_exists(module):
    if sys.version_info.major == 3:
        if sys.version_info.minor > 3:
            from importlib.util import find_spec as check_module
            spam_spec = check_module(module)
            found = spam_spec is not None
        else:
            from importlib import find_loader as check_module
            spam_loader = check_module(module)
            found = spam_loader is not None
        return found
    elif sys.version_info.major == 2:
        from pkgutil import find_loader as check_module    
        eggs_loader = check_module(module)
        found = eggs_loader is not None
        return found
    
class bcolors:
    BLACK_RED = '\x1b[4;30;41m' 
    GREEN_WHITE = '\x1b[0;32;47m' 
    BLACK_GREEN = '\x1b[0;30;42m' 
    END = '\x1b[0m'



def returnXY(field, w = 2060, frame='observed', keyword='lya'):
    """Return redshift, position of the slit, wavelength used for each mask
    given the DS9 entry
    """
    from astropy.table import Table
#    try:
    from .mapping import Mapping
#    except ValueError:
#        from Calibration.mapping import Mapping
    field = field.lower()
    if (w == 'lya') or (w == 'Lya') or (w == 'Lya'):
        w = 1215.67
    w = float(w)
    w *= 1e-4
    print('Selected Line is : %0.4f microns'%( w))

    try:
        #slit_dir = resource_filename('DS9FireBall', 'Slits')
        Target_dir = resource_filename('DS9FireBall', 'Targets')
        Mapping_dir = resource_filename('DS9FireBall', 'Mappings')
    except:
        #slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        Target_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Targets')
        Mapping_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mappings')
        
        
    if ('f1' in field) or ('119' in field):
        #csvfile = os.path.join(slit_dir,'F1_119.csv')
        targetfile = os.path.join(Target_dir,'targets_F1.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F1.pkl')#mapping-mask-det-180612-F1.pkl
    if ('f2' in field) or ('161' in field):
        #csvfile = os.path.join(slit_dir,'F2_-161.csv')
        targetfile = os.path.join(Target_dir,'targets_F2.csv')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F2.pkl')
    if ('f3' in field) or ('121' in field):
        #csvfile = os.path.join(slit_dir,'F3_-121.csv')
        targetfile = os.path.join(Target_dir,'targets_F3.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F3.pkl')
    if ('f4' in field) or ('159' in field):
        #csvfile = os.path.join(slit_dir,'F4_159.csv')
        targetfile = os.path.join(Target_dir,'targets_F4.txt')
        mappingfile = os.path.join(Mapping_dir,'mapping-mask-det-w-1806012-F4.pkl')
    #print('Selected field in : ', csvfile)
    print('targetfile = ',targetfile)  
    mapping = Mapping(filename=mappingfile)

    try:
        target_table = Table.read(targetfile)#, format='ascii')
    except:
        target_table = Table.read(targetfile, format='ascii', delimiter='\t')

    if 'f1' in field:
        idok = (target_table['slit_length_right'] != 0) &  (target_table['slit_length_left'] !=0)
        target_table = target_table[idok]
        xmask = target_table['xmask'] + (target_table['slit_length_right'] - target_table['slit_length_left'])/2.
        ymask = target_table['ymask'] + target_table['offset']
        z = target_table['z'] 
        internalCount = target_table['Internal-count']

    else:
        xmask = target_table['xmm']
        ymask = target_table['ymm']
        internalCount = target_table['Internal-count']
        z = target_table['Z'] 
    redshift = z
    slit = internalCount
    
    if frame.lower() == 'restframe':
        print('Working in rest frame wevelength' )
        #w = 1215.67
        wavelength = (1 + redshift) * w #* 1e-4
        y,x = mapping.map(wavelength, xmask, ymask, inverse=False)
    if frame.lower() == 'observedframe':
        print('Working in observed frame wevelength' )
        y,x = mapping.map(w, xmask, ymask, inverse=False)
    w *= 1e4
        
    #print(x[0],y[0])
    if keyword is None:
        return x, y, redshift, slit, w  
    elif keyword.lower() == 'lya':
        index = [len(sliti)<3 for sliti in slit]
        if ('f2' in field) or ('161' in field):
            mag = target_table['mgNUVMAG_L'] 
            #index = [magi>np.median(mag) for magi in mag]
            x, y, redshift, slit, mag, w = x[index], y[index], redshift[index], slit[index], mag[index], w
        else:
            x, y, redshift, slit, w = x[index], y[index], redshift[index], slit[index], w
        return x, y, redshift, slit, w#x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'qso':
        index = ['qso' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'lyc':
        index = ['lyc' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index]  
    elif keyword.lower() == 'ovi':
        index = ['ovi' in sliti for sliti in slit]
        return x[index], y[index], redshift[index], slit[index], w#[index] 





def DS9plot_spectra(xpapoint, w=None, n=30, rapport=1.8, continuum=False, DS9backUp = DS9_BackUp_path):
    """Plot spectra in local frame 
    """
    import matplotlib.pyplot as plt

    field, Frame, w, kernel, threshold = sys.argv[3:]#'f3 names'#sys.argv[3]
    kernel, threshold = int(kernel), int(threshold)
    print('Field, Frame, w, kernel = ', field, Frame, w, kernel)
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    x, y, redshift, slit, w = returnXY(field, w=w, frame=Frame)  

        
#    if '-' in line:
#        field, w =  line.split('-')
#        x, y, redshift, slit, w = returnXY(field, w=w, frame='rest')  
#    else:
#        x, y, redshift, slit, w = returnXY(field, w='lya', frame='rest') 
    #x, y = x+93, y-93
    create_DS9regions2(y,x, radius=10, form = 'box',save=True,color = 'yellow', savename='/tmp/centers')
    #d.set('regions /tmp/centers.reg')
    print('x = ', x)
    print('y = ', y)
    #print('redshift = ', redshift)
    #print('slit = ', slit)
    try:
        region = getregion(d)
    except:
        pass
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        mask = (y>Xinf) & (y<Xsup) & (x>Yinf) & (x<Ysup)
        print(mask)
        x, y, redshift, slit = x[mask], y[mask], redshift[mask], slit[mask]
        print('Selected objects are : ', slit)
    
    
    n1, n2 = int(n/4), 700 
    print('n1,n2 = ',n1,n2)
    redshift = redshift.tolist()

    lambdasup = []
    lambdainf = []
    sup = 2100#2133
    inf = 1100#1053
    x2w = 10./46
    flag_visible = (y>inf) & (y<sup) & (x>0) & (x<2070)
    if Frame == 'RestFrame':
        lya_z = 0.7
    if Frame == 'ObservedFrame':
        lya_z = 0        
        redshift = np.zeros(len(redshift))
    redshifti = np.array(redshift)[flag_visible]
    sliti = np.array(slit)[flag_visible]
    xi=np.array(x)[flag_visible]
    yi=np.array(y)[flag_visible]
    imagettes = []
    for i in range(len(xi)):
        imagettes.append(image[int(xi[i])-n1:int(xi[i]) +n1,int(yi[i])-n2:int(yi[i]) +n2])
        lambdainf.append(  - (sup - yi[i]) * x2w / (1*redshift[i]+1) + w)
        lambdasup.append( - (inf - yi[i]) * x2w / (1*redshift[i]+1) + w)
    #print(np.array(lambdainf)-np.array(lambdasup))
#    for i in range(len(x)):
#        if (y[i]>inf) & (y[i]<sup) & (x[i]>0) & (x[i]<2070):
#            imagettes.append(image[int(x[i])-n1:int(x[i]) +n1,int(y[i])-n2:int(y[i]) +n2])
#            redshifti.append(redshift[i])
#            sliti.append(slit[i])
#            xi.append(x[i]);yi.append(y[i])
#            lambdainf.append(  (inf - xi) * x2w + w)
#            lambdasup.append(  (sup - xi) * x2w + w)
            #boundsup.append()
    v1,v2 = int(len(xi)/2+1),2
    print('v1,v2=',v1,v2)
    #fig, axes = plt.subplots(v1, v2, figsize=(18,50),sharex=True)
    fig, axes = plt.subplots(v1, v2, figsize=(12.8,70/40*v1),sharex=True, sharey=True)
    xaxis = np.linspace(w-n2*x2w/(1+lya_z), w+n2*x2w/(1+lya_z), 2*n2)
    xinf = np.searchsorted(xaxis,lambdainf)
    xsup = np.searchsorted(xaxis,lambdasup)
    fig.suptitle('Spectra, lambda in %s'%(Frame) ,y=1)
    
    spectras = []
    for i, ax in enumerate(axes.ravel()[1:len(imagettes)+1]): 
        spectra = imagettes[i][:, ::-1].mean(axis=0)
        spectra[:xinf[i]] = np.nan
        spectra[xsup[i]:] = np.nan
        spectras.append(spectra)
        ax.step(xaxis,spectra,
                label = 'Slit: ' + sliti[i] +'\nz = %0.2f'%(redshifti[i])+'\nx,y = %i - %i'%(yi[i],xi[i]))
        ax.axvline(x=lambdainf[i],color='black',linestyle='dotted')
        ax.axvline(x=lambdasup[i],color='black',linestyle='dotted')
        ax.legend()
        ax.set_xlim(xaxis[[0,-1]])
        ax.tick_params(labelbottom=True)
    ax = axes.ravel()[0]
    print(np.array(imagettes).shape)
#    stack = np.nanmean(np.array(spectras),axis=0)
    stack = np.convolve(np.nanmean(np.array(spectras),axis=0),np.ones(kernel)/kernel, mode='same')#[kernel:-kernel]
#    print(np.nanmean(np.array(spectras),axis=0),stack.shape)
    ax.step(xaxis[kernel:-kernel],stack[kernel:-kernel],label = 'Stack',c='orange')  
    ax.legend()
    ax.set_xlim(xaxis[[0,-1]])
    ax.tick_params(labelbottom=True)
    for ax in axes[-1,:]:
        ax.set_xlabel('Wavelength [A] rest frame')
    fig.tight_layout()
    fig.savefig(filename[:-5] + '_Spectras.png')
    if v1>12:
        ScrollableWindow(fig)
    else:
        plt.show()
    detectLine(xaxis[kernel:-kernel],stack[kernel:-kernel], clipping=threshold, window=20)
#    plt.step(xaxis[kernel:-kernel],stack[kernel:-kernel],label = 'Stack',c='orange') 
#    plt.xlabel('Wavelength [A] rest frame')
#    plt.show()
    csvwrite(np.vstack((xaxis,stack)).T, DS9backUp + 'CSVs/%s_SpectraBigRange.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )

    return imagettes    

def detectLine(x,y, clipping=10, window=20):
    """Fit a gaussian
    interger the flux on 1/e(max-min) and then add the few percent calculated by the gaussian at 1/e 
    """
    #from astropy.io import fits
    #x, y = Table.read('/Users/Vincent/DS9BackUp/CSVs/190508-16H24_SpectraBigRange.csv')['col0'][8:-8],Table.read('/Users/Vincent/DS9BackUp/CSVs/190508-16H24_SpectraBigRange.csv')['col1'][8:-8]
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt    
    from scipy.optimize import curve_fit
    n=0.3
    fit = np.polyfit(x[y<y.mean()+n*np.std(y)], y[y<y.mean()+n*np.std(y)] , 1)
    #xmax = x[np.argmax(y)]
    fit_fn = np.poly1d(fit) 

#    popt1, pcov1 = curve_fit(gaussian, x,y-fit_fn(x), p0=[y.max()-y.min(), x[y.argmax()], 5])
  
    texp=1
    limfit =  fit_fn(x)#+gaussian(x, *popt1).max()/np.exp(1)

#    import matplotlib._cntr as cntr
#    c = cntr.Cntr(x, y, z)
#    nlist = c.trace(level, level, 0)
#    CS = nlist[:len(nlist)//2]
    std_signal = np.nanstd((y-limfit)[:100])
    image = np.array([np.zeros(len(y)),y-limfit,np.zeros(len(y))])
    CS = plt.contour(image,levels=np.nanmean(y-limfit) + clipping * std_signal);plt.close()
    #size = [cs[:,0].max() - cs[:,0].min()     for cs in CS.allsegs[0] ]
    if (y-limfit>np.nanmean(y-limfit) + clipping * std_signal).any():
        maxx = np.array([ int(cs[:,0].max())   for cs in CS.allsegs[0] ])
        minx =  np.array([ int(cs[:,0].min())   for cs in CS.allsegs[0] ])
        maxx.sort();minx.sort()
        meanx = np.array((maxx+minx)/2, dtype=int)
    else:
       maxx , minx, meanx = [], [], []
#    index = np.where(maxx>popt1[1])[0][0]#np.argmax(size)
#    print(minx[index],maxx[index])

    y0 = y-fit_fn(x)
    plt.figure()
    plt.xlabel('Spatial direction')
    plt.ylabel('ADU mean value')
    for i, xm in enumerate(meanx):
        popt1, pcov1 = curve_fit(gaussian, x[xm-window:xm+window],y0[xm-window:xm+window], p0=[y0[xm], x[xm], 5])
        plt.plot(x[xm-window:xm+window], gaussian(x[xm-window:xm+window], *popt1) + fit_fn(x[xm-window:xm+window]), label='%i Gaussian Fit, F = %0.2f - sgm=%0.1f'%(i,np.sum(gaussian(x, *popt1)),popt1[-1]))
    #mask = (x>minx[index]) & (x<maxx[index])
    plt.plot(x[meanx],y[meanx],'o',label='Detections')
    plt.plot(x, np.ones(len(x))*np.nanmean(y-limfit) + clipping * std_signal+fit_fn(x), label='Detection level: %i sigma'%(clipping))
    p = plt.plot(x, y , linestyle='dotted',label='Data')#, label='Data, F=%0.2fADU - SNR=%0.2f'%(np.nansum(y0)/texp,gaussian(x, *popt1).max()/np.std(y0[~mask])))
    #plt.fill_between(x[mask],y[mask],y2=fit_fn(x)[mask],alpha=0.2, label='Best SNR flux calculation: F=%0.2f'%(1.155*np.sum(y0[mask])/texp))
    plt.plot(x, fit_fn(x),label='Fitted background',linestyle='dotted',c=p[0].get_color())
    plt.legend()
    plt.show()
    return 

def DS9tsuite(xpapoint, Plot=False):
    """Create an image with subimages where are lya predicted lines and display it on DS9
    """
    #xpapoint = '7f000001:55985'
    path = os.path.dirname(os.path.realpath(__file__))   
    sys.argv.append('');sys.argv.append('')#;sys.argv.append('')#;sys.argv.append('')
    d = DS9(xpapoint)
#    d.set('frame delete all')
#    DictFunction = {'beta':[np.random.beta,1,1],'binomial':[np.random.binomial,10, 0.5],'geometric':[np.random.geometric, 0.5],'pareto':[np.random.pareto, 1],
#                 'poisson':[np.random.poisson, 1],'power':[np.random.power, 1],'rand':[np.random.rand],'standard_exponential':[np.random.standard_exponential],
#                 'standard_gamma':[np.random.standard_gamma, 1],'standard_normal':[np.random.standard_normal],'standard_t':[np.random.standard_t, 1],'randint':[np.random.randint, 1]}
#
#    for law in DictFunction.keys():
#        print (law)
#        sys.argv[3] = law
#        FollowProbabilityLaw(xpapoint)
#    DictFunction = {'pareto':[np.random.pareto],'beta':[np.random.beta,1],'gamma':[np.random.gamma],'geometric':[np.random.geometric],
#                     'power':[np.random.power],'poisson':[np.random.poisson, 1]}#,'binomial':[np.random.binomial,10, 0.5]
#    ApplyRealisation(xpapoint,'gamma')
#    ApplyRealisation(xpapoint,'poisson')
#    FollowProbabilityLaw(xpapoint,'rand')
#    ApplyRealisation(xpapoint,'pareto')
#    FollowProbabilityLaw(xpapoint,'rand')
#    ApplyRealisation(xpapoint,'beta')    
#    ApplyRealisation(xpapoint,'power')
#    FollowProbabilityLaw(xpapoint,'rand')
#    ApplyRealisation(xpapoint,'geometric')      
#    
#    CreateImageFromCatalogObject(xpapoint, nb=100)
#    DS9setup2(xpapoint)
#    DS9originalSettings(xpapoint)
#    
#    
#    DS9inverse(xpapoint)
#    DS9inverse(xpapoint)
#    DS9next(xpapoint)
#    DS9previous(xpapoint)
#    
#    image = CreateImageFromCatalogObject(xpapoint, nb=10)
#    AddHeaderField(xpapoint, field=my_conf.exptime[0], value=10)
#    d.set('regions '+ path + '/Regions/test.reg')
#    DS9Region2Catalog(xpapoint, new_name='/tmp/test.csv')
#    DS9Catalog2Region(xpapoint, name='/tmp/test.csv')
#    d.set('regions delete all')
#    d.set('regions command "box %0.3f %0.3f %0.1f %0.1f# color=red"' % (np.where(image==image.max())[1][0],np.where(image==image.max())[0][0],40,40))
#    SourcePhotometry(xpapoint)
#    DS9rp(xpapoint, Plot=Plot) 
#    sys.argv = []
#    DS9MaskRegions(xpapoint)
#    DS9InterpolateNaNs(xpapoint)
#    d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=red"' % (np.where(image==image.max())[1][0],np.where(image==image.max())[0][0],40,40))
#    DS9Trimming(xpapoint)
#
#    #sys.exit()
#    
#    
#    
#    #d.set('frame new')
#    print('''\n\n\n\n      TEST: diffuse focus test analysis   \n\n\n\n''') 
#    d.set('frame delete all')
#    sys.argv.append('');sys.argv.append('');sys.argv.append('');sys.argv.append('')
#
#    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
#    sys.argv[3] = 'f3'
#    DS9focus(xpapoint, Plot=Plot)
#    
#    print('''\n\n\n\n      TEST: Open    \n\n\n\n''')
#    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
#    BackgroundMeasurement(xpapoint)
#    DS9OverscanCorrection(xpapoint)
#    print('''\n\n\n\n      TEST: Setup   \n\n\n\n''')
#    DS9setup2(xpapoint)
#    
#    
#    
#    
#
#
#    print('''\n\n\n\n      TEST: Visualization Detector   \n\n\n\n''')
#    print('''\n\n\n\n      TEST: Stacking Detector   \n\n\n\n''')
#    sys.argv.append('');sys.argv.append('');sys.argv.append('');sys.argv.append('')
#    sys.argv[3] = ''
#    DS9visualisation_throughfocus(xpapoint)
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
#    sys.argv[3] = '402-404'
#    sys.argv[4] = '407-408'
#    DS9visualisation_throughfocus(xpapoint)
#    DS9stack(xpapoint)    
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
#    sys.argv[3] = '402-404-406'    #sys.argv[4] = '403-401-407'
#    DS9visualisation_throughfocus(xpapoint)
#    DS9stack(xpapoint)    
#
#
#    print('''\n\n\n\n      TEST: Visualization Guider   \n\n\n\n''')
#    print('''\n\n\n\n      TEST: stacking Guider   \n\n\n\n''')
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
#    sys.argv[3] = ''
#    DS9visualisation_throughfocus(xpapoint)
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
#    sys.argv[3] = '18405298-18407298'
#    DS9visualisation_throughfocus(xpapoint)
#    DS9stack(xpapoint)    
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
#    sys.argv[3] = '18405298-18405946-18407582'
#    DS9visualisation_throughfocus(xpapoint)
#    DS9stack(xpapoint)    
#
#    print('''\n\n\n\n      TEST: Next Guider   \n\n\n\n''')
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/guider/images/stack18405298.fits')
#    print('''\n\n\n\n      TEST: Throughfocus Guider   \n\n\n\n''')
#    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (326,902,40))#(812,783.2,40))
#    d.set('regions select all') 
#    sys.argv[3] = ''
#    DS9throughfocus(xpapoint, Plot=Plot)
#    sys.argv[3] = '18405298-18407298'
#    DS9throughfocus(xpapoint, Plot=Plot)
#
#    print('''\n\n\n\n      TEST: Next detector   \n\n\n\n''')
#
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/detector/images/image000404.fits')
#    print('''\n\n\n\n      TEST: Throughfocus detector   \n\n\n\n''')
#
#    DS9next(xpapoint)
#    print('''\n\n\n\n      TEST: Throughfocus detector   \n\n\n\n''')
#    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (1677,1266.2,40))
#    d.set('regions select all') 
#    sys.argv[3] = ''
#    DS9throughfocus(xpapoint, Plot=Plot)
#    sys.argv[3] = '404-408'
#    DS9throughfocus(xpapoint, Plot=Plot)
#
#    print('''\n\n\n\n      TEST: Radial profile   \n\n\n\n''') 
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/TestImage.fits')    
#    d.set('regions command "circle %0.3f %0.3f %0.1f # color=red"' % (1001,500.2,40))
#    d.set('regions select all') 
#    sys.argv[3] = '3'
#    #DS9rp(xpapoint)  
#    sys.argv[3] = ''
#    DS9rp(xpapoint, Plot=Plot)  
#    print('''\n\n\n\n      TEST: Centering spot   \n\n\n\n''') 
#    DS9center(xpapoint, Plot=Plot)  
#
#    print('''\n\n\n\n      TEST: Centering slit   \n\n\n\n''') 
# 
#    
#    d.set('regions delete all') 
#    d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=yellow"' % (100,501,20,10))
#    d.set('regions select all')
#    DS9center(xpapoint, Plot=Plot)  
#    
#    print('''\n\n\n\n      TEST: Show slit regions   \n\n\n\n''') 
#    d.set('frame delete all')
#    #d.set('frame new')
#    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
#    sys.argv[3] = 'f3'
#    Field_regions(xpapoint)
#    d.set('regions delete all') 
#    sys.argv[3] = 'f4-lya'
#    Field_regions(xpapoint)
#    d.set('regions delete all') 
#    sys.argv[3] = 'f1-names'
#    Field_regions(xpapoint)
#    d.set('regions delete all') 
    
    print('''\n\n\n\n      TEST: diffuse focus test analysis   \n\n\n\n''') 
    d.set('frame delete all')
    DS9open(xpapoint,path + '/test/detector/image000075-000084-Zinc-with_dark-121-stack.fits')    
    sys.argv[3] = 'f3'
    DS9focus(xpapoint, Plot=Plot)

    print('''\n\n\n\n      TEST: Imagette lya   \n\n\n\n''') 
    sys.argv[3:5] = 'F3','restframe','1216'
    create_multiImage(xpapoint)

    print('''\n\n\n\n      TEST: Photocounting   \n\n\n\n''') 
    d.set('frame delete all')
    d.set('frame new')
    sys.argv[3] = 5
    sys.argv[4:] = ''
    DS9open(xpapoint,path + '/test/detector/image000827.fits')    
    DS9photo_counting(xpapoint)
    
    print('''\n\n\n\n      TEST: Guider WCS   \n\n\n\n''') 
    d.set('frame delete all')
    d.set('frame new')
    DS9open(xpapoint,path + '/test/guider/stack8102274_pa+119_2018-06-11T06-14-15_wcs.fits')    
    DS9guider(xpapoint)  

    print('''\n\n\n\n      END OF THE TEST: EXITED OK   \n\n\n\n''') 

    return

    
    
#    DS9next(xpapoint)
#    #create a circle center on a line
#    d.set('regions command "circle %i %i %0.1f # color=red"' % (1843.93,615.27,40))
#    d.set('regions select all')
#    DS9center(xpapoint)
#    DS9rp(xpapoint)
#    DS9throughfocus(xpapoint)
#    DS9visualisation_throughfocus(xpapoint)
#    d.set('frame delete all')
#    d.set('frame new')
#    DS9open(xpapoint,'/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/FBGuider2018/stack8103716_pa+078_2018-06-11T06-21-24_new.fits')
#    DS9guider(xpapoint)
#    d.set('frame delete all')
#    d.set('frame new')
#    DS9open(xpapoint,'/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180614/photon_counting/image000016.fits')
#    DS9photo_counting(xpapoint)
#    d.set('frame delete all')
#    d.set('frame new')
#    DS9open(xpapoint,'/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018/AIT-Optical-FTS-201805/180612/image-000025-000034-Zinc-with_dark-161-stack.fits')
#    create_multiImage(xpapoint)
#    print('Test completed: OK')


def Field_regions(xpapoint, mask=''):
    """Display on DS9 some region files. GS and holes location if guider images,
    slits' locations if detector iamges.
    """
    #from astropy.io import fits
    d = DS9(xpapoint)
    #d.set("regions format ds9")
    #d.set("regions system image")
    path = d.get("file")
    ImageName = os.path.basename(path)
    if (ImageName[:5].lower() == 'image') or (ImageName[:5]== 'Stack'):
        Type = 'detector'
    if ImageName[:5] == 'stack':
        Type = 'guider'
    print ('Type = ', Type)
    try:
        slit_dir = resource_filename('DS9FireBall', 'Slits')
    except:
        slit_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Slits')
        
    print('Type = ', Type)
    if mask == '':
        try:
            mask = sys.argv[3]#'f3 names'#sys.argv[3]
        except:
            mask = ''
    print ('Masks = ', mask)
    mask = mask.lower()
    if ('d' in mask):
        print('ok')
        filename = os.path.join(slit_dir, 'DetectorFrame.reg' )
        d.set("region {}".format(filename))
    if ('g' in mask):
        filename = os.path.join(slit_dir, 'GuiderFrame.reg' )
        d.set("region {}".format(filename))
    if Type == 'detector':
        print('ok')
        if ('f1' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F1_119_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F1_119_Zn.reg')
            #if ('name' in mask):
                #   filename2 = os.path.dirname(os.path.realpath(__file__)) + '/Slits/F1_119_names.reg'        
        if ('f2' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F2_-161_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F2_-161_Zn.reg')
            if ('name' in mask):
                filename2 = os.path.join(slit_dir, 'F2_-161_names.reg')
        if ('f3' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F3_-121_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F3_-121_Zn.reg')
            if ('name' in mask):
                filename2 = os.path.join(slit_dir, '3_-121_names.reg')
        if ('f4' in mask):
            if ('lya' in mask):
                filename = os.path.join(slit_dir, 'F4_159_Lya.reg')
            else:
                filename = os.path.join(slit_dir, 'F4_159_Zn.reg')
            #if ('name' in mask):
                #filename2 = os.path.join(slit_dir, 'F4_159_names.reg')
        if ('grid' in mask):
            filename = os.path.join(slit_dir, 'grid_Zn.reg' )
        d.set("region {}".format(filename))
        d.set('contour load ' + os.path.join(slit_dir, 'detector_deffect.ctr' ))
        if d.get('tile')=='yes':
            d.set('frame last')
            for i in range(int(d.get('frame'))-1):
                d.set('frame next')
                d.set('regions ' + filename)
                d.set('contour load ' + os.path.join(slit_dir, 'detector_deffect.ctr' ))



    if Type == 'guider':
        if mask == 'no':
            d.set('contour clear')
            if d.get('tile')=='yes':
                d.set('frame last')
                for i in range(int(d.get('frame'))-1):
                    d.set('frame next')
                    d.set('contour clear')
                
        else:
            d.set('contour clear')
            #d.set('contour yes')

#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']
#            guidingstars[0] = header['CX0','CX0','USE0']

#            for line in header:                
#                if line in ['USE0','USE1','USE2','USE3','USE4','USE5','USE6','USE7']:
#                    print(line)
            header = d.get_fits()[0].header
            pa = int(header['ROTENC'])
            #DS9
            print('Position angle = ',pa)
            if (pa>117) & (pa<121):
                name1 = os.path.join(slit_dir, 'GSF1.reg')
                name2 = os.path.join(slit_dir, 'F1.ctr')
            if (pa>-163) & (pa<-159):
                name1 = os.path.join(slit_dir, 'GSF2.reg')
                name2 = os.path.join(slit_dir, 'F2.ctr')
            if (pa>-123) & (pa<-119):
                name1 = os.path.join(slit_dir, 'GSF3.reg')
                name2 = os.path.join(slit_dir, 'F3.ctr')
            if (pa>157) & (pa<161):
                name1 = os.path.join(slit_dir, 'GSF4.reg')
                name2 = os.path.join(slit_dir, 'F4.ctr')
            d.set('regions ' + name1)
            d.set('contour load ' + name2)
            if d.get('tile')=='yes':
                d.set('frame last')
                for i in range(int(d.get('frame'))-1):
                    d.set('frame next')
                    d.set('regions ' + name1)
                    d.set('contour load ' + name2)

                    guidingstars = np.zeros((8,3))
                    #namename = getfilename(d)#
                    header = d.get_fits()[0].header
                    for i in range(8):
                        guidingstars[i] = header['CY%i'%(i)],header['CX%i'%(i)],header['USE%i'%(i)]
                        if (int(guidingstars[i,2]) ==  257) or (int(guidingstars[i,2]) ==  1):
                            d.set('regions command "box %0.3f %0.3f 8 8  # color=yellow"' % (guidingstars[i,1],guidingstars[i,0]))
                print('guiding stars = ',guidingstars)
            else:
                guidingstars = np.zeros((8,3))
                header = d.get_fits()[0].header
                for i in range(8):
                    guidingstars[i] = header['CY%i'%(i)],header['CX%i'%(i)],header['USE%i'%(i)]
                    if (int(guidingstars[i,2]) ==  257) or (int(guidingstars[i,2]) ==  1):
                        d.set('regions command "box %0.3f %0.3f 8 8  # color=yellow"' % (guidingstars[i,1],guidingstars[i,0]))
                print('guiding stars = ',guidingstars)

    try:
        print('Putting regions, filename = ', filename)
    except UnboundLocalError:
        pass            

    try:
        d.set('regions {}'.format(filename2))
    except:
        pass
    print('Test completed: OK')
    return

def DS9XYAnalysis(xpapoint, DS9backUp = DS9_BackUp_path):
    """Analyze images from XY calibration, just need to zoom and the 
    selected spot for each images
    """
    from matplotlib import pyplot as plt
    d = DS9(xpapoint)    
    filename = getfilename(d)#filename = d.get('file')
    d.set('frame first')
    n1 = int(d.get('frame'))
    
    d.set('frame last')
    n2 = int(d.get('frame'))
    n=n2-n1+1
    print('Number of frame = ',n)
    d.set('frame first')
    Centers = np.zeros((n,2))
    SlitPos = np.zeros((n,2))
    SlitDetectedPos = np.zeros((n,2))
    DetectedRegFile = glob.glob(os.path.dirname(filename) + '/*detected.reg')

    mask = sys.argv[3]
    print(n)
    for frame in range(n):
        x, y = d.get('pan image').split(' ')
        xc, yc = int(float(x)), int(float(y))
        d.set('regions delete all')
        d.set('regions command "circle %0.3f %0.3f %0.3f # color=yellow"' % (xc,yc,30))
        d.set('regions select all')
        xnew, ynew = DS9center(xpapoint,Plot=False)#;plt.show()
        Centers[frame,:] = xnew, ynew
        if len(DetectedRegFile) == 1:
            d.set('regions file ' + DetectedRegFile[0])
        d.set('frame next')
    d.set('frame last')
    Field_regions(xpapoint, mask = mask)
        

    d.set('frame first')
    slitnames = []
    MappedRegions = []
    for frame in range(n):    
        d.set('regions select all')
        regions = d.get('regions').split('\n')
        SlitsOnImage = []
        for region in regions:
            if 'circle' in region:
                MappedRegions.append(region)
        regionsxy = np.zeros((len(MappedRegions),2))

        for i,region in enumerate(MappedRegions):
            a, reg, b = region.replace('(',')').split(')')
            a,name,b = b.replace('{','}').split('}')
            try:
                x,y,a1,a1,a1,a1,a1,a1,a1,a1,a1 = reg.split(',')
            except ValueError:
                x,y,r = reg.split(',')
            x,y = float(x), float(y)
            regionsxy[i,:] = x, y
            SlitsOnImage.append(name)            
        distance = np.sqrt(np.square((Centers[frame,:] - regionsxy)).sum(axis=1))
        index = np.argmin(distance)
        SlitPos[frame] = regionsxy[index]
        slitnames.append(SlitsOnImage[index])            
        print('Slit Detected positions are :')
        print(repr(SlitDetectedPos))    

    print('Slit predicted positions are :')
    print(repr(SlitPos))

    print('Centers of the spots are :')
    print(repr(Centers))
    
    plt.figure()#figsize=(5,7))
    plt.title('Distances to slit')
    plt.xlabel('x pixel detector')
    plt.ylabel('y pixel detector')
    Q = plt.quiver(Centers[:,0],Centers[:,1],SlitPos[:,0] - Centers[:,0],SlitPos[:,1] - Centers[:,1],scale=30,label='Dist to mapped mask')

    plt.quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
    plt.axis('equal')
    plt.figtext(.15, .65,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitPos[:,0] - Centers[:,0]).mean(), (SlitPos[:,1] - Centers[:,1]).mean(),(SlitPos[:,0] - Centers[:,0]).std(), (SlitPos[:,1] - Centers[:,1]).std(), max((SlitPos[:,0] - Centers[:,0]), key=abs), max((SlitPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'black')
    plt.figtext(.15, .35,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitDetectedPos[:,0] - Centers[:,0]).mean(), (SlitDetectedPos[:,1] - Centers[:,1]).mean(),(SlitDetectedPos[:,0] - Centers[:,0]).std(), (SlitDetectedPos[:,1] - Centers[:,1]).std(), max((SlitDetectedPos[:,0] - Centers[:,0]), key=abs), max((SlitDetectedPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'blue')
    for i in range(len(slitnames)):
        plt.text(SlitPos[i,0],SlitPos[i,1], slitnames[i], color='red',fontsize=14)
    plt.grid(linestyle='dotted')
    plt.legend()
    plt.savefig(os.path.dirname(filename) + '/XYCalibration.png')
    plt.show()
#    fig, ax = plt.subplots(1,2,figsize=(8, 8), sharex = True, sharey=True)
#    fig.suptitle('Distances to slit')
#    ax[0].set_xlabel('x pixel detector')
#    ax[0].set_ylabel('y pixel detector')
#    Q = ax[0].quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - Centers[:,0],SlitPos[:,1] - Centers[:,1],scale=30)
#    #Q = plt.quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - Centers[:,0],0,color='blue',scale=30)#, width=0.003)#, scale=Q)
#    #Q = plt.quiver(SlitPos[:,0],SlitPos[:,1],0,SlitPos[:,1] - Centers[:,1],color='blue',scale=30)#, width=0.003)#, units='x')
#    ax[0].quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
#    ax[0].axis('equal')
#    ax[0].set_xlim(1000,2000)
#    ax[0].set_ylim(0,2000)
#    #plt.xlim((1000,2000))
#    #ax[0].ylim((0,2000))
#    plt.figtext(.15, .65,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitPos[:,0] - Centers[:,0]).mean(), (SlitPos[:,1] - Centers[:,1]).mean(),(SlitPos[:,0] - Centers[:,0]).std(), (SlitPos[:,1] - Centers[:,1]).std(), max((SlitPos[:,0] - Centers[:,0]), key=abs), max((SlitPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'blue')
#    for i in range(len(slitnames)):
#        ax[0].text(SlitPos[i,0],SlitPos[i,1], slitnames[i], color='red',fontsize=14)
#    ax[0].grid(linestyle='dotted')
#
#
#    Q = ax[1].quiver(SlitPos[:,0],SlitPos[:,1],SlitPos[:,0] - SlitDetectedPos[:,0],SlitPos[:,1] - SlitDetectedPos[:,1],scale=30)
#    ax[1].quiverkey(Q, 0.2, 0.2, 2, '2 pixels', color='r', labelpos='E',coordinates='figure')
#    ax[1].axis('equal')
#    ax[1].set_xlim(1000,2000)
#    ax[1].set_ylim(0,2000)
#    #plt.xlim((1000,2000))
#    #ax[0].ylim((0,2000))
#    plt.figtext(.15, .65,'Xmean =  %0.1f\nYmean = %0.1f \nstd(X) =  %0.1f\nstd(y) = %0.1f\nMax(X) =  %0.1f\nMax(y) = %0.1f' % ((SlitPos[:,0] - Centers[:,0]).mean(), (SlitPos[:,1] - Centers[:,1]).mean(),(SlitPos[:,0] - Centers[:,0]).std(), (SlitPos[:,1] - Centers[:,1]).std(), max((SlitPos[:,0] - Centers[:,0]), key=abs), max((SlitPos[:,1] - Centers[:,1]), key=abs)), fontsize=12, color = 'blue')
#    for i in range(len(slitnames)):
#        ax[1].text(SlitPos[i,0],SlitPos[i,1], slitnames[i], color='red',fontsize=14)
#    ax[1].grid(linestyle='dotted')
#
#    plt.savefig(os.path.dirname(filename) + '/XYCalibration.png')
#    plt.show()
    return Centers



def DS9stack(xpapoint):
    """This function aims at stacking images with or without dark substraction (no interpolation of the dark for now).
    By pressing ’s’ a window ask for the detector images to stack: just write the first and the last number of the
    images (separated by ’-’ no space as in figure 5). If the images are not consecutive you have to write all the
    numbers (eg. ’13-26-86-89’). Do the same for the dark, put only one number if there is only one dark image
    or press directly ’ok’ if there is not any dark image. The stack image is saved in the repository containing the
    DS9 image with the number of the images stacked and the dark images used + ’stack.fits’.
    """
    from astropy.io import fits
    from .focustest import stackImages
    d = DS9(xpapoint)
    filename = d.get("file")
    fitsimage = fits.open(filename)[0]
    if fitsimage.header['BITPIX'] == -32:
        Type = 'guider'  
    else:
        Type = 'detector' 
    print('Type = ', Type)

    if Type == 'guider':
        files = Charge_path(filename)
        print(files)
        n = len(files)
        image = fitsimage
        lx,ly = image.data.shape
        stack = np.zeros((lx,ly,n),dtype='uint16')
        print('\nReading fits files...')
        numbers = []
        for i,file in enumerate(files):
            name = os.path.basename(file) 
            numbers.append(int(name[5:12]))
            with fits.open(file) as f:
                stack[:,:,i] = f[0].data
        image.data = np.nanmean(stack,axis=2)
        fname = os.path.dirname(filename)                
        filename = '{}/StackedImage_{}-{}.fits'.format(fname, np.array(numbers).min(), np.array(numbers).max())
        image.writeto(filename ,overwrite=True)
        print('Images stacked:',filename)            
        
    if Type == 'detector':
        entry = sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# sys.argv[3]#'325-334'# 
        print('Images to stack = ', entry)
        try:
            number_dark = sys.argv[4] #''#sys.argv[4] #''#'sys.argv[4] #'365'#'365-374'#''#sys.argv[4] 
        except:
            number_dark = ''
        print('Dark to remove = ', number_dark)
        numbers = entry.split('-')
    
        if len(numbers) == 2:
            print('2 numbers given for the stack')
            n1, n2 = entry.split('-')
            n1, n2 = int(n1), int(n2)
            numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
        print('Numbers used: {}'.format(numbers))           
    
        path = os.path.dirname(filename)
        try:
            d1,d2 = number_dark.split('-') 
            print('2 numbers given for the dark')
            print(d1,d2)
            dark = stackImages(path,all=False, DS=0, function = 'mean', numbers=np.arange(int(d1),int(d2)), save=True, name="Dark")[0]
            Image, filename = stackImages(path,all=False, DS=dark, function = 'mean', numbers=np.array([int(number) for number in numbers]), save=True, name="DarkSubtracted")#, name="Dark_{}-{}".format(int(d1),int(d2)))
        except ValueError:
            d1 = number_dark
            if d1 == '':
                dark = 0
                print('No dark')
                Image, filename = stackImages(path,all=False, DS=dark, function = 'mean', numbers=numbers, save=True, name="NoDark")
            else:
                dark = fits.open(path + '/image%06d.fits' % (int(d1)))[0].data
                Image, filename = stackImages(path,all=False, DS=dark, function = 'mean', numbers=numbers, save=True, name="Dark_{}".format(d1))

    d.set('tile yes')
    d.set('frame new')
    d.set("file {}".format(filename))   
    d.set("lock frame physical")
    return

def FlagHighBackgroundImages(stack, std=1, config=my_conf):
    """
    """
    det_part = stack[:,1053:2133,:]
    det_mean = np.nanmean(det_part)
    det_std = np.nanstd(det_part)
    means = np.nanmean(det_part, axis=(0,1))
    index = means < det_mean + 1 * det_std
    print(index)
    return index
    

def DS9stack_new(xpapoint, Type='mean', dtype=float, std=False):
    d = DS9(xpapoint)
    filename = d.get("file")
    Type = sys.argv[3]
    clipping = sys.argv[4]
    #if len(sys.argv) > 3+2: paths = Charge_path_new(filename, entry_point=3+2)
    paths = Charge_path_new(filename) if len(sys.argv) > 5 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    if 'int' in Type:
        dtype = 'uint16'
    if 'float' in Type:
        dtype = 'float'
    if 'std' in Type:
        std = True
    if 'median' in  Type:
        Type = np.nanmedian
    else:
        Type = np.nanmean
    image, name = StackImagesPath(paths, Type=Type,clipping=float(clipping), dtype=dtype, std=std) 
    d.set('tile yes')
    d.set('frame new')
    d.set("file {}".format(name))   
    d.set("lock frame physical")
    return 

def StackImagesPath(paths, Type=np.nanmean,clipping=3, dtype=float, fname='', std=False, DS9_BackUp_path=DS9_BackUp_path, config=my_conf):
    """Stack images of the files given in the path
    """
    from astropy.io import fits
    import re
    Xinf, Xsup, Yinf, Ysup = my_conf.physical_region
    stds = np.array([np.nanstd(fits.open(image)[0].data[Xinf:Xsup, Yinf:Ysup]) for image in paths])
    plt.figure()
    plt.hist(stds)
    plt.title( 'Stds - M = %0.2f  -  Sigma = %0.3f'%(np.nanmean(stds), np.nanstd(stds)));plt.xlabel('Stds');plt.ylabel('Frequecy')
    plt.savefig(DS9_BackUp_path +'Plots/%s_Outputs_%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),'stds'))
    index = stds < np.nanmean(stds) + clipping * np.nanstd(stds) 
    if std is False:
        stds = np.ones(len(paths[index]))
    else:
        stds /= np.nansum(stds[index])
        print('std = ', stds)
    n=len(paths)
    paths.sort()
    image = fits.open(paths[0])
    #name_images = [os.path.basename(path) for path in paths]
    lx,ly = image[0].data.shape
    stack = np.zeros((lx,ly),dtype=dtype)
    if std:
        for i,file in enumerate(paths[index]):
            try:
                with fits.open(file) as f:
                    print(file, stds[i])
                    #f[0].data[~np.isfinite(f[0].data)] = stack[~np.isfinite(f[0].data)]
                    stack[:,:] += f[0].data/stds[i]
                    del(f)
            except TypeError as e:
                print(e)
                n -= 1
        stack = stack/n
    elif Type == np.nanmedian:
        stack = np.array(Type(np.array([fits.open(file)[0].data for file in paths[index]]), axis=0), dtype=dtype)
    elif Type == np.nanmean:
        stack = Type(np.array([fits.open(file)[0].data for file in paths[index]]),dtype=dtype, axis=0)
    numbers = [int(re.findall(r'\d+',os.path.basename(filename))[-1]) for filename in paths[index]]
    images = '-'.join(list(np.array(numbers,dtype=str))) 
        
#    if cond_bckd:
#        index = FlagHighBackgroundImages(stack, std=1)
#    else:
#        index = True
        
#    if np.isfinite(stack).all():
#        stack.dtype = 'uint16'
    
    #print('All these images have not been taken into account because of hight background', '\n'.join(list(np.array(name_images)[~index]))  )
    #image[0].data = np.nanmean(stack[:,:,index],axis=2)#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    image[0].data = stack#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    image[0].header['STK_NB'] = images# '-'.join(re.findall(r'\d+',images))#'-'.join(list(np.array(name_images)[index]))   
    try:
        name = '{}/StackedImage_{}-{}{}.fits'.format(os.path.dirname(paths[0]), int(os.path.basename(paths[0])[5:5+6]), int(os.path.basename(paths[-1])[5:5+6]),fname)
    except ValueError:
        name = '{}/StackedImage_{}-{}{}'.format(os.path.dirname(paths[0]), os.path.basename(paths[0]).split('.')[0], os.path.basename(paths[-1]),fname)       
    print('Image saved : %s'%(name))
    fitswrite(image[0], name)
    print('n = ', n)

    return image  , name



#def StackImagesPath_old(paths, cond_bckd=True, fname='', std=None):
#    """Stack images of the files given in the path
#    """
#    from astropy.io import fits
#    import re
#    if std is None:
#        std = np.ones(len(paths))
#    else:
#        print(std)
#        #std *= np.nansum(1/std)#std=np.array([100,200],dtype='float64')
#        std /= np.nanmean(std)#std=np.array([100,200],dtype='float64')
#        print('std = ', std)
#    n=len(paths)
#    paths.sort()
#    image = fits.open(paths[0])
#    name_images = [os.path.basename(path) for path in paths]
#    lx,ly = image[0].data.shape
#    if 'CRv' in paths[0]:
#        stack = np.zeros((lx,ly,n))#,dtype='uint16')
#    else:
#        stack = np.zeros((lx,ly,n),dtype='uint16')
#    for i,file in enumerate(paths):
#        with fits.open(file) as f:
#            print(os.path.basename(file),'%0.3f'%(std[i]))
#            stack[:,:,i] = f[0].data/std[i]
#    if cond_bckd:
#        print('deleting images with high backgorund')
#        index = FlagHighBackgroundImages(stack, std=1)
#        image[0].data = np.nanmean(stack[:,:,index],axis=2)#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
#    else:
#        index = True
#        image[0].data = np.nanmean(stack[:,:,:],axis=2)#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
##    if np.isfinite(stack).all():
##        stack.dtype = 'uint16'
#    #images = '\n'.join(list(np.array(name_images)[index])) 
#    #print('All these images have not been taken into account because of hight background', '\n'.join(list(np.array(name_images)[~index]))  )
#    #image[0].header['STK_NB'] =  '-'.join(re.findall(r'\d+',images))#'-'.join(list(np.array(name_images)[index]))   
#    print(1)
#    try:
#        name = '{}/StackedImage_{}-{}.fits'.format(os.path.dirname(paths[0]), int(os.path.basename(paths[0])[5:5+6]), int(os.path.basename(paths[-1])[5:5+6]))
#    except ValueError:
#        name = '{}/StackedImage_{}-{}'.format(os.path.dirname(paths[0]), os.path.basename(paths[0]).split('.')[0], os.path.basename(paths[-1]))       
#    print(2)
#    fitswrite(image[0], name)
#    print('Image saved : %s'%(name))
#    return image  

def DS9focus(xpapoint, Plot=True):
    """Apply focus test class to the image
    """
    from .focustest import Focus  
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get("file")
    try:
        entry = sys.argv[3] #f3 -121'#sys.argv[3] #''#sys.argv[4] #''#'sys.argv[4] #'365'#'365-374'#''#sys.argv[4] 
    except:
        F = Focus(filename = filename, HumanSupervision=False, source='Zn', shape='holes', windowing=False, peak_threshold=50,plot=Plot)
        d.set('regions {}'.format(filename[:-5]+'detected.reg'))
    else:
        try:
            mask, pa = entry.split('-')
            if mask.lower() == 'grid':
                print('On passe par ici')
                F = Focus(filename = filename, quick=True, figsize=12,windowing=True, mask='grid' , plot=False,sources='holes',date=5)
            else:                
                F = Focus(filename = filename, quick=False, threshold = [7], fwhm = [9,12.5],
                      HumanSupervision=False, reversex=False, source='Zn',
                      shape='slits', windowing=True, mask=mask.capitalize(), pa=int(pa) ,MoreSources=0,peak_threshold=50,plot=Plot)
        except ValueError as e:
            print(e)
            mask = entry
            F = Focus(filename = filename, quick=False, threshold = [7], fwhm = [9,12.5],
                  HumanSupervision=False, reversex=False, source='all',
                  shape='slits', windowing=True, mask=mask.capitalize(),MoreSources=0,peak_threshold=50,plot=Plot)
    
        d.set('regions {}'.format(filename[:-5]+'detected.reg'))
    return F


def DS9throughslit(xpapoint, DS9backUp = DS9_BackUp_path, config=my_conf):
    """How to use: Open one an image of the through focus which is close to the focus. Click on region. Then click
    precisely on what you think is the centre of the PSF. Select the region you created and press t (throughfocus)
    or go in analysis menu: Through slit analysis. This will open a dialog box that asks what is the number of
    the images of the through focus. You can either put the numbers (eg: "10-21"). If the throughslit is not done
    straightforward then you will have to enter all the number of the images in the right order (eg "10-12-32-28-21").
    This will pick up the center of the region you put, compute the sum of the pixels in a 30pix box after
    removing the background in an anulus around the source. Then it plots the flux with respect to the throughslit
    and give the number of the image with the maximum of flux in title. Note that the x axis only give the index
    of the image, not its number.
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from .focustest import estimateBackground
    from .focustest import Gaussian
    from scipy.optimize import curve_fit

    print('''\n\n\n\n      START THROUGHSLIT \n\n\n\n''')
    try:
        entry = sys.argv[3]#'2-15'#'2-7-8-9-11-14-15'#'2-15'#'2-4-6-8-9'#sys.argv[3]
        numbers = entry.split('-')#[::-1]
        if len(numbers) == 2:
            n1,n2 = entry.split('-')
            n1,n2 = int(n1), int(n2)
            numbers = np.arange(int(min(n1,n2)),int(max(n1,n2)+1)) 
        print(numbers)
        
    except IndexError:
        n1=''
        n2=''
        numbers=None
    d = DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    path = []
    if (type(numbers) == list) or (type(numbers) == np.ndarray):
        print('Specified numbers are integers, opening corresponding files ...')
        for number in numbers:
            print (number)
            #path = os.path.dirname(filename) + '/image%06d.fits' % (number)
            path.append(os.path.dirname(filename) + '/image%06d.fits' % (int(number)))
            x = [int(i) for i in numbers]
        print (x)
    else:
        print('Not numbers, taking all the .fits images from the current repository')
        path = glob.glob(os.path.dirname(filename) + '/*.fits')
        x = np.arange(len(path))
    #    with fits.open(path) as f:
    #        files.append(f[0].data)        
    #    os.path.dirname(filename)
    #path = np.sort(path)
    path.sort()
    #print("\n".join(path)) 
    a = getregion(d)
    radius = 15
    print('Sum pixel is used (another estimator may be prefarable)')
    fluxes=[]
    
    for file in path:
        print (file)
        fitsfile = fits.open(file)[0]
        image = fitsfile.data
        #plt.figure(figsize=(sizefig,sizefig))
        #plt.imshow(image[int(a.yc)-radius:int(a.yc)+radius, int(a.xc)-radius:int(a.xc)+radius])#;plt.colorbar();plt.show()
        #plt.show()
        subimage = image[int(a.yc)-radius:int(a.yc)+radius, int(a.xc)-radius:int(a.xc)+radius]
        background = estimateBackground(image, [a.xc,a.yc], radius=30, n=1.8)
        #flux = np.nansum(image[center[0]-n:center[0]+n,center[1]-n:center[1]+n])-np.nansum(image[center_bg[0]-n:center_bg[0]+n,center_bg[1]-n:center_bg[1]+n])
        flux = np.nansum(subimage - background) #- estimateBackground(image, center, radius, n_bg)
        fluxes.append(flux)
    fluxesn = (fluxes - min(fluxes)) / max(fluxes - min(fluxes))    
    x = np.arange(len(path))+1
    popt, pcov = curve_fit(Gaussian, x, fluxesn, p0=[1, x.mean(),3,0])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
    xl = np.linspace(x.min(),x.max(),100)
    maxf = xl[np.where(Gaussian(xl,*popt)==np.nanmax(Gaussian(xl,*popt)))[0][0]]#[0]
    plt.figure()
    plt.plot(x, fluxesn,'o',label='data')
    plt.plot(xl, Gaussian(xl,*popt),'--',label='Gaussian fit')
    plt.legend()
    plt.plot(np.linspace(maxf, maxf, len(fluxes)), fluxesn/max(fluxesn))
    plt.grid(linestyle='dotted')
    plt.xlabel('# image')
    plt.title('Best image : {}'.format(maxf))
    plt.ylabel('Sum pixel') 
    name = 'Through slit analysis\n%0.3f - %s - %s'%(maxf,[int(a.xc),int(a.yc)],fitsfile.header['DATE'])
    print(name) 
    plt.title(name)
    plt.savefig(os.path.dirname(file) + '/' + name + '.jpg')
    plt.show()
    csvwrite(np.vstack((x, fluxesn,Gaussian(x,*popt))).T, DS9backUp + 'CSVs/%s_ThroughSlit.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )

    return 

def DS9snr(xpapoint):
    """Compute a rough SNR on the selected spot. Needs to be updates
    """
    #from astropy.io import fits
    #from .focustest import create_DS9regions2
    n1 = 1.2
    n2 = 1.8
    d = DS9(xpapoint)
    #filename = getfilename(d)#ffilename = d.get("file")
    fitsfile = d.get_fits()#fits.open(filename)
    image = fitsfile[0].data
    
    region = getregion(d)
    y, x = np.indices((image.shape))
    r = np.sqrt((x - region.xc)**2 + (y - region.yc)**2)  
    r = r.astype(np.int)
    if hasattr(region, 'r'):
        signal = image[r<region.r]
        maskBackground = (r >= n1 * region.r) & (r <= n2 * region.r)
    else:
        print('Need to be a circular region')
    signal_max = np.nanpercentile(signal,95)
    noise = np.sqrt(np.nanvar(image[maskBackground]))
    background = np.nanmean(image[maskBackground])
    SNR = (signal_max - background) / noise
    print('Signal = ', signal_max)
    print('Background = ', background)
    print('Noise = ', noise)
    print('SNR = ', SNR)
    #d.set('regions command "text %i %i # text={SNR = %0.2f}"' % (region.xc+10,region.yc+10,SNR))
    try:
        os.remove('/tmp/centers.reg')
    except OSError:
        pass
    create_DS9regions2([region.xc+n2*region.r+10],[region.yc], form = '# text',
                       save=True,color = 'yellow', savename='/tmp/centers',
                       text = ['SNR = %0.2f' % (SNR)])
    d.set('regions /tmp/centers.reg')
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=yellow"' % (region.xc,region.yc,n1 * region.r))
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=yellow"' % (region.xc,region.yc,n2 * region.r))
    return
    


def DS9createImage(xpapoint):
    """
    Should test: centering (gaussian, slit), radial profile, stack dark, SNR, 
    throughfocus, 
    Du coup dans l'image il faut quil y ai: du bruit, une gaussienne, une fente convoluee avec une gausienne
    Radial profile: OK
    centering spot: OK
    centering slit: OK
    through focus
    throughlit
    """
    import matplotlib; matplotlib.use('TkAgg')  
   # import matplotlib.pyplot as plt
    #from astropy.io import fits
    from .focustest import ConvolveSlit2D_PSF
    d=DS9(xpapoint)
#    n=20
    lx, ly = 1000, 1000#fitstest[0].data.shape
    x, y = np.arange(lx), np.arange(ly)
    xy = np.meshgrid(x,y)
    new_image = np.zeros((lx,ly))
    for xi, ampi in zip((np.linspace(100,1900,10)),(np.linspace(10,1000,10))):
        slit = (1/0.006)*ConvolveSlit2D_PSF(xy, ampi, 3, 9, int(xi), 1500, 3,3).reshape(ly,lx).T
        new_image = new_image + slit# + gaussian2
        #plt.imshow(slit[int(xi)-n:int(xi)+n,1500-n:1500+n]);plt.plt.colorbar();plt.show()
        #plt.imshow(new_image[int(xi)-n:int(xi)+n,1500-n:1500+n]);plt.colorbar();plt.show()
    for xi, ampi in zip((np.linspace(100,1900,10)),(np.linspace(10,1000,10))):
        gaussian = twoD_Gaussian(xy, ampi, int(xi), 1000, 5, 5, 0).reshape(ly,lx).T
        new_image = new_image + gaussian
    d.set_np2arr(new_image)
    return 
    
    




def DS9meanvar(xpapoint):
    """Compute mean standard deviation and skewness in this parth of the image
    """
    #from astropy.io import fits
    from scipy import stats
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup, Xinf, Xsup]
    print(area)
    image = d.get_fits()[0].data[area[0]:area[1],area[2]:area[3]]#picouet
    print ('Image : {}'.format(filename))
    print ('Mean : {}'.format(image.mean()))
    print ('Standard deviation : {}'.format(image.std()))
    print ('Skewness: {}'.format(stats.skew(image,axis=None)))
    return

 

def DS9lock(xpapoint):
    """Lock all the images in DS9 together in frame, smooth, limits, colorbar
    """
    d = DS9(xpapoint)
    #lock = d.get("lock scalelimits")
    l = sys.argv[-5:]
    ll = np.array(l,dtype='U3')
    print(l,ll)
    l = np.array(l,dtype=int)
    ll[l==1]='yes'
    ll[l==0]='no'
    if ll[0] == 'no':
        d.set("lock frame %s"%(ll[0]))
    else:
        d.set("lock frame physical")
        
    d.set("lock scalelimits  %s"%(ll[1]))
    if ll[2] == 'no':
        d.set("lock crosshair %s"%(ll[2]))
    else:
        d.set("lock crosshair physical")
    d.set("lock smooth  %s"%(ll[3]))
    d.set("lock colorbar  %s"%(ll[4]))
#    if lock == 'yes':
#        d.set("lock frame no")
#        d.set("lock scalelimits no")
#        d.set("lock crosshair no")
#        d.set("lock smooth no")
#        d.set("lock colorbar no")
#    if lock == 'no':
#        d.set("lock frame physical")
#        d.set("lock scalelimits yes")
#        d.set("crosshair lock physical")
#        d.set("lock crosshair physical")
#        d.set("lock smooth yes")
#        d.set("lock colorbar yes")
    return


def DS9inverse(xpapoint):
    """Inverse the image in DS9, can be used to then do some positive gaussian fitting, etc
    """
    #from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    fitsfile = d.get_fits()[0]#fits.open(filename)[0]

    image = fitsfile.data
    new_image = - image + image.max()
    fitsfile.data = new_image
    fitsfile.data = new_image
    fitsfile.writeto(filename[:-5] + '_Inverse.fits',overwrite=True)
    d.set('frame new')
    d.set('file {}'.format(filename[:-5] + '_Inverse.fits'))
    return


def DS9Update(xpapoint,Plot=True):
    """Always display last image of the repository and will upate with new ones
    """
    import time
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    flag=0
    while flag<1: 
        files = glob.glob(os.path.dirname(filename)+ '/*.fits')
        latest_file = max(files, key=os.path.getctime)
        if os.path.getctime(latest_file)>os.path.getctime(filename):
            filename = latest_file
            d.set('file ' + filename)
            print('New file!\nDisplaying : ' + latest_file)
        else:
            print('No file more recent top display')
            print('Waiting ...')
        time.sleep(5)
    return
        
        
def Lims_from_region(region):
    """Return the pixel locations limits of a DS9 region
    """
    try:
        xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
    except:
        print('Region is not a box, exracting box using the radius')
        xc, yc, r = int(region.xc), int(region.yc), int(region.r)
        w, h = r, r
    print('W = ', w)
    print('H = ', h)
    if w <= 2:
        w = 2		      
    if h <= 2:																			
        h = 2
    Yinf = int(np.floor(yc - h/2 -1))
    Ysup = int(np.ceil(yc + h/2 -1))
    Xinf = int(np.floor(xc - w/2 -1))
    Xsup = int(np.ceil(xc + w/2 -1))
    print('Xinf, Xsup, Yinf, Ysup = ', Xinf, Xsup, Yinf, Ysup)
    print('data[%i:%i,%i:%i]'%(Yinf, Ysup,Xinf, Xsup))
    return Xinf, Xsup, Yinf, Ysup


def DS9center(xpapoint,Plot=True):
    """How to use:
    This function can be used in two different ways:
    •To find the center of a circular spot (eg. fiber in autocoll for XY calibration): Click on region and select
    Circle shape (default). Then click precisely on what you think is the centre of the PSF. Select the region
    you created and press c (centering) or go in analysis menu: centering.
    •To find the center of a slit: Click on region and select box. Then put the box where you think the image
    of the slit is relatively flat (a few pixels). Select the region you created and press c (centering) or go in
    analysis menu: centering.
    In the first case, the function opens a dialog box that will ask the size of the fiber. Give it in pixels +
    Enter or press directly Enter if you only want to fit a 2D gaussian function. Then, the function will pick up the
    center of the region you put, compute the barycenter of the image and fit the spot by a 2D gaussian convolved
    with a 2D disk of the diameter you gave (or only a gaussian). To be sure that the fit worked correctly, after it
    converged it plots the line and the column of the PSF at its center with the computed 2D function that fits the
    spot (see figure 10).
    In the second case, the function does two 1D analysis. First it will pick up the box region and stack it in
    both directions to create two profiles. Then in will fit the two profiles by the 1D convolution of a sigma-free
    gaussian and a size-free box and plot it (see figure 11).
    In both cases it will return the computed center of the spot and print a new region with the computed
    center. It will also give the characteristics of the spot.
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    #from astropy.io import fits
    from .focustest import ConvolveBoxPSF
    from .focustest import create_DS9regions
    #from .focustest import create_DS9regions2
    from .focustest import estimateBackground
    from scipy.optimize import curve_fit
    from .focustest import Gaussian
    d = DS9(xpapoint)#DS9(xpapoint)
    #filename = getfilename(d)#ffilename = d.get("file")
    region = getregion(d)

    if hasattr(region, 'h'):
        xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
        print('W = ', w)
        print('H = ', h)
        if w <= 2:
            w = 2
        if h <= 2:
            h = 2
        Xinf = int(np.floor(yc - h/2 -1))
        Xsup = int(np.ceil(yc + h/2 -1))
        Yinf = int(np.floor(xc - w/2 -1))
        Ysup = int(np.ceil(xc + w/2 -1))
        data = d.get_fits()[0].data
        imagex = data[Xinf-15:Xsup+15,Yinf:Ysup].sum(axis=1)
        imagey = data[Xinf:Xsup,Yinf-15:Ysup+15].sum(axis=0)
        model = ConvolveBoxPSF
        x = np.arange(-len(imagex)/2,len(imagex)/2)
        y = np.arange(-len(imagey)/2,len(imagey)/2)
        try:
            poptx, pcovx = curve_fit(model, x, imagex, p0=[imagex.max(), 20, 0., 10., np.median(imagex)])#,  bounds=bounds)
            popty, pcovy = curve_fit(model, y, imagey, p0=[imagey.max(), 10, 0., 10., np.median(imagey)])#,  bounds=bounds)
            ampx, lx, x0x, sigma2x, offsetx = poptx
            ampy, ly, x0y, sigma2y, offsety = popty
        except RuntimeError:
            print('Optimal parameters not found: Number of calls to function has reached maxfev = 1400.')
        else:
            print('Poptx = ', poptx)
            print('Popty = ', popty)
        
        newCenterx = xc + x0y#popty[2]
        newCentery = yc + x0x#poptx[2]
        d.set('regions delete select')
        print('''\n\n\n\n     Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f] \n\n\n\n''' % (region.yc,region.xc,newCentery,newCenterx))
        #d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=yellow"' % (newCenterx+1,newCentery+1,region.w,region.h))
        d.set('regions command "box %0.3f %0.3f %0.2f %0.2f # color=yellow"' % (newCenterx,newCentery,region.w,region.h))
        #d.set('regions command "circle %i %i %0.1f # color=yellow"' % (newCenterx+1,newCentery+1,2))
        #d.set('regions command "circle %0.3f %0.3f %0.2f # color=yellow"' % (newCenterx,newCentery,2))
        try:
            os.remove('/tmp/centers.reg')
        except OSError:
            pass
        #create_DS9regions([newCenterx],[newCentery], radius=2, save=True, savename="/tmp/centers", form=['circle'], color=['yellow'], ID='test')
        create_DS9regions2([newCenterx],[newCentery-15], form = '# text',
                           save=True,color = 'yellow', savename='/tmp/centers',
                           text = ['%0.2f - %0.2f' % (newCenterx,newCentery)])
        d.set('regions /tmp/centers.reg')
        
        fig, axes = plt.subplots(2, 1, sharex=True)#, figsize=(8,6))
        axes[0].plot(x,imagex, 'bo', label='Spatial direction')
        axes[1].plot(y,imagey, 'ro', label='Spectral direction')
        axes[0].plot(x, model(x, *poptx), color='b')#,label='Spatial direction')
        axes[1].plot(y, model(y, *popty), color='r')#,label='Spatial direction')
        #plt.ylabel('Fitted profiles')
        axes[0].set_ylabel('Spatial direction');axes[1].set_ylabel('Spectral direction')
        axes[0].plot(x, Gaussian(x, imagex.max()-offsetx, x0x, sigma2x, offsetx), ':b',label='Deconvolved PSF') # Gaussian x, amplitude, xo, sigma_x, offset
        #axes[0].plot(x, Gaussian(x, ampx, x0x, sigma2x, offsetx), ':b',label='Deconvolved PSF') # Gaussian x, amplitude, xo, sigma_x, offset
        xcc = x - x0x
        #axes[0].plot(x, np.piecewise(x, [xc < -lx, (xc >=-lx) & (xc<=lx), xc>lx], [offsetx, ampx + offsetx, offsetx]), ':r', label='Slit size') # slit (UnitBox)
        axes[0].plot(x, np.piecewise(x, [xcc < -lx, (xcc >=-lx) & (xcc<=lx), xcc>lx], [offsetx, imagex.max() , offsetx]), ':r', label='Slit size') # slit (UnitBox)
        axes[0].plot([x0x, x0x], [imagex.min(), imagex.max()])
        axes[1].plot(y, Gaussian(y, imagey.max() - offsety, x0y, sigma2y, offsety), ':b',label='Deconvolved PSF') # Gaussian x, amplitude, xo, sigma_x, offset
        xcc = x - x0y
        axes[1].plot(x, np.piecewise(x, [xcc < -ly, (xcc >=-ly) & (xcc<=ly), xcc>ly], [offsety, imagey.max(), offsety]), ':r', label='Slit size') # slit (UnitBox)
        axes[1].plot([x0y, x0y], [imagey.min(), imagey.max()])
        plt.figtext(0.66,0.65,'Sigma = %0.2f +/- %0.2f pix\nSlitdim = %0.2f +/- %0.2f pix\ncenter = %0.2f +/- %0.2f' % ( np.sqrt(poptx[3]), np.sqrt(np.diag(pcovx)[3]/2.) , 2*poptx[1],2*np.sqrt(np.diag(pcovx)[1]), x0x, np.sqrt(np.diag(pcovx)[2])),bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})
        plt.figtext(0.67,0.25,'Sigma = %0.2f +/- %0.2f pix\nSlitdim = %0.2f +/- %0.2f pix\ncenter = %0.2f +/- %0.2f' % ( np.sqrt(popty[3]), np.sqrt(np.diag(pcovy)[3]/2.) , 2*popty[1],2*np.sqrt(np.diag(pcovy)[1]), x0y, np.sqrt(np.diag(pcovy)[2])),bbox={'facecolor':'red', 'alpha':0.2, 'pad':10})
        if Plot:
            plt.show()
        
    if hasattr(region, 'r'):
        xc, yc, r = int(region.xc), int(region.yc), int(region.r)
        Xinf = yc - r -1#int(region.yc - region.r)
        Xsup = yc + r -1#int(region.yc + region.r)
        Yinf = xc - r -1#int(region.xc - region.r)
        Ysup = xc + r -1#int(region.xc + region.r)
        data = d.get_fits()[0].data
        image = data[Xinf:Xsup,Yinf:Ysup]
        print('2D fitting with 100 microns fibre, to be updated by allowing each fiber size')
        background =  estimateBackground(data,[region.yc,region.xc],20,1.8 )
        image = image - background
        lx, ly = image.shape
        x = np.linspace(0,lx-1,lx)
        y = np.linspace(0,ly-1,ly)
        x, y = np.meshgrid(x,y)
        yo,xo = np.where(image == image.max())#ndimage.measurements.center_of_mass(image)
        maxx, maxy = xc - (lx/2 - xo), yc - (ly/2 - yo)
        print ('maxx, maxy = {}, {}'.format(maxx,maxy))
#        d.set('regions command "circle %i %i %0.1f # color=yellow"' % (maxy,maxx,1))
#        try:
#            os.remove('/tmp/centers.reg')
#        except OSError:
#            pass
#        create_DS9regions2([maxx],[maxy-10], form = '# text',
#                           save=True,color = 'yellow', savename='/tmp/centers',
#                           text = ['%0.2f - %0.2f' % (maxx,maxy)])
#        d.set('regions /tmp/centers.reg')
        bounds = ([1e-1*np.nanmax(image), xo-10 , yo-10, 0.5,0.5,-1e5], [10*np.nanmax(image), xo+10 , yo+10, 10,10,1e5])#(-np.inf, np.inf)#
        Param = (np.nanmax(image),int(xo),int(yo),2,2,np.percentile(image,15))
        print ('bounds = ',bounds)
        print('\nParam = ', Param)
        try:
            popt,pcov = curve_fit(twoD_Gaussian,(x,y),image.flat,
                                  Param,bounds=bounds)
            print('\nFitted parameters = ', popt)
        except RuntimeError:
            print('Optimal parameters not found: Number of calls to function has reached maxfev = 1400.')
            sys.exit() 
        fit = twoD_Gaussian((x,y),*popt).reshape((ly,lx))
        #imshow(twoD_Gaussian((x,y), *Param).reshape((ly,lx)));colorbar()
        #imshow(image);colorbar()
        #imshow(fit);colorbar()
        newCenterx = xc - (lx/2 - popt[1])#region.xc - (lx/2 - popt[1])
        newCentery = yc - (ly/2 - popt[2])#region.yc - (ly/2 - popt[2])
        print('''\n\n\n\n     Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f] \n\n\n\n''' % (region.yc,region.xc,newCentery,newCenterx))

        print(np.diag(pcov))

        #plt.show()
#        d.set('regions command "text %i %i # text={%0.2f}"' % (newCenterx+10,newCentery+10,newCentery))

        d.set('regions delete select')

        #d.set('regions command "circle %0.2f %0.2f %0.2f # color=yellow"' % (newCenterx,newCentery,5))
        try:
            os.remove('/tmp/centers.reg')
        except OSError:
            pass
#        create_DS9regions2([newCenterx],[newCentery-10], form = '# text',
#                           save=True,color = 'yellow', savename='/tmp/centers',
#                           text = ['%0.2f - %0.2f' % (newCenterx,newCentery)])
        create_DS9regions([newCenterx-1],[newCentery-1], radius=5, save=True, savename="/tmp/centers", form=['circle'], color=['white'], ID=[['%0.2f - %0.2f' % (newCenterx,newCentery)]])
        
        d.set('regions /tmp/centers.reg')
        if Plot:
            plt.plot(image[int(yo), :], 'bo',label='Spatial direction')
            plt.plot(fit[int(yo), :],color='b')#,label='Spatial direction')
            plt.plot(image[:,int(xo)], 'ro',label='Spatial direction')
            plt.plot(fit[:,int(xo)],color='r')#,label='Spatial direction')
            plt.ylabel('Fitted profiles')
            plt.figtext(0.66,0.55,'Sigma = %0.2f +/- %0.2f pix\nXcenter = %0.2f +/- %0.2f\nYcenter = %0.2f +/- %0.2f' % ( np.sqrt(popt[3]), np.sqrt(np.diag(pcov)[3]/2.), lx/2 - popt[1] , np.sqrt(np.diag(pcov)[1]), ly/2 - popt[2], np.sqrt(np.diag(pcov)[2])),bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})
            plt.legend()
    return newCenterx, newCentery

def t2s(h,m,s,d=0):
    """Transform hours, minutes, seconds to seconds [+days]
    """
    return 3600 * h + 60 * m + s + d*24*3600


def calc_emgainGillian(image, area=[0,-1,0,-1],plot_flag=False,save=False, DS9backUp = DS9_BackUp_path, config=my_conf):
        """Compute Bias and EMgain on a high voltage image taken with an EMCCD
        """
        from astropy.io import fits
        gain =  1.78
        if (type(image) == np.str) or (type(image) == np.str_) or (type(image) == str):        
            fitsimage = fits.open(image)[0]
            img_data = fitsimage.data
            ysize, xsize = img_data.shape
            img_section = img_data[area[0]:area[1],area[2]:area[3]]
        else:
            fitsimage = image#fits.open(image)[0]
            ysize, xsize = fitsimage.data.shape
            img_section = fitsimage.data[area[0]:area[1],area[2]:area[3]]
        nbins = 1000
        readnoise = 60
        #gain = float(gain)

        # Histogram of the pixel values
        n, bins = np.histogram(np.array(img_section[np.isfinite(img_section)]), bins = nbins, range=(-200,11800))
        bin_center = 0.5 * (bins[:-1] + bins[1:])
        #y0 = np.nanmin(n)

        n_log = np.log(n)

        # What is the mean bias value?
        idx = np.where(n == n.max())
        bias = bin_center[idx][0]
        n_bias = n[idx][0]
    
        # Range of data in which to fit the Gaussian to calculate sigma
        bias_lower = bias - float(1.5) * readnoise
        bias_upper = bias + float(2.0) * readnoise
        try:
            idx_lower = np.where(bin_center >= bias_lower)[0][0]
            idx_upper = np.where(bin_center >= bias_upper)[0][0]
        except IndexError:
            idx_lower = 0
            idx_upper = 900
        print(len(n))
        #gauss_range = np.where(bin_center >= bias_lower)[0][0]
    
        valid_idx = np.where(n[idx_lower:idx_upper] > 0)
        try:
            amp, x0, sigma = gaussianFit(bin_center[idx_lower:idx_upper][valid_idx], n[idx_lower:idx_upper][valid_idx], [n_bias, bias, readnoise])
        except RuntimeError as e:
            print(e)
            D = {'EMG_hist':0,'bias':0,'sigma':0,'frac_lost':0,
             'image':0,my_conf.gain[0]:0,'bias':0,'sigma':0,'bin_center':0,
             'n':0,'xlinefit':0,'ylinefit':0,'xgaussfit':0,
             'ygaussfit':0,'n_bias':0,'n_log':0,'threshold0':0,
             'threshold55':0,'exposure':0, 'gain':0,'temp': 0}
            return D
        #plt.figure()
        #plt.plot(bin_center[idx_lower:idx_upper], n[idx_lower:idx_upper], 'r.')
        #plt.show()

        # Fitted frequency values
        xgaussfit = np.linspace(bin_center[idx_lower], bin_center[idx_upper], 1000)
        #print xgaussfit
        ygaussfit = gaussian(xgaussfit, amp, x0, sigma)
        #print ygaussfit

        # Define index of "linear" part of the curve
        Tmin = 10
        Tmax = 30
        threshold_min = bias + (float(Tmin) * sigma)
        threshold_max = bias + (float(Tmax) * sigma)
        threshold_55 = bias + (float(5.5) * sigma)

        #print threshold_max

        # Lines for bias, 5.5*sigma line

        n_line = n_log.size
        zeroline = np.zeros([n_line], dtype = np.float32)
        threshold0 = zeroline + int(bias)
        threshold55 = zeroline + int(bias + 5.5*sigma)
        #thresholdmin = zeroline + int(threshold_min)
        #thresholdmax = zeroline + int(threshold_max)
        try:
            idx_threshmin = np.array(np.where(bin_center >= threshold_min))[0,0]
        except IndexError as e:
            print('Error, threshold min was %i and first bin is %i-%i'%(threshold_min,bin_center[0],bin_center[1]))
            print(e)
            D = {'EMG_hist':0,'bias':0,'sigma':0,'frac_lost':0,
             'image':0,my_conf.gain[0]:0,'bias':0,'sigma':0,'bin_center':0,
             'n':0,'xlinefit':0,'ylinefit':0,'xgaussfit':0,
             'ygaussfit':0,'n_bias':0,'n_log':0,'threshold0':0,
             'threshold55':0,'exposure':0, 'gain':0,'temp': 0}
            return D
        try:
            idx_threshmax = np.array(np.where(bin_center >= threshold_max))[0,0]
        except IndexError:
            print('Error, threshold max was %i and last bin was %i'%(threshold_max,bin_center.max()))
            idx_threshmax = np.array(np.where(bin_center == bin_center.max()))[0,0]
            
        valid_idx = np.where(n[idx_threshmin:idx_threshmax] > 0)

        slope, intercept = fitLine(bin_center[idx_threshmin:idx_threshmax][valid_idx], n_log[idx_threshmin:idx_threshmax][valid_idx])

        # Fit line
        #xlinefit = np.linspace(bias-20*sigma, bin_center[idx_threshmax], 1000)
        xlinefit = np.linspace(bin_center[idx_threshmin], bin_center[idx_threshmax], 1000)
        ylinefit = linefit(xlinefit, slope, intercept)


        emgain = (-1./slope) * (gain)

        #print xlinefit
        embias_idx = np.where(xlinefit >= bias)
        #print embias_idx
        embias_y1 = ylinefit[embias_idx[0][0]]
        #print embias_y1
        em5_idx = np.where(xlinefit >= bias+(5.5*sigma))
        #print em5_idx
        em5_y1 = ylinefit[em5_idx[0][0]]
        #print em5_y1

        total_counts = 0.5*(threshold_max-bias)*(embias_y1)
        thres_counts = 0.5*(threshold_max-threshold_55)*(em5_y1)
        #print total_counts, thres_counts
        frac_lost = (total_counts-thres_counts)/thres_counts
        print(frac_lost)
        exposure, gain, temp = fitsimage.header[my_conf.exptime[0]], fitsimage.header[my_conf.gain[0]], fitsimage.header[my_conf.temperature[0]]
#        plot_hist2(image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,
#                   ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp,plot_flag=plot_flag, ax=None)
        try:
            fitsimage.header['C_egain'] = emgain
        except ValueError as e:
            print (e)
            fitsimage.header['C_egain'] = -1
        fitsimage.header['C_bias'] = bias
        fitsimage.header['C_sigR0'] = sigma
        try:
            fitsimage.header['C_flost'] = frac_lost
        except ValueError:
            pass
        if save:
            fitswrite(fitsimage, image)
        D = {'EMG_hist':emgain,'bias':bias,'sigma':sigma,'frac_lost':frac_lost,
             'image':image,'emgain':emgain,'bias':bias,'sigma':sigma,'bin_center':bin_center,
             'n':n,'xlinefit':xlinefit,'ylinefit':ylinefit,'xgaussfit':xgaussfit,
             'ygaussfit':ygaussfit,'n_bias':n_bias,'n_log':n_log,'threshold0':threshold0,
             'threshold55':threshold55,'exposure':exposure, 'gain':gain,'temp': temp}
        #return (emgain,bias,sigma,frac_lost), (image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit, ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp)
        return D




def DS9DetectCosmics(xpapoint,filen=None,T=6*1e4, config=my_conf):
    from astropy.io import fits
    #from .focustest import create_DS9regions
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
#    try:
#        region = getregion(d)
#        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
#        area = [Yinf, Ysup,Xinf, Xsup]
#        print(Yinf, Ysup,Xinf, Xsup)
#    except ValueError:
#        area = [0,-1,1053,2133]
    try:
        region = getregion(d)
    except ValueError:
        Xinf, Xsup, Yinf, Ysup = my_conf.physical_region#[0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)  
        
    for filename in path:
        print(filename)
        #from astropy.io import fits
        fitsimage =  fits.open(filename)[0] 
        image = fitsimage.data
        cosmicRays = detectCosmics(image,T=T,area=area)
        if len(cosmicRays)>11000:
            print('Too many cosmics to detected, Please make sure the image is not saturated or change the detection threshold.')
            return 1
        else:
            #print('Deleting duplications...')
            #cosmicRays = delete_doublons_CR(cosmicRays,dist=5)
            print('Assigning detections to single cosmic-ray events...')
            cosmicRays = assign_CR(cosmicRays,dist=10)
    #        plot(cosmicRays[cosmicRays['id']==-1]['xcentroid'],cosmicRays[cosmicRays['id']==-1]['ycentroid'],'.')
            print('Distinguishing  Dark From CR...')
            cosmicRays = DistinguishDarkFromCR(cosmicRays, T=T, number=2)
            print('Determine cosmic-ray front')
            cosmicRays = Determine_front(cosmicRays)
            a=cosmicRays[cosmicRays['front']==1]
            a = a[(a['xcentroid']>1000) & (a['xcentroid']<2200)] 
            a['filename'] = path
            name = os.path.dirname(filename)+'/' +os.path.basename(filename)[:-5]+'.CRv.fits'
            csvwrite(a,name[:-5] + '.csv')
        
    if len(path) < 2:    
        create_DS9regions([list(a['xcentroid'])],[list(a['ycentroid'])], form=['circle'], radius=10, save=True, 
                   savename='/tmp/cr', color = ['yellow'],ID=None)
        d.set('region delete all')
        d.set('region {}'.format('/tmp/cr.reg'))                                        
    return cosmicRays







def DS9removeCRtails2(xpapoint,filen=None, length=20, config=my_conf):
    """Replace cosmic ray tails present in the image by NaN values and save
    it with the same name with .CVr.fits. Some primary particules hits might
    not be completely removed and would need some other specific care with
    the ReplaceByNaNs function
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    try:
        length = float(sys.argv[3])
    except IndexError:
        length = length
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
        
    try:
        region = getregion(d)
    except ValueError:
        Xinf, Xsup, Yinf, Ysup = my_conf.physical_region#[0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)   
        
    for filename in path:
        print(filename, length)
        result, name, cosmicRays = RemoveCRtail(filename,T=6*1e4, length=length, area=area)#2.1*e4, area=area) 
    if (len(path)<2) & (len(cosmicRays)>1):    
        d.set('region delete all')
        d.set('region {}'.format('/tmp/cr.reg'))                                        
        d.set('frame new')
        d.set('tile yes')
        d.set('lock frame physical')
        d.set('file ' + name)


    return cosmicRays



def DS9Catalog2Region(xpapoint, name=None, x='xcentroid', y='ycentroid'):
    """
    """
    from astropy.table import Table
    #from .focustest import create_DS9regions2
    if xpapoint is not None:
        d = DS9(xpapoint)
    if name is None:
        name = sys.argv[3]
    try:
        x, y = sys.argv[4].replace(',','-').split('-')
    except:
        pass
    cat = Table.read(name)
    print(cat)
    create_DS9regions2(cat[x],cat[y], radius=3, form = 'circle', save=True,color = 'yellow', savename='/tmp/centers')
    if xpapoint is not None:
        d.set('regions /tmp/centers.reg')    
    return cat , '/tmp/centers.reg'

def DS9Region2Catalog(xpapoint, name=None, new_name=None):
    """
    """
    from astropy.table import Table
    d = DS9(xpapoint)
    if name is not None:
        d.set('regions ' + name)
    regions = getregion(d, all=True)
    filename = d.get('file')
    try:
        x, y = np.array([r.xc for r in regions]), np.array([r.yc for r in regions])
    except AttributeError:
        x, y = np.array([r.xc for r in [regions]]), np.array([r.yc for r in [regions]])
    cat = Table((x-1,y-1),names=('xcentroid','ycentroid'))    
    print(cat)
    if new_name is None:
        new_name = '/tmp/regions.csv'
    csvwrite(cat, new_name)
    return cat


def DS9MaskRegions(xpapoint, length = 20):
    """
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    try:
        top, bottom, left, right = np.array(sys.argv[3:7], dtype=int)
    except (IndexError, ValueError) as e:
        print(e)
        top, bottom, left, right = 0, 0, 4, 0
    print('top, bottom, left, right = ',top, bottom, left, right)
    #if len(sys.argv) > 4+3: path = Charge_path_new(filename, entry_point=4+3)
    path = Charge_path_new(filename) if len(sys.argv) > 7 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    cosmicRays = DS9Region2Catalog(xpapoint, new_name='/tmp/cat.csv')
    cosmicRays['front'] = 1
    cosmicRays['dark'] = 0
    cosmicRays['id'] = np.arange(len(cosmicRays))
    print('path = ', path)
    for filename in path:
        print(filename, length)
        fitsimage, name = MaskRegions2(filename, regions=cosmicRays, top=top, bottom=bottom, left=left, right=right)
    if (len(path)<2):    
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)       
    return fitsimage, name


def MaskRegions2(filename, top, bottom, left, right, regions=None):
    """
    """
    from astropy.io import fits
    fitsimage = fits.open(filename)[0]

    #print(cosmicRays)
    maskedimage = MaskCosmicRays2(fitsimage.data, cosmics=regions, top=top, bottom=bottom, left=left, right=right, all=True, cols=None)
    fitsimage.data = maskedimage
    name = os.path.dirname(filename) + '/'+ os.path.basename(filename)[:-5] + '_masked.fits'
    if not os.path.exists(os.path.dirname(name) ):
        os.makedirs(os.path.dirname(name))
    fitswrite(fitsimage, name)
    return fitsimage, name

def MaskCosmicRays2(image, cosmics, top=0, bottom=0, left=4, right=0, cols=None,all=False):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    from tqdm import tqdm
    y, x = np.indices((image.shape))
    image = image.astype(float)
    if all is False:   
        cosmics = cosmics[(cosmics['front']==1) & (cosmics['dark']<1)]
    if cols is None:
        for i in tqdm(range(len(cosmics))):#range(len(cosmics)):
            image[(y>cosmics[i]['ycentroid']-bottom-0.1) & (y<cosmics[i]['ycentroid']+top+0.1) & (x<cosmics[i]['xcentroid']+right+0.1) & (x>-left-0.1 + cosmics[i]['xcentroid'])] = np.nan
    else:
        print('OK')
        for i in tqdm(range(len(cosmics))):#range(len(cosmics)):
            image[(y>cosmics[i]['ycentroid']-bottom-0.1) & (y<cosmics[i]['ycentroid']+top+0.1) & (x<cosmics[i]['xcentroid']+right+0.1) & (x>-left-0.1 + cosmics[i]['xcentroid'])] = np.nan
    return image

#def MaskRegions(filename, length, regions=None):
#    """
#    """
#    from astropy.io import fits
#    fitsimage = fits.open(filename)[0]
#
#    #print(cosmicRays)
#    maskedimage = MaskCosmicRays(fitsimage.data, cosmics=regions,length=length, all=True, cols=None)
#    fitsimage.data = maskedimage
#    name = os.path.dirname(filename) + os.path.basename(filename)[:-5] + '_masked.fits'
#    if not os.path.exists(os.path.dirname(name) ):
#        os.makedirs(os.path.dirname(name))
#    fitswrite(fitsimage, name)
#    return fitsimage, name



def MaskCosmicRaysCS(image, cosmics,all=False, size=None):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    from tqdm import tqdm
    y, x = np.indices((image.shape))
    ly, lx = image.shape
    image = image.astype(float)
    cosmics = cosmics[(cosmics['min_y']>0)&(cosmics['min_y']<ly) & (cosmics['max_y']>0)&(cosmics['max_y']<ly) &  (cosmics['max_x']>0)&(cosmics['max_x']<lx)]
    if size is None:
        for i in tqdm(range(len(cosmics))):
            
            image[(y>cosmics[i]['min_y']) & (y<cosmics[i]['max_y']) & (x<cosmics[i]['max_x']+cosmics[i]['size_opp']) & (x>-cosmics[i]['size'] + cosmics[i]['max_x'])] = np.nan#0#np.inf#0#np.nan
    elif size>1000:
        for i in tqdm(range(len(cosmics))):
            mask = (y>cosmics[i]['min_y']) & (y<cosmics[i]['max_y']) & (x<cosmics[i]['max_x']+cosmics[i]['size_opp'])
        #    if len(image[mask])<50*3000:
            image[mask] = np.nan
    else:
        for i in tqdm(range(len(cosmics))):
            mask = (y>cosmics[i]['min_y']) & (y<cosmics[i]['max_y']) & (x<cosmics[i]['max_x']+cosmics[i]['size_opp']) & (x>-size + cosmics[i]['max_x'])
           # if len(image[mask])<50*3000:
            image[mask] = np.nan        
    return image

def DS9removeCRtails_CS(xpapoint, threshold=60000,n=3,size=0, config=my_conf):
#    from astropy.io import fits
#    import matplotlib.pyplot as plt
#    from astropy.table import Table, Column
    #from .focustest import create_DS9regions
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
    try:
        region = getregion(d)
    except ValueError:
        Xinf, Xsup, Yinf, Ysup = my_conf.physical_region
        #area = [0,2069,1053,2133]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)   
   
    for filename in path:
        fitsimage, name, a = removeCRtails_CS(filename=filename, threshold=threshold,n=n,size=size, area=area, create=False)
    if (len(path)<2) & (len(a)>1):    
        create_DS9regions([list(a['max_x'])],[list(a['mean_y'])], form=['circle'], radius=10, save=True, 
                   savename='/tmp/cr', color = ['yellow'],ID=[list(a['len_contour'])])
        d.set('region delete all')
        d.set('regions /tmp/cr.reg')
        d.set('frame new')
        d.set('file '+ name)
        d.set('tile yes')
    return  

def removeCRtails_CS(filename, area, threshold=15000,n=3,size=0,DS9backUp = DS9_BackUp_path, create=False): 
    from astropy.io import fits
    import matplotlib.dates as mdates
    import re
    from astropy.table import Table
    locator = mdates.HourLocator(interval=1)
    locator.MAXTICKS = 50000
    ax=plt.gca()
    ax.xaxis.set_minor_locator(locator)
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    header = fitsimage.header
    print(threshold)
    CS = ax.contour(image, levels=threshold, colors='white', alpha=0.5)
    names = ('id','sizex','sizey','len_contour','max_x','min_y', 'max_y')
    cosmics = Table(np.zeros((len(CS.allsegs[0]),len(names))),names=names)
    cosmics['id'] = np.arange(len(CS.allsegs[0]))
    cosmics['sizex'] = [cs[:,0].max() - cs[:,0].min()     for cs in CS.allsegs[0] ]
    cosmics['sizey'] = [cs[:,1].max() - cs[:,1].min()     for cs in CS.allsegs[0] ]
    cosmics['len_contour'] = [ len(cs[:,1])   for cs in CS.allsegs[0] ]
    cosmics['max_x'] = [ int(cs[:,0].max()+n*1)   for cs in CS.allsegs[0] ]
    cosmics['min_y'] = [ int(cs[:,1].min()-n*1)   for cs in CS.allsegs[0] ]
    cosmics['max_y'] = [ int(cs[:,1].max()+n*2)   for cs in CS.allsegs[0] ]
    cosmics['mean_y'] = [ int((cs[:,1].max()+cs[:,1].max())/2)   for cs in CS.allsegs[0] ]
    cosmics['size'] = [ n*50   for cs in CS.allsegs[0] ]
    cosmics['size_opp'] = [ n*1 for cs in CS.allsegs[0] ]
    cosmics[my_conf.exptime[0]] = header[my_conf.exptime[0]]
    cosmics[my_conf.gain[0]] = header[my_conf.gain[0]]
    cosmics['number'] = re.findall(r'\d+',os.path.basename(filename))[-1]

    
    cosmics = cosmics[(cosmics['max_x']>500) & (cosmics['max_x']<2500)]



    mask1 = (cosmics['len_contour']<=20)
    mask2 = (cosmics['len_contour']>20) & (cosmics['len_contour']<2000)
    mask3 = (cosmics['len_contour']>50) & (cosmics['len_contour']<2000)
    mask4 = (cosmics['len_contour']>200) & (cosmics['len_contour']<2000)
    cosmics['size'][mask2] = n*200
    cosmics['size'][mask3] = n*3000
    if size>1000:
        cosmics['size'] = [ n*3000   for cs in CS.allsegs[0] ]
    cosmics['size_opp'][mask4] = n*3000
    cosmics['min_y'][(cosmics['len_contour']>200) & (cosmics['len_contour']<2000)] -= n*20
    cosmics['max_y'][(cosmics['len_contour']>200) & (cosmics['len_contour']<2000)] += n*20
    a = cosmics
    maskedimage = MaskCosmicRaysCS(image, cosmics=cosmics)
    savename = DS9backUp + 'CSVs/Cosmics_' +os.path.basename(filename)[:-5] + '.csv'
    csvwrite(a,savename)  
    
    print('%i cosmic rays found!'%(len(cosmics)))
    
    fitsimage.data = maskedimage
    name = os.path.dirname(filename)+'/CosmicRayFree/' +os.path.basename(filename)[:-5]+'.CRv_cs.fits'
    fitsimage.header['N_CR'] = len(cosmics)
    fitsimage.header['N_CR1'] = len(cosmics[mask1])
    fitsimage.header['N_CR2'] = len(cosmics[mask2])
    fitsimage.header['N_CR3'] = len(cosmics[mask3])
    fitsimage.header['N_CR4'] = len(cosmics[mask4])
    if 'NAXIS3' in fitsimage.header:
        fits.delval(filename,'NAXIS3')
        print('2D array: Removing NAXIS3 from header...')  
    fits.setval(filename, 'N_CR', value = len(cosmics))
    fits.setval(filename, 'N_CR1', value = len(cosmics[mask1]))
    fits.setval(filename, 'N_CR2', value = len(cosmics[mask2]))
    fits.setval(filename, 'N_CR3', value = len(cosmics[mask3]))
    fits.setval(filename, 'N_CR4', value = len(cosmics[mask4]))
    try:
        fitsimage.header['MASK'] = 100 * float(np.sum(~np.isfinite(maskedimage[:,1053:2133]))) / (maskedimage[:,1053:2133].shape[0]*maskedimage[:,1053:2133].shape[1])
        fits.setval(filename, 'MASK', value = 100 * float(np.sum(~np.isfinite(maskedimage[:,1053:2133]))) / (maskedimage[:,1053:2133].shape[0]*maskedimage[:,1053:2133].shape[1]))
    except ZeroDivisionError:
        fitsimage.header['MASK'] = 100 * float(np.sum(~np.isfinite(maskedimage))) / (maskedimage.shape[0]*maskedimage.shape[1])    
        fits.setval(filename, 'MASK', value = 100 * float(np.sum(~np.isfinite(maskedimage))) / (maskedimage.shape[0]*maskedimage.shape[1]))
    if create:          
        fitswrite(fitsimage,name)
        return fitsimage, name, cosmics
    elif len(cosmics)>0:
        fitswrite(fitsimage,name)
        return fitsimage, name, cosmics
    else:    #testv
        if not os.path.exists(os.path.dirname(name)):
            os.makedirs(os.path.dirname(name))
        symlink_force(filename, name)
        return fitsimage, filename, cosmics


def symlink_force(target, link_name):
    import errno
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

 
def RemoveCRtail(path,T=6*1e4, length=1000,area=[0,2069,1053,2133], config=my_conf):
    """
    """
    from astropy.io import fits
    fitsimage =  fits.open(path)[0] #d.get_fits()[0]#fits.open(path)[0] 
    image = fitsimage.data
    cosmicRays = detectCosmics(image,T=T, area=area)
    if len(cosmicRays)==0:
        return fitsimage, 'no cosmic detected', cosmicRays
    else:
        if len(cosmicRays)>11000:
            print('Too many cosmics to detected, Please make sure the image is not saturated or change the detection threshold.')
            return 1
        else:
            #print('Deleting duplications...')
            #cosmicRays = delete_doublons_CR(cosmicRays,dist=5)
            print('Assigning detections to single cosmic-ray events...')
            cosmicRays = assign_CR(cosmicRays,dist=20)
            print('Distinguishing  Dark From CR...')
            cosmicRays = DistinguishDarkFromCR(cosmicRays, T=T, number=2)
            print('Determine cosmic-ray front')
            cosmicRays = Determine_front(cosmicRays)
            a=cosmicRays[cosmicRays['front']==1]
            a['filename'] = path
            #a = a[(a['xcentroid']>1000) & (a['xcentroid']<2200)]             
            create_DS9regions([list(a['xcentroid'])],[list(a['ycentroid'])], form=['circle'], radius=10, save=True, 
                       savename='/tmp/cr', color = ['yellow'],ID=[list(a['id'])])
            print('Masking cosmic-ray events...')
            maskedimage = MaskCosmicRays(image, cosmics=cosmicRays,all=False, cols=1, length=length)
            fitsimage.data = maskedimage
            name = os.path.dirname(path)+'/CosmicRayFree/' +os.path.basename(path)[:-5]+'.CRv.fits'
            fitsimage.header['N_CR'] = cosmicRays['id'].max()
            fitsimage.header['MASK'] = len(np.where(maskedimage==np.nan)[0])
            fitswrite(fitsimage,name)
            csvwrite(a,name[:-5] + '.csv')
            return fitsimage, name, cosmicRays


def DS9DetectHotPixels(xpapoint, DS9backUp = DS9_BackUp_path, T1=None, T2=None, config=my_conf, nb=200):
    """
    """
    #from astropy.io import fits
    d=DS9(xpapoint)

 
    filename = d.get('file')
    fitsimage = d.get_fits()[0]#fits.open(path)[0] 
    image = fitsimage.data
    try:
        region = getregion(d)
    except ValueError:
        if image.shape == (2069, 3216):
            Xinf, Xsup, Yinf, Ysup = my_conf.physical_region
        else:
            Xinf, Xsup, Yinf, Ysup = [0,-1,0,-1]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)   


    if T1 is None:
        try:
            entry = sys.argv[3]
        except IndexError:
            print('No threshold entered, takiong the 200 brightest pixels')
            T1, T2 = GetThreshold(image[Yinf: Ysup,Xinf: Xsup], nb=200),7e4
        else:
            try:
                T1, T2 = entry.split('-')
            except ValueError:
                T1, T2 = entry, 7e4

        T1, T2 = int(T1), int(T2)
        T1, T2 = np.min([T1,T2]), np.max([T1,T2])
    print('Threshold = %i % s'%(T1,T2))
    
    D = DetectHotPixels(filename, area=area, DS9backUp = DS9_BackUp_path, T1=T1, T2=T2,nb=nb)

    d.set('region delete all')
    d.set('region {}.reg'.format(D['region']))                                        
    return  D['table']

def DS9Desmearing(xpapoint, DS9backUp = DS9_BackUp_path, T1=None, T2=None, config=my_conf, size=5,nb=None):
    from tqdm import tqdm

    d=DS9(xpapoint)
    d.set('region delete all')
    filename = d.get('file')
    fitsimage = d.get_fits()[0]#fits.open(path)[0] 
    image = fitsimage.data
    ly, lx = image.shape
    table = DS9DetectHotPixels(xpapoint, DS9backUp = DS9_BackUp_path, T1=T1, T2=T2, config=my_conf)
    print (table)
    
    background = [np.nanmean(image[yi-1,np.max([0,xi-size]):xi+1]+image[np.min([ly-1,yi+1]),np.max([0,xi-size]):xi+1])/2 for xi, yi in zip(table['xcentroid'],table['ycentroid'])]
#    background =[]
#    for xi, yi in zip(table['xcentroid'],table['ycentroid']):
#        print(xi,yi)
#        background.append(np.nanmean(image[yi-1,np.max([0,xi-size]):xi+1]+image[np.min([ly-1,yi+1]),np.max([0,xi-size]):xi+1])/2 )
#    
    #print(background)
    #plt.plot(image[table['ycentroid'][0],table['xcentroid'][0]-size:table['xcentroid'][0]+1])
    value = [np.nansum(image[yi,xi-size:xi+1]-bck) for xi, yi, bck in zip(table['xcentroid'],table['ycentroid'],background)]
    for i in tqdm(range(len(table['ycentroid']))):
        print(i)
        #print(i,table['ycentroid'][i],table['xcentroid'][i])
        #print(value[i],background[i])
        image[table['ycentroid'][i],table['xcentroid'][i]-size:table['xcentroid'][i]+1] = background[i]
        image[table['ycentroid'][i],table['xcentroid'][i]] = value[i]+background[i]
    fitsimage.data = image
    fitswrite(fitsimage,filename[:-5] + '_unsmeared.fits')
    d.set('frame new')
    d.set('file '+ filename[:-5] + '_unsmeared.fits')
    #plt.plot(image[table['ycentroid'][0],table['xcentroid'][0]-size:table['xcentroid'][0]+1])
    #plt.show()
    return table
    
    
def DetectHotPixels(filename, area, DS9backUp = DS9_BackUp_path, T1=None, T2=None, nb=200):
    """
    """
    from astropy.io import fits
    import random
    fitsimage = fits.open(filename)[0] 
    image = fitsimage.data
    Yinf, Ysup,Xinf, Xsup = area
    if T1 is None:
        T1, T2 = GetThreshold(image[Yinf: Ysup,Xinf: Xsup], nb=nb), 7e4
    cosmicRays = detectCosmics_new(image,T=T1 ,T2=T2, area=area)
    #cosmicRays = cosmicRays[(cosmicRays['xcentroid']<2200) & (cosmicRays['xcentroid']>1700) & (cosmicRays['ycentroid']<1000)]
    #plt.plot(cosmicRays['xcentroid'],cosmicRays['ycentroid'],'x')
    #cosmicRays = cosmicRays[:10*nb]


    #random.shuffle(cosmicRays)

#    cosmicRays = assign_CR(cosmicRays,dist=5)
#    cosmicRays = Determine_front(cosmicRays)
#    cosmicRays = cosmicRays[cosmicRays['front']==1]



    #plt.plot(cosmicRays[(cosmicRays['front']==1)]['xcentroid'],cosmicRays[(cosmicRays['front']==1)]['ycentroid'],'x')
#    index = random.sample(range(len(cosmicRays)), np.min([len(cosmicRays),nb]))
#    index.sort()
#    cosmicRays = cosmicRays[index]
    cosmicRays = cosmicRays[cosmicRays['doublons']==0]
    cosmicRays = cosmicRays#[:nb]

    #csvwrite(cosmicRays, filename + '_HotPixels.csv')
    name = DS9backUp + 'DS9Regions/HotPixels%s-%s'%(T1,T2)
    create_DS9regions([list(cosmicRays['xcentroid'])],[list(cosmicRays['ycentroid'])], form=['circle'], radius=1, save=True, savename=name, color = ['yellow'],ID=None)                
    return  {"table": cosmicRays,"region": name}



def detectCosmics_new(image,T=6*1e4, T2=None,area=[0,2069,1053,2133], n=3, config=my_conf):
    """Detect cosmic rays, for FIREBall-2 specfic case it is a simplistic case
    where only thresholding is enough
    """
    from astropy.table import Table, Column
    import matplotlib.dates as mdates
    locator = mdates.HourLocator(interval=1)
    locator.MAXTICKS = 50000
    ax=plt.gca()
    ax.xaxis.set_minor_locator(locator)
    CS1 = ax.contour(image, levels=T, colors='white', alpha=0.5).allsegs[0] if (image>T).any() else []
    if T2 is not None:
        CS2 = ax.contour(image, levels=T2, colors='white', alpha=0.5).allsegs[0] if (image>T2).any() else []
        contours = CS1 + CS2
        print('%i hot pixels above %i, %i hot pixels above %i'%(len(CS1),T,len(CS2),T2))
    names = ('id','sizex','sizey','len_contour','max_x','min_y', 'max_y')
    cosmics = Table(np.zeros((len(contours),len(names))),names=names)
    cosmics['id'] = np.arange(len(contours))
    cosmics['sizex'] = [cs[:,0].max() - cs[:,0].min()     for cs in contours ]
    cosmics['sizey'] = [cs[:,1].max() - cs[:,1].min()     for cs in contours ]

    cosmics['len_contour'] = [ len(cs[:,1])   for cs in contours ]
    cosmics['max_x'] = [ int(cs[:,0].max()+n*1)   for cs in contours ]
    cosmics['min_y'] = [ int(cs[:,1].min()-n*1)   for cs in contours ]
    cosmics['max_y'] = [ int(cs[:,1].max()+n*2)   for cs in contours ]
    cosmics['mean_y'] = [ int((cs[:,1].max()+cs[:,1].max())/2)   for cs in contours ]
    cosmics['size'] = [ n*50   for cs in contours ]
    cosmics['size_opp'] = [ n*1 for cs in contours ]
    imagettes = [image[int(cs[:,1].min()):int(cs[:,1].max())+1,int(cs[:,0].min()):int(cs[:,0].max())+1] for cs in contours]
    for cs in contours:
        print(int(cs[:,0].min()),int(cs[:,0].max())+1,int(cs[:,1].min()),int(cs[:,1].max()+1))
    cosmics['cx']  = [ np.where(ima==np.nanmax(ima))[1][0] for ima in imagettes]
    cosmics['cy']  = [ np.where(ima==np.nanmax(ima))[0][0] for ima in imagettes]
    cosmics['c0x'] = [int(cs[:,0].min())    for cs in contours ]
    cosmics['c0y'] = [int(cs[:,1].min())    for cs in contours ]
    cosmics['xcentroid'] = cosmics['c0x'] + cosmics['cx'] 
    cosmics['ycentroid'] = cosmics['c0y'] + cosmics['cy'] 
    cosmics['value'] = [image[y,x] for  x, y in zip(cosmics['xcentroid'],cosmics['ycentroid'])]
    #cosmics[my_conf.exptime[0]] = header[my_conf.exptime[0]]
    #cosmics[my_conf.gain[0]] = header[my_conf.gain[0]]
    #cosmics['number'] = re.findall(r'\d+',os.path.basename(filename))[-1]
    if len(cosmics)==0:
        print('No cosmic rays detected... Please verify the detection threshold')
        #sys.exit()
        cosmics.add_columns([Column(name='doublons'),Column(name='dark'),Column(name='id'),Column(name='distance')])
        return cosmics
    else:
        cosmics['doublons']=0
        cosmics['dark'] = -1
        cosmics['id'] = -1
        cosmics['distance'] = -1
        cosmics['sum_10'] = -1
        cosmics['contour'] = -1
        cosmics['Nb_saturated'] = -1
        print(len(cosmics), ' detections, youpi!')
        cosmics_n = delete_doublons_CR(cosmics, dist=1, delete_both=False)
        print(len(cosmics_n[cosmics_n['doublons']==0]))
        print(cosmics_n['xcentroid','ycentroid','value','doublons'])
    return cosmics_n

#plt.plot(cosmics['xcentroid'],cosmics['ycentroid'],'x')

       
def detectCosmics(image,T=6*1e4, T2=None,area=[0,2069,1053,2133], config=my_conf):
    """Detect cosmic rays, for FIREBall-2 specfic case it is a simplistic case
    where only thresholding is enough
    """
    from astropy.table import Table, Column
    if T2 is None:
        y, x = np.where(image>T)
        value = image[image>T]
    else:
        y, x = np.where((image>T) & (image<T2))
        value = image[(image>T) & (image<T2)]
    index = (x>area[2]) & (x<area[3]) & (y>area[0]) & (y<area[1])
    x, y, value = x[index], y[index], value[index]  
    cosmicRays = Table([x, y, value], names = ('xcentroid', 'ycentroid', 'value'))
    if len(cosmicRays)==0:
        print('No cosmic rays detected... Please verify the detection threshold')
        #sys.exit()
        cosmicRays.add_columns([Column(name='doublons'),Column(name='dark'),Column(name='id'),Column(name='distance')])
        return cosmicRays
    else:
        cosmicRays['doublons']=0
        cosmicRays['dark'] = -1
        cosmicRays['id'] = -1
        cosmicRays['distance'] = -1
        cosmicRays['sum_10'] = -1
        cosmicRays['contour'] = -1
        cosmicRays['Nb_saturated'] = -1
        print(len(cosmicRays), ' detections, youpi!')
    return cosmicRays

def delete_doublons_CR(sources, dist=4, delete_both=False):
    """Function that delete doublons detected in a table, 
    the initial table and the minimal distance must be specifies
    """
    from tqdm import tqdm
    if delete_both:
        for i in tqdm(range(len(sources))):
            distances = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i])
            a = distances >= dist
            #a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
            #a = distance2(sources[sources['doublons']==0]['xcentroid','ycentroid'],sources['xcentroid','ycentroid'][i]) > dist
            a = list(1*a)
            a.remove(0)
            if np.nanmean(a)<1:
                sources['doublons'][i]=1
                sources['distance'][i]= np.nanmin(distances[distances>0])
    else:
        for i in tqdm(range(len(sources))):
            distances = distance(sources['xcentroid'],sources['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i])
            a = distances >= dist
            #a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
            #a = distance2(sources[sources['doublons']==0]['xcentroid','ycentroid'],sources['xcentroid','ycentroid'][i]) > dist
            a = list(1*a)
            a.remove(0)
            if np.nanmean(a)<1:#if there is still a 0, means if there is a neighboor closer to min distance
                sources['doublons'][i]=1
                sources['distance'][i]= np.nanmin(distances[distances>0])        
    print(len(sources[sources['doublons']==0]), ' Comsic rays detected, youpi!')
    return sources


def plotOSregion(xpapoint, DS9backUp = DS9_BackUp_path, config=my_conf):
    """
    """    
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    d = DS9(xpapoint)
    data = d.get_arr2np()
    header = d.get_fits()[0].header
    filename = d.get('file')
    y = np.nanmean(data, axis=0)[:1070][::-1]
    x = np.arange(len(y))
    exp = lambda x, xo, a, b, offset: offset + b*np.exp(-a*(x-xo)) 
    end = 30
    p0=[0,5e-1,y.max()-y.min(),y.min()]
    popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0) 
    xo, a, b, offset = popt
    plt.figure()#figsize=(10,6))
#    plt.plot(x, y,'o', label='DS9 values')
#    plt.plot(x[:end], exp(x[:end],*popt), label='Exp Fit: %i*exp(-(x-%i)/%0.2f)+%i'%(b, xo, 1/a, offset))
    plt.plot(x, np.log10(y - offset),'o', label='DS9 values')
    plt.plot(x[:end], np.log10(exp(x[:end],*popt)-offset), label='Exp Fit: %i*exp(-(x-%i)/%0.2f)+%i'%(b, xo, 1/a, offset))

    plt.grid(True, linestyle='dotted')
    plt.title('Smearing analysis: Overscan profile - T=%s, gain=%i'%(header[my_conf.temperature[0]],header[my_conf.gain[0]]))
    plt.legend()
    plt.xlabel('Pixels - %s'%(filename))
    plt.ylabel("ADU value")
    plt.show() 
    csvwrite(np.vstack((x,y)).T, DS9backUp + 'CSVs/%s_OSprofile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return    

def even(f):
    import math
    return math.ceil(f / 2.) * 2

def DS9SmearingProfileAutocorr(xpapoint, DS9backUp=DS9_BackUp_path, name='', Plot=True, verbose=False, config=my_conf):
    d = DS9(xpapoint)
    filename = d.get('file')
    #print(sys.argv)
    Type = sys.argv[3].lower()#'2D-xy'.lower()
    verboseprint('Type =', Type, verbose=verbose)
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    
    try:
        region = getregion(d)
    except ValueError:
        try:
            reg = resource_filename('DS9FireBall', 'Regions')
        except:
            reg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Regions')
        if Type == '2d-xy':
            d.set('regions ' + reg + '/Autocorr2d.reg')
        else:
            d.set('regions ' + reg + '/Autocorr1dx.reg')
            
        region = getregion(d)
        verboseprint('No region defined! Taking default region in %s.\nDo not hesitate to change this default region if needed'%(reg + '/Autocorr.reg'), verbose=verbose)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    verboseprint(Yinf, Ysup,Xinf, Xsup,verbose=verbose)
    
    if len(path)>1:
        Plot=False
    D=[] 
    for filename in path:
        verboseprint(filename, verbose=verbose)
        if len(path)>1:
            D.append(SmearingProfileAutocorr(filename=filename, area=area, DS9backUp=DS9_BackUp_path, name='', Plot=Plot,Type=Type,verbose=verbose))
        else:
            D = SmearingProfileAutocorr(filename=filename, area=area, DS9backUp=DS9_BackUp_path, name='', Plot=Plot,Type=Type,verbose=verbose)
    #print(D)
    return D 
       
def verboseprint(*args, verbose=True):
    if bool(int(verbose)):
        print(*args)
    else:
        pass

def SmearingProfileAutocorr(filename, area=None, DS9backUp=DS9_BackUp_path, name='', Plot=True, Type='x', verbose=False, config=my_conf):
    """
    plot a stack of the cosmic rays
    """
    #from astropy.io import fits
    #from astropy.table import Table
    from scipy.optimize import curve_fit
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    if not verbose:
        np.warnings.filterwarnings('ignore')
#    import warnings
#    
#    warnings.simplefilter('ignore', UserWarning)
#    np.errstate(all='ignore')
    print('Type  = ', Type)
    if area is None:
        if Type == '2d-xy':
            area = [1500,1550,1500,1550]
        else:
            area = [890-470,890+470,1565-260,1565+260]#[1200,1800,1200,1800]
    D = TwoD_autocorrelation(filename, save=True, area=area, plot_flag=Plot, DS9backUp = DS9_BackUp_path, ds9=False,Type=Type)
    verboseprint('Autocorr = ', D['corr'],verbose=verbose)
    try:
        autocorr  = D['corr']/D['corr'].min()
    except ValueError:
         return {'Exp_coeff':1/-100,'NoiseReductionFactor':Smearing2Noise(exp_coeff=1/-100) }
    temp, gain =  D['temp'], D['gain'] 
    #imshow(autocorr)#d.set_np2arr(autocorr)
    lx, ly = autocorr.shape
    verboseprint(lx,ly,verbose=verbose)
    size= 6#
    #y = autocorr[int(lx/2)-size:int(lx/2)+size,int(ly/2)-1];x = np.arange(len(y));plot(x,y,'--o')
    if Type=='2d-xy':
        y = autocorr[int(lx/2)-1,int(ly/2)-size-1:int(ly/2)+size];x = np.arange(len(y));#plot(x,y,'--o')
        y1 = autocorr[int(lx/2)-1,int(ly/2)-size-1:int(ly/2)];x1 = np.arange(len(y1))
        y2 = autocorr[int(lx/2)-1,int(ly/2)-1:int(ly/2)+size]
        yy = (y1[::-1]+y2)/2
    if Type=='x':
        y = np.mean(autocorr[:,int(ly/2)-size-1:int(ly/2)+size],axis=0);x = np.arange(len(y));#plot(x,y,'--o')
        #print(y.shape)
        y1 = np.mean(autocorr[:,int(ly/2)-size:int(ly/2)+1],axis=0);x1 = np.arange(len(y1))
        #print(y1,x1)
        y2 = np.mean(autocorr[:,int(ly/2):int(ly/2)+size+1],axis=0)
        yy = (y1[::-1]+y2)/2    
#    plt.plot(x1,y1[::-1],'--')
#    plt.plot(x1,y2,'--')
#    plt.plt.plot(x1,yy,'--o')
    #plt.figure(figsize=(10,6))
    fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
    exp = lambda x, a, b, offset: offset + b*np.exp(-a*x) 
    #exp2 = lambda x, a, b, offset, a1, b1: offset + b*np.exp(-a*x) + b1*np.exp(-a1*x) 
    #cr_image = autocorr
    x, y, end  = x1, yy, size
    verboseprint(x,verbose=verbose)
    x100 = np.linspace(x[0],x[size],100)
    #plt.vlines(median(x),y.min(),y.max())
    #plt.text(median(x)+50,mean(y),'Amplitude = %i'%(max(y)-min(y)))
    try:
        p0=[5e-1,y.max()-y.min(),y.min()]
        #p1=[0,5e-1,y.max()-y.min(),y.min(),0,5e-1,y.max()-y.min()]
        popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0) 
        #popt1, pcov1 = curve_fit(exp2, x[:end], y[:end], p0=np.hstack((popt,[5e-1,y.max()-y.min()])))
    except (RuntimeError or TypeError) as  e:# ( or ValueError) as e :
        verboseprint(e,verbose=True)
        offset = np.min(y)
        a=-0.01
        offsetn = y.min()
    except ValueError as e:
        verboseprint(e,verbose=True)
        a=-0.01
        offset = 0
        offsetn = 0
    else:   
        a, b, offset = popt
        #a0, b0, offset0, a1, b1 = popt1
        offsetn = y.min()
        ax2.plot(x100, np.log10(exp(x100,*popt)-offsetn),linestyle='dotted', label='Exp Fit: %i*exp(-x/%0.2f)+%i'%(b, 1/a, offset))
        #ax2.plot(x[:end], np.log10(exp2(x[:end],*popt1)-offsetn), label='Exp Fit: %i*exp(-x/%0.2f)+ %i + %i*exp(-x/%0.2f)'%(b0, 1/a0, offset0,b1, 1/a1))
        
        ax1.plot(x100, exp(x100,*popt),linestyle='dotted', label='Exp Fit: %i*exp(-x/%0.2f)+%i'%(b, 1/a, offset))
        #ax1.plot(x[:end], exp2(x[:end],*popt1), label='Exp Fit: %i*exp(-x/%0.2f)+ %i + %i*exp(-x/%0.2f)'%(b0, 1/a0, offset0,b1, 1/a1))

    ax2.plot(x, np.log10( y - offsetn ) ,'-o',c='black', label='DS9 values - %0.2f Expfactor \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f'%(1/a,Smearing2Noise(exp_coeff=1/a)['Var_smear'],Smearing2Noise(exp_coeff=1/a)['Hist_smear']))
#    ax2.plot(x1,np.log10(y1[::-1]- offsetn),linestyle='dotted',c='grey')
 #   ax2.plot(x1,np.log10(y2- offsetn) ,linestyle='dotted',c='grey')  
    ax2.fill_between(x1,np.log10(y1[::-1]- offsetn),np.log10(y2- offsetn),alpha=0.2, label='Left-right difference in autocorr profile')
    ax1.plot(x, y ,'-o', c='black',label='DS9 values - %0.2f Expfactor \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f'%(1/a,Smearing2Noise(exp_coeff=1/a)['Var_smear'],Smearing2Noise(exp_coeff=1/a)['Hist_smear']))
    #ax1.plot(x1,y1[::-1],linestyle='dotted',c='grey')
    #ax1.plot(x1,y2,linestyle='dotted',c='grey')    
    ax1.fill_between(x1,y1[::-1],y2,alpha=0.2, label='Left-right difference in autocorr profile')
    ax1.grid(True, linestyle='dotted');ax2.grid(True, linestyle='dotted')
    fig.suptitle('%s Smearing analysis: Autocorrelation - T=%s, gain=%i, area=%s'%(Type.upper(), temp,float(gain), area),y=1)
    ax1.legend();ax2.legend()
    ax2.set_xlabel('Pixels - %s'%(os.path.basename(filename)))
    ax1.set_ylabel("ADU value");ax2.set_ylabel("Log ADU value")
    fig.tight_layout()
#    plt.savefig(DS9backUp + 'Plots/%s_CR_HP_profile%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss"), name) )
    if not os.path.exists(os.path.dirname(filename) +'/SmearingAutocorr'):
        os.makedirs(os.path.dirname(filename) +'/SmearingAutocorr')
    plt.savefig(os.path.dirname(filename) + '/SmearingAutocorr/%s_%s.png'%(os.path.basename(filename)[:-5],Type.upper()) )
    if Plot:
        plt.show() 
    else:
        plt.close()
    csvwrite(np.vstack((x,y)).T, DS9backUp + 'CSVs/%s_CR_HP_profile%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"), name) ,verbose=verbose)
    return {'Exp_coeff':1/a,'NoiseReductionFactor':Smearing2Noise(exp_coeff=1/a) }



def DS9SmearingProfile(xpapoint, DS9backUp=DS9_BackUp_path, name='', Plot=True, config=my_conf):
    """
    plot a stack of the cosmic rays
    """
    from astropy.table import Table
    d = DS9(xpapoint)
    filename = getfilename(d)
    #print(sys.argv)
    pathc, xy, thresholds = sys.argv[3:6]
    #if len(sys.argv) > 3+3: path = Charge_path_new(filename, entry_point=3+3)
    path = Charge_path_new(filename) if len(sys.argv) > 6 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    if len(path)>1:
        Plot=False
         
    #print(path)
    #print(os.path.basename(path)) 
    paths = path
    D=[]
    if os.path.exists(pathc):
        if '.reg' in os.path.basename(pathc):
            d.set('region {}'.format(pathc))                                        
        else:
            x, y = xy.split('-')
            cat, name = DS9Catalog2Region(xpapoint, name=pathc, x=x, y=y)
            d.set('region {}'.format(name))
    for path in paths:
        if 'Merged' not in os.path.basename(path): 
            filename = d.get('file')#fits.open(path[0])[0].data
                
            regions = getregion(d, all=True)
            if len(regions)>1:
                x, y = np.array([int(reg.xc) for reg in regions]), np.array([int(reg.yc) for reg in regions])
                area = None
            else: 
                try:
                    region = regions[0]
                except ValueError:
                    area = [200,600,1600,2000]#[1400,1800,1200,1600]#[200,600,1600,2000]#
                else:
                    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
                    area = [Yinf, Ysup,Xinf, Xsup]
                    print(Yinf, Ysup,Xinf, Xsup)
                print('detecting regions')
                print('thresholds = ',thresholds)
                if thresholds == '-':
                    T1, T2 = None, None
                elif thresholds.isdigit():
                    T1, T2 = int(float(thresholds)), None
                else:
                    T1, T2 = np.array(thresholds.split('-'),dtype=int)
                cat = DetectHotPixels(path, area=area, DS9backUp = DS9_BackUp_path, T1=T1, T2=T2)
                table = cat['table']
                x, y = table['xcentroid'], table['ycentroid']
                create_DS9regions2(x,y, radius=3, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers')
                x, y = x+1, y+1
                d.set('regions /tmp/centers.reg')
            xy=(x,y)
            D.append(SmearingProfile(filename=filename, path=None, xy=xy, area=area, DS9backUp=DS9_BackUp_path, name='', Plot=Plot))
        else:
            print('detecting regions')
            D.append(SmearingProfile(filename=None, path=path, xy=None, area=area, DS9backUp=DS9_BackUp_path, name='', Plot=Plot))
    return D


def SmearingProfile(filename=None, path=None, xy=None, area=None, DS9backUp=DS9_BackUp_path, name='', Plot=True, config=my_conf):
    """
    plot a stack of the cosmic rays
    """
    from astropy.io import fits
    from astropy.table import Table
    from scipy.optimize import curve_fit
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
        
    if filename is None:
        cat = Table.read(path)
        lx,ly, offset = 5,100, 20
        cr_image = np.nan * np.zeros((2*lx,2*ly+offset,len(cat)))
        n=len(cat)
        for image in np.unique(cat['filename'])  :
            #print(image)
            fitsimage = fits.open(image)[0] 
            data = fitsimage.data
            header = fitsimage.header
            cr2 = delete_doublons_CR(cat[cat['filename']==image],dist=30)
            x, y = cr2['xcentroid'], cr2['ycentroid']
    #        x, y = cr2['xcentroid'][cr2['distance']>10], cr2['ycentroid'][cr2['distance']>10]
            index = (np.sum(np.isnan(cr_image), axis=(0,1))==2*lx*(2*ly+offset)).argmax()
        
            for k in range(len(x)):
                j = x[k]
                i = y[k]
                print('k+index = ', k + index)
                try:
                    cr_image[:,:,k+index] = data[i-lx:i+lx,j-2*ly:j+offset]
                except ValueError as e:
                    print(e)
                    n -= 1
                    pass
        cr_im = np.nansum(cr_image,axis=2)
        cr_im /= n
    
    else:
        try:
            x, y = xy
        except TypeError:
            D = DetectHotPixels(filename, area=area, DS9backUp = DS9_BackUp_path, T1=None, T2=None)
            table = D['table']
            x, y = table['xcentroid'], table['ycentroid']
            x, y = x+1, y+1
        fitsimage = fits.open(filename)[0] 
        image = fitsimage.data
        header = fitsimage.header
        lx,ly, offset = 5,50, 20
        cr_image = np.nan * np.zeros((2*lx,2*ly+offset,len(x)))
        for k in range(len(x)):
            j = x[k]
            i = y[k]
            try:
                cr_image[:,:,k] = image[i-lx:i+lx,j-2*ly:j+offset]
            except ValueError:
                pass
        cr_im = np.nanmean(cr_image,axis=2)



    y = cr_im[4,:]
    y = y[:np.argmax(y)]
    y = y[::-1]
    x = np.arange(len(y))
    
    exp = lambda x, a, b, offset: offset + b*np.exp(-a*x) 
#    exp2 = lambda x, a, b, offset, a1, b1: offset + b*np.exp(-a*x) + b1*np.exp(-a1*x) 
    endd = 20
    x, y = x[:endd], y[:endd] 
    end = 8


    #plt.figure(figsize=(10,6))
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(10,8),sharex=True)
    
    #plt.vlines(median(x),y.min(),y.max())
    #plt.text(median(x)+50,mean(y),'Amplitude = %i'%(max(y)-min(y)))
    try:
        p0=[5e-1,y.max()-y.min(),y.min()]
        popt, pcov = curve_fit(exp, x[:end], y[:end], p0=p0) 
    except (RuntimeError or TypeError) as  e:# ( or ValueError) as e :
        print(e)
        offset = np.min(y)
        a=-0.01
        offsetn = y.min()
        popt = p0
    except ValueError:
        a=-0.01
        offset = 0
        offsetn = 0
    else:   
        a, b, offset = popt
        offsetn = offset
        ax2.plot(x[:end], np.arcsinh(exp(x[:end],*popt)-offsetn), color='#1f77b4', label='Exp Fit: %i*exp(-x/%0.2f)+%i'%(b, 1/a, offset))
        ax2.plot(x, np.arcsinh(exp(x,*popt)-offsetn), color='#1f77b4', linestyle='dotted')
        
        ax1.plot(x[:end], exp(x[:end],*popt), color='#1f77b4', label='Exp Fit: %i*exp(-x/%0.2f)+%i'%(b, 1/a, offset))
        ax1.plot(x, exp(x,*popt), color='#1f77b4', linestyle='dotted')
        
#    try:
#        p0=[5e-1,y.max()-y.min(),y.min()]
#        popt1, pcov1 = curve_fit(exp2, x[:end], y[:end], p0=np.hstack((popt,[5e-1,y.max()-y.min()])))
#    except (RuntimeError or TypeError) as  e:# ( or ValueError) as e :
#        print(e)
#        offset = np.min(y)
#        a=-0.01
#        offsetn = y.min()
#    except ValueError:
#        a=-0.01
#        offset = 0
#        offsetn = 0
#    else:   
#        a, b, offset = popt
#        a0, b0, offset0, a1, b1 = popt1
#        offsetn = offset0
#        ax2.plot(x[:end], np.log10(exp2(x[:end],*popt1)-offsetn), label='Exp Fit: %i*exp(-x/%0.2f)+ %i + %i*exp(-x/%0.2f)'%(b0, 1/a0, offset0,b1, 1/a1))
#        ax1.plot(x[:end], exp2(x[:end],*popt1), label='Exp Fit: %i*exp(-x/%0.2f)+ %i + %i*exp(-x/%0.2f)'%(b0, 1/a0, offset0,b1, 1/a1))

    ax2.plot(x, np.arcsinh( y - offsetn ) ,'o',color='#1f77b4', label='DS9 values - %i images \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f'%(cr_image.shape[-1],Smearing2Noise(exp_coeff=1/a)['Var_smear'],Smearing2Noise(exp_coeff=1/a)['Hist_smear']))
    ax1.plot(x, y ,'o',label='DS9 values - %i images \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f'%(cr_image.shape[-1],Smearing2Noise(exp_coeff=1/a)['Var_smear'],Smearing2Noise(exp_coeff=1/a)['Hist_smear']))
    
    #plt.plot(x, exp(x,*p0), label='p0')
    ax1.grid(True, linestyle='dotted');ax2.grid(True, linestyle='dotted')
    fig.suptitle('Smearing analysis: Hot pixel / Cosmic rays profile - T=%s, gain=%i'%(header[my_conf.temperature[0]],header[my_conf.gain[0]]),y=1)
    ax1.legend();ax2.legend()
    ax2.set_xlabel('Pixels - %s'%(os.path.basename(filename)))
    ax1.set_ylabel("ADU value");ax2.set_ylabel("~Log [Arcsinh] ADU value")
    fig.tight_layout()
    plt.savefig(DS9backUp + 'Plots/%s_CR_HP_profile%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss"), name) )
    if Plot:
        plt.show() 
    else:
        plt.close()
    csvwrite(np.vstack((x,y)).T, DS9backUp + 'CSVs/%s_CR_HP_profile%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"), name) )
    return {'Exp_coeff':1/a,'NoiseReductionFactor':Smearing2Noise(exp_coeff=1/a) }



def assign_CR(sources, dist=10):
    """Assign all the cosmic detected pixel (superior to the threshold) 
    to one cosmic ray hit event
    """
    from tqdm import tqdm
    groups = sources[sources['doublons']==0]
    groups = sources
#    for i, cr in tqdm(enumerate(groups)):
#        x,y = cr['xcentroid'],cr['ycentroid']
    for i in tqdm(range(len(sources))):
        x = groups[i]['xcentroid']
        y = groups[i]['ycentroid']
        index = (sources['xcentroid']>x-dist) & (sources['xcentroid']<x+dist) & (sources['ycentroid']>y-dist) & (sources['ycentroid']<y+dist)
        sources['id'][index]=i
    return sources

def DistinguishDarkFromCR(sources, T, number=2):
    """When only one pixel of the detection is above threshold it must be dark
    """
    from tqdm import tqdm
    for i in tqdm(range(len(np.unique(sources['id'])))):
        idd = np.unique(sources['id'])[i]
        sources['Nb_saturated'][(sources['id'] == idd)] = len(sources[(sources['id'] == idd) & (sources['value'] > T)])
#        if len(sources[(sources['id'] == idd) & (sources['value'] > T)]) < 2:
#            sources['dark'][sources['id'] == idd] = 1
#        else:
#            sources['dark'][sources['id'] == idd] = 0
    return sources


def PlotCR(sources):
    """Visualization of the CRs
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    plt.figure()#figsize=(7,10))
    plt.plot(sources[sources['dark']<1]['xcentroid'],sources[sources['dark']<1]['ycentroid'],'.',label='CR')
    plt.plot(sources[sources['dark']==1]['xcentroid'],sources[sources['dark']==1]['ycentroid'],'.',label='Dark')
    plt.axis('equal');plt.legend()
    plt.xlim((1000,2100))
    #plt.xlim((1000,2100))
    for idd in np.unique(sources['id']):
        plt.annotate('%i'%(idd), (sources[sources['id']==idd]['xcentroid'][0]+10,sources[sources['id']==idd]['ycentroid'][0]+10))
        plt.scatter(sources[sources['id']==idd]['xcentroid'][0],sources[sources['id']==idd]['ycentroid'][0], s=180, facecolors='none', edgecolors='r')
    return
   #plt.scatter(sources[sources['id']==idd]['xcentroid'][0],sources[sources['id']==idd]['ycentroid'][0], s=10*len(sources[sources['id']==idd]['xcentroid']), facecolors='none', edgecolors='r')
    

#for idd in np.unique(sources['id']):
#    print(sources[(sources['id'] == idd) & (sources['value'] > T)])
#plot(a[a['dark']<1]['xcentroid'],a[a['dark']<1]['ycentroid'],'.')
#plot(a[a['dark']==1]['xcentroid'],a[a['dark']==1]['ycentroid'],'.')
#
##
#sources=a
#for idd in sources['id']:
#    plot(len(sources[(sources['id'] == idd) & (sources['value'] > T)]),
#         sources[(sources['id'] == idd) & (sources['value'] > T)]['dark'].mean(),'.')
#    
    
#plot(sources[sources['id']==-1]['xcentroid'],sources[sources['id']==-1]['ycentroid'],'.')  
        
#def delete_doublons_CR1(sources, dist=10):
#    """Function that delete doublons detected in a table, 
#    the initial table and the minimal distance must be specifies
#    """
#    from tqdm import tqdm
#    sources['doublons']=0
#    for i in tqdm(range(len(sources))):
#        a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
#        #a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
#        #a = distance2(sources[sources['doublons']==0]['xcentroid','ycentroid'],sources['xcentroid','ycentroid'][i]) > dist
#        a = list(1*a)
#        a.remove(0)
#        if np.nanmean(a)<1:
#            sources['doublons'][i]=1
#    print(len(sources[sources['doublons']==0]), ' Comsic rays detected, youpi!')
#    return sources

def Determine_front(sources):
    """Function that delete doublons detected in a table, 
    the initial table and the minimal distance must be specifies
    """
    from astropy.table import Column
    from tqdm import tqdm
    try:    
        sources.add_column(Column(name='front', data=np.ones(len(sources))*0))
    except TypeError:
        sources.add_column(Column(name='front'))
        
        
    a = sources[(sources['doublons']==0)]# & (sources['dark']<1)]
    for id in tqdm(range(len(a))):
        #print('Id = ',id)
        index = sources['id']==id
        #print('Number of high value pixel in cosmic: ', len(sources[index]))
        for pixel in sources[index]:
            y = pixel['ycentroid']
            #x, y = pixel['xcentroid'], pixel['ycentroid']
            line = sources['ycentroid']==y
            frontmask = index & line  & (sources['xcentroid']==sources[line&index]['xcentroid'].max())
            sources['front'][frontmask]=1
            #sources[frontmask]#=1
        #print('Front pixels: ', sources[sources['id']==id]['front'].sum())
    #print(len(sources[sources['front']==1]), ' Front Comsic ray pixels detected, youpi!')
    return sources






def MaskCosmicRays(image, cosmics, length = 1000, cols=None,all=False):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    from tqdm import tqdm
    y, x = np.indices((image.shape))
    image = image.astype(float)
    if all is False:   
        cosmics = cosmics[(cosmics['front']==1) & (cosmics['dark']<1)]
    if cols is None:
        for i in tqdm(range(len(cosmics))):#range(len(cosmics)):
            image[(y==cosmics[i]['ycentroid']) & (x<cosmics[i]['xcentroid']+1) & (x>-length + cosmics[i]['xcentroid'])] = np.nan
    else:
        for i in tqdm(range(len(cosmics))):#range(len(cosmics)):
            image[(y>cosmics[i]['ycentroid']-cols-0.1) & (y<cosmics[i]['ycentroid']+cols+0.1) & (x<cosmics[i]['xcentroid']+4) & (x>-length + cosmics[i]['xcentroid'])] = np.nan
    return image

def distance(x1,y1,x2,y2):
    """
    Compute distance between 2 points in an euclidian 2D space
    """
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))

def DS9RemoveImage(xpapoint):
    """Substract an image, for instance bias or dark image, to the image
    in DS9. Can take care of several images
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    path2remove = sys.argv[3]
    a, b = sys.argv[4].split(',')
    print(a,b)
    a, b = float(a), float(b)
    #if len(sys.argv) > 5: path = Charge_path_new(filename, entry_point=5)
    path = Charge_path_new(filename) if len(sys.argv) > 5 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
    for filename in path:
        print(filename)
        result, name = SubstractImage(filename, path2remove=path2remove, a=a, b=b) 
                                        
    if len(path) < 2:
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return



def DS9_2D_autocorrelation(xpapoint):
    """Return 2D_autocorrelation plot
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    Type = sys.argv[3].lower()
    print('Type = ',Type)
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    plot_flag = False#(len(path) == 1)
    print('flag for plot = ', plot_flag)
    try:
        region = getregion(d)
    except ValueError:
        try:
            reg = resource_filename('DS9FireBall', 'Regions')
        except:
            reg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Regions')
        d.set('regions ' + reg + '/Autocorr.reg')
        region = getregion(d)
        print('No region defined! Taking default region in %s.\nDo not hesitate to change this default region if needed'%(reg + '/Autocorr.reg'))

    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)
    if plot_flag:
        d.set('regions command "box %0.3f %0.3f %0.1f %0.1f # color=yellow"' % (region.xc ,region.yc, 3*region.w, 3*region.h))
    for filename in path:
        print(filename)
        D = TwoD_autocorrelation(filename, save=True, area=area, plot_flag=plot_flag, ds9=xpapoint, Type=Type)
    corr, name =  D['corr'],   D['name']
        
    if len(path)<2:
        w = abs((area[1] - area[0]))
        h = abs((area[3] - area[2]))
        l=20
        d.set('frame new')
        d.set('file '+ name)
        d.set('regions command "projection %0.3f %0.3f %0.1f %0.1f  %0.1f # color=yellow"' % (3*h/2-l,3*w/2,3*h/2+l,3*w/2,0))
        d.set('regions command "projection %0.3f %0.3f %0.1f %0.1f  %0.1f # color=yellow"' % (3*h/2,3*w/2-l,3*h/2,3*w/2+l,0))
        DS9setup2(xpapoint)
    return corr


def DS9_2D_FFT(xpapoint, config=my_conf):
    """Return 2D_autocorrelation plot
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    plot_flag = (len(path) == 1)
    print('flag for plot = ', plot_flag)
    try:
        region = getregion(d)
    except ValueError:
        try:
            reg = resource_filename('DS9FireBall', 'Regions')
        except:
            reg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Regions')
        d.set('regions ' + reg + '/Autocorr.reg')
        region = getregion(d)
        print('No region defined! Taking default region in %s.\nDo not hesitate to change this default region if needed'%(reg + '/Autocorr.reg'))
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    print(Yinf, Ysup,Xinf, Xsup)
    for filename in path:
        print(filename)
        fft = TwoD_FFT(filename, save=True, area=area, plot_flag=plot_flag)
    return fft

def TwoD_FFT(filename, save=True, area=None, plot_flag=True, DS9backUp = DS9_BackUp_path):
    """Return 2D_autocorrelation plot
    """
    from astropy.io import fits
    from scipy import fftpack
    data = fits.open(filename)[0].data
    template = np.copy(data[area[0]:area[1], area[2]:area[3]]).astype('uint64')
    F = fftpack.fftshift( fftpack.fft2(template) )
    psd2D = np.abs(F)**2
    fitswrite(psd2D,'/tmp/test.fits')
    d = DS9()
    d.set('frame new')
    d.set('file /tmp/test.fits')
    return psd2D


        
def TwoD_autocorrelation(filename, save=True, area=None, plot_flag=True, DS9backUp = DS9_BackUp_path, ds9=None,Type='x', verbose=False, config=my_conf):
    """Return 2D_autocorrelation plot
    """
    from scipy import signal#from scipy import misc
    from astropy.io import fits
    w = abs((area[1] - area[0]))
    h = abs((area[3] - area[2]))
    new_area = [area[0] - w,area[1] + w, area[2] - h,area[3] + h]
    fitsimage = fits.open(filename)[0]
    data = fitsimage.data 
    finite = np.isfinite(np.mean(data[:, new_area[2]:new_area[3]],axis=1));
    data = data[finite,:]
    try:
        gain , temp = fitsimage.header[my_conf.gain[0]], fitsimage.header[my_conf.temperature[0]]
    except KeyError:
        gain, temp = 0,0 
    template = np.copy(data[area[0]:area[1], area[2]:area[3]]).astype('uint64')

    if Type == '2d-xy': 
        image = np.copy(data[new_area[0]:new_area[1], new_area[2]:new_area[3]]).astype('uint64')
        image = image - np.nanmin(data) + 100.0
        template = template - np.nanmin(data) + 100.0
        corr = signal.correlate2d(image, template, boundary='symm', mode='same')

    if Type == 'x':
        image = np.copy(data[area[0]:area[1], new_area[2]:new_area[3]]).astype('uint64')
        corr = np.zeros(image.shape)
        for i in range(template.shape[0]):
            corr[i,:]  = signal.correlate(image[i,:], template[i,:], mode='same')# / 128
    if Type == 'y':
        image = np.copy(data[new_area[0]:new_area[1], area[2]:area[3]]).astype('uint64')
        corr = np.zeros(image.shape)
        for i in range(template.shape[1]):
            corr[:,i]  = signal.correlate(image[:,i], template[:,i], mode='same')# / 128 
    #np.savetxt(DS9backUp + 'Arrays/%s_Autocorr.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")), corr)
    fitsimage = fits.HDUList([fits.PrimaryHDU(corr)])[0]
    fitsimage.header['PATH'] = filename
    name = DS9_BackUp_path + 'Images/%s_%sAutocorr.fits'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),os.path.basename(filename)[:-5])
    fitswrite(fitsimage, name, verbose=verbose)
    D = {'corr':corr, 'name':name, 'gain':gain, 'temp':temp}
    return D


def DS9AnalyzeImage(xpapoint, config=my_conf):
    """Return some plot to analyze the image
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    plot_flag = (len(path) == 1)
    print('flag for plot = ', plot_flag)
    for filename in path:
        print(filename)
        AnalyzeImage(filename, save=True, area=None, plot_flag=plot_flag, config=config)
    return

def AnalyzeImage(filename, save=True, area=None, plot_flag=True, config=my_conf):
    """Return some plot to analyze the image
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from scipy import stats 
    import scipy.ndimage as ndimage
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data#.astype('uint64')
    header = fitsimage.header
    if os.path.getsize(filename)>5638240:
        Type = 'detector'
        if area is None:
            area = [0,-1,1050,2150]
#        new_image = image[area[0]:area[1],area[2]:area[3]]
        width_ratios = (1,2.8)

    else:
        Type = 'guider'
        if area is None:
            area = [0,-1,0,-1] 
        width_ratios = (1,0.9)
    new_image = image[area[0]:area[1],area[2]:area[3]]
    lx, ly = image.shape
 #   width_ratios = (1,ly/lx+0.5)

    fig = plt.figure()#figsize=(15,8))    
    gs = GridSpec(6, 2, width_ratios=width_ratios,height_ratios=(1,1,1,1,0.3,0.2))
    ax1 = fig.add_subplot(gs[:-1, :1])
    axc = fig.add_subplot(gs[-1, :1])
    ax2 = fig.add_subplot(gs[0:2, 1])
    #ax3 = fig.add_subplot(gs[1,1])
    ax4 = fig.add_subplot(gs[2:-2, 1])
    #ax5 = fig.add_subplot(gs[-2:, 1])
    amp = 0    
    print(np.nanmean(image,axis=1), image)
    if Type == 'detector':
        offset = 20
        OScorr_n = ComputeOSlevel1(new_image, OSR1=[offset,-offset,offset,400], OSR2=[offset,-offset,2200,2400]) 
        OScorr = ComputeOSlevel1(image, OSR1=[offset,-offset,offset,400], OSR2=[offset,-offset,2200,2400]) 
        new_image = (new_image - OScorr_n)# / header[my_conf.exptime[0]]
        #ax2.set_ylim((0,1.1*np.nanmean(new_image,axis=1).max()))
        #ax3.set_ylim((0,1.1*np.nanmean(new_image,axis=0).max()))
        if header[my_conf.gain[0]] == 0:
            fitsimage.data = image - OScorr
            amp=1
            D = calc_emgainGillian(fitsimage,area=area)
            emgain,bias,sigma,frac_lost =  [D[x] for x in [my_conf.gain[0],'bias','sigma','frac_lost']]# D[my_conf.gain[0],'bias','sigma','frac_lost']
            b = [D[x] for x in ["image","emgain","bias","sigma","bin_center","n","xlinefit","ylinefit","xgaussfit", "ygaussfit","n_bias","n_log","threshold0","threshold55","exposure", "gain", "temp"]]
            try:
                plot_hist2(*b, ax=ax4,plot_flag=True)
            except TypeError:
                ax4.hist(new_image.flatten(),bins=1000,histtype='step',log=True)
                ax4.set_xlabel('Pixel value')
                ax4.set_ylabel('Log10(#Pixels)')                
            #ax4.set_xlim((-1e3,10000))
    if amp == 0:
        ax4.hist(new_image.flatten(),bins=1000,histtype='step',log=True)
        ax4.set_xlabel('Pixel value')
        ax4.set_ylabel('Log10(#Pixels)')
    cmap = plt.cm.cubehelix
    cmap.set_bad('black',0.9)
    new_image_sm = ndimage.gaussian_filter(new_image, sigma=(1.3, 1.3), order=0)

    im = ax1.imshow(new_image[::-1,:], cmap=cmap,vmin=np.nanpercentile(new_image_sm,10),vmax=np.nanpercentile(new_image_sm,99.99))
    print(np.nanmean(new_image,axis=1), new_image)
    #ax1.axis('equal')
    ax2.plot(np.nanmean(new_image_sm,axis=1), label='Column values [ADU/s]')
    ax2.plot(np.nanmean(new_image_sm,axis=0), label='Line values [ADU/s]')
    ax2.legend()
    #ax2.set_ylabel('Column/Line value')
    #ax3 = ComputeEmGain(filename, Path2substract=None, save=False, plot_flag=False, d=None, ax=ax3)
    #ax3.set_ylabel('Variance Intensity plot')
    fig.colorbar(im, cax=axc, orientation='horizontal');
    area = [500,600,500,600]
    im = image[area[0]:area[1],area[2]:area[3]]
    if Type == 'guider':
        ax1.set_title("Image Analysis - " + os.path.basename(filename) + ' - ' + header['DATE'])
        plt.figtext(.54, .1, 'Az = %0.3f deg\nEl = %0.3f deg\n'
                'Exp = %0.1f sec' % (header['AZ'], header['EL'], header['EXPOSURE']),
                fontsize=10,bbox={'facecolor':'orange', 'alpha':0.9, 'pad':10})
        plt.figtext(.64, .1, 'Press = %i mB\nRotenc = %i \n'
                'Mrot = %0.1f deg' % (header['PRESSURE'], header['ROTENC'], header['MROT']),
                fontsize=10,bbox={'facecolor':'orange', 'alpha':0.9, 'pad':10})
        plt.figtext(.75, .1, 'LINAENC = %0.2f \nLINBENC = %0.2f \n'
                'LINCENC = %0.2f' % (header['LINAENC'], header['LINBENC'], header['LINCENC']),
                fontsize=10,bbox={'facecolor':'orange', 'alpha':0.9, 'pad':10})
        plt.figtext(.87, .1, 'Mean = %0.1f e-/s\nStd = %0.1f e-/s\n'
                'SKew = %0.2f e-/s \nStdovers^2/<Im> = %0.3f' % (np.nanmean(new_image), np.nanstd(new_image), stats.skew(new_image,axis=None, nan_policy='omit'),
                                                                np.nanstd(im)**2/np.nanmean(im)),
                fontsize=10,bbox={'facecolor':'blue', 'alpha':0.3, 'pad':10})    
    else:
        fig.suptitle("Image Analysis [OS corrected] - " + os.path.basename(filename) + ' - ' + header['DATE'], y=0.99)
        plt.figtext(.4, .1, 'Exposure = %i sec\nGain = %i \n'
                'T det = %0.2f C' % (header[my_conf.exptime[0]], header[my_conf.exptime[0]], float(header[my_conf.temperature[0]])),
                fontsize=12,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
        plt.figtext(.6, .1, 'Mean = %0.5f \nStd = %0.3f \n'
                'SKew = %0.3f ' % (np.nanmean(new_image), np.nanstd(new_image),  stats.skew(new_image,axis=None, nan_policy='omit')),
                fontsize=12,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
        plt.figtext(.8, .1, 'Gain = %i\nStd^2/<Im> = %0.3f \n'
                'Bias = %i' % (header[my_conf.gain[0]], np.nanstd(im)**2/np.nanmean(im), OScorr.mean() ),
                fontsize=12,bbox={'facecolor':'blue', 'alpha':0.5, 'pad':10})
        


    fig.tight_layout()
    name = os.path.dirname(filename) + '/ImageAnalysis/'
    if save:
        try: 
            os.makedirs(name)
        except OSError:
            pass
        plt.savefig(name + os.path.basename(filename)[:-5] + '.png')        
    if plot_flag:
        plt.show() 
    else:
        plt.close()
    print(np.nanmean(new_image,axis=1), new_image)
    #plt.figure()
    #plt.plot(np.nanmean(new_image_sm,axis=1), label='Column values [ADU/s]')
    #plt.show()
    return


def SubstractImage(filename, path2remove, a , b, config=my_conf):
    """Substract an image, for instance bias or dark images,
    and save it
    """
    from astropy.io import fits
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    fitsimage2 = fits.open(path2remove)[0]
    image2substract = fitsimage2.data
    fitsimage.data = image - a * image2substract - b 
    name = filename[:-5] + '_subtracted_' + os.path.basename(path2remove)
    exp1, exp2 = int(fitsimage.header[my_conf.exptime[0]]), int(fitsimage2.header[my_conf.exptime[0]])
    fitsimage.header[my_conf.exptime[0]] = np.nanmax([exp2,exp1]) - a * (np.nanmin([exp2,exp1]))
    fitswrite(fitsimage, name)
    return fitsimage.data, name    

def DS9NoiseMinimization(xpapoint, radius=200, config=my_conf):
    """Substract a linear combination of images
    """
    #from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    Path2substract = sys.argv[3]
    if sys.version_info.major == 3:
        images2substract = glob.glob(os.path.join(Path2substract, '**/*.fits'), recursive=True)
    if sys.version_info.major == 2:
        images2substract = glob.glob(os.path.join(Path2substract, '**/*.fits'))
        images2substract +=  glob.glob(os.path.join(Path2substract, '*.fits'))
    images2substract.sort()    
    for filename in path:
        print(filename)
        name, areas = NoiseMinimization(filename, images2substract=images2substract, Path2substract=Path2substract, save=True, radius=radius)  
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
        areas = np.array(areas)
        create_DS9regions2(areas[:,2]+float(radius)/2,areas[:,0]+float(radius)/2, radius=radius, form = 'box',
                           save=True,color = 'yellow', savename='/tmp/centers')
        d.set('regions /tmp/centers.reg')
    return    
    #vpicouet

def NoiseMinimization(filename, images2substract, save=True, radius=50, Path2substract=''):
    """
    """
    from scipy.optimize import leastsq
    from astropy.io import fits
    images2substract.sort()
    im0 = fits.open(filename)[0].data
    lx, ly = im0.shape
    images = np.zeros((lx, ly, len(images2substract)))
    print('Images used to minimize standard deviation: \n', '\n'.join(images2substract))
    for i, im in enumerate(images2substract):
        images[:,:,i] = fits.open(im)[0].data
    areas = CreateAreas(im0, area=None, radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    result, cov_x = leastsq(func=residual, x0=0.3*np.ones(len(images2substract)), args=(im0,images,areas,radius))        
    image, name = SubstractImages(param=np.square(result), image_name=filename, images=images, save=True, Path2substract=Path2substract)
    std0 = np.std(residual(np.zeros(len(images2substract)),im0,images,areas,radius))
    std_end = np.std(residual(result,im0,images,areas,radius))
    print(bcolors.BLACK_RED + 'Standard deviation:  %0.2f  =====>   %0.2f '%(std0,std_end) + bcolors.END)
    return name, areas

    
   
def SubstractImages(param, image_name, images, save=True, Path2substract='', config=my_conf):
    """
    """
    from astropy.io import fits
    image = fits.open(image_name)[0]
    new_image = image.data - np.nansum(param * images,axis=2)
    if save:
        image.data = new_image
        par = '_{:0.2f}'*len(param)
        name = image_name[:-5] + par.format(*param) + '.fits'
        print('save : %s'%(name))
        image.header['PATHSUBS'] = Path2substract
        image.writeto(name,overwrite=True)        
    return new_image.ravel() - np.nanmean(new_image), name

def residual(param, image, images, areas, radius):
    """
    """
  #  n=300
  #  xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  
  #  yis = [1100, 1450, 1800]*3
  #  areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]
    #area=[40,330,1830,2100]

    im_tot = np.ones((radius,radius,len(areas)))
    for i, area in enumerate(areas):
        new_image = image[area[0]:area[1],area[2]:area[3]] - np.nansum(np.square(param) * images[area[0]:area[1],area[2]:area[3],:],axis=2)
        new_image -= np.nanmean(new_image)
        im_tot[:,:,i] = new_image
    
    return im_tot[np.isfinite(im_tot)] #- new_image.mean()


def ReturnPath(filename,number=None, All=False):
    import re
    number1 = re.findall(r'\d+',os.path.basename(filename))[-1]
    n = len(number1)
    filen1, filen2 = filename.split(number1)
    print(filen1, filen2)
    if number is not None:
        number = int(float(number))
        return filen1 + '%0{}d'.format(n)%(number) + filen2
    elif All:
        path = glob.glob('%s%s%s'%(filen1,'?'*n,filen2))
        np.sort(path)
        return path


def DS9PhotonTransferCurve(xpapoint, config=my_conf):
    """
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    d.set('region delete all')

    #if len(sys.argv) > 5: path = Charge_path_new(filename, entry_point=5)
    path = Charge_path_new(filename) if len(sys.argv) > 6 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    
    subtract, number, size = sys.argv[3:6]#'f3 names'#sys.argv[3]
    radius=np.array(size.split(','),dtype=int)
    print('subtract, number = ', subtract, number)
    if int(float(subtract)) == 0:
        subtract=False
        Path2substract = None
    elif os.path.isfile(number):
        Path2substract = number
    elif number.isdigit():
        Path2substract = ReturnPath(filename,number)
    elif number == '-' :
        Path2substract = None
        
    try:
        region = getregion(d)
    except ValueError:
        print('Please define a region.')
        area = my_conf.physical_region#[1053,2133,500,2000]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Xinf, Xsup,Yinf, Ysup]
        print(Xinf, Xsup,Yinf, Ysup)


    if len(path)==1:
        plot_flag=True
    else:
        plot_flag=False
    D=[]
    print('Path2substract, subtract = ', Path2substract, subtract)
    for filename in path:
        verboseprint(filename, verbose=my_conf.verbose)
        if len(path)>1:
            D.append(PhotonTransferCurve(filename, Path2substract=Path2substract, save=True, d=d, Plot=plot_flag, area=area, subtract=subtract, radius=radius))
        else:
            D = PhotonTransferCurve(filename, Path2substract=Path2substract, save=True, d=d, Plot=plot_flag, area=area, subtract=subtract, radius=radius)
    return

def PhotonTransferCurve(filename, Path2substract=None, save=True, Plot=True, d=None, ax=None, radius=[40,40], subtract=False, area=None, DS9backUp = DS9_BackUp_path,verbose=True, config=my_conf):
    """Compute EMgain with the variance intensity method
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    print("""##################\nSubtracting Image = %s \nPath to subtract = %s\nradius = %s \nArea = %s\n\nfilename = %s \nplot_flag = %s\n##################"""%(subtract,Path2substract, radius, area, filename, Plot))#,verbose=config.verbose)
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    try:
        texp = fitsimage.header[my_conf.exptime[0]]
    except KeyError:
        texp=1
    offset = 20
    image = image - ComputeOSlevel1(image, OSR1=[offset,-offset,offset,400],OSR2=[offset,-offset,2200,2400])
    if subtract:
        n=2
        if Path2substract is None:
            images = ReturnPath(filename, All=True)
            im='next'
            if images.index(filename)<len(images)-1:
                name = images[images.index(filename) + 1]
                image_n = fits.open(name)[0]
                data, exptime = image_n.data, image_n.header[my_conf.exptime[0]]
                if int(float(exptime)) == int(float(texp)):
                    verboseprint('Subtracting previous image: %s'%(name),verbose=config.verbose)
                    image_sub = image - data 
                    im=0
                else: 
                    verboseprint('Previous image do not have same exposure time',verbose=config.verbose)
                    im='next'
            if (im=='next') or (images.index(filename)==len(images)-1):
                name = images[images.index(filename) - 1]
                image_n = fits.open(name)[0]
                data, exptime = image_n.data, image_n.header[my_conf.exptime[0]]
                if int(float(exptime)) == int(float(texp)):
                    verboseprint('Subtracting next image: %s'%(name),verbose=config.verbose)
                    image_sub = image - data  
                else:
                    verboseprint('No image have the same exposure time: No subtraction!',verbose=config.verbose)
                    n=1
                    image_sub = image
        else:
            image_sub = image - fits.open(Path2substract)[0].data 
    else:
        n=1
        image_sub = image
    if area is None:
        area = my_conf.physical_region#[1053,2133,0,2000]
    areasd = CreateAreas(image, area=area, radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    areasOS1 = CreateAreas(image, area=[0, 400, area[2], area[3]], radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    areasOS2 = CreateAreas(image, area=[2200, 2400, area[2], area[3]], radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    len_area_det = len(areasd)
    areas = areasd + areasOS1 + areasOS2
    areas_OS = areasOS1 + areasOS2
    verboseprint('Number of regions : ', len(areas),verbose=verbose)
    var_all = []
    intensity_all = []
    var_phys = []
    intensity_phys = []
    var_os = []
    intensity_os = []
    for i, area in enumerate(areas):
        i, v = np.nanmean(image[area[0]:area[1],area[2]:area[3]]), np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])#MeanVarArea(image_sub, area)
        var_all.append(v);intensity_all.append(i)
    for i, area in enumerate(areasd):
        i, v = np.nanmean(image[area[0]:area[1],area[2]:area[3]]), np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])#MeanVarArea(image_sub, area)
        var_phys.append(v);intensity_phys.append(i)
    for i, area in enumerate(areas_OS):
        i, v = np.nanmean(image[area[0]:area[1],area[2]:area[3]]), np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])#MeanVarArea(image_sub, area)
        var_os.append(v);intensity_os.append(i)
 
    var_all, var_phys = np.array(var_all).flatten(), np.array(var_phys).flatten()
    a = 1
    #Index_phys = (var_phys < np.nanmedian(var_phys) + a * np.nanstd(var_phys)) & (intensity_phys < np.nanmedian(intensity_phys) + a * np.nanstd(intensity_phys))#.std()
    Index_phys = (var_phys < np.nanpercentile(var_phys,98)) & (intensity_phys <np.nanpercentile(intensity_phys,98))#.std()
    Index_all = var_all < np.nanmedian(var_phys) + a * np.nanstd(var_phys)#.std()
    Index_os = var_os < np.nanmedian(var_os) + a * np.nanstd(var_os)#.std()
    var_all, intensity_all = var_all[Index_all], np.array(intensity_all)[Index_all].flatten()
    var_phys, intensity_phys = var_phys[Index_phys], np.array(intensity_phys)[Index_phys].flatten()
    var_os, intensity_os = np.array(var_os)[Index_os], np.array(intensity_os)[Index_os].flatten()

    areas = np.array(areas)
    if type(radius)==int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    if d is not None:
        create_DS9regions2(areas[:,2]+float(r1)/2,areas[:,0]+float(r2)/2, radius=radius, form = 'box',
                           save=True,color = 'yellow', savename='/tmp/centers')
        d.set('regions /tmp/centers.reg')
    try:
        emgain = fitsimage.header[my_conf.gain[0]]
    except KeyError:
        emgain=1
    if emgain > 0:
        cst = 2
    else:
        cst = 1    
    fig, ax = plt.subplots()
    intensity_phys_n,var_phys_n = SigmaClipBinned(intensity_phys,var_phys/cst, sig=1, Plot=True, ax=ax, log=True)
    intensity_os_n,var_os_n = SigmaClipBinned(intensity_os,var_os/cst, sig=1, Plot=True, ax=ax, log=True)
    

#    ax.loglog(intensity_phys_n,var_phys_n,'+', label='loglog')
#    ax.plot(intensity_phys_n,var_phys_n,'x',label='lineair')
#    ax.plot(np.log(intensity_phys_n),np.log(var_phys_n),'x',label='log')
#    ax.plot(np.log10(intensity_phys_n),np.log10(var_phys_n),'x',label='log10')
    index = (np.log10(intensity_phys_n)>3.6)  &  (np.log10(intensity_phys_n)<4.5)
    fit = np.polyfit(np.log10(intensity_phys_n[index]), np.log10(var_phys_n[index]),1)
    fit_fn = np.poly1d(fit) 
    ax.plot(np.linspace(2,5,100), fit_fn(np.linspace(2,5,100)), '--', label='Linear regression, GainTot = %0.1f \n-> %0.1f smr corr (0.32)'%(fit[0],fit[0]/0.32))

    index1 = (np.isfinite(np.log10(intensity_os_n)))  &  (np.isfinite(np.log10(var_os_n)))
    fit1 = np.polyfit(np.log10(intensity_os_n[index1]), np.log10(var_os_n[index1]),1)
    fit_fn1 = np.poly1d(fit1) 
    ax.plot(np.linspace(-2,3,100), fit_fn1(np.linspace(-2,3,100)), '--', label='Linear regression, GainTot = %0.1f \n-> %0.1f smr corr (0.32)'%(fit[0],fit[0]/0.32))

 #   ax.plot(np.log10(intensity_phys_n[index]), fit_fn(np.log10(intensity_phys_n[index])), '--', label='Linear regression, GainTot = %0.1f \n-> %0.1f smr corr (0.32)'%(fit[0],fit[0]/0.32))
    ax.text(0.5,0.1,'y = %0.2f * x + %0.2f'%(fit[0], fit[1]),transform=ax.transAxes)
    ax.set_xlabel('Intensity [log(ADU)]')
    ax.set_ylabel('Variance [log(ADU)] / %s'%(cst))
    ax.set_title('Photon transfer curve')
    ax.grid(linestyle='dotted')
    ax.legend(loc='upper left')
#    emgain = fit[0]
    
#    ax, emgain_phys = PlotComputeEmGain(intensity_phys_n, var_phys_n, emgain , r1*r2, filename=filename, len_area_det=len_area_det, ax=ax1, cst='(%i x %i)'%(cst,n))

    csvwrite(np.vstack((intensity_phys_n,var_phys_n/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ,verbose=verbose)
    csvwrite(np.vstack((np.hstack((intensity_os_n,intensity_phys_n)),np.hstack((var_os_n,var_phys_n))/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ,verbose=verbose)

    x, y  = np.log10(np.hstack((intensity_os_n,intensity_phys_n))), np.log10(np.hstack((var_os_n,var_phys_n)))
    ax.set_ylim((0.97*np.nanmin(y[np.isfinite(y)]),1.03*np.nanmax(y[np.isfinite(y)])))
    ax.set_xlim((1.03*np.nanmin(x[np.isfinite(x)]),1.03*np.nanmax(x[np.isfinite(x)])))
    if save:
        if not os.path.exists(os.path.dirname(filename) +'/VarIntensDiagram'):
            os.makedirs(os.path.dirname(filename) +'/VarIntensDiagram')
        plt.savefig(os.path.dirname(filename) +'/VarIntensDiagram/' + os.path.basename(filename)[:-5] + '_.png')
    if Plot:
        plt.show()
    else:
        plt.close()
    #D = {'ax':ax, 'EMG_var_int_w_OS':emgain_all, 'EMG_var_int_wo_OS':emgain_phys}
    return 1#D
        
def DS9ComputeEmGain(xpapoint, subtract=True, verbose=False, config=my_conf):
    """Compute EMgain with the variance intensity method
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    d.set('region delete all')
    #if len(sys.argv) > 5: path = Charge_path_new(filename, entry_point=5)
    path = Charge_path_new(filename) if len(sys.argv) > 6 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    
    subtract, number, size = sys.argv[3:6]#'f3 names'#sys.argv[3]
    radius=np.array(size.split(','),dtype=int)
    print('subtract, number = ', subtract, number)
    if int(float(subtract)) == 0:
        subtract=False
        Path2substract = None
    elif os.path.isfile(number):
        Path2substract = number
    elif number.isdigit():
        Path2substract = ReturnPath(filename,number)
    elif number == '-' :
        Path2substract = None
        
    try:
        region = getregion(d)
    except ValueError:
        print('Please define a region.')
        area = my_conf.physical_region#[1053,2133,500,2000]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Xinf, Xsup,Yinf, Ysup]
        print(Xinf, Xsup,Yinf, Ysup)


    if len(path)==1:
        plot_flag=True
    else:
        plot_flag=False
    D=[]
    print('Path2substract, subtract = ', Path2substract, subtract)
    for filename in path:
        verboseprint(filename, verbose=verbose)
        if len(path)>1:
            D.append(ComputeEmGain(filename, Path2substract=Path2substract, save=True, d=d, Plot=plot_flag, area=area, subtract=subtract, radius=radius))
        else:
            D = ComputeEmGain(filename, Path2substract=Path2substract, save=True, d=d, Plot=plot_flag, area=area, subtract=subtract, radius=radius)
    #print(D)
    return D

def CreateAreas_old(image, area=None, radius=100, offset=20, config=my_conf):
    """Create areas in the given image
    """
    #image = a.data
    ly, lx = image.shape
    if area is None:
        if ly == 2069:
            xmin, xmax = my_conf.physical_region[:2]#1053, 2121#0,3000#1053, 2121
            rangex = xmax - xmin
        xi = np.arange(offset,rangex - offset - radius, radius)
        yi = np.arange(offset,ly - offset - radius, radius)
        xx, yy = np.meshgrid(xi,yi)
        areas = [[a, a + radius, b + xmin, b + radius+ xmin] for a,b in zip(yy.flatten(),xx.flatten())]
        return areas
    else:
        print(1)
 


def DS9AnalyzeOSSmearing(xpapoint, Plot=False):
    d = DS9(xpapoint)
    filename = getfilename(d)
    print(sys.argv)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    if len(path)==1:
        Plot = True
    for filename in path:
        a = AnalyzeOSSmearing(filename, Plot=Plot)
    return a


def AnalyzeOSSmearing(filename, Plot=False, config=my_conf):
    from astropy.io import fits
    from scipy.optimize import curve_fit
    image = fits.open(filename)[0].data
    area = [70,2000,1020,1080]
    subim = np.nanmean(image[area[0]:area[1],area[2]:area[3]],axis=0)
    x = np.arange(len(subim)-1)
    subim1 = (subim[1:] + subim[:-1])/2 
    subim1 = 100 * (subim1 - np.min(subim1)) / np.nanmax(subim1 - np.min(subim1))
    slope_index = np.argmax(subim1[1:] - subim1[:-1])
    if (slope_index<10) or (slope_index>len(x)-10):
        slope_index = 45
    y = subim1#[::-1]
    exp = lambda x, a, b, offset: offset + b*np.exp(a*(x)) 
    p0 = [ 0.13,  0.05, 10]
    p0_1 = [ 1*0.13/2,  1*0.05, 10]
    n1=30
    n2 = 2
    try:
        popt1, pcov = curve_fit(exp, x[:n1],y[:n1],p0=p0) 
    except (RuntimeError, TypeError) as e:
        popt1 = [999,0,1]
    try:
        popt2, pcov = curve_fit(exp, x[slope_index-20:slope_index+n2],y[slope_index-20:slope_index+n2],p0=p0_1)  
    except (RuntimeError, TypeError) as e:
        popt2 = [999,0,1]
    offsetn =    popt1[-1]   
    print(popt2)
#    plt.plot(x[slope_index-4:slope_index+4],y[slope_index-4:slope_index+4])
#    plt.show()
#    sys.exit()
#    popt1 = PlotFit1D(x[:30],np.log10(subim1)[:30],Plot=False, deg=1)        
#    popt2 = PlotFit1D(x[slope_index-3:slope_index+3],np.log10(subim1)[slope_index-3:slope_index+3],Plot=False, deg=1)        
  #  plt.plot(x, exp(x,*popt1) , c='black',linestyle='dotted')#,label='DS9 values - %0.2f Expfactor \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f'%(1/popt1[0],Smearing2Noise(exp_coeff=1/popt1[0])['Var_smear'],Smearing2Noise(exp_coeff=1/popt1[0])['Hist_smear']))
   # plt.plot(x, exp(x,*popt2) , c='black',linestyle='dotted')#,label='DS9 values - %0.2f Expfactor \n-> Noise reduc = %0.2f, Slope reduc =  %0.2f'%(1/popt1[0],Smearing2Noise(exp_coeff=1/popt1[0])['Var_smear'],Smearing2Noise(exp_coeff=1/popt1[0])['Hist_smear']))
    #plt.show()
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    ax2.plot(x, np.log10( y - offsetn ) ,'.',c='black', label='DS9 values')
    ax2.plot(x[:n1], np.log10(exp(x,*popt1)[:n1]- offsetn )  , c='blue',linestyle='dotted',label='%0.2f  Expfactor '%(1/popt1[0]))
    ax2.plot(x[slope_index-20:slope_index+n2], np.log10(exp(x,*popt2)[slope_index-20:slope_index+n2]- offsetn )  , c='red',linestyle='dotted',label='%0.2f  Expfactor '%(1/popt2[0]))

    ax1.plot(x, y ,'.', c='black', label='DS9 values')
    ax1.plot(x[:n1], exp(x[:n1],*popt1) , c='blue',linestyle='dotted',label='%0.2f  Expfactor '%(1/popt1[0]))
    ax1.plot(x[slope_index-20:slope_index+n2], exp(x,*popt2)[slope_index-20:slope_index+n2] , c='red',linestyle='dotted',label='%0.2f  Expfactor '%(1/popt2[0]))
    fig.suptitle('%s Smearing analysis: OS region',y=1)
    ax1.legend();ax2.legend()
    ax1.set_ylim((0.9*y.min(),1.1*y.max()))
    ax2.set_xlabel('Pixels - %s'%(os.path.basename(filename)))
    ax1.set_ylabel("ADU value");ax2.set_ylabel("Log ADU value")
    fig.tight_layout()
    if not os.path.exists(os.path.dirname(filename) +'/SmearingOS'):
        os.makedirs(os.path.dirname(filename) +'/SmearingOS')
    plt.savefig(os.path.dirname(filename) + '/SmearingOS/%s.png'%(os.path.basename(filename)[:-5]) )
    if Plot:
        plt.show() 
    else:
        plt.close()
    return {'Exp1':1/popt2[0],'Exp2':1/popt1[0]}

        
def DS9ComputeDirectEmGain(xpapoint, radius=[150,150], area=None, config=my_conf):
    from astropy.io import fits
    d = DS9(xpapoint)
    filename = getfilename(d)
    image1 = filename
    image2 = sys.argv[-1]
    if area is None:
        area = my_conf.physical_region#[1053,2133,0,2000]
    areas = CreateAreas(fits.open(image1)[0].data, area=area, radius=radius)
    
    a = ComputeDirectEmGain(image1, image2, areas)
    r1, r2 = radius
    areas = np.array(areas)
    create_DS9regions2(areas[:,2]+float(r1)/2,areas[:,0]+float(r2)/2, radius=radius, form = 'box',
                       save=True,color = 'yellow', savename='/tmp/centers')
    d.set('regions /tmp/centers.reg')
    return a

def ComputeDirectEmGain(image1, image2, areas, config=my_conf):
    from astropy.io import fits
    fitsim1 = fits.open(image1)[0]
    fitsim2 = fits.open(image2)[0]
    texp1, texp2 = fitsim1.header[my_conf.exptime[0]], fitsim2.header[my_conf.exptime[0]]
    os1, os2 = ComputeOSlevel1(fitsim1.data), ComputeOSlevel1(fitsim2.data)
    im1, im2 = (fitsim1.data - os1)/texp1, (fitsim2.data - os2)/texp2
    gain = []
    print(os.path.basename(image1), os.path.basename(image2), texp1, texp2)
    for i, area in enumerate(areas):
        #print(im1[area[0]:area[1],area[2]:area[3]])
        #print(im2[area[0]:area[1],area[2]:area[3]])
        gaini = np.nanmean(im1[area[0]:area[1],area[2]:area[3]])/np.nanmean(im2[area[0]:area[1],area[2]:area[3]])
        gain.append(gaini)
    gain = np.array(gain)
    if np.nanmean(gain)<1:
        gain = 1/gain
    plt.figure()
    plt.hist(gain)
    plt.title('Gain  %s - %s -  M = %0.2f  -  Sigma = %0.3f'%(os.path.basename(image1), os.path.basename(image1), np.nanmean(gain), np.nanstd(gain)))
    plt.xlabel('Gain');plt.ylabel('Frequecy')
    plt.savefig('%s_DirectGain.png'%(image1[:-5]))
    plt.show()
    return


def SigmaClipBinned(x,y, sig=1, Plot=True, ax=None, log=False):
    """
    """
    x, y = np.array(x), np.array(y)
    ob, bins = np.histogram(x,bins=[np.percentile(x,i) for i in np.linspace(0,100,2+len(x)/100)])
    index=[]
    xn, yn = [], []
#    mask1 = ((x>=bins[-2])&(x<=bins[-1]))
#    mask2 = ((x>=bins[-3])&(x<=bins[-2]))
#    print('Vars = ',np.var(np.array(y)[mask1])/np.var(np.array(y)[mask2]))
#    if np.var(np.array(y)[mask1]) > 2.5 * np.var(np.array(y)[mask2]):
#        offset=1
#    else:
#        offset=0
    offset=0
    for i in range(len(ob)-offset):
        mask = ((x>=bins[i])&(x<=bins[i+1]))#.astype(int)
        xi, yi = np.array(x)[mask], np.array(y)[mask]
        indexi = (yi < np.nanmedian(yi) + sig * np.nanstd(yi)) & (yi > np.nanmedian(yi) - sig * np.nanstd(yi)) &  (xi < np.nanmedian(xi) + 3*sig * np.nanstd(xi)) & (xi > np.nanmedian(xi) - 3*sig * np.nanstd(xi))
        index.append(indexi)
        xn.append(xi)
        yn.append(yi)
        if Plot:
            if log:
                p = ax.plot(np.log10(xi),np.log10(yi),'.',alpha=0.15)
                ax.plot(np.log10(xi[indexi]), np.log10(yi[indexi]),'.',alpha=0.9,c=p[0].get_color())
            else:
                p = ax.plot(xi,yi,'.',alpha=0.15)
                ax.plot(xi[indexi], yi[indexi],'.',alpha=0.9,c=p[0].get_color())
#            ax.scatter(xi[indexi], yi[indexi],alpha=0.3,c=p[0].get_color())

    all_index = np.hstack(index)
    xx, yy = np.hstack(xn), np.hstack(yn)
    return xx[all_index], yy[all_index]
                

def MeanVarArea(image, area, threshold=0.99):
    subim = image[area[0]:area[1],area[2]:area[3]]
    lx, ly = subim.shape
    percent = np.isfinite(subim).sum() / (lx*ly)
    if percent >= threshold:
        return np.nanmean(subim), np.nanvar(subim)
    else:
        return np.nan, np.nan

def ComputeEmGain(filename, Path2substract=None, save=True, Plot=True, d=None, ax=None, radius=[40,40], subtract=False, area=None, DS9backUp = DS9_BackUp_path,verbose=True, config=my_conf):
    """Compute EMgain with the variance intensity method
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    verboseprint("""##################\nSubtracting Image = %s \nPath to subtract = %s\nradius = %s \nArea = %s\n\nfilename = %s \nplot_flag = %s\n##################"""%(subtract,Path2substract, radius, area, filename, Plot),verbose=verbose)
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    try:
        texp = fitsimage.header[my_conf.exptime[0]]
    except KeyError:
        texp=1
    offset = 20
    image = image - ComputeOSlevel1(image, OSR1=[offset,-offset,offset,400],OSR2=[offset,-offset,2200,2400])
    if subtract:
        n=2
        if Path2substract is None:
            images = ReturnPath(filename, All=True)
            im='next'
            if images.index(filename)<len(images)-1:
                name = images[images.index(filename) + 1]
                image_n = fits.open(name)[0]
                data, exptime = image_n.data, image_n.header[my_conf.exptime[0]]
                if int(float(exptime)) == int(float(texp)):
                    verboseprint('Subtracting previous image: %s'%(name),verbose=verbose)
                    image_sub = image - data 
                    im=0
                else: 
                    verboseprint('Previous image do not have same exposure time',verbose=verbose)
                    im='next'
            if (im=='next') or (images.index(filename)==len(images)-1):
                name = images[images.index(filename) - 1]
                image_n = fits.open(name)[0]
                data, exptime = image_n.data, image_n.header[my_conf.exptime[0]]
                if int(float(exptime)) == int(float(texp)):
                    verboseprint('Subtracting next image: %s'%(name),verbose=verbose)
                    image_sub = image - data  
                else:
                    verboseprint('No image have the same exposure time: No subtraction!',verbose=verbose)
                    n=1
                    image_sub = image
        else:
            image_sub = image - fits.open(Path2substract)[0].data 
    else:
        n=1
        image_sub = image
    if area is None:
        area = my_conf.physical_region#[1053,2133,0,2000]
    areasd = CreateAreas(image, area=area, radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    areasOS1 = CreateAreas(image, area=[0, 400, area[2], area[3]], radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    areasOS2 = CreateAreas(image, area=[2200, 2400, area[2], area[3]], radius=radius)#    n = 300#300#300#    xis = [40]*3 + [400]*3 + [800]*3  + [1200]*3  + [1600]*3  #    yis = [1100, 1450, 1800]*len(xis)#    areas = [[xo, xo + n, yo, yo + n] for xo, yo in zip(xis,yis)]#area=[40,330,1830,2100]
    len_area_det = len(areasd)
    areas = areasd + areasOS1 + areasOS2
    areas_OS = areasOS1 + areasOS2
    verboseprint('Number of regions : ', len(areas),verbose=verbose)
    var_all = []
    intensity_all = []
    var_phys = []
    intensity_phys = []
    var_os = []
    intensity_os = []
    for i, area in enumerate(areas):
        i, v = np.nanmean(image[area[0]:area[1],area[2]:area[3]]), np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])#MeanVarArea(image_sub, area)
        var_all.append(v);intensity_all.append(i)
    for i, area in enumerate(areasd):
        i, v = np.nanmean(image[area[0]:area[1],area[2]:area[3]]), np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])#MeanVarArea(image_sub, area)
        var_phys.append(v);intensity_phys.append(i)
    for i, area in enumerate(areas_OS):
        i, v = np.nanmean(image[area[0]:area[1],area[2]:area[3]]), np.nanvar(image_sub[area[0]:area[1],area[2]:area[3]])#MeanVarArea(image_sub, area)
        var_os.append(v);intensity_os.append(i)
 
    var_all, var_phys = np.array(var_all).flatten(), np.array(var_phys).flatten()
    a = 1
    #Index_phys = (var_phys < np.nanmedian(var_phys) + a * np.nanstd(var_phys)) & (intensity_phys < np.nanmedian(intensity_phys) + a * np.nanstd(intensity_phys))#.std()
    Index_phys = (var_phys < np.nanpercentile(var_phys,98)) & (intensity_phys <np.nanpercentile(intensity_phys,98))#.std()
    Index_all = var_all < np.nanmedian(var_phys) + a * np.nanstd(var_phys)#.std()
    Index_os = var_os < np.nanmedian(var_os) + a * np.nanstd(var_os)#.std()
    var_all, intensity_all = var_all[Index_all], np.array(intensity_all)[Index_all].flatten()
    var_phys, intensity_phys = var_phys[Index_phys], np.array(intensity_phys)[Index_phys].flatten()
    var_os, intensity_os = np.array(var_os)[Index_os], np.array(intensity_os)[Index_os].flatten()

    areas = np.array(areas)
    if type(radius)==int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    if d is not None:
        create_DS9regions2(areas[:,2]+float(r1)/2,areas[:,0]+float(r2)/2, radius=radius, form = 'box',
                           save=True,color = 'yellow', savename='/tmp/centers')
        d.set('regions /tmp/centers.reg')
    try:
        emgain = fitsimage.header[my_conf.gain[0]]
    except KeyError:
        emgain=1
    if emgain > 0:
        cst = 2
    else:
        cst = 1    
    fig, (ax0,ax1) = plt.subplots(1,2)#,figsize=(14,6))        
    intensity_phys_n,var_phys_n = SigmaClipBinned(intensity_phys,var_phys/cst, sig=1, Plot=True, ax=ax0)
    intensity_os_n,var_os_n = SigmaClipBinned(intensity_os,var_os/cst, sig=1, Plot=True, ax=ax0)
    intensity_phys_n,var_phys_n = SigmaClipBinned(intensity_phys,var_phys/cst, sig=1, Plot=True, ax=ax1)
    
    ax, emgain_phys = PlotComputeEmGain(intensity_phys_n, var_phys_n, emgain , r1*r2, filename=filename, len_area_det=len_area_det, ax=ax1, cst='(%i x %i)'%(cst,n))
    ax, emgain_all = PlotComputeEmGain(np.hstack((intensity_os_n,intensity_phys_n)), np.hstack((var_os_n,var_phys_n)), emgain , r1*r2, filename=filename, len_area_det=len_area_det, ax=ax0, cst='(%i x %i)'%(cst,n))

    csvwrite(np.vstack((intensity_phys_n,var_phys_n/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ,verbose=verbose)
    csvwrite(np.vstack((np.hstack((intensity_os_n,intensity_phys_n)),np.hstack((var_os_n,var_phys_n))/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ,verbose=verbose)


    ax0.set_ylim((0.97*np.hstack((var_os_n,var_phys_n)).min(),1.03*np.hstack((var_os_n,var_phys_n)).max()))
    ax0.set_xlim((0.97*np.hstack((intensity_os_n,intensity_phys_n)).min(),1.03*np.hstack((intensity_os_n,intensity_phys_n)).max()))
    ax1.set_ylim((0.97*var_phys_n.min(),1.03*var_phys_n.max()))
    ax1.set_xlim((0.97*intensity_phys_n.min(),1.03*intensity_phys_n.max()))
    fig.suptitle('Variance intensity diagram - %s - G = %s - #regions = %i'%(os.path.basename(filename),emgain,areas[:,1].shape[0]),y=1)    
    fig.tight_layout()
    if save:
        if not os.path.exists(os.path.dirname(filename) +'/VarIntensDiagram'):
            os.makedirs(os.path.dirname(filename) +'/VarIntensDiagram')
        plt.savefig(os.path.dirname(filename) +'/VarIntensDiagram/' + os.path.basename(filename)[:-5] + '_.png')
    if Plot:
        plt.show()
    else:
        plt.close()
    D = {'ax':ax, 'EMG_var_int_w_OS':emgain_all, 'EMG_var_int_wo_OS':emgain_phys}
    return D


def DS9AddToCatalog(xpapoint):
    path = sys.argv[3]
    functions = np.array(np.array(sys.argv[4:],dtype=int),dtype=bool)
    dict_functions = {'ComputeEmGain':[ComputeEmGain,'EMG_var_int_w_OS'],'calc_emgainGillian':[calc_emgainGillian,'EMG_hist'],
                      'SmearingProfileAutocorr':[SmearingProfileAutocorr,'Exp_coeff'],'SmearingProfile':[SmearingProfile,'Exp_coeff'],
                      'CountHotPixels':[CountHotPixels,'#_HotPixels'],'AnalyzeOSSmearing':[AnalyzeOSSmearing,'Exp1','Exp2'],'FluxDeffect':[FluxDeffect,'Flux']}
    for function, f in zip(functions,dict_functions.keys()):
        if function:
            AddToCatalog(path,f,f)
    return


def AddToCatalog(path, function, field,Plot=False):
    from astropy.table import Table
    dict_functions = {'ComputeEmGain':[ComputeEmGain,'EMG_var_int_w_OS'],'calc_emgainGillian':[calc_emgainGillian,'EMG_hist'],
                      'SmearingProfileAutocorr':[SmearingProfileAutocorr,'Exp_coeff'],'SmearingProfile':[SmearingProfile,'Exp_coeff'],
                      'CountHotPixels':[CountHotPixels,'#_HotPixels'],'AnalyzeOSSmearing':[AnalyzeOSSmearing,'Exp1','Exp2'],'FluxDeffect':[FluxDeffect,'Flux']}
    cat = Table.read(path)
    output = np.zeros((len(cat['PATH']),len(dict_functions[function][1:])))

    for j, key in enumerate(dict_functions[function][1:]):
        for i, filename in enumerate(cat['PATH']):
            print(filename)
            output[i,j] = dict_functions[function][0](filename,Plot=Plot)[key]

        cat[field+'_%i'%(j)] = output[:,j]
    csvwrite(cat,path)
    return cat




def CountHotPixels(filename,area=None, T=9000, Plot=False, config=my_conf):
    from astropy.io import fits
    image = fits.open(filename)[0].data
    if area is None:
        area = [10,1000,1500,2000]
    image = image[area[0]:area[1],area[2]:area[3]]
    a = np.where(image>T)
    return {'#_HotPixels': len(a[0])}       

def CreateAreas(image, area=None, radius=100, offset=20, verbose=False, config=my_conf):
    """Create areas in the given image
    """
    #image = a.data$
    if type(radius)==int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    verboseprint('r1,r2=',r1,r2,verbose=verbose)
    ly, lx = image.shape
    if area is None:
        if ly == 2069:
            xmin, xmax = my_conf.physical_region[:2]#1053, 2121#0,3000#1053, 2121
            ymin, ymax = 0,ly
           # rangex = xmax - xmin
        else:
            xmin, xmax = 0, lx
            ymin, ymax = 0, ly            
    else:
        xmin, xmax = area[0], area[1]
        ymin, ymax = area[2], area[3]
    xi = np.arange(offset + xmin, xmax - offset - r1, r1)
    yi = np.arange(offset + ymin, ymax - offset - r2, r2)
    xx, yy = np.meshgrid(xi,yi)
    areas = [[a, a + r2, b, b + r1] for a,b in zip(yy.flatten(),xx.flatten())]
    return areas

def Smearing2Noise(exp_coeff=1.5):
    """
    """
    try:
        noisePath= resource_filename('DS9FireBall', 'CSVs')
    except:
        noisePath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'CSVs')
    coeff1 = np.loadtxt(os.path.join(noisePath,'GainHistVSsmearing.csv'), delimiter=',')
    coeff2 = np.loadtxt(os.path.join(noisePath,'GainVarVSsmearing.csv'), delimiter=',')
    x1, y1 = coeff1.T
    x2, y2 = coeff2.T
    Hist_smear = PlotFit1D(x1,y1, deg=6, Plot=False)
    #y = [1/Smearing2Noise(a) for a in np.linspace(0,2,100)]
    #plot(np.linspace(0,2,100),y)
#    n=100
#    x = np.linspace(0,n,n+1)
#    y = np.exp(-x/exp_coeff) *(1-np.exp(-1/exp_coeff))
#    return np.sqrt(np.square(y).sum())**2
    Var_smear = PlotFit1D(x2,y2, deg=6, Plot=False)
    return {'Hist_smear':1/Hist_smear(exp_coeff),'Var_smear':1/Var_smear(exp_coeff)}


def PlotComputeEmGain(intensity, var, emgain, n, filename, len_area_det, ax=None, DS9backUp = DS9_BackUp_path, name='',cst=2):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
        
    fit = np.polyfit(intensity,var,1)
    fit_fn = np.poly1d(fit) 
    if ax is None:
        fig = plt.figure()#figsize=(8,6))
        ax = fig.add_subplot(111)
    #print(len_area_det)
    #ax.plot(intensity[:], var[:]/cst, 'x', label='OS: Data in %i pix side area, G = %i '%(n,emgain))
    ax.plot(intensity, fit_fn(intensity), '--', label='Linear regression, GainTot = %0.1f \n-> %0.1f smr corr (0.32)'%(fit[0],fit[0]/0.32))
    ax.set_ylabel('Variance [ADU] / %s'%(cst))
    ax.text(0.5,0.1,'y = %0.2f * x + %0.2f'%(fit[0], fit[1]),transform=ax.transAxes)
    ax.legend(loc='upper left')
    ax.set_xlabel('Intensity [ADU]')
    ax.grid(linestyle='dotted')
    emgain = fit[0]
    return ax, emgain


def PlotComputeEmGain_old(intensity, var, emgain, n, filename, len_area_det, ax=None, DS9backUp = DS9_BackUp_path,name=''):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    if emgain > 0:
        cst = 2
    else:
        cst = 1
    fit = np.polyfit(intensity,var/cst,1)
    fit_fn = np.poly1d(fit) 
    fit_wo0 = np.polyfit(intensity[:len_area_det],var[:len_area_det]/cst,1)
    fit_fn_wo0 = np.poly1d(fit_wo0) 
    if ax is None:
        fig = plt.figure()#figsize=(8,6))
        ax = fig.add_subplot(111)
    print(len_area_det)
    ax.plot(intensity[:len_area_det], var[:len_area_det]/cst, 'x', label='Det: Data in %i pix side area, G = %i '%(n,emgain))
    ax.plot(intensity[len_area_det:], var[len_area_det:]/cst, 'o', label='OS: Data in %i pix side area, G = %i '%(n,emgain))
    ax.plot(intensity, fit_fn(intensity), '--', label='Linear regression, Engain = %0.1f \n-> %0.1f smr corr (0.32)'%(fit[0],fit[0]/0.32))
    ax.plot(intensity[:len_area_det], fit_fn_wo0(intensity[:len_area_det]), '--', label='Linear regression[WO OS region], Engain = %0.1f\n-> %0.1f smr corr (0.32)'%(fit_wo0[0],fit[0]/0.32))
    ax.set_ylabel('Variance [ADU] / %i'%(cst))
    ax.text(0.5,0.1,'y = %0.2f *u x + %0.2f [W OS]'%(fit[0], fit[1]),transform=ax.transAxes)
    ax.text(0.5,0.2,'y = %0.2f * x + %0.2f [WO OS]'%(fit_wo0[0], fit_wo0[1]),transform=ax.transAxes)
    ax.legend(loc='upper left')
    ax.set_xlabel('Intensity [ADU]')
    ax.grid(linestyle='dotted')
    ax.set_title('Variance intensity diagram - %s'%(os.path.basename(filename)))    
    csvwrite(np.vstack((intensity,var/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ) 
    
    emgain = fit[0]
    return ax, emgain





def DS9TotalReductionPipeline_new(xpapoint, delete=True, create=False):
    """Run the total reduction pipeline defined for fireball
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    print(sys.argv)
    n = 6
    print('cr, os, bck, cc, d, si',sys.argv[3:3+n])
    cr, osc, bck, delete, cc, si = np.array(np.array(sys.argv[3:3+n],dtype=int),dtype=bool)
    print('cr, os, bck, d, cc, si', cr, osc, bck, delete, cc, si)


    #if len(sys.argv) > 3+n: path = Charge_path_new(filename, entry_point=3+n)
    path = Charge_path_new(filename) if len(sys.argv) > 9 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    print('Path to reduce: ', path)
    if '/DS9BackUp/subsets/' in path[0]:
        print('You are using simlinks: setting create to True. Will create images even if no cosmic rays')
        create = True
    
    #DS9CreateHeaderCatalog(xpapoint=None, files=path, filename=None, info=True, redo=True)
#    for filename in path:
#        name = ApplyTotalReductionPipeline_new(xpapoint, filename,BackgroundCorrection=bck, cosmics=cr, osc=osc, create=create)
    paths = np.unique(np.array([os.path.dirname(pathi) for pathi in path]))
    filesn = []     
    for path in paths:
        if bck:
            #print('Adding test1')
            filesn += glob.glob(os.path.join(path,'CosmicRayFree','OS_corrected','bkgd_photutils_substracted','*_cs.fits'))
            if delete:
                files = glob.glob(os.path.join(path,'CosmicRayFree','*_cs.fits'))
                files += glob.glob(os.path.join(path,'CosmicRayFree','OS_corrected','*_cs.fits'))
                for file in files: 
                    os.remove(file)
                    print('%s file deleted'%(file))
        elif osc:
            #print('Adding test2')
            filesn += glob.glob(os.path.join(path,'CosmicRayFree','OS_corrected','*_cs.fits'))
            if delete:
                files = glob.glob(os.path.join(path,'CosmicRayFree','*_cs.fits'))
                for file in files: 
                    os.remove(file)
                    print('%s file deleted'%(file))
        elif cr:
            #print('Adding test3')
            filesn += glob.glob(os.path.join(path,'CosmicRayFree','*_cs.fits'))
        else:
            #print('Adding test4')
            filesn += glob.glob(os.path.join(path,'*.fits')) 
    print(filesn)
    if cc:
        t2 = DS9CreateHeaderCatalog(xpapoint=None, files=filesn, filename=sys.argv[-1], info=True, redo=True)    
        print(t2)  
        if si:
            StackAllImages(xpapoint,t2)
    TimerSMS(timeit.default_timer(), hour=1)
    return


    
#1h->TU et francais
#heure temp = 10h, heure image 11h(francais)


def ApplyTotalReductionPipeline_new(xpapoint,filename, BackgroundCorrection=False, area=[0,-1,0,-1],length=4, cosmics=True, osc=True, create=False):
    """Warning, you are about to run the total reduction piepeline. 
    Please make sure you are aware about how it works. 
    In particular note that master bias and dark should be stacked images (or folders containing the images),
    at the same gains/temp. Also make sure the exposure time is saved in the header of your images as my_conf.exptime[0].
    If not flat path is given, flat removal will not be performed. 
    """
    if cosmics:
        fitsimage, filename, cosmics = removeCRtails_CS(filename, threshold=40000,n=3,size=0,area=area, create=create)

#    try:
#        table_path = resource_filename('DS9FireBall', 'Regions') + '/HotPixelsFinalV1_190222.reg'
#    except:
#        table_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Regions')  + '/HotPixelsFinalV1_190222.reg'
#    
#    reg = DS9Region2Catalog(xpapoint,name='/Users/Vincent/Documents/FireBallPipe/Calibration/Regions/test.reg')
#    InterpolateNaNs(filename_n, stddev=1)
#    try:
    if osc: 
        fitsimage, filename = ApplyOverscanCorrection(filename, stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20], save=True)
#    except FileNotFoundError:
#        fitsimage, filename_n = ApplyOverscanCorrection(filename, stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20], save=True)
    if BackgroundCorrection:
        fitsimage, filename = BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path)
    return filename 


def DS9TotalReductionPipeline(xpapoint):
    """Run the total reduction pipeline defined for fireball
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    if len(sys.argv) == 7:
        path = Charge_path(filename, entry_point=6)
    elif (len(sys.argv) ==  6) :
        if (os.path.isdir(sys.argv[5]) is False) & (os.path.isfile(sys.argv[5]) is False) :
            path = Charge_path(filename, entry_point=5)
    elif (len(sys.argv) <  6) :
        path = [filename]#[d.get("file")]
    print('Path to reduce: ', path)
    masterBias = sys.argv[3]
    masterDark = sys.argv[4]
    try:
        masterFlat = sys.argv[5]
    except IndexError:
        masterFlat = None
            
    for filename in path:
        print(filename)
        result, name = ApplyTotalReductionPipeline(filename, masterBias, masterDark, masterFlat) 
                                        
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return

def ApplyTotalReductionPipeline(filename, masterBias, masterDark, masterFlat):
    """Warning, you are about to run the total reduction piepeline. 
    Please make sure you are aware about how it works. 
    In particular note that master bias and dark should be stacked images (or folders containing the images),
    at the same gains/temp. Also make sure the exposure time is saved in the header of your images as my_conf.exptime[0].
    If not flat path is given, flat removal will not be performed. 
    """
    from astropy.io import fits
    from astropy.convolution import convolve
    masterDark = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Dark/StackedImage_StackedImage_207-211_100s_9000_NaNsFree-StackedImage_247-251_100s_9400_NaNsFree.fits'
    masterBias = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Bias/StackedImage_0-1.fits'
    masterFlat = None
    filename = '/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/CRfree_and_NaNsFree/StackedImage_368-379.fits'
    if os.path.isdir(masterBias):
        masterBias, name = StackImagesPath(glob.glob(os.path.join(masterBias, 'image*.fits')))
    elif os.path.isfile(masterBias):
        masterBias = fits.open(masterBias)
    if os.path.isdir(masterDark):
        masterDark, name = StackImagesPath(glob.glob(os.path.join(masterDark, 'image*.fits')))
    elif os.path.isfile(masterDark):
        masterDark = fits.open(masterDark)

    Bias_OScorr, Bias_OScorrName = ApplyOverscanCorrection(masterBias.filename(), stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
    Bias_OScorr.data = convolve(Bias_OScorr.data, kernel=np.array([[1.,1.,0],[1.,1.,0],[0.,0.,0]])/4)
    Bias_OScorr.writeto('/Users/Vincent/Nextcloud/FIREBALL/TestsFTS2018-Flight/Flight/dobc_data/180922/redux/NoiseReduction/Bias/StackedImage_0-1_OScorr_convolved.fits')
    Dark_OScorr, Dark_OScorrName = ApplyOverscanCorrection(masterDark.filename(), stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
    
    
    Dark_OScorr_Biascorr = Dark_OScorr.data - Bias_OScorr.data  #SubstractImage(DarkName, Bias_OScorrName)
    if masterDark[0].header[my_conf.exptime[0]] != 0:
        Dark_OScorr_Biascorr_per_sec = Dark_OScorr_Biascorr / masterDark[0].header[my_conf.exptime[0]]
    else:
        print('Exposure time is null for master dark, can not proceed')
        sys.exit()
    if masterFlat is not None:
        name = '_OScorr_Biascorr_Darkcorr_Flatcorr.fits'
        if os.path.isdir(masterFlat):
            masterFlat, name = StackImagesPath(glob.glob(os.path.join(masterFlat, 'image*.fits')))        
        elif os.path.isfile(masterFlat):
            masterFlat = fits.open(masterFlat)
            
        Flat_OScorr, Flat_OScorrName = ApplyOverscanCorrection(masterFlat, stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
        Flat_OScorr_Biascorr = Flat_OScorr - Bias_OScorr#SubstractImage(DarkName, Bias_OScorrName)
        Flat_OScorr_Biascorr_Darkcorr = Flat_OScorr_Biascorr - (Dark_OScorr_Biascorr * masterFlat[0].header[my_conf.exptime[0]])#SubstractImage(DarkName, Bias_OScorrName)
    else:
        name = '_OScorr_Biascorr_Darkcorr.fits'
        Flat_OScorr_Biascorr_Darkcorr = 0
    finalImage = fits.open(filename)   
    FinalImage_OScorr, FinalImage_OScorrName = ApplyOverscanCorrection(finalImage.filename(), stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20])
    FinalImage_OScorr_Biascorr = FinalImage_OScorr.data - Bias_OScorr.data
    FinalImage_OScorr_Biascorr_Darkcorr = FinalImage_OScorr_Biascorr - (Dark_OScorr_Biascorr_per_sec * finalImage[0].header[my_conf.exptime[0]] )
    FinalImage_OScorr_Biascorr_Darkcorr_Flatcorr = FinalImage_OScorr_Biascorr_Darkcorr - Flat_OScorr_Biascorr_Darkcorr
    
    finalImage[0].data = FinalImage_OScorr_Biascorr_Darkcorr_Flatcorr
    fitswrite(finalImage[0], finalImage.filename()[:-5] + name)
    return finalImage, finalImage.filename()[:-5] + name



def DS9Trimming(xpapoint, config=my_conf):
    """Crop the image to have only the utility area
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
    try:
        region = getregion(d)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    except ValueError:
        area = [0,-1,1053,2133]
        
    for filename in path:
        print(filename)
        result, name = ApplyTrimming(filename, area=area) 
    print(name)
    if len(path) < 2:
        d.set('frame new')
        d.set('tile yes')
        d.set("file %s" %(name))  
        d.set('file ' + name)  
    return

def ApplyTrimming(path, area=[0,-0,1053,2133], config=my_conf):
    """Apply overscan correction in the specified region, given the two overscann areas
    """
    from astropy.io import fits
    fitsimage = fits.open(path)[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]
    #image = image[area[0]:area[1],area[2]:area[3]]
    fitsimage.data = image
    name = path[:-5] + '_Trimm.fits'
    fitswrite(fitsimage, name)
    return fitsimage, name

def DS9CLcorrelation(xpapoint, config=my_conf):
    """
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    try:
        region = getregion(d)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    except ValueError:
        area = [0,-1,1053,2133]
        
    for filename in path:
        print(filename)
        CLcorrelation(filename, area=area) 
    return

def CLcorrelation(path, area=[0,-1,1053,2133], DS9backUp = DS9_BackUp_path, config=my_conf):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    from astropy.io import fits
    from matplotlib import pyplot as plt
    fitsimage = fits.open(path)[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]
    #image = image[area[0]:area[1],area[2]:area[3]]
    imagex = np.nanmean(image, axis=1)
    imagey = np.nanmean(image, axis=0)
    nbins=100
    fig, ax = plt.subplots(2, 2)#, figsize=(12,7))
    ax[0,0].hist(imagex[1:]-imagex[:-1],bins=10*nbins,histtype='step',label='Lines')
    ax[1,0].hist(imagey[1:]-imagey[:-1],bins=10*nbins,histtype='step',label='Column', color='orange')
#    ax[0,1].hist((image[1:,:] - image[:-1,:]).flatten(),bins=nbins,histtype='step',label='Column', color='orange')
    #ax[0,1].hist((image[1:,:] - image[:-1,:]).flatten(),bins=10*nbins,histtype='step',label='Column', color='orange')
    x = (image[:,1:] - image[:,:-1]).flatten()
    y = (image[1:,:] - image[:-1,:]).flatten()
    ax[0,1].hist(x[np.isfinite(x)],bins=30*nbins,histtype='step',label='Line')
    ax[1,1].hist(x[np.isfinite(x)],bins=30*nbins,histtype='step',label='Line')
    ax[1,1].hist(y[np.isfinite(y)],bins=30*nbins,histtype='step',label='Column', color='orange')
    ax[1,1].set_xlim((-100,100))
    ax[0,1].set_xlim((-100,100))
    #ax[1,1].set_ylim((-10,40000))
    #ax[0,1].set_ylim((-10,40000))
    for axx in ax.flatten():
        axx.grid()
    ax[0,0].legend();    ax[0,1].legend()
    ax[1,0].legend();    ax[1,1].legend()
    ax[1,0].set_xlabel('abs(CL(n) - CL(n-1))')
    ax[1,1].set_xlabel('abs(P(n) - P(n-1))')
    print('ok')
    fig.suptitle('Histograms for Columns/Lines correlation analysis \n' + os.path.basename(path), y=0.95)
    plt.show()
    print('ok')
    return


def ComputeCIC(xpapoint, save=True, DS9backUp = DS9_BackUp_path):
    """Compute the CIC on a serie of images
    """
    from astropy.table import Table
    import matplotlib; matplotlib.use('TkAgg')  
    from matplotlib import pyplot as plt
    d = DS9(xpapoint)
    filename = d.get('file')
    try:
        cat = Table.read(sys.argv[3])
    except FileNotFoundError:
        cat = DS9CreateHeaderCatalog(xpapoint, info=True)
        cat = Table.read(os.path.join(os.path.dirname(filename), 'HeaderCatalog_info.csv'))

    t = cat[my_conf.exptime[0]]
    print(cat)
    fit1 = np.polyfit(t, cat['MeanADUValueTR'], 1)
    fit2 = np.polyfit(t, cat['MeanADUValueBR'], 1)
    fit3 = np.polyfit(t, cat['MeanADUValueTL'], 1)
    fit4 = np.polyfit(t, cat['MeanADUValueBL'], 1)
    fit = np.polyfit(t, cat['MeanADUValue'], 1)

    #xmax = x[np.argmax(y)]
    fit_fn = np.poly1d(fit) 
    fit_fn1 = np.poly1d(fit1) 
    fit_fn2 = np.poly1d(fit2) 
    fit_fn3 = np.poly1d(fit3) 
    fit_fn4 = np.poly1d(fit4) 
    
    
    #adu = np.linspace(cat['MeanADUValue'].min(),cat['MeanADUValue'].max(),1000)
    plt.figure()#figsize=(12,8))
    norm=0.1
    plt.plot(t,fit_fn(t), linestyle='dashed' , c='b', label='CIC=%0.2f'%(fit_fn[0]))
    plt.plot(t,fit_fn1(t), linestyle='dashed', c='orange', label='CIC=%0.2f'%(fit_fn1[0]))
    plt.plot(t,fit_fn2(t), linestyle='dashed', c='black', label='CIC=%0.2f'%(fit_fn2[0]))
    plt.plot(t,fit_fn3(t), linestyle='dashed', c='red', label='CIC=%0.2f'%(fit_fn3[0]))
    plt.plot(t,fit_fn4(t), linestyle='dashed', c='green', label='CIC=%0.2f'%(fit_fn4[0]))
    plt.scatter(t,cat['MeanADUValue'], label = 'All', c='b', alpha=norm)
    plt.scatter(t,cat['MeanADUValueTR'], label = 'Top right part', c='orange', alpha=norm)
    plt.scatter(t,cat['MeanADUValueBR'], label = 'Bottom right part', c='black', alpha=norm)
    plt.scatter(t,cat['MeanADUValueTL'], label = 'Top left part', c='red', alpha=norm)
    plt.scatter(t,cat['MeanADUValueBL'], label = 'Bottom left part', c='green', alpha=norm)
    plt.grid(True, linestyle='dotted')
    plt.title('CIC measurements #im %i-%i: T=%i, gain=%i'%(np.min(cat['IMNO']),np.max(cat['IMNO']),np.mean(cat[my_conf.temperature[0]]),np.mean(cat[my_conf.gain[0]])))
    plt.xlabel('Exposure time [s] ')
    plt.ylabel('ADU value')
    plt.legend()
    if save:
        plt.savefig(DS9backUp + 'Plots/%s_CICmeasurement.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")))    #plt.subplot_tool()
    plt.show()
    

    return 

def DS9SubstractBackground(xpapoint, save=True):
    """Apply the background substraction fonction to the DS9 images or several if given
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    for filename in path:
        print(filename)
        result, name = BackgroundSubstraction(filename, stddev=10, save=save)
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name) 
    return 

def BackgroundSubstraction(path, stddev=13, save=False, config=my_conf):
    """Generate a background of the given image using a 2D gaussian convolution that smoothes the signal
    Background is saved and then substracted from the given image. The bckd substracted image is saved as _BCKDsubstracted.fits
    """
    from astropy.io import fits
    from scipy import ndimage
    from astropy.convolution import interpolate_replace_nans
    from astropy.convolution import Gaussian2DKernel 
    if type(path) == str:
        print(path)
        image = fits.open(path)[0]
    else:
        image = path
    imagesub = image.copy()
    data = image.data
    if np.isfinite(image.data).all() is False:       
        kernel = Gaussian2DKernel(stddev=stddev)
        image.data = interpolate_replace_nans(image.data, kernel)    
    smoothed_image = ndimage.gaussian_filter(image.data, sigma=(stddev, stddev), order=0)

    imagesub.data = data.astype(np.int16) - smoothed_image.astype(np.int16)
    name = ''
    if save:

        name = path[:-5] + '_BCKDsubstracted.fits'
        fitswrite(imagesub, name)
        image.data = smoothed_image.astype('uint16')
        fitswrite(image,path[:-5] + '_smoothed.fits')
    return imagesub, name


def reject_outliers(data, stddev = 3.):
    """Reject outliers above std=m, taking care of nan values
    """
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    return data[s<stddev]


def DS9ComputeStandardDeviation(xpapoint,radius=50):
    """
    """
    #from astropy.io import fits
    #from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    #path = getfilename(d)#d.get("file")
    fitsimage = d.get_fits()[0]#fits.open(path)[0] 
    areas = CreateAreas(fitsimage.data, area=None, radius=radius)
    lx, ly = fitsimage.data.shape
    image3D = residual([0,0], fitsimage.data, np.zeros((lx,ly,2)), areas, radius)

    
    areas = np.array(areas)
    create_DS9regions2(areas[:,2]+float(radius)/2,areas[:,0]+float(radius)/2, radius=radius, form = 'box',
                       save=True,color = 'yellow', savename='/tmp/centers')
    d.set('regions /tmp/centers.reg')
    print(bcolors.BLACK_RED + 'Standard deviation = %0.1f'%(np.nanstd(image3D)) + bcolors.END)
    return image3D

def DS9HistogramDifferences(xpapoint):
    """Histogram differences
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    im2substract = sys.argv[3]
    for filename in path:
        print(filename)
        result = HistogramDifferences(filename, im2substract)
    return result

def HistogramDifferences(filename, im2substract, DS9backUp = DS9_BackUp_path):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    from matplotlib import pyplot as plt
    from astropy.io import fits
    im1 = fits.open(filename)[0].data
    im2 = fits.open(im2substract)[0].data
    nbins = 150#228
    nrange = (np.nanpercentile(np.hstack((im1,im2)),0.1), np.nanpercentile(np.hstack((im1,im2)),99.9)   )
    hist1, x1 = np.histogram(im1.flatten(), bins=nbins, range=nrange)
    hist2, x2 = np.histogram(im2.flatten(),  bins=nbins, range=nrange)
    c1 = (x1[1:] + x1[:-1]) / 2
    n = 10
    beg1 = 6 * int(nbins/n)
    end1 = 8 * int(nbins/n)
    beg2 = 1 * int(nbins/n)
    end2 = 3 * int(nbins/n)
    if np.nansum((hist1-hist2)[beg1:end1]) > np.nansum((hist2-hist1)[beg1:end1]):
        slope1, intercept1 = fitLine(c1[beg1:end1], np.log(hist1-hist2)[beg1:end1])
    else:
        slope1, intercept1 = fitLine(c1[beg1:end1], np.log(hist2-hist1)[beg1:end1])
    
    if np.nansum((hist1-hist2)[beg2:end2]) > np.nansum((hist2-hist1)[beg2:end2]):
        slope2, intercept2 = fitLine(c1[beg2:end2], np.log(hist1-hist2)[beg2:end2])
    else:
        slope2, intercept2 = fitLine(c1[beg2:end2], np.log(hist2-hist1)[beg2:end2])



    xlinefit1 = np.linspace(c1[beg1], c1[end1], 1000)
    ylinefit1 = linefit(xlinefit1, slope1, intercept1)
    xlinefit2 = np.linspace(c1[beg2], c1[end2], 1000)
    ylinefit2 = linefit(xlinefit2, slope2, intercept2)


    f, (ax1, ax2) = plt.subplots(2, sharex=True)#, figsize=(8,6))
    ax1.step(c1,np.log(hist1), label='H1')
    ax1.step(c1,np.log(hist2), label='H2')
    ax1.legend()
    #f.suptitle('Histogram difference')
    ax1.set_ylabel('Histograms')
    ax2.set_ylabel("Histograms' difference")
    ax2.set_xlabel("Pixel value [ADU]")
    ax1.set_title('%s\n%s'%(os.path.basename(filename), os.path.basename(im2substract)), fontsize=8)
    ax2.step(c1,np.log(hist1-hist2), label='H1 - H2')
    ax2.step(c1,np.log(-hist1+hist2), label='H2 - H1')
    ax2.plot(xlinefit1,ylinefit1, label='Linear fit')
    ax2.plot(xlinefit2,ylinefit2, label='Linear fit')
    ax2.text(-200,4.5,'y = %0.4f x + %0.2f  ==> Gtot = %0.1f'%(slope1, intercept1, -1./slope1))
    ax2.text(-200,3.5,'y = %0.4f x + %0.2f  ==> Gtot = %0.1f'%(slope2, intercept2, 1./slope2))
    ax2.legend()
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
    f.tight_layout()
    plt.show()
    print('Done')
    csvwrite(np.vstack((c1,np.log(hist1-hist2),np.log(-hist1+hist2))).T, DS9backUp + 'CSVs/%s_HistogramDifference.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return

#os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' ''')
#from Cocoa import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
#app = NSRunningApplication.runningApplicationWithProcessIdentifier_(os.getpid())
#app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps)

def get_size(start_path):
    """
    One line : sum( os.path.getsize(os.path.join(dirpath,filename)) for dirpath, dirnames, filenames in os.walk( path ) for filename in filenames )

    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total_size += os.path.getsize(fp)
            except FileNotFoundError as e:
                print(e)
    return total_size

def AnalyseDark(xpapoint, path=None,c_gain=0.53, config=my_conf):
    #filename = '/Users/Vincent/DS9BackUp/subsets/190424_10H34m15/190203/CosmicRayFree/StackedImage_1795-1799-Gain9200-Texp30-Temp-115.fits'
    #np.nanmean((image1- ComputeOSlevel1(image1))[1200:1900,1200:2000]/texp)
    from astropy.io import fits
    if path is None:
        path = sys.argv[-1]
    tab = Table( names=('file', 'subtracted_file', 'exp1', 'exp2', 'dark1','dark2','dark_bottom', 'dark_middle', 'dark_top','Temp1','Temp2'),dtype=('S200','S200',float,float,float,float,float,float,float,float,float))
    files = glob.glob(os.path.join(path , '**/StackedImage*.fits'),recursive=True)
    fpath = np.unique([os.path.dirname(f) for f in files])
    pol = np.poly1d([5.49129191e+00, 1.13611824e+03, 5.94756000e+04])#from LAM measurement 19 april 2019
    for pathi in fpath:
        filesi = glob.glob(os.path.join(pathi , 'StackedImage*.fits'))
        for file_init in  filesi:
            fitsimage = fits.open(file_init)[0]
            texp = fitsimage.header[my_conf.exptime[0]]
            texps = np.array([fits.getheader(file)[my_conf.exptime[0]] for file in filesi])
            for  file,texp2 in zip(filesi,texps):
                if texp2 < texp:
                    print(pathi)
                    print('Compute dark, ',file_init,file)
                    fitsimage2 = fits.open(file)[0] 
                    image2 = fitsimage2.data - ComputeOSlevel1(fitsimage2.data)
                    image1 = fitsimage.data - ComputeOSlevel1(fitsimage.data)
                    darkim1 = image1[100:1900,1200:2000]/ texp
                    darkim2 = image2[100:1900,1200:2000] /  texp2
                    dark1 = (image1[100:700,1200:2000] - image2[100:700,1200:2000]) / (texp - texp2)
                    dark2 = (image1[700:1200,1200:2000] - image2[700:1200,1200:2000]) / (texp - texp2)
                    dark3 = (image1[1200:1900,1200:2000] - image2[1200:1900,1200:2000]) / (texp - texp2)
                    T1, T2 = fitsimage.header['EMCCDBAC'],fitsimage2.header['EMCCDBAC']
                    tab.add_row([file_init,file,texp,texp2,3600*np.nanmean(darkim1)/pol(T1)/c_gain,3600*np.nanmean(darkim2)/pol(T2)/c_gain,
                                 3600*np.nanmean(dark1)/pol(T1)/c_gain,3600*np.nanmean(dark2)/pol(T1)/c_gain,3600*np.nanmean(dark3)/pol(T1)/c_gain,
                                 T1, T2])
    csvwrite(tab,os.path.join(path,'Dark_catalog.csv'))  
    plt.figure()
    plt.plot(tab['Temp1'],3600*tab['dark_bottom']/pol(tab['Temp1']),'o',label='Difference: Bottom part')
    plt.plot(tab['Temp1'],3600*tab['dark_middle']/pol(tab['Temp1']),'o',label='Difference: Medium part')
    plt.plot(tab['Temp1'],3600*tab['dark_top']/pol(tab['Temp1']),'o',label='Difference: Top part')
    plt.plot(tab['Temp1'],3600*tab['dark1']/pol(tab['Temp1']),'.',label='Dark image 1')
    plt.plot(tab['Temp1'],3600*tab['dark2']/pol(tab['Temp1']),'.',label='Dark image 2')
    plt.legend()
    plt.xlabel('EMCCD temperature [C]')
    plt.ylabel('Dark current [e-/pix/hour]')
    plt.title('Dark current evolution with T')
    plt.show()
    
    return tab                  



        

def DS9CreateHeaderCatalog(xpapoint, files=None, filename=None, info=True, redo=False, onlystacks=False, name='', config=my_conf):
    """0.5 second per image for info
    10ms per image for header info, 50ms per Mo so 240Go-> 
    """
    from astropy.table import vstack
    if sys.argv[3] == '1':
        info=True
    elif sys.argv[3] == '0':
        info = False
    if sys.argv[4] == '1':
        onlystacks=True
        print('Running analysis only on stacked images' )
        name = '_on_stack'
    elif sys.argv[4] == '0':
        onlystacks = False

    if xpapoint:
        d = DS9(xpapoint)
        if filename is None:
            filename = getfilename(d)        
        if files is None:
            #if len(sys.argv) > 5: files = Charge_path_new(filename, entry_point=5)
            path = Charge_path_new(filename) if len(sys.argv) > 5 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    if os.path.isdir(sys.argv[-1]):
        fname = sys.argv[-1]
    elif filename is not None:
        fname = filename
    else:
        fname = os.path.dirname(os.path.dirname(files[0]))
#            if filename is None:
#                filename = getfilename(d)
#            if len(sys.argv) > 3:
#                print('Not folder given, analyzing numbers...')
    paths = np.unique([os.path.dirname(f) for f in files ])
    print(files)
#    if redo:
    print('Redoing all the calculation')
    if len(paths) > 1:
        if onlystacks:
            t1s = [CreateCatalog(glob.glob(path + '/StackedImage*.fits')) for path in paths if (len(glob.glob(path + '/StackedImage*.fits'))>0)] 
        else:
            t1s = [CreateCatalog(glob.glob(path + '/image*.fits')) for path in paths if (len(glob.glob(path + '/image*.fits'))>0)] 
        
    else:
        t1s = [CreateCatalog(files)]
#    else:
#        print('Not recomputing catalog')
#        t1s = [CreateCatalog(glob.glob(path + '/image*.fits')) for path in paths if (len(glob.glob(path + '/image*.fits'))>0) & (os.path.isfile(path + '/HeaderCatalog.csv') is False)] 
    if len(t1s)>1:
        t1 = vstack(t1s)
        csvwrite(t1, os.path.join(fname,'TotalMergedCatalog.csv') )
    if info:
        t2s = [CreateCatalogInfo(t) for t in t1s] 
        if len(t2s)>1:
            t2 = vstack(t2s)
            name =  os.path.join(fname,'TotalMergedCatalog_info%s.csv'%(name)) 
            csvwrite(t2,name)
            return t2
        else:
            return t2s[0]
    else:
        if len(t1s)>1:
            return t1
        else:
            return t1s[0]


def GetThreshold(data, nb=100):
    """
    """
    n = np.dot(*data.shape)
    print(nb,n)
    pc = 1 - nb/n
    print('Percentile = ',100*pc)
    T = np.nanpercentile(data,100*pc)
    return T


def AddHeaderFieldCat(xpapoint, conv_kernel=5, comment='- '):
    from scipy import interpolate
    from astropy.io import fits
    import time
    d = DS9(xpapoint)
    filename = getfilename(d)
    Field, catalog, timediff, timeformatImage, timeformatCat = sys.argv[3:3+5]
    print('Field, catalog, timediff, formatImage, formatCat =',Field, catalog, timediff, timeformatImage, timeformatCat)
    #if len(sys.argv) > 3+5: path = Charge_path_new(filename, entry_point=3+5)
    path = Charge_path_new(filename) if len(sys.argv) > 8 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    header0 = fits.getheader(path[0])
    TimeFieldImage = FindTimeField(list(set(header0.keys())))
    if (timeformatImage == '-') or (timeformatImage == "'-'"):    
        timeformatImage = RetrieveTimeFormat(header0[TimeFieldImage])
    #Field = 'EMCCDBack[C]'
    cat = Table.read(catalog)
    TimeFieldCat = FindTimeField(cat.colnames)
    if (timeformatCat == '-') or (timeformatCat == "'-'"):
        timeformatCat = RetrieveTimeFormat(cat[TimeFieldCat][0])
        
    cat = AddTimeSec(cat, TUtimename=TimeFieldCat,  NewTime_s='DATE_s', timeformat=timeformatCat)
    cat.sort('DATE_s')
    plt.figure()
    print(type(cat[Field][0]), type(cat['DATE_s'][0]))
    ok = []
    try:
        yy = np.array(cat[Field], dtype=float)
        t = cat['DATE_s']
    except ValueError:
        ok = [a.replace('-','.').replace('.','0').isdigit() for a in cat[Field]]
        print(ok)
        t, yy = cat['DATE_s'][ok], np.array(cat[Field][ok], dtype=float)
    print(t,yy)
    plt.plot(t,yy,linestyle='dotted', label='Catalog Field')
    t,y = t[conv_kernel:-conv_kernel], np.convolve(yy, np.ones(conv_kernel)/conv_kernel, mode='same')[conv_kernel:-conv_kernel]
    plt.plot(t,y, label='Field convolved')
#    plt.show()
#    plt.plot(t)
#    plt.show()
#    print(t,y)
#    print(type(t),type(y))
#    print(len(t),len(y))
#    sys.exit()
#    t = sorted(set(t))
#    diff = len(y)-len(t)
#    print('diff = ', diff)
#    if diff<10:
#        y = y[:-diff]
    t, index = np.unique(t, return_index=True)
    y = y[index]
    FieldInterp = interpolate.interp1d(t, np.array(y),kind='linear')
    
    for filename in path:
        print(filename)
        header = fits.getheader(filename)
        timeseconds = time.mktime(datetime.datetime.strptime(header[TimeFieldImage], timeformatImage).timetuple())
        try:
            value = FieldInterp(timeseconds - float(timediff) * 3600)
            plt.plot(timeseconds, value, 'o', c = 'red')#label='Images after change time')
        except ValueError:
            value = -999
            plt.plot(timeseconds, np.nanmean(yy), 'P',c='black')#, label='Images out of interpoalation range: -999')
        value = np.round(value,12)
        if 'NAXIS3' in header:
            fits.delval(filename,'NAXIS3')
            print('2D array: Removing NAXIS3 from header...')            
        fits.setval(filename, Field[:8], value = value, comment = comment)
#        print(value)
#        fitswrite(fitsimage,filename)
    plt.plot([], [], 'P',c='black', label='Images out of interpoalation range: -999')
    plt.plot([], [], 'o', c = 'red', label='Images after change time')
    #plt.plot(np.linspace(t.min(),t.max(),1e5), FieldInterp(np.linspace(t.min(),t.max(),1e5)), c = 'grey', label='Linear interplolation')
    plt.title('Addition of the catlog field to image headers')
    plt.xlabel('Time in seconds')
    plt.ylabel(Field)
    plt.legend()
    plt.show()

    return



def AddTimeSec(table, TUtimename='time',  NewTime_s='DATE_s', timeformat="%m/%d/%y %H:%M:%S"):
    import time
    #sec = [datetime.datetime.strptime(moment, timeformat) for moment in table[TUtimename]]
    sec = [time.mktime(datetime.datetime.strptime(moment, timeformat).timetuple()) for moment in table[TUtimename]]
    table[NewTime_s] = np.array(sec)
    return table  

def FindTimeField(liste):
    timestamps = ['Datation GPS','Date','Time', 'Date_TU', 'UT Time', 'Date_Time', 'Time UT']
    timestamps_final = [field.upper() for field in timestamps] + [field.lower() for field in timestamps] + timestamps
    try:
        timestamps_final.remove('date')
    except ValueError:
        pass
    for timename in timestamps_final:
        if timename in liste:#table.colnames:
            timefield = timename
    try:
        print('Time field found : ', timefield)
        return timefield
    except UnboundLocalError:
        return liste[0]#table.colnames[0]

def RetrieveTimeFormat(time):
    formats = ["%d/%m/%Y %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S"]
    form = []
    for formati in formats:
        try:
           datetime.datetime.strptime(time, formati)
           form.append(True)
        except ValueError:
            form.append(False)
    return formats[np.argmax(form)]

 


def CreateCatalog(files, config=my_conf):
    """
    """
    from astropy.table import Table, Column
    from astropy.io import fits
    import warnings
    from tqdm import tqdm
    warnings.simplefilter('ignore', UserWarning)
    files.sort()
    path = files[0]
    header = fits.getheader(path)
    fields = list(set(header.keys())) + ['PATH'] + ['FPATH'] 
    #t = Table(names=fields,dtype=('S20,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    t = Table(names=fields,dtype=('S20,'*len(fields[:-2])).split(',')[:-1] + ['S200'] + ['S200'])#,dtype=('S10,'*30).split(',')[:-1])
    #files = glob.glob(os.path.dirname(path) + '/*.fits')
    #files.sort()
    for i in tqdm(range(len(files))):
        file = files[i]
        t.add_row()
        t['PATH'][i] = file
        t['FPATH'][i] = os.path.dirname(file)
        #print(100*i/len(files),'%')
        header = fits.getheader(file)
        fields_image = list(set(header.keys())) 
        for field in fields_image:
            if field in fields:
                try:
                    t[field][i] = float(header[field])
                except (TypeError, ValueError) as e:
                    #print(e)
                    try:
                        t[field][i] = header[field]
                    except ValueError as e:
                        #print(e)
                        t[field][i] = -1 
                    except KeyError as e:
                        #print(e)
                        pass 
    t['index'] = np.arange(len(t))
    #t.remove_columns(['EXTEND','SIMPLE','NAXIS','COMMENT','NAXIS3','SHUTTER','VSS','BITPIX','BSCALE'])
    for col in ['EXTEND','SIMPLE','NAXIS','ROS','NAXIS1','NAXIS2','NAXIS3','BSCALE','COMMENT','COMMENT']:
        try:
            t.remove_column(col)
        except KeyError:
            pass
    try:
        dates = [int(''.join(date.split('T')[0].split('-')[:3])[2:]) for date in t['DATE']]
        t.add_columns([Column(name='date', data=dates)])
    except KeyError:
        print('No date in this header')
        pass

    csvwrite(t,os.path.dirname(path) + '/HeaderCatalog.csv')
    #t.write(os.path.dirname(path) + '/HeaderCatalog.csv', overwrite=True)
    try:
        file = open(os.path.dirname(path) + '/Info.log','w') 
        file.write('##########################################################################\n') 
        file.write('\nNumber of images : %i'%(len(t))) 
        file.write('\nGains: ' + '-'.join([gain for gain in np.unique(t[my_conf.gain[0]])]))
        file.write('\nExposures: ' + '-'.join([exp for exp in np.unique(t[my_conf.exptime[0]])]))
        file.write('\nTemps: ' + '-'.join([te for te in np.unique(t[my_conf.temperature[0]])]))
        try:
            file.write('\nTemps emCCD: ' + ','.join([np.str(te) for te in np.unique(np.round(np.array(t['EMCCDBAC'],dtype=float),1))]))
        except KeyError:
            pass
        file.write('\nNumber of images per gain and exposure: %0.2f'%(len(t[(t[my_conf.gain[0]]==t[my_conf.gain[0]][0]) & (t[my_conf.exptime[0]]==t[my_conf.exptime[0]][0])])) )
        file.write('\n\n##########################################################################')      
        file.close() 
    except:
        pass
    return t



def FluxDeffect(filename, Plot=False, config=my_conf):
    from astropy.io import fits
    fitsimage = fits.open(filename)[0]
    texp = fitsimage.header[my_conf.exptime[0]]
    flux = (np.nanmean(fitsimage.data[1158-25:1158+25,2071-50:2071+50]) - np.nanmean(fitsimage.data[1158-25+50:1158+25+50,2071-50:2071+50]))/texp
    return {'Flux':flux}



def CreateCatalogInfo(t1, verbose=False, config=my_conf):
    """
    """
    from astropy.table import Table
    from astropy.io import fits
    from astropy.table import hstack
    from tqdm import tqdm
    files = t1['PATH']
    path = files[0]
    fields = ['Col2ColDiff', 'Line2lineDiff', 'OverscannRight', 'OverscannLeft', 'Gtot_var_int_w_OS',
              'TopImage', 'BottomImage', 'MeanADUValue', 'SaturatedPixels','MeanFlux','OS_SMEARING1','OS_SMEARING2',
              'stdXY', 'stdY', 'stdX', 'MeanADUValueTR', 'MeanADUValueBR', 'MeanADUValueBL', 'MeanADUValueTL','Gtot_var_int_wo_OS',
              'Smearing_coeff_phys','GainFactorVarIntens','GainFactorHist','BrightSpotFlux','Top2BottomDiff_OSL','Top2BottomDiff_OSR']
    #t = Table(names=fields,dtype=('S20,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    t = Table(names=fields,dtype=('float,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    #for i, file in enumerate(files):
#    for i,file in tqdm(enumerate(files)):
    for i in tqdm(range(len(files))):
        file = files[i]
        t.add_row()
        verboseprint(i,verbose=verbose)
        data = fits.getdata(file)
        try:
            texp = fits.getheader(file)[my_conf.exptime[0]]
        except KeyError:
            texp = fits.getheader(file)['EXPOSURE']
        column = np.nanmean(data[2:1950,1100:2100], axis=1)
        line = np.nanmean(data[2:1950,1100:2100], axis=0)
        offset = 20
        OSR1 = [offset,-offset,offset,400]
        OSR2 = [offset,-offset,2200,2400]
        OSR = data[OSR2[0]:OSR2[1],OSR2[2]:OSR2[3]]
        OSL = data[OSR1[0]:OSR1[1],OSR1[2]:OSR1[3]]
        t[i]['Col2ColDiff'] =  np.nanmean(abs(line[1:] - line[:-1])) 
        t[i]['Line2lineDiff'] = np.nanmean(abs(column[1:] - column[:-1])) 
        t[i]['OverscannRight'] = np.nanmean(OSR)
        t[i]['OverscannLeft'] = np.nanmean(OSL)
        t[i]['TopImage'] = np.nanmean(column[:20])
        t[i]['BottomImage'] = np.nanmean(column[-20:])
        t[i]['Top2BottomDiff_OSL'] = np.nanmean(OSL[:20,:]) - np.nanmean(OSL[-20:,:])
        t[i]['Top2BottomDiff_OSR'] = np.nanmean(OSR[:20,:]) - np.nanmean(OSR[-20:,:])
        t[i]['MeanFlux'] =  t[i]['MeanADUValue']/texp
        t[i]['MeanADUValueTR'] =  np.nanmean((data - ComputeOSlevel1(data))[1000:1950,1600:2100])
        t[i]['MeanADUValueBR'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1000,1600:2100])
        t[i]['MeanADUValueBL'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1000,1100:1600])
        t[i]['MeanADUValueTL'] =  np.nanmean((data - ComputeOSlevel1(data))[1000:1950,1100:1600])
        t[i]['SaturatedPixels'] = 100*float(np.sum(data[2:1950,1100:2100]>60000)) / np.sum(data[2:1950,1100:2100]>0)
        t[i]['stdXY'] = np.nanstd(data[2:1950,1100:2100])
        t[i]['BrightSpotFlux'] = (np.nanmean(data[1158-25:1158+25,2071-50:2071+50]) - np.nanmean(data[1158-25+50:1158+25+50,2071-50:2071+50]))/texp
        emgain = ComputeEmGain(file, None,True,False,None,None,[40,40],False,[1053,2133,500,2000])
        t[i]['Gtot_var_int_w_OS'],t[i]['Gtot_var_int_wo_OS'] = emgain['EMG_var_int_w_OS'], emgain['EMG_var_int_wo_OS']
        t[i]['Smearing_coeff_phys'] = SmearingProfileAutocorr(file,None,DS9_BackUp_path,'',False,'x')['Exp_coeff']
        t[i]['GainFactorVarIntens'] = 1/Smearing2Noise(t[i]['Smearing_coeff_phys'])['Var_smear']
        t[i]['GainFactorHist'] =  1/Smearing2Noise(t[i]['Smearing_coeff_phys'])['Hist_smear']
#        smearingOS = AnalyzeOSSmearing(file)
#        t[i]['OS_SMEARING1'] = smearingOS['Exp1']
#        t[i]['OS_SMEARING2'] =       smearingOS['Exp2']
        try:
            t[i]['stdX'] = np.nanstd(data[1500,1100:2100])
            t[i]['stdY'] = np.nanstd(data[2:1950,1500])
        except IndexError:
            t[i]['stdX'] = np.nanstd(data[500,:])
            t[i]['stdY'] = np.nanstd(data[:,500])            
        t[i]['MeanADUValue'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1950,1100:2100])
    new_cat = hstack((t1,t))
    #t.remove_columns(['EXTEND','SIMPLE','NAXIS','COMMENT','NAXIS3','SHUTTER','VSS','BITPIX','BSCALE'])
    #new_cat.write(os.path.dirname(path) + '/HeaderCatalog_info.csv', overwrite=True)
    csvwrite(new_cat,os.path.dirname(path) + '/HeaderCatalog_info.csv')
    print(new_cat)
    print(new_cat[my_conf.gain[0]].astype(float)>0)
    error_cat = new_cat[(new_cat[my_conf.gain[0]].astype(float)>0)&(new_cat['OverscannRight'].astype(float)>4500)]['PATH']
    if len(error_cat)>0:
        try:
            file = open(os.path.dirname(path) + '/Info.log','a') 
            file.write('\n\n##########################################################################\n')      
            file.write('ERROR - ERROR - ERROR - ERROR - ERROR - ERROR - ERROR - ERROR - \n') 
            file.write('\n##########################################################################')      
            file.write('\nNumber of images with EMGAIN error: %i'%(len(error_cat))) 
            file.write('\nPath of the images: '+repr(error_cat))
            file.close() 
        except:
            pass
    return new_cat    

def DS9PlotSpatial(xpapoint):
    """Plot the spatial dimension of the detector in order to detect a spectra
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    field = sys.argv[3]
    print(path)#    print('len(path)=',len(path))
    plot_flag = False
    if len(path) < 2:
        plot_flag = True
    for filename in path:
        print(filename)
        result = PlotSpatial1(filename, field=field, plot_flag=plot_flag)#2133+offset,-offset]) 
    return result


def createFieldMask(image, x, y, xlen, ylen):#vpi
    """With mask centered on 206nm
    """
    mask = image.copy()
    mask[:,:] = 1
    for xi, yi in zip(x,y):
        print(xi, yi)
        mask[int(xi) - xlen : int(xi) + xlen, :int(yi) - ylen] = 0
        mask[int(xi) - xlen : int(xi) + xlen,int(yi) + ylen:] = 0
#        print(mask[int(xi) - xlen : int(xi) + xlen, int(yi) - ylen : int(yi) + ylen])
    image[mask==0] = np.nan
    print(float(np.nansum(image)) / len(image.flatten()))
    return image




def PlotSpatial1(filename, field, save=True, plot_flag=True, DS9backUp = DS9_BackUp_path):
    """
    Without mask centered on 206nm
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from scipy.optimize import curve_fit
    #print('Entry = ', field)
    fitsfile = fits.open(filename)#d.get_fits()#fits.open(filename)
    image = fitsfile[0].data    
    x, y, redshift, slit, w = returnXY(field,keyword=None, frame='observedframe')  
    new_im = image[0:1870,1070:2100]
    #if plot_flag:
    offset = 5
    #stdd = np.nanstd(new_im,axis=1)[offset:-offset]
    new_im = np.convolve(np.nanmean(new_im,axis=1), np.ones(3)/3, mode='same')[offset:-offset]
    #new_im = np.nanmean(new_im,axis=1)[5:-5]
    n=2*10
    x1 = 360
    xfit1 = np.arange(x1,len(new_im))
    xfit2 = np.arange(x1)
    try:
        fit1 = np.polyfit(xfit1, new_im[x1:], n)
        fit2 = np.polyfit(xfit2, new_im[:x1], n)
    except ValueError as e:
        print ('Error with the fit, Nan value: ',e)
        p1 = lambda x: np.nanmean(new_im) + x * 0
        p2 = p1
    else:
        p1 = np.poly1d(fit1)
        p2 = np.poly1d(fit2)    
    f, (ax1, ax2) = plt.subplots(2,1, sharex=True, gridspec_kw = {'height_ratios':[3, 1]})#, figsize=(11,7))
    ax1.plot(new_im, label='Detector columns')
    ax1.plot(xfit1, p1(xfit1), label='Background fit', c='orange')
    #ax1.set_ylim((-50,100))
    ax1.plot(xfit2, p2(xfit2), c='orange')
    ax2.set_xlabel('Column index')
    ax1.set_ylabel('Average line Value [ADU] - K3 convolved')
    yy = [new_im[int(xi)] for xi in x]
    #plt.plot(x, yy,'P',label='Slits position')
    x += offset
    ax1.hlines(yy, x - 10, x + 10,label='Slits position', colors='black')
    for i, sliti in enumerate(slit):
        if i% 4 == 1:
            n =  1
        elif i% 4 == 2:
            n = + 0.5
        elif i% 4 == 3:
            n =  -1
        elif i% 4 == 0:
            n = -0.5
        ax1.text(x[i], yy[i] + np.nanmax(abs(new_im[x1:]-p1(xfit1))) * n,str(sliti),bbox=dict(facecolor='red', alpha=0.3),fontsize=8)
        ax1.vlines(x[i], yy[i] + 20.3 * n, yy[i], linestyles='dotted', colors='red')

    #plt.ylim(0.9*np.nanmin(new_im) -  1.3 * n, 1.1*np.nanmax(new_im)  +  1.3 * n)
#    stdd1 = np.nanstd(new_im[x1+100:]-p1(xfit1[100:]))
#    mean1 = np.nanmean(new_im[x1+100:]-p1(xfit1[100:]))

    stdd1 = np.nanstd(new_im[-100:]-p1(xfit1[-100:]))
    mean1 = np.nanmean(new_im[-100:]-p1(xfit1[-100:]))
    #print('xfit1 = ',xfit1)
    #print('xfit2 = ',xfit2)
    #n=100
 #   argmaxx = int(np.nanmax(new_im[x1+n:-n]-p1(xfit1[n:-n])))
    #plt.plot(new_im[x1+n:-n]-p1(xfit1[n:-n]));plt.show()
    argmaxx = np.argmax(new_im[x1:]-p1(xfit1))
    print('argmax = ', argmaxx)
    popt, pcov = curve_fit(gaussian, xfit1, new_im[x1:]-p1(xfit1), p0=(np.nanmax(new_im[x1:]-p1(xfit1)),xfit1[argmaxx],5))
    print('x0 = ', xfit1[argmaxx])
    print('popt = ', popt)

    
    for k in [3,5,7,10]:
        ax2.plot(np.hstack((xfit1,xfit2)), np.ones(len(np.hstack((xfit1,xfit2)))) * (mean1 + k * stdd1), color='grey', linewidth=0.8, linestyle='dotted')
        ax2.text(200, mean1 + k * stdd1, '%i sigma'%(k), fontsize=9)
    ax2.plot(xfit1, new_im[x1:]-p1(xfit1), color='orange')
    ax2.plot(xfit2, new_im[:x1]-p2(xfit2), color='orange', label='Background substracted')
    ax2.text(xfit1[argmaxx]+50, 700,"Amp = %0.1fADU\nFWHM = %0.2f''"%(popt[0],1.1*2.35*popt[2]),bbox=dict(facecolor='red', alpha=0.3),fontsize=8)
    ax2.plot(xfit1, gaussian(xfit1, *popt),'--', color='grey')
    ax2.plot(xfit2, gaussian(xfit2, *popt),'--', color='grey', label='Gaussian fit')
    ax2.text(1100, 30, 'Std(signal) = %0.1f'%(stdd1))
    ax1.legend()
    ax2.legend()
    ax1.grid(linestyle='dotted')
    ax2.grid(linestyle='dotted')
#    ax1.set_title('Spatial detection on F2 - Fort Sumner - 2018-09-23T02:28:27')
    ax1.set_title(os.path.basename(filename))
    f.tight_layout()
    #ax2.step(xfit1, new_im[x1:]-p1(xfit1), color='orange')
    #ax2.errorbar(xfit1, new_im[x1:]-p1(xfit1), yerr=stdd[x1:], color='orange')
    #plt.show()
    if save:#vpicouet
        if not os.path.exists(os.path.dirname(filename) +'/detection'):
            os.makedirs(os.path.dirname(filename) +'/detection')
        try:
            f.savefig(os.path.dirname(filename) +'/detection/'+ os.path.basename(filename).replace('.fits', '.detect.png'), dpi = 300, bbox_inches = 'tight')
        except ValueError:
            pass
    if plot_flag:
        #f.canvas.manager.window.raise_()
        plt.show()
    else:
        plt.close()
        #f.canvas.manager.window.activateWindow()
    csvwrite(np.vstack((np.arange(len(new_im)),new_im)).T, DS9backUp + 'CSVs/%s_SpatialPlot.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
    return x, y, redshift, slit, w



def DS9OverscanCorrection(xpapoint):
    """Overscan correction line by line
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    for filename in path:
        print(filename)
        offset = 20
        result, name = ApplyOverscanCorrection(filename, stddev=3, 
                                               OSR1=[offset,-offset,offset,400],#1053-offset], 
                                               OSR2=[offset,-offset,2200,2400])#2133+offset,-offset])
    if len(path) < 2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return result


def ApplyOverscanCorrection(path, stddev=3, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20], save=True, config=my_conf):
    """Apply overscan correction line by line using overscan region1234
    """    
    from astropy.io import fits
    fitsimage = fits.open(path)[0]
    image = fitsimage.data.astype(float).copy()
    OScorrection = ComputeOSlevel1(image, OSR1=OSR1, OSR2=OSR2)
    fitsimage.data = image - OScorrection
    #name = path[:-5] + '_OScorr.fits'
    name = os.path.join(os.path.dirname(path) + '/OS_corrected/%s'%(os.path.basename(path)))
    if save:
        fitswrite(fitsimage, name)
    return fitsimage, name

def ComputeOSlevel(image, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20], config=my_conf):
    """Apply average overscan correction  (not line by line)
    """
    OSregion = np.hstack((image[OSR1[0]:OSR1[1],OSR1[2]:OSR1[3]],image[OSR2[0]:OSR2[1],OSR2[2]:OSR2[3]]))
    OScorrection = np.nanmedian(OSregion,axis=1)#reject_outliers(OSregion, stddev=3))
    #OScorrection = OScorrection[..., np.newaxis]*np.ones((image.shape))
    return OScorrection

def ComputeOSlevel1(image, OSR1=[20,-20,20,1053-20], OSR2=[20,-20,2133+20,-20], config=my_conf):
    """Apply overscan correction line by line using overscan region
    """
    OSregion = np.hstack((image[:,OSR1[2]:OSR1[3]],image[:,OSR2[2]:OSR2[3]]))
    OScorrection = np.nanmedian(OSregion,axis=1)#reject_outliers(OSregion, stddev=3))
    OScorrection = OScorrection[..., np.newaxis]*np.ones((image.shape))
    #print(OScorrection)
    return OScorrection



def DS9replaceNaNs(xpapoint):
    """Replace the pixels in the selected regions in DS9 by NaN values
    """
   # from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#d.get("file")
    regions = getregion(d, all=True)
    fitsimage = d.get_fits()[0]#fits.open(filename)[0]
    image = fitsimage.data.astype(float).copy()
    print(regions)
    try:
        xc, yc, h, w = int(regions.xc), int(regions.yc), int(regions.h), int(regions.w)
        print('Only one region found...')        
        print('W = ', w)
        print('H = ', h)
        Xinf = int(np.floor(yc - h/2 -1))
        Xsup = int(np.ceil(yc + h/2 -1))
        Yinf = int(np.floor(xc - w/2 -1))
        Ysup = int(np.ceil(xc + w/2 -1))
        image[Xinf:Xsup+1,Yinf:Ysup+2] = np.nan
    except AttributeError:
        print('Several regions found...')
        for region in regions:
            x,y = np.indices(image.shape)
            try:
                xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
            except AttributeError:
                xc, yc, h, w = int(region.xc), int(region.yc), int(region.r), int(region.r)
                radius = np.sqrt(np.square(y-xc) + np.square(x-yc))
                print(radius)
                mask = radius<h
            else:
                print('W = ', w)
                print('H = ', h)
                Xinf = int(np.floor(yc - h/2 -1))
                Xsup = int(np.ceil(yc + h/2 -1))
                Yinf = int(np.floor(xc - w/2 -1))
                Ysup = int(np.ceil(xc + w/2 -1))
                mask = (x>Xinf) & (x<Xsup+1) & (y>Yinf) & (y<Ysup+1)
#            image[Xinf:Xsup+1,Yinf:Ysup+2] = np.nan
            image[mask] = np.nan
    fitsimage.data = image
    #fitsimage.writeto(filename,overwrite=True)
    filename = fitswrite(fitsimage,filename)
    #d.set('frame new')
    #d.set('tile yes')
    d.set('file '+ filename)
    d.set('pan to %0.3f %0.3f physical' % (xc,yc))
    #d.set('lock yes')
    #d.set("lock frame physical")
    return



def InterpolateNaNs(path, stddev=1):
    """Return int16 fits image with NaNs interpolated
    """
    from astropy.convolution import interpolate_replace_nans
    from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel          
    print(path)
    image = fits.open(path)[0]
    bitpix = int(image.header['BITPIX'])
    result = image.data#vpicouet
    offset = 2**bitpix/2 - np.nanmean(result) 
    result += offset
    stddev = stddev
    while (~np.isfinite(result).all()):
        print(len(np.where(~np.isfinite(result))[0]), 'NaN values found!')
        print('Infinite values in the image, inteprolating NaNs with 2D Gaussian kernel of standard deviation = ', stddev)
        kernel = Gaussian2DKernel(stddev=stddev)
        result = interpolate_replace_nans(result, kernel)#.astype('float16')
        stddev += 1
    result -= offset
#    imshow(result[1000:1500,1000:1500]);colorbar();    plt.figure()imshow(result1[1000:1500,1000:1500]);colorbar()
    #result2 = result - result.min()
    image.data = result.astype('uint32')
    name = path[:-5] + '_NaNsFree.fits'
    fitswrite(image, name)
    return result, name

def DS9InterpolateNaNs(xpapoint):
    """Replace NaNs value of the image in DS9 (or several if specified) by values interpolated around NaNs value
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
    for filename in path:
        if os.path.isfile(filename) is False:
            filename = filename[:-5] + '.CRv.fits'#os.path.join(os.path.dirname(filename),os.path.basename(fielname))
        print(filename)
        result, name = InterpolateNaNs(filename, stddev=2)
        if len(path)<2:
            d.set("lock frame physical")
            d.set('frame new')
            d.set('tile yes')
            d.set('file ' + name)  
    return


def AnalyzeBackground(filename):
    """Analyze background in 2 specfic areas of the images for FB2 because
    of caracteristics of the flight and save it in the header
    """
    from astropy.io import fits
    from scipy import stats
    fitsfile = fits.open(filename)[0]
    LowPartimage = fitsfile.data[16:330,1100:2100]
    HighPart = fitsfile.data[500:1950,1100:2100]
    print ('Image Low part: {}'.format(filename))
    print ('Mean : {}'.format(np.nanmean(LowPartimage)))
    print ('Standard deviation : {}'.format(np.nanstd(LowPartimage)))
    #print ('Skewness: {}'.format(float(stats.skew(LowPartimage,axis=None, nan_policy='omit').data)))    
   
    print ('Image Hight part: {}'.format(filename))
    print ('Mean : {}'.format(np.nanmean(HighPart)))
    print ('Standard deviation : {}'.format(np.nanstd(HighPart)))
    #print ('Skewness: {}'.format(float(stats.skew(HighPart,axis=None, nan_policy='omit').data)))   
    
    fitsfile.header['MEANHIGH'] = np.nanmean(HighPart)
    fitsfile.header['STDHIGH'] = np.nanstd(HighPart)
    try:    
        fitsfile.header['SKEWHIGH'] = float(stats.skew(HighPart,axis=None, nan_policy='omit').data)  
    except AttributeError:
        fitsfile.header['SKEWHIGH'] = float(stats.skew(HighPart,axis=None, nan_policy='omit'))  

    fitsfile.header['MEANLOW'] = np.nanmean(LowPartimage)
    fitsfile.header['STDLOW'] = np.nanstd(LowPartimage)
    try:
        fitsfile.header['SKEWLOW'] = float(stats.skew(LowPartimage,axis=None, nan_policy='omit').data)  
    except AttributeError:
        fitsfile.header['SKEWLOW'] = float(stats.skew(LowPartimage,axis=None, nan_policy='omit'))
    fitsfile.header.comments['SKEWHIGH'] = 'Skewness of a data set on [500:1950,1100:2100]'
    fitsfile.header.comments['STDHIGH'] = 'Mean ADU values of a data set on [500:1950,1100:2100]'
    fitsfile.header.comments['MEANHIGH'] = 'Std ADU avlues of a data set on [500:1950,1100:2100]'

    fitsfile.header.comments['SKEWLOW'] = 'Skewness of a data set on [16:330,1100:2100]'
    fitsfile.header.comments['STDLOW'] = 'Mean ADU values of a data set on [16:330,1100:2100]'
    fitsfile.header.comments['MEANLOW'] = 'Std ADU avlues of a data set on [16:330,1100:2100]'
#    try:
#        fitsfile.header.comments['N_CR'] = 'Number of detected cosmic rays'
#        fitsfile.header.comments['MASK'] = 'Number of pixels deleted due to cosmics'
#    except KeyError:
#        fitsfile.header['N_CR'] = 0
#        fitsfile.header['MASK'] = 0
#        fitsfile.header.comments['N_CR'] = 'Number of detected cosmic rays'
#        fitsfile.header.comments['MASK'] = 'Number of pixels deleted due to cosmics'
    fitsfile.writeto(filename, overwrite=True)
    return

 
ScrollableWindow = None
def Choose_backend(function):
    """
    """
    global ScrollableWindow 
    if 'spectra' in function:  
        if check_if_module_exists('PyQt5'):
            from PyQt5 import QtWidgets
            #import matplotlib.pyplot as plt;from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas;from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar    
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar    
            
            class ScrollableWindow(QtWidgets.QMainWindow):
                """
                """
                import matplotlib.pyplot as plt    
                #matplotlib.use('Qt5Agg')# Make sure that we are using QT5
                def __init__(self, fig):
                    self.qapp = QtWidgets.QApplication([])
                    QtWidgets.QMainWindow.__init__(self)
                    self.widget = QtWidgets.QWidget()
                    self.setCentralWidget(self.widget)
                    self.widget.setLayout(QtWidgets.QVBoxLayout())
                    self.widget.layout().setContentsMargins(0,0,0,0)
                    self.widget.layout().setSpacing(0)    
                    self.fig = fig
                    self.canvas = FigureCanvas(self.fig)
                    self.canvas.draw()
                    self.scroll = QtWidgets.QScrollArea(self.widget)
                    self.scroll.setWidget(self.canvas)
                    self.nav = NavigationToolbar(self.canvas, self.widget)
                    self.widget.layout().addWidget(self.nav)
                    self.widget.layout().addWidget(self.scroll)
                    self.resize(2000, 1000)
                    self.show()
                    self.qapp.exec_()#exit(self.qapp.exec_()) 
                    return


def DS9LoadCSV(xpapoint):
    from astropy.table import Table
    #from .focustest import create_DS9regions2
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
#    if len(sys.argv) > 3:
#    else:
#        path = [d.get("file")]
    sources = Table.read(filename[:-5] + '.csv')
    create_DS9regions2(sources['xcentroid'],sources['ycentroid'], radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers')
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers.reg'))                                        
    return


def DS9ExtractSources(xpapoint):
    """Extract sources for DS9 image and create catalog
    """
    #from .focustest import create_DS9regions2
    #x, y = x+93, y-93
    d = DS9(xpapoint)
    filename = getfilename(d)
    ErosionDilatation, threshold, fwhm, theta, iters, ratio, deleteDoublons = sys.argv[3:3+7]
    threshold = np.array(threshold.split(','),dtype=float)
    fwhm = np.array(fwhm.split(','),dtype=float)
    print('ErosionDilatation, threshold, fwhm, theta, iters, ratio, deleteDoublons = ', ErosionDilatation, threshold, fwhm, theta, iters, ratio, deleteDoublons)
    if len(sys.argv) > 3+7: path = Charge_path_new(filename, entry_point=3+7)

        
    for filename in path:
        print(filename)
        sources = ExtractSources(filename, fwhm=fwhm, threshold=threshold, theta=float(theta), ratio=float(ratio), n=int(ErosionDilatation), iters=int(iters), deleteDoublons=int(deleteDoublons))
        if len(path)<2:
            create_DS9regions2(sources['xcentroid'],sources['ycentroid'], radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers')
            d.set('region delete all')
            d.set('region {}'.format('/tmp/centers.reg')) 
        csvwrite(sources, filename[:-5] + '.csv')
#        try:                                          
#            sources.write(filename[:-5] + '.csv')
#        except UnicodeDecodeError:
#            reload(sys); sys.setdefaultencoding('utf8')
#            sources.write(filename[:-5] + '.csv')            
    return



def ExtractSources(filename, fwhm=5, threshold=8, theta=0, ratio=1, n=2, sigma=3, iters=5, deleteDoublons=3):
    """Extract sources for DS9 image and create catalog
    """
    from astropy.io import fits
    from scipy import ndimage
    from astropy.table import Table
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    from .focustest import delete_doublons
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data
    data2 = ndimage.grey_dilation(ndimage.grey_erosion(data, size=(n,n)), size=(n,n))       
    mean, median, std = sigma_clipped_stats(data2, sigma=sigma, iters=iters)    
#    if quick:    
#        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std,ratio = ratio, theta = theta)         
#        sources0 = daofind(data2 - median)  
#    if not quick:
    daofind = DAOStarFinder(fwhm=fwhm[0], threshold=threshold[0]*std,ratio = ratio, theta = theta)        
    sources0 = daofind(data2 - median)
    print('fwhm = {}, T = {}, len = {}'.format(fwhm[0],threshold[0],len(sources0)))
    for i in fwhm[1:]: 
        for j in threshold[1:]: 
            daofind = DAOStarFinder(fwhm=i, threshold=j*std,ratio =ratio, theta = theta)         
        
            sources1 = daofind(data2 - median) 
            print('fwhm = {}, T = {}, len = {}'.format(i,j,len(sources1)))
            try :
                sources0 = Table(np.hstack((sources0,sources1)))
            except TypeError:
                print('catalog empty')
                if len(sources0)==0:
                    sources0 = sources1
    sources = delete_doublons(sources0, dist = deleteDoublons)
    return sources


#class Fits(object):
#    """
#    """
#    def __init__(self, ds9):
#        """
#        """
#        from astropy.io import fits
#
#        
#        filename = ds9.get('file')
#        if len(filename) == 0:
#            self.filename = ''
#            self.header = None
#        else:
#            if '[IMAGE]' in os.path.basename(filename):
#                filename = filename.split('[IMAGE]')[0]
#            
#            if filename[0] == '.':
#                new_filename = backup_path + filename[1:]
#            
#            else:
#                new_filename = filename
#            
#            self.data = ds9.get_arr2np()
#            self.filename = new_filename
#            self.header = fits.open(self.filename)[0].header


def getfilename(ds9, config=my_conf):
    """
    """
    backup_path = os.environ['HOME'] + '/DS9BackUp'
    if not os.path.exists(os.path.dirname(backup_path)):
        os.makedirs(os.path.dirname(backup_path))
    filename = ds9.get('file')
    
    if '[IMAGE]' in os.path.basename(filename):
        filename = filename.split('[IMAGE]')[0]
    if len(filename)==0:
        new_filename = filename
    elif filename[0] == '.':
        new_filename = backup_path + filename[1:]
    else:
        new_filename = filename
    print(new_filename)
    return new_filename    

def HistogramSums(xpapoint, DS9backUp = DS9_BackUp_path, config=my_conf):
    """
    """
    from astropy.io import fits
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt   
    from astropy.table import Table
    d = DS9(xpapoint)
    #filename = d.get('file')
    filename = getfilename(d)
    if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    

    try:
        region = getregion(d)
    except ValueError:
        Yinf, Ysup,Xinf, Xsup = my_conf.physical_region#[0,2069,1172,2145]
    else:
        Yinf, Ysup, Xinf, Xsup = Lims_from_region(region)#[131,1973,2212,2562]
        #image_area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)

    fitsfile = fits.open(filename)[0]
    data = fitsfile.data[Xinf: Xsup, Yinf: Ysup]
    #value, bins = np.histogram(data,bins=500,range=(data.min(),np.percentile(data,99.999)))
#    value, bins = np.histogram(data,bins=500,range=(np.nanmin(data),np.percentile(data,99.995)))#np.nanmax(data)))#,np.percentile(data,99.99)))
    value, bins = np.histogram(data,bins=500,range=(2500,10000))#np.nanmax(data)))#,np.percentile(data,99.99)))
    bins_c = (bins[1:] + bins[:-1])/2

    for filename in path[1:]: 
        print(filename)
        fitsfile = fits.open(filename)[0]
        data = fitsfile.data[Xinf: Xsup, Yinf: Ysup]
        value1, bins = np.histogram(data,bins=bins, range=(np.nanmin(bins), np.nanmax(bins)))
        value += value1
    plt.figure()#figsize=(10,6))
    #plt.step(bins_c,np.log10(value))
    plt.step(bins_c,np.log10(value))
    plt.grid(True, which="both", linestyle='dotted')
    plt.xlabel('ADU value')
    plt.title("Histogram's sum")
    plt.ylabel('Log(#)')
    plt.grid()
    plt.show()
    csvwrite(Table(np.vstack((bins_c,value)).T), DS9backUp + 'CSVs/%s_HistogramSum.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")))#piki
    return

def AddHeaderField(xpapoint, field='', value='', comment='-'):
    """
    """
    from astropy.io import fits
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    
  
    if field == '':
        field = sys.argv[3]
    if value == '':
        value = sys.argv[4]
        try:
            value = float(value)
        except:
            pass
    try:
        comment = sys.argv[5]
    except IndexError:
        pass

   # if len(sys.argv) > 6: path = Charge_path_new(filename, entry_point=6); else: path = [filename]
    path = Charge_path_new(filename) if len(sys.argv) > 6 else [filename]
    
    for filename in path: 
        print(filename)
        header = fits.getheader(filename)
        if 'NAXIS3' in header:
            print('2D array: Removing NAXIS3 from header...')
            fits.delval(filename,'NAXIS3')
        fits.setval(filename, field[:8], value = value, comment = comment)

#        fitsimage = fits.open(filename)[0]
#        fitsimage.header[field] =  value
#        fitswrite(fitsimage, filename)
    if len(path)<2:
        d.set('frame clear')
        d.set('file '+ filename)
    return



    

def DS9BackgroundEstimationPhot(xpapoint, n=2, DS9backUp = DS9_BackUp_path, Plot=True):
    """
    """
    #from scipy import ndimage
#    import matplotlib.pyplot as plt   
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    sigma, bckd, rms, filters, boxs, percentile, mask, snr, npixels, dilate =  sys.argv[3:3+5+5]
    filter1, filter2 = np.array(filters.split(','), dtype=int)
    mask = bool(mask)
    sigma, percentile, snr, npixels, dilate = np.array([sigma, percentile, snr, npixels, dilate], dtype=int)
    box1, box2 = np.array(boxs.split(','), dtype=int)
    if len(sys.argv) > 3+5: path = Charge_path_new(filename, entry_point=3+5+5)
    if len(path)>1:
        Plot=False

    for filename in path:
        fitsfile, name = BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path,
                                                  sigma=float(sigma), bckd=bckd, rms=rms, filters=(filter1, filter2), boxs=(box1, box2),
                                                  exclude_percentile=percentile, mask=mask, snr=snr, npixels=npixels, dilate_size=dilate, Plot=Plot)

    if len(path)<2:
        d.set("lock frame physical")
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name) 
        DS9setup2(xpapoint)
    return fitsfile, name

#Background2D(mask, (50,50), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=BiweightLocationBackground(),exclude_percentile=20)


def BackgroundEstimationPhot(filename,  sigma, bckd, rms, filters, boxs,n=2, DS9backUp = DS9_BackUp_path,snr = 3,npixels = 15,dilate_size = 3,exclude_percentile = 5, mask=False, Plot=True):
    """
    """
    from astropy.io import fits
    from photutils import make_source_mask, Background2D, MeanBackground, MedianBackground
    from photutils import ModeEstimatorBackground, MMMBackground, SExtractorBackground, BiweightLocationBackground
    from photutils import StdBackgroundRMS, MADStdBackgroundRMS, BiweightScaleBackgroundRMS
    from astropy.stats import SigmaClip#, sigma_clipped_stats
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data#[400:1700,1200:2000]
    #data2 = data#ndimage.grey_dilation(ndimage.grey_erosion(data, size=(n,n)), size=(n,n))   
    masks = generatemask(data)
    functions = {'MeanBackground':MeanBackground,'MedianBackground':MedianBackground,'ModeEstimatorBackground':ModeEstimatorBackground,
                     'MMMBackground':MMMBackground, 'SExtractorBackground':SExtractorBackground, 'BiweightLocationBackground':BiweightLocationBackground}
    functions_rms = {'StdBackgroundRMS':StdBackgroundRMS,'MADStdBackgroundRMS':MADStdBackgroundRMS,'BiweightScaleBackgroundRMS':BiweightScaleBackgroundRMS}
    bkg=[]
    for i, mask_data in enumerate(masks):
        if mask:
            mask_source = make_source_mask(mask_data, snr=snr, npixels=npixels, dilate_size=dilate_size)
        else:
            mask_source = np.ones(mask_data.shape,dtype=bool)
        #mean, median, std = sigma_clipped_stats(mask, sigma=sigma, mask=mask_data)#sigma_clip = SigmaClip(sigma=sigma)
        bkg_estimator = functions[bckd]()
        bkgrms_estimator = functions_rms[rms]()
        if len(masks)>1:
            bkg.append(Background2D(mask_data, boxs, filter_size=filters,sigma_clip=SigmaClip(sigma=sigma), 
                               bkg_estimator=bkg_estimator,exclude_percentile=exclude_percentile,
                               bkgrms_estimator=bkgrms_estimator))#,mask=mask_source)
        else:
            bkg.append(Background2D(mask_data, boxs, filter_size=filters,sigma_clip=SigmaClip(sigma=sigma), 
                               bkg_estimator=bkg_estimator,exclude_percentile=exclude_percentile,
                               bkgrms_estimator=bkgrms_estimator,mask=mask_source))
        print('Mask %i, median = %0.2f'%(i,bkg[-1].background_median))  
        print('Mask %i, rms = %0.2f'%(i,bkg[-1].background_rms_median))
#        fitsfile.data[np.isfinite(mask)] = fitsfile.data[np.isfinite(mask)] - bkg[-1].background[np.isfinite(mask)]#.astype('uint16')
#        fitsfile.data = fitsfile.data - bkg[-1].background
        if i==0:
            fitsfile.data = fitsfile.data - bkg[-1].background#.astype('uint16')
        else:
            fitsfile.data[np.isfinite(mask)] = fitsfile.data[np.isfinite(mask)] - bkg[-1].background[np.isfinite(mask)]#.astype('uint16')
#        fitsfile.data = fitsfile.data - bkg[-1].background
        if len(masks)==2:
            masks[-1][np.isfinite(masks[-1])] = fitsfile.data[np.isfinite(masks[-1])]
    if len(masks)==2:
        diff = np.nanmean(fitsfile.data[np.isfinite(masks[1])]) - np.nanmean(fitsfile.data[np.isfinite(masks[0])])
        fitsfile.data[np.isfinite(masks[1])] -= diff
        diff = np.nanmean(np.hstack((fitsfile.data[np.isfinite(masks[0])],fitsfile.data[np.isfinite(masks[1])])))
        fitsfile.data[np.isfinite(masks[1])] -= diff
    else:
        diff = np.nanmean(fitsfile.data[np.isfinite(masks[0])])
    fitsfile.data[np.isfinite(masks[0])] -= diff
    if np.isfinite(data).all():
        fitsfile.data = fitsfile.data.astype('uint16')
        
    name = os.path.join(os.path.dirname(filename) + '/bkgd_photutils_substracted/%s'%(os.path.basename(filename)))
    fitswrite(fitsfile, name)
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt    
    if data.shape == (2069, 3216):
        a, b = my_conf.physical_region[:2]#1073, 2033
    else:
        a, b = 0, -1
        
#    fig = plt.figure()
#    plt.subplot(131)
#    plt.title('Data')
#    plt.imshow(data[0:-1,a:b], vmin=np.percentile(data[0:-1,a:b],50), vmax=np.percentile(data[0:-1,a:b],99));plt.colorbar()
#    plt.subplot(132)
#    im = plt.imshow(data[0:-1,a:b] - fitsfile.data[0:-1,a:b], vmin=np.percentile(data[0:-1,a:b] - fitsfile.data[0:-1,a:b],50), vmax=np.percentile(data[0:-1,a:b] - fitsfile.data[0:-1,a:b],99));plt.colorbar()
#    plt.title('Extracted background')
#    plt.subplot(133)
#    im = plt.imshow(fitsfile.data[0:-1,a:b], vmin=np.percentile(fitsfile.data[0:-1,a:b],50), vmax=np.percentile(fitsfile.data[0:-1,a:b],99))#;plt.colorbar()
#    fig.colorbar(im)
#    plt.title('Data - Background')
#    plt.savefig(DS9backUp + 'Plots/%s_BackgroundSubtraction.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M:%S")))    #plt.subplot_tool()
    if len(masks)==1:
        fig = plt.figure()
        plt.suptitle('bckd=%s, sigma=%s, snr=%s, npixels=%s, dilate_size=%s, exclude_percentile=%s  box=%s-  std = %0.8f'%(bckd, sigma, snr, npixels, dilate_size, exclude_percentile,boxs,np.nanstd(fitsfile.data) ),y=1)
        plt.subplot(221)
        plt.title('Data and background')
        plt.plot(np.nanmean(data[0:-1,a:b],axis=0))
        plt.plot(np.nanmean(bkg.background[0:-1,a:b]+diff,axis=0))
        plt.subplot(222)
        im = plt.plot(np.nanmean(fitsfile.data[0:-1,a:b],axis=0))
        plt.title('Residual: Data = background')
        plt.subplot(223)
        plt.title('Data and background')
        plt.plot(np.nanmean(data[0:-1,a:b],axis=1))
        plt.plot(np.nanmean(bkg.background[0:-1,a:b]+diff,axis=1))
        plt.subplot(224)
        plt.plot(np.nanmean(fitsfile.data[0:-1,a:b],axis=1))
        plt.title('Residual: Data = background')
        fig.tight_layout()
        plt.savefig(DS9backUp + 'Plots/%s_BackgroundSubtraction.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M:%S")))    #plt.subplot_tool()
    if len(masks)==2:
        diff1 = np.nanmean(masks[0]-bkg[0].background)
        diff2 = np.nanmean(masks[1]-bkg[1].background)
        fig = plt.figure()
        plt.suptitle('bckd=%s, sigma=%s, snr=%s, npixels=%s, dilate_size=%s, exclude_percentile=%s  box=%s-  std = %0.8f'%(bckd, sigma, snr, npixels, dilate_size, exclude_percentile,boxs,np.nanstd(fitsfile.data) ),y=1)
        plt.subplot(221)
        plt.title('Data and background')
        plt.plot(np.nanmean(masks[0][0:-1,a:b],axis=0))
        plt.plot(np.nanmean(masks[1][0:-1,a:b],axis=0))
        plt.plot(np.nanmean(bkg[0].background[0:-1,a:b]+diff1,axis=0))
        plt.plot(np.nanmean(bkg[1].background[0:-1,a:b]+diff2,axis=0))
        plt.subplot(222)
        plt.plot(np.nanmean(fitsfile.data[0:-1,a:b],axis=0))
        plt.title('Residual: Data = background')
        plt.subplot(223)
        plt.title('Data and background')
        plt.plot(np.nanmean(data[0:-1,a:b],axis=1))
        plt.plot(np.nanmean(bkg[0].background[0:-1,a:b]+diff1,axis=1))
        plt.plot(np.nanmean(bkg[1].background[0:-1,a:b]+diff2,axis=1))
        plt.subplot(224)
        plt.plot(np.nanmean(fitsfile.data[0:-1,a:b],axis=1))
        plt.title('Residual: Data = background')
        fig.tight_layout()
        plt.savefig(DS9backUp + 'Plots/%s_BackgroundSubtraction.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M:%S")))    #plt.subplot_tool()
    if Plot:
        plt.show()
    return fitsfile, name


#snr = 3;npixels = 15;dilate_size = 3
#for exclude_percentile in [1,5,10,15,20,25,30]:
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd='SExtractorBackground', rms=rms, filters=(10, 10), boxs=(20, 20),snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)
#exclude_percentile=15
#for dilate_size in [1,5,10,15,20,25,30]:
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd='SExtractorBackground', rms=rms, filters=(10, 10), boxs=(20, 20),snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)
#exclude_percentile=15
#for snr in [1,5,10,15,20,25,30]:
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd='SExtractorBackground', rms=rms, filters=(10, 10), boxs=(20, 20),snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)
#snr=3
#for npixels in [1,5,10,15,20,25,30]:
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd='SExtractorBackground', rms=rms, filters=(10, 10), boxs=(20, 20),snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)
#for bckd in functions.keys():
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd=bckd, rms=rms, filters=(10, 10), boxs=(20, 20),snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)
#bckd = 'MeanBackground'
#for box in [(2,2),(6,6),(10,10),(20,20),(40,40),(70,70),(100,100)]:
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd=bckd, rms=rms, filters=(10, 10), boxs=box,snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)
#box=20
#for sigma in [5,10,15,20,25,30]:
#    BackgroundEstimationPhot(filename, n=2, DS9backUp = DS9_BackUp_path, sigma=float(sigma), bckd=bckd, rms=rms, filters=(10, 10), boxs=(20, 20),snr = snr,npixels = npixels,dilate_size = dilate_size,exclude_percentile = exclude_percentile)



def generatemask(data, config=my_conf):
    """
    """
    if data.shape == (2069, 3216):
        from PIL import Image, ImageDraw
        img1 = Image.new('L', data.shape[::-1], 0)#, Image.new('L2', data.shape, 0)
        img2 = Image.new('L', data.shape[::-1], 0)#, Image.new('L2', data.shape, 0)
        polygon1 = [1907,366,   2122,389,   2122,0,    1065,0,      1065,352]
        polygon2 = [1907,366,   2122,389,   2122,2000, 1065,2000,   1065,352]
        ImageDraw.Draw(img1).polygon(polygon1, outline=1, fill=1)
        ImageDraw.Draw(img2).polygon(polygon2, outline=1, fill=1)
        m1 = np.array(img1)
        m2 = np.array(img2)
        mask1, mask2 = data.copy().astype(float), data.copy().astype(float)
        mask1[m1==0] = np.nan
        mask2[m2==0] = np.nan
        return mask1, mask2
    else:
        print('Data shape not understood, this function works only for FB detector images')
        return [data]


def generatemask1(data):
    """
    """
    if data.shape == (2069, 3216):
        general_mask = data.copy()
        general_mask[2000:, :] = np.nan
        general_mask[:,:1065] = np.nan
        general_mask[:,2122:] = np.nan
        mask1, mask2 = general_mask.copy(), general_mask.copy()
        mask1[:362,:] = np.nan
        mask2[362:,:] = np.nan
        return mask1, mask2
    else:
        print('Data shape not understood, this function works only for FB detector images')
        return data

def GroupingAlgorithm(xpapoint, n=2, fwhm=5):
    """
    """
    from astropy.io import fits
    #from astropy.stats import gaussian_sigma_to_fwhm
    #from photutils.psf.groupstars import DAOGroup
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data
    #from scipy import ndimage
#    data2 = ndimage.grey_dilation(ndimage.grey_erosion(data, size=(n,n)), size=(n,n)) 
#    daogroup = DAOGroup(crit_separation=2.5*fwhm)
#    fwhm = sigma_psf * gaussian_sigma_to_fwhm
    return data

def AperturePhotometry(xpapoint):
    """
    """
    from astropy.table import Table
    #from .focustest import create_DS9regions2
    #from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from photutils import aperture_photometry
    from photutils import CircularAperture, CircularAnnulus
    from astropy.table import hstack
    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    fitsfile = d.get_fits()[0]#fits.open(filename)[0]
    data = fitsfile.data
    try:
        source_catalog_name = sys.argv[3]
    except IndexError:
        source_catalog_name = filename[:-5] + '.csv'
    source_catalog = Table.read(source_catalog_name)
    try:        
        positions = [(x,y) for x,y in source_catalog['xcentroid', 'ycentroid']]
    except ValueError:
        positions = [(x,y) for x,y in source_catalog['xcenter', 'ycenter']]
    apertures = CircularAperture(positions, r=5)
    annulus_apertures = CircularAnnulus(positions, r_in=10, r_out=15)
    annulus_masks = annulus_apertures.to_mask(method='center')
    
    bkg_median = []
    for mask in annulus_masks:
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
    bkg_median = np.array(bkg_median)
    phot = aperture_photometry(data, apertures, error = 0.1 * data)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * apertures.area()
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    for col in phot.colnames:
        phot[col].info.format = '%.8g'  # for consistent table output
    print(phot) 
    phot = Table(phot)
    create_DS9regions2(phot['xcenter'],phot['ycenter'], radius=10, form = 'circle',save=True,color = 'yellow', savename='/tmp/centers1')
    create_DS9regions2(phot['xcenter'],phot['ycenter'], radius=15, form = 'circle',save=True,color = 'red', savename='/tmp/centers2')
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers1.reg'))                                        
    d.set('region {}'.format('/tmp/centers2.reg'))
    new_cat =  hstack((source_catalog,phot))    
    new_cat.remove_columns(['xcentroid','ycentroid','id_2'])                              
    csvwrite(new_cat, filename[:-5] + '.csv')#new_cat.write(filename[:-5] + '.csv', overwrite=True)
    return phot

def BuildingEPSF(xpapoint):
    """
    """
    from astropy.table import Table
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    from astropy.nddata import NDData
    from photutils.psf import extract_stars
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    try:
        source_catalog_name = sys.argv[3]
    except IndexError:
        source_catalog_name = filename[:-5] + '.csv'
    source_catalog = Table.read(source_catalog_name)
    stars_tbl = Table()
    try:        
        stars_tbl['x'],stars_tbl['y'] = source_catalog['xcentroid'], source_catalog['ycentroid']
    except ValueError:
        stars_tbl['x'],stars_tbl['y'] = source_catalog['xcenter'], source_catalog['ycenter']
    fitsfile = fits.open(filename)[0]
    data = fitsfile.data
    mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=2.)
    data -= median_val    
    nddata = NDData(data=data)
    stars = extract_stars(nddata, stars_tbl, size=25)   
    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols,squeeze=True)
    ax = ax.ravel()
    for i in range(nrows*ncols):
        norm = simple_norm(stars[i], 'log', percent=99.)
        ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
    from photutils import EPSFBuilder
    epsf_builder = EPSFBuilder(oversampling=4, maxiters=3,
                               progress_bar=False)
    epsf, fitted_stars = epsf_builder(stars)
    norm = simple_norm(epsf.data, 'log', percent=99.)
    plt.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
    plt.colorbar()
    return
   
def MorphologicalProperties(xpapoint):
    """
    """
    from astropy.table import Table
    #from astropy.io import fits
    from astropy.convolution import Gaussian2DKernel
    from photutils import Background2D, MedianBackground
    from photutils import detect_sources
    from astropy.stats import gaussian_fwhm_to_sigma
    from photutils import deblend_sources
    import matplotlib.pyplot as plt
    from photutils import source_properties, EllipticalAperture
    #from .focustest import create_DS9regions
    #from astropy.table import hstack

    d = DS9(xpapoint)
    filename = getfilename(d)#filename = d.get('file')
    fitsfile = d.get_fits()[0]#fits.open(filename)[0]
    data = fitsfile.data

    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (50, 50), filter_size=(3, 3),
                       bkg_estimator=bkg_estimator)
    threshold = bkg.background + (2. * bkg.background_rms)
    sigma = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    kernel.normalize()
    npixels = 5
    segm = detect_sources(data, threshold, npixels=npixels,
                          filter_kernel=kernel)
    segm_deblend = deblend_sources(data, segm, npixels=npixels,
                                   filter_kernel=kernel, nlevels=32,
                                   contrast=0.001)
    cat = source_properties(data, segm_deblend)
    tbl = cat.to_table()
    tbl['xcentroid'].info.format = '.2f'  # optional format
    tbl['ycentroid'].info.format = '.2f'
    tbl['cxx'].info.format = '.2f'
    tbl['cxy'].info.format = '.2f'
    tbl['cyy'].info.format = '.2f'
    
    r = 3.    # approximate isophotal extent
    apertures = []
    for obj in cat:
        position = (obj.xcentroid.value, obj.ycentroid.value)
        a = obj.semimajor_axis_sigma.value * r
        b = obj.semiminor_axis_sigma.value * r
        theta = obj.orientation.value
        apertures.append(EllipticalAperture(position, a, b, theta=theta))
    tbl = Table(tbl)
    print(tbl)
    ly, lx = data.shape
    fig, ax1 = plt.subplots(1, 1, figsize=(7*float(lx)/ly-1, 7))
    ax1.imshow(segm_deblend, origin='lower',
               cmap=segm_deblend.cmap(random_state=12345))
    ax1.set_title('Segmentation Image')
    for aperture in apertures:
        aperture.plot(color='white', lw=0.7, ax=ax1)
    plt.show()
    create_DS9regions([list(tbl['xcentroid'])],[list(tbl['ycentroid'])], more=[3*tbl['semimajor_axis_sigma'],3*tbl['semiminor_axis_sigma'],180*tbl['orientation']/np.pi], form = ['ellipse']*len(tbl),save=True,color = ['red']*len(tbl), savename='/tmp/centers',ID=[list(tbl['id'])])
#    create_DS9regions([list(tbl['xcentroid'])],[list(tbl['ycentroid'])], form=['ellipse'],  more=[2.35*tbl['semimajor_axis_sigma'],2.35*tbl['semiminor_axis_sigma']], radius=10, save=True, savename='/tmp/cr', color = ['yellow'],ID=None)
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers.reg'))                                        
    tbl.write(filename[:-5] + '.csv', overwrite=True)
    return segm_deblend

def EllipticalIsophoteAnalysis(xpapoint):
    """
    """
    from photutils.isophote import EllipseGeometry
    import matplotlib.pyplot as plt
    from photutils.isophote import Ellipse
    #from astropy.io import fits
#    from photutils.datasets import make_noise_image
#    from astropy.modeling.models import Gaussian2D
#    g = Gaussian2D(100., 75, 75, 20, 12, theta=40.*np.pi/180.)
#    ny = nx = 150
#    y, x = np.mgrid[0:ny, 0:nx]
#    noise = make_noise_image((ny, nx), type='gaussian', mean=0.,
#                             stddev=2., random_state=12345)
#    data = g(x, y) + noise
    d = DS9(xpapoint)
    #filename = getfilename(d)#filename = d.get('file')

    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    data = d.get_fits()[0].data[area[0]:area[1],area[2]:area[3]]#fits.open(filename)[0].data[area[0]:area[1],area[2]:area[3]]#picouet  
    geometry = EllipseGeometry(x0=75, y0=75, sma=20, eps=0.5,
                           pa=20.*np.pi/180.)
#    plt.imshow(data);plt.show()
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image()
    print(isolist.pa)    
    print(isolist.to_table())  
    plt.figure()#figsize=(8, 6))
    #plt.subplots_adjust(hspace=0.35, wspace=0.35)
    plt.subplot(2, 2, 1)
    plt.errorbar(isolist.sma, isolist.eps, yerr=isolist.ellip_err,
                 fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('Ellipticity')
    plt.subplot(2, 2, 2)
    plt.errorbar(isolist.sma, isolist.pa/np.pi*180.,
                 yerr=isolist.pa_err/np.pi* 80., fmt='o', markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('PA (deg)')
    plt.subplot(2, 2, 3)
    plt.errorbar(isolist.sma, isolist.x0, yerr=isolist.x0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('x0')
    plt.subplot(2, 2, 4)
    plt.errorbar(isolist.sma, isolist.y0, yerr=isolist.y0_err, fmt='o',
                 markersize=4)
    plt.xlabel('Semimajor Axis Length (pix)')
    plt.ylabel('y0')
    plt.tight_layout()
    plt.show()
    table = isolist.to_table()
    table.write('/tmp/isolist_fit.csv',overwrite=True)
    return isolist


def Centroiding(xpapoint):
    """
    """
    #from astropy.io import fits
    from photutils import centroid_com, centroid_1dg, centroid_2dg

    d = DS9(xpapoint)
    #filename = getfilename(d)#filename = d.get('file')

    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    data = d.get_fits()[0][area[0]:area[1],area[2]:area[3]] #fits.open(filename)[0].data[area[0]:area[1],area[2]:area[3]]#picouet 
    x1, y1 = centroid_com(data)
    print((x1, y1))    
    x2, y2 = centroid_1dg(data)
    print((x2, y2))    
    x3, y3 = centroid_2dg(data)
    print((x3, y3))
    create_DS9regions([[x1+Xinf,x2+Xinf,x3+Xinf]], [[y1+Yinf,y2+Yinf,y3+Yinf]], radius=2, form = ['circle']*3,save=True, ID=[['COM:%0.2f - %0.2f'%(x1,y1),'2x1D Gaussian:%0.2f - %0.2f'%(x2,y2),'2D Gaussian:%0.2f - %0.2f'%(x3,y3)]], color = ['yellow', 'red', 'green'], savename='/tmp/centers1')
    d.set('region delete all')
    d.set('region {}'.format('/tmp/centers1.reg'))   
   
    
#    import matplotlib.pyplot as plt
#    fig, ax = plt.subplots(1, 1)
#    ax.imshow(data, origin='lower', interpolation='nearest', cmap='viridis')
#    marker = '+'
#    ms, mew = 30, 2.
#    plt.plot(x1, y1, color='#1f77b4', marker=marker, ms=ms, mew=mew)
#    plt.plot(x2, y2, color='#17becf', marker=marker, ms=ms, mew=mew)
#    plt.plot(x3, y3, color='#d62728', marker=marker, ms=ms, mew=mew)   
#    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
#    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
#    ax2 = zoomed_inset_axes(ax, zoom=6, loc=9)
#    ax2.imshow(data, interpolation='nearest', origin='lower',
#               cmap='viridis', vmin=190, vmax=220)
#    ax2.plot(x1, y1, color='#1f77b4', marker=marker, ms=ms, mew=mew)
#    ax2.plot(x2, y2, color='#17becf', marker=marker, ms=ms, mew=mew)
#    ax2.plot(x3, y3, color='#d62728', marker=marker, ms=ms, mew=mew)
#    ax2.set_xlim(13, 15)
#    ax2.set_ylim(16, 18)
#    mark_inset(ax, ax2, loc1=3, loc2=4, fc='none', ec='0.5')
#    ax2.axes.get_xaxis().set_visible(False)
#    ax2.axes.get_yaxis().set_visible(False)
#    ax.set_xlim(0, data.shape[1]-1)
#    ax.set_ylim(0, data.shape[0]-1)
#    plt.show()
    return

def SimulateFIREBallemCCD(xpapoint, DS9backUp = DS9_BackUp_path):
    from astropy.io import fits
    d = DS9(xpapoint)
    lx, ly = np.array(sys.argv[3].split(','), dtype=int)
    if sys.argv[4] == '1':
        try:
            if len(sys.argv[5].split(','))==1:
                OS1, OS2 = int(sys.argv[5]), -1
            else:
                OS1, OS2 = np.array(sys.argv[5].split(','), dtype=int)
        except ValueError:
            OS1, OS2 = 0, -1
    else:
        OS1, OS2 = 0, -1
    ConversionGain, EmGain, Bias, RN, CIC, Dark, Smearing, exposure, flux, source, Rx, Ry, name, spectra, cube = sys.argv[6:]
    name = DS9backUp + 'CreatedImages/' + name
    image = SimulateFIREBallemCCDImage(ConversionGain=float(ConversionGain), EmGain=float(EmGain), Bias=Bias, RN=float(RN), CIC=float(CIC), Dark=float(Dark), 
                                       Smearing=float(Smearing), exposure=float(exposure), flux=float(flux), source=source, Rx=float(Rx)/2.35, 
                                       Ry=float(Ry)/2.35, size=(lx,ly),spectra=spectra, cube=cube, 
                                       OSregions=(OS1,OS2), name=name)
    fitsimage = fits.HDUList([fits.PrimaryHDU(image)])[0]
    fitsimage.header['CONVGAIN'] = (float(ConversionGain), 'Conversion Gain')
    fitsimage.header[my_conf.gain[0]] = (float(EmGain), 'Amplification gain')
    fitsimage.header['BIAS'] = (Bias, 'Detector bias')
    fitsimage.header['READNOIS'] = (float(RN), 'Read noise in ADU')
    fitsimage.header['CIC'] = (float(CIC), 'Charge induced current')
    fitsimage.header['DARK'] = (float(Dark), 'Dark current')
    fitsimage.header['SMEARING'] = (float(Smearing), 'Smeariong length')
    fitsimage.header[my_conf.exptime[0]] = (float(exposure), 'Exposure time in second')
    fitsimage.header['FLIUX'] = (float(flux), "Flux un e'-/pix/sec")
    fitsimage.header['SOURCE'] = (source, 'Type of source')
    fitsimage.header['Rx'] = (float(Rx), 'Spatial resolution in pixel FWHM')
    fitsimage.header['Ry'] = (float(Ry), 'Spectral resolution in pixel FWHM')
    fitsimage.header['OS-1'] = (float(OS1), 'Over scann left position')
    fitsimage.header['OS-2'] = (float(OS2), 'Over scann right position')
    fitsimage.header['SPECTRA'] = (spectra, 'path of the spectra in entry')
    fitsimage.header['MOCKCUBE'] = (cube, 'path of the spatio-spectral mock cube in entry')
    fitswrite(fitsimage, name)
    d.set('file '+ name)
    print(lx,ly)
    print(' ConversionGain, EmGain, Bias, RN, CIC, Dark, Smearing =', ConversionGain, EmGain, Bias, RN, CIC, Dark, Smearing)
    
    value, bins = np.histogram(image[:,OS1:OS2],bins=500,range=(np.percentile(image[:,OS1:OS2],1e-3),np.percentile(image[:,OS1:OS2],100-1e-3)))
    bins_c = (bins[1:] + bins[:-1])/2
    plt.figure()
    plt.step(bins_c,np.log10(value))
    plt.fill_between(bins_c,np.log10(value),alpha=0.2, step='pre')
    plt.grid(True, which="both", linestyle='dotted')
    plt.xlabel('ADU value')
    plt.title("Histogram")
    plt.ylabel('Log(#)')
    plt.grid(True)
    plt.show()
   
    return

def ConvolveSlit2D_PSF(xy, amp=1, l=3, L=9, xo=0 ,yo=0 , sigmax2 = 40, sigmay2 = 40):
    from scipy import special
    x, y = xy
    A1 = special.erf((l-(x-xo))/np.sqrt(2*sigmax2))
    A2 = special.erf((l+(x-xo))/np.sqrt(2*sigmax2)) 
    B1 = special.erf((L-(y-yo))/np.sqrt(2*sigmay2))
    B2 = special.erf((L+(y-yo))/np.sqrt(2*sigmay2)) 
    function = amp * (1/(16*l*L)) * (A1+A2) * (B1 + B2)
    return function.ravel()

def addAtPos(M1, M2, center):
    size_x, size_y = np.shape(M2)
    coor_x, coor_y = center
    end_x, end_y   = (coor_x + size_x), (coor_y + size_y)
    M1[coor_x:end_x, coor_y:end_y] = M1[coor_x:end_x, coor_y:end_y] + M2
    return M1


   
def createHole(radius=40,size=(200,200)):
    """
    Create a disk of the size specified in arcsec to then be convolved to the PSF :
    convolution.Gaussian2DKernel
    """
    Hole = np.zeros((size[0],size[1]))
    y, x = np.indices((Hole.shape))    
    r = np.sqrt((x - size[0]/2)**2 + (y - size[1]/2)**2)#    r = np.around(r)-1
    r = r.astype(np.int)
    Hole[r<radius]=1
    return Hole


def convolvePSF(radius_hole = 20, fwhmsPSF =  [5,6], unit = 10, size=(201,201), Plot = False):
    """
    Convolve a disk from createHole with a gaussian kernel with the specified size in arcsec
    """
    from astropy import convolution
    from astropy.convolution import Gaussian2DKernel
    PSF = Gaussian2DKernel(x_stddev=fwhmsPSF[0]/2.35,y_stddev=fwhmsPSF[1]/2.35,x_size=size[0],y_size=size[1]).array 
    PSF /= PSF.max() 
    Hole = createHole(radius=radius_hole,size=size)
    print('Hole created')
    conv = convolution.convolve(PSF, Hole) 
    print('Comvolution done')
    if Plot:
        plt.plot(conv[:,int(size/2)] / conv[:,int(size/2)].max(),'-',label= "PSF:" + np.str(fwhmsPSF) + "dec''")
        plt.plot(Hole[:,int(size/2)] / Hole[:,int(size/2)].max(),label="Hole:{}dec''".format(2*radius_hole))
        plt.legend(loc='upper right')
    return conv

        


def SimulateFIREBallemCCDImage(ConversionGain=0.53, EmGain=1500, Bias='Auto', RN=80, CIC=1, Dark=5e-4, Smearing=0.7, exposure=50, flux=1e-3, source='Spectra', Rx=8, Ry=8, size=[3216,2069], OSregions=[1066,2124], name='/tmp/image000001.fits', spectra=None, cube=None):
    from astropy.modeling.functional_models import Gaussian2D
    #from .focustest import convolvePSF, addAtPos
    OS1, OS2 = OSregions
    if Bias == 'Auto':
        if EmGain>1:
            Bias = 3000 / ConversionGain
        else:
            Bias = 6000 / ConversionGain
    image = np.zeros((size[1],size[0]),dtype = 'float32')

    #dark & flux
    source_im = 0 * image[:,OSregions[0]:OSregions[1]]
    lx, ly = source_im.shape
    y = np.linspace(0,lx-1,lx)
    x = np.linspace(0,ly-1,ly)
    x, y = np.meshgrid(x,y) 
    if os.path.isfile(cube):
        from .FitsCube import FitsCube
        fitsimage = FitsCube(filename=cube)

    elif source == 'Flat-field':
        source_im += flux
    elif source == 'Dirac':
        source_im += Gaussian2D.evaluate(x, y, 100*flux, ly/2, lx/2, Ry,  Rx, 0)
    elif source == 'Spectra':
        source_im += Gaussian2D.evaluate(x, y, 100*flux, ly/2, lx/2, 100 * Ry, Rx, 0)
    elif source == 'Slit':
        ConvolveSlit2D_PSF_75muWidth = lambda xy , amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
        source_im += ConvolveSlit2D_PSF_75muWidth((x,y),30000*flux,9,ly/2, lx/2,Rx,Ry).reshape(lx,ly)
    elif source == 'Fibre':
        print('Create fibre source')
        fibre= convolvePSF(radius_hole = 20, fwhmsPSF = [5,6], unit = 10, size=(201,201), Plot = False)#[:,OSregions[0]:OSregions[1]]
        source_im = addAtPos(source_im, fibre, (int(lx/2), int(ly/2)))
        print('Done')
    elif source[:5] == 'Field':
        ConvolveSlit2D_PSF_75muWidth = lambda xy , amp, L, xo, yo, sigmax2, sigmay2: ConvolveSlit2D_PSF(xy, amp, 2.5, L, xo, yo, sigmax2, sigmay2)
        ws = [2025,2062,2139]
        #x, y = [], []
        for i, w in enumerate(ws):    
            slits = returnXY(source[0].lower() + source[-1], w = w, frame='observedframe')
            xs = slits[0]
            ys = slits[1]
            index = (ys>OS1) & (ys<OS2) 
            print(xs,ys)
            for yi, xi in zip(np.array(ys[index])-OS1,xs[index]):
                print(xi,yi)
                source_im += ConvolveSlit2D_PSF_75muWidth((x,y),40000*flux,9,yi, xi,Rx,Ry).reshape(lx,ly)
                
    source_im2 = np.random.poisson((Dark + source_im) * exposure + CIC) 
    #enregistrer limage de poisson
    image[:,OSregions[0]:OSregions[1]] += source_im2
       
    if EmGain > 1:
        id_nnul = image != 0
        image[id_nnul] = np.random.gamma(image[id_nnul], EmGain)    
    if Smearing>0:
        exp = lambda x, a : np.exp(-x/a)
        smearingKernelx = exp(np.arange(5),Smearing)[::-1]#plt.plot(smearingKernel/np.sum(smearingKernel))
        smearingKernely = exp(np.arange(3),Smearing/2)[::-1]#plt.plot(smearingKernel/np.sum(smearingKernel))
        image = np.apply_along_axis(lambda m: np.convolve(m, smearingKernelx, mode='same'), axis=1, arr=image)
        image = np.apply_along_axis(lambda m: np.convolve(m, smearingKernely, mode='same'), axis=0, arr=image)
    fitswrite(image, name[:-5] + '_before_RN.fits')

    #np.convolve(image, smearingKernel)  
    readout = np.random.normal(float(Bias), float(RN), (size[1],size[0]))
    #image = image + CIC 
    #Smearing

    #signal.convolve2d(image, smearingKernel)
    #read noise
    return np.array((image + readout) * ConversionGain , dtype='int16')

def FollowProbabilityLaw(xpapoint, law=''):
    d = DS9(xpapoint)
    lx, ly = 1000, 1000
    if law == '':
        law = sys.argv[3]    #law = 'standard_exponential'
    DictFunction = {'beta':[np.random.beta,1,1],'binomial':[np.random.binomial,10, 0.5],'geometric':[np.random.geometric, 0.5],'pareto':[np.random.pareto, 1],
                     'poisson':[np.random.poisson, 1],'power':[np.random.power, 1],'rand':[np.random.rand],'randint':[np.random.randint, lx],'standard_exponential':[np.random.standard_exponential],
                     'standard_gamma':[np.random.standard_gamma, 1],'standard_normal':[np.random.standard_normal],'standard_t':[np.random.standard_t, 1]}
    if law not in DictFunction.keys():
        print(law + ' not recognize, Please choose a law in the following possibilities: \n','\n'.join(DictFunction.keys()))
    else:
        try:
            image = DictFunction[law][0](*DictFunction[law][1:],size=(lx,ly))
        except TypeError:
            image = DictFunction[law][0](*(DictFunction[law][1:]+[lx,ly]))
          #  image = DictFunction[law][0](*DictFunction[law][1:],lx,ly)
        fitswrite(image,'/tmp/test.fits')
        d.set('frame new')       
        d.set('file ' + '/tmp/test.fits')       
#        d.set_np2arr(image)
    return

def ApplyRealisation(xpapoint, law=''):
    d = DS9(xpapoint)
    data = d.get_arr2np()
    if law == '':
        law = sys.argv[3]    #law = 'standard_exponential'    law = sys.argv[3]    #law = 'standard_exponential'
    #lx=1
    DictFunction = {'gamma':[np.random.gamma],'beta':[np.random.beta,1],'geometric':[np.random.geometric],'poisson':[np.random.poisson],'pareto':[np.random.pareto],
                     'power':[np.random.power]}#,'binomial':[np.random.binomial,10, 0.5]
                    # 'standard_gamma':[np.random.standard_gamma, 1],'standard_normal':[np.random.standard_normal],'standard_t':[np.random.standard_t, 1],'rand':[np.random.rand],'standard_exponential':[np.random.standard_exponential],'randint':[np.random.randint]}
    if law not in DictFunction.keys():
        print(law + ' not recognize, Please choose a law in the following possibilities: \n','\n'.join(DictFunction.keys()))
    else:
#        image = DictFunction[law][0](data, *DictFunction[law][1:])
       
        image = DictFunction[law][0](data, *DictFunction[law][1:])
        d.set('frame new')       
        d.set_np2arr(image)
    return


#        image = np.random.standard_exponential(size=(lx,ly))
#        image =  np.random.beta(1, 1, size=(lx,ly))
#        image =  np.random.binomial(10, 0.5, size=(lx,ly))
#        image =  np.random.geometric(0.5, size=(lx,ly))
#        image =  np.random.pareto(1, size=(lx,ly))
#        image =  np.random.poisson(1, size=(lx,ly))
#        image =  np.random.power(1, size=(lx,ly))
#        image =  np.random.rand(lx,ly)
#        image =  np.random.randint(2, size=(lx,ly))
#        image =  np.random.standard_exponential(size=(lx,ly))
#        image =  np.random.standard_gamma(1, size=(lx,ly))
#        image =  np.random.standard_normal(size=(lx,ly))
#        image =  np.random.standard_t(1,size=(lx,ly))

def CreateImageFromCatalogObject(xpapoint, nb = int(1e3)):
    """
    """
    from astropy.table import Table
    from tqdm import tqdm
    from astropy.modeling.functional_models import Gaussian2D


    d = DS9(xpapoint)
    nb = nb
    lx, ly = 1000, 1000
    #if len(sys.argv)>3:
    if os.path.isfile( sys.argv[3]):
        catfile =   sys.argv[3] #law = 'standard_exponential'
        catalog = Table.read(catfile)
    else:
        x, y, angle = np.random.randint(lx, size=nb), np.random.randint(ly, size=nb), np.random.rand(nb)*2*np.pi - np.pi
        peak = np.random.exponential(10, size=nb)
        sigmax, sigmay = np.random.normal(3, 2, size=nb), np.random.normal(3, 2, size=nb)
        catalog = Table([x, y, peak, sigmax, sigmay, angle],  names=('x_mean', 'y_mean', 'amplitude','x_stddev','y_stddev', 'theta'))
        
    background = 1
    image = np.ones((lx,ly)) * background
    for i in tqdm(range(len(catalog))):
        x = np.linspace(0,lx-1,lx)
        y = np.linspace(0,ly-1,ly)
        x, y = np.meshgrid(x,y)
        try:
            image += Gaussian2D.evaluate(x, y, catalog[i]['peak'], catalog[i]['xcenter'], catalog[i]['ycenter'], catalog[i]['sigmax'],  catalog[i]['sigmay'], catalog[i]['angle'])
        except KeyError:
            image += Gaussian2D.evaluate(x, y, catalog[i]['amplitude'], catalog[i]['x_mean'], catalog[i]['y_mean'], catalog[i]['x_stddev'],  catalog[i]['y_stddev'], catalog[i]['theta'])
    image_real = np.random.poisson(image)
    fitswrite(image_real,'/tmp/galaxies000000.fits')
    d.set('file /tmp/galaxies000000.fits')
#    from photutils.datasets import make_gaussian_sources_image
#    nb = int(1e3)
#    lx, ly = 1000, 1000
#    x, y, angle = np.random.randint(lx, size=nb), np.random.randint(ly, size=nb), np.random.rand(nb)*2*np.pi - np.pi
#    peak = np.random.exponential(10, size=nb)
#    sigmax, sigmay = np.random.normal(3, 2, size=nb), np.random.normal(3, 2, size=nb)
#    catalog = Table([x, y, peak, sigmax, sigmay, angle], names=('x_mean', 'y_mean', 'amplitude','x_stddev','y_stddev', 'theta'))
#    image = make_gaussian_sources_image((lx,ly), catalog, oversample=1)    
    return image_real



def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, offset=0):
    """Defines a gaussian function in 2D 
    """
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    #A = amplitude/(2*np.pi*sigma_x*sigma_y)    #A to be there?
    g = offset + amplitude * np.exp( - 0.5*(((x-xo)/sigma_x)**2) - 0.5*(((y-yo)/sigma_y)**2))
    return g.ravel()

def galex2Ph_s_A(f200=2.9e-4, atm=0.37, throughput=0.13, QE=0.55, area=7854):
    """
    Convert galex fluxes into photons per seconds per angstrom
    """
    Ph_s_A = f200*(throughput*atm* QE*area)
    return Ph_s_A

def FB_ADU2Flux(ADU, EMgain=1000, ConversionGain=1.8, dispersion=46.6/10):#old emgain=453
    """
    Convert FB2 ADUs into photons per seconds per angstrom
    print((galex2Ph_s_A()-FB_ADU2Flux(5.6))/galex2Ph_s_A())
    """
    Flux = ADU * ConversionGain * dispersion / EMgain
    return Flux

#def(3600*np.arctan(0.010/500)*180/np.pi):
    


def Flux2FBADU(Flux, EMgain=453, ConversionGain=1.8, dispersion=46.6/10):
    """
    Convert FB2 ADUs into photons per seconds per angstrom
    """
    ADU = Flux * EMgain / ConversionGain / dispersion
    #Flux = ADU * ConversionGain * dispersion / EMgain / 10
    return ADU


def BackgroundMeasurement(xpapoint, config=my_conf):
    """
    """
    #from .focustest import create_DS9regions2
    from decimal import Decimal
    #x, y = x+93, y-93
    d = DS9(xpapoint)
    
    try:
        region = getregion(d)
    except ValueError:
        image_area = [1500,2000,1500,2000]
        Yinf, Ysup,Xinf, Xsup = image_area
    else:
        Yinf, Ysup, Xinf, Xsup = Lims_from_region(region)#[131,1973,2212,2562]
        image_area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)    
    
    if d.get('tile')=='yes':
        d.set('frame first')
        n1 = int(d.get('frame'))
        d.set('frame last')
        n2 = int(d.get('frame'))
        n=n2-n1+1
        print('Number of frame = ',n)
        d.set('frame first')
        for frame in range(n):
            data = d.get_fits()[0].data
            try:
                texp = float(d.get_fits()[0].header[my_conf.exptime[0]])
            except KeyError as e:
                print(e)
            else:
                xc = [int(2336),int((image_area[1]+image_area[0])/2)]
    #            yc = 1000
    #            w,l = 300,1900
                yc = int((image_area[2]+image_area[3])/2)#1000
                w,l = int(image_area[1]-image_area[0]),int(image_area[3]-image_area[2])
                create_DS9regions2([xc[0]],[yc], radius=[150,l], form = 'box',save=True,color = 'yellow', savename='/tmp/centers')
                create_DS9regions2([xc[1]],[yc], radius=[w,l], form = 'box',save=True,color = 'yellow', savename='/tmp/centers1')
                d.set('region delete all')  
                d.set('region {}'.format('/tmp/centers.reg'))
                d.set('region {}'.format('/tmp/centers1.reg'))
                reg = data[int(yc - l/2):int(yc + l/2),int(xc[1] - w/2):int(xc[1] + w/2)]
    #            regOS = data[int(yc - l/2):int(yc + l/2),int(xc[0] - w/2):int(xc[0] + w/2)]
                regOS = data[int(yc - l/2):int(yc + l/2),2200:2500]
    #            print(int(yc - l/2),int(yc + l/2),int(xc[1] - w/2),int(xc[1] + w/2))
    #            print(int(yc - l/2),int(yc + l/2),2200,2500)
    #            print(reg.shape)
    #            print(reg.shape)
    #            print(regOS)
                meanADU = np.nanmean(reg) - np.nanmean(regOS)
                stdADU = np.nanstd(reg)
                create_DS9regions2([xc[1]],[yc], radius=[w,l], form = '# text',save=True, text=['Flux=%0.3EADU/s/pix - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)],color = 'yellow', savename='/tmp/centers1')
                create_DS9regions2([xc[1]],[yc-100], radius=[w,l], form = '# text',save=True, text=['ADU mean=%0.2EADU/pix - EXP=%isec'%(Decimal(meanADU),texp)],color = 'yellow', savename='/tmp/centers2')
                create_DS9regions2([xc[1]],[yc-200], radius=[w,l], form = '# text',save=True, text=['ADU PHYS=%0.2EADU - ADU OS=%0.2EADU'%(Decimal(np.nanmean(reg)), Decimal(np.nanmean(regOS)))],color = 'yellow', savename='/tmp/centers3')
                d.set('region {}'.format('/tmp/centers1.reg')) 
                d.set('region {}'.format('/tmp/centers2.reg')) 
                d.set('region {}'.format('/tmp/centers3.reg')) 
            d.set('frame next')
    else:
        data = d.get_fits()[0].data
        try:
            texp = float(d.get_fits()[0].header[my_conf.exptime[0]])
        except KeyError as e:
            print(e)
        else:
            xc = [int(2336),int((image_area[1]+image_area[0])/2)]
#            yc = 1000
#            w,l = 300,1900
            yc = int((image_area[2]+image_area[3])/2)#1000
            w,l = int(image_area[1]-image_area[0]),int(image_area[3]-image_area[2])
            create_DS9regions2([xc[0]],[yc], radius=[150,l], form = 'box',save=True,color = 'yellow', savename='/tmp/centers')
            create_DS9regions2([xc[1]],[yc], radius=[w,l], form = 'box',save=True,color = 'yellow', savename='/tmp/centers1')
            d.set('region delete all')  
            d.set('region {}'.format('/tmp/centers.reg'))
            d.set('region {}'.format('/tmp/centers1.reg'))
            reg = data[int(yc - l/2):int(yc + l/2),int(xc[1] - w/2):int(xc[1] + w/2)]
#            regOS = data[int(yc - l/2):int(yc + l/2),int(xc[0] - w/2):int(xc[0] + w/2)]
            regOS = data[int(yc - l/2):int(yc + l/2),2200:2500]
#            print(int(yc - l/2),int(yc + l/2),int(xc[1] - w/2),int(xc[1] + w/2))
#            print(int(yc - l/2),int(yc + l/2),2200,2500)
#            print(reg.shape)
#            print(reg.shape)
#            print(regOS)
            meanADU = np.nanmean(reg) - np.nanmean(regOS)
            stdADU = np.nanstd(reg)
            create_DS9regions2([xc[1]],[yc], radius=[w,l], form = '# text',save=True, text=['Flux=%0.3EADU/s/pix - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)],color = 'yellow', savename='/tmp/centers1')
            create_DS9regions2([xc[1]],[yc-100], radius=[w,l], form = '# text',save=True, text=['ADU mean=%0.2EADU/pix - EXP=%isec'%(Decimal(meanADU),texp)],color = 'yellow', savename='/tmp/centers2')
            create_DS9regions2([xc[1]],[yc-200], radius=[w,l], form = '# text',save=True, text=['ADU PHYS=%0.2EADU - ADU OS=%0.2EADU'%(Decimal(np.nanmean(reg)), Decimal(np.nanmean(regOS)))],color = 'yellow', savename='/tmp/centers3')
            d.set('region {}'.format('/tmp/centers1.reg')) 
            d.set('region {}'.format('/tmp/centers2.reg')) 
            d.set('region {}'.format('/tmp/centers3.reg')) 
            
    return


def SourcePhotometry(xpapoint, config=my_conf):
    """vtest
    """
    #from .focustest import create_DS9regions
    from decimal import Decimal
    #x, y = x+93, y-93
    d = DS9(xpapoint)
    data = d.get_fits()[0].data
    texp = d.get_fits()[0].header[my_conf.exptime[0]]
    r = getregion(d, all=False)
    y, x = np.indices((data.shape))
    radius = np.sqrt((x - r.xc)**2 + (y - r.yc)**2)
    physdet = (x>1050) & (x<2050) & (y<2000)

    if hasattr(r, 'w'):
        smallreg = (x>r.xc - r.w/2) & (x<r.xc + r.w/2) & (y>r.yc - r.h/2) & (y<r.yc + r.h/2)
        background = (x>r.xc - r.w) & (x<r.xc + r.w) & (y>r.yc - r.h) & (y<r.yc + r.h)
    if hasattr(r, 'r'):
        smallreg = radius<r.r
        background = radius<2*r.r
    meanADU = np.nansum(data[smallreg & physdet] - np.nanmean(data[~ smallreg & physdet & background]) ) 
    stdADU = np.nanstd(data[~ smallreg & physdet & background])
    if hasattr(r, 'r'):
        create_DS9regions([r.xc], [r.yc], radius=int(2*r.r), form = ['circle'],save=True,color = ['yellow'], ID=[['mean=%.3EADU/s - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)]],savename='/tmp/centers')
    if hasattr(r, 'w'):
        create_DS9regions([r.xc], [r.yc], radius=[2*r.w,2*r.h], form = ['box'],save=True,color = ['yellow'], ID=[['mean=%.3EADU/s - std=%0.1fADU'%(Decimal(meanADU/texp),stdADU)]],savename='/tmp/centers')

    d.set('region {}'.format('/tmp/centers.reg')) 
    return

def CalculateDispersion(table, DS9backUp = DS9_BackUp_path):
    x=[]
    y=[]
    disp=[]
    for i in np.arange(table['id_slit'].max()):
        if len(table[table['id_slit']==i])==2:
    #        print(abs(table[table['id_slit']==i][0]['X_IMAGE'] - table[table['id_slit']==i][1]['X_IMAGE'] ))   
    #        print(table[table['id_slit']==i]['wavelength'])
            distance = abs(table[table['id_slit']==i][0]['X_IMAGE'] - table[table['id_slit']==i][1]['X_IMAGE'] ) 
            ldistance = float(abs(table[table['id_slit']==i][0]['wavelength'] - table[table['id_slit']==i][1]['wavelength'] )) *1000
            if (float(distance) / ldistance) < 47.1:
                x.append(table[table['id_slit']==i]['X_IMAGE'].mean())
                y.append(table[table['id_slit']==i]['Y_IMAGE'].mean())
                disp.append(float(distance) / ldistance)    
        if len(table[table['id_slit']==i])==3:
            distance = abs(table[(table['id_slit']==i) & (table['wavelength']>0.2050)][0]['X_IMAGE'] - table[(table['id_slit']==i) & (table['wavelength']>0.2050)][1]['X_IMAGE'] ) 
            ldistance = float(abs(table[(table['id_slit']==i) & (table['wavelength']>0.2050)][0]['wavelength'] - table[(table['id_slit']==i) & (table['wavelength']>0.2050)][1]['wavelength'] )) * 1000
            if (float(distance) / ldistance) < 47.1:
                x.append(table[(table['id_slit']==i) & (table['wavelength']>0.2050) ]['X_IMAGE'].mean())
                y.append(table[(table['id_slit']==i) & (table['wavelength']>0.2050) ]['Y_IMAGE'].mean())
                disp.append(float(distance) / ldistance)                
            if (float(distance) / ldistance) < 47.1:
                distance = abs(table[(table['id_slit']==i) & (table['wavelength']<0.2070)][0]['X_IMAGE'] - table[(table['id_slit']==i) & (table['wavelength']<0.2070)][1]['X_IMAGE'] ) 
                ldistance = float(abs(table[(table['id_slit']==i) & (table['wavelength']<0.2070)][0]['wavelength'] - table[(table['id_slit']==i) & (table['wavelength']<0.2070)][1]['wavelength'] )) * 1000   
                x.append(table[(table['id_slit']==i) & (table['wavelength']<0.2070) ]['X_IMAGE'].mean())
                y.append(table[(table['id_slit']==i) & (table['wavelength']<0.2070) ]['Y_IMAGE'].mean())
                disp.append(float(distance) / ldistance)    
            #plt.plot(x,y,'x')
    n=12
    a = 1200
    b = 2000#900
    c = 100
    d = 1800#1600
    x1 = np.linspace(a,b,n)
    y1 = np.linspace(c,d,n)
    xx, yy = np.meshgrid(x1,y1)
    dispmap = np.zeros((n,n))
    for k in range(len(disp)):   
        for i in range(n):
            for j in range(n):
                if (x[k]>x1[i]) &  (x[k]<x1[i] + (b-a)/(n-1)) & (y[k]>y1[j]) &  (y[k]<y1[j] + (d-c)/(n-1)):
                    dispmap[i,j] = disp[k]
    dispb = np.zeros((2*n,n))
    for i in range(n):
        dispb[2*i,:] = dispmap[i,:]
        dispb[2*i+1,:] = dispmap[i,:]
    dispb[dispb==0] = np.nan
    plt.figure()
    plt.title('Mask to det dispersion measured with zinc lamp')
    plt.imshow(13*dispb[:-2,:].T,cmap='coolwarm')#cm.PRGn)#,vmin=45.9, vmax=46.5)cmap='inferno'
    #plt.scatter(np.array(x1),np.array(y1),'o')
    #plt.yticks(np.arange(n),x.astype(intl))
    #plt.xticks(np.arange(n),y.astype(int))
    plt.xlabel('Field of view x-coordinate (arcmin)' )
    plt.ylabel('Field of view y-coordinate (arcmin)')
    plt.yticks(np.linspace(0,9,5),[-450/60,-225/60,0,225/60,450/60])
    plt.xticks(np.linspace(0,18,5),[-780/60,-390/60,0,390/60,780/60])
    cb = plt.colorbar(orientation='horizontal')
    cb.set_label("dispersion [microns/nanometer]")
    plt.savefig(DS9_BackUp_path +'Plots/%s_Outputs_%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),'Dispersion'))
    plt.show()
    return x,y,disp
    
def DS9MeasureDispersion(xpapoint):
    table = Table.read(sys.argv[-1])
    x,y,disp = CalculateDispersion(table)
    return x,y,disp

    


def main():
    """Main function where the arguments are defined and the other functions called
    """
    #path = os.path.dirname(os.path.realpath(__file__))
    print(__file__)
    print(__package__)
    print('Python version = ', sys.version)
    CreateFolders()
    if len(sys.argv)==1:
        try:
            AnsDS9path = resource_filename('DS9FireBall','FireBall.ds9.ans')
        except:
            #sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
            print(__file__)
            pass
        else:
            print(bcolors.BLACK_RED + 'To use DS9Utils, add the following file in the DS9 Preferences->Analysis menu :  \n' + AnsDS9path + bcolors.END)
            print(bcolors.BLACK_RED + 'And switch on Autoreload' + bcolors.END)
            #sys.exit()

    
    
    print(datetime.datetime.now())
    start = timeit.default_timer()

    DictFunction = {
                    #GENERIC FUNCTIONS
                    'setup':DS9setup2,'Update':DS9Update,
                    'inverse': DS9inverse,'AddHeaderFieldCat':AddHeaderFieldCat,
                    'next':DS9next, 'previous':DS9previous, 'ImageAnalysis': DS9AnalyzeImage,
                    'stack': DS9stack_new,'lock': DS9lock, 'open':DS9open, 'CreateHeaderCatalog':DS9CreateHeaderCatalog,
                    'StackAllImages': StackAllImages, 'DS9Catalog2Region':DS9Catalog2Region, 'AddHeaderField':AddHeaderField,
                    'DS9Region2Catalog':DS9Region2Catalog, 'DS9MaskRegions':DS9MaskRegions,'BackgroundMeasurement':BackgroundMeasurement,
                    'DS9realignImage':DS9realignImage,'DS9createSubset':DS9createSubset,

                    #AIT Functions
                    'centering':DS9center, 'radial_profile':DS9rp,
                    'throughfocus':DS9throughfocus, 'regions': Field_regions,
                    'throughfocus_visualisation':DS9visualisation_throughfocus, 
                    'snr': DS9snr, 'focus': DS9focus,'DS9AddToCatalog':DS9AddToCatalog,
                    'throughslit': DS9throughslit, 'meanvar': DS9meanvar,
                    'xy_calib': DS9XYAnalysis,'SourcePhotometry':SourcePhotometry,'DS9MeasureDispersion':DS9MeasureDispersion,
                    
                    'AnalyzeOSSmearing':DS9AnalyzeOSSmearing,
                    #Flight Functions
                    'ReplaceWithNans': DS9replaceNaNs,'InterpolateNaNs': DS9InterpolateNaNs,
                    'OverscanCorrection': DS9OverscanCorrection, 'Trimming': DS9Trimming,
                    'SubstractImage': DS9RemoveImage, 'TotalReduction': DS9TotalReductionPipeline_new,
                    'photo_counting':DS9photo_counting, 'WCS':DS9guider, 'Remove_Cosmics': DS9removeCRtails2,
                    'BackgroundSubstraction': DS9SubstractBackground, #'plot_all_spectra':DS9plot_all_spectra,
                    'ColumnLineCorrelation': DS9CLcorrelation, 'OriginalSettings': DS9originalSettings,
                    'NoiseMinimization': DS9NoiseMinimization, 'SmearingProfile': DS9SmearingProfile,
                    'lya_multi_image':create_multiImage, 'DS9plot_spectra':DS9plot_spectra,
                    'ComputeEmGain': DS9ComputeEmGain, 'DS9ComputeDirectEmGain':DS9ComputeDirectEmGain, '2D_autocorrelation': DS9_2D_autocorrelation,
                    'HistogramDifferences': DS9HistogramDifferences, 'ComputeStandardDeviation': DS9ComputeStandardDeviation,
                    'PlotSpatial': DS9PlotSpatial, 'PlotArea3D':PlotArea3D, 'ContinuumPhotometry':ContinuumPhotometry,
                    'HistogramSums': HistogramSums, 'DS9DetectCosmics':DS9DetectCosmics, 'DetectHotPixels':DS9DetectHotPixels,
                    'DS9MultipleThreshold': DS9MultipleThreshold, 'DS9removeCRtails_CS':DS9removeCRtails_CS,'DS9_2D_FFT':DS9_2D_FFT,
                    'ComputeCIC':ComputeCIC,'DS9SmearingProfileAutocorr':DS9SmearingProfileAutocorr, 'AnalyseDark':AnalyseDark,
                    'DS9PhotonTransferCurve':DS9PhotonTransferCurve,'DS9Desmearing':DS9Desmearing,
                    #Photutils
                    'LoadRegions': DS9LoadCSV, 'BackgroundEstimationPhot': DS9BackgroundEstimationPhot, 'ExtractSources':DS9ExtractSources,
                    'Centroiding': Centroiding,#'PSFphotometry': PSFphotometry,'PSFmatching':PSFmatching,'Datasets': Datasets
                    'GroupingAlgorithm': GroupingAlgorithm, 'AperturePhotometry': AperturePhotometry,'BuildingEPSF': BuildingEPSF,
                    'MorphologicalProperties': MorphologicalProperties, 'EllipticalIsophoteAnalysis': EllipticalIsophoteAnalysis,
                    
                    #Create test images
                    'DS9createImage': DS9createImage, 'FollowProbabilityLaw': FollowProbabilityLaw,'CreateImageFromCatalogObject':CreateImageFromCatalogObject,
                    'ApplyRealisation': ApplyRealisation,'SimulateFIREBallemCCD':SimulateFIREBallemCCD,
                    
                     #Others
                    'test':DS9tsuite, 'ChangeConfig':ChangeConfig
             }

    xpapoint = sys.argv[1]
    function = sys.argv[2]
    Choose_backend(function)
    
    print(bcolors.BLACK_RED + 'DS9Utils ' + ' '.join(sys.argv[1:]) + bcolors.END)# %s %s '%(xpapoint, function) + ' '.join())
    
    print(bcolors.GREEN_WHITE + """
          ********************************************************************
                                     Function = %s
          ********************************************************************"""%(function)+ bcolors.END)
    a = DictFunction[function](xpapoint)             
    stop = timeit.default_timer()
    print(bcolors.BLACK_GREEN + """
        ********************************************************************
                            Exited OK, test duration = {}s      
        ******************************************************************** """.format(stop - start) + bcolors.END)
    if (type(a) == list) and (type(a[0]) == dict):
        for key in a[0].keys():
#            if (type(a[0][key])==float) or (type(a[0][key])==int):
            l=[]
            for i in range(len(a)):
                l.append(a[i][key])
            print(key, l)
            try:
                l=np.array(l)
                plt.figure()
                plt.hist(l)
                plt.title(key + 'M = %0.2f  -  Sigma = %0.3f'%(np.nanmean(l), np.nanstd(l)));plt.xlabel(key);plt.ylabel('Frequecy')
                plt.savefig(DS9_BackUp_path +'Plots/%s_Outputs_%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),key))
            except TypeError:
                pass

    return 



if __name__ == '__main__':
    a = main()

