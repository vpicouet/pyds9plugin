#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:13 2018

@author: Vincent
"""

import time
import glob
import os
import sys
import numpy as np
from pyds9 import DS9
import datetime
from  pkg_resources  import resource_filename
from astropy.table import Table
from mpl_toolkits.mplot3d import axes3d
try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()


#from shutil import which
#print("which('DS9Utils') =", which('DS9Utils'))
#print("__file__ =", __file__)
#print("__package__ =", __package__)
#print('Python version = ', sys.version)
#import matplotlib; matplotlib.use('TkAgg')  
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from .BasicFunctions import DS9lock


################################


width = 23#root.winfo_screenmmwidth() / 25.4
height = 14#root.winfo_screenmmheight() / 25.4

#IPython_default = plt.rcParams.copy()
plt.rcParams['figure.figsize'] = (2*width/3, 3*height/5)
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['axes.grid'] = True
plt.rcParams['image.interpolation'] = None
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'x-large'


#################################
#import matplotlib.pyplot as plt
#from .DS9Utils_ok import *


    
#get_ipython().magic(u'matplotlib inline')
#import tkinter as tk
#root = tk.Tk()

#os.environ["DS9Function"] = os.path.join(os.path.dirname(__file__),'doc/ref/index.html')
#def SetDiplay():
#    """Set the sparameters of the plots
#    """
width = 23#root.winfo_screenmmwidth() / 25.4
height = 14#root.winfo_screenmmheight() / 25.4

#IPython_default = plt.rcParams.copy()
plt.rcParams['figure.figsize'] = (2*width/3, 3*height/5)
plt.rcParams['grid.linestyle'] = ':'
plt.rcParams['axes.grid'] = True
plt.rcParams['image.interpolation'] = None
#plt.rcParams['savefig.transparent'] = True
plt.rcParams['xtick.labelsize'] = 'large'
plt.rcParams['ytick.labelsize'] = 'large'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.titlesize'] = 'x-large'
#    return

    
    
    
    

def CreateFolders(DS9_BackUp_path=os.environ['HOME'] + '/DS9BackUp/'):
    """Create the folders in which are stored DS9 related data
    """
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
    return  DS9_BackUp_path

DS9_BackUp_path = CreateFolders(os.environ['HOME'] + '/DS9BackUp/')


def LoadDS9QuickLookPlugin():
    """Load the plugin in DS9 parameter file
    """
    try:
        AnsDS9path = resource_filename('pyds9plugin','QuickLookPlugIn.ds9.ans')
        help_path = resource_filename('pyds9plugin','doc/ref/index.html')
    except:
        #sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
        print(__file__)
        pass
    else:

        if os.path.isdir(os.path.join(os.environ['HOME'], '.ds9')):
            for file in glob.glob(os.path.join(os.environ['HOME'], '.ds9','*')):
                if AnsDS9path not in open(file).read():
                    var = input("Do you want to add the Quick Look plug-in to the DS9 %s fils? [y]/n"%(os.path.basename(file)))
                    if  var.lower() != 'n':
                        ds9file = open(file,'a') 
                        ds9file.write('array set panalysis { user2 {} autoload 1 user3 {} log 1 user4 {} user %s }'%(AnsDS9path))
                        ds9file.close() 
                        print(bcolors.BLACK_GREEN + """Plug-in added"""+ bcolors.END)
                    else:
                        print(bcolors.BLACK_RED + 'To use the Quick Look plug-in, add the following file in the DS9 Preferences->Analysis menu :  \n' + AnsDS9path + bcolors.END)
                        print(bcolors.BLACK_RED + 'And switch on Autoreload' + bcolors.END)
            sys.exit()
        else:
            print(bcolors.BLACK_RED + 'To use DS9Utils, add the following file in the DS9 Preferences->Analysis menu :  \n' + AnsDS9path + bcolors.END)
            print(bcolors.BLACK_RED + 'And switch on Autoreload' + bcolors.END)
        bash_file = os.path.join(os.environ['HOME'], '.bashrc')
        if os.path.isfile(bash_file):
                if 'pyds9plugin' not in open(bash_file).read():
                    var = input("Do you want to add DS9 Plug-in help path to %s to access it from ds9? [y]/n"%(bash_file))
                    if  var.lower() != 'n':
                        ds9file = open(bash_file,'a') 
                        ds9file.write('export DS9Function="%s"'%(help_path))
                        ds9file.close()                     
            
        #sys.exit()
        
    return

def PresentPlugIn():
    """Print presentation of the plug in.
    """
    print(bcolors.BLACK_GREEN + 
 """                                                                                 
                     DS9 Quick Look Plug-in                                      
                                                                                 
            Written by Vincent PICOUET <vincent.picouet@lam.fr>                  
            Copyright 2019                                                       
            visit https://people.lam.fr/picouet.vincent                          
                                                                                 
            DS9 Quick Look Plug-in comes with ABSOLUTELY NO WARRANTY             
            You may redistribute copies of DS9 Quick Look Plug-in                
            under the terms of the MIT License.                                  
                                                                                 
            To use it run:                                                       
            > ds9 &                                                              
            and play with the analysis commands!                                 
                                                                                 """+ bcolors.END)
    return




class config(object):
    """Configuration class
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

        self.ConversionGain = float(config[config['param']=='ConversionGain']['value'].data[0])
        self.Autocorr_region_1D = np.array(config[config['param']=='Autocorr_region_1D']['value'].data[0].split('-'), dtype=int)
        self.Autocorr_region_2D = np.array(config[config['param']=='Autocorr_region_2D']['value'].data[0].split('-'), dtype=int)
        return 
    
try:
    conf_dir = resource_filename('pyds9plugin', 'config')
except:
    conf_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config')
try:
    my_conf = config(path=conf_dir+'/config.csv')
except (IOError or FileNotFoundError) as e:
    print(e)
    pass

    





def FitsExt(fitsimage):
    """Returns the number of the first image in a fits object
    """
    ext = np.where(np.array([type(ext.data) == np.ndarray for ext in fitsimage])==True)[0][0]
    print('Taking extension: ',ext)
    return ext

def DS9setup2(xpapoint, config=my_conf, color='cool'):
    """This function aims at giving a quick and general visualisation of the image by applying specific thresholding
        and smoothing parameters. This allows to detect easily:
        •Different spot that the image contains
        •The background low frequency/medium variation
        •If a spot saturates
        •If the image contains some ghost/second pass spots. . .
    """
    try:
        scale, cuts, smooth, color, invert, grid = sys.argv[-6:]
    except ValueError:
        scale, cuts, smooth, color, invert, grid = "Log",  '50-99.99',  '0', color, '0', '0' 

    cuts = np.array(cuts.split('-'),dtype=float)
    d = DS9(xpapoint)
    region = getregion(d, all=False, quick=True,selected=True)
    if region is None:
        image_area = [0,-1,0,-1]
    else:
        image_area = Lims_from_region(None,coords=region)
    Xinf, Xsup, Yinf, Ysup = image_area   

    print(Yinf, Ysup,Xinf, Xsup)
    #filename = getfilename(d)
    #fitsimage = fits.open(filename)[0].data#d.get_pyfits()[0].data#d.get_pyfits()[0].data#d.get_arr2np()
    fitsimage = d.get_pyfits()[0].data#d.get_pyfits()[0].data#d.get_arr2np()
    #print(fitsimage)
    image = fitsimage[Yinf: Ysup,Xinf: Xsup]#[Xinf: Xsup, Yinf: Ysup]

    d.set("cmap %s"%(color))
    d.set("scale %s"%(scale))


    if grid=='1':
        d.set("grid yes") 
    elif grid=='0':
        d.set("grid no") 
    if invert=='1':
        d.set("cmap invert yes") 
    elif invert=='0':
        d.set("cmap invert no") 
    if smooth!='0':
        d.set("smooth function boxcar") 
        d.set("smooth radius %i"%(int(smooth))) 

    d.set("scale limits {} {} ".format(np.nanpercentile(image,cuts[0]),np.nanpercentile(image,cuts[1])))
    return 


#d.get('file') tiff


###################################################################################
def DS9createSubset(xpapoint, cat=None, number=2,dpath=DS9_BackUp_path+'subsets/', config=my_conf):
    """Cfeate a subset of images considering a selection and ordering rules
    """
    from astropy.table import Table
    #import pandas as pd
    #cat, query, fields = sys.rgv[-3:]
    cat, bumber, fields, query = sys.argv[-4:]
    fields = np.array(fields.split(','),dtype=str)
    print('cat, bumber, fields, query = ',cat, bumber, fields, query )
    try:
        cat = Table.read(cat)
    except:
        print('Impossible to open table, verify your path.')
    df = cat.to_pandas()
    new_table = df.query(query)    
    t2 = Table.from_pandas(new_table) 
    print(t2)
    print('SELECTION %i -> %i'%(len(cat),len(t2)))
    
    print('SELECTED FIELD  %s'%(fields))

    path_date = dpath+datetime.datetime.now().strftime("%y%m%d_%HH%Mm%S")
    if not os.path.exists(path_date):
        os.makedirs(path_date)       

    numbers = t2[list(fields)].as_array()
    for line,numbers in zip(t2, numbers):
        filename = line['PATH']
        #print(fields)
        number = list(numbers)#np.array(list(line[fields]))
        #print(numbers)
        f = '/'.join(['%s_%s'%(a,b) for a,b in zip(fields,number)])
        new_path = os.path.join(path_date,f)
        #print(new_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path) 
        #print('Copying file',os.path.basename(filename))
        symlink_force(filename,new_path + '/%s'%(os.path.basename(filename)))    
#    paths = [path_date]
#    for path in paths:
#        for i in range(len(fields)):
#            values = values[i]#np.unique(t2[fields[i]])
#            paths = create_repositories(path, fields[i], values)
#        
#        for value in values:
#            npath = os.path.join(path, field + '_' + value)
#            os.makedirs(npath)     
    return t2

def create_repositories(path, field, values):
    """Create repository have different names and values
    """
    paths = []
    for value in values:
        npath = os.path.join(path, '%s_%s'%(field, value))
        #os.makedirs(npath)  
        print(npath)
        paths.append(npath)
    return paths





    

def PlotFit1D(x=None,y=[709, 1206, 1330],deg=1, Plot=True, sigma_clip=None, title=None, xlabel=None, ylabel=None, P0=None, bounds=(-np.inf,np.inf), fmt='.'):
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
        std = np.nanstd(y)
    else:
        sigma_clip = [10,1]
        index = (x > np.nanmean(x) - sigma_clip[0] * np.nanstd(x)) & (x < np.nanmean(x) + sigma_clip[0] * np.nanstd(x)) & (y > np.nanmean(y) - sigma_clip[1] * np.nanstd(y)) & (y < np.nanmean(y) + sigma_clip[1] * np.nanstd(y))
        std = np.nanstd(y[index])
        
    if Plot:
        fig = plt.figure()#figsize=(10,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=(4,1))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        #fig, (ax1, ax2) = plt.subplots(2, sharex=True,figsize=(10,6),height_ratios=[4,1])
        ax1.plot(x, y, fmt,label='Data')
        ax1.plot(x, np.ones(len(x))*np.nanmean(y[index]) + std, linestyle='dotted',label='Standard deviation')
    xp = np.linspace(x.min(),x.max(), 1000)

    if type(deg)==int:
        z, res, rank, singular, rcond = np.polyfit(x, y, deg, full=True)
        pcov = None
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
            if P0 is None:
                P0 = [np.nanmax(y)-np.nanmin(y),1,np.nanmin(y)]
        if deg=='2exp':
            law = lambda x, b1, b2, a1, a2, offset:  b1*np.exp(-x/a1) + b2*np.exp(-x/a2) +offset
#            if P0 is None:
#                P0 = [np.nanmax(y)-np.nanmin(y),1,np.nanmin(y)]

        elif deg=='gaus':
            law = lambda x, a, xo, sigma, offset: a**2 * np.exp(-np.square((x-xo) / sigma) / 2) + offset
            if P0 is None:
                P0 = [np.nanmax(y)-np.nanmin(y),x[np.argmax(y)],np.std(y),np.nanmin(y)]                
        elif deg=='power':
            law = lambda x, amp, index, offset: amp * (x**index) + offset
            P0 = None
        try:
            popt, pcov = curve_fit(law, x, y, p0=P0, bounds=bounds)
        except RuntimeError as e:
            print(e)
            return np.zeros(len(P0))
        zp = law(xp,*popt)
        zz = law(x,*popt)
        name = 'Fit %s'%(np.round(np.array(popt, dtype=int),0))
        res = None
        #plt.plot(xp, , '--', label='Fit: ')
    if Plot:
        if deg=='gaus':
            ax1.text(popt[1],popt[0]**2,'Max = %0.1f std'%(popt[0]**2/std))
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
        #plt.show()
        return {'popt':popt, 'pcov': pcov, 'res': res, 'axes': [ax1,ax2], 'y': y, 'x': x}
    return {'popt':popt, 'pcov': pcov, 'res': res, 'y': y, 'x': x}





def CreateRegions(regions,savename='/tmp/region.reg', texts='               '):
    """Create DS9 regions files from imported python region from DS9 
    """
    regions_ = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image
    """
    colors = ['orange','green','red','pink','grey','black']*100
    for region, text, color in zip(regions, texts,colors):
        #print(region)
        regions_ +=  '\n'  
        regions_ +=  addRegion(region, color=color, text=text)
    with open(savename, "w") as text_file:
        text_file.write(regions_)        
     


def addRegion(region, color='Yellow', text=''):
    """Add a region 
    """
    def get_r(region):
        return  region.r if hasattr(region, 'r') else [region.w, region.h]
    def get_type(region):
        return  'circle' if hasattr(region, 'r') else 'box'
    form = get_type(region) 
    
    if form == 'circle':
        text = '%s(%0.2f,%0.2f,%0.2f) # color=%s width=4 text={%s}'%(form, region.xc, region.yc, get_r(region), color, text )
    if form == 'box':
        text = '%s(%0.2f,%0.2f,%0.2f,%0.2f) # color=%s width=4 text={%s}'%(form, region.xc, region.yc, get_r(region)[0], get_r(region)[1], color, text )
    #print(text)
    return text

def getDatafromRegion(d,region, ext):
    """Get data from region
    """
    Xinf, Xsup, Yinf, Ysup  = Lims_from_region(region=region, coords=None,  config=my_conf)
    data = d.get_pyfits()[ext].data[Yinf:Ysup,Xinf:Xsup]
    return data




def create_DS9regions(xim, yim, radius=20, more=None, save=True, savename="test", form=['circle'], color=['green'], ID=None, config=my_conf):
    """Returns and possibly save DS9 region (circles) around sources with a given radius
    """
    
    regions = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image
    """
    if (type(radius) == int) or (type(radius) == float):
        r, r1 = radius, radius
    else:
        r , r1 = radius
    #r = radius   
    #print(range(len(xim)))
    for i in range(len(xim)):
        #print(form[i])
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
                    rest += ' # color={}'.format(color[j])
                    #print(color[j])
                regions += '{}({:f},{:f},'.format(form[i], x+1, y+1) + rest
                if ID is not None:
                    regions += ' text={{{}}}'.format(ID[i][j])
                    #print(ID[i][j])
                regions +=  '\n'  
        except ValueError:
            pass

    if save:
        with open(savename+'.reg', "w") as text_file:
            text_file.write(regions)        
        verboseprint(('Region file saved at: ' +  savename + '.reg'),verbose=config.verbose)
        return 



def create_DS9regions2(xim, yim, radius=20, more=None, save=True, savename="test",text=0, form='circle', color='green', config=my_conf,DS9_offset=[1,1],system='image'):
    """Returns and possibly save DS9 region (circles) around sources with a given radius
    """
    regions = """
        # Region file format: DS9 version 4.1
        global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
        %s
        """%(system)
    if system == 'fk5':
        DS9_offset=[0,0]
    #print ('form = ' + form )
    if form != 'ellipse':
        if (type(radius) == int) or (type(radius) == float) :
            r1, r2 = radius, radius
        else:
            r1 , r2 = radius
        if form == 'box':
            rest = ',%.5f,%.5f,%.5f) # color=%s'%(r1,r2,0,color)
        if form == 'circle':
            rest = ',%.5f) # color=%s'%(r1,color)
        if form == 'bpanda':
            rest = ',0,360,4,0.1,0.1,%.2f,%.2f,1,0) # color=%s'%(r1,2*r1,color)
        if form == 'annulus':
            n=10
            radius = np.linspace(0,r1,n)
            rest = ',%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f) # color=%s'%(radius[0],radius[1],radius[2],radius[3],radius[4],radius[5],radius[6],radius[7],radius[8],radius[9],color)  
         
        for i, x, y in zip(np.arange(len(xim)),xim, yim):
            if form == '# text':
                rest = ') color=%s text={%s}'%(color,text[i])          #regions += """\ncircle({},{},{})""".format(posx, posy, radius)
            regions +="""\n%s(%.5f,%.5f"""%(form,x+DS9_offset[0],y+DS9_offset[1]) + rest
        
        
        
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
        verboseprint('Saving region file',verbose=my_conf.verbose)
        with open(savename+'.reg', "w") as text_file:
            text_file.write("{}".format(regions))        
#        r.write(savename+'.reg') 
#        np.savetxt(savename+'.reg',regions)
        verboseprint(('Region file saved at: ' +  savename + '.reg'),verbose=config.verbose)
        return 

    
def getdata(xpapoint, Plot=False):
    """Get data from DS9 display in the definied region
    """
    import matplotlib.pyplot as plt
    #from astropy.io import fits
    d = DS9(xpapoint)
    region = getregion(d, quick=True)
    #filename = getfilename(d)
        

    Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
    data = d.get_pyfits()[0].data
    if len(data.shape) == 2:
        data = data[Yinf:Ysup,Xinf:Xsup]
        if Plot:
            plt.imshow(data)
            plt.colorbar()
    if len(data.shape) == 3:
        data = data[:,Yinf:Ysup,Xinf:Xsup]
#    if hasattr(region, 'r'):
#        data[data>region.r] = np.nan

    return data


def fitsgaussian2D(xpapoint, Plot=True, n=300):
    """2D gaussian fitting of the encricled region in DS9
    """
    from astropy.io import fits
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    #from scipy.stats import multivariate_normal
    from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
    fwhm, center, test = sys.argv[-3:]
    if bool(int(test)):
        Plot=False
        d = DS9(xpapoint)
        region = getregion(d, quick=True)
        filename = getfilename(d)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
        data = fits.open(filename)[0].data
        size = Xsup - Xinf
        xinfs, yinfs = np.random.randint(1100,1900,size=n), np.random.randint(100,1900,size=n)
        images = [data[Yinf:Yinf+size,Xinf:Xinf+size] for Xinf, Yinf in zip(xinfs, yinfs)]
        print('Test: number of images = %s'%(len(images)))
    else:
        images = [getdata(xpapoint)]

    fluxes = []
    for i,image in enumerate(images):
        #print(xinfs[i],yinfs[i])
        #print(image)

        while np.isfinite(image).all() == False:
            kernel = Gaussian2DKernel(stddev=2)
            image = interpolate_replace_nans(image, kernel)#.astype('float16')
            print(np.isfinite(image).all())

        print(np.isfinite(image).all())
        lx, ly = image.shape
        x = np.linspace(0,lx-1,lx)
        y = np.linspace(0,ly-1,ly)
        x, y = np.meshgrid(x,y)

         
        if fwhm.split('-')[0] == '':
            if bool(int(center)):
                Param = (np.nanmax(image),lx/2,ly/2,2,2,np.percentile(image,15))
                bounds = ([-np.inf, lx/2-0.5,ly/2-0.00001, 0.5,0.5,-np.inf], [np.inf, lx/2+0.00001,ly/2+0.5, 10,10,np.inf])#(-np.inf, np.inf)#
            else:
                xo, yo = np.where(image == np.nanmax(image))[1][0],  np.where(image == np.nanmax(image))[0][0]
                Param = (np.nanmax(image),int(xo),int(yo),2,2,np.percentile(image,15))
                bounds = ([-np.inf, xo-10 , yo-10, 0.5,0.5,-np.inf], [np.inf, xo+10 , yo+10, 10,10,np.inf])#(-np.inf, np.inf)#
        else:
            stdmin, stdmax = np.array(fwhm.split('-'), dtype=float)/2.35
            if bool(int(center)):
                Param = (np.nanmax(image),lx/2,ly/2, (stdmin+stdmax)/2, (stdmin+stdmax)/2,np.percentile(image,15))
                bounds = ([-np.inf, lx/2-0.5,ly/2-0.00001, stdmin, stdmin,-np.inf], [np.inf, lx/2+0.00001,ly/2+0.5, stdmax, stdmax,np.inf])#(-np.inf, np.inf)#
            else:
                xo, yo = np.where(image == np.nanmax(image))[1][0],  np.where(image == np.nanmax(image))[0][0]
                Param = (np.nanmax(image),xo, yo, (stdmin+stdmax)/2, (stdmin+stdmax)/2,np.percentile(image,15))
                bounds = ([-np.inf, xo-10 , yo-10, stdmin, stdmin,-np.inf], [np.inf, xo+10 , yo+10, stdmax, stdmax,np.inf])#(-np.inf, np.inf)#
    
    
        
        
        print ('bounds = ',bounds)
        print('\nParam = ', Param)
        try:
            popt,pcov = curve_fit(twoD_Gaussian,(x,y),image.flat,Param,bounds=bounds)
        except RuntimeError:
            popt = [0,0,0,0,0,0]
        else:
            print('\npopt = ', popt)
        fluxes.append(2*np.pi*popt[3]*popt[4]*popt[0])
    print(fluxes)
    if Plot:
        z = twoD_Gaussian((x,y),*popt).reshape(x.shape)
        fig = plt.figure()    
        ax = fig.add_subplot(111, projection='3d')   
        ax.scatter(x,y, image, s=5, cmap='twilight_shifted',vmin=np.nanmin(image),vmax=np.nanmax(image))#, cstride=1, alpha=0.2)
      
        ax.plot_surface(x,y,z,  label = 'amp = %0.3f, sigx = %0.3f, sigy = %0.3f '%(popt[0],popt[3],popt[4]), cmap='twilight_shifted',vmin=np.nanmin(image),vmax=np.nanmax(image), shade=True, alpha=0.7)#
        plt.title('3D plot, FLUX = %0.1f'%(fluxes[0]))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Pixel value')
        #ax.axis('equal')
        ax.axis('tight')
        ax.text(popt[1],popt[2], np.nanmax(image),s='amp = %0.3f, sigx = %0.3f, sigy = %0.3f '%(popt[0],popt[3],popt[4]))    
        plt.show()
    else:

        L = np.array(fluxes)
        median, mean, std = np.median(L), np.mean(L),np.std(L)
        limit = median + 3 * std
        mask = L >  limit



        if len(xinfs[mask])>0:
            create_DS9regions([xinfs[mask]+size/2],[yinfs[mask]+size/2], radius=[size,size], form = ['circle']*len(xinfs[mask]),save=True, savename='/tmp/centers',ID=[L[mask].astype(int)])
            d.set('regions /tmp/centers.reg')
        fig = plt.figure()
        #plt.hist(fluxes, bins=np.linspace(min(fluxes),max(fluxes),100))
        plt.hist(L[~mask],label='mean = %0.1f, std = %0.1f'%(mean, std), alpha=0.3,bins=50)
        plt.hist(L[mask],label='Detections', alpha=0.3)#,bins=50)
        plt.vlines(limit, 0, 10, label='5 sigma limit')
        plt.ylabel('Frequency')
        plt.xlabel('Flux estimator [log(ADU)]')
        plt.legend()
        plt.show()
        return




def DS9guider(xpapoint):
    """Display on DS9 image from SDSS at the same location if a WCS header is 
    present in the image. If not, it send the image on the astrometry.net server
    and run a lost in space algorithm to have this header. Processing might take a few minutes
    """
    from astropy.io import fits
    from astropy.wcs import wcs

    d = DS9(xpapoint)
    filename = getfilename(d)#d.get("file")
#    header = fits.getheader(filename)#d.get_pyfits()[0].header
#    if ('WCSAXES' in header) & 0==1:
#        print('WCS header existing, checking Image servers')
#        d.set("grid")
#        d.set("scale mode 99.5")
#        try:# if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
#            d.set("dsssao")
#        except:
#            pass
#        d.set("lock scalelimits no")
#        d.set("lock frame wcs")
#        d.set("frame last")
#        d.set("scale squared")
#        
#    else:
    type_ = sys.argv[-14]
    print('Type = ', type_)
    if type_ == 'XY-catalog':
        name = '/tmp/centers_astrometry.fits'
        DS9Region2Catalog(xpapoint, name=None, new_name=name)
        filename = name
    params = sys.argv[-13:]
    print('params = ',params)
    print ('Nop header WCS - Applying lost in space algorithm: Internet needed!')
    print ('Processing might take a few minutes ~5-10')
    PathExec = os.path.dirname(os.path.realpath(__file__)) + '/astrometry3.py'
    Newfilename = filename[:-5] + '_wcs.fits'
    CreateWCS(PathExec, filename, Newfilename, params=params, type_=type_)
    
    wcs_header  = wcs.WCS(fits.getheader(filename)).to_header()
    filename = getfilename(d)
    for key in list(dict.fromkeys( wcs_header.keys())):
        print(key)
        try:
            fits.setval(filename, key, value = wcs_header[key] , comment = '')
        except ValueError:
            pass    
    fits.setval(filename, 'WCSDATE', value =datetime.datetime.now().strftime("%y%m%d-%HH%M") , comment = '')
    d.set("lock frame wcs")        
    d.set("analysis message {Astrometry.net performed successfully! The WCS header has been saved in you image.}")    

    
#    filename = d.set("file {}".format(Newfilename))
#    filename = getfilename(d)#d.get("file")
#    header = d.get_pyfits()[0].header
#    if header['WCSAXES'] == 2:
#        print('WCS header existing, checking Image servers')
#        d.set("grid")
#        d.set("scale mode 99.5")#vincent
#        try:# if urllib.request.urlopen("http://google.com",timeout=1):#Internet == True:
#            d.set("dsssao")
#        except:
#            pass
#        d.set("lock scalelimits no")
        
    return



def CreateWCS(PathExec, filename, Newfilename, params, type_='Image'):
    """Sends the image on the astrometry.net server
    and run a lost in space algorithm to have this header. 
    Processing might take a few minutes
    """
    options = [' --scale-units   ',
#    ' --scale-type ',
    ' --scale-lower ',
    '  --scale-upper ',
    ' --scale-est ',
    ' --scale-err ',
    ' --ra ',
    ' --dec ',
    ' --radius ',
    ' --downsample ',
    ' --tweak-order ',
 #   ' --use_sextractor ',
    ' --crpix-center ',
    ' --parity ',
    ' --positional_error ']
    if type_ == 'XY-catalog':
        upload = "--upload-xy "
        options += [' --image-width ',' --image-height ']
        d = DS9()
        image = d.get_pyfits()[0]
        lx, ly = image.shape
        params += [lx, ly]
        print(options, params)
    else:
        upload = "--upload "
    print(type_,upload)
    #params = ['-'] * len(options)
    parameters = ' '.join([option + str(param) for option,param in zip(options, params) if param!='-'])
    print(parameters)


    print(filename, Newfilename)
    print(os.path.dirname(filename) + "/--wait.fits")
    start = time.time()
    print('''\n\n\n\n      Start lost in space algorithm - might take a few minutes \n\n\n\n''')
    executable = "python " + PathExec + " --apikey apfqmasixxbqxngm --wcs " + Newfilename + " --private y " + upload + filename + parameters
    print(executable)
    #import subprocess;    subprocess(executable)
    stop = time.time()
    print('File created')
    print('Lost in space duration = {} seconds'.format(stop-start))
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

def process_region(regions, win,quick=False, config=my_conf, message=True):
    """Process DS9 regions to return pythonic regions
    """
    from collections import namedtuple
    processed_regions = []

    for i, region in enumerate(regions):
        try:
            name, info = region.split('(')
        except ValueError:
            if message:
                d = DS9();d.set('analysis message {It seems that you did not create a region. Please create a region and rerun the analysis}')
            #sys.exit() 
        coords = [float(c) for c in info.split(')')[0].split(',')]
        #print('Region %i: %s'%(i, region))
        if quick:
            #print('Salut')
            #print(regions)
            if len(regions)==1:
                verboseprint('Only one region, taking it', verbose=config.verbose)
                return np.array(coords, dtype=int)
            else:
                verboseprint('There are no regions here', verbose=config.verbose)
                raise ValueError
        else:
            if name == 'box':
                xc,yc,w,h,angle = coords
                dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))                
                X,Y,arr = parse_data(dat)
                box = namedtuple('Box', 'data xc yc w h angle')
                processed_regions.append(box(arr, xc, yc, w, h, angle))
            elif name == 'bpanda':
                xc, yc, a1, a2, a3, a4,a5, w, h,a6,a7 = coords
                dat = win.get("data physical %s %s %s %s no" % (xc - w/2,yc - h/2, w, h))
                X,Y,arr = parse_data(dat)
                box = namedtuple('Box', 'data xc yc w h angle')
                processed_regions.append(box(arr, xc, yc, w, h, 0))
            elif name == 'circle':
                xc,yc,r = coords
                #dat = win.get("data physical %s %s %s %s no" % (xc - r, yc - r, 2*r, 2*r))
                #X,Y,arr = parse_data(dat)
                Xc,Yc = np.floor(xc), np.floor(yc)
                #inside = (X - Xc)**2 + (Y - Yc)**2 <= r**2
                circle = namedtuple('Circle', 'data databox inside xc yc r')
                #processed_regions.append(circle(arr[inside], arr, inside, xc, yc, r))
                processed_regions.append(circle(0, 0, 0, xc, yc, r))
            elif name == '# vector':
                xc, yc, xl, yl = coords
                vector = namedtuple('Vector', 'data databox inside xc yc r')
                processed_regions.append(vector(xc, yc, xl, yl,0,0))
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
        return processed_regions#[0]
    else:
        return processed_regions

   
    
    
    


def getregion(win, debug=False, all=False, quick=False, config=my_conf,selected=False, message=True):
    """ Read a region from a ds9 instance.
    Returns a tuple with the data in the region.
    """
    win.set("regions format ds9")
    win.set("regions system Image")#rows = win.get("regions list")
    if all is False:
        regions = win.get("regions selected")
        verboseprint(regions, verbose=config.verbose)
        verboseprint(len([row for row in regions.split('\n')]), verbose=config.verbose)
        if len([row for row in regions.split('\n')])>=3:
            verboseprint('Taking only selected region', verbose=config.verbose)
            rows = regions
        #else:
        elif selected is False:
                verboseprint('no region selected', verbose=config.verbose)
                rows = win.get("regions all")
        else:
            return None
            
    else:
        verboseprint('Taking all regions', verbose=config.verbose)
        rows = win.get("regions all")
    rows = [row for row in rows.split('\n') ]
    if len(rows) < 3:
        verboseprint( "No regions found", verbose=config.verbose)

    if debug:
        verboseprint (rows[4])
        if rows[5:]:
            verboseprint('discarding %i regions' % len(rows[5:]) , verbose=config.verbose)
    #print(rows[5:])
    if all:
        #print(rows[3:], type(rows[3:]))
        region = process_region(rows[3:], win,quick=quick, message=message)
        if type(region) == list:
            return region
        else:
            return [region]
    else:
        return process_region([rows[-1]], win,quick=quick, message=message)
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
    from scipy.optimize import curve_fit
    color = 'black'
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
    ext = FitsExt(fits.open(files[0]))
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[ext]
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
    fig, axes = plt.subplots(2, 2,figsize=(10,6),sharex=True)#, figsize=(10,6)
    xtot = np.linspace(x.min(),x.max(),200)
    print(ENCa)
    try:
        opt1,cov1 = curve_fit(f,x,fwhm)
        axes[0,0].plot(xtot,f(xtot,*opt1),linestyle='dotted',color = color)
        bestx1 = xtot[np.argmin(f(xtot,*opt1))]
        axes[0,0].plot(np.ones(2)*bestx1,[min(fwhm),max(fwhm)],color = color)
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
        axes[1,0].plot(xtot,f(xtot,*opt2),linestyle='dotted',color = color)
        bestx2 = xtot[np.argmin(f(xtot,*opt2))]
        if len(ENCa) > 0:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx2,ENC(bestx2,ENCa)))
        else:
            axes[1,0].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx2))
        axes[1,0].plot(np.ones(2)*bestx2,[min(EE50),max(EE50)],color = color)
    except RuntimeError:
        opt2 = [0,0,0]
        bestx2 = np.nan
        pass     
    try:
        opt3,cov3 = curve_fit(f,x,EE80)
        axes[1,1].plot(xtot,f(xtot,*opt3),linestyle='dotted',color = color)
        bestx3 = xtot[np.argmin(f(xtot,*opt3))]
        if len(ENCa) > 0:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx3,ENC(bestx3,ENCa)))
        else:
            axes[1,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx3))
        axes[1,1].plot(np.ones(2)*bestx3,[min(EE80),max(EE80)],color = color)
    except RuntimeError:
        opt3 = [0,0,0]
        bestx3 = np.nan
        pass     
    try:
        opt4,cov4 = curve_fit(f,x,maxpix)
        axes[0,1].plot(xtot,f(xtot,*opt4),linestyle='dotted',color = color)
        bestx4 = xtot[np.argmax(f(xtot,*opt4))]
        axes[0,1].plot(np.ones(2)*bestx4,[min(maxpix),max(maxpix)],color = color)
        if len(ENCa) > 0:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = %0.2f' % (bestx4,ENC(bestx4,ENCa)))
        else:
            axes[0,1].set_xlabel('Best index = %0.2f, Actuator = ?' % (bestx4))
    except RuntimeError:
        opt4 = [0,0,0]
        bestx4 = np.nan
        pass            

    bestx6, bestx6 = np.nan, np.nan
    import matplotlib.ticker as mtick
    axes[1,1].yaxis.set_label_position("right")
    axes[0,1].yaxis.set_label_position("right")
    axes[1,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    axes[0,1].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    axes[1,1].yaxis.tick_right()
    axes[0,1].yaxis.tick_right()  
    fig.tight_layout()

    name = '{} - {} - {}'.format(os.path.basename(filename),[int(center[0]),int(center[1])],fitsfile.header['DATE'])
    fig.suptitle(name, y=1)
    fig.savefig(os.path.dirname(filename) + '/Throughfocus-{}-{}-{}.png'.format( int(center[0]) ,int(center[1]), fitsfile.header['DATE']))
    print(name) 
    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix','Best sigma','Best EE50','Best EE80','Best Maxpix','Best Varpix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    t.add_row((os.path.basename(filename),1,
               t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],min(fwhm),
               min(EE50),min(EE80),min(maxpix),min(sumpix),max(varpix),
               ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)))#tbm

    if Plot:
        axes[0,0].plot(x,fwhm, '--o',label='$\sigma$[pix]',color = color, linewidth=0.5)
        axes[0,0].legend()    
        axes[1,0].plot(x,EE50, '--o',label=r'$EE_{50\%}[pix]$',color = color, linewidth=0.5)
        axes[1,0].legend()    
        axes[1,1].plot(x,EE80, '--o',label=r'$EE_{80\%}[pix]$',color = color, linewidth=0.5)
        axes[1,1].legend()

        axes[0,1].plot(x,maxpix, '-o',label='$Max_{pix}$',color = color, linewidth=0.5)
        axes[0,1].legend()       
        plt.show()
        
        
        fig, axes = plt.subplots(1, 11,sharey=True)#,figsize=(24,3))
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('equal')
            #subname = os.path.basename(files[i])[6:-5] 
            try:
                axes[i].set_xlabel('%s - %0.2f'%(os.path.basename(files[i].split('.')[0]),float(ENCa[i])))
                #axes[i].set_title(float(ENCa[i]))
            except:
                axes[i].set_xlabel(os.path.basename(files[i].split('.')[0]))
                pass
        fig.suptitle(name,y=1)
        #fig.subplots_adjust(top=0.88)   
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
    ext = FitsExt(fits.open(files[0]))
    for file in files:
        print (file)
        filename = file
        with fits.open(filename) as f:
            #stack[:,:,i] = f[0].data
            fitsfile = f[ext]
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
   
    mean = np.nanmean(np.array([ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)]))
    print(mean)
    name = '%s - %i - %i - %s - %0.3f'%(os.path.basename(filename),int(center_pix[0]),int(center_pix[1]),fitsfile.header['DATE'],mean)

    fig.suptitle(name, y=0.99)
    fig.tight_layout()

    t = Table(names=('name','number', 't', 'x', 'y','Sigma', 'EE50','EE80', 'Max pix','Flux', 'Var pix','Best sigma','Best EE50','Best EE80','Best Maxpix','Best Varpix'), dtype=('S15', 'f4','f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    t.add_row((os.path.basename(filename),os.path.basename(filename)[5:11],
               t2s(h=h,m=m,s=s,d=day), d['Center'][0],d['Center'][1],min(fwhm),
               min(EE50),min(EE80),min(maxpix),min(sumpix),max(varpix),
               ENC(bestx1,ENCa),ENC(bestx2,ENCa),
               ENC(bestx3,ENCa),ENC(bestx4,ENCa),
               ENC(bestx6,ENCa)))
    #name = '%s - %i - %i'%(os.path.basename(filename),int(center_pix[0]),int(center_pix[1]),fitsfile.header['DATE'],mean)
    print(name) 
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


def AnalyzeSpot(data, center, size=40, n=1.5,radius=40, fit=True, center_type='barycentre', radius_ext=12, platescale=None,fibersize = 100,SigmaMax=4):
  """Function used to plot the radial profile and the encircled energy of a spot,
  Latex is not necessary
  """
  from scipy import interpolate

  from scipy.optimize import curve_fit
  rsurf, rmean, profile, EE, NewCenter, stddev = radial_profile_normalized(data, center, radius=radius, n=n, center_type=center_type)
  profile = profile[:size]#(a[:n] - min(a[:n]) ) / np.nansum((a[:n] - min(a[:n]) ))
  #popt, pcov = curve_fit(ConvolveDiskGaus2D, np.linspace(0,size,size), profile, p0=[2,2,2])#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):  3.85  
  fiber = fibersize / (2*1.08*(1/0.083))
  if fiber == 0:
      gaus = lambda x, a, sigma: a**2 * np.exp(-np.square(x / sigma) / 2)
      popt, pcov = curve_fit(gaus, rmean[:size], profile, p0=[1, 2])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
  else:
      popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[1,fiber,2, np.nanmean(profile)],bounds=([0,0.95*fiber-1e-5,1,-1],[2,1.05*fiber+1e-5,SigmaMax,1]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):      
  EE_interp = interpolate.interp1d(rsurf[:size], EE[:size],kind='cubic')
  ninterp = 10
  xnew = np.linspace(rsurf[:size].min(),rsurf[:size].max(),ninterp*len(rsurf[:size]))
  mina = min(xnew[EE_interp(xnew)[:ninterp*size]>79])
  minb = min(xnew[EE_interp(xnew)[:ninterp*size]>49])
  if fiber == 0:
      flux = 2*np.pi*np.square(popt[1])*np.square(popt[0])
      d = {"Flux":flux,"SizeSource":0,"Sigma":abs(popt[1]),"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
      print("Flux = {}\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(flux,0,popt[1],minb,mina,platescale,NewCenter))
  else:
      d = {"Flux":0,"SizeSource":popt[1],"Sigma":abs(popt[2]),"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
      print("Flux = 0\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(popt[1],popt[2],minb,mina,platescale,NewCenter))
  return d

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
    
    print('''\n\n\n\n      START THROUGHFOCUS \n\n\n\n''')
    d = DS9(xpapoint)
    filename = getfilename(d)#d.get("file ")
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    #path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] 
    a = getregion(d)[0]
    if sys.argv[3] == '-':
        path = getfilename(d,All=True)
    else:
        path = globglob(sys.argv[3])
    
    #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    WCS = bool(int(sys.argv[4]))
    try:
        ENCa_center, pas = np.array(sys.argv[4].split('-'),dtype=float)
    except ValueError:
        print('No actuator given, taking header ones for guider images, none for detector images')
        ENCa_center, pas = None, None
    except IndexError :
        print('No actuator given, taking header ones for guider images, none for detector images')
        ENCa_center, pas = None, None        
    x = np.arange(len(path))
    
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
    if WCS:
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

def DS9rp(xpapoint, Plot=True, config=my_conf, center_type=None, fibersize=None, log=None):
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
    if center_type is None:
        print(sys.argv[3])
        center_type = sys.argv[3]
#    except IndexError:
#        entry = ''
    if fibersize is None:
        fibersize = sys.argv[4] if sys.argv[4].replace('.','',1).isdigit() else 0 and print('Fiber size not understood, setting to 0')
    if fibersize is None:
        log = bool(int(sys.argv[-1]))
    print('log = ',log)

    filename =  getfilename(d)#d.get("file ")
    a = getregion(d)[0]
    fitsfile = d.get_pyfits()[0]


    print(center_type)
    spot = DS9plot_rp_convolved(data=fitsfile.data,
                                center = [np.int(a.xc),np.int(a.yc)],
                                fibersize=fibersize, center_type=center_type,log=log, name = filename)    
#    try:
#        plt.title('{} - {} - {}'.format(os.path.basename(filename),[np.int(a.xc),np.int(a.yc)],fitsfile.header['DATE']),y=0.99)
#    except KeyError:
#        print('No date in header')
    if Plot:
        plt.show()
    d.set('regions command "circle %0.3f %0.3f %0.3f # color=red"' % (spot['Center'][0]+1,spot['Center'][1]+1,40))
    return


def radial_profile_normalized(data, center, anisotrope=False, angle=30, radius=40, n=1.5, center_type='barycentre', radius_bg=70,n1=20, stddev=True, size=70):
    """Function that returns the radial profile of a spot
    given an input image + center.
    Use the azimuthal average to compute the profile and determine the encircled energy
    """
    from scipy import ndimage
    y, x = np.indices((data.shape)) 
    print(data)
    #5#10  
    print('center_type = ',center_type)
    if center_type.lower() == 'maximum':
        image = data[int(center[1])-n1:int(center[1])+n1,int(center[0])-n1:int(center[0])+n1]
        barycentre =  np.array([np.where(image == image.max())[0][0], np.where(image == image.max())[1][0]])#ndimage.measurements.center_of_mass(data[center[1]-n1:center[1]+n1,center[0]-n1:center[0]+n1])
    if center_type.lower() == 'barycentre':
        image = data[int(center[1])-n1:int(center[1])+n1,int(center[0])-n1:int(center[0])+n1]
        #print ('Need to add background substraction')
        background = estimateBackground(data,center,radius,1.8 )
        new_image = image - background
        #print(new_image,new_image.shape)
        index = new_image > 0.5 * np.nanmax(new_image)#.max()
        #print(index)
        new_image[~index] = 0    
        barycentre = ndimage.measurements.center_of_mass(new_image)#background#np.nanmin(image)
    if center_type.lower() == 'user':
        barycentre = [n1,n1]
    else:
        image = data[int(center[1])-n1:int(center[1])+n1,int(center[0])-n1:int(center[0])+n1]
        print('Center type not understood, taking barycenter one')
        background = estimateBackground(data,center,radius,1.8 )
        new_image = image - background
        #print(new_image,new_image.shape)
        index = new_image > 0.5 * np.nanmax(new_image)#.max()
        #print(index)
        new_image[~index] = 0 
        barycentre = ndimage.measurements.center_of_mass(new_image)#background#np.nanmin(image)
    new_center = np.array(center) + barycentre[::-1] - n1
    print('new_center = {}, defined with center type: {}'.format(new_center, center_type))
    
    if radius_bg:
        fond = estimateBackground(data, new_center, radius, n)
    else:
        fond = 0
    image = data - fond#(data - fond).astype(np.int)
    
    r = np.sqrt((x - new_center[0])**2 + (y - new_center[1])**2)#    r = np.around(r)-1
    rint = r.astype(np.int)
    
    
    image_normalized = image #/ np.nansum(image[r<radius])
    if anisotrope == True:
        theta = abs(180*np.arctan((y - new_center[1]) / (x - new_center[0])) / np.pi)#    theta = np.abs(180*np.arctan2(x - new_center[0],y - new_center[1]) / np.pi)
        tbin_spectral = np.bincount(r[theta<angle].ravel(), image_normalized[theta<angle].ravel())
        tbin_spatial = np.bincount(r[theta>90-angle].ravel(), image_normalized[theta>90-angle].ravel())
        nr_spectral = np.bincount(r[theta<angle].ravel())
        nr_spatial = np.bincount(r[theta>90-angle].ravel())
        EE_spatial = 100 * np.nancumsum(tbin_spatial) / np.nanmax(np.nancumsum(tbin_spatial)[:100] + 1e-5)
        EE_spectral = 100 * np.nancumsum(tbin_spectral) / np.nanmax(np.nancumsum(tbin_spectral)[:100] + 1e-5)
        return tbin_spectral / nr_spectral, tbin_spatial / nr_spatial, EE_spectral, EE_spatial
    else:
        tbin = np.bincount(rint.ravel(), image_normalized.ravel())
        nr = np.bincount(rint.ravel())
        rsurf = np.sqrt(np.nancumsum(nr) /np.pi)
        rmean = np.bincount(rint.ravel(), r.ravel())/nr 
        if stddev:
            #print(datetime.datetime.now())
            dist = np.array(rint[rint<radius].ravel(), dtype=int)
            data = image[rint<radius].ravel()
            stdd = [np.nanstd(data[dist==distance])/ np.sqrt(len(data[dist==distance])) for distance in np.arange(size)]
            #print(datetime.datetime.now())

        radialprofile = tbin / nr
        EE = np.nancumsum(tbin) * 100 / np.nanmax(np.nancumsum(tbin)[:radius] + 1e-5)
        return rsurf[:size], rmean[:size], radialprofile[:size], EE[:size], new_center[:size], stdd[:size]


def estimateBackground(data, center, radius=30, n=1.8):
    """Function that estimate the Background behing a source given an inner radius and a factor n to the outter radius
    such as the background is computed on the area which is on C2(n*radius)\C1(radius)
    """
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)
    mask = (r>=radius) & (r<=n*radius)
    fond = np.nanmean(data[mask])
    return fond



def DS9plot_rp_convolved(data, center, size=40, n=1.5, log=False, anisotrope=False,angle=30, radius=40, ptype='linear', fit=True, center_type='barycentre', maxplot=0.013, minplot=-1e-5, radius_ext=12, platescale=None,fibersize = 100,SigmaMax=4, DS9backUp = DS9_BackUp_path, config=my_conf, name=''):
  """Function used to plot the radial profile and the encircled energy of a spot,
  Latex is not necessary
  """
  import matplotlib; matplotlib.use('TkAgg')  
  import matplotlib.pyplot as plt
  from scipy.optimize import curve_fit
  from scipy import interpolate
  rsurf, rmean, profile, EE, NewCenter, stddev = radial_profile_normalized(data, center, anisotrope=anisotrope, angle=angle, radius=radius, n=n, center_type=center_type, size=size)
  profile = profile[:size]#(a[:n] - min(a[:n]) ) / np.nansum((a[:n] - min(a[:n]) ))
  
  
  fig, ax1 = plt.subplots()
  fiber = float(fibersize) #/ (2*1.08*(1/0.083))
  if fiber == 0:
      gaus = lambda x, a, sigma: a**2 * np.exp(-np.square(x / sigma) / 2)
      popt, pcov = curve_fit(gaus, rmean[:size], profile, p0=[1, 2])#,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
      if log:
          ax1.semilogy(np.linspace(0,size,10*size), gaus(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
      else:
          popt_m, pcov_m = curve_fit(Moffat1D,rmean[:size], profile,p0=[profile.max(),4,2.5])
          ax1.plot(rmean[:size], Moffat1D(rmean[:size],*popt_m),label=r'Moffat fit: A=%0.3f, $\alpha$=%0.2f, $\beta$=%0.2f'%(popt_m[0],popt_m[1],popt_m[2]))
          
  else:
      stddev /= profile.max()
      profile /= profile.max() 
      popt_m, pcov_m = curve_fit(Moffat1D,rmean[:size], profile,p0=[profile.max(),4,2.5])
      popt, pcov = curve_fit(ConvolveDiskGaus2D, rmean[:size], profile, p0=[np.nanmax(profile),fiber,2, np.nanmin(profile)],bounds=([1e-3*(profile.max()-profile.min()),0.8*fiber,1,profile.min()],[1e3*(profile.max()-profile.min()),1.2*fiber,SigmaMax,profile.max()]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):    
      if log:
          ax1.semilogy(np.linspace(0,size,10*size), ConvolveDiskGaus2D(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
      else:
          ax1.plot(rmean[:size], Moffat1D(rmean[:size],*popt_m),label=r'Moffat fit: A=%0.3f, $\alpha$=%0.2f, $\beta$=%0.2f'%(popt_m[0],popt_m[1],popt_m[2]))
          ax1.plot(np.linspace(0,size,10*size), ConvolveDiskGaus2D(np.linspace(0, size, 10*size), *popt), c='royalblue') #)r"$\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!"
          ax1.fill_between(rmean[:size], profile - 1.5*np.abs(profile - ConvolveDiskGaus2D(rmean[:size], *popt)), profile + 1.5*np.abs(profile - ConvolveDiskGaus2D(rmean[:size], *popt)), alpha=0.3, label=r"3*Residuals")
  if log:
      p = ax1.semilogy(rmean[:size], profile, '.', c='black', label='Normalized isotropic profile')#, linestyle='dotted')
      ax1.set_ylim(ymin=1e-4)#, ymax=20*np.nanmax(np.log10(profile))
  else:
      p = ax1.plot(rmean[:size], profile, linestyle='dotted', c='black', label='Normalized isotropic profile')
      ax1.errorbar(rmean, profile, yerr = stddev, fmt='o', color=p[0].get_color(), alpha=0.5)
      ax1.set_ylim((minplot, np.nanmax([np.nanmax(1.1*(profile)), maxplot])))

  ax1.set_xlabel('Distance to center [pix]', fontsize=12)                      
  ax1.set_ylabel('Radial Profile', color='b', fontsize=12)
  ax1.tick_params('y', colors='b')
  ax2 = ax1.twinx()
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
  ax2.set_ylim((0, 110))
  ax2.set_ylabel('Encircled Energy', color='r', fontsize=12)
  ax2.tick_params('y', colors='r')
  ax1.xaxis.grid(True)
  ax1.tick_params(axis='x', labelsize=12)
  ax1.tick_params(axis='y', labelsize=12)
  ax2.tick_params(axis='y', labelsize=12)                    
  ax1.legend(loc = (0.54,0.05),fontsize=12)
  if fiber == 0:
      flux = 2*np.pi*np.square(popt[1])*np.square(popt[0])
      plt.figtext(0.53,0.53,"Flux = %0.0f\n$r_{in}$ = %0.1f pix \n$\sigma_{PSF}$ = %0.3f pix \n$EE^{50-80\%%}$ = %0.2f - %0.2f p" % (flux,0,abs(popt[1]),minb,mina), 
                  fontsize=14,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})#    norm_gaus = np.pi*sigma    norm_exp = 2*np.pi * lam**2 * gamma(2/alpha)/alpha
      d = {"Flux":flux,"SizeSource":0,"Sigma":popt[1],"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
      print("Flux = {}\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(flux,0,popt[1],minb,mina,platescale,NewCenter))
  else:
      plt.figtext(0.53,0.53,"Amp = %0.3f\n$r_{in}$ = %0.3f pix \n$\sigma_{PSF}$ = %0.3f pix \n$EE^{50-80\%%}$ = %0.2f - %0.2f p" % (popt[0],popt[1],popt[2],minb,mina), 
                  fontsize=14,bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})#    norm_gaus = np.pi*sigma    norm_exp = 2*np.pi * lam**2 * gamma(2/alpha)/alpha
      d = {"Flux":0,"SizeSource":popt[1],"Sigma":popt[2],"EE50":mina,"EE80":minb,"Platescale":platescale,"Center":NewCenter}
      print("Flux = 0\nSizeSource = {}\nSigma = {} \nEE50 = {}\nEE80 = {}\nPlatescale = {}\nCenter = {}".format(popt[1],popt[2],minb,mina,platescale,NewCenter))
  plt.title('{} - {}'.format(os.path.basename(name),np.round(NewCenter,1)),y=1)
  fig.tight_layout()
  plt.savefig(DS9backUp + 'Plots/%s_RdialProfile.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%Mm%Ss")) )
  csvwrite(np.vstack((rmean[:size], profile,ConvolveDiskGaus2D(rmean[:size], *popt))).T, DS9backUp + 'CSVs/%s_RadialProfile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
  csvwrite(np.vstack((rsurf, EE)).T, DS9backUp + 'CSVs/%s_EnergyProfile.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M")) )
  return d

      
def ConvolveDiskGaus2D(r, amp = 2 , RR = 4, sig = 4/2.35, offset=0):
    """Convolution of a disk with a gaussian to simulate the image of a fiber
    """
    from scipy.integrate import quad#, fixed_quad, quadrature
    from scipy import special#, signal, misc

    integrand =  lambda eta,r_ :  special.iv(0,r_ * eta / np.square(sig)) * eta * np.exp(-np.square(eta)/(2*np.square(sig)))
    #def integrand2 (eta, r_):
     #   return special.iv(0,r_ * eta / np.square(sig)) * eta * np.exp(-np.square(eta)/(2*np.square(sig)))
    integ = [quad(integrand,0,RR,args=(r_,))[0] * np.exp(-np.square(r_)/(2*np.square(sig))) / (np.pi*np.square(RR*sig)) for r_ in r]
    #integ = [np.exp(-np.square(r_)/(2*np.square(sig))) / (np.pi*np.square(RR*sig)) * np.nansum(integrand (np.linspace(0,RR,1000),r_))  for r_ in r]    
    #error = [quad(integrand,0,RR,args=(r_,))[1] * np.exp(-np.square(r_/(2*np.square(sig)))) / (np.pi*np.square(RR*sig)) for r_ in r]
    return offset + amp* np.array(integ)#, error






def Charge_path_new(filename, entry_point = 3, entry=None, All=0, begin='-', end='-', liste='-', patht='-', config=my_conf):
    """From the entry gave in DS9 (either nothing numbers or beginning-end),
    reuturns the path of the images to take into account in the asked analysis
    """
    #from astropy.io import fits
    import re
    print(sys.argv)
    try:
        a, b, e, l, p = sys.argv[-5:]
        print('a, b, e, l, p = ', a, b, e, l, p)
    except ValueError:
        pass
    else:
        if a.isdigit() & (b=='-' or b.isdigit()) & (e=='-' or e.isdigit()) & (p=='-' or len(glob.glob(p, recursive=True))>0):
            All, begin, end, liste, patht = sys.argv[-5:]
        else:
            print('Taking function argument not sys.argv')
    print('All, begin, end, liste, path =', All, begin, end, liste, patht)


    print('glob = ',glob.glob(patht, recursive=True))
    if len(glob.glob(patht, recursive=True))>0:
        print('Folder given, going though all the repositories of %s'%(patht))
        path = glob.glob(patht, recursive=True)
        #path += glob.glob(os.path.join(patht , '**/*.FIT'), recursive=True)
        #path += glob.glob(os.path.join(patht , '**/*.fit'), recursive=True)

    elif int(float(All)) == 1:
        print('Not numbers, taking all the .fits images from the current repository')
        #path = glob.glob('%s%s%s'%(filen1,'?'*n,filen2))
        path = glob.glob(os.path.dirname(filename) + '/*.fits')

    elif (begin == '-') & (end == '-') & (liste == '-')  :
        path = [filename]



    else:
        number = re.findall(r'\d+',os.path.basename(filename))[-1]
        n = len(number)
        filen1, filen2 = filename.split(number)
        print(filen1, filen2)
            
        if liste.split('-')[0]!='':
            path = []
            print('Specified numbers are integers, opening corresponding files ...')
            numbers = np.array( liste.split('-'),dtype=int)
            print('Numbers used: {}'.format(numbers))               
            for number in numbers:
                path.append(filen1 + '%0{}d'.format(n)%(number) + filen2) #'%0{}d'.format(n)%(1) + filen2
    
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
    if len(path)==0:
        d = DS9();d.set('analysis message {Could not find any image, the DS9 loaded image is in the right folder and have a comparable name?}')
    return path     
                        


def DS9visualisation_throughfocus(xpapoint):
    """Visualization of a throughfocus via DS9
    """
    d = DS9(xpapoint)
    #filename = getfilename(d)
    path = globglob(sys.argv[-1])##and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    d.set('tile yes')
    try:
        a = getregion(d, message = False)
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



def getImage(xpapoint):
    """Get image encircled by a region in DS9.
    """
    #from astropy.io import fits
    d = DS9(xpapoint)
    filename = getfilename(d)
    region = getregion(d)
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
    area = [Yinf, Ysup,Xinf, Xsup]
    #fitsimage = fits.open(filename)[0]
    fitsimage = d.get_pyfits()[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]
    print('Region =', region)
    if hasattr(region[0], 'r'):
        print('Region = Circle Radius = ',region[0].r)
        image = np.array(image, dtype=float)
        y, x = np.indices((image.shape))
        lx, ly = image.shape
        r = np.sqrt((x - lx/2)**2 + (y - ly/2)**2)
        image[r>int(region[0].r)] = np.nan
        print(image)

    header = fitsimage.header
    return image, header, area, filename
 
    

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.nanmean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.nanmean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    #z_middle = np.nanmean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    #ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return

def PlotArea3D(xpapoint):
    """Plot the image area defined in DS9 in 3D, should add some kind of 2D
    polynomial fit
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    #from mpl_toolkits.mplot3d import Axes3D #Needed even if not used
    #from matplotlib import cm
    #from .polynomial import polyfit2d
    log = np.bool(int(sys.argv[-2]))
    smooth = sys.argv[-1]
    image, header, area, filename = getImage(xpapoint)
    if smooth != '-':
        from astropy.convolution import convolve, Gaussian2DKernel

        if len(smooth.split(','))==1:
            sm = float(smooth.split(',')[0])
            smx, smy = sm, sm
        else:
            smx, smy = np.array(smooth.split(','),dtype=int)
        kernel = np.ones((int(smx),int(smy))) / (int(smx)*int(smy))
        kernel = Gaussian2DKernel(x_stddev=smx,y_stddev=smy)    
        image = convolve(image,kernel)
        image = image[int(3*smx):-int(3*smx),int(3*smy):-int(3*smy)]
        image = image[int(3*smy):-int(3*smy),int(3*smx):-int(3*smx)]
    
    X,Y = np.indices(image.shape)
    x, y = image.shape
    xm,ym = np.meshgrid(np.arange(x),np.arange(y),indexing='ij')
    x, y = xm.ravel(), ym.ravel()
    if log:
        image = np.log10(image - np.nanmin(image) +1)
        
    #max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), image.max()-image.min()]).max() / 2.0
    fig = plt.figure()
    ax = fig.gca(projection='3d', adjustable='box')
    #ax.set_aspect('equal')
    X1 = np.reshape(xm, -1)
    Y1 = np.reshape(ym, -1)
    Z1 = np.reshape(image, -1)
    ax.scatter(X1, Y1, Z1, c='r', s=200/len(x))#1.1)#, cstride=1, alpha=0.2)
    #ax.plot_trisurf(X1, Y1, Z1, cmap='twilight_shifted',vmin=np.nanmin(image),vmax=np.nanmax(image), shade=True, alpha=0.8)
    ax.plot_surface(X, Y, image, cmap='twilight_shifted',vmin=np.nanmin(image),vmax=np.nanmax(image), shade=True, alpha=0.7)#, cstride=1, alpha=0.2)
    #coeff = polyfit2d(x, y, imager, [4,4])
    plt.title('3D plot - %s - area = %s'%(os.path.basename(filename),area))
    plt.xlabel('X')
    plt.ylabel('Y')
    print(image)
    #print(0.9 * np.nanmin(image[np.isfinite(image)]), 1.1 * np.nanmax(image[np.isfinite(image)]))
    ax.set_zlim((0.9 * np.nanmin(image[np.isfinite(image)]), 1.1 * np.nanmax(image[np.isfinite(image)])))
    #ax.set_xlim(mid_x - max_range, mid_x + max_range)
    #ax.set_ylim(mid_y - max_range, mid_y + max_range)
    set_axes_equal(ax)
    ax.set_zlabel('Pixels ADU value')
    #ax.axis('equal')
    ax.axis('tight')
    plt.show()
    return






def fitswrite(fitsimage, filename, verbose=True, config=my_conf, header=None):
    """Write fits image function with different tests
    """
    from astropy.io import fits
    if type(fitsimage) == np.ndarray:
        fitsimage = fits.HDUList([fits.PrimaryHDU(fitsimage,header=header)])[0]
    if len(filename)==0:
        print('Impossible to save image in filename %s, saving it to /tmp/image.fits'%(filename))
        filename = '/tmp/image.fits'
        fitsimage.header['NAXIS3'], fitsimage.header['NAXIS1'] = fitsimage.header['NAXIS1'], fitsimage.header['NAXIS3']
        fitsimage.writeto(filename,overwrite=True) 
#    try:
#        fitsimage = fitsimage[0]
#    except TypeError:
#        pass
    if hasattr(fitsimage, 'header'):
        if 'NAXIS3' in fitsimage.header:
            verboseprint('2D array: Removing NAXIS3 from header...',verbose=my_conf.verbose)
            fitsimage.header.remove('NAXIS3')
    elif hasattr(fitsimage[0], 'header'):
        if 'NAXIS3' in fitsimage[0].header:
            verboseprint('2D array: Removing NAXIS3 from header...',verbose=my_conf.verbose)
            fitsimage[0].header.remove('NAXIS3')        
    elif not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if not os.path.exists(os.path.dirname(filename)):
        verboseprint('%s not existing: creating folde...'%(os.path.dirname(filename)),verbose=my_conf.verbose)
        os.makedirs(os.path.dirname(filename))
    try:
        #plt.plot(np.nanmean(fitsimage.data,axis=0));plt.show()
        fitsimage.writeto(filename, overwrite=True)
    except IOError:
        print(bcolors.BLACK_RED + 'Can not write in this repository : ' + filename + bcolors.END)
        filename = '/tmp/' + os.path.basename(filename)
        print(bcolors.BLACK_RED + 'Instead writing new file in : ' + filename + bcolors.END)
        fitsimage.writeto(filename,overwrite=True) 
    verboseprint('Image saved: %s'%(filename),verbose=my_conf.verbose)
    return filename


def csvwrite(table, filename, verbose=True, config=my_conf):
    """Write a catalog in csv format
    """
    import importlib
    from astropy.table import Table
    if type(table) == np.ndarray:
        table = Table(table)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    try:                                          
        table.write(filename, overwrite=True, format='csv')
    except UnicodeDecodeError:
        print('UnicodeDecodeError: you should consider change the name of the file/folder...')
        importlib.reload(sys); sys.setdefaultencoding('utf8')
    try:
        table.write(filename, overwrite=True, format='csv')
    except IOError:
        print(bcolors.BLACK_RED + 'Can not write in this repository : ' + filename + bcolors.END)
        filename = '/tmp/' + os.path.basename(filename)
        print(bcolors.BLACK_RED + 'Instead writing new file in : ' + filename + bcolors.END)
        table.write(filename, overwrite=True, format='csv')
    verboseprint('Table saved: %s'%(filename),verbose=True)#my_conf.verbose)
    return table

def gaussian(x, amp, x0, sigma):
    """Gaussian funtion
    """
    return amp*np.exp(-(x-x0)**2/(2*sigma**2))



def Gaussian(x, amplitude, xo, sigma2, offset):
    """Defines a gaussian function with offset
    """
    xo = float(xo)
    #A = amplitude/np.sqrt(2 * np.pi * sigma2)   #attention is not used anymore  
    g = offset + amplitude * np.exp( - 0.5*(np.square(x-xo)/sigma2))
    return g.ravel()




    
class bcolors:
    """Color class to use in print
    """
    BLACK_RED = '\x1b[4;30;41m' 
    GREEN_WHITE = '\x1b[0;32;47m' 
    BLACK_GREEN = '\x1b[0;30;42m' 
    END = '\x1b[0m'



def Convolve2d(xpapoint):
    """2D concolution with a gaussian, box or image
    """
    d = DS9(xpapoint)
    from astropy.convolution import Gaussian2DKernel, convolve
    #from scipy.signal import convolve as scipy_convolve
    from astropy.io import fits
    type_, path, sizex, sizey = sys.argv[-4-5:-5]
    if os.path.isfile(path) is False:
        if type_.lower() == 'gaussian':
            print('Convolving with gaussian')
            kernel = Gaussian2DKernel(x_stddev=float(sizex),y_stddev=float(sizey))
        elif type_.lower() == 'box':
            print('Convolving with box')
            kernel = np.ones((int(sizex),int(sizey))) / (int(sizex)*int(sizey))
    else:
        print('Convolving with image: ', path)
        kernel = fits.open(path)[0].data
    #scipy_conv = scipy_convolve(img, kernel, mode='same', method='direct')
    filename = getfilename(d)
    paths = Charge_path_new(filename) if len(sys.argv) > 5 else [filename]
    for file in paths:
        fitsfile = fits.open(file)
        i = np.argmax([im.size for im in fitsfile[:4]])
        #fitsfile = fitsfile[i]
        data_conv = convolve(fitsfile[i].data,kernel)
        fitsfile[i].data = data_conv
        #fitswrite(fitsfile,'/tmp/convolved_%s'%(os.path.basename(file)))
        fitsfile.writeto(file[:-5] + '_convolved.fits', overwrite=True)
    if len(paths)==1:
        d.set('frame new')
        d.set('file ' + file[:-5] + '_convolved.fits')
    return





def stackImages(path,all=False, DS=0, function = 'mean', numbers=None, save=True, name=""):
    """Stack all images contained in a folder, if all=True all images contained in each sub folder
    """
    from astropy.io import fits
    exts = ('*.FIT', '*.fits', '*.fts','*.fit')
    files=[]
    if all == True:
        folders = os.walk(path).next()[1]#glob.glob(path+'/*')
        for path1 in folders:
            global stackImages
            stackImages(path+'/'+path1,all=False)
    else:
        if numbers is None:
            for ext in exts:
                files.extend(glob.glob(path + ext)) 
        else:
            print('Using files number specified')
            for i in numbers:
                for ext in exts:
                    files.extend(glob.glob("{}/image{:06d}{}".format(path, int(i), ext))) 
        print(print("\n".join(files)))
        n = len(files)
        image = fits.open(files[0])[0]
        lx,ly = image.data.shape
        stack = np.zeros((lx,ly,n))
        print('\nReading fits files...')
        for i,file in enumerate(files):
            with fits.open(file) as f:
                stack[:,:,i] = f[0].data
        if function=='mean':
            image.data = np.nanmean(stack,axis=2) - DS
        if function=='median':
            image.data = np.nanmedian(stack,axis=2) - DS
        print('Images stacked')
        if save:
            fname = path + '/'#os.path.splitext(files[0])[0][:-6] + '-'
            if 'NAXIS3' in image.header:
                image.header.remove('NAXIS3')             

            if numbers is None:
                name = fname + 'stack' + '-' + name + '.fits'                
                image.writeto(name ,overwrite=True)
                print('Stacked image save at: ' + name)
            else:
                name = '{}StackedImage_{}-{}-{}.fits'.format(fname, int(numbers[0]), int(numbers[-1]), name)
#                name = '{}StackedImage_{}-{}-{}.fits'.format(fname, numbers.min(), numbers.max(), name)
                image.writeto(name ,overwrite=True)
                fits.setval(name, 'DARKUSED', value = 0, comment = 'Images subtracted for dark subtraction')
                #add
                print('Stacked image save at: ' + name)                

    return image.data, name




def globglob(file):
    """Improved glob.glob routine wher we can use regular expression with this: /tmp/image[5-15].fits
    """
    try:
        paths = glob.glob( file, recursive=True)
    except:
        paths=[]
    if (len(paths)==0) & ('[' in file) and (']' in file):
        print('Go in loop')
        a, between = file.split('[')
        between, b = between.split(']')
        if ('-' in between) & (len(between)>3):
            paths = []
            n1, n2 = np.array(between.split('-'),dtype=int)
            range_ = np.arange(n1, n2+1)
            print(range_)
            files = [a + '%0.{}d'.format(len(str(n2)))%(i) + b for i in range_]
            files += [a + '%i'%(i) + b for i in range_]
            for path in np.unique(files):
                if os.path.isfile(path):
                    paths.append(path) 
    paths.sort()
    return paths

def DS9stack_new(xpapoint, dtype=float, std=False, Type=None, clipping=None):
    """DS9 stacking function
    """
    d = DS9(xpapoint)
    #filename = d.get("file")
    if Type is None:
        Type = sys.argv[4]
        clipping = sys.argv[5]
    paths = globglob(sys.argv[3])
    print(sys.argv[3],paths)
  
    
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
    if clipping=='-':
        clipping = 1e5
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
    fitsfile = fits.open(paths[0])
    #sizes = [sys.getsizeof(elem.data) for elem in fitsfile]#[1]
    #i = np.argmax(sizes)
    i = FitsExt(fitsfile)
    Xinf, Xsup, Yinf, Ysup = my_conf.physical_region
    stds = np.array([np.nanstd(fits.open(image)[i].data[Xinf:Xsup, Yinf:Ysup]) for image in paths])
    plt.figure()
    plt.hist(stds)
    plt.title( 'Stds - M = %0.2f  -  Sigma = %0.3f'%(np.nanmean(stds), np.nanstd(stds)));plt.xlabel('Stds');plt.ylabel('Frequecy')
    plt.savefig(DS9_BackUp_path +'Plots/%s_Outputs_%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),'stds'))
    index = stds < np.nanmean(stds) + clipping * np.nanstd(stds) 
    paths = np.array(paths)
    if std is False:
        #print(index)
        stds = np.ones(len(paths[index]))
    else:
        stds /= np.nansum(stds[index])
        print('std = ', stds)
    n=len(paths)
    paths.sort()
    #name_images = [os.path.basename(path) for path in paths]
    lx,ly = fitsfile[i].data.shape
    stack = np.zeros((lx,ly),dtype=dtype)
    if std:
        print('Using std method')
        for i,file in enumerate(paths[index]):
            try:
                with fits.open(file) as f:
                    print(file, stds[i])
                    #f[0].data[~np.isfinite(f[0].data)] = stack[~np.isfinite(f[0].data)]
                    stack[:,:] += f[i].data/stds[i]
                    del(f)
            except TypeError as e:
                print(e)
                n -= 1
        stack = stack/n
    elif Type == np.nanmedian:
        print('Using std nanmedian')
        stack = np.array(Type(np.array([fits.open(file)[i].data for file in paths[index]]), axis=0), dtype=dtype)
    elif Type == np.nanmean:
        print('Using std nanmean')
        stack = Type(np.array([fits.open(file)[i].data for file in paths[index]]),dtype=dtype, axis=0)
    try:
        numbers = [int(re.findall(r'\d+',os.path.basename(filename))[-1]) for filename in paths[index]]
    except IndexError:
        numbers = paths
    images = ' - '.join(list(np.array(numbers,dtype=str))) 
    print(paths,numbers)
#    if cond_bckd:
#        index = FlagHighBackgroundImages(stack, std=1)
#    else:
#        index = True
        
#    if np.isfinite(stack).all():
#        stack.dtype = 'uint16'
    
    #print('All these images have not been taken into account because of hight background', '\n'.join(list(np.array(name_images)[~index]))  )
    #image[0].data = np.nanmean(stack[:,:,index],axis=2)#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    new_fitsfile = fitsfile[i]
    print(new_fitsfile.data, stack)
    new_fitsfile.data = stack#,dtype='uint16')#AddParts2Image(np.nanmean(stack,axis=2)) 
    new_fitsfile.header['STK_NB'] = images# '-'.join(re.findall(r'\d+',images))#'-'.join(list(np.array(name_images)[index]))   
    try:
        name = '{}/StackedImage_{}-{}{}.fits'.format(os.path.dirname(paths[0]), int(os.path.basename(paths[0])[5:5+6]), int(os.path.basename(paths[-1])[5:5+6]),fname)
#        name = '{}/StackedImage_{}-{}{}.fits'.format(os.path.dirname(paths[0]),np.min(numbers), np.max(numbers),fname)
    except ValueError:
        name = '{}/StackedImage_{}-{}{}'.format(os.path.dirname(paths[0]), os.path.basename(paths[0]).split('.')[0], os.path.basename(paths[-1]),fname)       
#        name = '{}/StackedImage_{}-{}{}'.format(os.path.dirname(paths[0]), np.min(numbers), np.max(numbers),fname)       
    print('Image saved : %s'%(name))
    try:
        fitswrite(new_fitsfile, name)
    except RuntimeError as e:
        print('Unknown error to be fixed: ', e)
        fitswrite(new_fitsfile.data, name)
        
    print('n = ', n)

    return fitsfile  , name




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
    from scipy.optimize import curve_fit

    print('''\n\n\n\n      START THROUGHSLIT \n\n\n\n''')
    d = DS9(xpapoint)
    filename = getfilename(d)#ffilename = d.get("file")
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    path.sort()
    x = np.arange(len(path))
    #print("\n".join(path)) 
    a = getregion(d)[0]
    radius = 15
    print('Sum pixel is used (another estimator may be prefarable)')
    fluxes=[]
    ext = FitsExt(fits.open(path[0]))

    for file in path:
        print (file)
        fitsfile = fits.open(file)[ext]
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
  
    name =  DS9backUp + 'CSVs/%s_ThroughSlit.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"))
    csvwrite(np.vstack((x, fluxesn)).T,name )
    csvwrite(np.vstack((xl, Gaussian(xl,*popt))).T,name[:-4]+'_fit.csv' )
    plt_=False
    if plt_:
        plt.figure()
        plt.plot(x, fluxesn,'o',label='data')
        plt.plot(xl, Gaussian(xl,*popt),'--',label='Gaussian fit')
        plt.legend()
        plt.plot(np.linspace(maxf, maxf, len(fluxes)), fluxesn/max(fluxesn))
        plt.grid(linestyle='dotted')
        plt.xlabel('# image')
        plt.title('Best image : {}'.format(maxf))
        plt.ylabel('Estimated flux (Sum pixel)') 
        name = 'Through slit analysis\n%0.3f - %s - %s'%(maxf,[int(a.xc),int(a.yc)],fitsfile.header['DATE'])
        print(name) 
        plt.title(name)
        plt.savefig(os.path.dirname(file) + '/' + name + '.jpg')
        plt.show()
    else:
        a = Table(np.vstack((x, fluxesn)).T)
        b = Table(np.vstack((xl, Gaussian(xl,*popt))).T)
        a.write(name ,format='ascii')
        b.write(name[:-4]+'_fit.csv' ,format='ascii')
        d = DS9Plot(path=name[:-4]+'_fit.csv', title='Best_image:{}'.format(maxf), name='Fit', xlabel='# image', ylabel='Estimated_flux', type_='xy', xlim=None, ylim=None,shape='none')
        DS9Plot(d=d,path=name, title='Best image : {}'.format(maxf), name='Fit', xlabel='# image', ylabel='Estimated flux (Sum pixel)', type_='xy', xlim=None, ylim=None,shape='circle',New=False)
    return 


def DS9Plot(d=None,path='', title='', name='', xlabel='', ylabel='', type_='xy', xlim=None, ylim=None,New=True,shape='None'):
    """Use DS9 plotting display
    """
    if d is None:
        d=DS9()
    if New:
        d.set('plot')#line|bar|scatter
    d.set('plot load %s %s'%(path, type_))
    d.set("plot title %s"%(title))
    d.set("plot title x %s"%(xlabel))
    d.set("plot title y %s"%(ylabel))
    d.set("plot show yes");#d.set("plot stats yes")
    d.set("plot shape %s"%(shape))
    d.set("plot name %s"%(name))
    d.set("plot legend yes")
    d.set("plot color red")
    if xlim is not None:
        d.set("plot axis x auto no")
        
    return d


    

def DS9Update(xpapoint,Plot=True, reg=True,style='--o',lw=0.5):
    """Always display last image of the repository and will upate with new ones
    """
    import time
    d = DS9(xpapoint)#DS9(xpapoint)

    if reg:
        regions =  getregion(d,all=True)
        CreateRegions(regions,savename='/tmp/region.reg', texts=np.arange(len(regions)))
        d.set('regions file /tmp/region.reg')
    ext = FitsExt(d.get_pyfits()) 
    filename = getfilename(d)#ffilename = d.get("file")
    files = glob.glob(os.path.dirname(filename)+ '/*.fits')
    files.sort()
    files = files*100
    fig, (ax0, ax1, ax2) = plt.subplots(3,1,sharex=True,figsize=(5,10))
    colors = ['orange','green','red','pink','grey','black']*100
    d0=[]
    for (i,region,color) in zip(np.arange(len(regions)), regions,colors):
        data = getDatafromRegion(d, region, ext=ext)
        d0.append(Center_Flux_std(data,bck=0,method='Gaussian-Picouet'))
        dn=d0
        ax0.semilogy(i, d0[i]['flux'], color=color,label=str(i))#,linewidth=lw)
        #ax0.scatter(i, np.log10(d0['flux']),color=color,label=str(i))
        ax1.scatter(i, d0[i]['std'],color=color,label=str(i))#,linewidth=lw)
        ax2.scatter(i, 0,color=color,label=str(i))#,linewidth=lw)


    ax0.legend(loc='upper left')
    
    ax1.set_xlabel('Number of images')
    ax0.set_ylabel('Flux[region]')
    ax1.set_ylabel('std[region]')
    ax2.set_ylabel('Distance[region]')
    ax0.set_title(os.path.basename(filename))
    fig.tight_layout()
    dn=[{'x':np.nan,'y':np.nan,'flux':np.nan,'std':np.nan}]*len(regions)
    for i, file in enumerate(files):
        ax0.set_title(os.path.basename(file))
        ax0.set_xlim((np.max([i-30,0]),np.max([i+1,30])))
        dnm = dn.copy()
        dn=[]
        for j, (region,color) in enumerate(zip(regions,colors)):
            data = getDatafromRegion(d, region, ext=ext)
            dn.append(Center_Flux_std(data,bck=0,method='Gaussian-Picouet'))
            #print('New centers = ',dn)
            print([i-1,i], [dnm[j]['flux'],dn[j]['flux']])
            ax0.plot([i-1,i], [dnm[j]['flux'],dn[j]['flux']],style,color=color,linewidth=lw)
            ax1.plot([i-1,i], [dnm[j]['std'],dn[j]['std']],style,color=color,linewidth=lw)
            ax2.plot([i-1,i], [distance(dnm[j]['x'],dnm[j]['y'],d0[j]['x'],d0[j]['y']),distance(dn[j]['x'],dn[j]['y'],d0[j]['x'],d0[j]['y'])],style,color=color,linewidth=lw)

            Xinf, Xsup, Yinf, Ysup = Lims_from_region(regions[j])
            regions[j] = regions[j]._replace(xc=Xinf + dn[j]['x'])        
            regions[j] = regions[j]._replace(yc=Yinf + dn[j]['y'])        
        CreateRegions(regions,savename='/tmp/region.reg', texts=np.arange(len(regions)))
        #pan = d.get('pan')
        d.set('file ' + file) 
        #d.set('pan to %s image'%(pan))
        d.set('regions file /tmp/region.reg')
        #y = np.random.random()

        plt.pause(0.00001)
        time.sleep(0.01)
    plt.show()
    return

def Center_Flux_std(image,bck=0,method='Gaussian-Picouet'):
    """Fit a gaussian and give, center flux and std
    """
    from photutils import centroid_com, centroid_1dg, centroid_2dg
    from scipy.optimize import curve_fit
#    if bool(int(bck)):
#        print('Subtracting background...')
#        background =  estimateBackground(data,[region.yc,region.xc],20,1.8 )
#        image = image - background
    lx, ly = image.shape
    
    if method == 'Center-of-mass': 
        xn, yn = centroid_com(image)
    
    if method == '2x1D-Gaussian-fitting': 
        xn, yn = centroid_1dg(image)
        
    if method == '2D-Gaussian-fitting': 
        xn, yn = centroid_2dg(image)
        
    if method == 'Maximum': 
        yn, xn = np.where(image==np.nanmax(image))[0][0], np.where(image==np.nanmax(image))[1][0]
    elif  method == 'Gaussian-Picouet':
        x = np.linspace(0,lx-1,lx)
        y = np.linspace(0,ly-1,ly)
        x, y = np.meshgrid(x,y)
        yo,xo = np.where(image == image.max())#ndimage.measurements.center_of_mass(image)
#        maxx, maxy = xc - (lx/2 - xo), yc - (ly/2 - yo)
#        print ('maxx, maxy = {}, {}'.format(maxx,maxy))

        bounds = ([1e-1*np.nanmax(image), xo-10 , yo-10, 0.5,0.5,-1e5], [10*np.nanmax(image), xo+10 , yo+10, 10,10,1e5])#(-np.inf, np.inf)#
        Param = (np.nanmax(image),int(xo),int(yo),2,2,np.percentile(image,15))
        #print ('bounds = ',bounds)
        #print('\nParam = ', Param)
        try:
            popt,pcov = curve_fit(twoD_Gaussian,(x,y),image.flat,
                                  Param,bounds=bounds)
            #print('\nFitted parameters = ', popt)
        except RuntimeError:
            #print('Optimal parameters not found: Number of calls to function has reached maxfev = 1400.')
            return np.nan, np.nan
        #print(np.diag(pcov))

        #fit = twoD_Gaussian((x,y),*popt).reshape((ly,lx))
        xn, yn = popt[1], popt[2]
        d = {'x':xn,'y':yn,'flux':2*np.pi*np.square(popt[1])*np.square(popt[0]),'std':np.sqrt(popt[-2]**2+popt[-3]**2)}
    return d 



def DS9Update_old(xpapoint,Plot=True):
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
     

    


   
def Lims_from_region(region=None, coords=None,  config=my_conf):
    """Return the pixel locations limits of a DS9 region
    """
    #print(coords)
    if coords is not None:
        try:
            xc, yc, w, h = coords[:4]
        except ValueError:
            try:
                xc, yc, w= coords[:4] 
                h = coords[-1]
            except ValueError:
                return None
    else:
        if hasattr(region, 'xc'):
            if hasattr(region, 'h'):
                xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
            if hasattr(region, 'r'):
                xc, yc, h, w = float(region.xc), float(region.yc), int(2*region.r), int(2*region.r)

        else:
            region = region[0]
            if hasattr(region, 'h'):
                xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
            if hasattr(region, 'r'):
                xc, yc, h, w = float(region.xc), float(region.yc), int(2*region.r), int(2*region.r)
    xc, yc = giveValue(xc,yc)
    #print('W = ', w)
    #print('H = ', h)
    if w <= 2:
        w = 2             
    if h <= 2:                                                                          
        h = 2
    Yinf = int(np.floor(yc - h/2 -1))
    Ysup = int(np.ceil(yc + h/2 +1))
    Xinf = int(np.floor(xc - w/2 -1))
    Xsup = int(np.ceil(xc + w/2 +1))
    try:
        verboseprint('Xc, Yc =  = ',region.xc, region.yc, verbose=config.verbose)
        verboseprint('Xc, Yc =  = ',xc, yc, verbose=config.verbose)
        verboseprint('Xinf, Xsup, Yinf, Ysup = ', Xinf, Xsup, Yinf, Ysup, verbose=config.verbose)
        verboseprint('data[%i:%i,%i:%i]'%(Yinf, Ysup,Xinf, Xsup), verbose=config.verbose)
    except AttributeError:
        pass
    return np.max([0,Xinf]), Xsup, np.max([0,Yinf]), Ysup

def giveValue(x,y):
    """Accoutn for the python/DS9 different way of accounting to 0 pixel
    """
    x = int(x) if x%1>0.5 else int(x)-1
    y = int(y) if y%1>0.5 else int(y)-1
    return x, y


def ConvolveBoxPSF(x, amp=1, l=40, x0=0, sigma2=40, offset=0):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    a = special.erf((l - (x - x0))/np.sqrt(2*sigma2))
    b = special.erf((l + (x - x0))/np.sqrt(2*sigma2))
    function = amp * ( a + b )/4*l
    return offset + function

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
    from photutils import centroid_com, centroid_1dg, centroid_2dg

    from scipy.optimize import curve_fit
    d = DS9(xpapoint)#DS9(xpapoint)
    #filename = getfilename(d)#ffilename = d.get("file")
    #region = getregion(d)[0]
    regions = getregion(d,all=True)#[0]
    for region in regions:
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
            data = d.get_pyfits()[0].data
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
    
            method, bck = sys.argv[-2:]
            Xinf, Xsup, Yinf, Ysup = Lims_from_region(region)
            data = d.get_pyfits()[0].data
            image = data[Yinf:Ysup,Xinf:Xsup]
            #image = data[Xinf:Xsup,Yinf:Ysup]
            print(image)
            print('2D fitting with 100 microns fibre, to be updated by allowing each fiber size')
            if bool(int(bck)):
                print('Subtracting background...')
                background =  estimateBackground(data,[region.yc,region.xc],20,1.8 )
                image = image - background
            lx, ly = image.shape
            
            if method == 'Center-of-mass': 
                xn, yn = centroid_com(image)
            
            if method == '2x1D-Gaussian-fitting': 
                xn, yn = centroid_1dg(image)
                
            if method == '2D-Gaussian-fitting': 
                xn, yn = centroid_2dg(image)
                
            if method == 'Maximum': 
                yn, xn = np.where(image==np.nanmax(image))[0][0], np.where(image==np.nanmax(image))[1][0]
            
            elif  method == 'Gaussian-Picouet':
                xc, yc = int(region.xc), int(region.yc)
                x = np.linspace(0,lx-1,lx)
                y = np.linspace(0,ly-1,ly)
                x, y = np.meshgrid(x,y)
                yo,xo = np.where(image == image.max())#ndimage.measurements.center_of_mass(image)
                maxx, maxy = xc - (lx/2 - xo), yc - (ly/2 - yo)
                print ('maxx, maxy = {}, {}'.format(maxx,maxy))
        
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
                print(np.diag(pcov))
    
                fit = twoD_Gaussian((x,y),*popt).reshape((ly,lx))
                xn, yn = popt[1], popt[2]
    
                if Plot:
                    plt.figure()
                    plt.plot(image[int(yo), :], 'bo',label='Spatial direction')
                    plt.plot(fit[int(yo), :],color='b')#,label='Spatial direction')
                    plt.plot(image[:,int(xo)], 'ro',label='Spatial direction')
                    plt.plot(fit[:,int(xo)],color='r')#,label='Spatial direction')
                    plt.ylabel('Fitted profiles')
                    plt.figtext(0.66,0.55,'Sigma = %0.2f +/- %0.2f pix\nXcenter = %0.2f +/- %0.2f\nYcenter = %0.2f +/- %0.2f' % ( np.sqrt(popt[3]), np.sqrt(np.diag(pcov)[3]/2.), lx/2 - popt[1] , np.sqrt(np.diag(pcov)[1]), ly/2 - popt[2], np.sqrt(np.diag(pcov)[2])),bbox={'facecolor':'blue', 'alpha':0.2, 'pad':10})
                    plt.legend()
                    plt.show()
            print('Region = ', region.xc, region.yc, region.r)
            print('Region = ',Yinf,Ysup,Xinf,Xsup)
            print('lx, ly = ', lx, ly)
            print('Sub array center = ', xn, yn)
            #print('Sub array center = ', xn, yn)
    #        xc, yc = int(region.xc), int(region.yc)
    
            newCenterx = Xinf + xn + 1#region.xc - (lx/2 - popt[1])
            newCentery = Yinf + yn + 1#region.yc - (ly/2 - popt[2])
            print('''\n\n\n\n     Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f] \n\n\n\n''' % (region.yc,region.xc,newCentery,newCenterx))
    
    
            d.set('regions delete select')
    
            try:
                os.remove('/tmp/centers.reg')
            except OSError:
                pass
    
            create_DS9regions([newCenterx-1],[newCentery-1], radius=region.r, save=True, savename="/tmp/centers", form=['circle'], color=['white'], ID=[['%0.2f - %0.2f' % (newCenterx,newCentery)]])
            
            d.set('regions /tmp/centers.reg')
    
    return newCenterx, newCentery

def t2s(h,m,s,d=0):
    """Transform hours, minutes, seconds to seconds [+days]
    """
    return 3600 * h + 60 * m + s + d*24*3600





def DS9Catalog2Region(xpapoint, name=None, x='xcentroid', y='ycentroid', ID=None,system='image'):
    """
    """
    from astropy.wcs import WCS

#    cat = Table.read('/Users/Vincent/Documents/Work/Thibault/UV_CLAUDS_HSC_s16a_uddd_deep_COSMOS_kNNZP_MixPHZ_PHYSPARAM_masked.fits')
#    x = 'RA' 
#    y = 'DEC' 
    import astropy
    from astropy.io import fits
    from astropy.table import Table
    if xpapoint is not None:
        d = DS9(xpapoint)
    if name is None:
        name = sys.argv[3]
    try:
        x, y = sys.argv[4].replace(',','-').split('-')
    except:
        pass

    try:
        cat = Table.read(name)
    except astropy.io.registry.IORegistryError:
        cat = Table.read(name, format='ascii')

    if (ID is None) & (sys.argv[5] != '-'):
        ID = sys.argv[5] 

    form = sys.argv[-3] 
    size = sys.argv[-2] 
    wcs = bool(int(sys.argv[-1] ))
    
    if wcs:
        filename = getfilename(d)
        a = fits.getheader(filename)
        wcs=WCS(a)
        corners = [#wcs.pixel_to_world(0,a['NAXIS2']), 
                   wcs.pixel_to_world(0,a['NAXIS1']), 
                   #wcs.pixel_to_world(a['NAXIS1'],0), 
                   wcs.pixel_to_world(a['NAXIS2'],0), 
                   wcs.pixel_to_world(a['NAXIS1'],a['NAXIS2']), 
                   wcs.pixel_to_world(0,0)]
        ra_max = max([corner.ra.deg for corner in corners])
        ra_min = min([corner.ra.deg for corner in corners])
        dec_max = max([corner.dec.deg for corner in corners])
        dec_min = min([corner.dec.deg for corner in corners])
        cat = cat[(cat[x]<ra_max) & (cat[x]>ra_min) & (cat[y]<dec_max) & (cat[y]>dec_min)]
        d.set('regions system wcs')
        d.set('regions sky fk5')
        d.set('regions skyformat degrees')
        system='fk5'
        size=float(size)/3600
        
    print(cat[x,y])
    if (sys.argv[5] == '-') & (ID is None):
        create_DS9regions2(cat[x],cat[y], radius=float(size), form = form, save=True,color = 'yellow', savename='/tmp/centers', system=system)
    else:
        create_DS9regions([cat[x]],[cat[y]], radius=float(size), form = [form],save=True,color = ['yellow'], ID=[np.round(np.array(cat[ID], dtype=float),1)],savename='/tmp/centers', system=system)
   
    if xpapoint is not None:
        d.set('regions /tmp/centers.reg')    
    return cat , '/tmp/centers.reg'

def DS9Region2Catalog(xpapoint, name=None, new_name=None):
    """Save DS9 regions as a catalog
    """
    print(new_name)
    from astropy.table import Table
    d = DS9(xpapoint)
    if new_name is None:
        new_name = sys.argv[-1]
    if name is not None:
        d.set('regions ' + name)
    regions = getregion(d, all=True, quick=False)
    #print(regions)
    #filename = getfilename(d)
    if hasattr(regions[0], 'xc'):
        x, y = np.array([r.xc for r in regions]), np.array([r.yc for r in regions])
    else:
        x, y = np.array([r.xc for r in [regions]]), np.array([r.yc for r in [regions]])
    cat = Table((x-1,y-1),names=('xcentroid','ycentroid'))    
    print(cat)
    if new_name is None:
        new_name = '/tmp/regions.csv'
    print(new_name)
    cat.write( new_name,overwrite=True)
    return cat


def DS9MaskRegions(xpapoint, length = 20):
    """Replace DS9 defined regions as a catalog
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
    """Replace DS9 defined regions as a catalog
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




def RunFunction(Fonction, args, return_dict):
    """Run a function in order to performe multi processing
    """
    print(*args)
    out = Fonction(*args)
    print(out)
    return_dict['output'] = out
    return 




def symlink_force(target, link_name):
    """Create a symbolic link
    """
    import errno
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

 

def even(f):
    """Return the even related number
    """
    return np.ceil(f / 2.) * 2

       
def verboseprint(*args, verbose=True):
    """Print function with a boolean verbose argument
    """
    if bool(int(verbose)):
        print(*args)
    else:
        pass


def distance(x1,y1,x2,y2):
    """
    Compute distance between 2 points in an euclidian 2D space
    """
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


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
        region = getregion(d, quick=True)
    except ValueError:
        try:
            reg = resource_filename('pyds9plugin', 'Regions')
        except:
            reg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Regions')
        d.set('regions ' + reg + '/Autocorr.reg')
        region = getregion(d, quick=True)
        print('No region defined! Taking default region in %s.\nDo not hesitate to change this default region if needed'%(reg + '/Autocorr.reg'))

    Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
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
        region = getregion(d, quick=True)
    except ValueError:
        try:
            reg = resource_filename('pyds9plugin', 'Regions')
        except:
            reg = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Regions')
        d.set('regions ' + reg + '/Autocorr.reg')
        region = getregion(d, quick=True)
        print('No region defined! Taking default region in %s.\nDo not hesitate to change this default region if needed'%(reg + '/Autocorr.reg'))
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
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
    fitsfile = fits.open(filename)
    ext = FitsExt(fitsfile)
    data = fitsfile[ext].data
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
    fitsimage = fits.open(filename)
    fitsimage = fitsimage[FitsExt(fitsimage)]
    data = fitsimage.data 

    if area is None:
        lx, ly = data.shape
        area = [int(lx/3),2*int(lx/3),int(ly/3),2*int(ly/3)]
    w = abs((area[1] - area[0]))
    h = abs((area[3] - area[2]))
    new_area = [area[0] - w,area[1] + w, area[2] - h,area[3] + h]
    finite = np.isfinite(np.mean(data[:, new_area[2]:new_area[3]],axis=1));
    data = data[finite,:]
    gain, temp = fitsimage.header.get(my_conf.gain[0], default=0.), fitsimage.header.get(my_conf.temperature[0], default=0.)
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
    fitswrite(fitsimage, name, verbose=my_conf.verbose)
    D = {'corr':corr, 'name':name, 'gain':gain, 'temp':temp}
    return D

 

def ReturnPath(filename,number=None, All=False):
    """Return path using regexp
    """
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

        
def DS9ComputeEmGain(xpapoint, subtract=True, verbose=False, config=my_conf):
    """Compute EMgain with the variance intensity method
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    if len(d.get('regions').split('\n'))!=5:
        d.set('region delete all')
    #if len(sys.argv) > 5: path = Charge_path_new(filename, entry_point=5)
    path = Charge_path_new(filename) if len(sys.argv) > 8 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
  
    subtract, number, size, overscan, limits = sys.argv[3:8]#'f3 names'#sys.argv[3]
    overscan = int(overscan)
    limits =  np.array(limits.split(','),dtype=int)  
    radius=np.array(size.split(','),dtype=int)
#    offset = 20
#    OSR1  = [offset,-offset,limits[0],limits[1]]
#    if overscan == 1:
#        OSR2 = None
#    else:
#        OSR2  = [offset,-offset,limits[2],limits[3]]
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
        region = getregion(d, quick=True)
    except ValueError:
        print('Please define a region.')
        area = my_conf.physical_region#[1053,2133,500,2000]
    else:
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
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
            D.append(ComputeEmGain(filename, Path2substract=Path2substract, save=True, d=d, Plot=plot_flag, area=area, subtract=subtract, radius=radius, OSR1=[20,-20,0,400], OSR2=[20,-20,2200,2600]))
        else:
            D = ComputeEmGain(filename, Path2substract=Path2substract, save=True, d=d, Plot=plot_flag, area=area, subtract=subtract, radius=radius, OSR1=[20,-20,0,400], OSR2=[20,-20,2200,2600])
    #print(D)
    return D
 




def SigmaClipBinned(x,y, sig=1, Plot=True, ax=None, log=False):
    """Perform sigma clipped binning on a x, y dataset
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
            if ax is None:
                fig = plt.figure()#figsize=(12,4.5))
                ax = fig.add_subplot(111)
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
                



def ComputeEmGain(filename, Path2substract=None, save=True, Plot=True, d=None, ax=None, radius=[40,40], subtract=False, area=None, DS9backUp = DS9_BackUp_path,verbose=True, config=my_conf, OSR1=[20,-20,0,400], OSR2=[20,-20,2200,2400]):
    """Compute EMgain with the variance intensity method
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    from astropy.io import fits
    verboseprint("""##################\nSubtracting Image = %s \nPath to subtract = %s\nradius = %s \nArea = %s\n\nfilename = %s \nplot_flag = %s\n##################"""%(subtract,Path2substract, radius, area, filename, Plot),verbose=my_conf.verbose)
    fitsimage = fits.open(filename)[0]
    image = fitsimage.data
    try:
        texp = fitsimage.header[my_conf.exptime[0]]
    except KeyError:
        texp=1
    #offset = 20
    image = image - ComputeOSlevel1(image, OSR1=OSR1,OSR2=OSR1)
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
                    verboseprint('Subtracting previous image: %s'%(name),verbose=my_conf.verbose)
                    image_sub = image - data 
                    im=0
                else: 
                    verboseprint('Previous image do not have same exposure time',verbose=my_conf.verbose)
                    im='next'
            if (im=='next') or (images.index(filename)==len(images)-1):
                name = images[images.index(filename) - 1]
                image_n = fits.open(name)[0]
                data, exptime = image_n.data, image_n.header[my_conf.exptime[0]]
                if int(float(exptime)) == int(float(texp)):
                    verboseprint('Subtracting next image: %s'%(name),verbose=my_conf.verbose)
                    image_sub = image - data  
                else:
                    verboseprint('No image have the same exposure time: No subtraction!',verbose=my_conf.verbose)
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
    verboseprint('Number of regions : ', len(areas),verbose=my_conf.verbose)
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
    fig.subplots_adjust(left=None, bottom=0.3, right=None, top=None, wspace=None, hspace=None)
#    fig= plt.figure()#plt.subplots(1,2)#,figsize=(14,6))  
#    ax0 = fig.add_axes([0.10, 0.30, 0.84/2-0.05, 0.66])      
#    ax1 = fig.add_axes([0.6, 0.30, 0.84/2-0.05, 0.66])      
      
    intensity_phys_n,var_phys_n = SigmaClipBinned(intensity_phys,var_phys/cst, sig=1, Plot=True, ax=ax0)
    intensity_os_n,var_os_n = SigmaClipBinned(intensity_os,var_os/cst, sig=1, Plot=True, ax=ax0)
    intensity_phys_n,var_phys_n = SigmaClipBinned(intensity_phys,var_phys/cst, sig=1, Plot=True, ax=ax1)
    
    ax, emgain_phys = PlotComputeEmGain_old(intensity_phys_n, var_phys_n, emgain , r1*r2, filename=filename, len_area_det=len_area_det, ax=ax1, cst='(%i x %i)'%(cst,n))
    ax, emgain_all = PlotComputeEmGain(np.hstack((intensity_os_n,intensity_phys_n)), np.hstack((var_os_n,var_phys_n)), emgain , r1*r2, filename=filename, len_area_det=len_area_det, ax=ax0, cst='(%i x %i)'%(cst,n))

    csvwrite(np.vstack((intensity_phys_n,var_phys_n/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ,verbose=my_conf.verbose)
    csvwrite(np.vstack((np.hstack((intensity_os_n,intensity_phys_n)),np.hstack((var_os_n,var_phys_n))/cst)).T, DS9backUp + 'CSVs/%s_VarianceIntensity_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%H:%M:%S"),os.path.basename(filename)[:-5]) ,verbose=my_conf.verbose)


    ax0.set_ylim((0.97*np.hstack((var_os_n,var_phys_n)).min(),1.03*np.hstack((var_os_n,var_phys_n)).max()))
    ax0.set_xlim((0.97*np.hstack((intensity_os_n,intensity_phys_n)).min(),1.03*np.hstack((intensity_os_n,intensity_phys_n)).max()))
    ax1.set_ylim((0.97*var_phys_n.min(),1.03*var_phys_n.max()))
    ax1.set_xlim((0.97*intensity_phys_n.min(),1.03*intensity_phys_n.max()))
    fig.suptitle('Variance intensity diagram - %s - G = %s - #regions = %i'%(os.path.basename(filename),emgain,areas[:,1].shape[0]),y=1)    
    #fig.tight_layout()
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


def CreateAreas(image, area=None, radius=100, offset=20, verbose=False, config=my_conf):
    """Create areas in the given image
    """
    #image = a.data$
    if type(radius)==int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    verboseprint('r1,r2=',r1,r2,verbose=my_conf.verbose)
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


def PlotComputeEmGain(intensity, var, emgain, n, filename, len_area_det, ax=None, DS9backUp = DS9_BackUp_path, name='',cst=2):
    """Compute emgain based on variance intensity diagram
    """
    import matplotlib; matplotlib.use('TkAgg')  
    from .dataphile.demos import auto_gui
    obj = auto_gui.GeneralFit(intensity,var,ax=ax,linestyle=None, marker=None)
    obj.ax.set_ylabel('Variance [ADU] / %s'%(cst))

    return ax, emgain


def PlotComputeEmGain_old(intensity, var, emgain, n, filename, len_area_det, ax=None, DS9backUp = DS9_BackUp_path, name='',cst=2):
    """
    """
    import matplotlib; matplotlib.use('TkAgg')  
    import matplotlib.pyplot as plt
    #from .dataphile.demos import auto_gui
    #auto_gui.GeneralFit(intensity,var,ax=ax,linestyle=None, marker=None)
    fit = np.polyfit(intensity,var,1)
    fit_fn = np.poly1d(fit) 
    if ax is None:
        fig = plt.figure()#figsize=(8,6))
        ax = fig.add_subplot(111)
    ax.plot(intensity, fit_fn(intensity), '--', label='Linear regression, GainTot = %0.1f \n-> %0.1f smr corr (0.32)'%(fit[0],fit[0]/0.32))
    ax.set_ylabel('Variance [ADU] / %s'%(cst))
    ax.text(0.5,0.1,'y = %0.2f * x + %0.2f'%(fit[0], fit[1]),transform=ax.transAxes)
    ax.legend(loc='upper left')
    ax.set_xlabel('Intensity [ADU]')
    ax.grid(linestyle='dotted')
    emgain = fit[0]
    return ax, emgain




def DS9Trimming(xpapoint, config=my_conf, all_ext=False):
    """Crop the image to have only the utility area
    """
    from astropy.io import fits
    from astropy import wcs
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
    try:
        region = getregion(d, quick=True)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    except ValueError:
        area = [0,-1,1053,2133]
    w = wcs.WCS(fits.getheader(filename))
    if ('' in w.wcs.ctype)  & (0 in w.wcs.crval)  & (0 in w.wcs.crpix) :
        result, name = ApplyTrimming(filename, area=area, all_ext=False)
    else:
        for filename in path:
            print(filename)
            result, name = cropCLAUDS(path=filename, area=area, all_ext=False)
    print(name) 
    if len(path) < 2:
        d.set('frame new')
        d.set('tile yes')
        d.set("file %s" %(name))  
        d.set('file ' + name)  
    return


def cropCLAUDS(path='/Users/Vincent/Documents/Work/sextractor/calexp/calexp-HSC-I-9813-2,4.fits',area=[0,100,0,100], all_ext=False):
    """Cropping/Trimming function that keeps WCS header information
    """
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.io.fits import ImageHDU
    a = fits.open(path)
    size = np.array([area[1]-area[0],area[3]-area[2]])
    position = (int((area[2]+area[3])/2),int((area[1]+area[0])/2))
    print(position)
    print(size)
    for i in range(4):
        print(i)
        try:    
            di = Cutout2D(a[i].data, position=position,size=size,wcs=WCS(a[i].header))
            if i ==0:
                a[i] = fits.PrimaryHDU(data=di.data, header=di.wcs.to_header())
            else:
                a[i] = ImageHDU(data=di.data, header=di.wcs.to_header())
        except (ValueError,IndexError) as e:
            print(i,e)
            pass

    a.writeto(path[:-5] + '_trim.fits',overwrite=True)
    return a, path[:-5] + '_trim.fits'


def ApplyTrimming(path, area=[0,-0,1053,2133], config=my_conf, all_ext=False):
    """Apply overscan correction in the specified region, given the two overscann areas
    """
    from astropy.io import fits
    fitsimage = fits.open(path)
    fitsimage_ = fitsimage[FitsExt(fitsimage)]
    name = path[:-5] + '_Trimm.fits'
    if all_ext:
        for i in range(len(fitsimage)):
            if type(fitsimage[i].data)==np.ndarray :
                print('Croping extension ', i)
                fitsimage[i].data = fitsimage[i].data[area[0]:area[1],area[2]:area[3]]
        fitswrite(fitsimage, name)
    else:
        image = fitsimage_.data[area[0]:area[1],area[2]:area[3]]
    #image = image[area[0]:area[1],area[2]:area[3]]
        fitsimage_.data = image
        fitswrite(fitsimage_.data, name, header=fitsimage_.header)
    #fitsimage.writeto('/tmp/test.fits',overwrite=True)
    return fitsimage, name

def DS9CLcorrelation(xpapoint, config=my_conf):
    """Performs a column to column or or line to line auto-correlation on a DS9 image
    """
    d = DS9(xpapoint)
    filename = getfilename(d)
    #if len(sys.argv) > 3: path = Charge_path_new(filename, entry_point=3)
    path = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    try:
        region = getregion(d, quick=True)
        Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
        area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)
    except ValueError:
        area = [0,-1,1053,2133]
        
    for filename in path:
        print(filename)
        CLcorrelation(filename, area=area) 
    return

def CLcorrelation(path, area=[0,-1,1053,2133], DS9backUp = DS9_BackUp_path, config=my_conf):
    """Performs a column to column or or line to line auto-correlation on a DS9 image
    """
    import matplotlib; matplotlib.use('TkAgg')  
    from astropy.io import fits
    from matplotlib import pyplot as plt
    fitsimage = fits.open(path)[0]
    image = fitsimage.data[area[0]:area[1],area[2]:area[3]]
    #image = image[area[0]:area[1],area[2]:area[3]]
    imagex = np.nanmean(image, axis=1)
    imagey = np.nanmean(image, axis=0)
    nbins=300
    fig, ax = plt.subplots(2, 2)#, figsize=(12,7))
    ax[0,0].hist(imagex[1:]-imagex[:-1],bins=np.linspace(np.percentile(imagex[1:]-imagex[:-1],5),np.percentile(imagex[1:]-imagex[:-1],95),nbins),histtype='step',label='Lines')
    ax[1,0].hist(imagey[1:]-imagey[:-1],bins=np.linspace(np.percentile(imagey[1:]-imagey[:-1],5),np.percentile(imagey[1:]-imagey[:-1],95),nbins),histtype='step',label='Column', color='orange')
#    ax[0,0].hist(imagex[1:]-imagex[:-1],bins=10*nbins,histtype='step',label='Lines')
#    ax[1,0].hist(imagey[1:]-imagey[:-1],bins=10*nbins,histtype='step',label='Column', color='orange')
#    ax[0,1].hist((image[1:,:] - image[:-1,:]).flatten(),bins=nbins,histtype='step',label='Column', color='orange')
    #ax[0,1].hist((image[1:,:] - image[:-1,:]).flatten(),bins=10*nbins,histtype='step',label='Column', color='orange')
    x = (image[:,1:] - image[:,:-1]).flatten()
    y = (image[1:,:] - image[:-1,:]).flatten()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    print(x)
    print(y)
    ax[0,1].hist(x,bins=np.linspace(np.percentile(x,5),np.percentile(x,95),nbins),histtype='step',label='Line')
    ax[1,1].hist(x,bins=np.linspace(np.percentile(x,5),np.percentile(x,95),nbins),histtype='step',label='Line')
    ax[1,1].hist(y,bins=np.linspace(np.percentile(y,5),np.percentile(y,95),nbins),histtype='step',label='Column', color='orange')
#    ax[1,1].set_xlim((-100,100))
#    ax[0,1].set_xlim((-100,100))
    #ax[1,1].set_ylim((-10,40000))
    #ax[0,1].set_ylim((-10,40000))
    #ax[1,0].set_ylabel('Histograms for Columns/Lines correlation analysis')
    for axx in ax.flatten():
        axx.grid()
    ax[0,0].legend();    ax[0,1].legend()
    ax[1,0].legend();    ax[1,1].legend()
    ax[1,0].set_xlabel('abs(CL(n) - CL(n-1))')
    ax[1,1].set_xlabel('abs(P(n) - P(n-1))')
    print('ok')
    fig.tight_layout()
    fig.suptitle(os.path.basename(path), y=1)
    plt.show()
    print('ok')
    return





        

def DS9CreateHeaderCatalog(xpapoint, files=None, filename=None, info=True, redo=False, onlystacks=False, name='', config=my_conf):
    """0.5 second per image for info
    10ms per image for header info, 50ms per Mo so 240Go-> 
    """
    from astropy.table import vstack
    print('info, extension = ', sys.argv[4], sys.argv[5])
    if sys.argv[4] == '1':
        info=True
    elif sys.argv[4] == '0':
        info = False
    extentsions = np.array(sys.argv[5].split(','),dtype=int)
    if xpapoint:
        #d = DS9(xpapoint)
#        if filename is None:
#            filename = getfilename(d)        
        if files is None:
            #if len(sys.argv) > 5: files = Charge_path_new(filename, entry_point=5)
            files =   glob.glob( sys.argv[3], recursive=True) #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

    fname = os.path.dirname(os.path.dirname(files[0]))

    paths = np.unique([os.path.dirname(f) for f in files ])
    print(files)
    if len(paths) > 1:
        t1s = [CreateCatalog_new(files, ext=extentsions)]#[CreateCatalog_new(glob.glob(path + '/*.fits')) for path in paths if (len(glob.glob(path + '/*.fits'))>0)] 
    else:
        t1s = [CreateCatalog_new(files, ext=extentsions)]

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


def CreateCatalog_new(files, ext=0, config=my_conf):
    """Create header catalog from a list of fits file
    """
    from astropy.table import Table, Column, hstack
    from astropy.io import fits
    import warnings
    warnings.simplefilter('ignore', UserWarning)
    files.sort()
    path = files[0]
    fitsimage = fits.open(path)
    config.verbose = True
    
    file_header = []
    for n in ext:
        if n < len(fitsimage):
            keys = list(dict.fromkeys( fits.getheader(files[0],n).keys()))
        
            htable = Table(names=keys,dtype=('S20,'*len(keys[:])).split(',')[:-1])
            #a.add_row()
            for i, path  in enumerate(files):
                print(i)
                try:
                    header = fits.getheader(path,n)
                    keys_ = list(dict.fromkeys( header.keys()))
                    values = [header[key] for key in keys_]
                    htable.add_row()
                    for key in keys_:
                        if key in keys:
                            try:
                                htable[key][i] = float(header[key])#;print('File %i entered'%(i))
                            except (ValueError,TypeError) as e:
                                print(key, header[key])
                                print(e)
                                try:
                                    htable[key][i] = header[key]#;print('File %i entered'%(i))
                                except ValueError as e:
                                    print(e)
                                    print(key, values)
                except IndexError:
                    pass
            file_header.append(htable)
    table_header = hstack(file_header) 
    table_header.add_column(Column(np.arange(len(table_header)),name='INDEX'), index=0, rename_duplicate=True)
    table_header.add_column(Column(files,name='PATH'), index=1, rename_duplicate=True)
    table_header.add_column(Column([os.path.basename(file) for file in files]),name='FILENAME', index=2, rename_duplicate=True)
    table_header.add_column(Column([os.path.basename(os.path.dirname(file)) for file in files]),name='DIRNAME', index=3, rename_duplicate=True)

    #import IPython; IPython.embed()
    csvwrite(hstack(table_header),os.path.dirname(path) + '/HeaderCatalog.csv')
#    try:
#
#        
#        file = open(os.path.dirname(path) + '/Info.log','w') 
#        file.write('##########################################################################\n') 
#        file.write('\nNumber of images : %i'%(len(t))) 
#        file.write('\nGains: ' + '-'.join([gain for gain in np.unique(t[my_conf.gain[0]])]))
#        file.write('\nExposures: ' + '-'.join([exp for exp in np.unique(t[my_conf.exptime[0]])]))
#        file.write('\nTemps: ' + '-'.join([te for te in np.unique(t[my_conf.temperature[0]])]))
#        try:
#            file.write('\nTemps emCCD: ' + ','.join([np.str(te) for te in np.unique(np.round(np.array(t['EMCCDBAC'],dtype=float),1))]))
#        except KeyError:
#            pass
#        file.write('\nNumber of images per gain and exposure: %0.2f'%(len(t[(t[my_conf.gain[0]]==t[my_conf.gain[0]][0]) & (t[my_conf.exptime[0]]==t[my_conf.exptime[0]][0])])) )
#        file.write('\n\n##########################################################################')      
#        file.close() 
#
#    except:
#         pass
    return table_header



def CreateCatalogInfo(t1, verbose=False, config=my_conf, write_header=True):
    """Adding imformaton to a header catalog 
    """
    from astropy.table import Table
    from astropy.io import fits
    from astropy.table import hstack
    from tqdm import tqdm
    files = t1['PATH']
    path = files[0]
    fields = ['OverscannRight', 'OverscannLeft', 'Gtot_var_int_w_OS',
              'TopImage', 'BottomImage', 'MeanADUValue', 'SaturatedPixels','MeanFlux','OS_SMEARING1','OS_SMEARING2',
              'stdXY', 'stdY', 'stdX', 'MeanADUValueTR', 'MeanADUValueBR', 'MeanADUValueBL', 'MeanADUValueTL','Gtot_var_int_wo_OS',
              'Smearing_coeff_phys','GainFactorVarIntens','GainFactorHist','BrightSpotFlux','Top2BottomDiff_OSL','Top2BottomDiff_OSR',
              'Col2ColDiff', 'Line2lineDiff',  'Col2ColDiff_OSR', 'Line2lineDiff_OSR', 'ReadNoise','emGainHist']
    #t = Table(names=fields,dtype=('S20,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])
    t = Table(names=fields,dtype=('float,'*len(fields)).split(',')[:-1])#,dtype=('S10,'*30).split(',')[:-1])

    Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1#config.physical_region
    for i in tqdm(range(len(files))):
        file = files[i]
        t.add_row()
        fitsimage = fits.open(file)
        data = fitsimage[FitsExt(fitsimage)].data
        lx, ly = data.shape
#        try:
#            texp = fits.getheader(file)[my_conf.exptime[0]]
#        except KeyError:
#            texp = fits.getheader(file)[my_conf.exptime[1]]
        column = np.nanmean(data[Yinf:Ysup,Xinf:Xsup], axis=1)
        line = np.nanmean(data[Yinf:Ysup,Xinf:Xsup], axis=0)
        #offset = 20
        #OSR1 = [offset,-offset,offset,400]
        #OSR2 = [offset,-offset,2200,2400]
        #OSR = data[OSR2[0]:OSR2[1],OSR2[2]:OSR2[3]]
        #OSL = data[OSR1[0]:OSR1[1],OSR1[2]:OSR1[3]]
        t[i]['Col2ColDiff'] =  np.nanmedian(line[::2]) - np.nanmedian(line[1::2])#np.nanmedian(abs(line[1:] - line[:-1])) np.nanmedian(a[::2])
#        t[i]['Col2ColDiff_OSR'] =   np.nanmedian(np.nanmean(OSR,axis=0)[::2]) - np.nanmedian(np.nanmean(OSR,axis=0)[1::2])#np.nanmedian(abs(line[1:] - line[:-1])) 
        t[i]['Line2lineDiff'] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])#np.nanmedian(abs(column[1:] - column[:-1])) 
#        t[i]['Line2lineDiff_OSR'] =np.nanmedian(np.nanmean(OSR,axis=1)[::2]) - np.nanmedian(np.nanmean(OSR,axis=1)[1::2]) #np.nanmedian(abs(column[1:] - column[:-1])) 
#        t[i]['OverscannRight'] = np.nanmean(OSR)
#        t[i]['OverscannLeft'] = np.nanmean(OSL)
        t[i]['TopImage'] = np.nanmean(column[:20])
        t[i]['BottomImage'] = np.nanmean(column[-20:])
#        t[i]['Top2BottomDiff_OSL'] = np.nanmean(OSL[:20,:]) - np.nanmean(OSL[-20:,:])
#        t[i]['Top2BottomDiff_OSR'] = np.nanmean(OSR[:20,:]) - np.nanmean(OSR[-20:,:])
#        t[i]['MeanFlux'] =  t[i]['MeanADUValue']/texp
#        t[i]['MeanADUValueTR'] =  np.nanmean((data - ComputeOSlevel1(data))[1000:1950,1600:2100])
#        t[i]['MeanADUValueBR'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1000,1600:2100])
#        t[i]['MeanADUValueBL'] =  np.nanmean((data - ComputeOSlevel1(data))[2:1000,1100:1600])
#        t[i]['MeanADUValueTL'] =  np.nanmean((data - ComputeOSlevel1(data))[1000:1950,1100:1600])
        t[i]['SaturatedPixels'] = 100*float(np.sum(data[Yinf:Ysup,Xinf:Xsup]>2**16-10)) / np.sum(data[Yinf:Ysup,Xinf:Xsup]>0)
        t[i]['stdXY'] = np.nanstd(data[Yinf:Ysup,Xinf:Xsup])
#        t[i]['BrightSpotFlux'] = (np.nanmean(data[1158-25:1158+25,2071-50:2071+50]) - np.nanmean(data[1158-25+50:1158+25+50,2071-50:2071+50]))/texp
        emgain = ComputeEmGain(file, None,True,False,None,None,[40,40],False,[1053,2133,500,2000])
        
        
        
        
        t[i]['Gtot_var_int_w_OS'],t[i]['Gtot_var_int_wo_OS'] = emgain['EMG_var_int_w_OS'], emgain['EMG_var_int_wo_OS']
#The 4 are to be added
#        t[i]['Smearing_coeff_phys'] = SmearingProfileAutocorr(file,None,DS9_BackUp_path,'',False,'x')['Exp_coeff']
#        t[i]['GainFactorVarIntens'] = 1/Smearing2Noise(t[i]['Smearing_coeff_phys'])['Var_smear']
#        t[i]['GainFactorHist'] =  1/Smearing2Noise(t[i]['Smearing_coeff_phys'])['Hist_smear']
#        t[i]['MeanADUValue'] =  np.nanmean((data - ComputeOSlevel1(data))[Yinf:Ysup,Xinf:Xsup])
#        t[i]['ReadNoise'] = ComputeReadNoise(path=None, fitsimage=fitsimage, Plot=False)['RON']
#        t[i]['emGainHist'] = ComputeGainHistogram(path=[file], Plot=False)


        try:
            t[i]['stdX'] = np.nanstd(data[int(Yinf + (Ysup - Yinf)/2),Xinf:Xsup])
            t[i]['stdY'] = np.nanstd(data[Yinf:Ysup,int(Xinf + (Xsup - Xinf)/2)])
        except IndexError:
            t[i]['stdX'] = np.nanstd(data[int(lx/2),:])
            t[i]['stdY'] = np.nanstd(data[:,int(ly/2)])            
#        if write_header:
#            for field in fields:
#                try:
#                    print( t[i][field])
#                    fits.setval(file, field, value = t[i][field],overwrite=True)
#                except ValueError as e:
#                    print(e)
#                    fits.setval(file, field, value = '%s'%(t[i][field]),overwrite=True)
#                except KeyError as e:
#                    print(e)
#                    fits.setval(file, field, value = 'NaN',overwrite=True)
#                    
            
    new_cat = hstack((t1,t),join_type='inner')
    print(new_cat.colnames)
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
#            from tkinter import messagebox
#            messagebox.showwarning( "Header error","At least one image here is not corresponding to its header")     
            d = DS9();d.set('analysis message yesno {At least one image here is not corresponding to its header}')
        except:
            pass
    return new_cat    




#
#def PlotSpatial2(filename, field, save=True, plot_flag=True, DS9backUp = DS9_BackUp_path):
#    """
#    Without mask centered on 206nm
#    """
#    import matplotlib; matplotlib.use('TkAgg')  
#    import matplotlib.pyplot as plt
#    from astropy.io import fits
#    from scipy.optimize import curve_fit
#    #print('Entry = ', field)
#    fitsfile = fits.open(filename)#d.get_pyfits()#fits.open(filename)
#    image = fitsfile[0].data    
#    x, y, redshift, slit, mag, w = returnXY(field,keyword=None, frame='observedframe')  
#    new_im = image[0:1870,1070:2100]
#    #if plot_flag:
#    offset = 5
#    #stdd = np.nanstd(new_im,axis=1)[offset:-offset]
#    new_im = np.convolve(np.nanmean(new_im,axis=1), np.ones(3)/3, mode='same')[offset:-offset]
#    #new_im = np.nanmean(new_im,axis=1)[5:-5]
#    n=2*10
#    obj = detectLine2(np.arange(len(new_im)),new_im)
#    yy = [new_im[int(xi)] for xi in x]
#    if len(yy)>0:
#        obj.ax.hlines(yy, x - 10, x + 10,label='Slits position', colors='black')
#        for i, sliti in enumerate(slit):
#            if i% 4 == 1:
#                n =  1
#            elif i% 4 == 2:
#                n = + 0.5
#            elif i% 4 == 3:
#                n =  -1
#            elif i% 4 == 0:
#                n = -0.5
#            obj.ax.text(x[i], yy[i] + 20.3  * n,str(sliti),bbox=dict(facecolor='red', alpha=0.1),fontsize=8)
#            obj.ax.vlines(x[i], yy[i] + 20.3 * n, yy[i], linestyles='dotted', colors='red', alpha=0.3)
#    
#
#    plt.show()
#    return




def DS9replaceNaNs(xpapoint):
    """Replace the pixels in the selected regions in DS9 by NaN values
    """
   # from astropy.io import fits
    d = DS9(xpapoint)#DS9(xpapoint)
    filename = getfilename(d)#d.get("file")
    regions = getregion(d, all=True)
    fitsimage = d.get_pyfits()[0]#fits.open(filename)[0]
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


def DS9ExtractSources(xpapoint):
    """Extract sources for DS9 image and create catalog
    """
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

           
    return


def delete_doublons(sources, dist):
    """Function that delete doublons detected in a table, 
    the initial table and the minimal distance must be specifies
    """
    try:
        sources['doublons'] = 0
        for i in range(len(sources)):
            a = distance(sources[sources['doublons']==0]['xcentroid'],sources[sources['doublons']==0]['ycentroid'],sources['xcentroid'][i],sources['ycentroid'][i]) > dist
            a = list(1*a)
            a.remove(0)
            if np.nanmean(a)<1:
                sources['doublons'][i]=1
        return sources[sources['doublons']==0]
    except TypeError:
        print('no source detected')
#        quit()
        
        
def ExtractSources(filename, fwhm=5, threshold=8, theta=0, ratio=1, n=2, sigma=3, iters=5, deleteDoublons=3):
    """Extract sources for DS9 image and create catalog
    """
    from astropy.io import fits
    from scipy import ndimage
    from astropy.table import Table
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    fitsfile = fits.open(filename)
    sizes = [sys.getsizeof(elem.data) for elem in fitsfile]#[1]
    data = fitsfile[np.argmax(sizes)].data
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


def getfilename(ds9, config=my_conf, All=False):
    """Get the filename of the loaded image in DS9
    """
    if not All:
        backup_path = os.environ['HOME'] + '/DS9BackUp'
        if not os.path.exists(os.path.dirname(backup_path)):
            os.makedirs(os.path.dirname(backup_path))
        filename = ds9.get('file')
        
    #    if '[IMAGE]' in os.path.basename(filename):
    #        filename = filename.split('[IMAGE]')[0]
        if os.path.basename(filename)[-1] == ']':
            filename = filename.split('[')[0]
        if len(filename)==0:
            new_filename = filename
        elif filename[0] == '.':
            new_filename = backup_path + '/BackUps' + filename[1:]
            print('Filename in the DS9 backup repository, changing path to %s'%(new_filename))
        else:
            new_filename = filename
    if All:
        new_filename = [] 
        print('Taking images opened in DS9')
        current = ds9.get('frame')
        ds9.set('frame last')
        n2 = int(ds9.get('frame'))
        ds9.set('frame first')
        n1 = int(ds9.get('frame'))
        #new_filename.append(ds9.get('file'))
        for i in np.arange(n1,n2+1):
            ds9.set('frame %i'%(i))
            file = ds9.get('file')
            if os.path.isfile(file):
                new_filename.append(file)  
        ds9.set('frame ' + current)
    print(new_filename)
    #print('Simlink = ',os.path.islink(new_filename))
    return new_filename    



    


def AddHeaderField(xpapoint, field='', value='', comment='-'):
    """Add header fild to the loaded DS9 image
    """
    from astropy.io import fits
    d = DS9(xpapoint)
    filename = getfilename(d)
  
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

    path = Charge_path_new(filename) if len(sys.argv) > 6 else [filename]
    
    for filename in path: 
        print(filename)
        header = fits.getheader(filename)
        if 'NAXIS3' in header:
            print('2D array: Removing NAXIS3 from header...')
            fits.delval(filename,'NAXIS3')
        fits.setval(filename, field[:8], value = value, comment = comment)

    if len(path)<2:
        d.set('frame clear')
        d.set('file '+ filename)
    return



    

#Background2D(mask, (50,50), filter_size=(3,3),sigma_clip=sigma_clip, bkg_estimator=BiweightLocationBackground(),exclude_percentile=20)


def BackgroundEstimationPhot(filename,  sigma, bckd, rms, filters, boxs,n=2, DS9backUp = DS9_BackUp_path,snr = 3,npixels = 15,dilate_size = 3,exclude_percentile = 5, mask=False, Plot=True):
    """Estimate backgound in a fits image
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
  
    if len(masks)==1:
        fig = plt.figure()
        plt.suptitle('bckd=%s, sigma=%s, snr=%s, npixels=%s, dilate_size=%s, exclude_percentile=%s  box=%s-  std = %0.8f'%(bckd, sigma, snr, npixels, dilate_size, exclude_percentile,boxs,np.nanstd(fitsfile.data) ),y=1)
        plt.subplot(221)
        plt.title('Data and background')
        plt.plot(np.nanmean(data[0:-1,a:b],axis=0))
        plt.plot(np.nanmean(bkg.background[0:-1,a:b]+diff,axis=0))
        plt.subplot(222)
        plt.plot(np.nanmean(fitsfile.data[0:-1,a:b],axis=0))
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
#    if Plot:
#        plt.show()
    return fitsfile, name






    

        

def CreateImageFromCatalogObject(xpapoint, nb = int(1e3), path=''):
    """Create galaxy image form a sextractor catalog
    """
    import astropy
    from astropy.table import Table
    from tqdm import tqdm
    from astropy.modeling.functional_models import Gaussian2D
    from photutils.datasets import make_100gaussians_image

    d = DS9(xpapoint)
    #if len(sys.argv)>3:
    path = sys.argv[-1]
    if (os.path.isfile(path)):
        print('Opening sextractor catalog')
        catfile =   path #law = 'standard_exponential'
        try:
            catalog = Table.read(catfile)
        except  astropy.io.registry.IORegistryError:
            catalog = Table.read(catfile, format='ascii')
            
#    else:
#        x, y, angle = np.random.randint(lx, size=nb), np.random.randint(ly, size=nb), np.random.rand(nb)*2*np.pi - np.pi
#        peak = np.random.exponential(10, size=nb)
#        sigmax, sigmay = np.random.normal(3, 2, size=nb), np.random.normal(3, 2, size=nb)
#        catalog = Table([x, y, peak, sigmax, sigmay, angle],  names=('x_mean', 'y_mean', 'amplitude','x_stddev','y_stddev', 'theta'))
#        
        lx, ly = int(catalog['X_IMAGE'].max()), int(catalog['Y_IMAGE'].max())
        background = np.median(catalog['BACKGROUND'])
        image = np.ones((lx,ly)) * background
        for i in tqdm(range(len(catalog))):
            x = np.linspace(0,lx-1,lx)
            y = np.linspace(0,ly-1,ly)
            x, y = np.meshgrid(x,y)
            try:
                #image += Gaussian2D.evaluate(x, y, catalog[i]['peak'], catalog[i]['xcenter'], catalog[i]['ycenter'], catalog[i]['sigmax'],  catalog[i]['sigmay'], catalog[i]['angle'])
#                image += Gaussian2D.evaluate(x, y, catalog[i]['MAG_AUTO'], catalog[i]['X_IMAGE'], catalog[i]['Y_IMAGE'], catalog[i]['KRON_RADIUS'] * catalog[i]['A_IMAGE'],  catalog[i]['KRON_RADIUS'] * catalog[i]['B_IMAGE'], catalog[i]['THETA_IMAGE']).T
                image += Gaussian2D.evaluate(x, y, catalog[i]['MAG_AUTO'], catalog[i]['X_IMAGE'], catalog[i]['Y_IMAGE'],  catalog[i]['A_IMAGE'],  catalog[i]['B_IMAGE'], np.pi * catalog[i]['THETA_IMAGE']/180).T
            except KeyError:
                image += Gaussian2D.evaluate(x, y, catalog[i]['amplitude'], catalog[i]['x_mean'], catalog[i]['y_mean'], catalog[i]['x_stddev'],  catalog[i]['y_stddev'], catalog[i]['theta'])
        image_real = np.random.poisson(image).T
    else:
        print('No catalog given, creating new image.')
        image_real = make_100gaussians_image()
    name = '/tmp/image_%s.fits'%(datetime.datetime.now().strftime("%y%m%d-%HH%MM%S"))
    fitswrite(image_real,name)
    d.set('frame new')
    d.set('file ' + name )
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



def BackgroundFit1D(xpapoint, config=my_conf, exp=False, double_exp=False, Type='Linear'):
    """Fit background 1d with different features
    """
    from .dataphile.demos import auto_gui

    d = DS9(xpapoint)
    axis, background, function, nb_gaussians,  kernel, Type = sys.argv[-6:]
    print('axis, function, nb_gaussians = ',axis, function, nb_gaussians)
    try:
        region = getregion(d, quick=True)
    except ValueError:
        image_area = my_conf.physical_region#[1500,2000,1500,2000]
        Yinf, Ysup,Xinf, Xsup = image_area
    else:
        Yinf, Ysup, Xinf, Xsup = Lims_from_region(None,coords=region)#[131,1973,2212,2562]
        image_area = [Yinf, Ysup,Xinf, Xsup]
        print(Yinf, Ysup,Xinf, Xsup)    
    data = d.get_pyfits()[0].data[Xinf: Xsup,Yinf: Ysup]
    if axis=='y':
        y = np.nanmean(data,axis=1);x = np.arange(len(y))
        index = np.isfinite(y)
        x, y = x[index], y[index]
        if np.nanmean(y[-10:])>np.nanmean(y[:10]):
            y = y[::-1]
    else:
        y = np.nanmean(data,axis=0);x = np.arange(len(y))
        index = np.isfinite(y)
        x, y = x[index], y[index]
        if np.nanmean(y[-10:])>np.nanmean(y[:10]):
            y = y[::-1]
    if  Type == 'Log':
        y = np.log10(y - np.nanmin(y))
        index = np.isfinite(y)
        x, y = x[index], y[index]
        
    kernel = int(kernel)
    y = np.convolve(y, np.ones(kernel)/kernel, mode='same')[kernel:-kernel]
    x = x[kernel:-kernel]
    #auto_gui.Background(x,y)
    if  function == 'exponential1D':
        exp=True
    if  function == 'DoubleExponiential':
        double_exp=True
    print('Using general fit new')
    auto_gui.GeneralFit_new(x,y,nb_gaussians=int(nb_gaussians), background=int(background), exp=exp, double_exp=double_exp)
    
#    if function == 'polynomial1D':
#        auto_gui.GeneralFit(x,y,nb_gaussians=int(nb_gaussians), background=background)
#    elif  function == 'both':
#        auto_gui.GeneralFit(x,y,exp=True,nb_gaussians=int(nb_gaussians), background=background)
#    elif  function == 'exponential1D':
#        auto_gui.GeneralFit(x, y, background=background, exp=True)#,nb_gaussians=int(nb_gaussians))
    plt.show()
    return



def get_depth_image(xpapoint):
    """Get the depth of an astronomical image
    """
    from .getDepth import main_coupon
    d = DS9(xpapoint)
    filename = getfilename(d)
    mag_zp, aper_size, N_aper = sys.argv[-3-5:-5]
    if mag_zp.lower() == '-':
        mag_zp = None
    else:
        mag_zp = float(mag_zp)
    aper_size = float(aper_size)
    N_aper = int(N_aper)
    paths = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    for filename in paths:
        if mag_zp is None:
            if 'HSC' in filename:
                mag_zp = 27
            if ('VIRCAM' in filename) | ('MegaCam' in filename):
                mag_zp = 30
        print(filename)
        print('Zero point magnitude =', mag_zp)
        main_coupon(fileInName=filename, fileOutName=None, ext=1, ext_seg=1, mag_zp=mag_zp, sub=None, aper_size=aper_size, verbose=True, plot=True, fileSegName=None, type="image", nomemmap=False, N_aper=N_aper)
    return


def PlotSpectraDataCube(xpapoint):
    """Plot a datacub in the other dimension a a given region
    """
    from astropy.io import fits
    from scipy import ndimage

    d = DS9(xpapoint)
    filename = getfilename(d)
    fits_im = fits.open(filename.split('[')[0])
    header = fits_im[0].header
    x_ = np.arange(1,header['NAXIS3']+1)
    try:
        lambda_ = header['CRVAL3'] + (x_ -  header['CRPIX3']) * header['CD3_3']
    except:
        lambda_ = header['CRVAL3'] + (x_ -  header['CRPIX3']) * header['NAXIS3']
        
    #x = np.linspace(header['CRVAL3'] - header['CRVAL3'] * header['CD3_3'], header['NAXIS3'])
    reg = getregion(d,all=True)
    smoothl, smoothx = np.array(sys.argv[-2:],dtype=float)
    
    spectra = ndimage.gaussian_filter(fits_im[0].data, sigma=(smoothl,smoothx,smoothx), order=0)
    plt.figure()
    for i in range(len(reg)):
        x, y= reg[i].xc, reg[i].yc
        spectra_ = spectra[:,int(x),int(y)]
        plt.plot(lambda_,spectra_,label='Selected spectra %i'%(i))
#    x_sky, y_sky = np.random.randint(x -r ), np.random.randint(y-r) 
#    sky = ndimage.gaussian_filter(fits_im[0].data, sigma=(smoothl,smoothx,smoothx), order=0)[:,x_sky, y_sky]
#    plt.plot(lambda_,sky,label='Sky background')
    #plt.plot(lambda_,spectra,label='Selected spectra')
    plt.legend()
    plt.xlabel(r'$\lambda [\AA]$')
    plt.ylabel('Flux')
    plt.title('Spectra - spectral smoothing = %0.1f - spatial smoothing = %0.1f'%(smoothl, smoothx))
    plt.show()    

    return

def StackDataDubeSpectrally(xpapoint):
    """Stack data cube in the 3d direction
    """
    from astropy.io import fits
    from scipy import ndimage
    smooth = float(sys.argv[-1])
    d = DS9(xpapoint)
    filename = getfilename(d)
    fits_im = fits.open(filename.split('[')[0])
    header = fits_im[0].header
    #x = np.linspace(header['CRVAL3'] - header['CRVAL3'] * header['CD3_3'], header['NAXIS3'])
    if ('Box' in d.get('region')) or ('Circle' in d.get('region')):
        reg = getregion(d)
        x, y, r = reg[0].xc, reg[0].yc, reg[0].r
        spectra = ndimage.gaussian_filter(fits_im[0].data, sigma=(0, smooth, smooth), order=0)[:,int(x-2*r):int(x+2*r),int(y-2*r):int(y+2*r)]
        #x_sky, y_sky = np.random.randint(x -r ), np.random.randint(y-r) 
    else:
        spectra = ndimage.gaussian_filter(fits_im[0].data, sigma=(0, smooth, smooth), order=0)[:,:,:]
    #if sky_subtraction:
    try:
        x_ = np.arange(1,header['NAXIS3']+1)
        lambda_ = header['CRVAL3'] + (x_ -  header['CRPIX3']) * header['NAXIS3']
    except KeyError:
        lambda_ = header['CRVAL3'] + (x_ -  header['CRPIX3']) * header['CD3_3']
        
    try:
        inf, sup = np.array(sys.argv[-2].split('-'),dtype=int)
    except ValueError:
        mask = np.ones(len(lambda_),dtype=bool)
    else:
        mask = (lambda_>inf) & (lambda_<sup) 
    print(mask)
    fitswrite(np.sum(spectra[mask],axis=0),'/tmp/stacked.fits')
    d.set('frame new')
    d.set('tile yes')
    d.set('file /tmp/stacked.fits')
#    d.set_np2arr(spectra[mask])

#    plt.figure(figsize=(8,8))
#    plt.imshow(np.sum(spectra[mask],axis=0))
#    plt.axis('equal')
#    theta = np.arange(361)*(np.pi/180.)
#    plt.plot(y+r*np.sin(theta), x+r*np.cos(theta), c = 'white')
#    plt.colorbar()
#    plt.show()
    return

def RunSextractor(xpapoint, filename=None, detector=None, path=None):
    """Run sextraxtor software
    """
    import astropy
    d = DS9(xpapoint)
    filename = getfilename(d)
    dn = os.path.dirname(filename)
    fn = os.path.basename(filename)

    from shutil import which
    if which('sex') is None:
#        from tkinter import messagebox
#        messagebox.showwarning( title = 'Sextractor error', message="""Sextractor do not seem to be installed in you machine. If you know it is, please add the sextractor executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes.""")     
        d = DS9();d.set('analysis message {Sextractor do not seem to be installed in you machine. If you know it is, please add the sextractor executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes}')
    params = np.array(sys.argv[-34:-1], dtype='<U256')#str)
    print(params)
    DETECTION_IMAGE = sys.argv[-35]
    CATALOG_NAME, CATALOG_TYPE,  PARAMETERS_NAME,  DETECT_TYPE,  DETECT_MINAREA = params[:5]
    THRESH_TYPE,  DETECT_THRESH,  ANALYSIS_THRESH,  FILTER,  FILTER_NAME,  DEBLEND_NTHRESH = params[5:5+6]
    DEBLEND_MINCONT,  CLEAN,  CLEAN_PARAM,  MASK_TYPE,  WEIGHT_TYPE, WEIGHT_IMAGE,  PHOT_APERTURES = params[5+6:5+6+7]
    PHOT_AUTOPARAMS,  PHOT_PETROPARAMS,  PHOT_FLUXFRAC,  MAG_ZEROPOINT,  PIXEL_SCALE,  SEEING_FWHM = params[5+6+7:5+6+7+6]
    STARNNW_NAME,  BACK_TYPE,  BACK_SIZE,  BACK_FILTERSIZE,  BACKPHOTO_TYPE,  BACKPHOTO_THICK = params[5+6+7+6:5+6+7+6+6]
    BACK_FILTTHRESH,  CHECKIMAGE_TYPE, CHECKIMAGE_NAME = params[-3:]

    try:
        param_dir = resource_filename('pyds9plugin', 'Sextractor')
    except:
        param_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Sextractor')


    
    param_names =  ['CATALOG_NAME', 'CATALOG_TYPE',  'PARAMETERS_NAME',  'DETECT_TYPE',  'DETECT_MINAREA' , 'THRESH_TYPE',  'DETECT_THRESH',  'ANALYSIS_THRESH',  'FILTER',  'FILTER_NAME',  'DEBLEND_NTHRESH', 'DEBLEND_MINCONT',  'CLEAN',  'CLEAN_PARAM',  'MASK_TYPE',  'WEIGHT_TYPE', 'WEIGHT_IMAGE',  'PHOT_APERTURES','PHOT_AUTOPARAMS',  'PHOT_PETROPARAMS',  'PHOT_FLUXFRAC',  'MAG_ZEROPOINT',  'PIXEL_SCALE',  'SEEING_FWHM', 'STARNNW_NAME',  'BACK_TYPE',  'BACK_SIZE',  'BACK_FILTERSIZE',  'BACKPHOTO_TYPE',  'BACKPHOTO_THICK','BACK_FILTTHRESH',  'CHECKIMAGE_TYPE', 'CHECKIMAGE_NAME']
    params[0]= filename[:-5] + '_sex.fits' if  (params[0]=='-') or (params[0]=='.') else params[0]
    print(filename,params[0],filename + '_sex.fits')
    params[8]='Y' if  params[8]=='1' else 'N'
    params[12]='Y' if params[12]=='1' else 'N'
    if not os.path.exists(os.path.join(dn,'DetectionImages','PhotometricCatalog')):
        os.makedirs(os.path.join(dn,'DetectionImages','PhotometricCatalog'))     

    if CATALOG_NAME=='-':
        CATALOG_NAME = os.path.join(dn,'DetectionImages','PhotometricCatalog',fn[:-5].replace(',','-')+'_cat.fits')
        params[0] = CATALOG_NAME
    params[2] = os.path.join(param_dir,params[2])
    params[9] = os.path.join(param_dir,params[9])
    params[24] = os.path.join(param_dir,params[24])

    if DETECTION_IMAGE == '-':
        DETECTION_IMAGE = None
    else:
        DETECTION_IMAGE = ',' + DETECTION_IMAGE
    print(bcolors.BLACK_RED +'Image used for detection  = ' + str(DETECTION_IMAGE) + bcolors.END)
    print(bcolors.BLACK_RED + 'Image used for photometry  = '+ str(filename) + bcolors.END)
    
    print(bcolors.GREEN_WHITE + """
          ********************************************************************
                                     Parameters sextractor:
          ********************************************************************"""+ bcolors.END)
    print(bcolors.BLACK_RED + '\n'.join([name + ' = ' + str(value) for name, value in zip(param_names, params)]) + bcolors.END)
    os.system('sex -d > default.sex')

    if DETECTION_IMAGE is not None:
        print('sex ' + filename + DETECTION_IMAGE + ' -c  default.sex -' + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params)]))
        os.system('sex ' + filename + DETECTION_IMAGE + ' -c  default.sex -' + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params)]))
    else:   
        print('sex ' + filename + ' -c  default.sex -' + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params)])) 
        os.system('sex ' + filename + ' -c  default.sex -' + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params)]))

#    if os.path.isfile(CATALOG_NAME):
#        DS9Catalog2Region(xpapoint, name=CATALOG_NAME, x='X_IMAGE', y='Y_IMAGE', ID='MAG_AUTO')
#    else:
#        print('Can not find the output sextractor catalog...')
    colors =  ['White','Yellow','Orange']  

    if not os.path.exists(os.path.dirname(filename) + '/DetectionImages/reg/'):
        os.makedirs(os.path.dirname(filename) + '/DetectionImages/reg/')     
    if os.path.isfile(CATALOG_NAME):
        from astropy.table import vstack
        try:
            cat = Table.read(CATALOG_NAME)
        except astropy.io.registry.IORegistryError:
            cat = Table.read(CATALOG_NAME, format='ascii')
            
        cat= CleanSextractorCatalog(cat)#[:int(sys.argv[-1])]

        print(CATALOG_NAME)
        cat1 = cat[cat['MAG_AUTO']>=cat['MAG_ISO']]
        cat2 = cat[cat['MAG_AUTO']<cat['MAG_ISO']]
        cat3 = vstack((cat1,cat2))
        #x, y = [list(cat1['X_IMAGE'])+list(cat2['X_IMAGE'])], [list(cat2['X_IMAGE'])+list(cat2['X_IMAGE'])]
        #create_DS9regions([cat['X_IMAGE']],[cat['Y_IMAGE']], more=[cat['A_IMAGE']*cat['KRON_RADIUS'],cat['B_IMAGE']*cat['KRON_RADIUS'],cat['THETA_IMAGE']], form = ['ellipse']*len(cat),save=True,color = ['green']*len(cat), savename=os.path.dirname(dn) + '/DetectionImages/reg/' + fn[:-5].replace(',','-') ,ID=[np.array(cat['MAG_AUTO'],dtype=int)])
        #create_DS9regions([cat3['X_IMAGE']],[cat3['Y_IMAGE']], more=[cat3['A_IMAGE']*cat3['KRON_RADIUS'],cat3['B_IMAGE']*cat3['KRON_RADIUS'],cat3['THETA_IMAGE']], form = ['ellipse']*len(cat3),save=True,color = ['green']*len(cat1)+['red']*len(cat2), savename=os.path.dirname(filename) + '/DetectionImages/reg/' + os.path.basename(filename)[:-5].replace(',','-') ,ID=[np.array(cat3['MAG_AUTO'],dtype=int)])
        create_DS9regions([cat3['X_IMAGE']],[cat3['Y_IMAGE']], more=[cat3['A_IMAGE']*cat3['KRON_RADIUS'],cat3['B_IMAGE']*cat3['KRON_RADIUS'],cat3['THETA_IMAGE']], form = ['ellipse']*len(cat3),save=True,color = [np.random.choice(colors)]*len(cat3), savename=os.path.dirname(filename) + '/DetectionImages/reg/' + os.path.basename(filename)[:-5].replace(',','-') ,ID=[np.array(cat3['MAG_AUTO'],dtype=int)])
        #DS9Catalog2Region(xpapoint, name=CATALOG_NAME, x='X_IMAGE', y='Y_IMAGE', ID='MAG_AUTO')
        if path is None:
            d.set('regions ' +os.path.dirname(filename) + '/DetectionImages/reg/' + os.path.basename(filename)[:-5].replace(',','-') + '.reg')
    else:
        print('Can not find the output sextractor catalog...')
    for file in glob.glob(os.path.dirname(filename) + '/tmp/' + os.path.basename(filename)[:-5] + '*.fits' ):
        os.remove(file)
    return
   

def DS9SWARP(xpapoint):
    """Run swarp software from DS9
    """
    from shutil import which
    params = sys.argv[-31-3:-3]
    param_names =  ['BACK_DEFAULT', 'BACK_FILTERSIZE', 'BACK_FILTTHRESH', 'BACK_SIZE', 'BACK_TYPE', 'CELESTIAL_TYPE' ,'CENTER_TYPE', 'CENTER', 'COMBINE',  'COMBINE_BUFSIZE', 
                    'COMBINE_TYPE', 'COPY_KEYWORDS' ,'FSCALASTRO_TYPE', 'FSCALE_DEFAULT', 'GAIN_DEFAULT' ,'GAIN_KEYWORD' ,'IMAGEOUT_NAME' ,'IMAGE_SIZE', 'MEM_MAX' ,'OVERSAMPLING', 
                    'PIXEL_SCALE', 'PIXELSCALE_TYPE' ,'PROJECTION_ERR', 'PROJECTION_TYPE' ,'RESAMPLE', 'RESAMPLING_TYPE', 'SUBTRACT_BACK', 'WRITE_FILEINFO' ,'VERBOSE_TYPE', 'WRITE_XML', 'XML_NAME']
    BACK_DEFAULT, BACK_FILTERSIZE, BACK_FILTTHRESH, BACK_SIZE, BACK_TYPE, CELESTIAL_TYPE, CENTER_TYPE, CENTERI, COMBINEI,    COMBINE_BUFSIZE, COMBINE_TYPE, COPY_KEYWORDS, FSCALASTRO_TYPE, FSCALE_DEFAULT, GAIN_DEFAULT, GAIN_KEYWORD, IMAGEOUT_NAME, IMAGE_SIZE, MEM_MAX, OVERSAMPLING, PIXEL_SCALE, PIXELSCALE_TYPE,PROJECTION_ERR, PROJECTION_TYPE, RESAMPLE, RESAMPLING_TYPE, SUBTRACT_BACK, WRITE_FILEINFO, VERBOSE_TYPE, WRITE_XML, XML_NAME = params
    

    d = DS9(xpapoint)
#    filename = getfilename(d)
#    paths = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    print(sys.argv[3])
    if sys.argv[3] != '-':
        paths = glob.glob(sys.argv[3])
    else:
        paths = getfilename(d, All=True)

            
    

    params[19] = os.path.join(os.path.dirname(paths[0]),params[19])
    if which('swarp') is None:
#        from tkinter import messagebox
#        messagebox.showwarning( title = 'SWARP error', message="""SWARP do not seem to be installedin you machine. If you know it is, please add the SWARP executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes.""")     
        d = DS9();d.set('analysis message {WARP do not seem to be installed in you machine. If you know it is, please add the sextractor executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes}')

    else:
        os.chdir(os.path.dirname(paths[0]))
        os.system("sleep 0.1")
        os.system("swarp -d >default.swarp")
        #print("swarp %s  -IMAGEOUT_NAME %s -c default.swarp "%(' '.join(paths), os.path.join(os.path.dirname(paths[0]), 'coadd.fits') ) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
        print("swarp %s  -c default.swarp -"%(' '.join(paths)) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
        os.system("swarp %s  -c default.swarp -"%(' '.join(paths)) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
        #os.system("swarp %s  -IMAGEOUT_NAME %s -c default.swarp "%(' '.join(paths), os.path.join(os.path.dirname(paths[0]), 'coadd.fits') ) )#+ ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
#        os.system("swarp %s  -c default.swarp "%(' '.join(paths))+ ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) + ' -IMAGEOUT_NAME %s'%(os.path.join(os.path.dirname(paths[0]), 'coadd.fits')))
        os.system("rm default.swarp")
    d.set('frame new')
    d.set('tile no')
    d.set('file %s'%(os.path.join(os.path.dirname(paths[0]),'coadd.fits')))
    return




def DS9PSFEX(xpapoint):
    """Run PSFex software form DS9
    """
    from astropy.table import Table
    from shutil import which
    from astropy.io import fits
   
    params = sys.argv[-30:]
    param_names =  ['ENTRY_PATH' ,  'BASIS_TYPE' ,  'BASIS_NUMBER' ,  'PSF_SAMPLING' ,  'PSF_ACCURACY' ,  'PSF_SIZE' ,  'CENTER_KEYS' ,  'PHOTFLUX_KEY' ,  'PHOTFLUXERR_KEY' ,  
                    'PSFVAR_KEYS' ,  'PSFVAR_GROUPS' ,  'PSFVAR_DEGREES', 'SAMPLE_AUTOSELECT' ,  'SAMPLEVAR_TYPE' ,  'SAMPLE_FWHMRANGE' ,  'SAMPLE_VARIABILITY' ,  'SAMPLE_MINSN' ,  
                    'SAMPLE_MAXELLIP' ,  'CHECKPLOT_DEV' ,  'CHECKPLOT_TYPE' ,  'CHECKPLOT_NAME', 'CHECKIMAGE_TYPE' ,  'CHECKIMAGE_NAME' ,  'PSF_DIR' ,'HOMOBASIS_TYPE',  'HOMOPSF_PARAMS', 'VERBOSE_TYPE' ,  'WRITE_XML' ,  'XML_NAME' ,  'NTHREADS']
    param_dict = {}
    for key, val in zip(param_names, params):
        param_dict[key] = val
        print(bcolors.BLACK_RED + '%s : %s'%(key,param_dict[key]) +  bcolors.END )
    #print(param_dict)
    

    d = DS9(xpapoint)
#    filename = getfilename(d)
#    paths = Charge_path_new(filename) if len(sys.argv) > 3 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.aargv[-5:]))
    param_dict['ENTRY_PATH'] = glob.glob(param_dict['ENTRY_PATH'])
    #print(param_dict['ENTRY_PATH'],paths)
    new_list = []
    if type(param_dict['ENTRY_PATH']) is str:
        param_dict['ENTRY_PATH'] = [param_dict['ENTRY_PATH']]

    for path in param_dict['ENTRY_PATH']:
        a = Table.read(path, format="fits", hdu='LDAC_OBJECTS')
        hdu1 = fits.open(path)
        mask = (a['CLASS_STAR']>0.97) & (a['MAG_AUTO']>19) & (a['MAG_AUTO']<24) #& ((a['MU_MAX']-a['MAG_AUTO'])<-0.3)
#        threshold = FindCompactnessThreshold(a[mask])
        mask = np.ones(len(a),dtype=bool)#(a['CLASS_STAR']>0.97) & (a['MAG_AUTO']>19) & (a['MAG_AUTO']<24) & ((a['MU_MAX']-a['MAG_AUTO'])<threshold+0.2) & ((a['MU_MAX']-a['MAG_AUTO'])>threshold-0.3)
        #(a['MAG_AUTO']<24)&((a['MU_MAX']-a['MAG_AUTO'])<0.1)&((a['MU_MAX']-a['MAG_AUTO'])>-0.1)
        print('Number of objects : %i => %i'%(len(a),len(a[mask])))
        plt.scatter(a['MAG_AUTO'],a['MU_MAX']-a['MAG_AUTO'],s=2, label='All objects')
        #plt.hlines(threshold,-15, 30, linewidth=1, linestyle='dotted')
        plt.scatter(a[mask]['MAG_AUTO'],a[mask]['MU_MAX']-a[mask]['MAG_AUTO'],s=2, label='Star selections')
        plt.title('Number of objects : %i => %i'%(len(a),len(a[mask])))
        plt.legend()
        plt.xlabel('MAG_AUTO')
        plt.ylabel('MU_MAX - MAG_AUTO')
        plt.ylim((-5,5));plt.xlim((15,30))
        plt.savefig(path[:-5])
        if len(param_dict['ENTRY_PATH'])>1:
            plt.close()
        else:
            plt.show()

        name = path[:-5] + '_.fits'#os.path.join('/tmp',os.path.basename(path))
        hdu1[2].data = hdu1[2].data[mask]
        hdu1.writeto(name, overwrite=True)
        hdu1.close()
        
#        hdulist = convert_table_to_ldac(a_new) 
#        hdulist.writeto(name, overwrite=True) 
        #save_table_as_ldac(a_new, name, overwrite=True)
        new_list.append(name)
    paths = new_list

    if type(param_dict['ENTRY_PATH']) is str:
        param_dict['XML_NAME'] = os.path.dirname(paths[0]) + '/psfex.xml'
    
    if which('psfex') is None:
#        from tkinter import messagebox
#        messagebox.showwarning( title = 'PSFex error', message="""PSFex do not seem to be installedin you machine. If you know it is, please add the PSFex executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes.""")     
        d = DS9();d.set('analysis message {PSFex do not seem to be installed in you machine. If you know it is, please add the sextractor executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes}')
    else:
        print(os.path.dirname(paths[0]))
        print(os.getcwd())
        os.chdir(os.path.dirname(paths[0]))
        print(os.getcwd())
        os.system("sleep 0.1")
        os.system("psfex -d > default.psfex")
        print("psfex %s -c default.psfex -%s -HOMOKERNEL_SUFFIX %s_homo.fits "%(' '.join(paths),  ' -'.join([key + ' ' + str(param_dict[key]) for key in list(param_dict.keys())[1:]]) , param_dict['HOMOPSF_PARAMS'].replace(',','-') ))
        os.system("psfex %s -c default.psfex -%s -HOMOKERNEL_SUFFIX %s_homo.fits "%(' '.join(paths),  ' -'.join([key + ' ' + str(param_dict[key]) for key in list(param_dict.keys())[1:]]) , param_dict['HOMOPSF_PARAMS'].replace(',','-') ))
        os.system("rm default.psfex")

    if (xpapoint is not None) & (len(paths)==1):
        d.set('frame new')
        d.set('file ' +paths[0][:-5] + '%s_homo.fits'%(param_dict['HOMOPSF_PARAMS'].replace(',','-')))
#    for path in paths:
#        os.remove(path)
#
#    table = Table.read('/Users/Vincent/Documents/Work/sextractor/DetectionImages/Photometric_Catalogs/calexp-HSC-Y-9813-2-4_cat.xml',table_id=0)
#    table = Table.read('/Users/Vincent/Documents/Work/sextractor/DetectionImages/Photometric_Catalogs/psf_all_bands_just_for_stars.xml',table_id=0)
#    
#    x = np.linspace(0,1.8,50)
#    plt.figure()
#    plt.plot(1,1,c='black',linestyle='dotted', label='Gaussian')
#    plt.plot(1,1,c='black',linestyle='dashed', label='Moffat')
#    for cat in table:
#        p = plt.plot(x, gaus(x,1,0,cat['FWHM_WCS_Mean']/2.35,0,0),linestyle='dotted', label='_nolegend_')
#        plt.fill_between(x, gaus(x,1,0,cat['FWHM_WCS_Min']/2.35,0,0),gaus(x,1,0,cat['FWHM_WCS_Max']/2.35,0,0),alpha=0.3, color=p[0].get_color(), label='_nolegend_')
#        
#        std = cat['FWHM_WCS_Mean']/(2*np.sqrt(2**(1/cat['MoffatBeta_Mean'])-1))
#        
#        plt.plot(x, Moffat1D(x,1,std,cat['MoffatBeta_Mean']), linestyle='dashed', color=p[0].get_color(), label=' '.join(str(cat['Catalog_Name']).split('-')[1:3]) + ': FWHM = %0.2f - Beta = %0.1f'%(cat['FWHM_WCS_Mean'],cat['MoffatBeta_Mean']))
#        plt.fill_between(x, Moffat1D(x,1,std,cat['MoffatBeta_Max']),Moffat1D(x,1,std,cat['MoffatBeta_Min']),alpha=0.3, color=p[0].get_color(), label='_nolegend_')
#        
#        plt.title('%s stars'%(int(cat['NStars_Loaded_Total'])))
#    plt.legend()
#    plt.show()
    return





def Moffat1D(x, amp, std, alpha):
    """1D moffat function
    """
    return amp*np.power((1+np.square(x/std)),-alpha)


def gaus(x, a, b, sigma, lam, alpha):
    """1D gaussian centered on zero
    """
    gaus = a**2 * np.exp(-np.square(x / sigma) / 2) 
    return gaus 

def exp(x, a, b, sigma, lam, alpha):
    """1D exponential centered on zero
    """
    exp =  b**2 * np.exp(-(x/lam)**(1**2))
    return exp





def DS9saveColor(xpapoint, filename=None):
    """Fun STIFF software form DS()
    """
    d = DS9(xpapoint)
    from shutil import which
    #from astropy.io import fits
    #import matplotlib.image as mpimg
    params = sys.argv[-20:]
    param_names =  ['BINNING', 'BITS_PER_CHANNEL',  'COLOUR_SAT',  'COPY_HEADER',  'DESCRIPTION' , 'GAMMA',  'GAMMA_FAC',  'GAMMA_TYPE',  'IMAGE_TYPE',  'MAX_LEVEL',  'MAX_TYPE', 'MIN_LEVEL',  'MIN_TYPE', 'OUTFILE_NAME', 'SKY_LEVEL',  'SKY_TYPE',  'SATURATION_LEVEL']
    path1, path2, path3, BINNING, BITS_PER_CHANNEL, COLOUR_SAT, COPY_HEADER, DESCRIPTION, GAMMA, GAMMA_FAC, GAMMA_TYPE, IMAGE_TYPE, MAX_LEVEL, MAX_TYPE, MIN_LEVEL, MIN_TYPE, OUTFILE_NAME, SKY_LEVEL, SKY_TYPE, SATURATION_LEVEL = params

    d = DS9()
    d.set('frame last')
    if path1=='-':
        files = getfilename(d, All=True)
        #print('ok : ',path1,path2,path3,len(files))
        if (path1=='-') & (path2=='-') & (path3=='-') & ((len(files)==3)|(len(files)==1)):
            try:
                path1, path2, path3 = files
            except ValueError:
                path1 = files[0]
     
    if which('stiff') is None:
        d.set('analysis message {Stiff do not seem to be installed in you machine. If you know it is, please add the sextractor executable path to your $PATH variable in .bash_profile. Depending on your image, the analysis might take a few minutes}')
    else:
        os.chdir(os.path.dirname(path1))
        if (path2=='-') & (path3=='-'):
            print("stiff %s -"%(path1) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
            os.system("stiff %s -"%(path1) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
        else:
            print("stiff %s %s %s -"%(path1, path2, path3) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
            os.system("stiff %s %s %s -"%(path1, path2, path3) + ' -'.join([name + ' ' + str(value) for name, value in zip(param_names, params[3:])]) )
    d.set('frame delete all')
    d.set('rgb')
    d.set('tiff %s'%(os.path.join(os.path.dirname(path1),'stiff.tiff')))
    
    for color in ['red','green','blue']:
        d.set('rgb '+ color)
        d.set('scale minmax')#
      #  d.set('scale %s"%(scale))
        d.set('scale linear')

    return 

 

def CleanSextractorCatalog(cat):
    """Clean sextractor catalog before display
    """
    mask = (cat['A_IMAGE']<20) & (cat['B_IMAGE']<20) & (cat['MAG_AUTO']<40)
    return cat[mask]









def CosmologyCalculator(xpapoint):
    """Plot the different imnformation for a given cosmolovy and at  1 or 2 redshifts
    """
    cosmology, redshift, H0, Omega_m, Ode0, uncertainty = 'WMAP9','0.7-2', 70, 0.30, 0.7, 'H0:1'#sys.argv[-3:]
    cosmology, redshift, H0, Omega_m, Ode0, uncertainty = sys.argv[-6:]
    redshift = np.array(redshift.split('-'),dtype=float)
    if cosmology == 'w0waCDM':
        from astropy.cosmology import w0waCDM as cosmol
    elif cosmology == 'w0wzCDM':
        from astropy.cosmology import w0wzCDM as cosmol
    elif cosmology == 'wpwaCDM':
        from astropy.cosmology import wpwaCDM as cosmol
    elif cosmology == 'LambdaCDM':
        from astropy.cosmology import LambdaCDM as cosmol
    elif cosmology == 'wCDM':
        from astropy.cosmology import wCDM as cosmol
        
    if cosmology == 'WMAP9':
        from astropy.cosmology import WMAP9 as cosmo
    elif cosmology == 'WMAP7':
        from astropy.cosmology import WMAP7 as cosmo
    elif cosmology == 'WMAP5':
        from astropy.cosmology import WMAP5 as cosmo
    elif cosmology == 'Planck13':
        from astropy.cosmology import Planck13 as cosmo
    elif cosmology == 'Planck15':
        from astropy.cosmology import Planck15 as cosmo


    elif (cosmology=='wCDM') or  (cosmology=='LambdaCDM'):
        print('cosmology, redshift, H0, Omega_m, Ode0, uncertainty =', cosmology, redshift, H0, Omega_m, Ode0, uncertainty )
        H0, Omega_m, Ode0 = np.array([ H0, Omega_m, Ode0], dtype=float)
        param, uncertainty = uncertainty.split(':')
        uncertainty = float(uncertainty)
        cosmo = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0)
        print('param, uncertainty = ',param, uncertainty)
        if param.lower() == 'h0':
            cosmo1 = cosmol(H0=H0*(1-0.01*uncertainty), Om0=Omega_m, Ode0=Ode0)
            cosmo2 = cosmol(H0=H0*(1+0.01*uncertainty), Om0=Omega_m, Ode0=Ode0)
        elif param.lower() == 'om0':
            cosmo1 = cosmol(H0=H0, Om0=Omega_m*(1-0.01*uncertainty), Ode0=Ode0)
            cosmo2 = cosmol(H0=H0, Om0=Omega_m*(1+0.01*uncertainty), Ode0=Ode0)
        else:
            cosmo1 = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0*(1-0.01*uncertainty))
            cosmo2 = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0*(1+0.01*uncertainty))
    elif cosmology == 'default_cosmology':
        from astropy.cosmology import default_cosmology 
        cosmo = default_cosmology.get()
        cosmo1 = cosmo2 = cosmo       
    else:
        cosmo1 = cosmo2 = cosmo

    info = {}
    info['luminosity_distance'] = cosmo.luminosity_distance(redshift)
    info['age'] = cosmo.age(redshift) 
    info['kpc_proper_per_arcsec'] = 1/cosmo.arcsec_per_kpc_proper(redshift)#.to(u.kpc/u.arcsec)
    info['kpc_comoving_per_arcsec'] = 1/cosmo.arcsec_per_kpc_comoving(redshift)#.to(u.kpc/u.arcsec)
    info['arcsec_per_proper_kpc'] = cosmo.arcsec_per_kpc_proper(redshift)#.to(u.kpc/u.arcsec)
    info['arcsec_per_comoving_kpc'] = cosmo.arcsec_per_kpc_comoving(redshift)#.to(u.kpc/u.arcsec)
    info['angular_diameter_distance'] = cosmo.angular_diameter_distance(redshift)
    info['comoving_distance'] = cosmo.comoving_distance(redshift)
    info['comoving_volume'] = cosmo.comoving_volume(redshift)
    #info['hubble_distance'] = cosmo.hubble_distance(redshift)
    info['lookback_distance'] = cosmo.lookback_distance(redshift)
    info['lookback_time'] = cosmo.lookback_time(redshift)
    #info['nu_relative_density'] = cosmo.nu_relative_density(redshift)
    info['scale_factor'] = cosmo.scale_factor(redshift)
    #info['w'] = cosmo.w(redshift)
    info['efunc'] = cosmo.efunc(redshift)
    
    
    zs = np.linspace(0,5,50)

    if type(redshift) is float:
        redshifts = np.array([redshift],dtype=float)
    else:
        redshifts = np.array(redshift,dtype=float)

    fig, (ax1,ax2,ax3) = plt.subplots(3, 3, figsize=(18,9.5),sharex=True)
    t = 'U4'
    l = ' - '
    #p = ax1[0].plot(zs,cosmo.angular_diameter_distance(zs)/1000,label="Angular diameter distance = %0.3f"%(cosmo.angular_diameter_distance(redshift).value/1000))
    p = ax1[0].plot(zs,cosmo.angular_diameter_distance(zs)/1000,label="Angular diameter distance = %s"%(l.join(np.array(cosmo.angular_diameter_distance(redshifts).value/1000,dtype=t))))
    ax1[0].fill_between(zs,cosmo1.angular_diameter_distance(zs)/1000,cosmo2.angular_diameter_distance(zs)/1000,alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax1[0].set_ylabel('Gpc')
    ax1[0].legend()   
    for redshift in redshifts:    
        ax1[0].hlines((cosmo.angular_diameter_distance(redshift)/1000).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax1[0].vlines(redshift,0,(cosmo.angular_diameter_distance(redshift)/1000).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        
    p = ax1[0].plot(zs,cosmo.comoving_distance(zs)/1000,label="Comoving distance = %s"%(l.join(np.array(cosmo.comoving_distance(redshifts).value/1000,dtype=t))))
    ax1[0].fill_between(zs,cosmo1.comoving_distance(zs)/1000,cosmo2.comoving_distance(zs)/1000,alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    for redshift in redshifts:    
        ax1[0].hlines((cosmo.comoving_distance(redshift)/1000).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax1[0].vlines(redshift,0,(cosmo.comoving_distance(redshift)/1000).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')


    p = ax1[1].plot(zs,cosmo.luminosity_distance(zs)/1000,label="Luminosity distance = %s"%(l.join(np.array(cosmo.luminosity_distance(redshifts).value/1000,dtype=t))))
    ax1[1].fill_between(zs,cosmo1.luminosity_distance(zs)/1000,cosmo2.luminosity_distance(zs)/1000,alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax1[1].set_ylabel('Gpc')
    ax1[1].legend()  
    for redshift in redshifts:    
        ax1[1].hlines((cosmo.luminosity_distance(redshift)/1000).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax1[1].vlines(redshift,0,(cosmo.luminosity_distance(redshift)/1000).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
 

    p = ax1[2].plot(zs,cosmo.critical_density(zs)/1e-29,label="Critical density = %s"%(l.join(np.array(cosmo.critical_density(redshifts).value/1e-29,dtype=t))))
    ax1[2].fill_between(zs,cosmo1.critical_density(zs)/1e-29,cosmo2.critical_density(zs)/1e-29,alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax1[2].set_ylabel('10e-29 g/cm^3')
    ax1[2].legend() 
    for redshift in redshifts:    
        ax1[2].hlines((cosmo.critical_density(redshift)/1e-29).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax1[2].vlines(redshift,0,(cosmo.critical_density(redshift)/1e-29).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
  

    p = ax2[0].plot(zs,cosmo.comoving_volume(zs)/1e9,label="Comoving volume = %s"%(l.join(np.array(cosmo.comoving_volume(redshifts).value/1e9,dtype=t))))
    ax2[0].fill_between(zs,cosmo1.comoving_volume(zs)/1e9,cosmo2.comoving_volume(zs)/1e9,alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax2[0].set_ylabel('Gpc^3')
    ax2[0].legend() 
    for redshift in redshifts:    
        ax2[0].hlines((cosmo.comoving_volume(redshift)/1e9).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax2[0].vlines(redshift,0,(cosmo.comoving_volume(redshift)/1e9).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')

    p = ax2[1].plot(zs,cosmo.lookback_time(zs),label="Lookback time = %s"%(l.join(np.array(cosmo.lookback_time(redshifts).value,dtype=t))))
    ax2[1].fill_between(zs,cosmo1.lookback_time(zs),cosmo2.lookback_time(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    for redshift in redshifts:    
        ax2[1].hlines((cosmo.lookback_time(redshift)).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax2[1].vlines(redshift,0,(cosmo.lookback_time(redshift)).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')

    p = ax2[1].plot(zs,cosmo.age(zs),label="age = %s"%(l.join(np.array(cosmo.age(redshifts).value,dtype=t))))
    ax2[1].fill_between(zs,cosmo1.age(zs),cosmo2.age(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax2[1].legend()     
    ax2[1].set_ylabel('Gyr')
    for redshift in redshifts:        
        ax2[1].hlines((cosmo.age(redshift)).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax2[1].vlines(redshift,0,(cosmo.age(redshift)).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')

    p = ax2[2].plot(zs,cosmo.distmod(zs),label="Dist mod (mu) = %s"%(l.join(np.array(cosmo.distmod(redshifts).value,dtype=t))))
    ax2[2].fill_between(zs,cosmo1.distmod(zs),cosmo2.distmod(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax2[2].legend()   #ax2[2].set_ylim()[0]
    ax2[2].set_ylabel('mag')
    for redshift in redshifts:    
        ax2[2].hlines((cosmo.distmod(redshift)).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax2[2].vlines(redshift,cosmo.distmod(zs)[1].value,(cosmo.distmod(redshift)).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
    
    p = ax3[0].plot(zs,cosmo.efunc(zs),label="efunc = %s"%(l.join(np.array(cosmo.efunc(redshifts),dtype=t))))
    ax3[0].fill_between(zs,cosmo1.efunc(zs),cosmo2.efunc(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax3[0].set_ylabel('E(z)')
    ax3[0].legend()   
    for redshift in redshifts:    
        ax3[0].hlines((cosmo.efunc(redshift)),0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax3[0].vlines(redshift,1,(cosmo.efunc(redshift)),linestyle='dotted', color=p[0].get_color(), label='_nolegend_')

    p = ax3[1].plot(zs,cosmo.scale_factor(zs),label="Scale factor = %s"%(l.join(np.array(cosmo.scale_factor(redshifts),dtype=t))))
    ax3[1].fill_between(zs,cosmo1.scale_factor(zs),cosmo2.scale_factor(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax3[1].legend()      
    ax3[1].set_xlabel('Redshift')      
    ax3[1].set_ylabel('a')      
    for redshift in redshifts:    
        ax3[1].hlines((cosmo.scale_factor(redshift)),0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax3[1].vlines(redshift,0,(cosmo.scale_factor(redshift)),linestyle='dotted', color=p[0].get_color(), label='_nolegend_')

    p = ax3[2].plot(zs,1/cosmo.arcsec_per_kpc_proper(zs),label="Proper = %s"%(l.join(np.array(1/cosmo.arcsec_per_kpc_proper(redshifts).value,dtype=t))))
    ax3[2].fill_between(zs,cosmo1.arcsec_per_kpc_proper(zs),cosmo2.arcsec_per_kpc_proper(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    for redshift in redshifts:    
        ax3[2].hlines(1/(cosmo.arcsec_per_kpc_proper(redshift)).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax3[2].vlines(redshift,0,1/(cosmo.arcsec_per_kpc_proper(redshift)).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
    p = ax3[2].plot(zs,1/cosmo.arcsec_per_kpc_comoving(zs),label="Comoving = %s"%(l.join(np.array(1/cosmo.arcsec_per_kpc_comoving(redshifts).value,dtype=t))))
    ax3[2].fill_between(zs,cosmo1.arcsec_per_kpc_comoving(zs),cosmo2.arcsec_per_kpc_comoving(zs),alpha=0.2, color=p[0].get_color(), label='_nolegend_')
    ax3[2].legend()
    ax3[2].set_ylabel("'/kpc")      
    for redshift in redshifts:    
        ax3[2].hlines(1/(cosmo.arcsec_per_kpc_comoving(redshift)).value,0,redshift,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
        ax3[2].vlines(redshift,0,1/(cosmo.arcsec_per_kpc_comoving(redshift)).value,linestyle='dotted', color=p[0].get_color(), label='_nolegend_')
  
    print('%s : H0=%s, Om0=%s, Ode0=%s, Tcmb0=%s, Neff=%s, Ob0=%s'%(cosmology,cosmo.H0, cosmo.Om0, cosmo.Ode0, cosmo.Tcmb0, cosmo.Neff, cosmo.Ob0))
    plt.suptitle('%s : H0=%s, Om0=%0.3f, Ode0=%0.3f, Tcmb0=%s, Neff=%0.2f, Ob0=%s'%(cosmology,cosmo.H0, cosmo.Om0, cosmo.Ode0, cosmo.Tcmb0, cosmo.Neff, cosmo.Ob0),y=1)
    plt.tight_layout()
    plt.show()
    
    
    #il reste mu, probleme sur Vc,. pho_c
    for key in info.keys():
        print(bcolors.BLACK_RED + '%s : %s'%(key,info[key]) +  bcolors.END )
    
    
    return


def Convertissor(xpapoint):
    """Converts an astropy unit in another one
    """
    #from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    from decimal import Decimal
    #name = dir(u)
    unit_dict = u.__dict__#{'m':u.m,'cm':u.cm}
    val, unit1_, unit2_, redshift = sys.argv[-4:]
    try:
        unit1 = unit_dict[unit1_]
    except KeyError:
        unit1 = u.imperial.__dict__[unit1_]
    try:           
        unit2 = unit_dict[unit2_]
    except KeyError:
        unit2 = u.imperial.__dict__[unit2_]

    print(unit1_,unit1,unit2_,unit2)
    print('%0.2E %s = %0.2E %s'%(Decimal(val), unit1, Decimal((val*unit1).to(unit2)),unit2))
#    from tkinter import messagebox
#    messagebox.showwarning( title = 'Convertissor', message='%0.2E %s = %0.2E %s'%(Decimal(val), unit1, Decimal((val*unit1).to(unit2)),unit2))     
    d = DS9();d.set('analysis message {%0.2E %s = %0.2E %s}'%(Decimal(val), unit1, Decimal((val*unit1).to(unit2)),unit2))

    return


#DS9Plot(path='/Users/Vincent/DS9BackUp/CSVs/190311-13H13_CR_HP_profile.csv')


#def Help(xpapoint):
#    return '/Users/Vincent/Github/DS9functions/pyds9plugin/doc/ref/index.html'

def WaitForN(xpapoint):
    """Wait for N in the test suite to go to next function
    """
    d = DS9(xpapoint)
    while d.get('nan')!='grey':
        time.sleep(0.01)
    d.set('nan black')
    return


def next_step(xpapoint):
    """Goes to next function in the test suite
    """
    d = DS9(xpapoint)
    d.set('nan grey')
    return

def DS9tsuite(xpapoint):
    """Teste suite: run several fucntions in DS9
    """
    #if connect(host='http://google.com') if False: sys.exit()
    d = DS9(xpapoint)
    d.set('nan black')#sexagesimal
    d.set("""analysis message {Test suite for beginners, please make sure you have a good internet connection so that images can be downloaded. This help, will go through most of the must-know functions of this plug-in. Between each function a message window will pop-up to explain you what does the function. After it has run you can run it again and change the parameters. When this is done you can just press the n key (next) so that you go to the next function.}""")

#############################################
    
    
    d.set('frame new')
    d.set('tile no')
    d.set("analysis message {First of all, let's load an image. You can change the cut/colormap/smoothing by yourself. }")
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/calexp-HSC-G-9813-1-5_1.fits')
    d.set('zoom to fit')
    WaitForN(xpapoint)
    d.set("""analysis message {Now let's use the first function: change all the display setups directly. Next time you will be able to access directly this function by hitting shift+s (for settings) or go in Analysis->Generic functions->Setup->Change display parameters. This function will make you gain a lot of time. If you create a region before hitting shift+s it will compute the threshold based on the encircled data! Do not forget to hit n when you want to go to next function!}""")
    d.set('analysis task 7')#setup
    WaitForN(xpapoint)
    d.set("analysis message {Now let's use SExtractor to perform a source extraction on this image. You can access this function via: Analysis->Astromatic->Setup->Change display parameters. Do not hesitate to change the detection threshold or other parameters to try to optimize your source extraction!}")
    d.set('analysis task 29')
    WaitForN(xpapoint)
    
    #######################3
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/2_Visualization-TF-TS/TF/TF_WCS/stack8099989_pa-161_2018-06-11T06-02-49_wcs.fits')
    DS9setup2(xpapoint)
    WaitForN(xpapoint)
    d.set('analysis task 34')
    WaitForN(xpapoint)
        #####################
    
    
    
    d.set("analysis message {Now let's do some resampling and co-add of FITS images having WCS header. It uses SWARP software.}")
    d.set('frame delete all')
    for file in glob.glob('/Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/SWARP/calexp2_trim?.fits'):
        d.set('frame new')
        d.set('file ' + file)
        DS9setup2(xpapoint)
    d.set('tile yes')
#    d.set("analysis message {You can type Shift+l to directly control all the lock parameters between the different frames}")
#    d.set('analysis task 9')
#    d.set('lock frame wcs')



#    d.set('frame new')
#    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/SWARP/calexp2.fits')
#    d.set('zoom to fit')
#    d.set('zoom out')
#    d.set('frame next')
#    d.set('zoom to fit')
#    d.set('zoom out')
    WaitForN(xpapoint)
    d.set("analysis message {As you can see the different images are cut... SWARP perform background subtraction and noise-level measurement for automatic weighting. Do not hesitqte to re-run the function and press n when you are satisfierd and want to go to next function.}")
    d.set('analysis task 31')
    WaitForN(xpapoint)
    #d.set("analysis message {Now they are attached! Let's now use multi-band images to simulate visible images. }")
    d.set('frame delete all')
    d.set('tile yes')
#    d.set('lock cmap no')
#    d.set("lock colorbar no") 
#
#    d.set('lock scalelimits no')
#    d.set("lock frame no")

    d.set('frame new') 
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/STIFF/G_Trimm.fits')
    DS9setup2(xpapoint, color='cool')
    d.set('frame new')
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/STIFF/R_Trimm.fits')
    DS9setup2(xpapoint, color='green')
    d.set('frame new')
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/STIFF/I_Trimm.fits')
    DS9setup2(xpapoint, color='red')
#    d.set('lock frame physical')
    WaitForN(xpapoint)
#    d.set('wcs skyformat degrees')
#    d.set('wcs reset')
    d.set('lock frame image')
    d.set('lock frame no')

    d.set("analysis message {We are now gonna use STIFF function to generate from scientific FITS image more popular TIFF format images for illustration puporses. Enter the path of a same image in three different bands (R,G,I) and STIFF will perform an accurate reproduction of the original surface brightness and colour, automatioc contrast/brightness adjustments, etc.}")
    #d.set('wcs reset')
    d.set('analysis task 32')
    WaitForN(xpapoint)
    

#
#
#    d.set('regions format ds9')
#    d.set('regions system image')
#    d.set('tile no')
#    d.set('regions command  "circle 8000 8000 20 # color=yellow"')
#    d.set('pan to 8000 8000')
#    d.set('zoom to 4')
#    d.set('regions select all')
#    d.set("analysis message {Create a region }")        
#    WaitForN(xpapoint)
    d.set("analysis message {Now some more simple features that do not need any other code to be run. Create a region on the part of the galaxy for instance (select it) and hit n : This will generate a 3d plot of a region area.  }")    
    WaitForN(xpapoint)
    d.set('analysis task 17')
    WaitForN(xpapoint)
    d.set("analysis message {No create a region a a bright star and select it. Then press n to use the function 2D gaussian fit!}")    
    WaitForN(xpapoint)
    d.set('analysis task 12')
    WaitForN(xpapoint)
    d.set("analysis message {Or a radial profile to compute the seeing and the encircled energy.}")    
    d.set('analysis task 20')    
    WaitForN(xpapoint)
    d.set('frame delete all')
    d.set('frame new') 
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/Command/image000034.fits')
    DS9setup2(xpapoint, color='cool')
    d.set('zoom to 1')
    d.set('regions command  "box 1598 1071 702 38 # color=yellow"')
    WaitForN(xpapoint)
    d.set("analysis message {This package also allows you to perform interactive 1D fitting with user adjustment of the initial fitting parameters. Choose 3 gaussian in order to fit the different features. }")    
    d.set('analysis task 11')
    WaitForN(xpapoint)
    
    d.set('frame delete all')
    files  = glob.glob('/Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/TF/stack*_Trimm.fits')
    files.sort()
    d.set('lock frame image')
    d.set('lock scalelimits yes')
    for file in files:
        d.set('frame new')
        d.set('file ' + file)
    DS9setup2(xpapoint)
 
        
    d.set("analysis message {Here is a throughfocus.Please create a region around a close to focus spot. Then hit n.}")    
    WaitForN(xpapoint)
    d.set('analysis task 19')
    WaitForN(xpapoint)
    
    #add throughfocus
    #add throughfocus analysis
    #add astrometry.net
    #add swarp
    

    
    d.set('frame delete all')
    d.set('frame new')
    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/Guidance/stack25055816_Trimm.fits')
    DS9setup2(xpapoint, color='cool')
    d.set("analysis message {Let us ty some real time simplistic guiding code in the case you need it for your observing nights. Select around ~3 stars by creating a circle regin around them and hit m.}")    
    WaitForN(xpapoint)
    sys.exit()
    d.set('analysis task 26')
    WaitForN(xpapoint)
    d.set("analysis message {You are now ready to use the DS9 Quick Look plugin by yourself.}")    
    sys.exit()
  
#    
#    WaitForN(xpapoint)
#    #####Cubes
#    d.set('file /Users/Vincent/Nextcloud/Work/Keynotes/DS9Presentation/CESAM/Test/_104002019_objcube.fits')
#    DS9setup2(xpapoint, color='cool')
#    d.set("analysis message {This package also allows you to work on data cube such as spatio-spectral 3D images. For instance here, we will extract the spectra at different location in order to compare them.}")    
#    WaitForN(xpapoint)
#    a = d.set('analysis task Guidance')    
    
    return



def DS9RemoveImage(xpapoint):
    """Substract an image, for instance bias or dark image, to the image
    in DS9. Can take care of several images
    """
    from .BasicFunctions import ExecCommand
    d = DS9(xpapoint)
    filename = getfilename(d)
    path2remove, exp, eval_ = sys.argv[3:6]
    #exp = sys.argv[4]
    #a, b = sys.argv[4].split(',')
    print('Expression to be evaluated: ', exp)
    #a, b = float(a), float(b)
    #if len(sys.argv) > 5: path = Charge_path_new(filename, entry_point=5)
    path = Charge_path_new(filename) if len(sys.argv) > 5 else [filename] #and print('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))

        
    for filename in path:
        print(filename)
        result, name = ExecCommand(filename, path2remove=path2remove, exp=exp, config=my_conf, eval_=bool(int(eval_))) 
                                        
    if len(path) < 2:
        d.set('frame new')
        d.set('tile yes')
        d.set('file ' + name)  
    return


def main():
    """Main function where the arguments are defined and the other functions called
    """
#    path = os.path.dirname(os.path.realpath(__file__))
#    import pkg_resources;print('Version = ', pkg_resources.require("pyds9plugin")[0].version)
#    print("which('DS9Utils') =", which('DS9Utils'))
#    print("__file__ =", __file__)
#    print("__package__ =", __package__)
#    print('Python version = ', sys.version)
    if len(sys.argv)==1:
        PresentPlugIn()
        LoadDS9QuickLookPlugin()
        SetDiplay()
                

    print(datetime.datetime.now())
    start = time.time()
    
    DictFunction_Generic =  {'setup':DS9setup2,'Update':DS9Update,'fitsgaussian2D': fitsgaussian2D, 'DS9createSubset':DS9createSubset,
                             'DS9Catalog2Region':DS9Catalog2Region, 'AddHeaderField':AddHeaderField,'BackgroundFit1D':BackgroundFit1D,
                             'test':DS9tsuite,  'Convolve2d':Convolve2d,'PlotSpectraDataCube':PlotSpectraDataCube,'StackDataDubeSpectrally': StackDataDubeSpectrally,
                             'stack': DS9stack_new,'lock': DS9lock,'CreateHeaderCatalog':DS9CreateHeaderCatalog,'SubstractImage': DS9RemoveImage,
                             'DS9Region2Catalog':DS9Region2Catalog, 'DS9MaskRegions':DS9MaskRegions,'CreateImageFromCatalogObject':CreateImageFromCatalogObject,
                             'PlotArea3D':PlotArea3D, 'OriginalSettings': DS9originalSettings,'next_step':next_step }
                   
    DictFunction_AIT =     {'centering':DS9center, 'radial_profile':DS9rp,
                            'throughfocus':DS9throughfocus, 
                            'throughfocus_visualisation':DS9visualisation_throughfocus, 
                            'throughslit': DS9throughslit,
                            'ReplaceWithNans': DS9replaceNaNs,'InterpolateNaNs': DS9InterpolateNaNs,'Trimming': DS9Trimming,
                            'ColumnLineCorrelation': DS9CLcorrelation,
                            'DS9_2D_FFT':DS9_2D_FFT,'2D_autocorrelation': DS9_2D_autocorrelation,
                             }


    DictFunction_SOFT =   {'DS9SWARP':DS9SWARP,'DS9PSFEX': DS9PSFEX,'RunSextractor':RunSextractor,'DS9saveColor':DS9saveColor,
                           'ExtractSources':DS9ExtractSources,'CosmologyCalculator':CosmologyCalculator,'Convertissor': Convertissor,'WCS':DS9guider}
 
   
    DictFunction = {}
    for d in (DictFunction_Generic, DictFunction_AIT, DictFunction_SOFT):#, DictFunction_Calc, DictFunction_SOFT, DictFunction_FB, DictFunction_Delete): #DictFunction_CLAUDS
        DictFunction.update(d)


    xpapoint = sys.argv[1]
    function = sys.argv[2]
    #Choose_backend(function)
    
    print(bcolors.BLACK_RED + 'DS9Utils ' + ' '.join(sys.argv[1:]) + bcolors.END)# %s %s '%(xpapoint, function) + ' '.join())
    
    print(bcolors.GREEN_WHITE + """
          *******************************************************************************************************
                                                      Function = %s
          *******************************************************************************************************"""%(function)+ bcolors.END)
    a = DictFunction[function](xpapoint=xpapoint)             
    stop = time.time()
    print(bcolors.BLACK_GREEN + """
        *******************************************************************************************************
                   date : %s     Exited OK, duration = %s      
        ******************************************************************************************************* """%(datetime.datetime.now().strftime("%y/%m/%d %HH%Mm%S"), str(datetime.timedelta(seconds=np.around(stop - start,1)))[:9]) + bcolors.END)
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
                plt.title(key + '%i objets: M = %0.2f  -  Sigma = %0.3f'%(len(l), np.nanmean(l), np.nanstd(l)));plt.xlabel(key);plt.ylabel('Frequecy')
                plt.savefig(DS9_BackUp_path +'Plots/%s_Outputs_%s.png'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),key))
                csvwrite(np.vstack((np.arange(len(l)), l)).T,DS9_BackUp_path +'CSVs/%s_Outputs_%s.csv'%(datetime.datetime.now().strftime("%y%m%d-%HH%M"),key))
            except TypeError:
                pass

    return a



if __name__ == '__main__':
    a = main()