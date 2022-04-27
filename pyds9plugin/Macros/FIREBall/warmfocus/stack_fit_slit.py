import matplotlib.pyplot as plt
from astropy.table import Table
from pyds9plugin.DS9Utils import plot_surface

def slitm(x, amp, l, x0, FWHM, offset):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np

    a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
    function = amp * (a + b) / (a + b).ptp()#4 * l
    return  function + offset



    # p.app.exec_()

fit='spatial'
fit='spectral'
plot_=False
d=DS9n()
regs=getregion(d,selected=True)
image = d.get_pyfits()[0].data
xx,yy,fwhm=[],[],[]
names=[]
ptps=[]
for i,region in enumerate(regs[:]):
    x,y = int(regs[i].xc),int(regs[i].yc)
    names.append(regs[i].id)
    x_inf, x_sup, y_inf, y_sup = lims_from_region(region=region, coords=None)
    # if fit=='spatial':
    #     subim = image[y_inf:y_sup, x_inf:x_sup].T
    # else:
    subim = image[y_inf-20:y_sup+20, x_inf:x_sup]
    # subim = image[y_inf:y_sup, x_inf:x_sup]
    xx.append(x)
    yy.append(y)
    # plt.figure()
    # plt.imshow(subim)
    # plt.show()
    if fit=='spatial':
        y = np.nanmedian(subim,axis=0)
    else:
        y = np.nanmedian(subim,axis=1)
    y_conv = np.convolve(y,np.ones(3)/3,mode='same')

    x = np.arange(len(y))
    min1,min2 = np.min(y[:int(len(y)/2)]), np.min(y[int(len(y)/2):])
    min1,min2 = y[0],y[-1]#np.min(y[:int(len(y)/2)]), np.min(y[int(len(y)/2):])
    mask = (x>np.where(y==min1)[0][0]) & (x<np.where(y==min2)[0][-1])
    slit_m2 = lambda x,  l, x0, FWHM : slitm(x, amp=y[mask].ptp() , l=l, x0=x0, FWHM=FWHM, offset=y.min())#y.min()
    bounds = [8,10,2],[20,50,14]

    if plot_:
        fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)#,figsize=(5,13))
        if fit=='spatial':
            ax1.imshow(subim)
        else:
            ax1.imshow(subim.T)
        ax1.set_title(regs[i].id + ": y=%i"%int(regs[i].yc))
        ax2.plot(x,y,":")
    else:
        ax2=None

    if fit=='spatial':
        popt = PlotFit1D(x[mask],y[mask],deg='gaus', plot_=plot_,ax=ax2)['popt']
    else:
        popt = PlotFit1D(x[mask],y[mask],deg=slitm, plot_=plot_,ax=ax2,P0=[y.ptp(),4,x.mean()+1,2,y.min()])['popt']

    # popt = PlotFit1D(x[mask],y_conv[mask],deg=slit_m2, plot_=True,ax=ax2,P0=[8,x.mean()+1,2.1],bounds=bounds)['popt']
    if len(popt)==3:
        fwhmi, slit_w = popt[-1],popt[0]
    else:
        fwhmi, slit_w = popt[-2],popt[1]


    fwhm.append(fwhmi)
    ptps.append(y.ptp())

    if plot_:
        ax2.plot(x[mask],y[mask],'-',label='FWHM = %0.1f\nslit length=%0.1f'%(fwhmi, slit_w))

        ax2.plot(x[mask],y_conv[mask],'-',)#label='FWHM = %0.1f\nslit length=%0.1f'%(popt[0],popt[-1]))
        ax2.legend(loc='upper right')
        ax2.set_xlim((x.min(),x.max()))
        fig.tight_layout()
        plt.show()

xx,yy,fwhm = np.array(xx),np.array(yy), np.array(fwhm)
# print(names)
mask = (fwhm>0.1) & (fwhm<100)# & (np.array(ptps)>60)
# plt.figure()#figsize=(5,10))
# plt.scatter(xx[mask],yy[mask],c=abs(fwhm[mask]),s=np.array(ptps)[mask]*20)#,vmin=50,vmax=150)
# plt.scatter(xx[~mask],yy[~mask],c=abs(fwhm[~mask]),s=np.array(ptps)[~mask])#,vmin=50,vmax=150)
# for i, txt in enumerate(fwhm):
#     plt.annotate("%s:\nsig=%0.1f"%(names[i],fwhm[i]), (xx[i]-40,yy[i]+30))
# plt.colorbar()
# # plt.axis('equal')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

cat = Table(data=[xx,yy,fwhm,ptps],names=['x','y','fwhm','amplitude'])
cat.write('/tmp/fwhm_%s.csv'%(fit),overwrite=True)


# x = cat['x']
# y = cat['y']
# z = cat['fwhm']
# X, Y, Z = fit_surface(xx,yy,fwhm)
plot_surface(cat,'x','y','fwhm')








#
# plt.figure()#figsize=(5,10))
# plt.scatter(np.array(ptps)[mask], fwhm[mask],s=np.array(ptps)[mask]*20)#,vmin=50,vmax=150)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
