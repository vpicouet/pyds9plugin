import matplotlib.pyplot as plt
from astropy.table import Table
from pyds9plugin.DS9Utils import plot_surface
from tqdm.tk import trange, tqdm
from scipy.optimize import curve_fit

from pyds9plugin.Macros.Fitting_Functions.functions import slit, smeared_slit
slitm=slit
# def slitm(x, amp, l, x0, FWHM, offset):
#     """Convolution of a box with a gaussian
#     """
#     from scipy import special
#     import numpy as np
#     l/=2
#     a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
#     b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
#     function = amp * (a + b) / (a + b).ptp()#4 * l
#     return  function + offset

def change_val_list(popt,val,new_val):
    popt1 = list(map(lambda item: new_val if item==val else item, popt))
    return popt1


# def variable_smearing_kernels(
#     image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
# ):
#     """Creates variable smearing kernels for inversion
#     """
#     import numpy as np

#     n = 15
#     smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
#     if type_ == "exp":
#         smearing_kernels = np.exp(
#             -np.arange(n)[::int(np.sign(smearing_length[0])), np.newaxis, np.newaxis] / abs(smearing_length)
#         )
#     else:
#         assert 0 <= Smearing <= 1
#         smearing_kernels = np.power(Smearing, np.arange(n))[
#             :, np.newaxis, np.newaxis
#         ] / np.ones(smearing_length.shape)
#     smearing_kernels /= smearing_kernels.sum(axis=0)
#     return smearing_kernels

# def smeared_slit(x, amp, l, x0, FWHM, offset,Smearing):
#     """Convolution of a box with a gaussian
#     """
#     # Smearing=0.8
#     from scipy import special
#     import numpy as np
#     from scipy.sparse import dia_matrix

#     l/=2
#     a = special.erf((l - (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
#     b = special.erf((l + (x - x0)) / np.sqrt(2 * (FWHM/2.35)**2))
#     function = amp * (a + b) / (a + b).ptp()#+1#4 * l
#     # function = np.vstack((function,function)).T
#     smearing_kernels = variable_smearing_kernels(
#         function, Smearing, SmearExpDecrement=50000)
#     n = smearing_kernels.shape[0]
#     # print(smearing_kernels.sum(axis=1))
#     # print(smearing_kernels.sum(axis=1))
#     A = dia_matrix(
#         (smearing_kernels.reshape((n, -1)), np.arange(n)),
#         shape=(function.size, function.size),
#     )
#     function = A.dot(function.ravel()).reshape(function.shape)
#     # function = np.mean(function,axis=1)
#     return  function + offset

    # p.app.exec_()

if not os.path.exists(os.path.dirname(filename)+'/fits'):
    os.mkdir(os.path.dirname(filename)+'/fits')
# fit='spatial'
# fit='spectral'

d=DS9n()
regs=getregion(d,selected=True)
image = d.get_pyfits()[0].data
# xx,yy,fwhm=[],[],[]
# names=[]
# ptps=[]


def Measure_PSF_slits(image, regs, plot_=True):
    cat = Table(names=['name','color','line','x','y','amp_x','lx','x0','fwhm_x','off_x','amp_y','ly','y0','fwhm_y','off_y','smearing','fwhm_x_unsmear'],dtype=[str,str]+[float]*15)

    for region in tqdm(regs[:]):
        x,y = int(region.xc),int(region.yc)
        if (x>1060) & (y<1950) & (y>40) & (x<2060):
            # if region.id=="":
                # region.id = "%i"%(region.yc)
            x_inf, x_sup, y_inf, y_sup = lims_from_region(region=region, coords=None)
            # if fit=='spatial':
            #     subim = image[y_inf:y_sup, x_inf:x_sup].T
            # else:
            n=20
            subim1 = image[y_inf-n:y_sup+n, x_inf:x_sup]
            subim2 = image[y_inf:y_sup, x_inf-n:x_sup+n]
            subim3 = image[y_inf-n:y_sup+n, x_inf-n:x_sup+n]
            # subim = image[y_inf:y_sup, x_inf:x_sup]
            # xx.append(x)
            # yy.append(y)
            # plt.figure()
            # plt.imshow(subim)
            # plt.show()
            y_spatial = np.nanmedian(subim1,axis=1)
            y_spectral = np.nanmedian(subim2,axis=0)
            x_spatial = np.arange(len(y_spatial))
            x_spectral = np.arange(len(y_spectral))
            # n_conv=1
            # y_conv = np.convolve(y,np.ones(n_conv)/n_conv,mode='same')

            # x = np.arange(len(y))
            # min1,min2 = np.min(y[:int(len(y)/2)]), np.min(y[int(len(y)/2):])
            # min1,min2 = y[0],y[-1]#np.min(y[:int(len(y)/2)]), np.min(y[int(len(y)/2):])
            # mask = (x>np.where(y==min1)[0][0]) & (x<np.where(y==min2)[0][-1])
            # slit_m2 = lambda x,  l, x0, FWHM : slitm(x, amp=y[mask].ptp() , l=l, x0=x0, FWHM=FWHM, offset=y.min())#y.min()
            # bounds = [8,10,2],[20,50,14]

            if plot_:
                fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,4))#,sharex=True)#,figsize=(5,13))
                # if fit=='spatial':
                ax1.imshow(subim3)
                # else:
                    # ax1.imshow(subim.T)
                ax1.set_title(  "Slit #%s: x=%i, y=%i"%(region.id,region.xc,region.yc))
                # ax2.plot(x_spatial,y_spatial,":k")
                # ax3.plot(x_spectral,y_spectral,":k")
            else:
                ax2=None
                ax3=None
            P0 = [y_spectral.ptp(),3.3,len(y_spectral)/2,1.7,np.median(y_spectral),1.2]
            bounds = [[0.7*y_spectral.ptp(),0,0,0,np.nanmin(y_spectral),0.1], [y_spectral.ptp(),len(y_spectral),len(y_spectral),10,np.nanmax(y_spectral),6]]
            try:
                popt_spectral_deconvolved,pcov = curve_fit(smeared_slit,x_spectral,y_spectral, p0=P0,bounds=bounds)#,bounds=bounds
            except (ValueError, RuntimeError) as e:
                popt_spectral_deconvolved = [0.1]*6
            print(popt_spectral_deconvolved)
            # bounds = [[0.7*y_spatial.ptp(), 10, 0, 0, np.nanmin(y_spatial)], [y_spatial.ptp(),  len(y_spatial), len(y_spatial), 10, np.nanmax(y_spatial)]]
            # popt_spatial = PlotFit1D(x_spatial,y_spatial,deg=slitm, plot_=plot_,ax=ax2,P0=[y_spatial.ptp(),20,x_spatial.mean()+1,2,y_spatial.min()],c='k',lw=2,bounds=bounds)['popt']
            try:
                bounds = [[0.7*y_spatial.ptp(), 20, 0, 0, np.nanmin(y_spatial)], [y_spatial.ptp(),  len(y_spatial), len(y_spatial), 18, np.nanmax(y_spatial)]]
                popt_spatial = PlotFit1D(x_spatial,y_spatial,deg=slitm, plot_=plot_,ax=ax2,P0=[y_spatial.ptp(),22,x_spatial.mean()+1,2,y_spatial.min()],c='k',lw=2,bounds=bounds)['popt']
                bounds = [[0.7*y_spectral.ptp(), 3, 0, 0, np.nanmin(y_spectral)],
                        [y_spectral.ptp(),8 , len(y_spectral), 15, np.nanmax(y_spectral)]]# len(y_spectral)
                popt_spectral = PlotFit1D(x_spectral, y_spectral, deg=slitm, plot_=plot_, ax=ax3, P0=[y_spectral.ptp(
                ), 4, x_spectral.mean()+1, 2, y_spectral.min()], c='k', ls='--', lw=0.5, bounds=bounds)['popt']
                popt_spatial = abs(np.array(popt_spatial))
                popt_spectral = abs(np.array(popt_spectral))
            except ValueError:
                print('error: ', region.id)
                popt_spatial, popt_spectral = [0,0,0,0,0],[0,0,0,0,0]
            # popt = PlotFit1D(x[mask],y_conv[mask],deg=slit_m2, plot_=True,ax=ax2,P0=[8,x.mean()+1,2.1],bounds=bounds)['popt']
            # if len(popt)==3:
            #     fwhmi, slit_w = popt[-1],popt[0]
            # else:
            #     fwhmi, slit_w = popt[-2],popt[1]
            # print([regs[i].id,regs[i].xc,regs[i].yc,*popt_spatial,*popt_spectral])
            # print(len([regs[i].id,regs[i].xc,regs[i].yc,*popt_spatial,*popt_spectral]))
            if region.color=="red":
                line=214
            elif region.color=="yellow":
                line=206
            elif region.color=="blue":
                line=203
            cat.add_row([region.id,region.color,line,region.xc,region.yc,*popt_spectral,*popt_spatial,popt_spectral_deconvolved[-1],popt_spectral_deconvolved[-3]])

            # amp, l, x0, FWHM
            # fwhm.append(fwhmi)
            # ptps.append(y.ptp())

            if plot_:
                ax2.plot(x_spatial,y_spatial,":k")
                ax2.plot(x_spatial,slitm(x_spatial,*change_val_list(popt_spatial,popt_spatial[1],0.001)),"-k",lw=0.5,label="Slit length=%0.1f\nFWHM=%0.1f"%(popt_spatial[1],popt_spatial[-2]))
                ax2.plot(x_spatial,slitm(x_spatial,*change_val_list(popt_spatial,popt_spatial[-2],0.001)),"-k",lw=0.5)

                ax3.plot(x_spectral,slitm(x_spectral,*change_val_list(popt_spectral,popt_spectral[1],0.001)),"--k",lw=0.5,label="Slit width=%0.1f\nFWHM=%0.1f"%(popt_spectral[1],popt_spectral[-2]))
                ax3.plot(x_spectral,slitm(x_spectral,*change_val_list(popt_spectral,popt_spectral[-2],0.001)),"-k",lw=0.5)

                ax3.plot(x_spectral,smeared_slit(x_spectral,*popt_spectral_deconvolved),"-k",lw=2,label="Unsmeared\nFWHM=%0.1f\nSmearing=%0.1f"%(popt_spectral_deconvolved[3],popt_spectral_deconvolved[-1]))
                try:
                    ax3.plot(x_spectral,smeared_slit(x_spectral,*change_val_list(change_val_list(popt_spectral_deconvolved,popt_spectral_deconvolved[-1],0.1),popt_spectral_deconvolved[1],0.1)),"-k",lw=0.5)
                except ValueError:
                    pass
                ax3.plot(x_spectral,y_spectral,":k",lw=2)

                # ax2.plot(x[mask],y_conv[mask],'-',)#label='FWHM = %0.1f\nslit length=%0.1f'%(popt[0],popt[-1]))
                ax2.set_xlabel('y')
                ax1.set_ylabel('y')
                ax1.set_xlabel('x')
                ax3.set_xlabel('x')
                ax2.legend(loc='center left',fontsize=10)
                ax3.legend(loc='center left',fontsize=10)
                ax2.set_xlim((x_spatial.min(),x_spatial.max()))
                ax3.set_xlim((x_spectral.min(),x_spectral.max()))
                fig.tight_layout()
                if region.id=="":
                    plt.savefig(os.path.dirname(filename)+'/fits/%s_%s.png' %
                                (os.path.basename(filename)[:2], int(region.yc)))
                else:
                    plt.savefig(os.path.dirname(filename)+'/fits/%s_%s_%s.png'%(os.path.basename(filename)[:2],region.id,line))
                plt.close()
                # plt.show()
                # sys.exit()
    # xx,yy,fwhm = np.array(xx),np.array(yy), np.array(fwhm)
    # print(names)
    # mask = (fwhm>0.1) & (fwhm<100)# & (np.array(ptps)>60)
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
    print(filename.replace('.fits','.csv'))
    cat["l203"] = False
    cat["l214"] = False
    cat["l206"] = False
    cat["l203"][cat['line']==203.0]=True
    cat["l214"][cat['line'] == 214.0] = True
    cat["l206"][cat['line'] == 206.0] = True
    cat.write(filename.replace('.fits','.csv'),overwrite=True)
    return cat, filename


# x = cat['x']
# y = cat['y']
# z = cat['fwhm']
# X, Y, Z = fit_surface(xx,yy,fwhm)
# plot_surface(cat,'x','y','fwhm_y')

cat, filename = Measure_PSF_slits(image,regs)

field = os.path.basename(filename)[:2]
from matplotlib.colors import LogNorm


mask = (cat['fwhm_y'] > 0.5) & (cat['fwhm_x'] > 0.5) #& (cat['x'] <1950)  & (cat['x'] >50) #& (cat['fwhm_x'] < 8) & (cat['fwhm_y'] < 8) & (cat['fwhm_x_unsmear'] < 8)  
    # = (cat['fwhm_x_unsmear']>0.5)&
fig, axes = plt.subplots(2,2,sharex='col',sharey='row', gridspec_kw={'height_ratios': [1, 3],'width_ratios': [3, 1]},figsize=(8,5))
ax0,ax1,ax2,ax3 = axes.flatten()
m = 'o'
size=3
print(cat['fwhm_x_unsmear'],cat['fwhm_y'],cat['fwhm_x'])
print(cat['fwhm_x_unsmear'][mask])
norm= LogNorm(vmin=np.min(cat['fwhm_x_unsmear'][mask]),vmax=np.max(cat['fwhm_y'][mask]))
im=ax2.scatter(cat['y'][mask]-50,cat['x'][mask],c=cat['fwhm_y'][mask],s=50,norm =norm)#,marker=',',)
ax2.scatter(cat['y'][mask],cat['x'][mask],c=cat['fwhm_x'][mask],s=50,norm = norm)#,vmin=3,vmax=6)
ax2.scatter(cat['y'][mask]+50,cat['x'][mask],c=cat['fwhm_x_unsmear'][mask],s=50,norm = norm,marker='>')#,vmin=3,vmax=6)
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatter
cax = make_axes_locatable(ax2).append_axes('right', size='2%', pad=0.02)
fig.colorbar(im, cax=cax, orientation='vertical',ticks=[1,2,3,4,5,6,7], format=LogFormatter(10, labelOnlyBase=False) )

# cat['color'][(cat['color']==np.ma.core.MaskedConstant).mask]='green'
ax3.set_ylim(ax2.get_ylim())
# for color in ['red']:#,'yellow','green']:
for color in ['red','yellow','blue']:
    mask = (cat['color']==color)#.mask
    p=ax0.plot(cat[mask]['y'],cat[mask]['fwhm_y'],marker='s',lw=0,ms=size,c=color.replace('yellow','orange'))
    fit = PlotFit1D(cat[mask]['y'],cat[mask]['fwhm_y'],deg=2, plot_=True,ax=ax0,c=p[0].get_color())
    ax0.plot(cat['y'],cat['fwhm_x'],m,c=color.replace('yellow','orange'),alpha=0.3)
    p=ax0.plot(cat['y'][mask],cat['fwhm_x_unsmear'][mask],marker='>',lw=0,ms=size,c=color.replace('yellow','orange'))
    fit = PlotFit1D(cat['y'][mask],cat['fwhm_x_unsmear'][mask],deg=2, plot_=True,ax=ax0,c=p[0].get_color(),ls='--')
    ax0.set_ylim((1.5,8))
    # c=abs(fwhm[mask]),s=np.array(ptps)[mask]*50
    # ax2.scatter(cat['x'],cat['fwhm_y'],'.')
    p=ax3.plot(cat[mask]['fwhm_y'],cat[mask]['x'],m,marker='s',lw=0,ms=size,c=color.replace('yellow','orange'))
    fit = PlotFit1D(cat[mask]['x'],cat[mask]['fwhm_y'],deg=1, plot_=False,ax=ax3,c=p[0].get_color())
    ax3.plot(fit['function'](cat['x'][mask]),cat['x'][mask],c=p[0].get_color(),ls='-')
    # ax3.plot(cat['fwhm_x'],cat['x'],m)
    p=ax3.plot(cat[mask]['fwhm_x_unsmear'],cat[mask]['x'],marker='>',ms=size,lw=0,c=color.replace('yellow','orange'))
    fit = PlotFit1D(cat['x'][mask],cat['fwhm_x_unsmear'][mask],deg=1, plot_=False,ax=ax3,c=p[0].get_color(),ls='--')
    ax3.plot(fit['function'](cat['x'][mask]),cat['x'][mask],c=p[0].get_color(),ls=':')
ax3.set_xlim((1.5,8))
ax1.axis('off')
ax1.plot(-1,-1,label='Spatial FWHM',marker='s',lw=0,c='k')
ax1.plot(-1,-1,m,label='Spectral FWHM',c='k')
ax1.plot(-1,-1,label='Unsmeared\nspectral FWHM',marker='>',lw=0,c='k')
ax1.plot(-1,-1,m,label='lambda=213',lw=0,c='r')
ax1.plot(-1,-1,m,label='lambda=206',c='orange')
ax1.plot(-1,-1,m,label='lambda=202',lw=0,c='g')
ax2.set_xlim((0,2000))
ax0.set_ylabel('Resolution')
ax3.set_xlabel('Resolution')
ax2.set_xlabel('Y')
ax2.set_ylabel('X')
ax1.legend(fontsize=7,title=field)
fig.tight_layout()
plt.show()





#
# plt.figure()#figsize=(5,10))
# plt.scatter(np.array(ptps)[mask], fwhm[mask],s=np.array(ptps)[mask]*20)#,vmin=50,vmax=150)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
