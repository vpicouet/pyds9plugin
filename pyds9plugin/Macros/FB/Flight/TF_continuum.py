from astropy.io import fits
# from pyds9plugin.DS9Utils import stack_images_path, globglob, DS9n, verboseprint
import glob
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

#TODO verify it works with cosmics
path=""
region = getregion(d, quick=True, selected=True)
x_inf, x_sup, y_inf, y_sup = lims_from_region(None, coords=region)

if path == "":
    path = get_filename(d, All=True, sort=False)
else:
    path = globglob(args.path, xpapoint=args.xpapoint)

filename = get_filename(d)
header = fits.open(filename)[0].header
ns = np.arange(len(path))

fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(13,5))
ll=[]
flux=[]
for i, f in enumerate(path):
    image = fits.open(f)[0].data[y_inf:y_sup, x_inf:x_sup]
    y = image.mean(axis=1)
    x = np.arange(len(y))
    l = ax1.plot(x,y,".",label=i)
    a = PlotFit1D(x,y,deg="gaus",ax=ax1,color=l[0].get_color(),P0=[y.ptp(),x.mean(),3,y.min()])
    ll.append(abs(a["popt"][2]))
    flux.append((y-y.min()).sum())
flux=np.array(flux)
ax2.plot( ns,ll,"ok")
ax3.plot( ns,flux,"ok")
PlotFit1D(ns,ll,deg=2,ax=ax2,color="k")#,P0=[y.ptp(),x.mean(),3,y.min()])
PlotFit1D(ns,flux,deg="gaus",ax=ax2,color="k",P0=[flux.ptp(),ns.mean(),3,flux.min()])
ax1.legend()
ax1.set_ylabel("ADU")
ax1.set_xlabel("pixels")
ax1.set_xlim((x.min(),x.max()))
ax2.set_ylabel("sigma")
ax3.set_ylabel("ADU sum")
ax2.set_xlim((0,len(path)))
ax3.set_xlim((0,len(path)))
ax2.set_ylim((np.min(ll),np.max(ll)))
ax3.set_ylim((np.min(flux),np.max(flux)))
ax2.set_xlabel("# image")
ax3.set_xlabel("# image")
fig.tight_layout()
plt.show()
