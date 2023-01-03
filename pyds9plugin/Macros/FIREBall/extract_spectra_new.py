from astropy.table import Table
import numpy as np
from pyds9plugin.DS9Utils import blockshaped, get, DS9n, verboseprint, getregion, get_filename
from astropy.table import Column
import matplotlib.pyplot as plt
import os
d=DS9n()

new_name = "/tmp/test.fits"#get(d, 'Path where you want to save the region catalog.', exit_=True)
verboseprint(new_name)

path = get_filename(d)
image = d.get_pyfits()[0].data


d.set("regions select all")
regions = getregion(d, all=False, quick=False, selected=True)
if regions is None:
    message(d, "It seems that you did not create any region. Please create regions and re-run the analysis.")
    sys.exit()
if hasattr(regions[0], "xc"):
    x, y, r1, r2 = (
        np.array([r.xc for r in regions]),
        np.array([r.yc for r in regions]),
        np.array([r.r if hasattr(r, "r") else r.w for r in regions]),
        np.array([r.r if hasattr(r, "r") else r.h for r in regions]),
    )
else:
    x, y, r1, r2 = (
        np.array([r.xc for r in [regions]]),
        np.array([r.yc for r in [regions]]),
        np.array([r.r if hasattr(r, "r") else r.w for r in [regions]]),
        np.array([r.r if hasattr(r, "r") else r.h for r in [regions]]),
    )

cat = Table((x - 1, y - 1, r1, r2), names=("x", "y", "w", "h"))
if hasattr(regions[0], "id"):
    cat["ids"] = np.array([r.id for r in regions])
else:
    cat["ids"] = np.arange(len(cat))
verboseprint(cat)
images = []
w = int(cat[0]["w"])
h = int(cat[0]["h"])
for x, y in zip(cat["x"].astype(int), cat["y"].astype(int)):
    im = image[x - w : x + w, y - h : y + h]
    if im.size == 8 * w * h:
        images.append(im)
    else:
        images.append(np.nan * np.zeros((2 * w, 2 * h)))  # *np.nan)

images = np.array(images)
verboseprint(images)



cat["name"] = [line["ids"].split(" ")[0] for line in cat]
cat["Z"] = [float(line["ids"].split(" ")[1]) for line in cat]
cat["X_IMAGE"] = [int(line["ids"].split(" ")[2]) for line in cat]


dispersion = 4.64
cat = cat.to_pandas()
queries = ["Z<0.01","(Z>0.044 & Z<0.072) | (Z>0.081 & Z<0.117)","(Z>0.285 & Z<0.320) | (Z>0.331 & Z<0.375)","(Z>0.59 & Z<0.682) | (Z>0.696 & Z<0.78) "," (Z>0.926 & Z<0.98)| (Z>0.996 & Z<1.062) ","(Z>1.184 & Z<1.245) | (Z>1.263 & Z<1.338)"]
for q,line in zip(queries,[2060,1908.7,1549.5,1215.67,1033.8,911.8]):
    if len(cat.query(q))>0:
        cat.loc[cat.eval(q), 'em_line'] =  line
        cat.loc[cat.eval(q), 'wave'] = (cat.query(q)['Z']+1)* line
        cat.loc[cat.eval(q), 'X_IMAGE_line'] =  cat.query(q)['X_IMAGE']  + ((cat.query(q)['Z']+1)* line-2060)*dispersion
cat = Table.from_pandas(cat)




cat["var"] = np.nanvar(images, axis=(1, 2))
cat["std"] = np.nanstd(images, axis=(1, 2))
cat["mean"] = np.nanmean(images, axis=(1, 2))
cat["median"] = np.nanmedian(images, axis=(1, 2))
cat["min"] = np.nanmin(images, axis=(1, 2))
cat["max"] = np.nanmax(images, axis=(1, 2))

if new_name is None:
    new_name = "/tmp/regions.csv"
verboseprint(new_name)
long = True

# appertures = blockshaped(physical_region-table['pre_scan'] , 40, 40)
# vars_ = n/p.nanvar(appertures,axis=(1,2))
xprofiles = [np.nanmean(im[:,10:-10],axis=1) for im in images]
if len(im.shape)>2:
    yprofiles = [np.nanmean(im[:,:,:],axis=2) for im in images]
else:
    yprofiles = [np.nanmean(im[:,:],axis=0) for im in images]


if long:
    cat["x_profile"] = xprofiles
    cat["y_profile"] = yprofiles
    
    
cat["max_all"] = [np.nanmax(np.nanmean(im[0,10:-10,:],axis=1)) for im in images]
if 1==1:
    print("match with target file")
cat.sort(["max_all"],reverse=True )





#%%
fig, axes  = plt.subplots(8,3,figsize=(10,14), gridspec_kw={'width_ratios': [0.3, 1,1]},sharex="col")#Nominal.PlotNoise()
axes[0][1].set_title(os.path.basename(path))
for ax, line in zip(axes, cat):
    
    wavelengths = np.linspace(2060-line["X_IMAGE_line"]/dispersion,2060+(1000-line["X_IMAGE_line"])/dispersion,len(line["x_profile"][1]))
    l=ax[0].plot(line["y_profile"][-1])
    ax[0].plot(line["y_profile"][2],alpha=0.2,c=l[0].get_color())
    # ax[1].fill_between(wavelengths,line["x_profile"][0],line["x_profile"][1],alpha=0.3,label="%s - %0.3f "%(line["ids"],line["max_all"]))
*()    ax[1].plot(wavelengths[3:-3],line["x_profile"][1][3:-3],alpha=0.3,label="%s - z=%0.2f "%(line["ids"].split(" ")[0],float(line["ids"].split(" ")[1])))
    l=ax[2].plot(wavelengths,line["x_profile"][-1])
    lims = ax[2].get_ylim()
    ax[2].plot(wavelengths,line["x_profile"][-2],alpha=0.2,c=l[0].get_color())
    ax[2].set_ylim(lims)
    ax[1].set_xlim(((2000,2150)))
    ax[2].set_xlim(((2000,2150)))
    ax[1].legend()
fig.tight_layout()
fig.savefig(path.replace(".csv.fits",".png"))

#%%


# # if 'csv' in new_name:
# cat.write(new_name, overwrite=True, format="ascii.ecsv")
# # else:
cat.write(new_name, overwrite=True)
