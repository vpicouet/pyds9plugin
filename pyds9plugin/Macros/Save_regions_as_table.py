from astropy.table import Table
from pyds9plugin.DS9Utils import blockshaped


new_name = get(d, 'Path where you want to save the region catalog.', exit_=True)
verboseprint(new_name)

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
verboseprint(cat)
images = []
w = int(cat[0]["w"])
h = int(cat[0]["h"])
for x, y in zip(cat["x"].astype(int), cat["y"].astype(int)):
    im = image[x - w : x + w, y - h : y + h]
    if im.size == 4 * w * h:
        images.append(im)
    else:
        images.append(np.nan * np.zeros((2 * w, 2 * h)))  # *np.nan)

images = np.array(images)
verboseprint(images)
cat["var"] = np.nanvar(images, axis=(1, 2))
cat["std"] = np.nanstd(images, axis=(1, 2))
cat["mean"] = np.nanmean(images, axis=(1, 2))
cat["median"] = np.nanmedian(images, axis=(1, 2))
cat["min"] = np.nanmin(images, axis=(1, 2))
cat["max"] = np.nanmax(images, axis=(1, 2))
if new_name is None:
    new_name = "/tmp/regions.csv"
verboseprint(new_name)
# long = True

# appertures = blockshaped(physical_region-table['pre_scan'] , 40, 40)
# vars_ = n/p.nanvar(appertures,axis=(1,2))
# xprofile = [np.nanmean(im,axis=1) for im in images]
# yprofile = [np.nanmean(im,axis=2) for im in images]
# if long:
#     cat["x_profile"] =  Column([xprofile], name="x_profile")   
#     cat["y_profile"] =  Column([xprofile], name="y_profile")   

# if 1==1:
#     print("match with target file")




if 'csv' in new_name:
    cat.write(new_name, overwrite=True)#, format="ecsv")
else:
    cat.write(new_name, overwrite=True)
