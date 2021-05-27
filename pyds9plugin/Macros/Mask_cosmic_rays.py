from astropy.io import fits
import re
from astropy.table import Table
import matplotlib.pyplot as plt
# global np
# global image
# image = ds9
area=[0,-1,0,-1]
verboseprint("area=", area)
threshold = 40000
# n = 1#masking_factor
# global n
n=1
size=0

def MaskCosmicRaysCS(image, cosmics, all=False, size=None):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    # global np
    # import numpy as np
    from tqdm import tqdm, tqdm_gui
    y, x = np.indices((image.shape))
    ly, lx = image.shape
    image = image.astype(float)
    cosmics = cosmics[(cosmics["min_y"] > 0) & (cosmics["min_y"] < ly) & (cosmics["max_y"] > 0) & (cosmics["max_y"] < ly) & (cosmics["max_x"] > 0) & (cosmics["max_x"] < lx)]
    if size is None:
        for i in tqdm(range(len(cosmics))):
            mask = (y > cosmics[i]["min_y"]) & (y < cosmics[i]["max_y"]) & (x < cosmics[i]["max_x"] + cosmics[i]["size_opp"]) & (x > -cosmics[i]["size"] + cosmics[i]["max_x"])
            image[mask] = np.nan  # 0#np.inf#0#np.nan
            # np.where((y>cosmics[i]['min_y']) & (y<cosmics[i]['max_y']) & (x<cosmics[i]['max_x']+cosmics[i]['size_opp']) & (x>-cosmics[i]['size'] + cosmics[i]['max_x']), image, np.nan)
    elif size > 1000:
        for i in tqdm(range(len(cosmics))):
            mask = (y > cosmics[i]["min_y"]) & (y < cosmics[i]["max_y"]) & (x < cosmics[i]["max_x"] + cosmics[i]["size_opp"])
            #    if len(image[mask])<50*3000:
            image[mask] = np.nan
    else:
        for i in tqdm(range(len(cosmics))):
            mask = (y > cosmics[i]["min_y"]) & (y < cosmics[i]["max_y"]) & (x < cosmics[i]["max_x"] + cosmics[i]["size_opp"]) & (x > -size + cosmics[i]["max_x"])
            # if len(image[mask])<50*3000:
            image[mask] = np.nan
    print(image[mask])
    return image

# fitsimage = fits.open(filename)[0]

# AllLine =False
#    locator = mdates.HourLocator(interval=1)
#    locator.MAXTICKS = 50000
ax = plt.gca()
#    ax.xaxis.set_minor_locator(locator)
#    CS = ax.contour(image, levels=threshold, colors='white', alpha=0.5)
cr_places = np.array(ds9 > threshold, dtype=int)
CS = ax.contour(cr_places, levels=1)
plt.close()
print('Numbe of cosmics = ', len(CS.allsegs[0]))
names = ("id", "sizex", "sizey", "len_contour", "max_x", "min_y", "max_y")
cosmics = Table(np.zeros((len(CS.allsegs[0]), len(names))), names=names)
cosmics["id"] = np.arange(len(CS.allsegs[0]))
cosmics["sizex"] = [cs[:, 0].max() - cs[:, 0].min() for cs in CS.allsegs[0]]
cosmics["sizey"] = [cs[:, 1].max() - cs[:, 1].min() for cs in CS.allsegs[0]]
cosmics["len_contour"] = [len(cs[:, 1]) for cs in CS.allsegs[0]]
cosmics["max_x"] = [int(cs[:, 0].max() + n * 1) for cs in CS.allsegs[0]]
cosmics["min_y"] = [int(cs[:, 1].min() - n * 1) for cs in CS.allsegs[0]]
cosmics["max_y"] = [int(cs[:, 1].max() + n * 2) for cs in CS.allsegs[0]]
cosmics["mean_y"] = [int((cs[:, 1].max() + cs[:, 1].max()) / 2) for cs in CS.allsegs[0]]
cosmics["size"] = [n * 50 for cs in CS.allsegs[0]]
cosmics["size_opp"] = [n * 1 for cs in CS.allsegs[0]]
# cosmics[my_conf.exptime[0]] = header[my_conf.exptime[0]]
# cosmics[my_conf.gain[0]] = header[my_conf.gain[0]]
# cosmics["number"] = re.findall(r"\d+", os.path.basename(filename))[-1]
contours = CS.allsegs[0]
imagettes = []
for cs in contours:
    imagettes.append(ds9[int(cs[:, 1].min()) : int(cs[:, 1].max()) + 1, int(cs[:, 0].min()) : int(cs[:, 0].max()) + 1] )
#    for cs in contours:
#        verboseprint(int(cs[:,0].min()),int(cs[:,0].max())+1,int(cs[:,1].min()),int(cs[:,1].max()+1))
# import numpy as np
a=[];

cosmics["cx"] = [np.where(ima == np.nanmax(ima))[1][0] for ima in imagettes]
cosmics["cy"] = [np.where(ima == np.nanmax(ima))[0][0] for ima in imagettes]
cosmics["c0x"] = [int(cs[:, 0].min()) for cs in contours]
cosmics["c0y"] = [int(cs[:, 1].min()) for cs in contours]
cosmics["xcentroid"] = cosmics["c0x"] + cosmics["cx"]
cosmics["ycentroid"] = cosmics["c0y"] + cosmics["cy"]
cosmics["value"] = [ds9[y, x] for x, y in zip(cosmics["xcentroid"], cosmics["ycentroid"])]
# index = (cosmics["ycentroid"] > area[0]) & (cosmics["ycentroid"] < area[1]) & (cosmics["xcentroid"] > area[2]) & (cosmics["xcentroid"] < area[3])
# cosmics = cosmics#[index]
# cosmics = cosmics[(cosmics['max_x']>500) & (cosmics['max_x']<2500)]

mask1 = cosmics["len_contour"] <= 20
mask2 = (cosmics["len_contour"] > 20) & (cosmics["len_contour"] < 2000)
mask3 = (cosmics["len_contour"] > 50) & (cosmics["len_contour"] < 2000)
mask4 = (cosmics["len_contour"] > 200) & (cosmics["len_contour"] < 2000)
cosmics["size"][mask2] = n * 200
cosmics["size"][mask3] = n * 3000
if size > 1000:
    cosmics["size"] = n * 3000  # [ n*3000   for cs in CS.allsegs[0] ]
cosmics["size_opp"][mask4] = n * 3000
cosmics["min_y"][(cosmics["len_contour"] > 200) & (cosmics["len_contour"] < 2000)] -= n * 20
cosmics["max_y"][(cosmics["len_contour"] > 200) & (cosmics["len_contour"] < 2000)] += n * 20
a = cosmics
ds9 = MaskCosmicRaysCS(ds9, cosmics=cosmics)
# savename = DS9backUp + "CSVs/Cosmics_" + os.path.basename(filename)[:-5] + ".csv"
# csvwrite(a, savename)
#
# verboseprint("%i cosmic rays found!" % (len(cosmics)))
#
# fitsimage.data = maskedimage
# name = os.path.dirname(filename) + "/CosmicRayFree/" + os.path.basename(filename)[:-5] + ".CRv_cs.fits"
# fitsimage.header["N_CR"] = len(cosmics)
# fitsimage.header["N_CR1"] = len(cosmics[mask1])
# fitsimage.header["N_CR2"] = len(cosmics[mask2])
# fitsimage.header["N_CR3"] = len(cosmics[mask3])
# fitsimage.header["N_CR4"] = len(cosmics[mask4])
# if "NAXIS3" in fitsimage.header:
#     fits.delval(filename, "NAXIS3")
#     verboseprint("2D array: Removing NAXIS3 from header...")
# fits.setval(filename, "N_CR", value=len(cosmics))
# fits.setval(filename, "N_CR1", value=len(cosmics[mask1]))
# fits.setval(filename, "N_CR2", value=len(cosmics[mask2]))
# fits.setval(filename, "N_CR3", value=len(cosmics[mask3]))
# fits.setval(filename, "N_CR4", value=len(cosmics[mask4]))
# try:
#     fitsimage.header["MASK"] = 100 * float(np.sum(~np.isfinite(maskedimage[:, 1053:2133]))) / (maskedimage[:, 1053:2133].shape[0] * maskedimage[:, 1053:2133].shape[1])
#     fits.setval(filename, "MASK", value=100 * float(np.sum(~np.isfinite(maskedimage[:, 1053:2133]))) / (maskedimage[:, 1053:2133].shape[0] * maskedimage[:, 1053:2133].shape[1]))
# except ZeroDivisionError:
#     fitsimage.header["MASK"] = 100 * float(np.sum(~np.isfinite(maskedimage))) / (maskedimage.shape[0] * maskedimage.shape[1])
#     fits.setval(filename, "MASK", value=100 * float(np.sum(~np.isfinite(maskedimage))) / (maskedimage.shape[0] * maskedimage.shape[1]))
