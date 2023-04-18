
# from astropy.table import hstack
from astropy.table import Table

# paths = get(d,"Path of the images you want to stack:")
paths="/Users/Vincent/Documents/shared/test/stack????????.fits"
print(paths)
if paths!="":
    files=globglob(paths)
else:
    files=get_filename(d, All=True, sort=False)
files=get_filename(d, All=True, sort=False)
files.sort()
region = getregion(d, quick=True, message=False, selected=True)
print("region = ", region)
Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
new_image = np.hstack([fits.open(f)[0].data[Yinf:Ysup, Xinf:Xsup] for f in files]) 
print(new_image)
name = files[0].replace(".fits","_TF.fits")
fitswrite(new_image,name)
cat_name = name.replace(".fits","_cat.fits")
x, y = "X_IMAGE", "Y_IMAGE"



param_dict = {"DETECT_THRESH":10,
"GAIN":0,
"DETECT_MINAREA":10,
"DEBLEND_NTHRESH":64,
"DEBLEND_MINCONT":0.3,
"PHOT_APERTURES":"5,20,80",
"CLEAN":0,
"CLEAN_PARAM":0,
"CATALOG_NAME":cat_name,
}

run_sep(name, name, param_dict)
cat = Table.read(cat_name)
cat.sort("X_IMAGE")




d.set("frame new; file "+name)
if 1==1:
    reg_file="/tmp/test.reg"
    create_ds9_regions(
        [cat["X_IMAGE"]],
        [cat["Y_IMAGE"]],
        more=[
            cat["A_IMAGE"] * cat["KRON_RADIUS"] / 2,
            cat["B_IMAGE"] * cat["KRON_RADIUS"] / 2,
            cat["THETA_IMAGE"],
        ],
        form=["ellipse"] * len(cat),
        save=True,
        ID=[np.around(cat["FWHM_IMAGE"], 1).astype(str)],
        color=["white"] * len(cat),
        savename=reg_file,
        font=10,
    )
    d.set("regions " + reg_file)


else:
    command = """catalog import FITS %s ; catalog x %s ;
            catalog y %s ; catalog symbol shape
            ellipse  ; catalog symbol Size
            "$A_IMAGE * $KRON_RADIUS/2" ; catalog symbol
            Size2 "$B_IMAGE * $KRON_RADIUS/2"; catalog
            symbol angle "$THETA_IMAGE" ; catalog symbol Text "$FWHM_IMAGE" ; mode catalog;  """
    d.set(f_string(command % (cat_name, x, y)))






















# # créer la figure et les sous-graphes
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

# # définir les couleurs des axes y
# color1 = 'k'
# color2 = 'grey'





# fig = plt.figure()
# fig.set_figheight(6)
# fig.set_figwidth(6)
 
# ax1 = plt.subplot2grid(shape=(3, 3), loc=(0, 0), colspan=3)
# ax2 = plt.subplot2grid(shape=(3, 3), loc=(1, 0), colspan=1)
# ax3 = plt.subplot2grid(shape=(3, 3), loc=(1, 1), colspan=1)
# ax4 = plt.subplot2grid(shape=(3, 3), loc=(1, 2), colspan=1)
# ax.imshow(new_image, cmap='gray',log=True)
# for i in range(len(cat)):
#     ellipse = Ellipse((cat["X_IMAGE"], cat["Y_IMAGE"]), cat["A_IMAGE"], cat["B_IMAGE"], cat["ANGLE"], edgecolor='r', facecolor='none')
#     ax.add_artist(ellipse)
# plt.tight_layout()
# plt.show()










# # tracer les courbes sur chaque sous-graphe
# ax1.plot(x, cat["FWHM_IMAGE"],"o", color=color1)
# ax1.set_xlim(ax1.get_xlim())
# ax1.set_ylim(ax1.get_ylim())
# PlotFit1D(x, cat["FWHM_IMAGE"],deg=2, ax=ax1,c=color1)
# ax1.set_ylabel('FWHM (pix)', color=color1)
# ax1.tick_params(axis='y', labelcolor=color1)
# ax1.set_title('Subplot 1')
# ax1b = ax1.twinx()
# # ax1b.plot(x, cat["flux"], color=color2)
# # ax1b.set_ylabel('exp(x)', color=color2)
# # ax1b.tick_params(axis='y', labelcolor=color2)

# ax2.errorbar(x, cat["MAG_APER_0"],fmt="o",ls=":", yerr=cat["MAGERR_APER_0"],color=color2)
# ax2.errorbar(x, cat["MAG_APER_1"],fmt="o",ls="--",yerr=cat["MAGERR_APER_1"], color=color2)
# ax2.errorbar(x, cat["MAG_APER_2"],fmt="o-", yerr=cat["MAGERR_APER_2"],color=color2)
# ax2.set_yscale("log")
# # ax2.set_ylabel('exp(x)', color=color2)
# ax2.tick_params(axis='y', labelcolor=color2)
# ax2.set_title('Subplot 2')
# ax2b = ax2.twinx()
# ax2b.plot(x, cat["peak"], color=color1)
# # ax2b.set_ylabel('sin(x)', color=color1)
# ax2b.tick_params(axis='y', labelcolor=color1)

# ax3.plot(x, cat["A_IMAGE"], color=color1)
# ax3.plot(x, cat["B_IMAGE"], color=color1)
# # ax3.set_ylabel('cos(x)', color=color1)
# ax3.tick_params(axis='y', labelcolor=color1)
# ax3.set_title('Subplot 3')
# ax3b = ax3.twinx()
# ax3b.plot(x, cat["THETA_IMAGE"],"o", color=color2)
# ax3b.set_ylabel('Theta', color=color2)
# ax3b.tick_params(axis='y', labelcolor=color2)

# ax4.plot(x, cat["cxx"], color=color2)
# ax4.plot(x, cat["cyy"], color=color2)
# ax4.set_ylabel('cxx, cyy)', color=color2)
# ax4.tick_params(axis='y', labelcolor=color2)
# ax4.set_title('Subplot 4')
# ax4b = ax4.twinx()
# ax4b.plot(x, cat["cxy"], color=color1)
# ax4b.set_ylabel('cxy', color=color1)
# ax4b.tick_params(axis='y', labelcolor=color1)
# fig.tight_layout()
# plt.show()
