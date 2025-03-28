import sys, os
FBPIPE_path = "/Users/Vincent/Nextcloud/LAM/FIREBALL/FireBallPipe"
FBPIPE_path = "/Users/Vincent/GitHub/FireBallPipe"
os.chdir(FBPIPE_path)
sys.path.insert(1, './Calibration')
sys.path.insert(1, FBPIPE_path)

# from guider2UV.guider2UV import Guider2UV, diff_skycoord, fit_model#, plot_fit
# from guider2UV.MaskAstrometry import LocalScienceMaskProjector
# from Calibration.mapping import Mapping
# from mapping_mask_det_2022 import create_mapping

from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from pyds9plugin.DS9Utils import *  # DS9n, plot_surface, getregion
from tqdm import trange, tqdm
from scipy.optimize import curve_fit
import os, sys
from pyds9plugin.Macros.Fitting_Functions.functions import slit, smeared_slit

from pyds9plugin.DS9Utils import create_ds9_regions
slitm = slit


def change_val_list(popt, val, new_val):
    popt1 = list(map(lambda item: new_val if item == val else item, popt))
    return popt1

#possibility 1

# xx,yy,fwhm=[],[],[]
# names=[]
# ptps=[]


def Measure_PSF_slits(image, regs, plot_=True, filename=None,slit_width=None,ds=1):
    cat = Table(
        names=[
            "name",
            "color",
            "line",
            "x",
            "y",
            "w",
            "h",
            "amp_x",
            "lx",
            "x0",
            "fwhm_x",
            "off_x",
            "amp_y",
            "ly",
            "y0",
            "fwhm_y",
            "off_y",
            "smearing",
            "fwhm_x_unsmear",
            "lx_unsmear",
            "x0_unsmear",
            "amp_x_unsmear",
        ],
        dtype=[str, str] + [float] * 20,
    )
    for region in tqdm(regs[:]):
        x, y = int(region.xc), int(region.yc)
        try:
            w, h = int(region.w), int(region.h)
        except AttributeError:
            break
        if image.size < 3251072:
            limx1, limx2 = 20, 1040
        else:
            limx1, limx2 = 1100, 2130
        if (x > limx1) & (y < 1990) & (y > 0) & (x < limx2):
            # if x > 0:  # & (y < 1950) & (y > 250) & (x < 2050):
            x_inf, x_sup, y_inf, y_sup = lims_from_region(region=region, coords=None)
            n = 15
            # subim1 = image[y_inf - n : y_sup + n, x_inf:x_sup]
            # subim2 = image[y_inf:y_sup, x_inf - n : x_sup + n]
            subim3 = image[y_inf - n : y_sup + n, x_inf - n : x_sup + n][::-1, :]
            subim1 = subim3[:, n:-n]
            subim2 = subim3[n:-n, :]
            # y_spatial = np.nanmedian(subim1, axis=1)
            # y_spectral = np.nanmedian(subim2, axis=0)  # [::-1]
            y_spatial = np.nanmean(subim1, axis=1)[::-1]
            y_spectral = np.nanmean(subim2, axis=0)

            x_spatial = np.arange(len(y_spatial))
            x_spectral = np.arange(len(y_spectral))

            if plot_:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))
            else:
                ax2 = None
                ax3 = None
            if slit_width is None:
                ds=0.3
                if "tilted" in filename:
                    slit_width = 6.5
                if "F1" in filename:
                    slit_width = 6.5
                if "F2" in filename:
                    slit_width = 6
                if "F3" in filename:
                    slit_width = 6
                if "F4" in filename:
                    slit_width = 8.5
                if "QSO" in filename:
                    slit_width = 6
                else:
                    slit_width = 6
        
                slit_min, slit_max, slit_med = slit_width-ds,slit_width+ds,slit_width
                print(filename,":\n", slit_min, slit_max, slit_med)
                
            else:
                slit_min, slit_max, slit_med = slit_width-ds,slit_width+ds,slit_width

            P0 = [
                y_spectral.ptp(),
                slit_med,
                len(y_spectral) / 2,
                1.7,
                np.median(y_spectral),
                -1.2,
            ]
            bounds = [
                [0.7 * y_spectral.ptp(), slit_min, 0, 0, np.nanmin(y_spectral), -6],
                [
                    y_spectral.ptp(),
                    # len(y_spectral),
                    slit_max,
                    len(y_spectral),
                    10,
                    np.nanmax(y_spectral),
                    6,
                ],
            ]
            try:
                popt_spectral_deconvolved, pcov = curve_fit(smeared_slit, x_spectral, y_spectral, p0=P0, bounds=bounds)
            except (ValueError, RuntimeError) as e:
                popt_spectral_deconvolved = [0.1] * 6
            # print(popt_spectral_deconvolved)
            # bounds = [[0.7*y_spatial.ptp(), 10, 0, 0, np.nanmin(y_spatial)], [y_spatial.ptp(),  len(y_spatial), len(y_spatial), 10, np.nanmax(y_spatial)]]
            # popt_spatial = PlotFit1D(x_spatial,y_spatial,deg=slitm, plot_=plot_,ax=ax2,P0=[y_spatial.ptp(),20,x_spatial.mean()+1,2,y_spatial.min()],c='k',lw=2,bounds=bounds)['popt']
            slit_length_min = 0  # 20
            try:
                bounds = [
                    [
                        0.7 * y_spatial.ptp(),
                        slit_length_min,
                        0,
                        0,
                        np.nanmin(y_spatial),
                    ],
                    [
                        y_spatial.ptp(),
                        len(y_spatial),
                        len(y_spatial),
                        50,
                        np.nanmax(y_spatial),
                    ],
                ]
                popt_spatial = PlotFit1D(
                    x_spatial,
                    y_spatial,
                    deg=slitm,
                    plot_=False,
                    ax=ax2,
                    P0=[y_spatial.ptp(), 22, x_spatial.mean() + 1, 2, y_spatial.min()],
                    c="k",
                    lw=2,
                    bounds=bounds,
                )["popt"]
                bounds = [
                    [0.7 * y_spectral.ptp(), slit_min, 0, 0, np.nanmin(y_spectral)],
                    [y_spectral.ptp(), slit_max, len(y_spectral), 20, np.nanmax(y_spectral)],
                ]
                popt_spectral = PlotFit1D(
                    x_spectral,
                    y_spectral,
                    deg=slitm,
                    plot_=False,
                    ax=ax3,
                    P0=[
                        y_spectral.ptp(),
                        slit_med,
                        x_spectral.mean() + 1,
                        2,
                        y_spectral.min(),
                    ],
                    c="k",
                    ls="--",
                    lw=0.5,
                    bounds=bounds,
                )["popt"]
                popt_spatial = abs(np.array(popt_spatial))
                popt_spectral = abs(np.array(popt_spectral))
            except ValueError as e:
                print("error: ", region.id, e)
                popt_spatial, popt_spectral = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
            
            if region.color == "red":
                line = 214
            elif region.color == "blue":
                line = 203
            elif (region.color == "yellow") | (region.color == "green"):
                line = 206
            else:
                line = 206
                # line = -99
            # print(region.id)
            cat.add_row(
                [   region.id,
                    region.color,
                    line,
                    region.xc,
                    region.yc,
                    region.w,
                    region.h,
                    *popt_spectral,
                    *popt_spatial,
                    popt_spectral_deconvolved[-1],
                    popt_spectral_deconvolved[-3],
                    popt_spectral_deconvolved[1],
                    popt_spectral_deconvolved[2],
                    popt_spectral_deconvolved[0],
                ]
            )

            # amp, l, x0, FWHM
            # fwhm.append(fwhmi)
            # ptps.append(y.ptp())

            if plot_:
                # lx, ly = subim3.shape
                extendx = x_inf - n + np.arange(len(x_spectral) + 1) + 0.5
                extendy = y_inf - n + np.arange(len(x_spatial) + 1) + 0.5
                extend = [extendx.min(), extendx.max(), extendy.min(), extendy.max()]
                x_spectral_c = x_inf + 1 - n + np.arange(len(x_spectral))
                x_spatial_c = y_inf + 1 - n + np.arange(len(x_spatial))
                # print(extendx)
                # print(x_spectral_c)
                # print(extendy)
                # print(x_spatial_c)
                ax3.grid(alpha=0.1)
                ax1.grid(alpha=0.1)
                ax2.grid(alpha=0.1)
                # x_spectral_c = x_spectral
                # x_spatial_c = x_spatial
                ax1.imshow(subim3, extent=extend)
                ax1.axvline(extendx[n], c="k")
                ax1.axvline(extendx[-n], c="k")
                ax1.axhline(extendy[n], c="k")
                ax1.axhline(extendy[-n], c="k")
                xc_smearing = x_inf - n + 1 + cat[-1]["x0_unsmear"]
                xc = x_inf - n + 1 + cat[-1]["x0"]
                yc = y_inf - n + 1 + cat[-1]["y0"]
                ax1.plot([xc], [yc], "or")
                ax1.plot([xc_smearing], [yc], "ob")
                ax2.axvline(y_inf - n + 1 + cat[-1]["y0"], c="k")
                ax3.axvline(x_inf - n + 1 + cat[-1]["x0_unsmear"], c="k")
                ax3.axvline(x_inf - n + 1 + cat[-1]["x0"], c="k")

                ax1.set_title(
                    "Slit #%s: x=%i, y=%i" % (region.id, region.xc, region.yc)
                )

                # print("y_inf,n, cat[-1]['x0'] = ", x_inf, n, cat[-1]["y0"])
                ax2.set_title("center = %0.1f" % (yc))
                ax3.set_title("centers = %0.1f -  %0.1f" % (xc, xc_smearing,))
                ax2.axvline(yc, c="k")
                ax3.axvline(xc_smearing, c="k")
                ax3.axvline(xc, c="k")
                ax2.plot(x_spatial_c, y_spatial, ":k", marker=".")
                ax3.plot(x_spectral_c, y_spectral, ":k", marker=".")
                ax2.plot(x_spatial_c, slitm(x_spatial, *popt_spatial), "k", lw=2)
                ax2.plot(
                    x_spatial_c,
                    slitm(
                        x_spatial,
                        *change_val_list(popt_spatial, popt_spatial[1], 0.001)
                    ),
                    "-k",
                    lw=0.5,
                    label="Slit length=%0.1f\nFWHM=%0.1f"
                    % (popt_spatial[1], popt_spatial[-2]),
                )
                ax2.plot(
                    x_spatial_c,
                    slitm(
                        x_spatial,
                        *change_val_list(popt_spatial, popt_spatial[-2], 0.001)
                    ),
                    "-k",
                    lw=0.5,
                )

                ax3.plot(
                    x_spectral_c,
                    slitm(
                        x_spectral,
                        *change_val_list(popt_spectral, popt_spectral[1], 0.001)
                    ),
                    "--k",
                    lw=0.5,
                    label="Slit width=%0.1f\nFWHM=%0.1f"
                    % (popt_spectral[1], popt_spectral[-2]),
                )
                ax3.plot(
                    x_spectral_c,
                    slitm(
                        x_spectral,
                        *change_val_list(popt_spectral, popt_spectral[-2], 0.001)
                    ),
                    "-k",
                    lw=0.5,
                )

                ax3.plot(
                    x_spectral_c,
                    smeared_slit(x_spectral, *popt_spectral_deconvolved),
                    "-k",
                    lw=2,
                    label="Unsmeared\nFWHM=%0.1f\nSmearing=%0.1f\nslit_width=%0.1f"
                    % (
                        popt_spectral_deconvolved[3],
                        popt_spectral_deconvolved[-1],
                        popt_spectral_deconvolved[1],
                    ),
                )
                try:
                    ax3.plot(
                        x_spectral_c,
                        smeared_slit(
                            x_spectral,
                            *change_val_list(
                                change_val_list(
                                    popt_spectral_deconvolved,
                                    popt_spectral_deconvolved[-1],
                                    0.1,
                                ),
                                popt_spectral_deconvolved[1],
                                0.1,
                            )
                        ),
                        "-k",
                        lw=0.5,
                    )
                except ValueError:
                    pass
                ax3.plot(x_spectral_c, y_spectral, ":k", lw=2)

                # ax2.plot(x[mask],y_conv[mask],'-',)#label='FWHM = %0.1f\nslit length=%0.1f'%(popt[0],popt[-1]))
                ax2.set_xlabel("y")
                ax1.set_ylabel("y")
                ax1.set_xlabel("x")
                ax3.set_xlabel("x")
                ax2.legend(loc="center left", fontsize=10)
                ax3.legend(loc="center left", fontsize=10)
                ax2.set_xlim((x_spatial_c.min(), x_spatial_c.max()))
                ax3.set_xlim((x_spectral_c.min(), x_spectral_c.max()))
                fig.tight_layout()
                if os.path.exists(os.path.dirname(filename) + "/fits/"):
                    path = os.path.dirname(filename) + "/fits/"
                else:
                    path = "/tmp/"
                if region.id == "":
                    plt.savefig(
                        path + "%s_%s.png"
                        % (os.path.basename(filename).split(".fits")[0], int(region.yc))
                    )
                else:
                    plt.savefig(
                        path + "%s_%s_%s.png"
                        % (
                            region.id,
                            line,
                            os.path.basename(filename).split(".fits")[0],
                        )
                    )
                # plt.show()
                plt.close()
    print(filename.replace(".fits", ".csv"))
    try:
        cat["l203"] = False
    except TypeError:
        pass
    else:
        cat["l214"] = False
        cat["l206"] = False
        cat["l203"][cat["line"] == 203.0] = True
        cat["l214"][cat["line"] == 214.0] = True
        cat["l206"][cat["line"] == 206.0] = True
        cat["X_IMAGE"] = cat["x"] - cat["w"] / 2 - n + cat["x0"]
        cat["X_IMAGE_unsmear"] = cat["x"] - cat["w"] / 2 - n + cat["x0_unsmear"]
        cat["Y_IMAGE"] = cat["y"] - cat["h"] / 2 - n + cat["y0"]
        # TODO add xinf yinf
    cat.write(filename.replace(".fits", ".csv"), overwrite=True)
    return cat, filename






plot_=True
d = DS9n()

if 1==0:
    d.set("regions select all")
    regs = getregion(d, selected=True)
    image = d.get_pyfits()[0].data
    filename = get_filename(d)
    cat, filename = Measure_PSF_slits(image, regs, filename=filename,plot_=plot_)

else:
    image = fits.open(filename)[0].data
    regs = getregion(d,file=filename.replace(".fits",".reg"),message=False)
    print(filename.replace(".fits",".reg"))
    cat, filename = Measure_PSF_slits(image, regs, filename=filename,plot_=plot_)



# sys.exit()

#%%

Field = os.path.basename(filename).split("_")[0]
print(Field)
mag= create_mapping(Field,file=filename.replace(".fits", ".csv"))#Table.read(f) )


slit_path = FBPIPE_path + "/Calibration/Targets/2022/targets_%s.csv"%(Field)
print(slit_path)
slits = Table.read(slit_path)

map_name = FBPIPE_path + '/Calibration/Mappings/2023/mask_to_det_mapping/mapping-mask-det-w-0-%s_%s.pkl'%(Field,datetime.datetime.now().strftime("%y%m%d"))
mapping = Mapping(map_name)


wcolors = {0.20255:'blue', 0.20619:'green', 0.21382:'red'}
ws = [0.20255, 0.20619, 0.21382]
colors = [wcolors[i] for i in ws]
if "Internal-count" in slits.colnames:
    ID =slits["Internal-count"]
else:
    ID =slits["name"]
offset =0 #1088#0#1072
# print(slits.colnames)
# if field in ["F1","F4"]:
#     create_ds9_regions(xdetpix, ydetpix, form=['box']*3, radius=[10,20], save=True, 
#                           savename=path.replace(".fits","correct.reg"), color = colors, ID=[ID]*3)
# else:
#     create_ds9_regions(xdetpix, ydetpix, form=['box']*3, radius=[10,32], save=True, 
#                           savename=path.replace(".fits","correct.reg"), color = colors, ID=[ID]*3)

for offset, ftype in zip([0,1088],["OS","noOS"]):

    xdetpix = []
    ydetpix = []
    x,y =  slits["x_mm"], slits["y_mm"] if "x_mm" in slits.colnames() else slits["xmm"], slits["ymm"]

    for w in ws:
        # if "x_mm" in slits.colnames:
        xydetpix = mapping.map(w, x,y)
        create_ds9_regions(xdetpix, ydetpix, form=['box']*3, radius=[10,size_slit], save=True, 
                                savename="/Users/Vincent/Github/pyds9plugin/pyds9plugin/regions/2023_flight/fields/"+ os.path.basename(filename).replace(".fits","_%s_%i.reg"%(w)), color = colors, ID=[ID]*3)
        xdetpix.append(xydetpix[0]-offset)
        ydetpix.append(xydetpix[1])
    size_slit = 20 if Field in ["F1","F4"] else 32

    create_ds9_regions(xdetpix, ydetpix, form=['box']*3, radius=[10,size_slit], save=True, 
                            savename=filename.replace(".fits","_%s_Zn.reg"%(ftype)), color = colors, ID=[ID]*3)
    create_ds9_regions(xdetpix, ydetpix, form=['box']*3, radius=[10,size_slit], save=True, 
                            savename="/Users/Vincent/Github/pyds9plugin/pyds9plugin/regions/2023_flight/fields/"+ os.path.basename(filename).replace(".fits","_%s_Zn.reg"%(ftype)), color = colors, ID=[ID]*3)



#if frame.lower() == "restframe":
    verboseprint("Working in rest frame wevelength")
    w = 1215.67
    wavelength = (1 + redshift) * w  # * 1e-4
    yi, xi = mapping.map(wavelength, x, y, inverse=False)
    create_ds9_regions(xdetpix, ydetpix, form=['box']*3, radius=[10,size_slit], save=True, 
                            savename="/Users/Vincent/Github/pyds9plugin/pyds9plugin/regions/2023_flight/fields/"+ os.path.basename(filename).replace(".fits","_%s_Lya.reg"%(ftype)), color = colors, ID=[ID]*3)

    # if frame.lower() == "observedframe":
    #     verboseprint("Working in observed frame wevelength")
    #     y, x = mapping.map(w, xmask, ymask, inverse=False)


    xdetpix = []
    ydetpix = []
    w=0.20619
    if "x_mm" in slits.colnames:
        xydetpix = mapping.map(w, slits["x_mm"], slits["y_mm"])
    else:
        xydetpix = mapping.map(w, slits["xmm"], slits["ymm"])
    xdetpix.append(xydetpix[0]-offset)
    ydetpix.append(xydetpix[1])
    print(len(slits),len(ydetpix))
    print(slits,ydetpix)

    create_ds9_regions([ [1000-offset] for i in range(len(slits))] , [[yi] for yi in ydetpix[0]], form=['projection']*len(slits), radius=[[2000-offset for i in range(len(slits))],  [yi for yi in ydetpix[0]] ], save=True, 
                                savename=filename.replace(".fits","_%s_proj.reg"%(ftype)),color=["yellow"]*len(slits))

    create_ds9_regions([ [1000-offset] for i in range(len(slits))] , [[yi] for yi in ydetpix[0]], form=['projection']*len(slits), radius=[[2000-offset for i in range(len(slits))],  [yi for yi in ydetpix[0]] ], save=True, 
                                savename="/Users/Vincent/Github/pyds9plugin/pyds9plugin/regions/2023_flight/fields/"+  os.path.basename(filename).replace(".fits","_%s_proj.reg"%(ftype)),color=["yellow"]*len(slits))













#%%

    # d.set("regions delete all")
    # create_ds9_regions(
    #     [cat["X_IMAGE"]],
    #     [cat["Y_IMAGE"]],
    #     # radius=[table_to_array(cat["h", "w"]).T],
    #     radius=[np.array(cat["lx_unsmear"]), np.array(cat["ly"])],
    #     save=True,
    #     savename= filename.replace(".fits","_c.reg"),
    #     form=["box"],
    #     color=cat["color"],
    #     ID=[cat["name"]],#None,  # cat["name"],
    # )
    # d.set("regions %s" % (filename.replace(".fits","_c.reg")))


# def plot_res(cat, filename):
#     field = os.path.basename(filename)[:2]
#     from matplotlib.colors import LogNorm

#     mask = (cat["fwhm_y"] > 0.5) & (
#         cat["fwhm_x"] > 0.5
#     )  # & (cat['x'] <1950)  & (cat['x'] >50) #& (cat['fwhm_x'] < 8) & (cat['fwhm_y'] < 8) & (cat['fwhm_x_unsmear'] < 8)
#     # = (cat['fwhm_x_unsmear']>0.5)&
#     fig, axes = plt.subplots(
#         2,
#         2,
#         sharex="col",
#         sharey="row",
#         gridspec_kw={"height_ratios": [1, 3], "width_ratios": [3, 1]},
#         figsize=(8, 5),
#     )
#     ax0, ax1, ax2, ax3 = axes.flatten()
#     m = "o"
#     size = 3
#     print(cat["fwhm_x_unsmear"], cat["fwhm_y"], cat["fwhm_x"])
#     print(cat["fwhm_x_unsmear"][mask])
#     norm = LogNorm(
#         vmin=np.min(cat["fwhm_x_unsmear"][mask]), vmax=np.max(cat["fwhm_y"][mask])
#     )
#     im = ax2.scatter(
#         cat["y"][mask] - 50, cat["x"][mask], c=cat["fwhm_y"][mask], s=50, norm=norm
#     )  # ,marker=',',)
#     ax2.scatter(
#         cat["y"][mask], cat["x"][mask], c=cat["fwhm_x"][mask], s=50, norm=norm
#     )  # ,vmin=3,vmax=6)
#     ax2.scatter(
#         cat["y"][mask] + 50,
#         cat["x"][mask],
#         c=cat["fwhm_x_unsmear"][mask],
#         s=50,
#         norm=norm,
#         marker=">",
#     )  # ,vmin=3,vmax=6)
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
#     from matplotlib.ticker import LogFormatter

#     cax = make_axes_locatable(ax2).append_axes("right", size="2%", pad=0.02)
#     fig.colorbar(
#         im,
#         cax=cax,
#         orientation="vertical",
#         ticks=[1, 2, 3, 4, 5, 6, 7],
#         format=LogFormatter(10, labelOnlyBase=False),
#     )

#     # cat['color'][(cat['color']==np.ma.core.MaskedConstant).mask]='green'
#     ax3.set_ylim(ax2.get_ylim())
#     # for color in ['red']:#,'yellow','green']:
#     for color in ["red", "yellow", "blue"]:
#         mask = cat["color"] == color  # .mask
#         p = ax0.plot(
#             cat[mask]["y"],
#             cat[mask]["fwhm_y"],
#             marker="s",
#             lw=0,
#             ms=size,
#             c=color.replace("yellow", "orange"),
#         )
#         fit = PlotFit1D(
#             cat[mask]["y"],
#             cat[mask]["fwhm_y"],
#             deg=2,
#             plot_=True,
#             ax=ax0,
#             c=p[0].get_color(),
#         )
#         ax0.plot(
#             cat["y"], cat["fwhm_x"], m, c=color.replace("yellow", "orange"), alpha=0.3
#         )
#         p = ax0.plot(
#             cat["y"][mask],
#             cat["fwhm_x_unsmear"][mask],
#             marker=">",
#             lw=0,
#             ms=size,
#             c=color.replace("yellow", "orange"),
#         )
#         fit = PlotFit1D(
#             cat["y"][mask],
#             cat["fwhm_x_unsmear"][mask],
#             deg=2,
#             plot_=True,
#             ax=ax0,
#             c=p[0].get_color(),
#             ls="--",
#         )
#         ax0.set_ylim((1.5, 8))
#         # c=abs(fwhm[mask]),s=np.array(ptps)[mask]*50
#         # ax2.scatter(cat['x'],cat['fwhm_y'],'.')
#         p = ax3.plot(
#             cat[mask]["fwhm_y"],
#             cat[mask]["x"],
#             m,
#             marker="s",
#             lw=0,
#             ms=size,
#             c=color.replace("yellow", "orange"),
#         )
#         fit = PlotFit1D(
#             cat[mask]["x"],
#             cat[mask]["fwhm_y"],
#             deg=1,
#             plot_=False,
#             ax=ax3,
#             c=p[0].get_color(),
#         )
#         ax3.plot(
#             fit["function"](cat["x"][mask]), cat["x"][mask], c=p[0].get_color(), ls="-"
#         )
#         # ax3.plot(cat['fwhm_x'],cat['x'],m)
#         p = ax3.plot(
#             cat[mask]["fwhm_x_unsmear"],
#             cat[mask]["x"],
#             marker=">",
#             ms=size,
#             lw=0,
#             c=color.replace("yellow", "orange"),
#         )
#         fit = PlotFit1D(
#             cat["x"][mask],
#             cat["fwhm_x_unsmear"][mask],
#             deg=1,
#             plot_=False,
#             ax=ax3,
#             c=p[0].get_color(),
#             ls="--",
#         )
#         ax3.plot(
#             fit["function"](cat["x"][mask]), cat["x"][mask], c=p[0].get_color(), ls=":"
#         )
#     ax3.set_xlim((1.5, 8))
#     ax1.axis("off")
#     ax1.plot(-1, -1, label="Spatial FWHM", marker="s", lw=0, c="k")
#     ax1.plot(-1, -1, m, label="Spectral FWHM", c="k")
#     ax1.plot(-1, -1, label="Unsmeared\nspectral FWHM", marker=">", lw=0, c="k")
#     ax1.plot(-1, -1, m, label="lambda=213", lw=0, c="r")
#     ax1.plot(-1, -1, m, label="lambda=206", c="orange")
#     ax1.plot(-1, -1, m, label="lambda=202", lw=0, c="g")
#     ax2.set_xlim((0, 2000))
#     ax0.set_ylabel("Resolution")
#     ax3.set_xlabel("Resolution")
#     ax2.set_xlabel("Y")
#     ax2.set_ylabel("X")
#     ax1.legend(fontsize=7, title=field)
#     fig.tight_layout()
#     plt.show()


# plot_res(cat,filename)



# sys.exit()

