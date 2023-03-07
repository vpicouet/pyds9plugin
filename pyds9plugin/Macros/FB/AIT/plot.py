#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 23:42:13 2018

@author: Vincent
"""
#%%
from __future__ import division
import os, sys
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import dblquad
from scipy import ndimage, linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

ft = 16
alpha_2018 = 0.3


# np.sqrt(
#     (1.7 / 2.35) ** 2 + (2.4 / 2.3) ** 2 + (2.8 / 2.35) ** 2 + (0.7 / 2.35) ** 2
# )  # *2.355


LO_2018 = 1.8
guider_2018 = 1.8
FC_2018 = 2.4

detector_2018 = 0.33
spectro_2018 = 4
total_instru = np.sqrt(guider_2018 ** 2 + detector_2018 ** 2 + LO_2018 ** 2 + FC_2018 ** 2 + spectro_2018 ** 2)
autocoll_2018 = 4.95  # 12.3*1.15# pix
flight_2018 = 1 + autocoll_2018
flight_r_2018 = 7

# plt.style.use('/Users/Vincent/anaconda2/pkgs/matplotlib-2.1.2-py27h6d6146d_0/lib/python2.7/site-packages/matplotlib/mpl-data/stylelib/presentation.mplstyle')
# plt.figure(figsize=(10, 6))
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

l = 5
ax1.fill_between(
    [-0.5, 5.5],
    detector_2018 - 0.5,
    1.1 * spectro_2018,
    color="0.9",
    cmap="coolwarm",
    label="Subsystems (transparent:2018, opaque:2023)",
)  # , zorder=0)

spectro_2022 = np.array([5, 5])#7.7
FC_2022_2mm = np.array([5.82, 6.75])
# FC_2022_2mm_avg = 6.75
FC_2022_4mm = np.array([2.01, 4.22])


FC_2023 = np.array([1.8,1.8])
spectro_2023 = np.array([5, 5])#7
# FC_2022_4mm_avg = 4.22


ax1.hlines(guider_2018, -0.5, 0.5, linewidth=l, color="k", alpha=alpha_2018)
ax1.hlines(LO_2018, 0.5, 1.5, linewidth=l, color="k", alpha=alpha_2018)
ax1.hlines(FC_2018, 1.5, 2.5, linewidth=l, color="k", alpha=alpha_2018)
# spectro =   # before was 2.82.8#1.5*2.25#2.2pix or 6pix=75mu=6ercsec
ax1.plot([2.5, 3.5], [3.3, 3.3], linewidth=l, color="k", alpha=alpha_2018)#5
ax1.hlines(detector_2018, 3.5, 4.5, linewidth=l, color="k", alpha=alpha_2018)
ax1.hlines(
    autocoll_2018,
    -0.5,
    4.5,
    linewidth=l,
    color="orange",
    linestyles=u"dotted", label=r"Measured end-to-end resolution",
    alpha=alpha_2018,
)
ax1.hlines(
    total_instru,
    -0.5,
    4.5,
    linewidth=l,
    color="g",
    linestyles="dotted",
    alpha=alpha_2018,
)

# ax2.plot([4.5, 5.5], 13 * np.array([2, 4]), linewidth=l, color="k", alpha=alpha_2018)
ax2.plot([4.5, 5.5], [1430, 2600], linewidth=l, color="k", alpha=alpha_2018)

if 1==0:
    ax1.plot([1.5, 2.5], FC_2022_4mm, linewidth=l, color="k", alpha=1)
    ax1.plot([1.5, 2.5], FC_2022_2mm, linewidth=l, color="k", alpha=1)
    ax1.plot([2.5, 3.5], spectro_2022, linewidth=l, color="k")  # , alpha=0.5)
    inst_2022 = np.sqrt(guider_2018 ** 2 + detector_2018 ** 2 + LO_2018 ** 2 + FC_2022_2mm ** 2 + spectro_2022 ** 2)
    ax1.plot([-0.5, 4.5],inst_2022,linewidth=l,color="g",label=r"Predicted resolution from sub-systems",ls=":")
    ax1.hlines(8.5,-0.5,4.5,linewidth=l,color="orange",label=r"Measured end-to-end resolution",linestyles=u"dotted",)
    # ax2.plot([4.5, 5.5], 13 * np.array([3.5, 7]), linewidth=l, color="k")
    ax2.plot([4.5, 5.5], [1150, 1250], linewidth=l, color="k")
else:
    ax1.plot([1.5, 2.5], FC_2023, linewidth=l, color="k", alpha=1)
    ax1.plot([2.5, 3.5], spectro_2023, linewidth=l, color="k")  # , alpha=0.5)
    inst_2023 = np.sqrt(guider_2018 ** 2 + detector_2018 ** 2 + LO_2018 ** 2 + FC_2023 ** 2 + spectro_2023 ** 2)
    ax1.plot([-0.5, 4.5],inst_2023,linewidth=l,color="g",label=r"Predicted resolution from sub-systems",ls=":")
    # ax2.plot([4.5, 5.5], 13 * np.array([3.5, 7]), linewidth=l, color="k")
    ax2.plot([4.5, 5.5], [1150,1600], linewidth=l, color="k")
    
ax1.quiver(1, 2, 0, 3)
ax1.text(1, 3.5, "?", fontsize=15)
# plt.hlines(FC_2022_2mm_center, 1.5, 2.5, linewidth=1, color="k", alpha=1)

# plt.hlines(FC_2022_4mm_avg, 1.5, 2.5, linewidth=l, color="k", alpha=1)
# plt.hlines(FC_2022_2mm_avg, 1.5, 2.5, linewidth=l, color="k", alpha=1)
# plt.hlines(spectro_2022, 2.5, 3.5, linewidth=l, color="k", alpha=1)

# rectangle=plt.Rectangle((2.5, 4.4), 1, 7.7- 4.4, fc="k", ec="k")
# plt.gca().add_patch(rectangle)





# ax1.hlines(
#     np.sqrt(guider ** 2 + detector ** 2 + LO ** 2 + 5.82 ** 2 + 5 ** 2),
#     -0.5,
#     4.5,
#     linewidth=1,
#     color="g",
#     label=r"Predicted resolution from sub-systems (center)",
#     linestyles=u"dotted",
# )






ax1.hlines(
    flight_r_2018,
    -0.5,
    4.5,
    linewidth=l,
    color="r",
    label=r"In-flight best resolution (2018)",
    alpha=alpha_2018,
)
ax1.axvline(4.5, c="k", lw=l + 1)

ax1.set_ylabel(r"arc-seconds FWHM", fontsize=ft)
ax1.set_title(r"FIREBall-2 resolution", fontsize=ft + 2)
ax1.set_xticks(
    np.arange(6),
    (
        "guider",
        "Large\noptics",
        "Focal\nCorrector",
        "Spectro:\nspatial",
        "Detector",
        "Spectro:\nspectral",
    ),
    fontsize=ft,
)
ax2.set_ylabel("Y2-axis", fontsize=ft)
ax2.set_ylabel(r"$\lambda/d\lambda$ Resolution [compact vs diffuse]", fontsize=ft)
ax1.set_xlim((-1, 5.8))
ax1.set_ylim((-0.5, 8))
ax2.set_ylim((2700, 0))
ax1.legend(loc="lower left", framealpha=1, fontsize=8)  # loc=(0.6, 0.12)
# plt.savefig('/Users/Vincent/Nextcloud/Work/MyPapers/2019/ESA_FB_performance_Short/latex/resolution5.png')
ax1.grid()
plt.show()
# np.sqrt(0.29 ** 2 + 0 ** 2 + (1.7 / 2.35) ** 2 + (3.4 / 2.35) ** 2 + (2.4 / 2.35) ** 2)
# total_instru = np.sqrt(guider_2018 ** 2 + detector_2018 ** 2 + LO_2018 ** 2 + FC_2018 ** 2 + spectro_2018 ** 2)
