#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:58:39 2023

@author: Vincent
"""
import sys
sys.path.append('/Users/Vincent/Github/arctic/python/')

import arcticpy 
import autoarray as aa


frame = fits.open("/Users/Vincent/Nextcloud/LAM/FIREBALL/TestsFTS2018-Flight/E2E-AIT-Flight/smearing_dark/-110_8600_1s_modified.fits")[0].data
from arcticpy.ccd import CCDPhase, CCD
from arcticpy.roe import ROE
from arcticpy.traps import (
    TrapInstantCapture,
    TrapSlowCapture,
    TrapInstantCaptureContinuum,
    TrapSlowCaptureContinuum,
)

ly,lx = frame.shape
frame=frame[:,::-1].T
# if ly>lx:
#     frame=frame[::-1,:].T
#     # parallel_offset=int(getregion(d)[0].xc+1000)
# else:
#     frame=frame
    # parallel_offset=int(getregion(d)[0].yc+1000)
# frame=ds9

#%%
density=40
timescale=0.5
axis=1
plt.figure()
plt.plot(frame[80:100,4],":",label='non corrected')
for density in [0.4]:#,20,30,40]:
    for timescale in [5]:#[0.1,0.3,0.6,0.8][-1:]:#7,14,20
        traps = [TrapInstantCapture(density=density, release_timescale=timescale)]
        # traps = [TrapInstantCapture(density=10, release_timescale=0.2),TrapInstantCapture(density=12, release_timescale=0.7)]
        ccd = CCD(full_well_depth=5000, well_fill_power=0.478, well_notch_depth=0)#FWD=84700
        roe = ROE(dwell_times=[1])
    # plt.figure()
    # im = plt.imshow(X=frame[1:], aspect="equal")#, vmax=6600)#, vmin=2300, vmax=2800)
    # plt.colorbar(im)
    # # plt.axis("off")
    # plt.savefig(f"{path}/{name}_input.png", dpi=400)
    # plt.close()
    # print(f"Saved {path}/{name}_input.png")

    
        # Remove CTI
        image_cti_removed = arcticpy.remove_cti(
            image=frame,
            n_iterations=5,
            parallel_traps=traps,
            parallel_ccd=ccd,
            parallel_roe=roe,
            # serial_roe=roe,
            # serial_ccd=ccd,
            # serial_traps=traps,
    
            # parallel_offset=parallel_offset,
            parallel_express=5,
            verbosity=1,
        )
    
        # Plot the corrected image
        # fits.HDUList([fits.PrimaryHDU(frame    )]).writeto('/tmp/test_smear.fits',overwrite=True)
        # fits.HDUList([fits.PrimaryHDU(image_cti_removed    )]).writeto('/tmp/test.fits',overwrite=True)
    
    
        plt.plot(image_cti_removed[80:100,4],'--',label="density %0.1f scale %0.1f"%(traps[0].density,traps[0].release_timescale))
        # plt.plot(image_cti_removed.mean(axis=axis)[2:],'--',label="density %0.1f scale %0.1f"%(traps[0].density,traps[0].release_timescale))
plt.legend()
plt.title("Which densisty and time scale unsmears the best the data?")
# plt.savefig(f"{path}/{name}_line_d%0.1f_t%0.1f.png"%(traps[0].density,traps[0].release_timescale), dpi=400)
plt.show()

#%%
d=DS9n()
d.set_np2arr(image_cti_removed.T[:,::-1])

# plt.close()

#%%

density, timescale = np.array(get(d,'What density and release timescale you want to use? eg: 30-0.5', exit_=True).split('-'),dtype=float)
traps = [TrapInstantCapture(density=density, release_timescale=timescale)]

ds9 = ac.remove_cti(
    image=ds9,
    iterations=1,
    parallel_traps=traps,
    parallel_ccd=ccd,
    parallel_roe=roe,
    parallel_offset=1000,
    # parallel_express=2,
    parallel_express=5,
    verbosity=1,

)



fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(10,3))
plt.figure()
im = ax0.imshow(X=frame[2:], aspect="equal")#, vmax=6600)#, vmin=2300, vmax=2800)

im = ax1.imshow(X=image_cti_removed[2:], aspect="equal")#, vmin=2300, vmax=2800)
# plt.colorbar(im)
ax0.set_title("Original image")
ax1.set_title("Corrected image")
ax2.set_title("Profile")
ax2.plot(frame.mean(axis=axis)[2:],label='non corrected')
ax2.plot(image_cti_removed.mean(axis=axis)[2:],'--',label='corrected')
ax2.legend()
# plt.axis("off")
fig.tight_layout()
fig.savefig(f"{path}/{name}_corrected_d%0.1f_t%0.1f.png"%(traps[0].density,traps[0].release_timescale), dpi=400)
print(f"Saved {path}/{name}_corrected.png")









#%%



image_path = "image_path/image_name"

# Load each quadrant of the image  (see pypi.org/project/autoarray)
image_A, image_B, image_C, image_D = [
    aa.acs.ImageACS.from_fits(
        file_path=image_path + ".fits",
        quadrant_letter=quadrant,
        bias_subtract_via_bias_file=True,
        bias_subtract_via_prescan=True,
    ).native
    for quadrant in ["A", "B", "C", "D"]
]

# Automatic CTI model  (see CTI_model_for_HST_ACS() in arcticpy/src/cti.py)
date = 2400000.5 + image_A.header.modified_julian_date
roe, ccd, traps = cti.CTI_model_for_HST_ACS(date)

# Or manual CTI model  (see class docstrings in src/<traps,roe,ccd>.cpp)
traps = [
    arctic.TrapInstantCapture(density=0.6, release_timescale=0.74),
    arctic.TrapInstantCapture(density=1.6, release_timescale=7.70),
    arctic.TrapInstantCapture(density=1.4, release_timescale=37.0),
]
roe = arctic.ROE()
ccd = arctic.CCD(full_well_depth=84700, well_fill_power=0.478)

# Remove CTI  (see remove_cti() in src/cti.cpp)
image_out_A, image_out_B, image_out_C, image_out_D = [
    arctic.remove_cti(
           image=image,
           n_iterations=5,
           parallel_roe=roe,
           parallel_ccd=ccd,
           parallel_traps=traps,
           parallel_express=5,
           verbosity=1,
    )
    for image in [image_A, image_B, image_C, image_D]
]

# Save the corrected image
aa.acs.output_quadrants_to_fits(
    file_path=image_path + "_out.fits",
    quadrant_a=image_out_A,
    quadrant_b=image_out_B,
    quadrant_c=image_out_C,
    quadrant_d=image_out_D,
    header_a=image_A.header,
    header_b=image_B.header,
    header_c=image_C.header,
    header_d=image_D.header,
    overwrite=True,
)