"""
Correct CTI in an image from the Hubble Space Telescope (HST) Advanced Camera
for Surveys (ACS) instrument.

It takes a while to correct a full image, so a small patch far from the readout
register (where CTI has the most effect) is used for this example.
"""


# # header = fits.open('/Users/Vincent/Github/arcticpy/examples/acs/jc0a01h8q_raw.fits')[0].header
# n=10
# for a in [header.cards[n+i][0] for i in range(len(header.cards[n:]))]:
#     print(a)
#     try:
#         fits.setval('/Users/Vincent/Github/arcticpy/examples/acs/StackedImage_24-42-NoDark_modified.fits', a, value=header[a], comment="")
#     except ValueError:
#         pass
sys.path.append('/Users/Vincent/Github/arcticpy')
import arcticpy as ac
# from arcticpy.roe import ROE, ROETrapPumping
# from arcticpy import model_for_FIREBall
import os
from autoconf import conf
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from arcticpy.roe import ROE, ROETrapPumping
from arcticpy.ccd import CCD, CCDPhase
from arcticpy.trap_managers import AllTrapManager
from arcticpy.traps import TrapInstantCapture



# Path to this file
# path = os.path.dirname(os.path.realpath(__file__))

# Set up some configuration options for the automatic fits dataset loading
# conf.instance = conf.Config(config_path=f"{path}/config")

# Load the HST ACS dataset
# path += "/acs"
# name = "jc0a01h8q_raw"
# name="slit"
# # name='hot_pix'
# # name="slits2"
# # name='hot_pix2'
# name="slits_modified"

# frame = ac.acs.FrameACS.from_fits(file_path=f"{path}/{name}.fits", quadrant_letter="A")
# exposure = frame.exposure_info.modified_julian_date
# mask =
# row_start, row_end, column_start, column_end = -70, -40, -205, -190
ly,lx = ds9.shape
if ly>lx:
    frame=region[::-1,:].T
    parallel_offset=int(getregion(d)[0].xc+1000)
else:
    frame=region
    parallel_offset=int(getregion(d)[0].yc+1000)

    #fits.open(f"{path}/{name}.fits")[0].data
# exposure = 56571.277233796296
# Extract an example patch of a few rows and columns, offset far from readout
# row_start, column_start = 1890,   1045
# row_end = row_start+50
# column_end = column_start+50
# row_offset = len(frame) + row_start
# frame = frame#[row_start:row_end, column_start:column_end]
#



# frame.mask = frame.mask[row_start:row_end, column_start:column_end]
# fits.HDUList([fits.PrimaryHDU(frame)]).writeto('/Users/Vincent/Github/arcticpy/examples/acs/slit.fits', overwrite=True,)
# Plot the initial image
# sys.exit()
# Set CCD, ROE, and trap parameters for HST ACS at this Julian date
# traps, ccd, roe = ac.model_for_HST_ACS(date=2400000.5 + exposure)
# traps, ccd, roe = ac.model_for_HST_ACS(date=2400000.5 + exposure)
#best for slit: timescale=0.7 density=6 (12?)
#best for hot pix: timescale=0.1 density=40
density=40
timescale=0.5
axis=1
plt.figure()
plt.plot(frame.mean(axis=axis)[2:],":",label='non corrected')
for density in [10,20,30,40]:#,20,30,40]:
# for timescale in [0.1,0.3,0.6,0.8]:#7,14,20
    traps = [TrapInstantCapture(density=density, release_timescale=timescale)]
    # traps = [TrapInstantCapture(density=10, release_timescale=0.2),TrapInstantCapture(density=12, release_timescale=0.7)]
    ccd = CCD(full_well_depth=84700, well_fill_power=0.478, well_notch_depth=0)
    roe = ROE(dwell_times=[1])
    # plt.figure()
    # im = plt.imshow(X=frame[1:], aspect="equal")#, vmax=6600)#, vmin=2300, vmax=2800)
    # plt.colorbar(im)
    # # plt.axis("off")
    # plt.savefig(f"{path}/{name}_input.png", dpi=400)
    # plt.close()
    # print(f"Saved {path}/{name}_input.png")


    # Remove CTI
    image_cti_removed = ac.remove_cti(
        image=frame,
        iterations=1,
        parallel_traps=traps,
        parallel_ccd=ccd,
        parallel_roe=roe,
        # serial_roe=roe,
        # serial_ccd=ccd,
        # serial_traps=traps,

        parallel_offset=parallel_offset,
        parallel_express=2,
    )

    # Plot the corrected image
    # fits.HDUList([fits.PrimaryHDU(frame    )]).writeto('/tmp/test_smear.fits',overwrite=True)
    # fits.HDUList([fits.PrimaryHDU(image_cti_removed    )]).writeto('/tmp/test.fits',overwrite=True)


    plt.plot(image_cti_removed.mean(axis=axis)[2:],'--',label="density %0.1f scale %0.1f"%(traps[0].density,traps[0].release_timescale))
plt.legend()
plt.title("Which densisty and time scale unsmears the best the data?")
# plt.savefig(f"{path}/{name}_line_d%0.1f_t%0.1f.png"%(traps[0].density,traps[0].release_timescale), dpi=400)
plt.show()
# plt.close()



density, timescale = np.array(get(d,'What density and release timescale you want to use? eg: 30-0.5', exit_=True).split('-'),dtype=float)
traps = [TrapInstantCapture(density=density, release_timescale=timescale)]

ds9 = ac.remove_cti(
    image=ds9,
    iterations=1,
    parallel_traps=traps,
    parallel_ccd=ccd,
    parallel_roe=roe,
    parallel_offset=1000,
    parallel_express=2,
)
#
#
#
# fig, (ax0,ax1,ax2) = plt.subplots(1,3,figsize=(10,3))
# plt.figure()
# im = ax0.imshow(X=frame[2:], aspect="equal")#, vmax=6600)#, vmin=2300, vmax=2800)
#
# im = ax1.imshow(X=image_cti_removed[2:], aspect="equal")#, vmin=2300, vmax=2800)
# # plt.colorbar(im)
# ax0.set_title("Original image")
# ax1.set_title("Corrected image")
# ax2.set_title("Profile")
# ax2.plot(frame.mean(axis=axis)[2:],label='non corrected')
# ax2.plot(image_cti_removed.mean(axis=axis)[2:],'--',label='corrected')
# ax2.legend()
# # plt.axis("off")
# fig.tight_layout()
# fig.savefig(f"{path}/{name}_corrected_d%0.1f_t%0.1f.png"%(traps[0].density,traps[0].release_timescale), dpi=400)
# print(f"Saved {path}/{name}_corrected.png")
