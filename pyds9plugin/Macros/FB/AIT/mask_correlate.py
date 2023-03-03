hot_columns = fits.open(filename.replace('.fits', '.masks.fits'))[0].data
mask = hot_columns > 0.1
# mask2 =   np.hstack([mask[:,-1:], mask[:,:-1]])
# mask3 =   np.hstack([mask[:,-2:], mask[:,:-2]])
# mask4 =   np.hstack([mask[:,-3:], mask[:,:-3]])

mask2 =   np.vstack([mask[-1:,:], mask[:-1,:]])
mask3 =   np.vstack([mask[-2:,:], mask[:-2,:]])
mask4 =   np.vstack([mask[-3:,:], mask[:-3,:]])

mask5 =   np.vstack([mask[1:,:],mask[:1,:]])
mask6 =   np.vstack([mask[2:,:],mask[:2,:]])
mask7 =   np.vstack([mask[3:,:],mask[:3,:]])


total_mask = mask | mask2 | mask3| mask4 | mask5 | mask6| mask7
ds9[total_mask] = np.nan
ds9 = interpolate_replace_nans(ds9, Gaussian2DKernel(x_stddev=1, y_stddev=1))





# def reduce_dark(name):
#     command = (
#                 """/Users/Vincent/opt/anaconda3/envs/py38/bin/python %s  -v --single_mask %s --batch_size %i --proba_thresh %s --prior_modif  True  %s"""
#                 % (
#                     "/Users/Vincent/Github/pyds9plugin/pyds9plugin/MaxiMask-1.1/maximask.py",
#                     False,
#                     8,
#                     False,
#                     name,
#                 )
#             )

#     print(command)
#     import subprocess

#     try:
#         a = subprocess.call(command, shell=True, stderr=subprocess.STDOUT)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError(
#             "command '{}' return with error (code {}): {}".format(
#                 e.cmd, e.returncode, e.output
#             )
#         )
#     fits_image = fits.open(name)[0]
#     image = np.array(fits_image.data,dtype=float)
#     mask = fits.open(name.replace(".fits", ".masks.fits"))[0].data
#     image[mask>0.001]=np.nan 
#     image=interpolate_replace_nans(image,Gaussian2DKernel(x_stddev=1,y_stddev=1))
#     corr_name = name.replace(".fits","_corr.fits")
#     fits.HDUList([fits.PrimaryHDU(image, header=fits_image.header)]).writeto(corr_name, overwrite=True)

#     return corr_name