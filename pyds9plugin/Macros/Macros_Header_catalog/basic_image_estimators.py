

# if filename not in table["Path"]:
# from astropy.io import fits
ds9 = fitsfile[0].data
header = fitsfile[0].header
# table['median'] = np.nanmedian(fitsfile[0].data) if  fitsfile[0].data is not None else np.nan
print(ds9[:,1000:2000].shape)
table['median'] = np.nanmedian(ds9[:,1000:2000])
table['mean'] = np.nanmean(ds9[:,1000:2000])
# header['median'] = np.nanmedian(ds9)
# header['mean'] = np.nanmean(ds9)
# fits.setval(filename, 'median', value= np.nanmedian(ds9), comment="")
# fits.setval(filename, 'mean', value= np.nanmean(ds9), comment="")
# import re
# SATURATION = 2 ** 16 -1
# import numpy as np

# data = fitsfile[0].data
# if data is None:
#     data = np.nan * np.ones((2,2))
# columns = np.nanmean(data, axis=1)
# lines = np.nanmean(data, axis=0)
# ly,lx = data.shape
# table['median'] = np.nanmedian(data)
# table['mean'] = np.nanmean(data)
# table["Lines_difference"] = np.nanmedian(lines[::2]) - np.nanmedian(lines[1::2])
# table["Lines_difference"] = np.nanmedian(columns[::2]) - np.nanmedian(columns[1::2])
# table["SaturatedPixels"] = 100 * np.mean(data > SATURATION)
# table["imno"] = int(re.findall('[0-9]+', os.path.basename(filename))[0])