# table['median'] = np.nanmedian(fitsfile[0].data) if  fitsfile[0].data is not None else np.nan
SATURATION = 2 ** 16 -1
import numpy as np

data = fitsfile[0].data
if data is None:
    data = np.nan * np.ones((2,2))
columns = np.nanmean(data, axis=1)
lines = np.nanmean(data, axis=0)
table['median'] = np.nanmedian(data)
table["Lines_difference"] = np.nanmedian(lines[::2]) - np.nanmedian(lines[1::2])
table["Lines_difference"] = np.nanmedian(columns[::2]) - np.nanmedian(columns[1::2])
table["SaturatedPixels"] = 100 * np.mean(data > SATURATION)
