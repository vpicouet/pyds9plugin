# table['median'] = np.nanmedian(fitsfile[0].data) if  fitsfile[0].data is not None else np.nan
from astropy.table import Column
import re
SATURATION = 2 ** 16 -1
import numpy as np

data = fitsfile[0].data
if data is None:
    data = np.nan * np.ones((2,2))
columns = np.nanmean(data, axis=1)
lines = np.nanmean(data, axis=0)
ly,lx = data.shape
table['median'] = np.nanmedian(data)
table["Lines_difference"] = np.nanmedian(lines[::2]) - np.nanmedian(lines[1::2])
table["Lines_difference"] = np.nanmedian(columns[::2]) - np.nanmedian(columns[1::2])
table["SaturatedPixels"] = 100 * np.mean(data > SATURATION)
table["imno"] = int(re.findall('[0-9]+', os.path.basename(filename))[0])

table['overscann'] = np.nanmedian(data[:,:int(lx/2)])
table['physical'] = np.nanmedian(data[:,int(lx/2):])
table['flux'] = table['physical'] - table['overscann']

table['flux_slit'] = np.mean(data[1475:1495,2045:2058] -  table['overscann'])
table['flux_smearing'] = np.mean(data[1475:1495,2058:2065] -  table['overscann'])


table['image'] = Column([data[915:987,1176:1267]], name="twoD_std")
# Column([ np.array(block_reduce(data[:-9,500:-516], block_size=(n,n), func=np.nanstd, cval=np.nanstd(data)),dtype=int)], name="twoD_std")
