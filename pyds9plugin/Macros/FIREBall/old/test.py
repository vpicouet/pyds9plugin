import time
import datetime
start = time.time()
from astropy.table import Table
import numpy as np
from pyds9plugin.DS9Utils import *

d=DS9n()
ds9 = d.get_pyfits()[0].data
d.set_np2arr(ds9+1)

stop = time.time()
verboseprint(
"""
*******************************************************************************************************
           date : %s     Exited OK, duration = %s
******************************************************************************************************* """
% (datetime.datetime.now().strftime("%y/%m/%d %HH%Mm%S"), str(datetime.timedelta(seconds=stop - start))[:-3])
)

