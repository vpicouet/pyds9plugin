# convert to fits
from pyds9plugin.DS9Utils import fitswrite
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

from astropy.io import fits
# Open the JPEG image
img = Image.open(filename)
img_array = np.array(img)
if len(img_array.shape)==3:
    img_array = np.mean(img_array,axis=2)
img_array = convolve2d(img_array, np.ones((2,2))/4, mode='same')
fitswrite(img_array,filename.replace(".jpg",".fits"))