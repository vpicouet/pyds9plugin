#%%
from pyds9plugin.DS9Utils import *
import sys, glob

path = get(d, "Path of the images you want to analyze:")
images = globglob(path)
data = fits.open(images[0])[0]
region = getregion(d, quick=True, message=False)    
if region is not None:
    Xinf, Xsup, Yinf, Ysup = lims_from_region(None, coords=region)
else:
    lx,ly = data.shape
    Yinf, Ysup, Xinf, Xsup = 0,lx,0,ly

print(Yinf, Ysup, Xinf, Xsup)
new_image = np.zeros(((Ysup-Yinf),(Xsup-Xinf)*len(images)))

for i,image in enumerate(images):
    fitsfile = fits.open(image)[0]
    data = fitsfile.data
    new_image[:,i*(Xsup-Xinf):(i+1)*(Xsup-Xinf)] = data[Yinf:Ysup, Xinf:Xsup]
fitsfile.data = new_image
name = os.path.dirname(path)+"/test.fits"
fitsfile.writeto(name,overwrite=True)
d.set("frame new")
d.set("file %s"%(name))