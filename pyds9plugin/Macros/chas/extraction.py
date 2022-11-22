new_image=convolve(ds9,np.ones((11,11)))
ds9[new_image<20] = 0
filename = "/tmp/test.fits"
fitswrite(ds9,filename)
d.set("frame new; file " + filename)
d.set('analysis task "SExtractor "')

