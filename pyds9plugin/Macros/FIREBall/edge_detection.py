from scipy.ndimage import sobel
ds9=np.hypot(sobel(ds9,axis=0,mode='constant'),sobel(ds9,axis=1,mode='constant'))
