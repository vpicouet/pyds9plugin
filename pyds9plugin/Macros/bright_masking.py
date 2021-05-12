from scipy.ndimage import grey_dilation, binary_erosion

T = np.percentile(ds9,90)
EROSION, DILATATION = 10, 10

ds9[ds9 < T] = 0
ds9[ds9 > T] = 1
ds9 = binary_erosion(ds9, structure=np.ones((EROSION,EROSION))).astype(int)
ds9 = grey_dilation(ds9, size=(DILATATION, DILATATION))
