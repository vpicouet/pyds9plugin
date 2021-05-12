data = fitsfile[0].data
if data is None:
    data = np.nan * np.ones((10,10))
lx, ly = data.shape
Xinf, Xsup, Yinf, Ysup = 1, -1, 1, -1
column = np.nanmean(data[Yinf:Ysup, Xinf:Xsup], axis=1)
line = np.nanmean(data[Yinf:Ysup, Xinf:Xsup], axis=0)
table["Col2ColDiff"] = np.nanmedian(line[::2]) - np.nanmedian(line[1::2])
table["Line2lineDiff"] = np.nanmedian(column[::2]) - np.nanmedian(column[1::2])
table["TopImage"] = np.nanmean(column[:20])
table["BottomImage"] = np.nanmean(column[-20:])
table["SaturatedPixels"] = 100 * float(np.sum(data[Yinf:Ysup, Xinf:Xsup] > 2 ** 16 - 10)) / np.sum(data[Yinf:Ysup, Xinf:Xsup] > 0)
table["stdXY"] = np.nanstd(data[Yinf:Ysup, Xinf:Xsup])
try:
    table["stdX"] = np.nanstd(data[int(Yinf + (Ysup - Yinf) / 2), Xinf:Xsup])
    table["stdY"] = np.nanstd(data[Yinf:Ysup, int(Xinf + (Xsup - Xinf) / 2)])
except IndexError:
    table["stdX"] = np.nanstd(data[int(lx / 2), :])
    table["stdY"] = np.nanstd(data[:, int(ly / 2)])
