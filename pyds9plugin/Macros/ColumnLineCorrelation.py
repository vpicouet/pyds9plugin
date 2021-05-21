DS9_BackUp_path = os.environ["HOME"] + "/DS9QuickLookPlugIn/"
region = getregion(d, quick=True,message=False,selected=True)
if region is not None:
    Xinf, Xsup, Yinf, Ysup = Lims_from_region(None, coords=region)
    area = [Yinf, Ysup, Xinf, Xsup]
else:
    area = [0, -1, 0, -1]

image = ds9[area[0] : area[1], area[2] : area[3]]
imagex = np.nanmean(image, axis=1)
imagey = np.nanmean(image, axis=0)
nbins = 300
bins1 = np.linspace(np.percentile(imagex[1:] - imagex[:-1], 5), np.percentile(imagex[1:] - imagex[:-1], 95), nbins)
bins2 = np.linspace(np.percentile(imagey[1:] - imagey[:-1], 5), np.percentile(imagey[1:] - imagey[:-1], 95), nbins)
x = (image[:, 1:] - image[:, :-1]).flatten()
y = (image[1:, :] - image[:-1, :]).flatten()
x = x[np.isfinite(x)]
y = y[np.isfinite(y)]
bins3 = np.linspace(np.percentile(x, 5), np.percentile(x, 95), nbins)
bins4 = np.linspace(np.percentile(y, 5), np.percentile(y, 95), nbins)
vals1, b_ = np.histogram(imagex[1:] - imagex[:-1], bins=bins1)
vals2, b_ = np.histogram(imagey[1:] - imagey[:-1], bins=bins2)
vals3, b_ = np.histogram(x, bins=bins3)
vals4, b_ = np.histogram(y, bins=bins4)

np.savetxt(DS9_BackUp_path + "/CSVs/1.dat", np.array([(bins1[1:] + bins1[:-1]) / 2, vals1]).T)
np.savetxt(DS9_BackUp_path + "/CSVs/2.dat", np.array([(bins2[1:] + bins2[:-1]) / 2, vals2]).T)
np.savetxt(DS9_BackUp_path + "/CSVs/3.dat", np.array([(bins3[1:] + bins3[:-1]) / 2, vals3]).T)
np.savetxt(DS9_BackUp_path + "/CSVs/4.dat", np.array([(bins4[1:] + bins4[:-1]) / 2, vals4]).T)

commands = []
commands.append("plot line open")
commands.append("plot axis x grid no ")
commands.append("plot axis y grid no ")
commands.append("plot title y 'Lines' ")
commands.append("plot load %s/CSVs/1.dat xy  " % (DS9_BackUp_path))
commands.append("plot add graph ")
commands.append("plot axis x grid no")
commands.append("plot axis y grid no ")
commands.append("plot load %s/CSVs/3.dat xy  " % (DS9_BackUp_path))
commands.append("plot add graph ")
commands.append("plot title y 'delta chisqr' ")
commands.append("plot load %s/CSVs/2.dat xy " % (DS9_BackUp_path))
commands.append("plot title y 'Columns' ")
commands.append("plot axis x grid no ")
commands.append("plot axis y grid no ")
commands.append("plot title x 'Column/Line average difference' ")
commands.append("plot add graph ")
commands.append("plot load %s/CSVs/4.dat xy " % (DS9_BackUp_path))
commands.append("plot title x 'Pixel value difference' ")
commands.append("plot axis x grid no ")
commands.append("plot axis y grid no ")
commands.append("plot layout grid")
d.set(" ; ".join(commands))
