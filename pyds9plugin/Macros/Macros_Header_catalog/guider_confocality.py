import re
data = fitsfile[0].data
valmax = np.nanmax(data)
# print(np.where(data==valmax))
yc, xc = np.where(data==valmax)
n=300
table["xc"] = int(xc/n)*n
table["yc"] = int(yc/n)*n

