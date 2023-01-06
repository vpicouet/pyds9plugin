#%%
from pyds9plugin.testing.startup_spyder import *

print(1)
d = DS9n()
# path = get(d, "Path of the files to analyzed")
path = "/Volumes/SSDPICOUET/LAM/FIREBALL/2022/DetectorData/220610/DIFFUSE_FOCUS_W_TEMP/*[05].fits"

files = glob.glob(path)
regs = getregion(d, selected=True)
files.sort()
n = 15


fig, axs = plt.subplots(len(regs), 1, figsize=(10, 7), sharex=True, sharey=True)
axs = axs.flatten()
for j, file in enumerate(files):
    im = fits.open(file)[0]
    temp = im.header["TEMPB"]
    for i, reg in enumerate(regs):
        x_inf, x_sup, y_inf, y_sup = lims_from_region(region=regs[i], coords=None)
        subim = im.data[y_inf - n : y_sup + n, x_inf - n : x_sup + n][::-1, :]
        stack = np.nanmean(subim, axis=0)
        stack = (stack - stack.min()) / (stack - stack.min()).ptp()
        axs[i].plot(
            stack, label="T=%0.1f" % (float(temp)), alpha=(len(files) - j) / len(files)
        )


axs[0].legend()
axs[0].set_xlim((20, 45))
plt.show()

