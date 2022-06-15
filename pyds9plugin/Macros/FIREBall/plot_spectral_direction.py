#%%
from pyds9plugin.testing.startup_spyder import *

# loop on all the files in the directory /Volumes/GoogleDrive-105248178238021002216/.shortcut-targets-by-id/1ZgB7kY-wf7meXrq8v-1vIzor75aRdLDn/FIREBall-2/FB2_2022/Detector_Data/220612/test_controller
# files = glob.glob(
#     "/Volumes/GoogleDrive-105248178238021002216/.shortcut-targets-by-id/1ZgB7kY-wf7meXrq8v-1vIzor75aRdLDn/FIREBall-2/FB2_2022/Detector_Data/220612/test_controller/im*.fits"
# )
files = glob.glob(os.path.dirname(filename) + "/im*.fits")
conv_gain = 5
files.sort()
# labs = ["Zn-open", "D2-open", "D2-long", "Zn-open"]
i = 0
plt.figure(figsize=(10, 10))
for file in files:
    fitsimage = fits.open(file)
    image = fitsimage[0].data
    header = fitsimage[0].header
    temp = header["TEMPB"]
    n = header["IMNO"]
    line = np.mean(image, axis=0)[1200:2300]
    os = np.mean(image, axis=0)[50:800]
    exp = float(header["EXPTIME"])
    val = (line - np.nanmedian(os)) / exp
    if exp > 0:
        plt.plot(
            val,
            label="%s : F= %0.3f \t /%is =\t  %0.3f  "
            % (n, exp * np.mean(val), exp, np.mean(val)),  # + labs[i],
        )
        i += 1
# plt.xlim((1200, 2300))
# plt.ylim((0, 0.25))
plt.xlabel("Column")
plt.ylabel("Flux: e-/sec")
plt.legend()
plt.show()
# %%
