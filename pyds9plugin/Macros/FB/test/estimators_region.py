from decimal import Decimal

# d = DS9n()
region = getregion(d, quick=True, message=False)
# except ValueError:
if region is None:
    image_area = [1500, 2000, 1500, 2000]
    Yinf, Ysup, Xinf, Xsup = image_area
else:
    Yinf, Ysup, Xinf, Xsup = lims_from_region(None, coords=region)
    # [131,1973,2212,2562]
    image_area = [Yinf, Ysup, Xinf, Xsup]
    verboseprint(Yinf, Ysup, Xinf, Xsup)

try:
    texp = float(header["EXPTIME"])
except KeyError as e:
    verboseprint(e)
    try:
        texp = float(header["EXPOSURE"])/1000
    except KeyError as e:
        verboseprint(e)
data = ds9
xc = [int(500), int((image_area[1] + image_area[0]) / 2)]
yc = int((image_area[2] + image_area[3]) / 2)  # 1000
w, l = (
    int(image_area[1] - image_area[0]),
    int(image_area[3] - image_area[2]),
)
n=2
# print(d.get('file'))
# print(filename)
# print(d.get('xpa'))
print(int(yc - l / 2) , int(yc + l / 2), int(xc[1] - w / 2) +n, int(xc[1] + w / 2)+n)
reg = data[int(yc - l / 2) : int(yc + l / 2),int(xc[1] - w / 2)+n : int(xc[1] + w / 2)+n]
# reg = data[int(xc[1] - w / 2) : int(xc[1] + w / 2),int(yc - l / 2) : int(yc + l / 2)]
reg_background = data[int(yc - l / 2) : int(yc + l / 2),int(xc[1] - w /2 +3*w) : int(xc[1] + w /2 +  3*w)]
reg_smearing1 = data[int(yc - l / 2) : int(yc + l / 2),int(xc[1] - w/2 + n - w) : int(xc[1] + w/2 + n - w)]
reg_smearing2 = data[int(yc - l / 2) : int(yc + l / 2),int(xc[1] - w/2 + n + w) : int(xc[1] + w/2 + n + w)]
if np.mean(reg_smearing1)>np.mean(reg_smearing2):
    reg_smearing=reg_smearing1
else:
    reg_smearing=reg_smearing2

# regOS = data[int(yc - l / 2) : int(yc + l / 2), 2200:2500]
regOS = data[int(yc - l / 2) : int(yc + l / 2), 50:900]
meanADU = np.nanmean(reg) - np.nanmean(regOS)
stdADU = np.nanstd(reg)
print('\nANALYSE\n')
# print('\nOS\n')
print("OS = %i, STD=%0.2f"% (np.nanmean(regOS), np.nanstd(regOS)))
# print('\n')
print("Dark = %i - %i = %0.1fADU/pix "% (np.nanmean(reg_background),np.nanmean(regOS),np.nanmean(reg_background) - np.nanmean(regOS)))
print("     = %is x %0.2fADU/s/pix, STD=%0.2f"% (texp, (np.nanmean(reg_background) - np.nanmean(regOS)) / texp, np.nanstd(reg_background)))
# print('\n\n')
print("Smea = %i - %i = %0.1fADU/pix "% (np.nanmean(reg_smearing),np.nanmean(reg_background),np.nanmean(reg_smearing) - np.nanmean(reg_background)))
print("     = %is x %0.2fADU/s/pix, STD=%0.2f"% (texp, (np.nanmean(reg_smearing) - np.nanmean(reg_background)) / texp, np.nanstd(reg_smearing)))

print("Reg = %i - %i = %0.1fADU/pix "% (np.nanmean(reg),np.nanmean(reg_background),np.nanmean(reg) - np.nanmean(reg_background)))
print("     = %is x %0.2fADU/s/pix, STD=%0.2f"% (texp, (np.nanmean(reg) - np.nanmean(reg_background)) / texp, np.nanstd(reg)))

#
# fig, (ax1,ax2,ax3) = plt.subplots(1,3)
# ax1.imshow(reg,vmin=np.nanmin(reg_background),vmax=np.nanmax(reg))
# ax2.imshow(reg_smearing,vmin=np.nanmin(reg_background),vmax=np.nanmax(reg))
# ax3.imshow(reg_background,vmin=np.nanmin(reg_background),vmax=np.nanmax(reg))
# plt.show()
