ims = []
lim=100
n=200
# files=glob.glob("/Volumes/VINCENT/GOBC/img_save/bkgd_photutils_substracted/stack4*.fits")
# files=glob.glob("/Volumes/VINCENT/GOBC/img_save/bkgd_photutils_substracted/stack4*.fits")

# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43738157-43739957].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43739958-43769958].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43767407-43771232].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43778807-43781957].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43782007-43795682].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43803557-43807157].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43800682-43807282].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43807282-43850432].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43860557-43863407].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43867757-43870607].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[43876907-43879907].fits")
# files=globglob("/Volumes/VINCENT/GOBC/img_save/stack[44233583-44236957].fits")
path = get(d,"Path of the images you want to stack:")
if os.path.isdir(path):
    files=globglob(path+"/*.fits")
else:
    files=globglob(path)

# files=globglob("/Users/Vincent/Pictures/Images_without213/image000*.fits")
for file in files:
    fitsimage = fits.open(file)
    im = np.array(fitsimage[0].data[lim:-lim,lim:-lim],dtype=float)
    print("nanmax = ", np.nanmax(im))
    x_max, y_max = np.where(im==np.nanmax(im)) 
    x_max, y_max = x_max[0], y_max[0]
    xs,ys = np.indices(im.shape)
    im[(xs<x_max-n)|(xs>x_max+n) | (ys<y_max-n)|(ys>y_max+n)]=np.nan
    if 1==0:
        plt.figure()
        plt.title(os.path.basename(file))
        plt.imshow(im[(xs>x_max-n)&(xs<x_max+n) & (ys>y_max-n)&(ys<y_max+n)].reshape(2*n,2*n))
        plt.show()

    ims.append(im)

ims_stack = np.nanmean(ims,axis=0)
ims_stack[~np.isfinite(ims_stack)] = np.nanmin(ims_stack)
fitsimage[0].data = ims_stack
h = fitsimage[0].header
#     filename = "/Users/Vincent/Nextcloud/LAM/FIREBALL/2023/Autocoll/Off_focus_exploration/sum_A%0.2f_B%0.2f_C%0.2f.fits"%(h["LINAENC"],h["LINBENC"],h["LINCENC"])
try:
    filename = os.path.dirname(files[0]) + "/sum_%s.fits"%(h["ROTENC"])
except KeyError:
    try:
        filename = os.path.dirname(files[0]) + "/sum_%s_%s.fits"%(h["TEMPDATE"].replace("/",""),h["TEMPTIME"])
    except KeyError:
        filename = os.path.dirname(files[0]) + "/sum.fits"#%(h["TEMPDATE"].replace("/",""),h["TEMPTIME"])

print(filename)
fitsimage.writeto(filename,overwrite=True)



