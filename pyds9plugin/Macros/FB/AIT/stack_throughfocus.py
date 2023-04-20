

files = glob.glob("/Volumes/VINCENT/GOBC/today/stack????????.fits")
files.sort()
tf_length=11
chunks = [files[x:x+tf_length] for x in range(0, len(files), tf_length)][:-1]
n=30
new_image = np.zeros((n*2*len(chunks),n*2*tf_length+1))
for j, tfs in enumerate(chunks):
    print(i)
    data = fits.open(tfs[int(tf_length/2)])[0].data
    valmax = np.nanmax(data)
    print(np.where(data==valmax))
    yc, xc = np.where(data==valmax)
    yc, xc  = int(yc), int(xc)
    yc, xc  = int(yc)+50, int(xc)-100
    for i,f in enumerate(tfs):
        print(j)
        data = fits.open(f)[0].data
        sub = data[yc-n:yc+n,xc-n:xc+n]
        print(sub.shape)
        print(new_image[j*2*n:(j+1)*2*n,i*2*n:(i+1)*2*n].shape)
        new_image[j*2*n:(j+1)*2*n,i*2*n:(i+1)*2*n] = sub
        plt.imshow(new_image)      
        plt.show()
fitswrite(new_image,"/Volumes/VINCENT/GOBC/today/stacked_image_2.fits")
plt.imshow(new_image)      
    
d=DS9n()
d.set_np2arr(new_image)