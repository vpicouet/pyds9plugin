n=5
std_ds9 = np.convolve(np.std(ds9,axis=1),np.ones(n)/n,mode="same")
ds9[ds9 > (np.median(ds9,axis=1)+0.1*std_ds9).reshape(-1,1)]=np.nan
