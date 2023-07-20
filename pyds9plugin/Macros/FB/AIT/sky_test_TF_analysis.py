from astropy.table import vstack, Table
# path = "/Users/Vincent/Library/CloudStorage/GoogleDrive-vp2376@columbia.edu/.shortcut-targets-by-id/1ZgB7kY-wf7meXrq8v-1vIzor75aRdLDn/FIREBall-2/FB2_2023/GOBC_data/230623_sky_test/sky_test_organized_230623/ROTENC_-21/TF1/"

for path in glob.glob("/Users/Vincent/Library/CloudStorage/GoogleDrive-vp2376@columbia.edu/.shortcut-targets-by-id/1ZgB7kY-wf7meXrq8v-1vIzor75aRdLDn/FIREBall-2/FB2_2023/GOBC_data/230623_sky_test/sky_test_organized_230623/ROTENC_**/TF?/")[::-1]:
    print(path)
    
    min_arcsec = 5
    
    files = glob.glob(path + "stack????????_cat.fits")
    files.sort()
    cats = [Table.read(file) for file in files]
    lengths = [len(cat) for cat in cats]
    begin = 5 #= np.argmin(lengths)
    
    cat_center = Table.read(files[begin])
    cat_center = cat_center[(cat_center["X_IMAGE"]>100) & (cat_center["Y_IMAGE"]>100) & (cat_center["X_IMAGE"]<1180) & (cat_center["Y_IMAGE"]<980)]
    for i, line in enumerate(cat_center[:]):
        # print(line["ALPHA_J2000"])
        lengths = []
        # TFs = []
        lines = []
        for cat in cats:
            cat['x_real_center'] = line["X_IMAGE"]
            cat['y_real_center'] =  line["Y_IMAGE"]
            cat['MAG_APER_0'] = cat["MAG_APER"][:,0]
            cat['MAG_APER_1'] = cat["MAG_APER"][:,-1]
            cat['FLUX_APER_0'] = cat["FLUX_APER"][:,0]
            cat['FLUX_APER_1'] = cat["FLUX_APER"][:,-1]
            cat = cat[(cat["X_IMAGE"]>100) & (cat["Y_IMAGE"]>100) & (cat["X_IMAGE"]<1180) & (cat["Y_IMAGE"]<980)]
            cat["distance"] = 3600*np.sqrt((cat["ALPHA_J2000"]-line["ALPHA_J2000"])**2+(cat["DELTA_J2000"]-line["DELTA_J2000"])**2)
            # print(cat[cat["distance"]<min_arcsec])
            length = len(cat[cat["distance"]<min_arcsec])
            lengths.append(length)
            if length==1:
                lines.append(cat[cat["distance"]<min_arcsec][0])
            
            
            # if length==1:
            #     TFs.append(cat[cat["distance"]<min_arcsec]["VIGNET"][0])
        lengths = np.array(lengths)
        
        if np.sum(lengths[lengths==1])==11:
            tf = vstack(lines)
            filename = files[begin].replace(".fits","_%i_%i_cat.fits"%(line["X_IMAGE"],line["Y_IMAGE"]))
            tf.write(filename,overwrite=True)
    
            throughfocus_new(xpapoint=None, plot_=True, shift=10, argv=" -N _%i -n 11 -p %s  "%(i,filename))
            # throughfocus_new(xpapoint=None, plot_=True, shift=10, argv="--WCS 1 -th 1 -N _%i -n 11 -c %s,%s -p %s  "%(i,cat_center[i]["Y_IMAGE"],cat_center[i]["X_IMAGE"],path+"stack????????.fits"))
