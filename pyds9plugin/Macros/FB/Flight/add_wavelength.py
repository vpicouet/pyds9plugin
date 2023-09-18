from astropy.table import Table
# d=DS9n()
names = d.get('plot').split()
name = names[0]
# for i, name in enumerate(names):
#     d.set("plot current %s " % (name))
i=0
d.set("plot save /tmp/test_%s.dat" % (i))

a = Table.read("/tmp/test_%s.dat" % (i),format="ascii")
# ds9entry(None,"Please give target name",quit_=False,)
# names = {"BQSO2":"0.77,2000","BQSO2":"0.77,2000","BQSO2":"0.77,2000","BQSO2":"0.77,2000","BQSO2":"0.77,2000","BQSO2":"0.77,2000"}
name = d.get("regions selected").split("text={")[-1]
if "BQSO3" in name:
    x,z = 1902, 0.623
elif "BQSO1" in name:
    x,z = 1653, 0.297
elif "TBS" in name:
    x,z = 1619, 0
elif "BQSO2" in name:
    x,z = 1330, 0.772
elif "QSO2" in name:
    x,z = 2000, 0


# a["col1"] = (213.9 - (1+z)*(l)) - 

a["col1"] = np.arange(2062.6604 - 0.2 * (x-1072),3000,0.2)[:len(a)]
a0 = a["col1"][0]
a.rename_column("col1",a["col1"][0])
a.rename_column("col2",a["col2"][0])
a.write("/tmp/spectra_observed_frame.dat",format="ascii",overwrite=True)#,names=[None,None] )
a["%s"%(a0)] /= (1+z)
a.rename_column("%s"%(a0),a["%s"%(a0)][0])
a.write("/tmp/spectra_rest_frame.dat",format="ascii",overwrite=True)#,names=[None,None] )



# d.set("plot load %s xy" % ("/tmp/test.dat"))
d.set("plot line %s xy" % ("/tmp/spectra_observed_frame.dat"))
# d.set("plot line color %s" % (c))
d.set("plot title '%s - observed frame'"%(name[:-1]))

d.set("plot line %s xy" % ("/tmp/spectra_rest_frame.dat"))
d.set("plot title '%s - rest frame'"%(name[:-1]))

# d.set("plot legend yes")

