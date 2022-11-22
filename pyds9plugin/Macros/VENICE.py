from astropy.table import Table
from astropy.table import Column


def GenerateMask(path, path2=None, ra="RA", dec="DEC", flag="flag", mask="/Users/Vincent/Nextcloud/LAM/Work/CLAUDS/CLAUDS/ZphotCatalogs/DS9Regions/regVenice/VeniceAndMine2.reg"):
    import subprocess

    if path2 is None:
        path2 = "!" + path
        # path2=path[:-5]+'_masked.fits'
    process = subprocess.Popen('/Users/Vincent/Nextcloud/LAM/Work/LePhare/HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m  %s -f all -cat %s  -xcol %s -ycol %s -o "%s" -flagName %s' % (mask, path, ra, dec, path2, flag), shell=True, stdout=subprocess.PIPE)
    process.wait()
    # print('/Users/Vincent/Nextcloud/Work/LePhare/HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m  %s -f all -cat %s  -xcol %s -ycol %s -o "%s" -flagName %s'%(mask, path,ra,dec, path2,flag))
    return


d.set("regions system image")
a = d.get("regions selected")
d.set("regions save /tmp/venice.reg")

x = np.linspace(0, header["NAXIS1"]-1, header["NAXIS1"])
y = np.linspace(0, header["NAXIS2"]-1, header["NAXIS2"])
xx, yy = np.meshgrid(x, y)
table = Table([Column(xx.flatten(), name="RA"), Column(yy.flatten(), name="DEC")])
table.write("/tmp/cat.fits", overwrite=True)
GenerateMask(path="/tmp/cat.fits", path2=None, ra="RA", dec="DEC", mask="/tmp/venice.reg", flag="flag")
table = Table.read("/tmp/cat.fits")
flag = np.array(table["flag"]).reshape(1000, 1000)
d.set("frame new")
d.set_np2arr(flag)
