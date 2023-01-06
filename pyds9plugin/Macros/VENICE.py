from astropy.table import Table
from astropy.table import Column
import numpy as np
from tqdm import tqdm
###
from pyds9plugin.DS9Utils import DS9n
import os
###


def GenerateMask(path, path2=None, ra="RA", dec="DEC", flag="flag", mask="/Users/Vincent/Nextcloud/LAM/Work/CLAUDS/CLAUDS/ZphotCatalogs/DS9Regions/regVenice/VeniceAndMine2.reg"):
    import subprocess

    if path2 is None:
        path2 = "!" + path
        # path2=path[:-5]+'_masked.fits'
    process = subprocess.Popen('/Users/Vincent/Nextcloud/LAM/Work/LePhare/HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m  %s -f all -cat %s  -xcol %s -ycol %s -o "%s" -flagName %s' % (mask, path, ra, dec, path2, flag), shell=True, stdout=subprocess.PIPE)
    process.wait()
    # print('/Users/Vincent/Nextcloud/Work/LePhare/HSC-SSP_brightStarMask_Arcturus/venice-4.0.3/bin/venice -m  %s -f all -cat %s  -xcol %s -ycol %s -o "%s" -flagName %s'%(mask, path,ra,dec, path2,flag))
    return



d=DS9n()
ds9 = d.get_pyfits()[0].data
header = d.get_pyfits()[0].header
regs = d.get("regions").split("\n")
regs.remove("")
print(regs)

name = "/tmp/venice.reg"
if os.path.isfile(name):
    os.remove(name)
# Creating mask region (.reg)
with open(name, "a") as file:
    file.write(
        """# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
image
"""
    )

# Because VENICE can only handle regions with less than 100 edges
for i, reg in enumerate(tqdm(regs[3:])):
    if "polygon" in reg:
        region = np.array(reg[8:-1].split(",")).reshape(-1,2)
        if region.shape[0] > 99:
            region = region[:: int(region.shape[0] / 50)]
        if region.shape[0] > 99:
            region = region[:: int(region.shape[0] / 50)]
        new_line = ("polygon(" + ",".join([str(np.round(float(a), 5)) + "," + str(np.round(float(b), 5)) for a, b in zip( region[:, 0],region[:, 1])])+ ")\n")
    else:
        new_line = reg + " \n"
    with open(name, "a") as file:
        file.write(new_line)
d = DS9n()
d.set("regions delete all")
d.set("regions " + name)



d.set("regions system image")
a = d.get("regions selected")
# d.set("regions save /tmp/venice.reg")

x = np.linspace(0, header["NAXIS1"]-1, header["NAXIS1"])
y = np.linspace(0, header["NAXIS2"]-1, header["NAXIS2"])
xx, yy = np.meshgrid(x, y)
table = Table([Column(xx.flatten(), name="RA"), Column(yy.flatten(), name="DEC")])
table.write("/tmp/cat.fits", overwrite=True)
GenerateMask(path="/tmp/cat.fits", path2=None, ra="RA", dec="DEC", mask="/tmp/venice.reg", flag="flag")
table = Table.read("/tmp/cat.fits")
flag = np.array(table["flag"],dtype=float).reshape(header["NAXIS2"],header["NAXIS1"])
flag[flag==1] = ds9[flag==1]
# flag = np.array(table["flag"]).reshape(1000,1000)
flag[flag==0] = np.nan
d.set("frame new")
d.set_np2arr(flag)
