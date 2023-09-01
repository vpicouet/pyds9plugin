
header['NEWVALUE'] = 'test'

cxis = np.array([header["CX%i"%(i)] for i in range(8)])
cyis = np.array([header["CY%i"%(i)] for i in range(8)])
txis = np.array([header["TX%i"%(i)] for i in range(8)])
tyis = np.array([header["TY%i"%(i)] for i in range(8)])

ids_tx = np.array(["%0.1f - %0.1f"%(header["TX%i"%(i)], header["TY%i"%(i)]) for i in range(8)])
ids_cx = np.array(["%0.1f - %0.1f"%(header["CX%i"%(i)], header["CY%i"%(i)]) for i in range(8)])

tmp_region = filename.replace(".fits","_CXY.reg")
tmp_region_ = filename.replace(".fits","_CXYp.reg")
tmp_region2 = filename.replace(".fits","_TXY.reg")

create_ds9_regions(
    [cxis],
    [cyis],
    radius=[10],
    save=True,
    savename=tmp_region,
    form=["circle"],
    color=["yellow"],
    # ID=None,
    ID=[ids_cx],
)

create_ds9_regions(
    [cxis],
    [cyis],
    radius=[1],
    save=True,
    savename=tmp_region_,
    form=["circle"],
    color=["yellow"],
    ID=None,
    # ID=[ids_cx],
)
d.set("region {}".format(tmp_region))
d.set("region {}".format(tmp_region_))



create_ds9_regions(
    [txis],
    [tyis],
    radius=[10],
    save=True,
    savename=tmp_region2,
    form=["box"],
    color=["yellow"],
    ID=[ids_tx],
)
d.set("region {}".format(tmp_region))