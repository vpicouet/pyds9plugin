
def create_contour_regions(xpapoint=None, argv=[]):
    """ Creates regions based on contour
    """
    from scipy.ndimage import grey_dilation, binary_erosion
    import numpy as np

    B, T, erosion, dilatation = np.array(sys.argv[-4:], dtype=float)
    d = DS9n(xpapoint)
    path = getfilename(d)
    d.set("""wcs skyformat degrees ;
             regions system wcs ;
             regions sky fk5 ;
             regions skyformat degrees""")
    d.set("scale limits %s %s" % (T, T + 1))
    d.set("region select all")
    d.set("regions color white")
    a = yesno(d, "Do you wish to continue?")
    if a != "n":
        im = d.get_pyfits()
        ds9 = im[0].data
        ds9[ds9 < T] = 0
        ds9[ds9 > T] = 1
        if erosion > 0:
            ds9 = binary_erosion(
                ds9, iterations=1, structure=np.ones((int(1), int(erosion)))
            ).astype(int)
            ds9 = binary_erosion(
                ds9, iterations=1, structure=np.ones((int(erosion), int(1)))
            ).astype(int)
        if dilatation > 0:
            ds9 = grey_dilation(ds9, size=(dilatation, dilatation))
        im[0].data = ds9
        d.set("frame new")
        d.set_pyfits(im)
        # d.set("block to 1")
        d.set("contour levels 0.1")
        d.set("contour smooth 5")
        d.set("contour yes")
        d.set("contour convert")
        d.set("regions system wcs")
        d.set("regions sky fk5")
        path = "/tmp/regions.reg"
        d.set("regions save " + path)
        d.set("regions delete all")
        path = simplify_mask(path)
        d.set("regions  " + path)


def simplify_mask(path):
    """Simplify masks so that they can be used by VENICE
       who can only use <100 edges regions """
    import numpy as np

    new_path = path[:-4] + "_smaller.reg"
    reg = open(path, "r")
    new_reg = open(new_path, "w")
    a = 0
    for i, line in enumerate(reg):
        if "polygon" not in line:
            new_reg.write(line)

        else:
            points = line.split("polygon(")[1].split(")\n")[0].split(",")
            number = len(points)
            if number < 200:
                new_reg.write(line)
            else:
                x, y = points[::2], points[1::2]
                k = 100  # 10
                a += 1
                while len(x) >= 100:
                    del x[::k]
                    del y[::k]
                    k -= 1
                new_line = (
                    "polygon("
                    + ",".join(
                        [
                            str(np.round(float(xi), 5))
                            + ","
                            + str(np.round(float(yi), 5))
                            for xi, yi in zip(x, y)
                        ]
                    )
                    + ")\n"
                )
                new_reg.write(new_line)
    print("%i regions rescaled!" % (a))
    return new_path
