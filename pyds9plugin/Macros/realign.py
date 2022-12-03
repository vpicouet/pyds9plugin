


def FindPixelsPositionsOnVector(vector):
    """
    """
    xc, yc, length, angle = vector.data, vector.databox, vector.inside, vector.xc
    xs = xc + np.linspace(0, length, length) * np.cos(np.pi * angle / 180)
    ys = yc + np.linspace(0, length, length) * np.sin(np.pi * angle / 180)
    return xs, ys, angle


def RecrateArrayFromIndexList(xs, ys, angle, image):
    """
    """
    xs, ys = xs.astype(int), ys.astype(int)
    n1, n2 = xs.min() - (xs.max() - xs.min()) / 2, 2936 - xs.max() - (xs.max() - xs.min()) / 2  # int(1500 - (xs.max()-xs.min())/2)
    # a = array[ys[0]-20:ys[0]+20,xs[0]-n:xs[0]+n]
    n1, n2 = 700, 700
    a = image[ys[0], xs[0] - n1 : xs[0] + n2]

    for i in range(len(xs) - 1):
        a = np.vstack((a, image[ys[i + 1], xs[i + 1] - n1 : xs[i + 1] + n2]))
    # imshow(a)
    if angle > 270.0:
        a = a[::-1, :]
        verboseprint(angle)
        up = image[ys[0] :, xs[-1] - n1 : xs[-1] + n2][::-1, :]
        down = image[: ys[-1], xs[0] - n1 : xs[0] + n2][::-1, :]
        new_image = np.vstack((up, a, down))
    elif (angle > 180.0) & (angle < 270.0):
        verboseprint(angle)
        up = image[ys[0] :, xs[0] - n1 : xs[0] + n2][::-1, :]
        down = image[: ys[-1], xs[-1] - n1 : xs[-1] + n2][::-1, :]
        new_image = np.vstack((up, a, down))
    # imshow(new_image)
    return new_image


def DS9realignImage(xpapoint):
    """
    """
    d = DS9n(xpapoint)
    filename = d.get("file")
    try:
        vector = getregion(d)
    except ValueError:
        pass
    # fitsimage = fits.open(filename)[0]
    fitsimage = d.get_pyfits()[0]
    image = fitsimage.data
    xs, ys, angle = FindPixelsPositionsOnVector(vector)
    new_im = RecrateArrayFromIndexList(xs, ys, angle, image)
    fitsimage.data = new_im
    name = filename[:-5] + "_reAlighned.fits"
    fitswrite(fitsimage, name)
    d.set("frame new")
    d.set("file {}".format(name))  # a = OpenFile(xpaname,filename = filename)

    return

vector = getregion(d)
xs, ys, angle = FindPixelsPositionsOnVector(vector)
ds9 = RecrateArrayFromIndexList(xs, ys, angle, ds9)

