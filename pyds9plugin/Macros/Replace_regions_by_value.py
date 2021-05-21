
regions = getregion(d, selected=True)
if getregion(d, selected=True) is None:
    d.set("analysis message {It seems that you did not create or select the region. Please make sure to click on the region after creating it and re-run the analysis.}")
    sys.exit()
value = get(d, 'Value to replace in the regions (eg. inf, nan, 0.1, -2)', exit_=True)

# overwrite = bool(int(args.overwrite))
ds9 = ds9.astype(float).copy()
# verboseprint(regions)
try:
    xc, yc, h, w = int(regions.xc), int(regions.yc), int(regions.h), int(regions.w)
    verboseprint("Only one region found...")
    verboseprint("W = ", w)
    verboseprint("H = ", h)
    Xinf = int(np.floor(yc - h / 2 - 1))
    Xsup = int(np.ceil(yc + h / 2 - 1))
    Yinf = int(np.floor(xc - w / 2 - 1))
    Ysup = int(np.ceil(xc + w / 2 - 1))
    ds9[Xinf : Xsup + 1, Yinf : Ysup + 2] = value  # np.nan
except AttributeError:
    verboseprint("Several regions found...")
    for region in regions:
        x, y = np.indices(ds9.shape)
        try:
            xc, yc, h, w = int(region.xc), int(region.yc), int(region.h), int(region.w)
        except AttributeError:
            xc, yc, h, w = int(region.xc), int(region.yc), int(region.r), int(region.r)
            radius = np.sqrt(np.square(y - xc) + np.square(x - yc))
            mask = radius < h
        else:
            Xinf = int(np.floor(yc - h / 2 - 1))
            Xsup = int(np.ceil(yc + h / 2 - 1))
            Yinf = int(np.floor(xc - w / 2 - 1))
            Ysup = int(np.ceil(xc + w / 2 - 1))
            mask = (x > Xinf) & (x < Xsup + 1) & (y > Yinf) & (y < Ysup + 1)
        ds9[mask] = value  # np.nan
