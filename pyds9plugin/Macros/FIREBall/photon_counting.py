

def DS9photo_counting(xpapoint, save=False, config=my_conf):
    """Calculate threshold of the image and apply phot counting
    """
    from astropy.io import fits
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    d = DS9n(xpapoint)
    filename = d.get("file")
    try:
        region = getregion(d)
    except ValueError:
        Xinf, Xsup, Yinf, Ysup = my_conf.physical_region  # [0,2069,1172,2145]
    else:
        Xinf, Xsup, Yinf, Ysup = lims_from_region(region)
    image_area = [Yinf, Ysup, Xinf, Xsup]
    verboseprint(Yinf, Ysup, Xinf, Xsup)
    try:
        threshold = float(sys.argv[3])
    except IndexError:
        threshold = 5.5
    verboseprint("Threshold = %0.2f" % (threshold))
    # if len(sys.argv) > 4: path = Charge_path_new(filename, entry_point=4)
    path = globglob(sys.argv[-1])  # Charge_path_new(filename) if len(sys.argv) > 4 else [filename] #and verboseprint('Multi image analysis argument not understood, taking only loaded image:%s, sys.argv= %s'%(filename, sys.argv[-5:]))
    if len(path) == 1:
        plot_flag = True
    else:
        plot_flag = False
    verboseprint("plot_flag = ", plot_flag)

    for filename in path:
        verboseprint(filename)
        try:
            fitsimage = fits.open(filename)
        except IOError as e:
            verboseprint(bcolors.BLACK_RED + "FILE NOT FOUND: " + bcolors.END, e)
        else:
            image = fitsimage[0].data
            # emgain,bias,sigma,amp,slope,intercept = calc_emgain(image,area=image_area,plot_flag=True)
            verboseprint(type(filename))
            #            a, b = calc_emgainGillian(filename,area=image_area,plot_flag=plot_flag)
            #            verboseprint(len(a),len(b))
            #            (emgain,bias,sigma,frac_lost) = a

            D = calc_emgainGillian(filename, area=image_area, plot_flag=plot_flag)
            emgain, bias, sigma, frac_lost = [D[x] for x in ["emgain", "bias", "sigma", "frac_lost"]]  # D[my_conf.gain[0],'bias','sigma','frac_lost']
            b = [D[x] for x in ["image", "emgain", "bias", "sigma", "bin_center", "n", "xlinefit", "ylinefit", "xgaussfit", "ygaussfit", "n_bias", "n_log", "threshold0", "threshold55", "exposure", "gain", "temp"]]

            # image,emgain,bias,sigma,bin_center,n,xlinefit,ylinefit,xgaussfit,ygaussfit,n_bias,n_log,threshold0,threshold55,exposure, gain, temp = b
            try:
                plot_hist2(*b, ax=None, plot_flag=plot_flag)
            except TypeError:
                pass
            else:
                if plot_flag:
                    plt.show()
                if save:
                    new_image = apply_pc(image=image, bias=bias, sigma=sigma, area=0, threshold=threshold)
                    print(new_image.shape)
                    fitsimage[0].data = new_image
                    #                    if not os.path.exists(os.path.dirname(filename) +'/Thresholded_images'):
                    #                        os.makedirs(os.path.dirname(filename) +'/Thresholded_images')
                    name = os.path.dirname(filename) + "/Thresholded_images/" + os.path.basename(filename)[:-5] + "_THRES.fits"
                    # name = '/tmp/test_pc.fits'
                    fitswrite(fitsimage[0], name)
                    if len(path) == 0:
                        d.set("frame new")
                        d.set("file " + name)
    return
