#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 20:30:33 2020

@author: Vincent
"""
import numpy as np
import os, glob, sys

DS9_BackUp_path = os.environ["HOME"] + "/DS9QuickLookPlugIn"
os.system("echo 0 > %s" % (DS9_BackUp_path + "/.verbose.txt"))
os.system("echo 0 > %s" % (DS9_BackUp_path + "/.message.txt"))

from pyds9plugin.DS9Utils import *
from shutil import copyfile, rmtree
from pyds9 import DS9
from pkg_resources import resource_filename
from astropy.io import fits
from IPython import get_ipython

ipython = get_ipython()

DS9_BackUp_path = os.environ["HOME"] + "/DS9QuickLookPlugIn"
test_folder = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/testing"  # resource_filename('pyds9plugin', 'testing')
im = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/images/stack.fits"  # resource_filename('pyds9plugin', 'images/stack.fits')
if os.path.exists(test_folder) is False:
    os.mkdir(test_folder)
files_folder = test_folder + "/files"
if os.path.exists(files_folder) is False:
    os.mkdir(files_folder)


def main():
    d = DS9n()
    name = d.get("xpa").split("\t")[-1]
    print("\n    Setup    \n")

    # d.set("frame delete all")
    # present_plugIn()
    # open_file(
    #     argv="-p /Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/stack1*.fits"
    # )
    open_file(argv="-p /Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/m33_hi.fits -t Multi-Frames-As-Cube")
    open_file(argv="-p /Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/m33_hi.fits -t Multi-Frames")
    open_file(argv="-p /Users/Vincent/Downloads/Safari/human_brain_from_itk_example.tif -t IMPORT-3D")
    open_file(argv="-p /Users/Vincent/Desktop/mail.jpg -t RGB")
    open_file(argv="-p /Users/Vincent/Desktop/test_cat.fits -t PRISM")

    # d.set("frame 5")
    # d.set('regions command "circle 35 35 20"')
    # d.set("regions select all")
    # throughfocus(argv="-x %s" % (name))
    # #
    # light_curve(argv="-x %s" % (name))
    # explore_throughfocus(argv="-x %s -p /Users/Vincent/Desktop/stack_ldac.fits" % (name))
    # d.set("frame delete all")
    # open_file(argv="-p /Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/m33_hi.fits")
    # d.set('regions command "circle 100 100 100"')
    # d.set("regions select all")
    # plot_3d(argv="-x %s" % (name))
    # open_file(argv="-p /Users/Vincent/Desktop/filters_guillaume.jpg -t RGB")
    # d.set('regions command "circle 35 35 20"')
    # d.set("regions select all")
    # plot_area_3d_color(d)
    # open_file(argv="-p /Users/Vincent/Desktop/stack.fits")
    # d.set('regions command "circle 35 35 20"')
    # d.set("regions select all")
    # ds9entry(message="test")
    # get_image()

    # fit_gaussian_2d(argv="-x %s" % (name))
    # verbose(argv="-x %s" % (name))
    # LoadDS9QuickLookPlugin()
    # ds9_psfex(
    #     argv="-x %s -p %s " % (name, "/Users/Vincent/Desktop/stack_ldac.fits")
    # )
    # a = fits.open('/Users/Vincent/Desktop/stack.fits')[0]
    # fitswrite(a.data, '/tmp/test.fits', verbose=True, header=None)
    # fitswrite(a, '/tmp/test.fits', verbose=True, header=a.header)
    # check_file()
    # globglob('/Users/Vincent/Nextcloud/LAM/Work/Keynotes/DS9Presentation/2_Visualization-TF-TS/TF/TF_WCS/stack8100093_pa-161_2018-06-11T0*.fits')
    # return_path('/Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/stack18447179.fits')
    
    # d.set('regions command "box 500 500 200 200"')
    # d.set('regions select all')
    # throw_apertures(argv="-x %s" % (name))
    # ds9_stiff(argv="-x %s -p1 %s" % (name, "/Users/Vincent/Desktop/stack.fits"))
    # cosmology_calculator(argv="-x %s" % (name))
    # PlotFit1D(np.arange(10), np.arange(10), deg=2)
    # PlotFit1D(np.arange(10), np.arange(10), deg='gaus')
    # PlotFit1D(np.arange(10), np.arange(10), deg=2)
    # original_settings(argv="-x %s" % (name))
    # d.set('regions command "circle 35 35 20"')
    # d.set("regions select all")

    # getregion(d,all=False)
    # getregion(d,all=False,selected=False)
    

    # open_file(argv="-p /Users/Vincent/Nextcloud/LAM/Work/Keynotes/DS9Presentation/2_Visualization-TF-TS/TF/TF_WCS/stack8100093_pa-161_2018-06-11T06-03-20_wcs.fits")
    # astrometry_net(argv="-x %s" % (name))
    # open_file(argv="-p /Users/Vincent/Desktop/stack.fits")
    # fit_ds9_plot(argv="-x %s -g 1 -m 1 -v 1 -b Exponential" % (name))
    # interactive_plotter(argv="-x %s " % (name), function=lambda x, A=1, s=2, B=3: A * x ** 3 + s * x + B)

    
    
    
    
    # maxi_mask(argv="-x %s -F 0-0-0-0-0-0-0-0-0-0-0-0-0-1" % (name))

    # extract_sources(argv="-x %s" % (name))
    # background_estimation(argv="-x %s" % (name))
    # stack_images(argv="-x %s -p /Users/Vincent/Documents/shared/pyds9plugin/pyds9plugin/Images/stack18448*.fits" % (name))
    # column_line_correlation(argv="-x %s " % (name))
    # d.set("regions delete all")
    # d.set('regions command "box 35 35 35 35"')
    # d.set("regions select all")
    # trim(argv="-x %s " % (name))
    # fit_ds9_plot(argv="-x %s -o User-defined-interactively -p /Users/Vincent/Github/pyds9plugin/pyds9plugin/testing/test.dat" % (name))
    # manual_fitting(argv="-x %s -p /Users/Vincent/Github/pyds9plugin/pyds9plugin/testing/test.dat" % (name))
    # import_table_as_region(argv="-x %s -p /Users/Vincent/Desktop/stack_modified_uint8_uint8_uint8_cat.fits -xy X_IMAGE,Y_IMAGE" % (name))
    # open_file(argv="-p /Users/Vincent/Nextcloud/LAM/Work/Keynotes/DS9Presentation/image-000025-000034-Zinc-with_dark-161-stack.fits")
    # d.set('regions delete all')
    # d.set('regions command "box 1820 982 15 30"')
    # d.set('regions select all')
    # center_region(argv="-x %s" % (name))
    # d.set('regions command "box 1820 982 15 30"')
    # d.set('regions select all')
    # fill_regions(argv="-x %s -v 5" % (name))
    
    # # fit_gaussian_2d
    # # astrometry_net
    # # create_wcs
    # # execute_command
    # # guidance
    # # center_flux_std
    # # apply_query
    # # compute_gain
    
    # get_depth_image(argv="-x %s" % (name))
    # compute_fluctuation(argv="-x %s" % (name))
    # get(d, "bla")
    # replace_string_in_file("/Users/Vincent/Documents/these.csv", "Match", "MATCH", path2=None)
    
    # lock(argv="-x %s  -f image -c  image -l 1 -l  1 -m 1" % (name))
    # setup(argv="-x %s" % (name))
    # d.set('regions command "circle 100 100 20"')
    # d.set("regions select all")
    # setup(argv="-x %s" % (name))
    # original_settings(argv="-x %s" % (name))
    # path = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/*.fits"
    # create_header_catalog(argv="-x %s -p %s" % (name, path))
    # print("\n    Header    \n")
    # print(1)
    # organize_files(
    #     argv="-x %s -s %s -a %s -p %s"
    #     % (
    #         name,
    #         "NAXIS>1",
    #         "Directory,BITPIX",
    #         "/Users/Vincent/DS9QuickLookPlugIn/HeaderDataBase",
    #     )
    # )
    # cosmology_calculator(argv="-x %s" % (name))
    # convertissor(argv="-x %s -v 1" % (name))
    # fitsconverter('/Users/Vincent/Downloads/Safari/human_brain_from_itk_example.tif')
    # # 
    # convert_image(argv="-x %s -t 8,uint8" % (name))
    # python_command(argv="-x %s -e ds9+=1" % (name))
    # python_command(argv="-x %s -e header['test']=1" % (name))
    # globglob('/Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/stack[18-20]446850.fits')
    # apply_query(cat=None, query='NUMBER>10', path='/Users/Vincent/Desktop/test_cat.fits', new_path=None, delete=True)
    # print(get_columns('/Users/Vincent/Desktop/test_cat.fits'))
    # number_ds9_frames()
    # save_region_as_catalog(argv="-x %s -p /tmp/test.csv" % (name))
    # divide_catalog('/Users/Vincent/Catalogs/SextractorBase/test/COSMOS_UDD_v2_Sept2020_only_mag_z_ml_test_outside.txt')
    # read_big_ascii_table('/Users/Vincent/Catalogs/SextractorBase/test/COSMOS_UDD_v2_Sept2020_only_mag_z_ml_test_outside.txt')
    # ds9_plot(d, path="%s/test.dat" % (test_folder))
    # d.set("regions delete all")
    # d.set('regions command "circle 477 472 20"')
    # d.set("regions select all")
    # radial_profile(argv="-x %s -d 5" % (name))
    # d.set("regions delete all")
    # d.set('regions command "circle 477 472 20"')
    # d.set("regions select all")
    # aperture_photometry(argv="-x %s" % (name))
    # d.set("regions delete all")
    # d.set('regions command "circle 477 472 20"')
    # d.set("regions select all")
    # radial_profile(argv="-x %s" % (name))
    # center_region(argv="-x %s" % (name))
    # run_sextractor(argv="-x %s" % (name))
    # ds9_swarp(argv="-x %s -p %s" % (name, "/Users/Vincent/Desktop/stack.fits"))
    # d.set("regions delete all")
    # d.set('regions command "circle 477 472 20"')
    # d.set("regions select all")
    # plot_3d(argv="-x %s" % (name))
    # open_file(argv="-p /Users/Vincent/Desktop/filters_guillaume.jpg")
    # create_image_from_catalog(argv="-x %s" % (name))
    # create_image_from_catalog(argv="-x %s -p /Users/Vincent/Desktop/test_cat.fits" % (name))
    # open_table("/Users/Vincent/Desktop/test_cat.fits")
    # test_suite(argv="-x %s -t 5-All-In-One" % (name))
    print("\n    TEST COMPLETED 100%     \n")
    os.system("echo 1 > %s" % (DS9_BackUp_path + ".verbose.txt"))
    # kill_long_process(function="*test.py*")
    return#    


# d.set('plot current add ')#line|bar|scatter
# d.set('plot load %s %s'%('/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat', 'xy'))
# d.set('plot  current graph 1')
# d.set('plot load %s %s'%('/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat', 'xy'))
# d.set("plot title {%s}"%(title))
# d.set("plot title x {%s}"%(xlabel))
# d.set("plot title y {%s}"%(ylabel))

if __name__ == "__main__":
    try:
        os.system("echo 0 > %s" % (DS9_BackUp_path + "/.verbose.txt"))
        os.system("echo 0 > %s" % (DS9_BackUp_path + "/.message.txt"))
        copyfile(im, test_folder + "/files/" + os.path.basename(im))
        a = main()
        [
            os.remove(file)
            for file in glob.glob(
                os.environ["HOME"]
                + "/Github/DS9functions/pyds9plugin/testing/files/*"
            )
        ]

    finally:
        os.system("echo 1 > %s" % (DS9_BackUp_path + "/.verbose.txt"))
        os.system("echo 1 > %s" % (DS9_BackUp_path + "/.message.txt"))


# os.system('DS9Utils %s '%(name))
# os.system('DS9Utils %s '%(name))
