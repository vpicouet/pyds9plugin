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
from IPython import get_ipython

ipython = get_ipython()
# get_ipython().magic("%load_ext line_profiler")
# get_ipython().magic("%load_ext coverage")

from line_profiler import LineProfiler
import pstats
from io import StringIO


test_folder = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/testing"  # resource_filename('pyds9plugin', 'testing')
im = "/Users/Vincent/Github/pyds9plugin/pyds9plugin/images/stack.fits"  # resource_filename('pyds9plugin', 'images/stack.fits')
if os.path.exists(test_folder) is False:
    os.mkdir(test_folder)
files_folder = test_folder + "/files"
profiles_folder = test_folder + "/files"
if os.path.exists(files_folder) is False:
    os.mkdir(files_folder)


def Profile_old(
    file=None,
    command="open_file('%s','%s/stack.fits')",
    argv=None,
    profiles_folder=profiles_folder,
    functions="",
):
    if file is None:
        file = command.split("(")[0]
    if argv is not None:
        sys.argv = ["DS9Utils", command.split("(")[0]] + argv
    ipython.magic(
        "lprun -u 1e-1  -T %s/Prun_%s.py -s -r -f %s %s %s  "
        % (profiles_folder, file, command.split("(")[0], functions, command)
    )
    return


def Profile(
    command="lock(xpapoint=name, argv='-x %s  -f image -c  image -l 1 -l  1 -m 1')",
    profiles_folder=profiles_folder,
    functions=[lock,],
    file=None,
):
    result = StringIO()
    if file is None:
        file = command.split("(")[0]

    line_prof = LineProfiler(*functions)
    # line_prof.run(command).print_stats(result)
    line_prof.run(command)  # .print_stats(result)
    result = result.getvalue()
    print(result)
    # chop the string into a csv-like buffer
    # result='ncalls'+result.split('ncalls')[-1]
    # result='\n'.join([','.join(line.rstrip().split(None,6)) for line in result.split('\n')])
    # save it to disk
    f = open("%s/Prun_%s.py" % (profiles_folder, file), "w")
    f.write(result)
    f.close()
    return line_prof


def Profile_new(
    command="lock(xpapoint=name, argv='-x %s  -f image -c  image -l 1 -l  1 -m 1')",
    profiles_folder=profiles_folder,
    functions=[lock,],
    file=None,
    args=None,
):
    result = StringIO()
    if file is None:
        file = command.split("(")[0]

    line_prof = LineProfiler(*functions)
    # line_prof.run(command).print_stats(result)
    a = line_prof.runcall(command, *args)  #
    print("jkblon", a)
    line_prof.print_stats(result)
    result = result.getvalue()
    print(result)
    # chop the string into a csv-like buffer
    # result='ncalls'+result.split('ncalls')[-1]
    # result='\n'.join([','.join(line.rstrip().split(None,6)) for line in result.split('\n')])
    # save it to disk
    print("test")
    f = open("%s/Prun_%s.py" % (profiles_folder, file), "w")
    f.write(result)
    f.close()
    return line_prof


def Profile_new2(
    command="lock(xpapoint=name, argv='-x %s  -f image -c  image -l 1 -l  1 -m 1')",
    profiles_folder=profiles_folder,
    functions=[lock,],
    file=None,
    args=None,
):
    result = StringIO()
    if file is None:
        file = command.split("(")[0]

    line_prof = LineProfiler(*functions)
    # line_prof.run(command).print_stats(result)
    a = line_prof.runcall(command, *args)  #
    print("jkblon", a)
    line_prof.print_stats(result)
    result = result.getvalue()
    print(result)
    # chop the string into a csv-like buffer
    # result='ncalls'+result.split('ncalls')[-1]
    # result='\n'.join([','.join(line.rstrip().split(None,6)) for line in result.split('\n')])
    # save it to disk
    print("test")
    f = open("%s/Prun_%s.py" % (profiles_folder, file), "w")
    f.write(result)
    f.close()
    return line_prof


# Profile(file='prun_open.py',command="DS9open(None,'%s/stack.fits')"%(files_folder))


def main():
    d = DS9()
    xpapoint = d.get("xpa").split("\t")[-1]
    name = xpapoint
    print("\n    Setup    \n")
    # os.system('DS9Utils %s open  " %s/stack.fits" Slice  0 '%(name,files_folder))
    # Profile(command="DS9open(None,'%s/stack.fitsfiles_folder))
    # %lprun -f open_file(name,argv='  -p %s -t Slice  -c 0 '%(im))

    # Profile_old(command="open_file('%s')"%(name),argv=[im,'Slice','0'] )#,functions=[open_file,])
    # sys.exit()
    # Profile_old(command="open_file('%s','%s/stack.fits')" )#,functions=[open_file,])
    # %lprun  -f open_file open_file(name,argv='  -p %s -t Slice  -c 0 '%(im))

    # Profile(command="open_file('%s',argv='  -p %s -t Slice  -c 0 ')"%(name,im),functions=[open_file,])
    Profile_new(
        command=open_file,
        args=(None, None, "  -p %s -t Slice  -c 0 " % (im)),
        functions=[open_file,],
        file="test",
    )

    # sys.exit()
    #    ipython.magic("lprun -u 1e-1  -T /tmp/prun_open.py -s -r -f DS9open -f get DS9open('%s','%s/stack.fits')  "%(name,files_folder))
    # ipython.magic("lprun -u 1e-1  -T /tmp/prun_get.py -s -r -f d.get  d.get('file')" )

    Profile(
        command="lock('%s',argv='-f image -c  image -l 1 -l  1 -m 1')" % (name),
        functions=[lock,],
    )
    Profile(
        command="lock('%s',argv='-f wcs -c  wcs -l 1 -l  1 -m 1')" % (name),
        functions=[lock,],
    )
    Profile(
        command="lock('%s',argv='-f none -c  none -l 1 -l  0 -m 1')" % (name),
        functions=[lock,],
    )

    # sys.exit()
    Profile(command="setup(argv='-x %s')" % (name), functions=[setup,])
    # sys.exit()
    d = DS9n()
    d.set('regions command "circle 100 100 20"')
    d.set("regions select all")
    Profile(command="setup(argv='-x %s')" % (name), functions=[setup,])

    Profile(
        command="original_settings(argv='-x %s')" % (name),
        functions=[original_settings,],
    )
    d.set("regions delete all")
    d.set('regions command "circle 100 100 200"')
    d.set("regions select all")
    Profile(command="setup(argv='-x %s')" % (name), functions=[setup,])

    print("\n    Header    \n")

    #    Profile(command="DS9CreateHeaderCatalog('%s')"%(xpapoint),argv=['?',"%s"%(resource_filename('pyds9plugin', 'Images/*.fits')),'0','-'])
    os.system(
        'DS9Utils create_header_catalog -x %s  -p "%s"   '
        % (name, "/Users/Vincent/Github/pyds9plugin/pyds9plugin/Images/*.fits")
    )  # ,functions='-f CreateCatalog_new')
    # create_header_catalog(arv="-x %s  -p '%s'  -n 0  "%(name,))

    Profile(
        command="organize_files(argv='-x %s -p %s')"
        % (
            name,
            glob.glob("/Users/Vincent/DS9QuickLookPlugIn/HeaderDataBase/*.csv")[
                0
            ],
        ),
        functions=[organize_files,],
    )

    # os.system('DS9Utils %s AddHeaderField test 1 no_comment "%s/Desktop/*.fits"  '%( os.environ['HOME'], name))

    # print('\n    Regions    \n' )

    # d.set('regions command "circle 100 100 200"')
    # Profile(command="DS9Region2Catalog('%s')"%(xpapoint),argv=[files_folder+'/test.csv'])
    # d.set('regions delete all')
    # Profile(command="DS9Catalog2Region('%s')"%(xpapoint),argv=['0', files_folder+'/test.csv','x,y','-','circle','10','0','-'])
    # d.set('regions select all')
    # Profile(command="ReplaceWithNans('%s')"%(xpapoint),argv=['nan','0'],functions='-f InterpolateNaNs -f fitswrite')

    print("\n    Image Processing     \n")
    Profile(
        command="python_command(argv = '-x %s -e /Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FIREBall/EMCCD_fit_analytical.py')"
        % (name),
        functions=[python_command,parallelize],)
    from pyds9plugin.Macros.FIREBall.EMCCD_fit_analytical import emccd_model
    Profile_new(
        command=python_command,
        args=(None, "  -x None -e /Users/Vincent/Github/pyds9plugin/pyds9plugin/Macros/FIREBall/EMCCD_fit_analytical.py"),
        functions=[python_command,parallelize,emccd_model,],
        file="test",
    )

    Profile(
        command="python_command(argv = '-x %s -e ds9+=np.random.normal(0,0.5*np.nanstd(ds9),size=ds9.shape)')"
        % (name),
        functions=[python_command,],
    )
    ds9_plot(d, path="%s/test.dat" % (test_folder))

    # Profile(command="BackgroundFit1D('%s')"%(xpapoint),argv='x none none 1 1 1 0 none'.split(' '))
    d.set("regions delete all")
    d.set('regions command "circle 477 472 20"')
    d.set("regions select all")
    # Profile(command="fit_gaussian_2d(argv = '-x %s -p 1')"%(name),functions=[fit_gaussian_2d,])

    d.set("regions delete all")
    d.set('regions command "box 477 472 100 99"')
    d.set("regions select all")
    Profile(command="trim(argv = '-x %s')" % (name), functions=[trim,])
    d.set("frame delete")

    print("\n    Other     \n")

    print("\n    INSTRUMENTATION AIT     \n")
    d.set('regions command "circle 500 500 50"')
    d.set("regions select all")
    Profile(
        command="radial_profile(argv = '-x %s')" % (name),
        functions=[radial_profile,],
    )
    d.set("regions delete all")
    d.set('regions command "circle 500 500 50"')
    d.set("regions select all")

    Profile(
        command="center_region(argv = '-x %s')" % (name),
        functions=[center_region,],
    )

    Profile(
        command="open_file('%s',argv='  -p %s -t Slice  -c 0 ')" % (name, im),
        functions=[open_file,],
    )

    Profile(
        command="run_sextractor(argv='-x %s')" % (name),
        functions=[run_sextractor,],
    )

    d.set("regions delete all")
    d.set('regions command "circle 50 50 20"')
    d.set("regions select all")
    # Profile(command="plot_3d(argv='-x %s')"%(name),functions=[plot_3d,])

    print("\n    TEST COMPLETED 100%     \n")

    return


# d.set('plot current add ')#line|bar|scatter
# d.set('plot load %s %s'%('/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat', 'xy'))
# d.set('plot  current graph 1')
# d.set('plot load %s %s'%('/Users/Vincent/Github/DS9functions/pyds9plugin/testing/test.dat', 'xy'))
# d.set("plot title {%s}"%(title))
# d.set("plot title x {%s}"%(xlabel))
# d.set("plot title y {%s}"%(ylabel))

if __name__ == "__main__":
    try:
        # os.system('DS9Utils')
        # os.system('DS9Utils setup -h ')
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
        os.system("echo 0 > %s" % (DS9_BackUp_path + "/.verbose.txt"))
        os.system("echo 1 > %s" % (DS9_BackUp_path + "/.message.txt"))


# os.system('DS9Utils %s '%(name))
# os.system('DS9Utils %s '%(name))
