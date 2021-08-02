#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 10:35:13 2018

@author: V. Picouet

Copyright Vincent Picouet (01/01/2019)

vincent@picouet.fr

This software is a computer program whose purpose is to perform quicklook
image processing and analysis. It can ionteract with SAOImage DS9 Software
when loaded as an extension.

This software is governed by the CeCILL-B license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-B
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-B license and that you accept its terms.
"""
import resource
import time
import glob
import os
import sys

import datetime
import argparse

try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()
from functools import wraps

DS9_BackUp_path = os.path.join(os.environ["HOME"], "DS9QuickLookPlugIn")
tmp_path = os.path.join(DS9_BackUp_path, "tmp")
tmp_image = os.path.join(tmp_path, "image.fits")
tmp_region = os.path.join(tmp_path, "regions.reg")


def resource_filename(a="", b=""):
    return os.path.join(os.path.dirname(__file__), b)


def get_name_doc():
    import inspect

    outerframe = inspect.currentframe().f_back
    name = outerframe.f_code.co_name
    try:
        doc = outerframe.f_back.f_globals[name].__doc__
    except KeyError:
        doc = ""
    return name, doc


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.stdout.write("error: %s\n" % message)
        sys.exit(2)

    def parse_args_modif(self, argv, required=True):
        if len(argv) == 0:
            args = self.parse_args()
        else:
            verboseprint(["test"] + argv.split())
            args = self.parse_args(["test"] + argv.split())
        if hasattr(args, "path") is False:
            args.path = None
        if required & (args.xpapoint is None) & (args.path in (None, "")):
            self.error("at least one of --xpapoint and --path required")
        return args


def create_parser(namedoc, path=False):
    n, doc = namedoc
    formatter = lambda prog: argparse.HelpFormatter(
        prog, max_help_position=32, width=136
    )
    parser = MyParser(
        description=bcolors.BOLD + "%s: %s" % (n + bcolors.END, doc),
        usage=bcolors.FAIL
        + "DS9Utils %s [-h] [-x xpapoint] [--optional OPTIONAL]"
        % (bcolors.BOLD + n + bcolors.END + bcolors.FAIL)
        + bcolors.END,
        formatter_class=formatter,
    )
    parser.add_argument("function", help="Function to perform [here %s]" % (n))
    parser.add_argument(
        "-x",
        "--xpapoint",
        help="""XPA access point for DS9 communication.
                If none is provided, it will take the last DS9 window if one,
                else it will run the function without DS9.""",
        metavar="",
    )
    if path:
        parser.add_argument(
            "-p",
            "--path",
            help="Path of the image(s) to process, regexp accepted",
            metavar="",
            default="",
        )
    return parser


def read_v(path):
    """Read a table and try ascii or CSV if an error is raised"""
    from astropy.table import Table

    if os.path.isfile(path):
        try:
            cat = Table.read(path)
        except Exception:
            try:
                cat = Table.read(path, format="ascii")
            except Exception:
                cat = Table.read(path, format="csv")
    else:
        raise ValueError(path + " is not a file.")
    return cat
    # Table.read_v = staticmethod(read_v)


def verbose(xpapoint=None, verbose=None, argv=[]):
    """Change the configuration
    """

    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv)
    verbose_path = os.path.join(DS9_BackUp_path, ".verbose.txt")
    d = DS9n(args.xpapoint)
    if verbose is None:
        v = bool(int(os.popen("cat %s" % (verbose_path)).read()))
        if v:
            if yesno(d, "Are you sur you want to enter QUIET mode?"):
                os.system("echo 0 > %s" % (verbose_path))
        else:
            if yesno(d, "Are you sur you want to enter VERBOSE mode?"):
                os.system("echo 1 > %s" % (verbose_path))
    else:
        os.system("echo %s > %s" % (verbose, verbose_path))
    return


class FakeDS9(object):
    def __init__(self, **kwargs):
        """For sharing a porfoilio
        """
        self.total = []

    def get(self, value=""):
        return True

    def set(self, value=""):
        return True


def DS9n(xpapoint=None, stop=False):
    """Open a DS9 communication with DS9 software, if no session opens a new
    one else link to the last created session. Possibility to give the ssession
    you want to link"""
    from pyds9 import DS9, ds9_targets

    targets = ds9_targets()
    if targets:
        xpapoints = [target.split(" ")[-1] for target in targets]
    else:
        xpapoints = []
    if ((xpapoint == "None") | (xpapoint is None)) & (len(xpapoints) == 0):
        verboseprint("No DS9 target found")
        return FakeDS9()
    elif len(xpapoints) != 0:
        # verboseprint("%i targets found" % (len(xpapoints)))
        if xpapoint in xpapoints:
            pass
            # verboseprint("xpapoint %s in targets" % (xpapoint))
        else:
            if stop:
                sys.exit()
            else:
                # verboseprint("xpapoint %s NOT in targets" % (xpapoint))
                xpapoint = xpapoints[0]

    try:
        # verboseprint("DS9(%s)" % (xpapoint))
        d = DS9(xpapoint)
        return d
    except (FileNotFoundError, ValueError) as e:
        verboseprint(e)
        d = DS9()


def create_folders(DS9_BackUp_path=DS9_BackUp_path):
    """Create the folders in which are stored DS9 related data
    """
    if not os.path.exists(os.path.dirname(DS9_BackUp_path)):
        os.makedirs(os.path.dirname(DS9_BackUp_path))
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + "/Plots"):
        os.makedirs(os.path.dirname(DS9_BackUp_path) + "/Plots")
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + "/CSVs"):
        os.makedirs(os.path.dirname(DS9_BackUp_path) + "/CSVs")
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + "/HeaderDataBase"):
        os.makedirs(os.path.dirname(DS9_BackUp_path) + "/HeaderDataBase")
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + "/subsets"):
        os.makedirs(os.path.dirname(DS9_BackUp_path) + "/subsets")
    if not os.path.exists(os.path.dirname(DS9_BackUp_path) + "/tmp"):
        os.makedirs(os.path.dirname(DS9_BackUp_path) + "/tmp")
    if not os.path.exists(DS9_BackUp_path + "/.verbose.txt"):
        os.system("echo 0 > %s" % (DS9_BackUp_path + "/.verbose.txt"))
    if not os.path.exists(DS9_BackUp_path + "/.message.txt"):
        os.system("echo 1 > %s" % (DS9_BackUp_path + "/.message.txt"))
    return DS9_BackUp_path


if (len(sys.argv) == 1) | ("LoadDS9QuickLookPlugin" in sys.argv):
    create_folders()

# Do not chnage order
# if sys.stdin is not None:
#     verbose(xpapoint=None, verbose=0)
# else:
#     verbose(xpapoint=None, verbose=0)
message_ = bool(int(os.popen("cat %s/.message.txt" % (DS9_BackUp_path)).read()))
verbose_ = bool(int(os.popen("cat %s/.verbose.txt" % (DS9_BackUp_path)).read()))


def log(v=None):
    """Logger of all pyds9plugin activity on pyds9plugin_activity.log"""
    import logging
    from logging.handlers import RotatingFileHandler

    # création de l'objet logger qui va nous servir à écrire dans les logs
    logger = logging.getLogger("pyds9plugin")
    # on met le niveau du logger à DEBUG, comme ça il écrit tout
    if v is None:
        v = 1
    if v == 0:
        logger.setLevel(logging.ERROR)
    if v == 1:
        logger.setLevel(logging.DEBUG)
    log_path = DS9_BackUp_path + "pyds9plugin_activity.log"
    file_handler = RotatingFileHandler(log_path, "a", 1000000, 1)
    logger.addHandler(file_handler)
    logging.getLogger("matplotlib.font_manager").disabled = True
    return logger


logger = log()


def f_string(string):
    """Delete new lines and trailing spaces in string"""
    string = " ".join(string.rstrip().replace("\n", "").split())
    return string


def yesno(d, question="", verbose=message_):
    """Opens a native DS9 yes/no dialog box."""
    question = f_string(question)
    if verbose:
        verboseprint(question)
        if isinstance(d, FakeDS9):
            return input("%s [y/n]" % (question)) == "y"
        else:
            return bool(int(d.get("analysis message yesno {%s}" % (question))))
    else:
        return True


def message(d, question="", verbose=message_):  #
    """Opens a native DS9 message dialog box with a message."""
    question = f_string(question)
    if verbose:
        if isinstance(d, FakeDS9):
            print(question)
            return  # input("%s [y/n]" % (question)) == "y"
        else:
            return bool(
                int(
                    d.set(
                        "analysis message {%s}" % (question.replace("\n", " "))
                    )
                )
            )
    else:
        return True


def verboseprint(*args, logger=logger, verbose=verbose_):  # False
    """Prints a message only if verbose is set to True"""
    st = " ".join([str(arg) for arg in args])
    logger.critical(st)
    if bool(int(verbose)):
        print(*args)
        # if sys.stdin is None:
        from tqdm import tqdm

        with tqdm(
            total=1,
            bar_format="{postfix[0]} {postfix[1][value]:>s}",
            postfix=["", dict(value="")],
            file=sys.stdout,
        ) as t:
            for i in range(0):
                t.update()
    return


def get(d, sentence, exit_=True):
    """Opens a native DS9 entry dialog box asking you to answer something."""
    try:
        path = d.get("""analysis entry {%s}""" % (sentence))
    except (TypeError) as e:
        verboseprint(1, e)
        time.sleep(0.2)
        try:
            path = d.get("""analysis entry {%s}""" % (sentence))
        except (TypeError) as e:
            print(2, e)
            time.sleep(0.2)
            d = DS9n()
            path = d.get("""analysis entry {%s}""" % (sentence))
    if exit_ & (path == ""):
        sys.exit()
    else:
        return path


def lock(xpapoint=None, argv=[]):
    """Lock all the images in DS9 in frame/limits/colorbar [DS9 required]
    """
    import numpy as np

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-f",
        "--frame",
        default="image",
        choices=["image", "none", "wcs"],
        metavar="",
    )
    parser.add_argument(
        "-c",
        "--crosshair",
        default="image",
        choices=["image", "none", "wcs"],
        metavar="",
    )
    parser.add_argument("-l", "--scalelimits", default="1", metavar="")
    parser.add_argument("-s", "--smooth", default="0", type=str, metavar="")
    parser.add_argument("-m", "--cmap", default="1", type=str, metavar="")
    args = parser.parse_args_modif(argv, required=False)

    ds9 = DS9n(args.xpapoint)
    param_int = [args.scalelimits, args.smooth, args.cmap]
    param_string = np.array(param_int, dtype="U3")
    param_int = np.array(param_int, dtype=int)
    param_string[param_int == 1] = "yes"
    param_string[param_int == 0] = "no"
    d = []
    d.append("lock frame %s" % (args.frame))
    d.append("lock crosshair %s" % (args.crosshair))
    d.append("lock scalelimits  %s" % (param_string[-3]))
    d.append("lock smooth  %s" % (param_string[-2]))
    d.append("lock colorbar  %s" % (param_string[-1]))
    ds9.set(" ; ".join(d))
    return


def fn_timer(function):
    """Prints the time the function took to run"""

    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        verboseprint("[%s] Total time=%ss" % (function.__name__, str(t1 - t0)))
        return result

    return function_timer


def fn_memory_load(function):
    """Prints the memory the function used"""

    @wraps(function)
    def function_timer(*args, **kwargs):
        m1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        result = function(*args, **kwargs)
        m2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        mem1, mem2, mem3 = (m2 - m1) / 1e6, m2 / 1e6, m1 / 1e6
        verboseprint(
            "Memory used %s: %0.1f MB (%0.1f - %0.1f)"
            % (function.__name__, mem1, mem2, mem3)
        )
        return result

    return function_timer


def display_arguments(function):
    """Prints the arguments a function use"""

    @wraps(function)
    def display_and_call(*args, **kwargs):
        args_ = ", ".join([str(arg) for arg in args])
        opt_args_ = ", ".join(
            [kw + "=" + str(kwargs[kw]) for kw in kwargs.keys()]
        )
        verboseprint(function.__name__ + "(%s, %s)" % (args_, opt_args_))

    return display_and_call


@fn_timer
@fn_memory_load
@display_arguments
def compute_fluctuation(
    xpapoint=None,
    file_out_name=None,
    ext=1,
    ext_seg=1,
    sub=None,
    verbose=False,
    plot=False,
    seg=None,
    type="image",
    nomemmap=False,
    argv=[],
):
    """Compute image(s) depth by apertures method"""
    import numpy as np

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-a", "--aperture", default="5", help="Aperture radius in pixels"
    )
    parser.add_argument(
        "-n",
        "--number_apertures",
        default="1000",
        help="Number of apertures to throw in the image",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv)
    d = DS9n(args.xpapoint)
    file_in_name = get_filename(d)

    mag_zp = 30
    image, header, area, filename, offset = get_image(xpapoint)

    pix_scale = 1
    mag_zp = 30
    sigma = [5.0, 10.0]
    n_aper, aper_size = int(args.number_apertures), float(args.aperture)
    flux, n_aper_used, results = throw_apers(
        image,
        pix_scale,
        2 * float(aper_size),
        n_aper,
        seg=seg,
        type=type,
        sub_bkg=sub,
    )
    result = results[np.isfinite(results["aperture_sum"])]
    fresult = results[~np.isfinite(results["aperture_sum"])]

    # create_ds9_regions(
    #     [np.array(fresult["xcenter"]) + offset[0]],
    #     [np.array(fresult["ycenter"]) + offset[1]],
    #     radius=np.ones(len(fresult)) * float(aper_size),
    #     form=["circle"] * len(fresult),
    #     save=True,
    #     ID=[np.array(fresult["aperture_sum"], dtype=int)],
    #     color=["Yellow"] * len(fresult),
    #     savename=tmp_region,
    #     system="image",
    #     font=1,
    # )
    # d.set("regions %s" % (tmp_region))

    create_ds9_regions(
        [np.array(result["xcenter"]) + offset[0]],
        [np.array(result["ycenter"]) + offset[1]],
        radius=np.ones(len(result)) * float(aper_size),
        form=["circle"] * len(result),
        save=True,
        ID=[np.array(result["aperture_sum"], dtype=int)],
        color=["green"] * len(result),
        savename=tmp_region,
        system="image",
        font=1,
    )
    d.set("regions %s" % (tmp_region))

    d, d_err, flux_std = depth(flux, mag_zp, sigma, type)
    print_d = "{0:3.2f}".format(d[0])
    print_sigma = "{0:3.2f}".format(sigma[0])
    for i in range(1, len(sigma)):
        print_d += " {0:3.2f}".format(d[i])
        print_sigma += " {0:3.2f}".format(sigma[i])
    title = '{0}:\n Depth in {1:3.2f}" diam. apertures: {2:s} ({3:s} sigmas) +/- {4:3.2f}. flux_std = {5:3.2f}'.format(
        os.path.basename(file_in_name),
        aper_size,
        print_d,
        print_sigma,
        d_err[0],
        flux_std,
    )

    if plot:
        import matplotlib.pyplot as plt

        plot_histo(flux, flux_std, aper_size, title)
        plt.savefig(file_in_name[:-5] + "_depth.png")
        plt.show()
    #        plt.close()del
    return {
        "depth": d,
        "depth_eroor": d_err,
        "flux_std": flux_std,
        "n_aper_used": n_aper_used,
    }


def fits_ext(fitsimage):
    """Returns the first extension of a fits file containing an image"""
    import numpy as np

    ext = np.where(np.array([type(ex.data) == np.ndarray for ex in fitsimage]))
    verboseprint("Taking extension: %s" % (ext[0][0]))
    return ext[0][0]


# @fn_timer


def LoadDS9QuickLookPlugin(xpapoint=None):
    """Load the plugin in DS9 parameter file
    """
    from shutil import which

    d = DS9n()
    AnsDS9path = resource_filename("pyds9plugin", "QuickLookPlugIn.ds9.ans")
    AnsDS9path_old = resource_filename(
        "pyds9plugin", "QuickLookPlugIn_DS9<8.2.ds9.ans"
    )
    help_path = resource_filename("pyds9plugin", "doc/ref/index.html")
    new_file = os.path.join(os.path.dirname(AnsDS9path), "DS9Utils")
    symlink_force(which("DS9Utils"), new_file)
    print("DS9 analysis file = ", AnsDS9path)
    print("DS9 (version <8.2) analysis file = ", AnsDS9path_old)
    if len(glob.glob(os.path.join(os.environ["HOME"], ".ds9/ds9*.prf"))) > 0:
        for file in glob.glob(
            os.path.join(os.environ["HOME"], ".ds9", "ds9*.prf")
        ):
            if "QuickLookPlugIn" not in open(file).read():
                if "user4 {}" not in open(file).read():
                    if (
                        float(".".join(os.path.basename(file).split(".")[1:-1]))
                        > 8.1
                    ):
                        print(
                            bcolors.BLACK_RED
                            + file
                            + " : You already have an analysis file here. To use the Quick Look plug-in, add the following file in the DS9 Preferences->Analysis menu And switch on Autoload:  \n"
                            + AnsDS9path
                            + bcolors.END
                        )
                    else:
                        print(
                            bcolors.BLACK_RED
                            + file
                            + " : You already have an analysis file here. To use the Quick Look plug-in, add the following file in the DS9 Preferences->Analysis menu And switch on Autoload:  \n"
                            + AnsDS9path_old
                            + bcolors.END
                        )

                else:
                    var = input(
                        "Do you want to add the Quick Look plug-in to the DS9 %s files? [y]/n:"
                        % (os.path.basename(file))
                    )  # = "y"
                    if var.lower() != "n":
                        if (
                            float(
                                ".".join(os.path.basename(file).split(".")[1:-1])
                            )
                            > 8.1
                        ):
                            replace_string_in_file(
                                path=file,
                                string1="user4 {}",
                                string2="user4 {%s}" % (AnsDS9path),
                            )
                        else:
                            replace_string_in_file(
                                path=file,
                                string1="user4 {}",
                                string2="user4 {%s}" % (AnsDS9path_old),
                            )
                        print(
                            bcolors.BLACK_GREEN + "Plug-in added" + bcolors.END
                        )
                    else:
                        print(
                            bcolors.BLACK_RED
                            + "To use the Quick Look plug-in, add the following file in the DS9 Preferences->Analysis menu And switch on Autoload:  \n"
                            + AnsDS9path
                            + bcolors.END
                        )
                        sys.exit()
                        print(
                            bcolors.BLACK_RED
                            + "For DS9 versions < 8.2 please use:  \n"
                            + AnsDS9path_old
                            + bcolors.END
                        )
                        sys.exit()
            else:
                print(file + " : Analysis file already in preferences")
    else:
        message(
            d,
            """In order to add the plugin to DS9 go to
                      Preferences-Analysis, paste the path that is going to
                      appear. Click on auto-load and save the preferences.""",
        )
        d.set("analysis text {%s}" % (AnsDS9path))
        print(
            bcolors.BLACK_RED
            + "To use DS9Utils, add the following file in the DS9 Preferences->Analysis menu And switch on Autoload:  \n"
            + AnsDS9path
            + bcolors.END
        )
    DS9Utils_link = os.path.join(os.path.dirname(AnsDS9path), "DS9Utils")
    if os.path.isfile(DS9Utils_link) is False:
        symlink_force(
            which("DS9Utils"),
            os.path.join(os.path.dirname(AnsDS9path), "DS9Utils"),
        )
    html_file = (
        "file:/Users/Vincent/Github/pyds9plugin/pyds9plugin/doc/ref/index.html"
    )
    if html_file in open(AnsDS9path).read():
        replace_string_in_file(
            path=AnsDS9path, string1=html_file, string2="file:%s" % (help_path),
        )
    if html_file in open(AnsDS9path_old).read():
        replace_string_in_file(
            path=AnsDS9path_old,
            string1=html_file,
            string2="file:%s" % (help_path),
        )
        sys.exit()

    return


def replace_string_in_file(path, string1, string2, path2=None):
    """Replaces string in a txt file"""
    fin = open(path, "rt")
    data = fin.read()
    data = data.replace(string1, string2)
    fin.close()
    if path2 is not None:
        path = path2
        if os.path.exists(path):
            os.remove(path)
        fin = open(path, "x")
    else:
        fin = open(path, "wt")
    fin.write(data)
    fin.close()
    return


def present_plugIn():
    """Print presentation of the plug in.
    """
    from shutil import which

    if which("DS9Utils") is None:
        print(
            """"DS9Utils does not seem to be installed in your PATH.
                  Please add it and re-run the command."""
        )
    if os.path.isfile(os.path.dirname(__file__) + "/DS9Utils") is False:
        symlink_force(
            which("DS9Utils"), resource_filename("pyds9plugin", "DS9Utils")
        )

    print(
        bcolors.BLACK_GREEN
        + """
                     DS9 Quick Look Plug-in

            Written by Vincent PICOUET <vincent.picouet@lam.fr>
            License: CeCILL-B
            visit https://people.lam.fr/picouet.vincent/pyds9plugin
            for more information:

            For better experience use last version of DS9 (>8.1)

            To use it run:
            > DS9Utils LoadDS9QuickLookPlugin
            Then launch DS9 and play with the analysis commands!
            You can also access it via command line:
            > DS9Utils function [-h] [--optionals OPTIONALS]
            Find bellow the list of the available functions
                                                                                 """
        + bcolors.END
    )
    return


# @fn_timer
# @profile
def setup(xpapoint=None, color="cool", argv=[]):
    """Give a general visualisation of the image by applying thresholds
    [DS9 required]
    """
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-s",
        "--scale",
        default="Log",
        help="Scale to apply",
        type=str,
        choices=[
            "Log",
            "Linear",
            "Power",
            "Square-Root",
            "ASINH",
            "SINH",
            "Histogram-Equalization",
            "zscale",
            "minmax",
        ],
    )
    parser.add_argument(
        "-l",
        "--limits",
        default="50-99.9",
        help="Scale limit in percentile of data",
        metavar="",
    )
    parser.add_argument(
        "-c", "--color", default="cool", help="Colormap to use", metavar=""
    )

    args = parser.parse_args_modif(argv, required=False)

    d = DS9n(args.xpapoint)
    d.set("wcs degrees")
    cuts = np.array(args.limits.split("-"), dtype=float)
    region = getregion(d, all=False, quick=True, selected=True)
    from astropy.io import fits

    try:
        fitsimage = fits.open(get_filename(d))
    except FileNotFoundError:
        fitsimage = d.get_pyfits()
    fitsimage = fitsimage[fits_ext(fitsimage)].data
    lx, ly = fitsimage.shape[0], fitsimage.shape[1]
    if region is not None:
        image_area = lims_from_region(None, coords=region)
        x_inf, x_sup, y_inf, y_sup = image_area
        if (x_sup < 0) | (y_sup < 0):
            image_area = [
                int(lx / 2),
                int(lx / 2) + 50,
                int(ly / 2),
                int(ly / 2) + 50,
            ]
    else:
        image_area = [
            int(lx / 2),
            int(lx / 2) + 50,
            int(ly / 2),
            int(ly / 2) + 50,
        ]
    x_inf, x_sup, y_inf, y_sup = image_area
    image = fitsimage[y_inf:y_sup, x_inf:x_sup]
    try:
        image_ok = image[np.isfinite(image)]
        d.set(
            "cmap %s ; scale %s ; scale limits %0.3f %0.3f "
            % (
                args.color,
                args.scale,
                np.nanpercentile(image_ok, cuts[0]),
                np.nanpercentile(image_ok, cuts[1]),
            )
        )
    except ValueError:
        d.set("cmap %s ; scale %s " % (args.color, args.scale))

    return


def organize_files(xpapoint=None, dpath=DS9_BackUp_path + "subsets/", argv=[]):
    """From a fits file database, create a subset of images considering
    a selection and ordering rules
    """
    from astropy.table import Table, vstack
    from shutil import copyfile
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        help="Path of the header data base to filter/organize",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-s",
        "--selection",
        default="",
        help="Selection to be applied.  Use \| for OR and \& for AND eg: ((FLUX*EXPTIME>1) \& (FLUX*EXPTIME<30)) \| (EMGAIN>0)",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-a",
        "--arange",
        default="Directory",
        help="Coma separated fields, order matters for folder creation. eg: Directory,NAXIS2",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-n",
        "--number",
        default="all",
        help="Number of same files to take",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv)
    d = DS9n(args.xpapoint)
    cat_path, number, fields, query = (
        args.path,
        args.number,
        args.arange,
        args.selection,
    )
    if number == "all":
        number = 1000
    cat_path = cat_path.rstrip()[::-1].rstrip()[::-1]
    fields = np.array(fields.split(","), dtype=str)
    if os.path.isdir(cat_path):
        files = glob.glob(os.path.join(cat_path, "*.csv"))
        files.sort(key=lambda x: os.path.getmtime(x))
        file = files[-1]
        if yesno(
            d,
            """%s is a directory not a table. Do you wish to take the most
               recent csv table of this directory: %s?"""
            % (cat_path, os.path.basename(file)),
        ):
            cat_path = file
        else:
            sys.exit()
    try:
        cat = Table.read(cat_path)
    except Exception as e:
        cat = Table.read(cat_path, format="csv")
        print(e)
        logger.warning(e)
    cat = delete_multidim_columns(cat)
    if query != "":
        df = cat.to_pandas()
        verboseprint(query)
        try:
            new_table = df.query(query)
        except Exception:
            query = ds9entry(
                args.xpapoint,
                "UndefinedVariableError in query, please rewrite a query.",
                quit_=False,
            )
            new_table = df.query(query)
        t2 = Table.from_pandas(new_table)
    else:
        t2 = cat
    if len(t2) == 0:
        d = DS9n(args.xpapoint)
        message(d, "No header verifying your condition, please verify it.")
    verboseprint(t2)
    verboseprint("SELECTION %i -> %i" % (len(cat), len(t2)))
    verboseprint("SELECTED FIELD  %s" % (fields))
    path_date = dpath + datetime.datetime.now().strftime("%y%m%d_%HH%Mm%S")
    if not os.path.exists(path_date):
        os.makedirs(path_date)

    t3 = t2.copy()
    t3.remove_rows(np.arange(len(t2)))
    for field in fields:
        for value in np.unique(t2[field]):
            t3 = vstack((t3, t2[t2[field] == value][-int(number) :]))
    t2 = t3
    try:
        numbers = t2[list(fields)].as_array()
    except KeyError:
        numbers = [""] * len(t2)

    for line, numbers in zip(t2, numbers):
        filename = line["Path"]
        number = list(numbers)  # np.array(list(line[fields]))
        f = "/".join(["%s_%s" % (a, b) for a, b in zip(fields, number)])
        new_path = os.path.join(path_date, f)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        symlink_force(filename, new_path + "/%s" % (os.path.basename(filename)))

    copyfile(cat_path, os.path.join(path_date, os.path.basename(cat_path)))
    csvwrite(t2, os.path.join(path_date, "HeaderCatalogSubet.csv"))
    message(d, "Images are saved as symbolik links there : %s" % (path_date))
    return t2


def create_repositories(path, field, values):
    """Create repository have different names and values
    """
    paths = []
    for value in values:
        npath = os.path.join(path, "%s_%s" % (field, value))
        # os.makedirs(npath)
        verboseprint(npath)
        paths.append(npath)
    return paths


def PlotFit1D(
    x=None,
    y=[709, 1206, 1330],
    deg=1,
    plot_=True,
    sigma_clip=None,
    title=None,
    xlabel=None,
    ylabel=None,
    P0=None,
    bounds=(-1e10, 1e10),
    fmt=".",
    ax=None,
    # c="black",
    Type="normal",
    sigma=None,
    # ls=":",
    interactive=False,
    **kwargs
):
    """ PlotFit1D(np.arange(100),np.arange(100)**2
    + 1000*np.random.poisson(1,size=100),2)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from decimal import Decimal
    import numpy as np

    x, y = np.array(x), np.array(y)

    if x is None:
        x = np.arange(len(y))

    if sigma is not None:
        x = x[np.isfinite(y)]
        y = y[np.isfinite(y)]
        sigma = sigma[np.isfinite(y)]
    else:
        x = x[np.isfinite(y)]
        y = y[np.isfinite(y)]

    x = np.array(x)
    y = np.array(y)

    if sigma_clip is not None:
        index = (
            (x > np.nanmean(x) - sigma_clip[0] * np.nanstd(x))
            & (x < np.nanmean(x) + sigma_clip[0] * np.nanstd(x))
            & (y > np.nanmean(y) - sigma_clip[1] * np.nanstd(y))
            & (y < np.nanmean(y) + sigma_clip[1] * np.nanstd(y))
        )
        x, y = x[index], y[index]
        std = np.nanstd(y)
    else:

        sigma_clip = [10, 1]
        index = (
            (x > np.nanmean(x) - sigma_clip[0] * np.nanstd(x))
            & (x < np.nanmean(x) + sigma_clip[0] * np.nanstd(x))
            & (y > np.nanmean(y) - sigma_clip[1] * np.nanstd(y))
            & (y < np.nanmean(y) + sigma_clip[1] * np.nanstd(y))
        )
        std = np.nanstd(y[index])

    if plot_ & (ax is None):
        fig = plt.figure()  # figsize=(10,6))
        gs = gridspec.GridSpec(2, 1, height_ratios=(4, 1))
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if sigma is None:
            ax1.plot(x, y, fmt, label="Data", **kwargs)
        else:
            ax1.errorbar(x, y, fmt=fmt, yerr=sigma, **kwargs)

    xp = np.linspace(x.min(), x.max(), 1000)

    def linear_func(p, x):
        m, c = p
        return m * x + c

    if type(deg) == int:
        z, res, rank, singular, rcond = np.polyfit(x, y, deg, full=True)
        pcov = None
        popt = np.poly1d(z)
        law = np.poly1d(z)
        if (deg == 1) & (Type == "ortho"):
            import scipy

            linear_model = scipy.odr.Model(linear_func)
            data = scipy.odr.RealData(x, y)
            odr = scipy.odr.ODR(data, linear_model, beta0=[0.0, 1.0])
            out = odr.run()
            popt = np.poly1d(out.beta)
            law = np.poly1d(out.beta)
        zp = popt(xp)
        zz = popt(x)
        degs = [" %0.2f * x^%i" % (a, i) for i, a in enumerate(popt.coef[::-1])]
        # name = "Fit: " + "+".join(degs) + ", R=%0.2E" % (Decimal(res[0]))
    else:
        from scipy.optimize import curve_fit

        if deg == "exp":
            law = lambda x, b, a, offset: b * np.exp(-x / a) + offset
            if P0 is None:
                P0 = [np.nanmax(y) - np.nanmin(y), 1, np.nanmin(y)]
        if deg == "2exp":
            law = (
                lambda x, b1, b2, a1, a2, offset: b1 * np.exp(-x / a1)
                + b2 * np.exp(-x / a2)
                + offset
            )
        elif deg == "gaus":
            law = (
                lambda x, a, xo, sigma, offset: a ** 2
                * np.exp(-np.square((x - xo) / sigma) / 2)
                + offset
            )
            if P0 is None:
                P0 = [
                    np.nanmax(y) - np.nanmin(y),
                    x[np.argmax(y)],
                    np.std(y),
                    np.nanmin(y),
                ]
        elif deg == "power":
            law = lambda x, amp, index, offset: amp * (x ** index) + offset
            P0 = None
        elif callable(deg):
            law = deg

        if interactive:
            print("Interactive Fit")
            from IPython import get_ipython

            get_ipython().run_line_magic("matplotlib", "")
            if len(P0) == 1:
                interactiv_manual_fitting(
                    x,
                    y,
                    initial="%s(x,a*%f)" % (law.__name__, *P0,),
                    dict_={law.__name__: law},
                )
            if len(P0) == 2:
                print("Interactive Fit")
                interactiv_manual_fitting(
                    x,
                    y,
                    initial="%s(x,a*%f,b*%f)" % (law.__name__, *P0,),
                    dict_={law.__name__: law},
                )
            if len(P0) == 3:
                interactiv_manual_fitting(
                    x,
                    y,
                    initial="%s(x,a*%f,b*%f,c*%f)" % (law.__name__, *P0,),
                    dict_={law.__name__: law},
                )
            if len(P0) == 4:
                interactiv_manual_fitting(
                    x,
                    y,
                    initial="%s(x,a*%f,b*%f,c*%f,d*%f)" % (law.__name__, *P0,),
                    dict_={law.__name__: law},
                )
            plt.show()
            get_ipython().run_line_magic("matplotlib", "inline")
            return {
                "popt": np.zeros(len(P0)),
                "pcov": np.zeros((len(P0), len(P0))),
                "res": 0,
                "y": y,
                "x": x,
                "curve": [],
            }
        try:
            popt, pcov = curve_fit(law, x, y, p0=P0, bounds=bounds, sigma=sigma)
        except RuntimeError as e:
            logger.warning(e)
            print(law)
            print(type(law))
            print(e)
            if interactive:
                if input("Do you want to fit it manually? [y/n]") == "y":
                    from IPython import get_ipython

                    get_ipython().run_line_magic("matplotlib", "")
                    if len(P0) == 1:
                        interactiv_manual_fitting(
                            x,
                            y,
                            initial="%s(x,a*%f)" % (law.__name__, *P0,),
                            dict_={law.__name__: law},
                        )
                    if len(P0) == 2:
                        interactiv_manual_fitting(
                            x,
                            y,
                            initial="%s(x,a*%f,b*%f)" % (law.__name__, *P0,),
                            dict_={law.__name__: law},
                        )
                    if len(P0) == 3:
                        interactiv_manual_fitting(
                            x,
                            y,
                            initial="%s(x,a*%f,b*%f,c*%f)"
                            % (law.__name__, *P0,),
                            dict_={law.__name__: law},
                        )
                    if len(P0) == 4:
                        interactiv_manual_fitting(
                            x,
                            y,
                            initial="%s(x,a*%f,b*%f,c*%f,d*%f)"
                            % (law.__name__, *P0,),
                            dict_={law.__name__: law},
                        )
                    get_ipython().run_line_magic("matplotlib", "inline")
                    return {
                        "popt": np.zeros(len(P0)),
                        "pcov": np.zeros((len(P0), len(P0))),
                        "res": 0,
                        "y": y,
                        "x": x,
                        "curve": [],
                    }
            else:
                return {
                    "popt": np.zeros(len(P0)),
                    "pcov": np.zeros((len(P0), len(P0))),
                    "res": 0,
                    "y": y,
                    "x": x,
                    "curve": [],
                }
        res = -99
        res = np.sum(np.square(y - deg(x, *popt)))
        zp = law(xp, *popt)
        zz = law(x, *popt)
        name = "Fit %s, R=%0.2E" % (
            np.round(np.array(popt, dtype=int), 0),
            Decimal(res),
        )
    if plot_:
        if ax is None:
            if deg == "gaus":
                ax1.text(
                    popt[1],
                    popt[0] ** 2,
                    "Max = %0.1f std" % (popt[0] ** 2 / std),
                )
            if title:
                fig.suptitle(title, y=1)
            if xlabel:
                ax2.set_xlabel(xlabel)
            if ylabel:
                ax1.set_ylabel(ylabel)
            ax1.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )
            ax2.set_ylabel("Error")
            line = ax1.plot(xp, zp, **kwargs)  # ls, label=name, c=c)
            ax2.plot(x, y - zz, fmt, **kwargs)  # label=name, c=c)
            ax2.set_xlim(ax1.get_xlim())
            ax2.plot([-1e100, 1e100], [0, 0], **kwargs)  # ls, c=c)
            ax1.grid(linestyle="dotted")
            ax2.grid(linestyle="dotted")
            ax1.legend()
            plt.tight_layout()
        else:
            xp = np.linspace(
                np.nanmin(xp) - 2 * xp.ptp(),
                np.nanmax(xp) + 2 * xp.ptp(),
                5 * len(xp),
            )
            try:
                line = ax.plot(xp, np.poly1d(z)(xp), **kwargs)
            except UnboundLocalError:
                line = ax.plot(xp, law(xp, *popt), **kwargs)
            ax1, ax2 = ax, ax
        return {
            "popt": popt,
            "pcov": pcov,
            "res": res,
            "axes": [ax1, ax2],
            "y": y,
            "x": x,
            "curve": line,
            "sigma": sigma,
            "y_fit": zz,
            "function": law,
        }
    else:
        return {
            "popt": popt,
            "pcov": pcov,
            "res": res,
            "y": y,
            "x": x,
            "curve": [],
            "sigma": sigma,
            "y_fit": zz,
            "function": law,
        }
    return {
        "popt": popt,
        "pcov": pcov,
        "res": res,
        "y": y,
        "x": x,
        "curve": [],
        "sigma": sigma,
        "y_fit": zz,
        "function": law,
    }


def create_regions(regions, savename=tmp_region, texts="               "):
    """Create DS9 regions files from imported python region from DS9
    """
    regions_ = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    image
    """
    colors = ["orange", "green", "red", "pink", "grey", "black"] * 100
    for region, text, color in zip(regions, texts, colors):
        verboseprint(region)
        regions_ += "\n"
        regions_ += add_region(region, color=color, text=text)
    with open(savename, "w") as text_file:
        text_file.write(regions_)


def add_region(region, color="Yellow", text=""):
    """Add a region
    """

    def get_r(region):
        return region.r if hasattr(region, "r") else [region.w, region.h]

    def get_type(region):
        return "circle" if hasattr(region, "r") else "box"

    form = get_type(region)

    if form == "circle":
        text = "%s(%0.2f,%0.2f,%0.2f) # color=%s width=4 text={%s}" % (
            form,
            region.xc,
            region.yc,
            get_r(region),
            color,
            text,
        )
    if form == "box":
        text = "%s(%0.2f,%0.2f,%0.2f,%0.2f) # color=%s width=4 text={%s}" % (
            form,
            region.xc,
            region.yc,
            get_r(region)[0],
            get_r(region)[1],
            color,
            text,
        )
    # verboseprint(text)
    return text


def get_data_from_region(d, region, ext):
    """Get data from region
    """
    x_inf, x_sup, y_inf, y_sup = lims_from_region(region=region, coords=None)
    data = d.get_pyfits()[ext].data[y_inf:y_sup, x_inf:x_sup]
    return data


def aperture_photometry(xpapoint=None, argv=[]):
    """Computes photometry in given aperture(s) [DS9 required]
    """
    from astropy.table import Table
    import numpy as np
    from astropy.stats import sigma_clipped_stats
    from photutils import aperture_photometry
    from photutils import CircularAperture, CircularAnnulus

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-a",
        "--apertures",
        default="30,30",
        help="Aperture radius in pixel",
        type=str,
    )
    parser.add_argument(
        "-z",
        "--zero_point_magnitude",
        default="0",
        help="Zero point magnitude of the image",
        type=str,
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    fitsfile = d.get_pyfits()[0]
    data = fitsfile.data
    zero_point_magnitude = float(args.zero_point_magnitude)
    regions = getregion(d, all=False, quick=True, selected=True)
    d.set("regions delete select")
    Phot = aperture_photometry(
        data, CircularAperture((10, 10), r=5), error=0.1 * data
    )
    Phot["annulus_median"] = 0
    Phot["aper_pix"] = 0
    Phot["aper_bkg"] = 0
    Phot["aper_sum_bkgsub"] = 0
    Phot.remove_row(0)
    apers = np.array(args.apertures.split(","), dtype=float)
    if regions is None:
        message(d, "Please select a region before running this analysis.")
        sys.exit()
    for reg in regions:
        # id = 'M = '
        for aper in apers:
            positions = (reg[0], reg[1])
            apertures = CircularAperture(positions, r=aper)
            annulus_apertures = CircularAnnulus(
                positions, r_in=aper, r_out=1.2 * aper
            )
            annulus_masks = annulus_apertures.to_mask(method="center")
            bkg_median = []
            mask = annulus_masks
            annulus_data = mask.multiply(data)
            annulus_data_1d = annulus_data[mask.data > 0]
            _, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
            bkg_median.append(median_sigclip)
            bkg_median = np.array(bkg_median)
            phot = aperture_photometry(data, apertures, error=0.1 * data)
            phot["annulus_median"] = bkg_median
            phot["aper_pix"] = aper
            phot["aper_bkg"] = bkg_median * apertures.area
            phot["aper_sum_bkgsub"] = phot["aperture_sum"] - phot["aper_bkg"]
            Phot.add_row(phot[0])
    Phot["MAG_APER"] = np.around(
        -2.5 * np.log10(Phot["aperture_sum"] - Phot["aper_bkg"])
        + zero_point_magnitude,
        1,
    )
    for col in Phot.colnames:
        Phot[col].info.format = "%.8g"  # for consistent table output
    verboseprint(Phot)
    Phot = Table(Phot)
    for aper, color in zip(
        apers, 10 * ["green", "yellow", "white"][: len(apers)]
    ):
        t_sub = Phot[Phot["aper_pix"] == aper]
        create_ds9_regions(
            [t_sub["xcenter"]],
            [t_sub["ycenter"]],
            radius=[aper],
            color=[color] * len(t_sub),
            form=["circle"] * len(t_sub),
            save=True,
            savename=tmp_region,
            ID=[t_sub["MAG_APER"]],
        )
        d.set("regions %s" % (tmp_region))

    return phot


def create_ds9_regions(
    xim,
    yim,
    radius=20,
    more=None,
    save=True,
    savename=tmp_region,
    form=["circle"],
    DS9_offset=[1, 1],
    color=["green"],
    ID=None,
    system="image",
    font=10,
    lw=1,
):
    """Returns and possibly save DS9 regions around sources with a given radius
    """
    import numpy as np

    regions = """# Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=%s font="helvetica %s normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    %s
    """ % (
        lw,
        font,
        system,
    )
    # if system == "fk5":
    #     DS9_offset = [0, 0] # should be deleted or removed from arguments

    if (
        (type(radius) == int)
        or (type(radius) == float)
        or (type(radius) == np.int64)
        or (type(radius) == np.float64)
    ):
        r, r1 = (
            radius,
            radius,
        )  # np.ones(len(xim))*radius, np.ones(len(xim))*radius #radius, radius
    else:
        try:
            r, r1 = radius  # , radius
        except ValueError:
            r = radius
    for i in range(len(xim)):
        if form[i] == "box":
            try:
                rest = "{:.4f},{:.4f})".format(r, r1)  # r[i], r1[i]
            except UnboundLocalError:
                rest = "{:.4f},{:.4f})".format(r[i], r[i])  # r[i], r1[i]

            rest += " # color={}".format(color[i])
        elif form[i] == "circle":
            rest = "{:.4f})".format(r[i])  # [i]
            rest += " # color={}".format(color[i])
        try:
            for j, (x, y) in enumerate(np.nditer([xim[i], yim[i]])):

                if form[0] == "ellipse":
                    rest = "{:.6f},{:.6f},{:.6f})".format(
                        more[0][j], more[1][j], more[2][j]
                    )
                    rest += " # color={}".format(color[j])
                    # print(color[j])
                regions += (
                    "{}({:.6f},{:.6f},".format(form[i], x + 0, y + 0) + rest
                )
                if ID is not None:
                    regions += " text={{{}}}".format(ID[i][j])
                    # print(ID[i][j])
                regions += "\n"
        except ValueError as e:
            logger.warning(e)
            pass

    if save:
        with open(savename, "w") as text_file:
            text_file.write(regions)
        verboseprint(("Region file saved at: " + savename))
        return


def getdata(xpapoint=None, plot_=False, selected=False):
    """Get data from DS9 display in the definied region
    """
    import numpy as np

    d = DS9n(xpapoint)

    regions = getregion(d, quick=True, selected=selected, dtype=float)
    # BUG problem here when test!
    if type(regions) != list:
        regions = [regions]
    datas = []
    verboseprint(regions)
    if regions[0] is None:
        datas = [d.get_pyfits()[0].data]
    else:
        for region in regions:
            verboseprint(region)
            verboseprint("region = %s" % (region))
            x_inf, x_sup, y_inf, y_sup = lims_from_region(
                None, coords=region, dtype=float
            )
            verboseprint(
                "x_inf, x_sup, y_inf, y_sup = %s, %s, %s, %s"
                % (x_inf, x_sup, y_inf, y_sup)
            )
            data = d.get_pyfits()[0].data
            if len(data.shape) == 2:
                if plot_:
                    import matplotlib.pyplot as plt

                    plt.imshow(
                        data[
                            give_value(y_inf + 0.5) : give_value(y_sup),
                            give_value(x_inf + 0.5) : give_value(x_sup),
                        ]
                    )
                    plt.colorbar()
                data = data[
                    np.max([give_value(y_inf + 0.5), 0]) : give_value(y_sup),
                    np.max([give_value(x_inf + 0.5), 0]) : give_value(x_sup),
                ]
                datas.append(data)
            if len(data.shape) == 3:
                data = data[
                    :,
                    np.max([give_value(y_inf + 0.5), 0]) : give_value(y_sup),
                    np.max([give_value(x_inf + 0.5), 0]) : give_value(x_sup),
                ]
                datas.append(data)
    if len(datas) > 1:
        return datas
    else:
        return datas[0]


def raise_create_region(d):
    message(
        d,
        """Please create and select a region (Circle/Box)
before runnning this analysis""",
    )


def raise_create_plot(d):
    message(
        d,
        """Please create a plot by creating a Region->Shape->Projection
or an histogram of any region! """,
    )
    return


def fit_gaussian_2d(xpapoint=None, n=300, cmap="twilight_shifted", argv=[]):
    """2D gaussian fitting on the encircled region in DS9 [DS9 required]
    """
    from astropy.io import fits
    from scipy.optimize import curve_fit
    from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--plot",
        default="1",
        help="Interactive plot for gaussian fitting",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=False)
    fluxes = []
    fwhm, center, test = "-", 0, 1
    plot_ = bool(int(args.plot))
    d = DS9n(args.xpapoint)
    region = getregion(d, selected=True, message=True, quick=True)  # [0]
    if bool(int(test)):
        plot_ = True
        image = d.get_pyfits()[0].data
        try:
            # print(region)
            x_inf, x_sup, y_inf, y_sup = lims_from_region(None, coords=region)
            # print(x_inf, x_sup, y_inf, y_sup)
        except Exception:
            raise_create_region(d)
            sys.exit()
        image = image[y_inf:y_sup, x_inf:x_sup]
        # print(image)
        # print(x_inf, x_sup, y_inf, y_sup)
        while np.isfinite(image).all() == False:  # keep == since array
            kernel = Gaussian2DKernel(x_stddev=2, y_stddev=2)
            image = interpolate_replace_nans(image, kernel)
            verboseprint(np.isfinite(image).all())
        lx, ly = image.shape
        lx, ly = ly, lx
        x = np.linspace(0, lx - 1, lx)
        y = np.linspace(0, ly - 1, ly)
        x, y = np.meshgrid(x, y)
        if fwhm.split("-")[0] == "":
            if bool(int(center)):
                param = (
                    np.nanmax(image),
                    lx / 2,
                    ly / 2,
                    2,
                    2,
                    0,
                    np.percentile(image, 15),
                )
                bounds = (
                    [
                        -np.inf,
                        lx / 2 - 0.5,
                        ly / 2 - 0.00001,
                        0.5,
                        0.5,
                        -np.inf,
                    ],
                    [np.inf, lx / 2 + 0.00001, ly / 2 + 0.5, 10, 10, np.inf],
                )  # (-np.inf, np.inf)#
            else:
                xo, yo = (
                    np.where(image == np.nanmax(image))[1][0],
                    np.where(image == np.nanmax(image))[0][0],
                )
                param = (
                    np.nanmax(image),
                    int(xo),
                    int(yo),
                    2,
                    2,
                    0,
                    np.percentile(image, 15),
                )
                bounds = (
                    [-np.inf, xo - 10, yo - 10, 0.5, 0.5, -np.inf],
                    [np.inf, xo + 10, yo + 10, 10, 10, np.inf],
                )  # (-np.inf, np.inf)#
        else:
            stdmin, stdmax = np.array(fwhm.split("-"), dtype=float) / 2.35
            if bool(int(center)):
                param = (
                    np.nanmax(image),
                    lx / 2,
                    ly / 2,
                    (stdmin + stdmax) / 2,
                    (stdmin + stdmax) / 2,
                    0,
                    np.percentile(image, 15),
                )
                bounds = (
                    [
                        -np.inf,
                        lx / 2 - 0.5,
                        ly / 2 - 0.00001,
                        stdmin,
                        stdmin,
                        -np.inf,
                    ],
                    [
                        np.inf,
                        lx / 2 + 0.00001,
                        ly / 2 + 0.5,
                        stdmax,
                        stdmax,
                        np.inf,
                    ],
                )  # (-np.inf, np.inf)#
            else:
                xo, yo = (
                    np.where(image == np.nanmax(image))[1][0],
                    np.where(image == np.nanmax(image))[0][0],
                )
                param = (
                    np.nanmax(image),
                    xo,
                    yo,
                    (stdmin + stdmax) / 2,
                    (stdmin + stdmax) / 2,
                    0,
                    np.percentile(image, 15),
                )
                bounds = (
                    [-np.inf, xo - 10, yo - 10, stdmin, stdmin, -np.inf],
                    [np.inf, xo + 10, yo + 10, stdmax, stdmax, np.inf],
                )  # (-np.inf, np.inf)#
        try:
            verboseprint(bounds)
            popt, pcov = curve_fit(gaussian_2dim, (x, y), image.flat, param)
        except RuntimeError as e:
            logger.warning(e)
            popt = [0, 0, 0, 0, 0, 0]
        verboseprint("popt = %s" % (popt))
        fluxes.append(2 * np.pi * popt[3] * popt[4] * popt[0])
        xn, yn = popt[1], popt[2]
        verboseprint("New center = %s %s " % (popt[1], popt[2]))
        verboseprint("New center = %s %s " % (x_inf, y_inf))
        verboseprint(
            x_inf + xn + 1,
            y_inf + yn + 1,
            2.35 * popt[3],
            2.35 * popt[4],
            180 * popt[5] / np.pi,
        )
        d.set(
            'regions format ds9 ; regions system detector ; regions command "ellipse %0.1f %0.1f %0.1f %0.1f %0.1f # color=yellow "'
            % (
                x_inf + xn + 1,
                y_inf + yn + 1,
                2.35 * popt[3],
                2.35 * popt[4],
                180 * popt[5] / np.pi,
            )
        )

    if plot_:
        from pyvista import Plotter, StructuredGrid, PolyData, set_plot_theme

        z = gaussian_2dim((x, y), *popt).reshape(x.shape)
        xx, yy = np.indices(image.shape)
        set_plot_theme("document")
        p = Plotter(
            notebook=False,
            window_size=[2 * 1024, 2 * 768],
            line_smoothing=True,
            point_smoothing=True,
            polygon_smoothing=True,
            splitting_position=None,
            title="3D plot, FLUX = %0.1f" % (fluxes[0])
            + "amp = %0.3f, sigx = %0.3f, sigy = %0.3f, angle = %id "
            % (popt[0], popt[3], popt[4], (180 * popt[5] / np.pi) % 180),
        )
        value = image
        z, image = image, z
        value = image.shape[0] / (image.max() - image.min()) / 3
        fit = StructuredGrid()
        data_mesh = StructuredGrid()
        data_mesh.points = PolyData(
            np.c_[
                xx.reshape(-1),
                yy.reshape(-1),
                ((z - np.nanmin(z)) * value).reshape(-1),
            ]
        ).points
        data_mesh["Intensity"] = image.ravel()
        data_mesh.dimensions = [z.shape[1], z.shape[0], 1]
        points = np.c_[
            xx.reshape(-1),
            yy.reshape(-1),
            ((image - np.nanmin(z)) * value).reshape(-1),
        ]
        data_points = PolyData(points)
        fit.points = data_points.points
        fit["z"] = image.ravel()
        fit.dimensions = [image.shape[1], image.shape[0], 1]
        p1 = p.add_mesh(
            fit,
            opacity=0.7,
            nan_opacity=0,
            use_transparency=False,
            name="3D plot, FLUX = %0.1f" % (fluxes[0]),
            flip_scalars=True,
            scalar_bar_args={"title": "Value"},
            scalars=z.flatten() + image.flatten(),
        )
        p2 = p.add_mesh(
            data_mesh,
            scalars=z.flatten() + image.flatten(),
            opacity=1 - 0.7,
            nan_opacity=0,
            use_transparency=False,
            flip_scalars=True,
            scalar_bar_args={"title": "Value"},
        )  # y=True, opacity=0.3,,pickable=True)
        p.add_text(
            "Gaussian fit: F = %0.0f, FWHMs = %0.1f, %0.1f, angle=%0.0fd"
            % (
                2 * np.pi * popt[3] * popt[4] * popt[0],
                popt[3],
                popt[4],
                (180 * popt[5] / np.pi) % 180,
            ),
            name="mylabel",
            position=(70, 10),
        )
        dict_ = {}

        def callback(value):
            p1.GetProperty().SetOpacity(value)
            p2.GetProperty().SetOpacity(1 - value)
            return

        def update_text(text):
            p.add_text(text, name="mylabel", position=(70, 10))

        def picking_callable(mesh):
            dict_["mesh"] = mesh
            # verboseprint(dict_["mesh"])
            x_inf, x_sup, y_inf, y_sup = np.array(
                dict_["mesh"].bounds[:4], dtype=int
            )
            data = (z[x_inf:x_sup, y_inf:y_sup] - np.nanmin(z)) * value
            x_new, y_new = (
                x[x_inf:x_sup, y_inf:y_sup],
                y[x_inf:x_sup, y_inf:y_sup],
            )
            xo, yo = (
                np.where(data == np.nanmax(data))[1][0],
                np.where(data == np.nanmax(data))[0][0],
            )
            P0 = (
                np.ptp(data),
                xo + x_inf,
                yo + y_inf,
                2,
                2,
                0,
                np.percentile(data, 15),
            )
            P0 = (
                np.ptp(data),
                yo + y_inf,
                xo + x_inf,
                2,
                2,
                0,
                np.percentile(data, 15),
            )
            dict_["P0"] = P0
            args, cov = curve_fit(
                gaussian_2dim, (x_new, y_new), data.flatten(), p0=P0
            )  # ,bounds=bounds)
            dict_["args"] = args
            points[:, -1] = (
                gaussian_2dim((x, y), *args).reshape(x.shape).reshape(-1)
            )
            dict_["new_fit"] = gaussian_2dim((x, y), *args).reshape(x.shape)
            p.update_coordinates(points, mesh=fit)
            p.update_scalars(z.flatten() + points[:, -1] / value, mesh=fit)
            update_text(
                "Gaussian fit: F = %0.0f, FWHMs = %0.1f, %0.1f, angle=%0.0fd"
                % (
                    2 * np.pi * args[3] * args[4] * args[0],
                    args[3],
                    args[4],
                    (180 * args[5] / np.pi) % 180,
                )
            )  #
            xn, yn = args[1], args[2]
            d.set(
                'regions command "ellipse %0.1f %0.1f %0.1f %0.1f %0.1f # color=yellow "'
                % (
                    x_inf + xn + 1,
                    y_inf + yn + 1,
                    2.35 * args[3],
                    2.35 * args[4],
                    180 * args[5] / np.pi,
                )
            )
            return  # scalars

        p.enable_cell_picking(
            mesh=data_mesh,
            callback=picking_callable,
            through=True,
            show=True,
            show_message="Press r to select another region to fit a 2D gaussian. \n r again to quit selection mode.",
            style="wireframe",
            line_width=2,
            color="black",
            font_size=14,
            start=False,
        )

        p.add_slider_widget(
            callback,
            rng=[0, 1],
            value=0.7,
            title="Transparency ratio",
            color=None,
            pass_widget=False,
            event_type="always",
            style=None,
        )
        p.clear_box_widgets()
        p.add_axes()
        p.show()
    return


def astrometry_net(xpapoint=None, argv=[]):
    """Uses astrometry.net to compute position on the sky and return header
    """
    from astropy.io import fits
    from astropy.wcs import wcs

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument("--type", choices=("Image", "XY-catalog"), metavar="")
    parser.add_argument(
        "--scale-units",
        default="arcsecperpix",
        help="Units for scale estimate",
        metavar="",
    )  # choices=('arcsecperpix', 'arcminwidth', 'degwidth', 'focalmm'),
    parser.add_argument(
        "--scale-lower",
        default="",
        type=str,
        help="Scale lower-bound",
        metavar="",
    )
    parser.add_argument(
        "--scale-upper",
        default="",
        type=str,
        help="Scale upper-bound",
        metavar="",
    )
    parser.add_argument(
        "--scale-est", default="", type=str, help="Scale estimate", metavar=""
    )
    parser.add_argument(
        "--scale-err",
        default="",
        type=str,
        help="Scale estimate error (in PERCENT)",
        metavar="",
    )
    parser.add_argument(
        "--ra", default="", type=str, help="RA center", metavar=""
    )
    parser.add_argument(
        "--dec", default="", type=str, help="Dec center", metavar=""
    )
    parser.add_argument(
        "--radius",
        default="",
        type=str,
        help="Search radius around RA,Dec center",
        metavar="",
    )
    parser.add_argument(
        "--downsample",
        default="",
        type=str,
        help="Downsample image by this factor",
        metavar="",
    )
    parser.add_argument(
        "--tweak-order",
        default="",
        type=str,
        help="SIP distortion order (default: 2)",
        metavar="",
    )
    parser.add_argument(
        "--crpix-center",
        default="",
        help="Set reference point to center of image?",
        metavar="",
    )
    parser.add_argument(
        "--parity",
        default="0",
        choices=("0", "1"),
        help="Parity (flip) of image",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "--positional_error",
        default="",
        dest="positional_error",
        type=str,
        help="How many pixels a star may be from where it should be.",
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)

    d = DS9n(args.xpapoint)
    filename = get_filename(d)  # d.get("file")
    type_ = args.type
    verboseprint("Type = %s" % (type_))
    if type_ == "XY-catalog":
        name = "/tmp/centers_astrometry.fits"
        save_region_as_catalog(argv="--path " + name)
        filename = name
    verboseprint(
        "No header WCS - Applying lost in space algorithm: Internet needed!"
    )
    verboseprint("Processing might take a few minutes ~5-10")
    PathExec = os.path.dirname(os.path.realpath(__file__)) + "/astrometry3.py"
    Newfilename = filename[:-5] + "_wcs.fits"
    create_wcs(PathExec, filename, Newfilename, params=args, type_=type_)

    wcs_header = wcs.WCS(fits.getheader(Newfilename)).to_header()
    filename = get_filename(d)
    for key in list(dict.fromkeys(wcs_header.keys())):
        verboseprint(key)
        try:
            fits.setval(filename, key, value=wcs_header[key], comment="")
        except ValueError as e:
            logger.warning(e)
    fits.setval(
        filename,
        "WCSDATE",
        value=datetime.datetime.now().strftime("%y%m%d-%HH%M"),
        comment="",
    )
    d.set("lock frame wcs")
    message(
        d,
        """Astrometry.net performed successfully!
                  The WCS header has been saved in you image.""",
    )
    return


def create_wcs(PathExec, filename, Newfilename, params, type_="Image"):
    """Sends the image on the astrometry.net server
    and run a lost in space algorithm to have this header.
    Processing might take a few minutes
    """
    options = [
        "scale-units",
        # ' --scale-type ',
        "scale-lower",
        "scale-upper",
        "scale-est",
        "scale-err",
        "ra",
        "dec",
        "radius",
        "downsample",
        "crpix-center",
        "parity",
        "positional_error",
        "tweak-order",
        #' --use_sextractor ',
    ]
    if type_ == "XY-catalog":
        d = DS9n()
        image = d.get_pyfits()[0]
        lx, ly = image.shape
        upload = " --image-width %i --image-height %i  --upload-xy " % (lx, ly)
        verboseprint(options)
        verboseprint(params)
    else:
        upload = "--upload "
    verboseprint(type_)
    verboseprint(upload)
    # params = ['-'] * len(options)
    param_dict = {}
    # params.scale
    for key in zip(options):
        if getattr(params, key[0].replace("-", "_")) != "":
            param_dict[key[0]] = getattr(params, key[0].replace("-", "_"))

    parameters = " --" + " --".join(
        [k + " " + str(param_dict[k]) for k in list(param_dict.keys())[:]]
    )
    verboseprint(parameters)
    verboseprint(os.path.dirname(filename) + "/--wait.fits")
    start = time.time()
    verboseprint("Start lost in space algorithm - might take a few minutes")
    executable = (
        "/Users/Vincent/opt/anaconda3/bin/python3 "
        + PathExec
        + " --apikey apfqmasixxbqxngm --wcs "
        + Newfilename
        + " --private y "
        + upload
        + filename
        + " "
        + parameters
    )
    verboseprint(executable)
    import subprocess

    result = subprocess.run(
        executable.split(), stderr=sys.stderr, stdout=sys.stderr
    )
    print(result.stderr)
    stop = time.time()
    verboseprint("File created")
    verboseprint("Lost in space duration = {} seconds".format(stop - start))
    return


def original_settings(xpapoint=None, argv=[]):
    """Return to original settings [DS9 required]
    """
    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv, required=False)

    d = DS9n(args.xpapoint)
    d.set("cmap grey")  # d.set("regions delete all")
    d.set("scale linear")
    d.set("scale mode minmax")
    d.set("grid no")
    d.set("smooth no")
    d.set("lock bin no")
    return d


def ds9entry(xpapoint=None, message="", quit_=False):
    """Opens DS9 native entry dialog box
    """
    d = DS9n(xpapoint)
    if isinstance(d, FakeDS9):
        answer = input("%s" % (message))
    else:
        answer = d.get("analysis entry {%s}" % (message))
        answer = answer.rstrip()[::-1].rstrip()[::-1]
    if quit_ & (answer == ""):
        sys.exit()
    else:
        return answer


def parse_data(data, map=map, float=float):
    """Parse DS9 defined region to give pythonic regions
    """
    import numpy as np

    vals = []
    xy = []
    for s in data.split("\n"):
        coord, val = s.split("=")
        val = val.strip() or np.nan
        xy.append(list(map(float, coord.split(","))))
        vals.append(float(val))
    vals = np.array(vals)
    xy = np.floor(np.array(xy)).astype(int)
    x, y = xy[:, 0], xy[:, 1]
    w = x.ptp() + 1
    h = y.ptp() + 1
    arr = np.empty((w, h))
    X = np.empty((w, h))
    Y = np.empty((w, h))
    indices = x - x.min(), y - y.min()
    arr[indices] = vals
    X[indices] = x
    Y[indices] = y
    return X.T, Y.T, arr.T


def process_region(regions, win, quick=False, message=True, dtype=int):
    """Process DS9 regions to return pythonic regions
    """
    from collections import namedtuple
    import numpy as np

    processed_regions = []

    for i, region in enumerate(regions):
        try:
            name, info = region.split("(")
        except ValueError:
            if message:
                d = win
                raise_create_region(d)
                sys.exit()
        coords = [float(c) for c in info.split(")")[0].split(",")]
        if quick:
            # verboseprint(regions)
            if len(regions) == 1:
                verboseprint("Only one region, taking it")
                return np.array(coords, dtype=dtype)
            elif len(regions) == 0:
                verboseprint("There are no regions here")
                raise ValueError
            else:
                processed_regions.append(np.array(coords, dtype=int))
        else:
            if name == "box":
                xc, yc, w, h, angle = coords
                box = namedtuple("Box", "data xc yc w h angle")
                processed_regions.append(box(0, xc, yc, w, h, angle))
            elif name == "bpanda":
                xc, yc, a1, a2, a3, a4, a5, w, h, a6, a7 = coords
                box = namedtuple("Box", "data xc yc w h angle")
                processed_regions.append(box(0, xc, yc, w, h, 0))
            elif name == "circle":
                xc, yc, r = coords
                Xc, Yc = np.floor(xc), np.floor(yc)
                circle = namedtuple("Circle", "data databox inside xc yc r")
                processed_regions.append(circle(0, 0, 0, xc, yc, r))
            elif name == "# vector":
                xc, yc, xl, yl = coords
                vector = namedtuple("Vector", "data databox inside xc yc r")
                processed_regions.append(vector(xc, yc, xl, yl, 0, 0))
            elif name == "ellipse":
                if len(coords) == 5:
                    xc, yc, a2, b2, angle = coords
                else:
                    xc, yc, a1, b1, a2, b2, angle = coords
                w = 2 * a2
                h = 2 * b2
                dat = win.get(
                    "data physical %s %s %s %s no" % (xc - a2, yc - b2, w, h)
                )
                X, Y, arr = parse_data(dat)
                Xc, Yc = np.floor(xc), np.floor(yc)
                inside = ((X - Xc) / a2) ** 2 + ((Y - Yc) / b2) ** 2 <= 1
                if len(coords) == 5:
                    ellipse = namedtuple(
                        "Ellipse", "data databox inside xc yc a b angle"
                    )
                    return ellipse(arr, arr, inside, xc, yc, a2, b2, angle)

                inside &= ((X - Xc) / a1) ** 2 + ((Y - Yc) / b1) ** 2 >= 1
                annulus = namedtuple(
                    "EllipticalAnnulus",
                    "data databox inside xc yc a1 b1 a2 b2 angle",
                )
                processed_regions.append(
                    annulus(arr, arr, inside, xc, yc, a1, b1, a2, b2, angle)
                )
            elif name == "polygon":
                # return(coords)
                processed_regions.append(coords)
            else:
                raise ValueError("Can't process region %s" % name)
    # if len(processed_regions) == 1:
    #     return processed_regions  # [0]
    # else:
    return processed_regions


# @fn_timer
def getregion(
    win,
    debug=False,
    all=False,
    quick=False,
    selected=False,
    message=True,
    system="Image",
    dtype=int,
):
    """ Read a region from a ds9 instance.
    Returns a tuple with the data in the region.
    """
    win.set("regions format ds9 ; regions system %s" % (system))
    if all is False:
        regions = win.get("regions selected")
        if len([row for row in regions.split("\n")]) >= 3:
            rows = regions
        elif selected is False:
            verboseprint("no region selected")
            try:
                rows = win.get("regions all")
            except TypeError:
                raise_create_region(win)
                sys.exit()
        else:
            return None

    else:
        verboseprint("Taking all regions")
        rows = win.get("regions all")
        try:
            rows = win.get("regions all")
        except TypeError:
            raise_create_region(win)
            sys.exit()

    rows = [row for row in rows.split("\n")]
    if len(rows) < 3:
        verboseprint("No regions found")
    if all or selected:
        if (
            ("circle" in rows[2])
            | ("box" in rows[2])
            | ("projection" in rows[2])
            | ("ellipse" in rows[2])
        ):
            region = process_region(
                rows[2:], win, quick=quick, message=message, dtype=dtype
            )
        else:
            region = process_region(
                rows[3:], win, quick=quick, message=message, dtype=dtype
            )
        if type(region) == list:
            return region
        else:
            return [region]

    else:
        return process_region([rows[-1]], win, quick=quick, message=message)


def enc(x, ENCa):
    """Return encoder step of FB2 tip-tilds focus A
    """
    a = (ENCa[-1] - ENCa[0]) / (len(ENCa) - 1) * x + ENCa[0]
    return a


def gaussian(x, amp, x0, sigma):
    """Gaussian funtion
    """
    import numpy as np

    return amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def throughfocus_wcs(
    center,
    files,
    # x=None,
    fibersize=0,
    center_type="barycentre",
    SigmaMax=4,
    plot_=True,
    Type=None,
    ENCa_center=None,
    pas=None,
    WCS=False,
    DS9backUp=DS9_BackUp_path,
    offsets=10,
):
    """Same algorithm than throughfocus except it works on WCS coordinate
    and not on pixels. Then the throughfocus can be run on stars even
    with a sky drift
    """
    from astropy.io import fits
    from astropy.table import Table, vstack
    import numpy as np

    from scipy.optimize import curve_fit

    fwhm = []
    EE50 = []
    EE80 = []
    maxpix = []
    sumpix = []
    varpix = []
    xo = []
    yo = []
    images = []
    ENCa = []
    ext = fits_ext(fits.open(files[0]))
    x = offsets
    for file in files:
        filename = file
        with fits.open(filename) as f:
            fitsfile = f[ext]
            image = fitsfile.data
        header = fitsfile.header
        nombre = 5
        if ENCa_center is not None:
            ENCa = np.linspace(
                ENCa_center - nombre * pas,
                ENCa_center + nombre * pas,
                2 * nombre + 1,
            )[::-1]
        if WCS:
            from astropy import units as u
            from astropy import wcs

            w = wcs.WCS(header)
            center_wcs = center
            center_pix = w.all_world2pix(
                center_wcs[0] * u.deg, center_wcs[1] * u.deg, 0
            )
            center_pix = [int(center_pix[0]), int(center_pix[1])]
        else:
            center_pix = center
        d = analyze_spot(
            image,
            center=center_pix,
            fibersize=fibersize,
            center_type=center_type,
            SigmaMax=SigmaMax,
        )
        background = 1 * estimate_background(image, center)
        n = 25
        subimage = (image - background)[
            int(center_pix[1]) - n : int(center_pix[1]) + n,
            int(center_pix[0]) - n : int(center_pix[0]) + n,
        ]
        images.append(subimage)

        max20 = subimage.flatten()
        max20.sort()
        fwhm.append(d["Sigma"])
        EE50.append(d["EE50"])
        EE80.append(d["EE80"])
        xo.append(d["Center"][0])
        yo.append(d["Center"][1])
        maxpix.append(max20[-20:].mean())
        sumpix.append(d["Flux"])
        varpix.append(subimage.var())
    f = lambda x, a, b, c: a * (x - b) ** 2 + c
    xtot = np.linspace(x.min(), x.max(), 200)
    # if Type == "guider":
    #     x = np.array(ENCa)
    #     xtot = np.linspace(x.min(), x.max(), 200)
    #     ENC = lambda x, a: x
    # if Type == "detector":
    #     x = np.arange(len(files))
    #     xtot = np.linspace(x.min(), x.max(), 200)
    #     if len(ENCa) == 0:
    #         ENC = lambda x, a: 0
    #     else:
    #         ENC = (
    #             lambda x, a: (ENCa[-1] - ENCa[0]) / (len(ENCa) - 1) * x + ENCa[0]
    #         )
    try:
        opt1, cov1 = curve_fit(f, x, fwhm)
        bestx1 = xtot[np.argmin(f(xtot, *opt1))]
        np.savetxt("/tmp/fwhm_fit.dat", np.array([xtot, f(xtot, *opt1)]).T)
    except RuntimeError as e:
        logger.warning(e)
        opt1 = [np.nan, np.nan, np.nan]  # [0,0,0]
        bestx1 = np.nan
        pass
    try:
        opt2, cov2 = curve_fit(f, x, EE50)
        bestx2 = xtot[np.argmin(f(xtot, *opt2))]
        np.savetxt("/tmp/EE50_fit.dat", np.array([xtot, f(xtot, *opt2)]).T)
    except RuntimeError:
        opt2 = [np.nan, np.nan, np.nan]  # [0,0,0]
        bestx2 = np.nan
        pass
    try:
        opt3, cov3 = curve_fit(f, x, EE80)
        bestx3 = xtot[np.argmin(f(xtot, *opt3))]
        np.savetxt("/tmp/EE80_fit.dat", np.array([xtot, f(xtot, *opt3)]).T)
    except RuntimeError:
        opt3 = [np.nan, np.nan, np.nan]  # [0,0,0]
        bestx3 = np.nan
        pass
    try:
        opt4, cov4 = curve_fit(f, x, maxpix)
        bestx4 = xtot[np.argmax(f(xtot, *opt4))]
        np.savetxt("/tmp/maxpix_fit.dat", np.array([xtot, f(xtot, *opt4)]).T)
    except RuntimeError:
        bestx4 = np.nan
        opt4 = [np.nan, np.nan, np.nan]  # [0,0,0]

        pass
    try:
        opt6, cov6 = curve_fit(f, x, varpix)
        bestx6 = xtot[np.argmax(f(xtot, *opt6))]
    except RuntimeError:
        opt6 = [0, 0, 0]
        bestx6 = np.nan
        pass
    # mean = np.nanmean(
    #     np.array(
    #         [
    #             enc(bestx1, ENCa),
    #             enc(bestx2, ENCa),
    #             enc(bestx3, ENCa),
    #             enc(bestx4, ENCa),
    #             enc(bestx6, ENCa),
    #         ]
    #     )
    # )
    name = "%s - %i - %i - %s " % (
        os.path.basename(filename),
        int(center_pix[0]),
        int(center_pix[1]),
        0,
    )
    # verboseprint(name)
    t = Table(
        names=(
            "name",
            "number",
            "x",
            "y",
            "Sigma",
            "EE50",
            "EE80",
            "Max pix",
            "Flux",
            "Var pix",
        ),
        dtype=("S15", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4", "f4",),
    )
    t.add_row(
        (
            os.path.basename(filename),
            os.path.basename(filename)[5:11],
            # t2s(h=h, m=m, s=s, d=day),
            d["Center"][0],
            d["Center"][1],
            min(fwhm),
            min(EE50),
            min(EE80),
            min(maxpix),
            min(sumpix),
            max(varpix),
        )
    )
    np.savetxt("/tmp/fwhm.dat", np.array([x, fwhm]).T)
    np.savetxt("/tmp/EE50.dat", np.array([x, EE50]).T)
    np.savetxt("/tmp/EE80.dat", np.array([x, EE80]).T)
    np.savetxt("/tmp/maxpix.dat", np.array([x, maxpix]).T)
    try:
        OldTable = Table.read(os.path.dirname(filename) + "/Throughfocus.csv")
    except IOError as e:
        logger.warning(e)
        t.write(os.path.dirname(filename) + "/Throughfocus.csv")
    else:
        t = vstack((OldTable, t))
        t.write(os.path.dirname(filename) + "/Throughfocus.csv", overwrite=True)
    d = []
    d.append("plot line open")  # d.append("plot axis x grid no ")
    d.append("plot axis y grid no ")
    d.append(
        "plot title 'Fit: Best FWHM = %0.2f - Position = %0.2f' "
        % (np.nanmin(f(xtot, *opt1)), xtot[np.argmin(f(xtot, *opt1))])
    )
    d.append("plot title y 'FWHM' ")
    d.append("plot load /tmp/fwhm.dat xy")
    d.append("plot line shape circle ")
    d.append("plot line width 0 ")
    d.append("plot line shape color black")  # d.append("plot legend yes ")
    d.append("plot legend position right ")
    d.append("plot load /tmp/fwhm_fit.dat xy")
    d.append("plot line dash yes ")
    d.append("plot title legend ''")
    d.append("plot name 'FWHM = %i, FWHM_fit = %0.1f' " % (1, 1))
    d.append("plot add graph ")
    d.append("plot axis y grid no ")
    d.append(
        "plot title 'Fit: Best FWHM = %0.2f - Position = %0.2f' "
        % (np.nanmax(f(xtot, *opt4)), xtot[np.argmax(f(xtot, *opt4))])
    )
    d.append("plot load /tmp/maxpix.dat xy")
    d.append("plot title y 'Max pix' ")
    d.append("plot line shape circle ")
    d.append("plot line width 0 ")
    d.append("plot line shape color black")  # d.append("plot legend yes ")
    d.append("plot legend position right ")
    d.append("plot load /tmp/maxpix_fit.dat xy")
    d.append("plot line dash yes ")
    d.append("plot title legend ''")
    d.append("plot name 'FWHM = %i, FWHM_fit = %0.1f' " % (1, 1))
    d.append("plot add graph ")
    d.append("plot axis y grid no ")
    d.append(
        "plot title 'Fit: Best EE50 = %0.2f - Position = %0.2f' "
        % (np.nanmin(f(xtot, *opt2)), xtot[np.argmin(f(xtot, *opt2))])
    )
    d.append("plot load /tmp/EE50.dat xy")
    d.append("plot title y 'Radial profile' ")
    d.append("plot line shape circle ")
    d.append("plot line width 0 ")
    d.append("plot line shape color black")  # d.append("plot legend yes ")
    d.append("plot legend position right ")
    d.append("plot load /tmp/EE50_fit.dat xy")
    d.append("plot line dash yes ")
    d.append("plot title legend ''")
    d.append("plot name 'FWHM = %i, FWHM_fit = %0.1f' " % (1, 1))
    d.append("plot add graph ")
    d.append("plot axis y grid no ")
    d.append(
        "plot title 'Fit: Best EE80 = %0.2f - Position = %0.2f' "
        % (np.nanmin(f(xtot, *opt3)), xtot[np.argmin(f(xtot, *opt3))])
    )
    d.append("plot load /tmp/EE80.dat xy")
    d.append("plot title y 'Radial profile' ")
    d.append("plot line shape circle ")
    d.append("plot line width 0 ")
    d.append("plot line shape color black")  # d.append("plot legend yes ")
    d.append("plot legend position right ")
    d.append("plot load /tmp/EE80_fit.dat xy")
    d.append("plot line dash yes ")
    d.append("plot title legend ''")
    d.append("plot name 'FWHM = %i, FWHM_fit = %0.1f' " % (1, 1))
    d.append("plot layout GRID ; plot layout STRIP scale 100")
    d.append("plot font legend size 9 ")
    d.append("plot font labels size 13 ")
    ds9 = DS9n()
    ds9.set(" ; ".join(d))
    return images  # fwhm, EE50, EE80


def analyze_spot(
    data,
    center,
    size=40,
    n=1.5,
    radius=40,
    fit=True,
    center_type="barycentre",
    radius_ext=12,
    platescale=None,
    fibersize=100,
    SigmaMax=4,
):
    """Function used to plot the radial profile and the encircled energy,
    """
    from scipy import interpolate
    from scipy.optimize import curve_fit
    import numpy as np

    rsurf, rmean, profile, EE, NewCenter, stddev = radial_profile_normalized(
        data, center, radius=radius, n=n, center_type=center_type
    )
    profile = profile[
        :size
    ]  # (a[:n] - min(a[:n]) ) / np.nansum((a[:n] - min(a[:n]) ))
    fiber = fibersize / (2 * 1.08 * (1 / 0.083))
    if fiber == 0:
        gaus = lambda x, a, sigma: a ** 2 * np.exp(-np.square(x / sigma) / 2)
        popt, pcov = curve_fit(
            gaus, rmean[:size], profile, p0=[1, 2]
        )  # ,bounds=([0,0],[1,5]))#[1,1,1,1,1] (x,a,b,sigma,lam,alpha):
    else:
        popt, pcov = curve_fit(
            convolve_diskgaus_2d,
            rmean[:size],
            profile,
            p0=[1, fiber, 2, np.nanmean(profile)],
            bounds=(
                [0, 0.95 * fiber - 1e-5, 1, -1],
                [2, 1.05 * fiber + 1e-5, SigmaMax, 1],
            ),
        )
    EE_interp = interpolate.interp1d(rsurf[:size], EE[:size], kind="cubic")
    ninterp = 10
    xnew = np.linspace(
        rsurf[:size].min(), rsurf[:size].max(), ninterp * len(rsurf[:size])
    )
    mina = min(xnew[EE_interp(xnew)[: ninterp * size] > 79])
    minb = min(xnew[EE_interp(xnew)[: ninterp * size] > 49])
    if fiber == 0:
        flux = 2 * np.pi * np.square(popt[1]) * np.square(popt[0])
        d = {
            "Flux": flux,
            "SizeSource": 0,
            "Sigma": abs(popt[1]),
            "EE50": mina,
            "EE80": minb,
            "Platescale": platescale,
            "Center": NewCenter,
        }
        verboseprint(d)
    else:
        d = {
            "Flux": 0,
            "SizeSource": popt[1],
            "Sigma": abs(popt[2]),
            "EE50": mina,
            "EE80": minb,
            "Platescale": platescale,
            "Center": NewCenter,
        }
        verboseprint(d)
    return d


def throughfocus(xpapoint=None, plot_=True, argv=[]):
    """Perform a throughfocus analysis and return the best focused image
    [DS9 required]
    """
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="Paths of the images you want to analyse. Use regexp ",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-s",
        "--sort",
        default="AlphaNumerical",
        help="Way to sort files to create throughfocus profile",
        type=str,
        choices=["AlphaNumerical", "CreationDate", "DS9-Order"],
        metavar="",
    )
    parser.add_argument(
        "-w",
        "--WCS",
        default="0",
        help="Perform throughfocus using WCS coord (when drifting on sky)",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-v",
        "--value",
        default="",
        help="If throughfocus images not taken uniformely, provide the offset of each image separated by a coma, eg: 0,0.5,2,3",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=False)

    verboseprint("""\n\n\n\n      START THROUGHFOCUS \n\n\n\n""")
    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    if getregion(d, selected=True) is None:
        raise_create_region(d)
        sys.exit()
    a = getregion(d)[0]
    if args.path == "":
        path = get_filename(d, All=True, sort=False)
    else:
        path = globglob(args.path, xpapoint=args.xpapoint)

    WCS = bool(int(args.WCS))
    sort = args.sort
    if sort == "CreationDate":
        verboseprint("Sorting by creation date")
        path.sort(key=os.path.getctime)
    elif sort == "AlphaNumerical":
        verboseprint("Sorting in alpha-numerical order")
        path.sort()

    try:
        ENCa_center, pas = np.array(args.value.split("-"), dtype=float)
    except ValueError as e:
        logger.warning(e)
        verboseprint("No actuator given, taking header ones for guider images")
        ENCa_center, pas = None, None
    except IndexError as e:
        logger.warning(e)
        verboseprint("No actuator given, taking header ones for guider images")
        ENCa_center, pas = None, None
    x = np.arange(len(path))
    if len(path) < 3:
        message(
            d,
            """You need at least 3 images to perform a throughfocus analysis.
            Select at least 3 images and re-run the analysis.""",
        ), sys.exit()

    if args.value == "":
        offsets = np.arange(len(path))
    else:
        offsets = np.array(
            [float(value) for value in args.value.split(",") if value != ""]
        )
        if len(offsets) != len(path):
            message(
                d,
                """You entered %i offsets but you have %i images. Please re-run
                   the analysis and provide consistent number of offsets."""
                % (len(offsets), len(path)),
            ), sys.exit()

    image = fits.open(filename)[0]
    rp = analyze_spot(
        image.data, center=[np.int(a.xc), np.int(a.yc)], fibersize=0
    )
    x, y = rp["Center"]
    d.set("regions system image")
    verboseprint(
        "\n\n\n\n     Centring on barycentre of the DS9 image "
        "(need to be close to best focus) : %0.1f, %0.1f"
        "--> %0.1f, %0.1f \n\n\n\n"
        % (a.xc, a.yc, rp["Center"][0], rp["Center"][1])
    )
    if image.header["BITPIX"] == -32:
        Type = "guider"
    else:
        Type = "detector"

    if WCS:
        from astropy import wcs

        w = wcs.WCS(image.header)

        center_wcs = w.all_pix2world(x, y, 0)
        # d.set('crosshair {} {} physical'.format(x,y))
        alpha, delta = float(center_wcs[0]), float(center_wcs[1])
        verboseprint("alpha, delta = %s %s" % (alpha, delta))
        center = [alpha, delta]
    else:
        center = [x, y]
    datas = throughfocus_wcs(
        center=center,
        files=path,
        fibersize=0,
        center_type="user",
        SigmaMax=6,
        plot_=plot_,
        Type=Type,
        ENCa_center=ENCa_center,
        pas=pas,
        WCS=WCS,
        offsets=offsets,
    )

    from astropy.convolution import convolve, Gaussian2DKernel

    dat = [
        (data - np.nanmin(data)) / np.max(np.ptp(datas, axis=(1, 2)))
        for data in datas
    ]  # /(data-np.nanmin(data)).ptp()
    datc = [convolve(data, Gaussian2DKernel(x_stddev=1)) for data in dat]
    ptp = [data.ptp() for data in dat]
    a = Table([dat, datc, ptp], names=("VIGNET1", "VIGNET2", "AMPLITUDE"))
    pyvista_throughfocus(a)


def explore_throughfocus(xpapoint=None, argv=[]):
    """Create focus exploration based on sextractor catalog including VIGNETS
    """
    from astropy.convolution import convolve, Gaussian2DKernel
    import numpy as np
    from astropy.table import Table

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="SExtractor catalog with VIGNETS",
        metavar="",
    )
    parser.add_argument(
        "-s",
        "--sort",
        default="MAG_AUTO",
        help="Column to use to sort the PSFs",
        metavar="",
        choices=[
            "MAG_AUTO",
            "FWHM_IMAGE",
            "THETA_IMAGE",
            "ELLIPTICITY",
            "X_IMAGE",
            "Y_IMAGE",
            "AMPLITUDE",
        ],
    )
    args = parser.parse_args_modif(argv)
    d = DS9n(args.xpapoint)
    a = Table.read(args.path)
    if len(a.colnames):
        a = Table.read(args.path, format="fits", hdu="LDAC_OBJECTS")
    if "VIGNET" not in a.colnames:
        message(
            d,
            """There is no VIGNET in the input catalog. Please make sure to use
               the parameter file sex_vignet.param when using SExtractor.""",
        )
        sys.exit()
    mask = np.nanmin(a["VIGNET"], axis=(1, 2)) > -1e30
    a = a[mask]
    a["VIGNET1"] = [
        (data - np.nanmin(data)) / (data - np.nanmin(data)).ptp()
        for data in a["VIGNET"]
    ]  #
    a["VIGNET2"] = [
        convolve(data, Gaussian2DKernel(x_stddev=1)) for data in a["VIGNET1"]
    ]

    a["AMPLITUDE"] = [data.ptp() for data in a["VIGNET"]]
    a.sort(args.sort)
    pyvista_throughfocus(a)
    return


def pyvista_throughfocus(a):
    """Explore throughfocus using pyvista in 3d
    """
    from pyvista import Plotter, set_plot_theme  # StructuredGrid, PolyData,

    # from pyvistaqt import BackgroundPlotter
    import numpy as np

    set_plot_theme("document")
    p = Plotter(
        notebook=False,
        window_size=(1500, 1600),
        line_smoothing=True,
        point_smoothing=True,
        polygon_smoothing=True,
        splitting_position=None,
        title="Throughfocus",
    )
    value = a["VIGNET1"][0].shape[0]
    mesh = create_mesh(a["VIGNET1"][0], value=value)
    p.add_mesh(
        mesh,
        scalars=a["VIGNET1"][0],
        opacity=0.9,
        nan_opacity=0,
        use_transparency=False,
        name="Data",
        flip_scalars=True,
        scalar_bar_args={"title": "Value"},
        show_scalar_bar=True,
    )  # , cmap='jet')
    dict_ = {"smooth": "VIGNET1", "number": 0}
    # p.show()
    labels = list(np.arange(len(a)) + 1)

    def update_text(text):
        text = int(text)
        p.add_text("Image: %i" % (text), name="mylabel")

    def throughfocus_callback(val):
        points = mesh.points.copy()
        data = a[dict_["smooth"]][int(val)]
        points[:, -1] = value * (data - np.nanmin(data)).reshape(-1)
        p.update_coordinates(points, render=False)
        scalar = a[dict_["smooth"]][int(val)].ravel()
        p.update_scalars(scalar, render=False)
        p.update_scalar_bar_range([np.nanmin(scalar), np.nanmax(scalar)])
        dict_["number"] = val
        return

    def create_gif(val):

        points = mesh.points.copy()
        p.open_gif("/tmp/throughfocus.gif")
        p.add_text("Image: 0", name="mylabel")
        # n = 1
        images = np.vstack([a, a[::-1]])
        for i, (data, lab) in enumerate(
            zip(images, (labels + labels[::-1]))
        ):  # +datas+datas+datas+datas+datas+datas:
            points[:, -1] = value * (data - np.nanmin(data)).reshape(-1)
            p.update_coordinates(points)  # , render=False)
            p.update_scalars(data.ravel())  # , render=False)
            update_text("Image: %i" % (lab))
            p.write_frame()

        return

    def smooth_callback(val):
        if val:
            name = "VIGNET2"
        else:
            name = "VIGNET1"

        points = mesh.points.copy()
        data = a[name][int(dict_["number"])]
        points[:, -1] = value * (data - np.nanmin(data)).reshape(-1)
        p.update_coordinates(points, render=False)
        scalar = a[name][int(dict_["number"])].ravel()
        p.update_scalars(scalar, render=False)
        p.update_coordinates(points)  # , render=False)
        p.update_scalars(data.ravel())  # , render=False)
        dict_["smooth"] = name

    p.add_text_slider_widget(
        throughfocus_callback,
        data=["%i" % (i) for i in np.arange(len(a))],
        value=0,
        event_type="always",
    )
    p.add_checkbox_button_widget(create_gif)
    p.add_text(
        "Create GIF in /tmp/thoughfocus.gif", name="button", position=(70, 10)
    )
    p.add_checkbox_button_widget(smooth_callback, position=(10, 80), value=False)
    p.add_text("Smooth", name="buttonSmooth", position=(70, 80))
    p.show()
    # p.app.exec_()


def radial_profile(xpapoint=None, plot_=True, fibersize=None, argv=[]):
    """Compute and plot the radial profile of the encircled source in DS9
    [DS9 required]
    """
    import numpy as np

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-c",
        "--centering",
        default="Maximum",
        help="Algorithm used for centering",
        type=str,
        choices=[
            "Maximum",
            "User",
            "Center-of-mass",
            "2x1D-Gaussian-fitting",
            "2D-Gaussian-fitting",
        ],
    )
    parser.add_argument(
        "-d",
        "--source_diameter",
        default="0",
        help="Fiber diameter in pixel. 0 for pure gaussian fitting, else data will be fitted by (Disk * Gaussian)",
        metavar="",
        type=str,
    )
    args = parser.parse_args_modif(argv, required=False)

    d = DS9n(args.xpapoint)
    center_type = args.centering

    if fibersize is None:
        if args.source_diameter.replace(".", "", 1).isdigit():
            fibersize = args.source_diameter
        else:
            fibersize = 0
            verboseprint("Fiber size not understood, setting it to 0")
    # if fibersize is None:
    log = False
    # verboseprint("log = ", log)

    filename = get_filename(d)  # d.get("file ")
    a = getregion(d)[0]
    fitsfile = d.get_pyfits()[0]
    try:
        ds9_plot_radial_profile(
            data=fitsfile.data,
            center=[np.int(a.xc), np.int(a.yc)],
            fibersize=fibersize,
            center_type=center_type,
            log=log,
            name=filename,
            radius=int(a.r),
            size=int(a.r),
            ds9=d,
        )  # int(a.r))
    except AttributeError:
        message(
            d,
            """Please define a circle region and select it. The radius
                      of the region is used to define the radial profile.""",
        )
        sys.exit()
    return


def radial_profile_normalized(
    data,
    center,
    anisotrope=False,
    angle=30,
    radius=40,
    n=1.5,
    center_type="barycentre",
    radius_bg=70,
    n1=20,
    size=70,
):
    """Function that returns the radial profile of a spot
    given an input image + center.
    """
    from scipy import ndimage
    import numpy as np

    y, x = np.indices((data.shape))
    verboseprint("center_type = %s" % (center_type))
    n1 = np.nanmin([n1, int(center[1]), int(center[0])])
    image = data[
        int(center[1]) - n1 : int(center[1]) + n1,
        int(center[0]) - n1 : int(center[0]) + n1,
    ]
    if center_type.lower() == "maximum":
        barycentre = np.array(
            [
                np.where(image == image.max())[0][0],
                np.where(image == image.max())[1][0],
            ]
        )
    if center_type.lower() == "barycentre":
        background = estimate_background(data, center, radius, 1.8)
        new_image = image - background
        index = new_image > 0.5 * np.nanmax(new_image)
        new_image[~index] = 0
        barycentre = ndimage.measurements.center_of_mass(new_image)
    if center_type.lower() == "user":
        barycentre = [n1, n1]
    else:
        verboseprint("Center type not understood, taking barycenter one")
        background = estimate_background(data, center, radius, 1.8)
        new_image = image - background
        index = new_image > 0.5 * np.nanmax(new_image)  # .max()
        new_image[~index] = 0
        barycentre = ndimage.measurements.center_of_mass(
            new_image
        )  # background#np.nanmin(image)
    new_center = np.array(center) + barycentre[::-1] - n1
    verboseprint(
        "new_center = {}, defined with center type: {}".format(
            new_center, center_type
        )
    )
    if radius_bg:
        fond = estimate_background(data, new_center, radius, n)
    else:
        fond = 0
    image = data - fond  # (data - fond).astype(np.int)

    r = np.sqrt((x - new_center[0]) ** 2 + (y - new_center[1]) ** 2)
    #    r = np.around(r)-1
    rint = r.astype(np.int)

    image_normalized = image
    # / np.nansum(image[r<radius])
    if anisotrope == True:
        theta = abs(
            180 * np.arctan((y - new_center[1]) / (x - new_center[0])) / np.pi
        )
        tbin_spectral = np.bincount(
            r[theta < angle].ravel(), image_normalized[theta < angle].ravel()
        )
        tbin_spatial = np.bincount(
            r[theta > 90 - angle].ravel(),
            image_normalized[theta > 90 - angle].ravel(),
        )
        nr_spectral = np.bincount(r[theta < angle].ravel())
        nr_spatial = np.bincount(r[theta > 90 - angle].ravel())
        EE_spatial = (
            100
            * np.nancumsum(tbin_spatial)
            / np.nanmax(np.nancumsum(tbin_spatial)[:100] + 1e-5)
        )
        EE_spectral = (
            100
            * np.nancumsum(tbin_spectral)
            / np.nanmax(np.nancumsum(tbin_spectral)[:100] + 1e-5)
        )
        return (
            tbin_spectral / nr_spectral,
            tbin_spatial / nr_spatial,
            EE_spectral,
            EE_spatial,
        )
    else:
        tbin = np.bincount(rint.ravel(), image_normalized.ravel())
        nr = np.bincount(rint.ravel())
        rsurf = np.sqrt(np.nancumsum(nr) / np.pi)
        rmean = np.bincount(rint.ravel(), r.ravel()) / nr
        dist = np.array(rint[rint < radius].ravel(), dtype=int)
        data = image[rint < radius].ravel()
        stdd = [
            np.nanstd(data[dist == distance])
            / np.sqrt(len(data[dist == distance]))
            for distance in np.arange(size)
        ]
        radialprofile = tbin / nr
        EE = (
            np.nancumsum(tbin)
            * 100
            / np.nanmax(np.nancumsum(tbin)[:radius] + 1e-5)
        )
        return (
            rsurf[:size],
            rmean[:size],
            radialprofile[:size],
            EE[:size],
            new_center[:size],
            stdd[:size],
        )


def estimate_background(data, center, radius=30, n=1.8):
    """Function that estimate the Background behing a source given an inner
    radius and a factor n to the outter radius such as the background is
    computed on the area which is on C2(n*radius)/C1(radius)
    """
    import numpy as np

    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)
    mask = (r >= radius) & (r <= n * radius)
    fond = np.nanmean(data[mask])
    return fond


def convolve_diskgaus_2d(r, amp=2, RR=4, sig=4 / 2.35, offset=0):
    """Convolution of a disk with a gaussian to simulate the image of a fiber
    """
    from scipy.integrate import quad
    from scipy import special
    import numpy as np

    integrand = (
        lambda eta, r_: special.iv(0, r_ * eta / np.square(sig))
        * eta
        * np.exp(-np.square(eta) / (2 * np.square(sig)))
    )
    integ = [
        quad(integrand, 0, RR, args=(r_,))[0]
        * np.exp(-np.square(r_) / (2 * np.square(sig)))
        / (np.pi * np.square(RR * sig))
        for r_ in r
    ]
    return offset + amp * np.array(integ)


def ds9_plot_radial_profile(
    data,
    center,
    size=40,
    n=1.5,
    log=False,
    anisotrope=False,
    angle=30,
    radius=40,
    ptype="linear",
    fit=True,
    center_type="barycentre",
    maxplot=0.013,
    minplot=-1e-5,
    radius_ext=12,
    platescale=None,
    fibersize=100,
    SigmaMax=4,
    DS9backUp=DS9_BackUp_path,
    name="",
    ds9=None,
):
    """Function used to plot the radial profile and the encirc energy of a spot
    """

    from scipy.optimize import curve_fit
    from scipy import interpolate
    import numpy as np

    rsurf, rmean, profile, EE, NewCenter, stddev = radial_profile_normalized(
        data,
        center,
        anisotrope=anisotrope,
        angle=angle,
        radius=radius,
        n=n,
        center_type=center_type,
        size=size,
    )
    profile = profile[:size]
    # (a[:n] - min(a[:n]) ) / np.nansum((a[:n] - min(a[:n]) ))
    rmean_long = np.linspace(0, rmean[:size].max(), 1000)

    fiber = float(fibersize)
    if fiber == 0:
        gaus = lambda x, a, sigma: a ** 2 * np.exp(-np.square(x / sigma) / 2)
        popt, pcov = curve_fit(gaus, rmean[:size], profile, p0=[1, 2])
        try:
            popt_m, pcov_m = curve_fit(
                moffat_1d, rmean[:size], profile, p0=[profile.max(), 4, 2.5]
            )
        except RuntimeError:
            popt_m = [profile.max(), 4, 2.5]
    else:
        stddev /= profile.max()
        profile /= profile.max()
        popt_m, pcov_m = curve_fit(
            moffat_1d, rmean[:size], profile, p0=[profile.max(), 4, 2.5]
        )
        popt, pcov = curve_fit(
            convolve_diskgaus_2d,
            rmean[:size],
            profile,
            p0=[np.nanmax(profile), fiber, 2, np.nanmin(profile)],
            bounds=(
                [
                    1e-3 * (profile.max() - profile.min()),
                    0.8 * fiber,
                    1,
                    profile.min(),
                ],
                [
                    1e3 * (profile.max() - profile.min()),
                    1.2 * fiber,
                    SigmaMax,
                    profile.max(),
                ],
            ),
        )

    EE_interp = interpolate.interp1d(rsurf[:size], EE[:size], kind="cubic")
    ninterp = 10
    xnew = np.linspace(
        rsurf[:size].min(), rsurf[:size].max(), ninterp * len(rsurf[:size])
    )
    mina = min(xnew[EE_interp(xnew)[: ninterp * size] > 79])
    minb = min(xnew[EE_interp(xnew)[: ninterp * size] > 49])

    np.savetxt("/tmp/1.dat", np.array([rmean, profile, stddev]).T)
    np.savetxt(
        "/tmp/2.dat",
        np.array(
            [xnew[EE_interp(xnew) > 0], EE_interp(xnew)[EE_interp(xnew) > 0]]
        ).T,
    )
    np.savetxt(
        "/tmp/3.dat", np.array([rmean_long, moffat_1d(rmean_long, *popt_m)]).T
    )

    np.savetxt("/tmp/4.dat", np.array([[minb, minb, 0], [0, 50, 50]]).T)
    np.savetxt("/tmp/5.dat", np.array([[mina, mina, 0], [0, 80, 80]]).T)
    np.savetxt("/tmp/6.dat", np.array([[0, size], [100, 100]]).T)

    if fiber == 0:
        flux = 2 * np.pi * np.square(popt[1]) * np.square(popt[0])
        d_ = {
            "Flux": flux,
            "SizeSource": 0,
            "Sigma": popt[1],
            "EE50": mina,
            "EE80": minb,
            "Platescale": platescale,
            "Center": NewCenter,
        }
        verboseprint(d_)
    else:
        d_ = {
            "Flux": 0,
            "SizeSource": popt[1],
            "Sigma": popt[2],
            "EE50": mina,
            "EE80": minb,
            "Platescale": platescale,
            "Center": NewCenter,
        }
        verboseprint(d_)
        csvwrite(
            np.vstack(
                (
                    rmean[:size],
                    profile,
                    convolve_diskgaus_2d(rmean[:size], *popt),
                )
            ).T,
            DS9backUp
            + "CSVs/%s_RadialProfile.csv"
            % (datetime.datetime.now().strftime("%y%m%d-%HH%M")),
        )
    csvwrite(
        np.vstack((rsurf, EE)).T,
        DS9backUp
        + "CSVs/%s_EnergyProfile.csv"
        % (datetime.datetime.now().strftime("%y%m%d-%HH%M")),
    )
    d = []
    d.append("plot line open")
    d.append("plot axis x grid no ")
    d.append("plot axis y grid no ")
    d.append(
        "plot title '%s , %0.1f - %0.1f' "
        % (os.path.basename(name), NewCenter[0], NewCenter[1])
    )
    d.append("plot title y 'Radial profile' ")
    d.append("plot load /tmp/1.dat xyey")
    d.append("plot legend yes ")
    d.append("plot legend position top ")
    d.append("plot title legend ''")
    d.append(
        "plot name 'Data: Flux = %i, FWHM_fit = %0.1f' "
        % (d_["Flux"], abs(popt[1]))
    )
    # d.append("plot line shape circle ")
    d.append("plot line dash yes ")
    # d.append("plot line shape color black")
    d.append("plot error color black ")
    d.append("plot load /tmp/3.dat xy  ")
    d.append(
        "plot name 'Moffat: Amp=%0.1f, alpha=%0.1f, beta=%0.1f' "
        % (popt_m[0], popt_m[1], popt_m[2])
    )
    d.append("plot line color cornflowerblue ")
    d.append("plot add graph ")
    d.append("plot axis x grid no")
    d.append("plot axis y grid no ")
    d.append("plot load /tmp/2.dat xy ")
    d.append("plot line color black ")
    d.append("plot title y 'Encircled energy' ")
    d.append("plot name 'Data' ")
    d.append("plot title x 'Distance from center' ")
    d.append("plot load /tmp/4.dat xy ")
    d.append("plot line color red ")
    d.append("plot legend yes ")
    d.append("plot legend position bottom ")
    d.append("plot title legend ''")
    d.append("plot name 'EE50 = %0.1fpix' " % (minb))
    d.append("plot load /tmp/5.dat xy ")
    d.append("plot name 'EE80 = %0.1fpix' " % (mina))
    d.append("plot line dash yes ")
    d.append("plot line color red ")
    d.append("plot load /tmp/6.dat xy ")
    d.append("plot name '100 percent limit' " % (mina))
    d.append("plot line dash yes ")
    d.append("plot line color black ")
    d.append("plot layout STRIP ; plot layout STRIP scale 100")
    d.append("plot font legend size 9 ")
    d.append("plot font labels size 13 ")
    d.append("plot current graph 1 ")
    ds9.set(" ; ".join(d))
    return d_


def get_image(xpapoint=None):
    """Get image encircled by a region in DS9.
    """
    import numpy as np

    d = DS9n(xpapoint)
    filename = get_filename(d)
    region = getregion(d, selected=True)
    try:
        x_inf, x_sup, y_inf, y_sup = lims_from_region(region)
    except Exception:
        raise_create_region(d)
        sys.exit()

    area = [y_inf, y_sup, x_inf, x_sup]
    fitsimage = d.get_pyfits()[0]
    if len(fitsimage.shape) == 2:
        image = fitsimage.data[area[0] : area[1], area[2] : area[3]]
    elif len(fitsimage.shape) == 3:
        image = fitsimage.data[
            int(d.get("cube")) - 1, area[0] : area[1], area[2] : area[3]
        ]

    # verboseprint("Region =%s"%( region))
    if hasattr(region[0], "r"):
        # verboseprint("Region = Circle Radius = ", region[0].r)
        image = np.array(image, dtype=float)
        y, x = np.indices((image.shape))
        lx, ly = image.shape
        r = np.sqrt((x - lx / 2) ** 2 + (y - ly / 2) ** 2)
        image[r > int(region[0].r)] = np.nan
        # verboseprint(image)
    header = fitsimage.header
    return image, header, area, filename, [x_inf, y_inf]


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """
    import numpy as np

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.nanmean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.nanmean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    # ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    return


def create_mesh(array, value=None):
    """
    """
    from pyvista import StructuredGrid, PolyData
    import numpy as np

    xx, yy = np.indices(array.shape)
    if value is None:
        value = array.shape[0] / (array.max() - array.min())
    data_mesh = StructuredGrid()
    data_mesh.points = PolyData(
        np.c_[
            xx.reshape(-1),
            yy.reshape(-1),
            value * (array - np.nanmin(array)).reshape(-1),
        ]
    ).points
    data_mesh["Intensity"] = (array - np.nanmin(array)).reshape(-1)
    data_mesh.dimensions = [array.shape[1], array.shape[0], 1]
    return data_mesh


def plot_area_3d_color(d):
    """
    """
    from pyvista import Plotter, set_plot_theme
    from matplotlib import cm
    import numpy as np

    size = [2000, 700]
    color = cm.get_cmap("Greens_r", 128)
    d.set("rgb channel green")
    data_green = getdata(d)
    d.set("rgb channel red")
    data_red = getdata(d)
    d.set("rgb channel blue")
    data_blue = getdata(d)

    set_plot_theme("document")
    value = data_blue.shape[0] / np.max(
        [data_blue.ptp(), data_green.ptp(), data_red.ptp()]
    )

    mesh_blue = create_mesh(data_blue, value=value)
    mesh_green = create_mesh(data_green, value=value)
    mesh_red = create_mesh(data_red, value=value)

    p = Plotter(
        notebook=False,
        window_size=size,
        line_smoothing=True,
        point_smoothing=True,
        polygon_smoothing=True,
        splitting_position=None,
        title="3D",
        shape=(1, 4),
    )
    p.add_mesh(
        create_mesh(data_blue, value=value),
        opacity=0.9,
        nan_opacity=0,
        use_transparency=False,
        name="Data",
        flip_scalars=True,
        cmap=cm.get_cmap("Blues_r", 128),
        show_scalar_bar=False,
    )
    p.add_axes()
    p.subplot(0, 1)
    p.add_mesh(
        create_mesh(data_green, value=value),
        opacity=0.9,
        nan_opacity=0,
        use_transparency=False,
        name="Data",
        flip_scalars=True,
        cmap=cm.get_cmap("Greens_r", 128),
        show_scalar_bar=False,
    )
    p.add_axes()
    p.subplot(0, 2)
    p.add_mesh(
        create_mesh(data_red, value=value),
        opacity=0.9,
        nan_opacity=0,
        use_transparency=False,
        name="Data",
        flip_scalars=True,
        cmap=cm.get_cmap("Reds_r", 128),
        show_scalar_bar=False,
    )
    p.link_views()
    p.add_axes()
    p.subplot(0, 3)
    xx, yy = np.indices(data_blue.shape)  # np.meshgrid(x, y)
    p.add_mesh(
        mesh_green,
        opacity=0.9,
        nan_opacity=0,
        use_transparency=False,
        flip_scalars=True,
        cmap=cm.get_cmap("Greens_r", 128),
        show_scalar_bar=False,
    )
    # zb = np.nanmax(create_mesh(data_green, value=value).points[:, 2]) + (
    #     (data_blue - np.nanmin(data_blue[np.isfinite(data_blue)])) * value
    # )
    blue = p.add_mesh(
        mesh_blue,
        scalars=data_blue.ravel(),
        opacity=0.7,
        nan_opacity=0,
        use_transparency=False,
        name="Data2",
        flip_scalars=True,
        scalar_bar_args={"title": "Value"},
        cmap=cm.get_cmap("Blues_r", 128),
        show_scalar_bar=False,
    )
    # zr = -np.nanmax(mesh_green.points[:, 2]) + (
    #     (data_red - np.nanmin(data_red[np.isfinite(data_red)])) * value
    # )
    red = p.add_mesh(
        mesh_red,
        scalars=data_red.ravel(),
        opacity=0.7,
        nan_opacity=0,
        use_transparency=False,
        name="Data3",
        flip_scalars=True,
        scalar_bar_args={"title": "Value"},
        cmap=cm.get_cmap("Reds_r", 128),
        show_scalar_bar=False,
    )
    return blue, red


def plot_3d(xpapoint=None, color=False, argv=[]):
    """Plots the DS9 region selected in 3D [DS9 required]
    """
    from pyvista import Plotter, StructuredGrid, PolyData, set_plot_theme

    # from pyvistaqt import BackgroundPlotter
    import numpy as np
    from astropy.convolution import convolve, Gaussian2DKernel
    from astropy.io import fits

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="Path of the image to display in 3D",
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)

    d = DS9n(args.xpapoint)
    filename = globglob(args.path, xpapoint=args.xpapoint)
    if (
        (".tif" in filename[0])
        | (".tiff" in filename[0])
        | (".png" in filename[0])
        | (".jpeg" in filename[0])
        | (".jpg" in filename[0])
    ):
        color = True
    if color:
        size = [int(0.8 * 1024), int(2.5 * 768)]
        d.set("rgb channel green")
    else:
        size = (2 * 1024, 2 * 768)
    if isinstance(d, FakeDS9):
        sys.exit()  # data = fits.open(filename[0])[0].data.T
    else:
        data = getdata(args.xpapoint, selected=True)  # problem in test
    verboseprint(data)
    if type(data) != list:
        if (len(data.shape) == 2) & (color):
            plot_area_3d_color(d)
        elif (len(data.shape) == 2) & (~color):
            xx, yy = np.indices(data.shape)  # np.meshgrid(x, y)
            set_plot_theme("document")
            p = Plotter(
                notebook=False,
                window_size=size,
                line_smoothing=True,
                point_smoothing=True,
                polygon_smoothing=True,
                splitting_position=None,
                title="3D",
            )
            # if d.get("scale") == "log":
            #     data = data  # np.log10(data - np.nanmin(data))
            value = data.shape[0] / (np.nanmax(data) - np.nanmin(data))
            # np.min([1,data.shape[0]/(np.nanmax(data) - np.nanmin(data)) ])
            # value_= data.shape[0]#value
            mesh = StructuredGrid()
            data_points = ((data - np.nanmin(data[np.isfinite(data)]))) / (
                np.nanmax(data) - np.nanmin(data)
            )
            log_data_points = (
                np.log10(data_points - np.nanmin(data_points) + 1)
            ) / np.nanmax(
                np.log10(data_points - np.nanmin(data_points) + 1)
            )  # -1
            data_points_c = convolve(data_points, Gaussian2DKernel(x_stddev=1))
            log_data_points_c = convolve(
                log_data_points, Gaussian2DKernel(x_stddev=1)
            )
            points = PolyData(
                np.c_[
                    xx.reshape(-1), yy.reshape(-1), 1 * data_points.reshape(-1)
                ]
            ).points
            mesh.points = points
            mesh["test"] = data.reshape(-1)
            mesh.dimensions = [data.shape[1], data.shape[0], 1]
            mesh = create_mesh(data_points, value=None)
            mesh_c = create_mesh(data_points_c, value=None)

            range_ = [np.nanpercentile(data, 0), np.nanpercentile(data, 100)]
            range_ = [np.nanmin(data), np.nanmax(data)]
            range_ = [np.nanpercentile(data, 30), np.nanpercentile(data, 99)]
            scalars = data.flatten()
            p.add_mesh(
                mesh,
                rng=range_,
                scalars=scalars,
                opacity=0.7,
                nan_opacity=0,
                use_transparency=False,
                name="Data",
                flip_scalars=True,
                scalar_bar_args={"title": "Value"},
            )
            contours = mesh.contour()
            contours_c = mesh_c.contour()
            p.add_mesh(contours, color="white", line_width=5)
            p.add_mesh(contours_c, color="white", line_width=5)
            p.update_coordinates(np.nan * mesh.contour().points, mesh=contours_c)
            d = {
                "log": False,
                "value": value,
                "Contour": True,
                "smooth": False,
                "mesh": mesh,
                "data": (data - np.nanmin(data[np.isfinite(data)]))
                / (data - np.nanmin(data[np.isfinite(data)])).ptp(),
            }
            d["data_points"] = data_points

            def callback(value):
                points = mesh.points
                points[:, -1] = d["data_points"].reshape(-1) * value
                # if d["log"] is False:
                #     points[:, -1] = d["data_points"].reshape(-1) * value
                # else:  #
                #     points[:, -1] = d["data_points"].reshape(-1) * value
                d["value"] = value
                d["points"] = points
                change_contour()
                return

            def log_callback(value):
                if value is True:
                    if d["smooth"]:
                        data = log_data_points_c  # * d['value']
                    else:
                        data = log_data_points  # * d['value']
                else:
                    if d["smooth"]:
                        data = data_points_c  # * d['value']
                    else:
                        data = data_points  # * d['value']
                d["data_points"] = data
                points[:, -1] = d["value"] * data.reshape(-1)
                d["points"] = points
                d["log"] = value
                change_contour()

                return

            def contour_callback(value):
                d["Contour"] = value
                change_contour()
                return

            def gif_callback(value):
                path = p.generate_orbital_path(n_points=36, shift=mesh.length)
                p.open_gif("/tmp/orbit.gif")
                p.orbit_on_path(path, write_frames=True)
                return

            def smooth_callback(value):
                if value is True:
                    if d["log"]:
                        data = log_data_points_c
                    else:
                        data = data_points_c
                else:
                    if d["log"]:
                        data = log_data_points
                    else:
                        data = data_points
                d["data_points"] = data
                points[:, -1] = d["value"] * data.reshape(-1)
                d["points"] = points
                d["smooth"] = value
                change_contour()
                return

            def change_contour():
                p.update_coordinates(d["points"], mesh=mesh)
                p.update_coordinates(d["points"], mesh=mesh_c)
                # p.update_scalars(d["points"], mesh=mesh)
                # p.update_scalars(d["points"], mesh=mesh_c)
                if d["Contour"]:
                    if d["smooth"] is True:
                        p.update_coordinates(
                            mesh_c.contour().points, mesh=contours_c
                        )
                        p.update_coordinates(
                            np.nan * mesh.contour().points, mesh=contours
                        )
                    else:
                        p.update_coordinates(
                            mesh.contour().points, mesh=contours
                        )
                        p.update_coordinates(
                            np.nan * mesh_c.contour().points, mesh=contours_c
                        )
                else:
                    p.update_coordinates(
                        np.nan * mesh.contour().points, mesh=contours
                    )
                    p.update_coordinates(
                        np.nan * mesh_c.contour().points, mesh=contours_c
                    )
                return

            p.add_slider_widget(
                callback,
                rng=[0, 10 * np.max([1, data.shape[0]])],
                value=data.shape[0],
                title="Stretching",
                color=None,
                pass_widget=False,
                event_type="always",
                style=None,
            )
            p.add_checkbox_button_widget(log_callback)
            p.add_checkbox_button_widget(
                contour_callback, position=(10, 80), value=True
            )
            p.add_checkbox_button_widget(gif_callback, position=(10, 70 + 80))
            p.add_text("Log scale", name="log", position=(70, 10))
            p.add_text("Contour", name="contour", position=(70, 80))
            p.add_text("Create a GIF", name="gif", position=(70, 70 + 80))
            p.add_checkbox_button_widget(
                smooth_callback, position=(10, 80 + 70 + 70), value=False
            )
            p.add_text(
                "Smooth", name="buttonSmooth", position=(70, 80 + 70 + 70)
            )
            p.clear_box_widgets()
            p.add_axes()  # interactive=True)
            p.show()
            # if isinstance(p, Plotter):
            #     p.show()
            # else:
            #     p.app.exec_()

        else:
            create_cube(d, data)
    else:
        datas = data
        set_plot_theme("document")
        value = datas[0].shape[0] / np.max(np.ptp(datas, axis=(1, 2)))
        cols = int(np.round(np.sqrt(len(datas))))
        rows = cols
        while rows * cols < len(datas):
            rows += 1
        cols, rows = rows, cols  # cols = 1; rows = len(datas)
        p = Plotter(
            notebook=False,
            window_size=[2000, 1500],
            line_smoothing=True,
            point_smoothing=True,
            polygon_smoothing=True,
            splitting_position=None,
            title="3D",
            shape=(rows, cols),
        )
        for i, data in enumerate(datas):
            p.subplot(int(i / cols), i % cols)  # data.flatten()
            p.add_mesh(
                create_mesh(data, value=value),
                scalars=None,
                opacity=0.9,
                nan_opacity=0,
                use_transparency=False,
                name="Data",
                flip_scalars=True,
                scalar_bar_args={"title": "Value"},
                show_scalar_bar=False,
            )
            p.link_views()
            p.add_axes()
        verboseprint(p)
        p.show()


def create_cube(d, data):
    """Replaces a 3D image  by a 3D cube for pyvista to plot
    """
    from pyvista import Plotter, set_plot_theme, wrap
    import numpy as np

    set_plot_theme("document")

    lx, ly, lz = data.shape
    if d.get("scale") == "log":
        data = np.log10(
            np.array(
                data[:, :, :] - np.nanmin(data[np.isfinite(data)]) + 1,
                dtype=float,
            )
        )
    else:
        data = np.array(data[:, :, :], dtype=float)
    mask = np.ones(len(data.ravel()), dtype=bool)  #
    xx, yy, zz = np.indices(data.shape)  # np.me
    starting_mesh = wrap(
        np.array([yy.ravel()[mask], zz.ravel()[mask], xx.ravel()[mask]]).T
    )
    verboseprint(data.ravel()[mask])
    starting_mesh["Intensity"] = data.ravel()[mask]

    def create_mesh(
        DensityMin=0.5, DensityMax=0.5, StretchingFactor=0.5, PointSize=5
    ):
        mask = (
            data.ravel() > np.nanpercentile(data[np.isfinite(data)], DensityMin)
        ) & (
            data.ravel() < np.nanpercentile(data[np.isfinite(data)], DensityMax)
        )
        mesh = wrap(
            np.array(
                [
                    yy.ravel()[mask],
                    zz.ravel()[mask],
                    StretchingFactor * xx.ravel()[mask]
                    - np.nanmean(StretchingFactor * xx.ravel()[mask]),
                ]
            ).T
        )
        mesh["Intensity"] = data.ravel()[mask]
        return mesh

    p = Plotter(notebook=False, window_size=[2 * 1024, 2 * 768], title="3D")

    class Change3dMesh:
        def __init__(self, mesh):
            self.output = mesh  # Expected PyVista mesh type
            self.kwargs = {
                "DensityMin": 0.5,
                "DensityMax": 0.5,
                "StretchingFactor": 0.5,
                "PointSize": 5,
            }

        def __call__(self, param, value):
            self.kwargs[param] = value
            self.update()

        def update(self):
            if (
                (self.kwargs["DensityMin"] == -1)
                & (self.kwargs["DensityMax"] == 100)
                & (self.kwargs["StretchingFactor"] == 0.5)
                & (self.kwargs["PointSize"] == 5)
            ):
                self.output.overwrite(self.output)
                return
            else:
                result = create_mesh(**self.kwargs)
                self.output.overwrite(result)
                p.update_scalar_bar_range(
                    [result["Intensity"].min(), result["Intensity"].max()]
                )
                return

    engine = Change3dMesh(starting_mesh)
    a = p.add_mesh(
        starting_mesh,
        show_edges=True,
        point_size=engine.kwargs["PointSize"],
        nan_opacity=0,
        cmap="jet",
        name="Data",
    )
    mmax = 0.92
    p.add_slider_widget(
        callback=lambda value: engine("DensityMin", int(value)),
        rng=[0.0, 99.9],
        value=90,
        title="",
        pointa=(0.025, mmax),
        pointb=(0.31, mmax),  # "Density Threshold Min",
    )
    p.add_slider_widget(
        callback=lambda value: engine("DensityMax", int(value)),
        rng=[0.1, 100],
        value=100,
        title="Density Threshold Min/Max",
        #        pointa=(.35, .9), pointb=(.64, .83),
        pointa=(0.025, 0.83),
        pointb=(0.31, 0.83),
    )
    p.add_slider_widget(
        callback=lambda value: engine("StretchingFactor", value),
        rng=[0, 1],
        value=0.5,
        title="Stretching Factor",
        pointa=(0.67, mmax),
        pointb=(0.98, mmax),
    )
    p.add_slider_widget(
        callback=a.GetProperty().SetOpacity,
        rng=[0, 1],
        value=0.5,
        title="Opacity",
        pointa=(0.35, mmax),
        pointb=(0.64, mmax),
        event_type="always",
    )
    p.add_axes()
    p.show()


def throw_apertures(xpapoint=None, argv=[]):
    """Throws aperture in image in order to compute depth
    """
    from astropy.io import fits
    import numpy as np

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-a",
        "--aperture",
        default="10,10",
        help="Aperture radius in pixels",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-f",
        "--form",
        default="circle",
        help="Aperture form",
        type=str,
        choices=["box", "circle"],
    )
    parser.add_argument(
        "-d",
        "--distribution",
        default="Random",
        help="Apertures position",
        type=str,
        choices=["Random", "Equidistributed"],
    )
    parser.add_argument(
        "-n",
        "--number_apertures",
        default="1000",
        help="Number of apertures to throw in the image",
        type=str,
        metavar="",
    )

    args = parser.parse_args_modif(argv)
    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    fitsimage = fits.open(filename)
    fitsimage = fitsimage[fits_ext(fitsimage)]
    image = fitsimage.data
    n_aper = int(args.number_apertures)
    region = getregion(d, quick=True, selected=True)
    if region is None:
        area = [0, image.shape[1], 0, image.shape[0]]
    else:
        x_inf, x_sup, y_inf, y_sup = lims_from_region(None, coords=region)
        area = [x_inf, x_sup, y_inf, y_sup]
    radius = np.array(args.aperture.split(","), dtype=int)
    if type(radius) == int:
        r1, r2 = radius, radius
    else:
        try:
            r1, r2 = radius
        except TypeError:
            r1 = r2 = radius
    if args.distribution == "Equidistributed":
        areasd = create_areas(image, area=area, radius=radius)
        areas = areasd
    else:
        print(area)
        areas = np.array(
            [
                np.random.randint(area[2], area[3], n_aper),
                np.random.randint(area[2], area[3], n_aper),
                np.random.randint(area[0], area[1], n_aper),
            ]
        ).T

    create_ds9_regions(
        [np.array(areas)[:, 2] + float(r1) / 2],
        [np.array(areas)[:, 0] + float(r2) / 2],
        radius=[r1, r2] * len(areas),
        save=True,
        savename=tmp_region,
        form=[args.form],
        color=["yellow"],
        ID=None,
    )
    d.set("regions %s" % (tmp_region))
    return


def execute_command(
    filename,
    path2remove,
    exp,
    xpapoint=None,
    eval_=False,
    overwrite=False,
    d=FakeDS9(),
):
    """Combine two images and an evaluable expression
    """
    from scipy.ndimage import (
        grey_dilation,
        grey_erosion,
        gaussian_filter,
        median_filter,
        sobel,
        binary_propagation,
        binary_opening,
        binary_closing,
        label,
    )
    from astropy.io import fits
    from astropy.convolution import convolve
    import numpy as np
    from scipy import fftpack
    from scipy import signal
    from scipy.signal import correlate, correlate2d
    from astropy.convolution import interpolate_replace_nans, Gaussian2DKernel

    try:
        fitsimage = fits.open(filename)
        ext = fits_ext(fitsimage)
        fitsimage = fitsimage[ext]
    except ValueError:
        filename = tmp_image
        fitsimage = fits.open(
            resource_filename("pyds9plugin", "Images/stack18446524.fits")
        )[0]
    ds9 = fitsimage.data
    header = fitsimage.header
    if os.path.isfile(path2remove) is False:
        if "image" in exp:
            d = DS9n()
            message(d, "{Image not found, please verify your path!")
        else:
            image = 0
    else:
        fitsimage2 = fits.open(path2remove)[ext]
        image = fitsimage2.data
    ds9 = np.array(ds9, dtype=float)

    ldict = {
        "ds9": ds9,
        "header": header,
        "image": image,
        "convolve": convolve,
        "grey_dilation": grey_dilation,
        "grey_erosion": grey_erosion,
        "gaussian_filter": gaussian_filter,
        "median_filter": median_filter,
        "sobel": sobel,
        "binary_propagation": binary_propagation,
        "binary_opening": binary_opening,
        "binary_closing": binary_closing,
        "label": label,
        "fftpack": fftpack,
        "np": np,
        "signal": signal,
        "correlate": correlate,
        "correlate2d": correlate2d,
        "interpolate_replace_nans": interpolate_replace_nans,
        "Gaussian2DKernel": Gaussian2DKernel,
        "d": d,
    }
    new_dict = {}
    new_dict.update(ldict)
    new_dict.update(globals())

    try:
        if os.path.isfile(exp):
            verboseprint("Executing file %s" % (exp))
            # exec(open(exp).read(),globals()+ldict)#, globals(), ldict)
            exec(open(exp).read(), new_dict)  # dict_function.update(d)

        else:
            verboseprint("Executing expression : %s" % (exp))
            exec(exp, new_dict)  # globals(), ldict)  # , locals(),locals())
    except (SyntaxError, NameError) as e:
        import traceback

        verboseprint(e)
        verboseprint(traceback.format_exc())

        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        verboseprint(exc_type, fname, exc_tb.tb_lineno)
        d = DS9n(xpapoint)
        if yesno(
            d,
            "Could not execute the command. Do you wish to see examples of one line python commands?",
        ):

            verboseprint(
                """********************************************************************************
                                        Execute Python Command
                    Generic functions -> Image Processing -> Execute Python Command
            ********************************************************************************

            * This command allows you to modify your image by interpreting a python command
            * Enter a basic command in the Expression field such as:
            *    ds9=ds9[:100,:100]                           -> Trim the image
            *    ds9+=np.random.normal(0,0.1,size=ds9.shape)  -> Add noise to the image
            *    ds9[ds9>2]=np.nan                            -> Mask a part of the image
            *    ds9=1/ds9, ds9+=1, ds9+=1                    -> Different basic expressions
            *    ds9=convolve(ds9,np.ones((1,9)))[1:-1,9:-9]  -> Convolve unsymetrically"
            *    ds9+=np.linspace(0,1,ds9.size).reshape(ds9.shape) -> Add background
            *    ds9+=30*(ds9-gaussian_filter(ds9, 1))        -> Sharpening
            *    ds9=np.hypot(sobel(ds9,axis=0,mode='constant'),sobel(ds9,axis=1,mode='constant')) -> Edge Detection
            *    ds9=np.abs(fftshift(fft2(ds9)))**2           -> FFT
            *    ds9=correlate2d(ds9.astype('uint64'),ds9,boundary='symm',mode='same') -> Autocorr
            *    ds9=interpolate_replace_nans(ds9, Gaussian2DKernel(x_stddev=5, y_stddev=5)) -> Interpolate NaNs

            * You also have the possibility to combine another image in your python
            * expression. It cam be very interesting for dark subtraction or convolution
            * for instance:
            *    ds9= ds9-image    (Must be the same size!)   -> Subtract the image
            *    ds9=convolve(ds9,image)  (image must be odd) -> Convolve with image
            * To do this you must copy the path of the image in the image field.

            * The last field is to apply your python expression to several images!
            * To do so, you can write their a regular expression matching the files on
            * which you want to sun the command! """,
                verbose="1",
            )
        sys.exit()
    ds9 = new_dict["ds9"]
    same = fitsimage.data == ds9
    if type(same) is not bool:
        same = same.all()
    if same & (fitsimage.header == header):
        verboseprint("ds9 did not change")
        return None, filename
    else:
        fitsimage.data = ds9
        fitsimage.header = header
        if (ds9.astype(int) == ds9).all():
            fitsimage.data = np.int16(fitsimage.data)
            fitsimage.header["BITPIX"] = 16
        if overwrite is False:
            name = filename[:-5] + "_modified.fits"
        else:
            name = filename
        fitsimage.header["DS9"] = filename
        fitsimage.header["IMAGE"] = path2remove
        try:
            fitsimage.header["COMMAND"] = exp
        except ValueError as e:
            verboseprint(e)
            verboseprint(len(exp))
            fitsimage.header.remove("COMMAND")
        try:
            fitsimage.writeto(name, overwrite=True)
        except RuntimeError:
            fitswrite(ds9, name)
        return fitsimage.data, name


def fitswrite(fitsimage, filename, verbose=True, header=None):
    """Write fits image function with different tests
    """
    from astropy.io import fits
    import numpy as np

    if type(fitsimage) == np.ndarray:
        fitsimage = fits.HDUList([fits.PrimaryHDU(fitsimage, header=header)])[0]
    if len(filename) == 0:
        verboseprint(
            "Impossible to save image in filename %s, saving it to %s"
            % (filename, tmp_image)
        )
        filename = tmp_image
        fitsimage.header["NAXIS3"], fitsimage.header["NAXIS1"] = (
            fitsimage.header["NAXIS1"],
            fitsimage.header["NAXIS3"],
        )
        fitsimage.writeto(filename, overwrite=True)
    if hasattr(fitsimage, "header"):
        if "NAXIS3" in fitsimage.header:
            verboseprint("2D array: Removing NAXIS3 from header...")
            fitsimage.header.remove("NAXIS3")
        if "SKEW" in fitsimage.header:
            fitsimage.header.remove("SKEW")
    elif hasattr(fitsimage[0], "header"):
        if "NAXIS3" in fitsimage[0].header:
            verboseprint("2D array: Removing NAXIS3 from header...")
            fitsimage[0].header.remove("NAXIS3")
        if "SKEW" in fitsimage[0].header:
            fitsimage[0].header.remove("SKEW")

    elif not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    if not os.path.exists(os.path.dirname(filename)):
        verboseprint(
            "%s not existing: creating folde..." % (os.path.dirname(filename))
        )
        os.makedirs(os.path.dirname(filename))
    try:
        fitsimage.writeto(filename, overwrite=True)
    except IOError:
        verboseprint("Can not write in this repository : " + filename)
        filename = "/tmp/" + os.path.basename(filename)
        verboseprint("Instead writing new file in : " + filename)
        fitsimage.writeto(filename, overwrite=True)
    verboseprint("Image saved: %s" % (filename))
    return filename


def csvwrite(table, filename, verbose=True):
    """Write a catalog in csv format
    """
    import importlib
    from astropy.io import ascii
    from astropy.table import Table
    import numpy as np

    if type(table) == np.ndarray:
        table = Table(table)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    try:
        table.write(
            filename,
            overwrite=True,
            format="csv",
            fill_values=[(ascii.masked, "N/A")],
        )
    except UnicodeDecodeError as e:
        verboseprint(e)
        importlib.reload(sys)
        sys.setdefaultencoding("utf8")
    try:
        table.write(
            filename,
            overwrite=True,
            format="csv",
            fill_values=[(ascii.masked, "N/A")],
        )
    except IOError:
        verboseprint("Can not write in this repository : " + filename)
        filename = "/tmp/" + os.path.basename(filename)
        verboseprint("Instead writing new file in : " + filename)
        table.write(
            filename,
            overwrite=True,
            format="csv",
            fill_values=[(ascii.masked, "N/A")],
        )
    verboseprint("Table saved: %s" % (filename))
    return table


def Gaussian(x, amplitude, xo, sigma2, offset):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = offset + amplitude * np.exp(-0.5 * (np.square(x - xo) / sigma2))
    return g.ravel()


def check_file(xpapoint=None):
    """Check the properties of the DS9 file
    """
    from astropy.io import fits
    from astropy.table import Table

    d = DS9n(xpapoint)
    path = get_filename(d)
    fitsim = fits.open(path)

    for i, fitsi in enumerate(fitsim):
        print(
            "\n\n********************** ",
            i,
            " **********************\nImage = ",
            fitsi.is_image,
        )
        try:
            print("Shape = ", fitsi.data.shape)
            print("Size = ", fitsi.size)
        except AttributeError:
            print("Size = ", fitsi.size)
        if not fitsi.is_image:
            print(Table(fitsi.data))


class bcolors:
    """Color class to use in print
    """

    BLACK_RED = "\x1b[4;30;41m"
    GREEN_WHITE = "\x1b[0;32;47m"
    BLACK_GREEN = "\x1b[0;30;42m"
    END = "\x1b[0m"
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[32m"
    ORANGE = "\033[33m"
    DARKGREY = "\033[90m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def globglob(file, xpapoint=None, sort=True, ds9_im=False):
    """Improved glob.glob routine where we can use regular
    expression with this: /tmp/image[5-15].fits
    """
    import numpy as np

    file = file.rstrip()[::-1].rstrip()[::-1]
    try:
        paths = glob.glob(r"%s" % (file), recursive=True)
    except Exception as e:
        verboseprint(e)
        paths = []
    # verboseprint(file, paths)
    if (len(paths) == 0) & ("[" in file) and ("]" in file):
        verboseprint("Go in loop")
        a, between = file.split("[")
        between, b = between.split("]")
        if ("-" in between) & (len(between) > 3):
            paths = []
            n1, n2 = np.array(between.split("-"), dtype=int)
            range_ = np.arange(n1, n2 + 1)
            verboseprint(range_)
            files = [a + "%0.{}d".format(len(str(n2))) % (i) + b for i in range_]
            files += [a + "%i" % (i) + b for i in range_]
            for path in np.unique(files):
                if os.path.isfile(path):
                    paths.append(path)
        if ("," in between) & (len(between) > 3):
            paths = []
            ns = np.array(between.split(","), dtype=int)
            ns.sort()
            verboseprint(ns)
            files = [a + "%0.{}d".format(len(str(ns[-1]))) % (i) + b for i in ns]
            files += [a + "%i" % (i) + b for i in ns]
            for path in np.unique(files):
                if os.path.isfile(path):
                    paths.append(path)

    if sort:
        paths.sort()
    if (len(paths) == 0) & (file == ""):  # changed --""
        verboseprint("No file specified, using loaded one.")
        d = DS9n(xpapoint)
        paths = [get_filename(d)]

    elif (len(paths) == 0) & (file != ""):
        d = DS9n(xpapoint)
        if (
            (len(paths) == 0)
            & (file != "")
            & (file == d.get("file"))
            & (file != get_filename(d))
        ):
            verboseprint(
                "Loaded image not on drive, saving it to run the analysis"
            )
            paths = [get_filename(d)]
        elif ds9_im:
            message(
                d,
                """No file is matching the pathname pattern. Please
                          verify your entry. Running the analysis on the
                          DS9 loaded image.""",
            )
            paths = [get_filename(d)]
        else:
            verboseprint("No image to work on...")
            paths = []
    return paths


def stack_images(xpapoint=None, std=False, clipping=None, argv=[]):
    """Stack same size images
    """
    import numpy as np

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-t",
        "--type",
        default="median",
        help="Type of the merge",
        type=str,
        choices=["mean", "median", "std"],
    )
    parser.add_argument(
        "-c", "--clip", default="100", help="Clip the images", type=str,
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    paths = globglob(args.path)

    dtype = "float"
    if "std" in args.type:
        std = True
    if "median" in args.type:
        Type = np.nanmedian
    else:
        Type = np.nanmean
    image, name = stack_images_path(
        paths,
        Type=Type,
        clipping=float(args.clip),
        dtype=dtype,
        std=std,
        name=os.path.dirname(paths[0]) + "stack_image.fits",
    )
    d.set("tile yes ; frame new ; file {}".format(name))
    try:
        d.set("lock frame physical")
    except ValueError:
        pass
    return


def stack_images_path(
    paths,
    Type="np.nanmean",
    clipping=3,
    dtype=float,
    fname="",
    std=False,
    name=None,
):
    """Stack images of the files given in the path
    """
    from astropy.io import fits
    import re
    import numpy as np

    fitsfile = fits.open(paths[0])

    i = fits_ext(fitsfile)
    x_inf, x_sup, y_inf, y_sup = 0, -1, 0, -1  # my_conf.physical_region
    stds = np.array(
        [
            np.nanstd(fits.open(image)[i].data[x_inf:x_sup, y_inf:y_sup])
            for image in paths
        ]
    )
    index = stds < np.nanmean(stds) + clipping * np.nanstd(stds)
    paths = np.array(paths)
    if std is False:
        stds = np.ones(len(paths[index]))
    else:
        stds /= np.nansum(stds[index])
    n = len(paths)
    paths.sort()
    lx, ly = fitsfile[i].data.shape
    stack = np.zeros((lx, ly), dtype=dtype)
    if std:
        verboseprint("Using std method")
        for i, file in enumerate(paths[index]):
            try:
                with fits.open(file) as f:
                    # f[0].data[~np.isfinite(f[0].data)] = stack[~np.isfinite(f[0].data)]
                    stack[:, :] += f[i].data / stds[i]
                    del f
            except TypeError as e:
                verboseprint(e)
                n -= 1
        stack = stack / n
    elif Type == np.nanmedian:
        stack = np.array(
            Type(
                np.array([fits.open(file)[i].data for file in paths[index]]),
                axis=0,
            ),
            dtype=dtype,
        )
    else:
        stack = Type(
            np.array([fits.open(file)[i].data for file in paths[index]]),
            dtype=dtype,
            axis=0,
        )
    try:
        numbers = [
            int(re.findall(r"\d+", os.path.basename(filename))[-1])
            for filename in paths[index]
        ]
    except IndexError:
        numbers = paths
    images = " - ".join(list(np.array(numbers, dtype=str)))
    new_fitsfile = fitsfile[i]
    new_fitsfile.data = stack
    new_fitsfile.header["STK_NB"] = images
    if name is None:
        try:
            name = "{}/StackedImage_{}-{}{}.fits".format(
                os.path.dirname(paths[0]),
                int(os.path.basename(paths[0])[5 : 5 + 6]),
                int(os.path.basename(paths[-1])[5 : 5 + 6]),
                fname,
            )
        except ValueError:
            name = "{}/StackedImage_{}-{}{}".format(
                os.path.dirname(paths[0]),
                os.path.basename(paths[0]).split(".")[0],
                os.path.basename(paths[-1]),
                fname,
            )
    else:
        name = os.path.join(os.path.dirname(paths[0]), name)
    verboseprint("Image saved : %s" % (name))
    try:
        fitswrite(new_fitsfile, name)
    except RuntimeError as e:
        verboseprint("Unknown error to be fixed: ", e)
        fitswrite(new_fitsfile.data, name)
    return fitsfile, name


def light_curve(xpapoint=None, DS9backUp=DS9_BackUp_path, argv=[]):
    """Perform a light_curve analysis and return the centered image
    [DS9 required]
    """
    from astropy.io import fits
    from astropy.table import Table
    from scipy.optimize import curve_fit
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="Paths of the images you want to analyse. Use regexp ",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    if args.path == "":
        path = get_filename(d, All=True, sort=False)
    else:
        path = globglob(args.path, xpapoint=args.xpapoint)
    path.sort()
    x = np.arange(len(path))
    a = getregion(d)[0]
    radius = 15
    fluxes = []
    ext = fits_ext(fits.open(path[0]))

    for file in path:
        fitsfile = fits.open(file)[ext]
        image = fitsfile.data
        subimage = image[
            int(a.yc) - radius : int(a.yc) + radius,
            int(a.xc) - radius : int(a.xc) + radius,
        ]
        background = estimate_background(image, [a.xc, a.yc], radius=30, n=1.8)
        flux = np.nansum(
            subimage - background
        )  # - estimate_background(image, center, radius, n_bg)
        fluxes.append(flux)
    fluxesn = (fluxes - min(fluxes)) / max(fluxes - min(fluxes))
    x = np.arange(len(path)) + 1
    popt, pcov = curve_fit(Gaussian, x, fluxesn, p0=[1, x.mean(), 3, 0])
    xl = np.linspace(x.min(), x.max(), 100)
    maxf = xl[
        np.where(Gaussian(xl, *popt) == np.nanmax(Gaussian(xl, *popt)))[0][0]
    ]  # [0]

    name = DS9backUp + "CSVs/%s_ThroughSlit.csv" % (
        datetime.datetime.now().strftime("%y%m%d-%HH%M")
    )
    csvwrite(np.vstack((x, fluxesn)).T, name)
    csvwrite(np.vstack((xl, Gaussian(xl, *popt))).T, name[:-4] + "_fit.csv")
    plt_ = True
    if plt_:
        np.savetxt("/tmp/throughslit.dat", np.array([x, fluxesn]).T)
        np.savetxt(
            "/tmp/throughslit_fit.dat", np.array([xl, Gaussian(xl, *popt)]).T
        )
        np.savetxt(
            "/tmp/middle.dat",
            np.array(
                [np.linspace(maxf, maxf, len(fluxes)), fluxesn / max(fluxesn)]
            ).T,
        )

        d = []
        d.append("plot line open")  # d.append("plot axis x grid no ")
        d.append("plot axis y grid no ")
        d.append("plot title y 'Flux' ")
        d.append(
            "plot title x 'Image number - Maximum flux at: %0.1f' " % (maxf)
        )
        d.append("plot legend position right ")
        d.append("plot load /tmp/throughslit_fit.dat xy")
        d.append("plot line dash yes ")
        d.append("plot load /tmp/middle.dat xy")
        d.append("plot load /tmp/throughslit.dat xy")
        d.append("plot line shape circle ")
        d.append("plot line width 0 ")
        d.append("plot line shape color black")  # d.append("plot legend yes ")
        d.append("plot font legend size 9 ")
        d.append("plot font labels size 13 ")
        ds9 = DS9n()
        ds9.set(" ; ".join(d))
    else:
        a = Table(np.vstack((x, fluxesn)).T)
        b = Table(np.vstack((xl, Gaussian(xl, *popt))).T)
        a.write(name, format="ascii")
        b.write(name[:-4] + "_fit.csv", format="ascii")
        d = ds9_plot(
            path=name[:-4] + "_fit.csv",
            title="Best_image:{}".format(maxf),
            name="Fit",
            xlabel="# image",
            ylabel="Estimated_flux",
            type_="xy",
            xlim=None,
            ylim=None,
            shape="none",
        )
        ds9_plot(
            d=d,
            path=name,
            title="Best image : {}".format(maxf),
            name="Fit",
            xlabel="# image",
            ylabel="Estimated flux (Sum pixel)",
            type_="xy",
            xlim=None,
            ylim=None,
            shape="circle",
            New=False,
        )
    return


def ds9_plot(
    d=None,
    path="",
    title="",
    name="",
    xlabel="",
    ylabel="",
    type_="xy",
    xlim=None,
    ylim=None,
    New=True,
    shape="None",
):
    """Use DS9 plotting display
    """
    if d is None:
        d = DS9n()
    if New:
        d.set("plot")  # line|bar|scatter
    d.set("plot load %s %s" % (path, type_))
    d.set("plot title {%s}" % (title))
    d.set("plot title x {%s}" % (xlabel))
    d.set("plot title y {%s}" % (ylabel))
    if xlim is not None:
        d.set("plot axis x auto no")
    return d


def guidance(xpapoint=None, plot_=True, reg=True, style="o", lw=0.5, argv=[]):
    """Always display last image of the repository and will upate with new ones
    [DS9 required]
    """
    import matplotlib.pyplot as plt
    import time
    import numpy as np

    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv, required=False)

    d = DS9n(args.xpapoint)
    if reg:
        regions = getregion(d, selected=True)
        create_regions(
            regions, savename=tmp_region, texts=np.arange(len(regions))
        )
        d.set("regions file %s" % (tmp_region))
    ext = fits_ext(d.get_pyfits())
    filename = get_filename(d)
    files = glob.glob(os.path.dirname(filename) + "/*.fits")
    files.sort()
    files = files * 100
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(5, 10))
    colors = ["orange", "green", "red", "pink", "grey", "black"] * 100
    d0 = []
    for (i, region, color) in zip(np.arange(len(regions)), regions, colors):
        data = get_data_from_region(d, region, ext=ext)
        d0.append(center_flux_std(data, bck=0, method="Gaussian-Picouet"))
        dn = d0
        ax0.semilogy(i, d0[i]["flux"], color=color, label=str(i))
        # ax0.scatter(i, np.log10(d0['flux']),color=color,label=str(i))
        ax1.scatter(i, d0[i]["std"], color=color, label=str(i))
        ax2.scatter(i, 0, color=color, label=str(i))
    ax0.legend(loc="upper left")
    ax1.set_xlabel("Number of images")
    ax0.set_ylabel("Flux[region]")
    ax1.set_ylabel("std[region]")
    ax2.set_ylabel("Distance[region]")
    ax0.set_title(os.path.basename(filename))
    fig.tight_layout()
    dn = [{"x": np.nan, "y": np.nan, "flux": np.nan, "std": np.nan}] * len(
        regions
    )
    for i, file in enumerate(files):
        ax0.set_title(os.path.basename(file))
        ax0.set_xlim((np.max([i - 30, 0]), np.max([i + 1, 30])))
        dnm = dn.copy()
        dn = []
        for j, (region, color) in enumerate(zip(regions, colors)):
            data = get_data_from_region(d, region, ext=ext)
            dn.append(center_flux_std(data, bck=0, method="Gaussian-Picouet"))
            verboseprint([i - 1, i], [dnm[j]["flux"], dn[j]["flux"]])
            ax0.plot(
                [i - 1, i],
                [dnm[j]["flux"], dn[j]["flux"]],
                style,
                color=color,
                linewidth=lw,
            )
            ax1.plot(
                [i - 1, i],
                [dnm[j]["std"], dn[j]["std"]],
                style,
                color=color,
                linewidth=lw,
            )
            ax2.plot(
                [i - 1, i],
                [
                    distance(dnm[j]["x"], dnm[j]["y"], d0[j]["x"], d0[j]["y"]),
                    distance(dn[j]["x"], dn[j]["y"], d0[j]["x"], d0[j]["y"]),
                ],
                style,
                color=color,
                linewidth=lw,
            )

            x_inf, x_sup, y_inf, y_sup = lims_from_region(regions[j])
            regions[j] = regions[j]._replace(xc=x_inf + dn[j]["x"])
            regions[j] = regions[j]._replace(yc=y_inf + dn[j]["y"])
        create_regions(
            regions, savename=tmp_region, texts=np.arange(len(regions))
        )
        d.set("file " + file)
        d.set("regions file %s" % (tmp_region))
        plt.pause(0.00001)
        time.sleep(0.01)
    plt.show()
    return


def center_flux_std(image, bck=0, method="Gaussian-Picouet"):
    """Fit a gaussian and give, center flux and std
    """
    from photutils import centroid_com, centroid_1dg, centroid_2dg
    from scipy.optimize import curve_fit
    import numpy as np

    lx, ly = image.shape
    if method == "Center-of-mass":
        xn, yn = centroid_com(image)
    if method == "2x1D-Gaussian-fitting":
        xn, yn = centroid_1dg(image)
    if method == "2D-Gaussian-fitting":
        xn, yn = centroid_2dg(image)
    if method == "Maximum":
        yn, xn = (
            np.where(image == np.nanmax(image))[0][0],
            np.where(image == np.nanmax(image))[1][0],
        )
    elif method == "Gaussian-Picouet":
        x = np.linspace(0, lx - 1, lx)
        y = np.linspace(0, ly - 1, ly)
        x, y = np.meshgrid(x, y)
        yo, xo = np.where(
            image == image.max()
        )  # ndimage.measurements.center_of_mass(image)
        bounds = (
            [1e-1 * np.nanmax(image), xo - 10, yo - 10, 0.5, 0.5, -1e5],
            [10 * np.nanmax(image), xo + 10, yo + 10, 10, 10, 1e5],
        )  # (-np.inf, np.inf)#
        param = (
            np.nanmax(image),
            int(xo),
            int(yo),
            2,
            2,
            0,
            np.percentile(image, 15),
        )
        try:
            popt, pcov = curve_fit(
                gaussian_2dim, (x, y), image.flat, param, bounds=bounds
            )
        except RuntimeError as e:
            verboseprint(e)
            return np.nan, np.nan
        xn, yn = popt[1], popt[2]
        d = {
            "x": xn,
            "y": yn,
            "flux": 2 * np.pi * np.square(popt[1]) * np.square(popt[0]),
            "std": np.sqrt(popt[-2] ** 2 + popt[-3] ** 2),
        }
    return d


def lims_from_region(region=None, coords=None, dtype=int):
    """Return the pixel locations limits of a DS9 region
    """
    import numpy as np

    if coords is not None:
        if len(coords) == 1:
            if len(coords[0]) > 3:
                xc, yc, w, h = coords[0][:4]
            else:
                xc, yc, w = coords[0][:3]
                h = w  # = 2 * coords[0][-1], 2 * coords[0][-1]
        else:
            if len(coords) > 3:
                xc, yc, w, h = coords[:4]
            else:
                xc, yc, w = coords[:3]
                h, w = 2 * coords[-1], 2 * coords[-1]
    else:
        if hasattr(region, "xc"):
            if hasattr(region, "h"):
                xc, yc, h, w = (
                    float(region.xc),
                    float(region.yc),
                    float(region.h),
                    float(region.w),
                )
            if hasattr(region, "r"):
                xc, yc, h, w = (
                    float(region.xc),
                    float(region.yc),
                    float(2 * region.r),
                    float(2 * region.r),
                )

        else:
            region = region[0]
            if hasattr(region, "h"):
                xc, yc, h, w = (
                    float(region.xc),
                    float(region.yc),
                    float(region.h),
                    float(region.w),
                )
            if hasattr(region, "r"):
                xc, yc, h, w = (
                    float(region.xc),
                    float(region.yc),
                    float(2 * region.r),
                    float(2 * region.r),
                )
    #########To use with extreme precaution!!!
    # xc, yc = give_value(xc), give_value(yc)
    ##############
    h, w = abs(h), abs(w)
    verboseprint("W = %s" % (w))
    verboseprint("H = %s" % (h))
    if w <= 2:
        w = 2
    if h <= 2:
        h = 2
    y_inf = int(np.floor(yc - h / 2 - 1))
    y_sup = int(np.ceil(yc + h / 2 + 1))
    x_inf = int(np.floor(xc - w / 2 - 1))
    x_sup = int(np.ceil(xc + w / 2 + 1))
    # try:
    #     # verboseprint("Xc, Yc =  ", region.xc, region.yc)
    #     # verboseprint("Xc, Yc = ", xc, yc)
    # verboseprint("x_inf, x_sup, y_inf, y_sup = ", x_inf, x_sup, y_inf, y_sup)
    #     # verboseprint("data[%i:%i,%i:%i]" % (y_inf, y_sup, x_inf, x_sup))
    # except AttributeError as e:
    #     verboseprint(e)
    #     pass
    if dtype == float:
        return (
            np.max([0, xc - w / 2]),
            xc + w / 2,
            np.max([0, yc - h / 2]),
            yc + h / 2,
        )
    else:
        return (
            give_value(np.max([1, xc - w / 2])),
            give_value(xc + w / 2),
            give_value(np.max([1, yc - h / 2])),
            give_value(yc + h / 2),
        )


def give_value(x):
    """Accoutn for the python/DS9 different way of accounting to 0 pixel
    """
    x = int(x) if x % 1 > 0.5 else int(x) - 1
    return x


def convolve_box_psf(x, amp=1, l=40, x0=0, sigma2=40, offset=0):
    """Convolution of a box with a gaussian
    """
    from scipy import special
    import numpy as np

    a = special.erf((l - (x - x0)) / np.sqrt(2 * sigma2))
    b = special.erf((l + (x - x0)) / np.sqrt(2 * sigma2))
    function = amp * (a + b) / 4 * l
    return offset + function


def center_region(xpapoint=None, plot_=True, argv=[]):
    """Centers DS9 region on spot [DS9 required]"""
    import numpy as np
    from photutils import centroid_com, centroid_1dg, centroid_2dg
    from scipy.optimize import curve_fit

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-m",
        "--method",
        default="Maximum",
        help="",
        type=str,
        choices=[
            "Maximum",
            "Center-of-mass",
            "2x1D-Gaussian-fitting",
            "2D-Gaussian-fitting",
        ],
    )
    parser.add_argument(
        "-b",
        "--background_removal",
        default="0",
        help="",
        type=str,
        choices=["1", "0"],
    )
    args = parser.parse_args_modif(argv, required=False)
    method, bck = args.method, args.background_removal

    d = DS9n(args.xpapoint)
    regions = getregion(d, selected=True, message=True)  # [0]
    d.set("regions delete select")
    if regions is None:
        d = DS9n(args.xpapoint)
        raise_create_region(d)
        sys.exit()

    for region in regions:
        if hasattr(region, "h"):
            xc, yc, h, w = (
                int(region.xc),
                int(region.yc),
                int(region.h),
                int(region.w),
            )
            if w <= 2:
                w = 2
            if h <= 2:
                h = 2
            x_inf = int(np.floor(yc - h / 2 - 1))
            x_sup = int(np.ceil(yc + h / 2 - 1))
            y_inf = int(np.floor(xc - w / 2 - 1))
            y_sup = int(np.ceil(xc + w / 2 - 1))
            data = d.get_pyfits()[0].data
            imagex = data[x_inf - 15 : x_sup + 15, y_inf:y_sup].sum(axis=1)
            imagey = data[x_inf:x_sup, y_inf - 15 : y_sup + 15].sum(axis=0)
            model = convolve_box_psf
            x = np.arange(-len(imagex) / 2, len(imagex) / 2)
            y = np.arange(-len(imagey) / 2, len(imagey) / 2)
            try:
                poptx, pcovx = curve_fit(
                    model,
                    x,
                    imagex,
                    p0=[imagex.max(), 20, 0.0, 10.0, np.median(imagex)],
                )  # ,  bounds=bounds)
                popty, pcovy = curve_fit(
                    model,
                    y,
                    imagey,
                    p0=[imagey.max(), 10, 0.0, 10.0, np.median(imagey)],
                )  # ,  bounds=bounds)
                ampx, lx, x0x, sigma2x, offsetx = poptx
                ampy, ly, x0y, sigma2y, offsety = popty
            except RuntimeError as e:
                verboseprint(e)
            else:
                verboseprint("Poptx = %s" % (poptx))
                verboseprint("Popty = %s" % (popty))
            newCenterx = xc + x0y  # popty[2]
            newCentery = yc + x0x  # poptx[2]
            verboseprint(
                """Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f]"""
                % (region.yc, region.xc, newCentery, newCenterx)
            )
            try:
                os.remove(tmp_region)
            except OSError:
                pass
            create_ds9_regions(
                [newCenterx - 1],
                [newCentery - 1],
                radius=[region.w, region.h],
                save=True,
                savename=tmp_region,
                form=["box"],
                color=["white"],
                ID=[["%0.2f - %0.2f" % (newCenterx, newCentery)]],
            )
            d.set("regions %s" % (tmp_region))
            # import matplotlib
            # matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, sharex=True)
            axes[0].plot(x, imagex, "bo", label="Spatial direction")
            axes[1].plot(y, imagey, "ro", label="Spectral direction")
            axes[0].plot(x, model(x, *poptx), color="b")
            axes[1].plot(y, model(y, *popty), color="r")
            axes[0].set_ylabel("Spatial direction")
            axes[1].set_ylabel("Spectral direction")
            axes[0].plot(
                x,
                Gaussian(x, imagex.max() - offsetx, x0x, sigma2x, offsetx),
                ":b",
                label="Deconvolved PSF",
            )  # Gaussian x, amplitude, xo, sigma_x, offset
            x = np.linspace(
                -np.max([len(imagey), len(imagex)]) / 2,
                np.max([len(imagey), len(imagex)]) / 2,
                2000,
            )
            xcc = x - x0x
            axes[0].plot(
                x,
                np.piecewise(
                    x,
                    [xcc < -lx, (xcc >= -lx) & (xcc <= lx), xcc > lx],
                    [offsetx, imagex.max(), offsetx],
                ),
                ":r",
                label="Slit size",
            )  # slit (UnitBox)
            axes[0].plot([x0x, x0x], [imagex.min(), imagex.max()])
            axes[1].plot(
                y,
                Gaussian(y, imagey.max() - offsety, x0y, sigma2y, offsety),
                ":b",
                label="Deconvolved PSF",
            )  # Gaussian x, amplitude, xo, sigma_x, offset
            xcc = x - x0y
            axes[1].plot(
                x,
                np.piecewise(
                    x,
                    [xcc < -ly, (xcc >= -ly) & (xcc <= ly), xcc > ly],
                    [offsety, imagey.max(), offsety],
                ),
                ":r",
                label="Slit size",
            )
            axes[1].plot([x0y, x0y], [imagey.min(), imagey.max()])
            plt.figtext(
                0.15,
                0.75,
                "Sigma = %0.2f +/- %0.2f pix\nSlitdim = %0.2f +/- %0.2f pix\ncenter = %0.2f +/- %0.2f"
                % (
                    np.sqrt(poptx[3]),
                    np.sqrt(np.diag(pcovx)[3] / 2.0),
                    2 * poptx[1],
                    2 * np.sqrt(np.diag(pcovx)[1]),
                    x0x,
                    np.sqrt(np.diag(pcovx)[2]),
                ),
                bbox={"facecolor": "blue", "alpha": 0.2, "pad": 10},
            )
            plt.figtext(
                0.15,
                0.35,
                "Sigma = %0.2f +/- %0.2f pix\nSlitdim = %0.2f +/- %0.2f pix\ncenter = %0.2f +/- %0.2f"
                % (
                    np.sqrt(popty[3]),
                    np.sqrt(np.diag(pcovy)[3] / 2.0),
                    2 * popty[1],
                    2 * np.sqrt(np.diag(pcovy)[1]),
                    x0y,
                    np.sqrt(np.diag(pcovy)[2]),
                ),
                bbox={"facecolor": "red", "alpha": 0.2, "pad": 10},
            )
            if plot_:
                plt.show()
        if hasattr(region, "r"):
            x_inf, x_sup, y_inf, y_sup = lims_from_region(region)
            data = d.get_pyfits()[0].data
            image = data[y_inf:y_sup, x_inf:x_sup]
            lx, ly = image.shape
            if method == "Center-of-mass":
                xn, yn = centroid_com(image)
            elif method == "2x1D-Gaussian-fitting":
                xn, yn = centroid_1dg(image)
            elif method == "2D-Gaussian-fitting":
                xn, yn = centroid_2dg(image)
            elif method == "Maximum":
                yn, xn = (
                    np.where(image == np.nanmax(image))[0][0],
                    np.where(image == np.nanmax(image))[1][0],
                )
            elif method == "Gaussian-Picouet":
                xc, yc = int(region.xc), int(region.yc)
                x = np.linspace(0, lx - 1, lx)
                y = np.linspace(0, ly - 1, ly)
                x, y = np.meshgrid(x, y)
                yo, xo = np.where(
                    image == image.max()
                )  # ndimage.measurements.center_of_mass(image)
                maxx, maxy = xc - (lx / 2 - xo), yc - (ly / 2 - yo)
                verboseprint("maxx, maxy = {}, {}".format(maxx, maxy))

                bounds = (
                    [1e-1 * np.nanmax(image), xo - 10, yo - 10, 0.5, 0.5, -1e5,],
                    [10 * np.nanmax(image), xo + 10, yo + 10, 10, 10, 1e5],
                )
                param = (
                    np.nanmax(image),
                    int(xo),
                    int(yo),
                    2,
                    2,
                    0,
                    np.percentile(image, 15),
                )
                try:
                    popt, pcov = curve_fit(
                        gaussian_2dim, (x, y), image.flat, param, bounds=bounds,
                    )
                    verboseprint("\nFitted parameters = %s" % (popt))
                except RuntimeError as e:
                    verboseprint(e)
                    sys.exit()
                verboseprint(np.diag(pcov))

                fit = gaussian_2dim((x, y), *popt).reshape((ly, lx))
                xn, yn = popt[1], popt[2]

                if plot_:
                    plt.figure()
                    plt.plot(image[int(yo), :], "bo", label="Spatial dir")
                    plt.plot(fit[int(yo), :], color="b")
                    plt.plot(image[:, int(xo)], "ro", label="Spatial dir")
                    plt.plot(fit[:, int(xo)], color="r")
                    plt.ylabel("Fitted profiles")
                    plt.figtext(
                        0.66,
                        0.55,
                        "Sigma = %0.2f +/- %0.2f pix\nXcenter = %0.2f +/- %0.2f\nYcenter = %0.2f +/- %0.2f"
                        % (
                            np.sqrt(popt[3]),
                            np.sqrt(np.diag(pcov)[3] / 2.0),
                            lx / 2 - popt[1],
                            np.sqrt(np.diag(pcov)[1]),
                            ly / 2 - popt[2],
                            np.sqrt(np.diag(pcov)[2]),
                        ),
                        bbox={"facecolor": "blue", "alpha": 0.2, "pad": 10},
                    )
                    plt.legend()
                    plt.show()
            newCenterx = x_inf + xn + 1  # region.xc - (lx/2 - popt[1])
            newCentery = y_inf + yn + 1  # region.yc - (ly/2 - popt[2])
            verboseprint(
                """Center change : [%0.2f, %0.2f] --> [%0.2f, %0.2f]"""
                % (region.yc, region.xc, newCentery, newCenterx)
            )
            d.set("regions delete select")

            try:
                os.remove(tmp_region)
            except OSError:
                pass

            create_ds9_regions(
                [newCenterx - 1],
                [newCentery - 1],
                radius=[region.r],
                save=True,
                savename=tmp_region,
                form=["circle"],
                color=["white"],
                ID=[["%0.2f - %0.2f" % (newCenterx, newCentery)]],
            )
            create_ds9_regions(
                [newCenterx],
                [newCentery],
                radius=[region.r],
                save=True,
                savename=tmp_region,
                form=["circle"],
                color=["white"],
                ID=[["%0.2f - %0.2f" % (newCenterx, newCentery)]],
            )
            d.set("regions %s" % (tmp_region))
    return newCenterx, newCentery


def t2s(h, m, s, d=0):
    """Transform hours, minutes, seconds to seconds [+days]
    """
    return 3600 * h + 60 * m + s + d * 24 * 3600


def delete_multidim_columns(table):
    """Deletes nulti dimmensional columns in a table
    """
    for column in table.colnames:
        if len(table[column].shape) > 1:
            table.remove_column(column)
    return table


def apply_query(cat=None, query=None, path=None, new_path=None, delete=False):
    """Apply panda easy to use SQL querys to astropy table
    """
    import astropy
    from astropy.table import Table

    if path is not None:
        try:
            cat = Table.read(path)
        except astropy.io.registry.IORegistryError:
            cat = Table.read(path, format="ascii")
    if delete:
        cat = delete_multidim_columns(cat)
    if query is not None:
        df = cat.to_pandas()
        new_table = df.query(query)
        cat = Table.from_pandas(new_table)
    if new_path is not None:
        if ".fits" in os.path.basename(new_path):
            cat.write(new_path, overwrite=True)
        elif ".csv" in os.path.basename(new_path):
            cat.write(new_path, overwrite=True, format="csv")
        else:
            cat.write(new_path, overwrite=True, format="ascii")
    return cat


def import_table_as_region(
    xpapoint=None, name=None, ID=None, system="image", argv=[]
):
    """Import a catalog as regions in DS9
    """
    import astropy
    from astropy.io import fits
    from astropy.table import Table
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p", "--path", help="Path of the catalog to load", metavar="",
    )
    parser.add_argument(
        "-xy",
        "--fields",
        help="Name of the x and y fields: ex = xcentroid,ycentroid",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="",
        help="field to put in region name: ex = magnitude, name, etc",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-f",
        "--form",
        default="circle",
        help="form of the regions to display",
        type=str,
        metavar="",
        choices=["circle", "box"],
    )
    parser.add_argument(
        "-r",
        "--radius",
        help="Size in pixel or arcseconds",
        type=str,
        metavar="",
        default="10",
    )
    parser.add_argument(
        "-w",
        "--WCS",
        help="Check if the catalog fields are in degree-WCS, then radius must be in arc-second",
        type=str,
        metavar="",
        default="0",
    )
    parser.add_argument(
        "-s",
        "--selection",
        default="",
        help="selection of the region in table: Use | for OR and \& for AND",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    try:
        cat = Table.read(args.path.rstrip()[::-1].rstrip()[::-1])
    except astropy.io.registry.IORegistryError:
        cat = Table.read(args.path.rstrip()[::-1].rstrip()[::-1], format="ascii")
    cat = delete_multidim_columns(cat)
    form = args.form
    size = args.radius
    wcs = bool(int(args.WCS))
    query = args.selection
    if args.fields == "":
        x, y = cat.colnames[:2]
    else:
        x, y = args.fields.replace(",", "-").split("-")
    if args.name != "":
        ID = args.name
    if query != "":
        df = cat.to_pandas()
        new_table = df.query(query)
        cat = Table.from_pandas(new_table)
    if wcs:
        from astropy.wcs import WCS

        filename = get_filename(d)
        fitsfile = fits.open(filename)
        ext = fits_ext(fitsfile)

        a = fits.getheader(filename, ext=ext)
        wcs = WCS(a)
        corners = [  # wcs.pixel_to_world(0,a['NAXIS2']),
            wcs.pixel_to_world(0, a["NAXIS1"]),
            # wcs.pixel_to_world(a['NAXIS1'],0),
            wcs.pixel_to_world(a["NAXIS2"], 0),
            wcs.pixel_to_world(a["NAXIS1"], a["NAXIS2"]),
            wcs.pixel_to_world(0, 0),
        ]
        ra_max = max([corner.ra.deg for corner in corners])
        ra_min = min([corner.ra.deg for corner in corners])
        dec_max = max([corner.dec.deg for corner in corners])
        dec_min = min([corner.dec.deg for corner in corners])
        cat = cat[
            (cat[x] < ra_max)
            & (cat[x] > ra_min)
            & (cat[y] < dec_max)
            & (cat[y] > dec_min)
        ]
        d.set("regions system wcs ; regions sky fk5 ; regions skyformat degrees")
        system = "fk5"
        size = float(size)
        size *= abs(wcs.wcs.cd[0][0])

    verboseprint(cat)
    if (ID == "") or (ID is None):
        create_ds9_regions(
            [cat[x]],
            [cat[y]],
            radius=np.ones(len(cat)) * float(size),
            form=[form],
            save=True,
            color=["yellow"],
            ID=None,
            savename=tmp_region,
            system=system,
        )
    else:
        create_ds9_regions(
            [cat[x]],
            [cat[y]],
            radius=np.ones(len(cat)) * float(size),
            form=[form],
            save=True,
            color=["yellow"],
            ID=[np.round(np.array(cat[ID], dtype=float), 1)],
            savename=tmp_region,
            system=system,
        )
    d.set("regions %s" % (tmp_region))
    return cat, tmp_region


def save_region_as_catalog(xpapoint=None, name=None, new_name=None, argv=[]):
    """Save DS9 regions as a catalog [DS9 required]
    """
    import numpy as np
    from astropy.table import Table

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-p",
        "--path",
        help="Path where to save the region file.",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)

    verboseprint(new_name)

    d = DS9n(args.xpapoint)
    image = d.get_pyfits()[0].data

    if new_name is None:
        new_name = args.path
    if name is not None:
        d.set("regions " + name)
    d.set("regions select all")
    regions = getregion(d, all=False, quick=False, selected=True)
    if regions is None:
        message(
            d,
            """It seems that you did not create any region.
                      Please create regions and re-run the analysis.""",
        )
        sys.exit()
    if hasattr(regions[0], "xc"):
        x, y, r1, r2 = (
            np.array([r.xc for r in regions]),
            np.array([r.yc for r in regions]),
            np.array([r.r if hasattr(r, "r") else r.w for r in regions]),
            np.array([r.r if hasattr(r, "r") else r.h for r in regions]),
        )
    else:
        x, y, r1, r2 = (
            np.array([r.xc for r in [regions]]),
            np.array([r.yc for r in [regions]]),
            np.array([r.r if hasattr(r, "r") else r.w for r in [regions]]),
            np.array([r.r if hasattr(r, "r") else r.h for r in [regions]]),
        )
    cat = Table((x - 1, y - 1, r1, r2), names=("x", "y", "w", "h"))
    verboseprint(cat)
    images = []
    w = int(cat[0]["w"])
    h = int(cat[0]["h"])
    for x, y in zip(cat["x"].astype(int), cat["y"].astype(int)):
        im = image[x - w : x + w, y - h : y + h]
        if im.size == 4 * w * h:
            images.append(im)
        else:
            images.append(np.nan * np.zeros((2 * w, 2 * h)))  # *np.nan)

    images = np.array(images)
    verboseprint(images)
    cat["var"] = np.nanvar(images, axis=(1, 2))
    cat["std"] = np.nanstd(images, axis=(1, 2))
    cat["mean"] = np.nanmean(images, axis=(1, 2))
    cat["median"] = np.nanmedian(images, axis=(1, 2))
    cat["min"] = np.nanmin(images, axis=(1, 2))
    cat["max"] = np.nanmax(images, axis=(1, 2))
    if new_name is None:
        new_name = "/tmp/regions.csv"
    verboseprint(new_name)
    if "csv" in new_name:
        cat.write(new_name, overwrite=True, format="csv")
    else:
        cat.write(new_name, overwrite=True)
    return cat


def mask_regions(xpapoint=None, length=20, argv=[]):
    """Replace DS9 defined regions as a catalog
    """
    import numpy as np

    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    try:
        top, bottom, left, right = np.array(sys.argv[3:7], dtype=int)
    except (IndexError, ValueError) as e:
        verboseprint(e)
        top, bottom, left, right = 0, 0, 4, 0
    path = globglob(sys.argv[-1])
    name = "/tmp/cat.csv"
    cosmic_rays = save_region_as_catalog(args.xpapoint, new_name=name)
    cosmic_rays["front"] = 1
    cosmic_rays["dark"] = 0
    cosmic_rays["id"] = np.arange(len(cosmic_rays))
    for filename in path:
        fitsimage, name = mask_regions_2(
            filename,
            regions=cosmic_rays,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )
    if len(path) < 2:
        d.set("frame new ; tile yes ; file %s" % (name))
    return fitsimage, name


def mask_regions_2(filename, top, bottom, left, right, regions=None):
    """Replace DS9 defined regions as a catalog
    """
    from astropy.io import fits

    fitsimage = fits.open(filename)[0]
    maskedimage = mask_cosmic_rays(
        fitsimage.data,
        cosmics=regions,
        top=top,
        bottom=bottom,
        left=left,
        right=right,
        all=True,
        cols=None,
    )
    fitsimage.data = maskedimage
    name = (
        os.path.dirname(filename)
        + "/"
        + os.path.basename(filename)[:-5]
        + "_masked.fits"
    )
    if not os.path.exists(os.path.dirname(name)):
        os.makedirs(os.path.dirname(name))
    fitswrite(fitsimage, name)
    return fitsimage, name


def mask_cosmic_rays(
    image, cosmics, top=0, bottom=0, left=4, right=0, cols=None, all=False
):
    """Replace pixels impacted by cosmic rays by NaN values
    """
    from tqdm import tqdm_gui  # tqdm,
    import numpy as np

    y, x = np.indices((image.shape))
    image = image.astype(float)
    if all is False:
        cosmics = cosmics[(cosmics["front"] == 1) & (cosmics["dark"] < 1)]
    for i in tqdm_gui(range(len(cosmics))):  # range(len(cosmics)):
        image[
            (y > cosmics[i]["ycentroid"] - bottom - 0.1)
            & (y < cosmics[i]["ycentroid"] + top + 0.1)
            & (x < cosmics[i]["xcentroid"] + right + 0.1)
            & (x > -left - 0.1 + cosmics[i]["xcentroid"])
        ] = np.nan

    return image


def symlink_force(target, link_name):
    """Create a symbolic link
    """
    import errno

    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def even(f):
    """Return the even related number
    """
    import numpy as np

    return np.ceil(f / 2.0) * 2


def distance(x1, y1, x2, y2):
    """
    Compute distance between 2 points in an euclidian 2D space
    """
    import numpy as np

    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))


def return_path(filename, number=None, all_images=False):
    """Return path using regexp
    """
    import re
    import numpy as np

    number1 = re.findall(r"\d+", os.path.basename(filename))[-1]
    n = len(number1)
    filen1, filen2 = filename.split(number1)
    if number is not None:
        number = int(float(number))
        return filen1 + "%0{}d".format(n) % (number) + filen2
    elif all_images:
        path = glob.glob("%s%s%s" % (filen1, "?" * n, filen2))
        np.sort(path)
        return path


def create_areas(image, area=None, radius=100, offset=20, verbose=False):
    """Create areas in the given image
    """
    import numpy as np

    if type(radius) == int:
        r1, r2 = radius, radius
    else:
        r1, r2 = radius
    ly, lx = image.shape
    if area is None:
        if ly == 2069:
            ymin, ymax = 0, ly
        else:
            xmin, xmax = 0, lx
            ymin, ymax = 0, ly
    else:
        xmin, xmax = area[0], area[1]
        ymin, ymax = area[2], area[3]
    xi = np.arange(offset + xmin, xmax - offset - r1, r1)
    yi = np.arange(offset + ymin, ymax - offset - r2, r2)
    xx, yy = np.meshgrid(xi, yi)
    areas = [[a, a + r2, b, b + r1] for a, b in zip(yy.flatten(), xx.flatten())]
    return areas


def trim(xpapoint=None, all_ext=False, argv=[]):
    """Crop/trim the image, WCS compatible [DS9 required]
    """
    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-s", "--system", default="Image", type=str, choices=["Image"]
    )
    args = parser.parse_args_modif(argv)

    import numpy as np

    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    syst = args.system
    path = globglob(args.path, xpapoint=args.xpapoint)

    verboseprint("system = ", syst)
    verboseprint("path = ", path)
    region = getregion(d, quick=True, selected=True, system=syst, dtype=float)
    x_inf, x_sup, y_inf, y_sup = lims_from_region(
        None, coords=region[0], dtype=float
    )
    if (len(region[0]) != 5) & (len(region[0]) != 4):
        message(
            d,
            """Trimming only works on box regions.
                  Select a box region and re-run the analysis.""",
        )
        sys.exit()
    else:
        for filename in path:
            verboseprint(filename)
            verboseprint("Using WCS information.")
            result, name = crop_wcs(
                path=filename,
                position=[region[0][0] - 1, region[0][1] - 1],
                size=np.array([region[0][3], region[0][2]], dtype=int),
                all_ext=False,
            )
    if len(path) < 2:
        d.set("frame new ; tile yes ; file %s" % (name))
    return


def crop_wcs(path, position=[0, 0], size=[10, 10], all_ext=False):
    """Cropping/Trimming function that keeps WCS header information
    """
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS
    from astropy.io.fits import ImageHDU

    a = fits.open(path)
    for i in range(1):
        try:
            di = Cutout2D(
                a[i].data, position=position, size=size, wcs=WCS(a[i].header)
            )
            if i == 0:
                a[i] = fits.PrimaryHDU(data=di.data, header=di.wcs.to_header())
            else:
                a[i] = ImageHDU(data=di.data, header=di.wcs.to_header())
            # a[i].header["CD1_1"] = b[i].header["CD1_1"]
            # a[i].header["CD2_2"] = b[i].header["CD2_2"]

        except (ValueError, IndexError) as e:
            verboseprint(i, e)
    a.writeto(path[:-5] + "_trim.fits", overwrite=True)
    return a, path[:-5] + "_trim.fits"


def column_line_correlation(xpapoint=None, argv=[]):
    """Performs a column to column or or line to line auto-correlation
    on a DS9 image
    """
    parser = create_parser(get_name_doc(), path=True)
    args = parser.parse_args_modif(argv, required=True)

    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    path = globglob(args.path, xpapoint=args.xpapoint)
    region = getregion(d, quick=True, message=False, selected=True)
    if region is not None:
        x_inf, x_sup, y_inf, y_sup = lims_from_region(None, coords=region)
        area = [y_inf, y_sup, x_inf, x_sup]
    else:
        area = [0, -1, 0, -1]

    for filename in path:
        verboseprint(filename)
        cl_correlation(filename, area=area)
    return


def cl_correlation(path, area=[0, -1, 1053, 2133], DS9backUp=DS9_BackUp_path):
    """Performs a column to column or or line to line auto-correlation
    on a DS9 image
    """
    from astropy.io import fits
    import numpy as np

    fitsimage = fits.open(path)[0]
    image = fitsimage.data[area[0] : area[1], area[2] : area[3]]
    imagex = np.nanmean(image, axis=1)
    imagey = np.nanmean(image, axis=0)
    nbins = 300
    bins1 = np.linspace(
        np.percentile(imagex[1:] - imagex[:-1], 5),
        np.percentile(imagex[1:] - imagex[:-1], 95),
        nbins,
    )
    bins2 = np.linspace(
        np.percentile(imagey[1:] - imagey[:-1], 5),
        np.percentile(imagey[1:] - imagey[:-1], 95),
        nbins,
    )
    x = (image[:, 1:] - image[:, :-1]).flatten()
    y = (image[1:, :] - image[:-1, :]).flatten()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    bins3 = np.linspace(np.percentile(x, 5), np.percentile(x, 95), nbins)
    bins4 = np.linspace(np.percentile(y, 5), np.percentile(y, 95), nbins)
    vals1, b_ = np.histogram(imagex[1:] - imagex[:-1], bins=bins1)
    vals2, b_ = np.histogram(imagey[1:] - imagey[:-1], bins=bins2)
    vals3, b_ = np.histogram(x, bins=bins3)
    vals4, b_ = np.histogram(y, bins=bins4)

    np.savetxt(
        DS9_BackUp_path + "/CSVs/1.dat",
        np.array([(bins1[1:] + bins1[:-1]) / 2, vals1]).T,
    )
    np.savetxt(
        DS9_BackUp_path + "/CSVs/2.dat",
        np.array([(bins2[1:] + bins2[:-1]) / 2, vals2]).T,
    )
    np.savetxt(
        DS9_BackUp_path + "/CSVs/3.dat",
        np.array([(bins3[1:] + bins3[:-1]) / 2, vals3]).T,
    )
    np.savetxt(
        DS9_BackUp_path + "/CSVs/4.dat",
        np.array([(bins4[1:] + bins4[:-1]) / 2, vals4]).T,
    )

    d = []
    d.append("plot line open")
    d.append("plot axis x grid no ")
    d.append("plot axis y grid no ")
    d.append("plot title y 'Lines' ")
    d.append("plot load %s/CSVs/1.dat xy  " % (DS9_BackUp_path))
    d.append("plot add graph ")
    d.append("plot axis x grid no")
    d.append("plot axis y grid no ")
    d.append("plot load %s/CSVs/3.dat xy  " % (DS9_BackUp_path))
    d.append("plot add graph ")
    d.append("plot title y 'delta chisqr' ")
    d.append("plot load %s/CSVs/2.dat xy " % (DS9_BackUp_path))
    d.append("plot title y 'Columns' ")
    d.append("plot axis x grid no ")
    d.append("plot axis y grid no ")
    d.append("plot title x 'Column/Line average difference' ")
    d.append("plot add graph ")
    d.append("plot load %s/CSVs/4.dat xy " % (DS9_BackUp_path))
    d.append("plot title x 'Pixel value difference' ")
    d.append("plot axis x grid no ")
    d.append("plot axis y grid no ")
    d.append("plot layout grid")
    ds9 = DS9n()
    ds9.set(" ; ".join(d))
    return


def create_header_catalog(xpapoint=None, files=None, info=False, argv=[]):
    """Generate fits files database based on header information
    0.5 second per image for info
    10ms per image for header info, 50ms per Mo so 240Go->
    """
    from astropy.table import vstack
    from astropy.io import fits
    import numpy as np

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-i",
        "--info",
        default="None",
        help="Addition file in pyds9plugin/Macros/Macros_Header_catalog/ that can be added for image analysis",
        type=str,
        metavar="",
    )  # , choices=['image','none','wcs'])#metavar='',
    args = parser.parse_args_modif(argv, required=True)
    extension = "-"
    verboseprint("info, extension = %s, %s" % (args.info, extension))

    if files is None:
        files = globglob(args.path, ds9_im=False)
        verboseprint("%s : %s" % (args.path, files))
        while len(files) == 0:
            d = DS9n(args.xpapoint)
            path = get(
                d,
                "No file matching the regular expression. Use regular exp.",
                exit_=True,
            )
            files = globglob(path, ds9_im=False)
    fname = os.path.dirname(os.path.dirname(files[0]))
    verboseprint(fname)
    ext = len(fits.open(files[0]))
    d = DS9n(args.xpapoint)
    if ext != 1:
        if yesno(
            d,
            "Your image contains %i extensions. Do you wish to analyze only the primary header?"
            % (ext),
        ):
            extentsions = [0]
        else:
            extentsions = np.arange(ext)
    else:
        extentsions = [0]
    verboseprint("Extensions to add: %s" % (extentsions))
    verboseprint(files)
    t1s = [
        create_catalog(
            files,
            ext=extentsions,
            info=os.path.join(
                os.path.dirname(__file__),
                "Macros/Macros_Header_catalog/" + args.info,
            ),
        )
    ]
    from datetime import datetime

    path_db = os.environ[
        "HOME"
    ] + "/DS9QuickLookPlugIn/HeaderDataBase/HeaderCatalog_%s.csv" % (
        datetime.now().strftime("%y%m%d-%HH%Mm%Ss")
    )
    verboseprint(t1s)
    t1 = vstack(t1s)
    csvwrite(t1, path_db)
    question = (
        "Analysis saved in %s! Do you want to load the file with PRISM?"
        % (path_db)
    )
    if yesno(d, question):
        d.set("prism import csv " + path_db)
    return


def get_columns(path):
    """IMPORTANT get column names from fits table path quickly wo opening it
    """
    if ".fits" in path:
        from astropy.io import fits

        header = fits.open(path)[1].header
        a = list(dict.fromkeys(header.keys()))
        cols = [header[item] for item in a if "TYPE" in item]
    else:
        with open(path) as f:
            first_line = f.readline()
        cols = first_line.split()
    return cols


def parallelize(
    function=lambda x: print(x),
    action_to_paralize=[],
    parameters=[],
    number_of_thread=10,
):
    """Use multi-processing to run the function on all the entries
    """
    from tqdm import tqdm
    from multiprocessing import Process, Manager
    from pyds9plugin.BasicFunctions import RunFunction

    info = [
        action_to_paralize[x : x + int(number_of_thread)]
        for x in range(0, len(action_to_paralize), int(number_of_thread))
    ]
    for i in tqdm(range(len(info))):
        subinfo = info[i]
        jobs = []
        manager = Manager()
        return_dict = manager.dict()
        for inf in subinfo:
            p = Process(
                target=RunFunction,
                args=(function, [inf] + parameters, return_dict,),
            )
            jobs.append(p)
            p.start()
        for job in jobs:
            job.join()
    if len(action_to_paralize) > 0:
        try:
            return return_dict["output"]
        except (KeyError, UnboundLocalError, TypeError) as e:
            print(e)
            return


def create_catalog(files, ext=[0], info=None):
    """Create header catalog from a list of fits file
    """
    from astropy.table import Column, vstack  # hstack,
    from datetime import datetime
    import warnings
    from tqdm import tqdm  # _gui
    import numpy as np

    warnings.simplefilter("ignore", UserWarning)
    files.sort()
    path = files[0]

    file_header = []
    files_name = []
    for i in tqdm(
        range(len(files)), file=sys.stdout, desc="Number of files: ", ncols=99,
    ):
        try:
            table = create_table_from_header(files[i], exts=ext, info=info)
            if table is not None:
                file_header.append(table)
                files_name.append(files[i])
        except OSError:
            # verboseprint('Empty or corrupt FITS file :', files[i])
            pass
    table_header = vstack(file_header)
    table_header.add_column(
        Column(np.arange(len(table_header)), name="Index", dtype=str),
        index=0,
        rename_duplicate=True,
    )
    table_header.add_column(
        Column(files_name, name="Path"), index=-1, rename_duplicate=True
    )
    table_header.add_column(
        Column([os.path.basename(file) for file in files_name]),
        name="Filename",
        index=1,
        rename_duplicate=True,
    )
    table_header.add_column(
        Column([os.path.basename(os.path.dirname(f)) for f in files_name]),
        name="Directory",
        index=2,
        rename_duplicate=True,
    )
    try:
        table_header.add_column(
            Column(
                [
                    datetime.strptime(
                        time.ctime(os.path.getctime(file)),
                        "%a %b %d %H:%M:%S %Y",
                    ).strftime("%y%m%d.%H%M")
                    for file in files_name
                ]
            ),
            name="CreationTime",
            index=3,
            rename_duplicate=True,
        )
        table_header.add_column(
            Column(
                [
                    datetime.strptime(
                        time.ctime(os.path.getmtime(file)),
                        "%a %b %d %H:%M:%S %Y",
                    ).strftime("%y%m%d.%H%M")
                    for file in files_name
                ]
            ),
            name="ModificationTime",
            index=4,
            rename_duplicate=True,
        )
    except ValueError:
        pass
    table_header.add_column(
        Column(["%0.2f" % (os.stat(f).st_size / 1e6) for f in files_name]),
        name="FileSize_Mo",
        index=5,
        rename_duplicate=True,
    )
    csvwrite(
        table_header.filled(""), os.path.dirname(path) + "/HeaderCatalog.csv"
    )
    return table_header


def remove_duplicates(hdr):
    """Removes duplicate in fits headers
    """
    from collections import Counter

    cnts = Counter(hdr.keys())
    keys = {key: n for key, n in cnts.items() if n >= 2}
    for key in keys:
        for number in range(keys[key]):
            hdr.remove(key)
    return hdr


def create_table_from_header(path, exts=[0], info=None):
    """Create table from fits header
    """
    from astropy.table import Table, hstack
    from astropy.io import fits
    import numpy as np

    tabs = []
    if exts == "all":
        with fits.open(path) as file:
            exts = np.arange(len(file))
    for ext in exts:
        try:
            header = remove_duplicates(fits.getheader(path, ext))
            cat = Table(
                data=np.array([val for val in header.values()]),
                names=[key for key in header.keys()],
                dtype=["S20" for key in header.keys()],
            )
            tabs.append(cat)
        except IndexError:
            pass
    verboseprint(info, os.path.isfile(info))
    if len(tabs) > 0:
        table = hstack(tabs)
        # verboseprint(table.colnames)
        if os.path.isfile(info):
            fitsfile = fits.open(path)
            ldict = {
                "fitsfile": fitsfile,
                "header": header,
                "np": np,
                "table": table,
            }
            verboseprint("Executing file %s" % (exp))
            try:
                exec(open(info).read(), globals(), ldict)
            except (SyntaxError, NameError) as e:
                print(e)
        return table


def rescale(img, target_type):
    """Rescale data based on dtype"""
    import numpy as np

    imin = np.nanmin(img)
    imax = np.nanmax(img)
    try:
        target_type_min, target_type_max = (
            np.iinfo(target_type).min,
            np.iinfo(target_type).max,
        )
    except ValueError:
        target_type_min, target_type_max = (
            np.finfo(target_type).min,
            np.finfo(target_type).max,
        )
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


# @profile
def convert_image(xpapoint=None, argv=[]):
    """Convert and scale file into other type
    """
    from astropy.io import fits
    import numpy as np

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-t",
        "--type",
        default="8,uint8",
        help="Conversion type of the image",
        type=str,
        choices=[
            "8,uint8",
            "16,int16",
            "32,int32",
            "64,int64",
            "-32,float32",
            "-64,float64",
        ],
        required=True,
    )
    parser.add_argument(
        "-r",
        "--rescale",
        default=1,
        help="Rescale or not the image",
        metavar="",
        type=int,
    )
    args = parser.parse_args_modif(argv)
    d = DS9n(args.xpapoint)
    rescale_ = args.rescale
    type = args.type
    path = args.path
    filenames = globglob(path, ds9_im=True)

    python_type = type.split(",")[-1]
    for filename in filenames:
        fitsimage = fits.open(filename)  #
        verboseprint(getattr(np, python_type))
        if rescale_:
            data = rescale(fitsimage[0].data, getattr(np, python_type))
            fitsimage[0].data = getattr(np, python_type)(data)
        else:
            fitsimage[0].data = getattr(np, python_type)(fitsimage[0].data)
        if len(filenames) == 1:
            load = True
        else:
            load = False
        save_fits_images(
            d,
            fitsimage,
            filename.replace(".fits", "_%s.fits" % (python_type)),
            load=load,
        )
    return


def save_fits_images(d, file, filename, load=False):
    if os.path.exists(filename):
        if yesno(
            d,
            "%s already exists. Do you want to replace it?"
            % (os.path.basename(filename)),
        ):
            file[0].writeto(filename, overwrite=True)
            if load:
                d.set("frame new")
                d.set("file %s" % (filename))
        else:
            if load:
                d.set("frame new")
                d.set_pyfits(file)
    else:
        file[0].writeto(filename)
        if load:
            d.set("frame new")
            d.set("file %s" % (filename))

    return


def fill_regions(xpapoint=None, argv=[]):
    """Replace the pixels in the selected regions in DS9 by NaN values
    [DS9 required]
    """
    import numpy as np
    from numpy import inf, nan

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-v",
        "--value",
        help="Value to replace in the regions",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default="0",
        help="Overwrite the image",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv)

    # verboseprint(inf + nan)
    d = DS9n(args.xpapoint)
    filename = get_filename(d)  # d.get("file")
    regions = getregion(d, selected=True)
    if getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region.
 make sure to click on the region after creating
it and re-run the analysis.""",
        )
        return
    fitsimage = d.get_pyfits()[0]  # fits.open(filename)[0]
    value = eval(args.value)
    overwrite = bool(int(args.overwrite))
    image = fitsimage.data.astype(float).copy()
    # verboseprint(regions)
    try:
        xc, yc, h, w = (
            int(regions.xc),
            int(regions.yc),
            int(regions.h),
            int(regions.w),
        )

        x_inf = int(np.floor(yc - h / 2 - 1))
        x_sup = int(np.ceil(yc + h / 2 - 1))
        y_inf = int(np.floor(xc - w / 2 - 1))
        y_sup = int(np.ceil(xc + w / 2 - 1))
        image[x_inf : x_sup + 1, y_inf : y_sup + 2] = value
    except AttributeError:
        verboseprint("Several regions found...")
        for region in regions:
            x, y = np.indices(image.shape)
            try:
                xc, yc, h, w = (
                    int(region.xc),
                    int(region.yc),
                    int(region.h),
                    int(region.w),
                )
            except AttributeError:
                xc, yc, h, w = (
                    int(region.xc),
                    int(region.yc),
                    int(region.r),
                    int(region.r),
                )
                radius = np.sqrt(np.square(y - xc) + np.square(x - yc))
                mask = radius < h
            else:

                x_inf = int(np.floor(yc - h / 2 - 1))
                x_sup = int(np.ceil(yc + h / 2 - 1))
                y_inf = int(np.floor(xc - w / 2 - 1))
                y_sup = int(np.ceil(xc + w / 2 - 1))
                mask = (
                    (x > x_inf) & (x < x_sup + 1) & (y > y_inf) & (y < y_sup + 1)
                )
            image[mask] = value  # np.nan
    fitsimage.data = image
    if overwrite:
        filename = fitswrite(fitsimage, filename)
    else:
        filename = fitswrite(fitsimage, filename + "_modified.fits")
    d.set("file " + filename)
    return


def extract_sources(xpapoint=None, argv=[]):
    """Extract sources from images(s)
    """
    import numpy as np

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-e", "--erosion", default="2", type=str, metavar="",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default="10",
        help="Extraction threshold",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-f",
        "--fwhm",
        default="8",
        help="The full-width half-maximum (FWHM) of the major axis of the Gaussian kernel in pixels. Enter eg. 8,10 to loop on FWHM",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-a",
        "--angle",
        default="0",
        help="The position angle (in degrees) of the major axis of the Gaussian kernel measured counter-clockwise from the positive x axis",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-i",
        "--iteration",
        default="5",
        help="The number of iterations to perform sigma clipping, or None to clip until convergence is achieved when calculating the statistics",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-r",
        "--ratio",
        default="1",
        help="The ratio of the minor to major axis standard deviations of the Gaussian kernel",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-d",
        "--distance",
        default="30",
        help="When entering several thresholds/FWHMs, the minimal distance between two sources in order to delete duplicates",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-N",
        "--number_processors",
        default=os.cpu_count() - 2,
        help="Number of processors to use for multiprocessing analysis.",
        metavar="",
    )

    args = parser.parse_args_modif(argv, required=True)
    d = DS9n(args.xpapoint)
    threshold = np.array(args.threshold.split(","), dtype=float)
    fwhm = np.array(args.fwhm.split(","), dtype=float)
    path = globglob(args.path, xpapoint=args.xpapoint)
    sources = parallelize(
        function=extract_sources_photutils,
        parameters=[
            fwhm,
            threshold,
            float(args.angle),
            float(args.ratio),
            int(args.erosion),
            3,
            int(args.iteration),
            int(args.distance),
        ],
        action_to_paralize=path,
        number_of_thread=args.number_processors,
    )
    verboseprint(sources)
    if len(path) < 2:
        create_ds9_regions(
            [sources["xcentroid"]],
            [sources["ycentroid"]],
            radius=[10],
            save=True,
            savename=tmp_region,
            form=["circle"],
            color=["yellow"],
            ID=None,
        )
        d.set("region delete all ; region {}".format(tmp_region))
    return


def delete_doublons(sources, dist):
    """Function that delete doublons detected in a table,
    the initial table and the minimal distance must be specifies
    """
    import numpy as np

    try:
        sources["doublons"] = 0
        for i in range(len(sources)):
            a = (
                distance(
                    sources[sources["doublons"] == 0]["xcentroid"],
                    sources[sources["doublons"] == 0]["ycentroid"],
                    sources["xcentroid"][i],
                    sources["ycentroid"][i],
                )
                > dist
            )
            a = list(1 * a)
            a.remove(0)
            if np.nanmean(a) < 1:
                sources["doublons"][i] = 1
        return sources[sources["doublons"] == 0]
    except TypeError:
        verboseprint("no source detected")


def extract_sources_photutils(
    filename,
    fwhm=5,
    threshold=8,
    theta=0,
    ratio=1,
    n=2,
    sigma=3,
    iters=5,
    distance_doublons=3,
):
    """Extract sources for DS9 image and create catalog
    """
    from astropy.io import fits
    from scipy import ndimage
    from astropy.table import Table
    from astropy.stats import sigma_clipped_stats
    from photutils import DAOStarFinder
    import numpy as np

    fitsfile = fits.open(filename)
    sizes = [sys.getsizeof(elem.data) for elem in fitsfile]  # [1]
    data = fitsfile[np.argmax(sizes)].data
    data2 = ndimage.grey_dilation(
        ndimage.grey_erosion(data, size=(n, n)), size=(n, n)
    )
    mean, median, std = sigma_clipped_stats(data2, sigma=sigma, maxiters=iters)
    daofind = DAOStarFinder(
        fwhm=fwhm[0], threshold=threshold[0] * std, ratio=ratio, theta=theta
    )
    sources0 = daofind(data2 - median)
    verboseprint(
        "fwhm = {}, T = {}, len = {}".format(
            fwhm[0], threshold[0], len(sources0)
        )
    )
    for i in fwhm[1:]:
        for j in threshold[1:]:
            daofind = DAOStarFinder(
                fwhm=i, threshold=j * std, ratio=ratio, theta=theta
            )

            sources1 = daofind(data2 - median)
            verboseprint(
                "fwhm = {}, T = {}, len = {}".format(i, j, len(sources1))
            )
            try:
                sources0 = Table(np.hstack((sources0, sources1)))
            except TypeError:
                verboseprint("catalog empty")
                if len(sources0) == 0:
                    sources0 = sources1
    sources = delete_doublons(sources0, dist=distance_doublons)
    csvwrite(sources, filename[:-5] + ".csv")
    return sources


def get_filename(ds9, All=False, sort=True):
    """Get the filename of the loaded image in DS9
    """
    if isinstance(ds9, FakeDS9):
        return ""
    if not All:
        backup_path = os.environ["HOME"] + "/DS9QuickLookPlugIn"
        if not os.path.exists(os.path.dirname(backup_path)):
            os.makedirs(os.path.dirname(backup_path))
        filename = ds9.get("file")
        if filename == "":
            try:
                fits_im = ds9.get_pyfits()[0]
                filename = backup_path + tmp_image
                fitswrite(fits_im, filename)
            except TypeError:
                return filename

        if os.path.basename(filename)[-1] == "]":
            filename = filename.split("[")[0]
        if len(filename) == 0:
            new_filename = filename
        elif os.path.isfile(filename) is False:
            try:
                fits_im = ds9.get_pyfits()[0]
                filename = backup_path + tmp_image
                fitswrite(fits_im, filename)
                new_filename = filename
            except TypeError:
                return filename

        elif filename[0] == ".":
            new_filename = backup_path + "/BackUps" + filename[1:]
            verboseprint(
                "Filename in the DS9 backup repository, changing path to %s"
                % (new_filename)
            )
        else:
            new_filename = filename
    else:
        new_filename = []
        verboseprint("Taking images opened in DS9")
        current = ds9.get("frame")
        number = number_ds9_frames(ds9.get("xpa info").split("\t")[-1])
        for i in range(number):
            ds9.set("frame next")
            file = ds9.get("file")
            if os.path.isfile(file):
                new_filename.append(file)
        ds9.set("frame " + current)
        if sort:
            new_filename.sort()
    verboseprint(new_filename)
    return new_filename


def number_ds9_frames(xpapoint=None):
    """Returns number of frame used in DS9
    """
    d = DS9n(xpapoint)
    d.set("frame last")
    last = d.get("frame")
    d.set("frame first")
    a = ""
    number = 1
    while a != last:
        d.set("frame next")
        a = d.get("frame")
        number += 1
    return number


def add_field_to_header(xpapoint=None, field="", value="", argv=[]):
    """Add header field to image header
    """
    from astropy.io import fits

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-f", "--field", default="FIELD", help="Header field to add", type=str
    )
    parser.add_argument(
        "-v",
        "--value",
        default="VALUE",
        help="Value to add",
        metavar="",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--comment",
        default="",
        help="Comment to add",
        metavar="",
        type=str,
    )

    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    filename = get_filename(d)

    if field == "":
        field = args.field
    if value == "":
        value = args.value
        try:
            value = float(value)
        except ValueError:
            pass
    try:
        comment = args.comment
    except IndexError:
        pass
    path = globglob(args.path, xpapoint=args.xpapoint)
    for filename in path:
        verboseprint(filename)
        header = fits.getheader(filename)
        # if "NAXIS3" in header:
        #     verboseprint("2D array: Removing NAXIS3 from header...")
        #     fits.delval(filename, "NAXIS3")
        fits.setval(filename, field[:8], value=value, comment=comment)

    if len(path) < 2:
        d.set("frame clear ; file " + filename)
    return


def background_estimation(
    xpapoint=None, n=2, DS9backUp=DS9_BackUp_path, plot_=True, argv=[]
):
    """Estimate image(s) background
    """
    import numpy as np

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-s", "--sigma", default="3", metavar="", help="", type=str
    )
    parser.add_argument(
        "-b",
        "--background",
        default="MeanBackground",
        metavar="",
        help="Different background estimators that can be used",
        type=str,
        choices=[
            "MeanBackground",
            "MedianBackground",
            "ModeEstimatorBackground",
            "MMMBackground",
            "SExtractorBackground",
            "BiweightLocationBackground",
        ],
    )
    parser.add_argument(
        "-r",
        "--rms",
        default="StdBackgroundRMS",
        help="Different RMS estimators that can be used",
        type=str,
        choices=[
            "StdBackgroundRMS",
            "MADStdBackgroundRMS",
            "BiweightScaleBackgroundRMS",
        ],
    )
    parser.add_argument(
        "-f",
        "--filter",
        default="3,3",
        metavar="",
        help="The window size of the 2D median filter to apply to the low-resolution background map",
        type=str,
    )
    parser.add_argument(
        "-box",
        "--box",
        default="40,40",
        metavar="",
        help="Size of the box to be used",
        type=str,
    )
    parser.add_argument(
        "-per",
        "--percentile",
        default="20",
        metavar="",
        help="Percentile of the image to exclude in the background estimation",
        type=str,
    )
    parser.add_argument(
        "-snr",
        "--snr",
        default="3",
        metavar="",
        help="The snr per pixel above the background for which to consider a pixel as possibly being part of a source",
        type=str,
    )
    parser.add_argument(
        "-n",
        "--npixels",
        default="3",
        metavar="",
        help="The number of connected pixels, each greater than threshold, that an object must have to be detected.",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--dilate",
        default="5",
        metavar="",
        help="The size of the array used to dilate the segmentation image.",
        type=str,
    )
    parser.add_argument(
        "-m",
        "--mask",
        default="0",
        help="Check this and fill following entries to detect and mask sources found with these settings.",
        type=str,
        choices=["0", "1"],
    )
    parser.add_argument(
        "-N",
        "--number_processors",
        default=os.cpu_count() - 2,
        help="Number of processors to use for multiprocessing analysis. Default use your total number of processors - 2.",
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)
    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    sigma, bckd, rms, filters, boxs, percentile, mask, snr, npixels, dilate = (
        args.sigma,
        args.background,
        args.rms,
        args.filter,
        args.box,
        args.percentile,
        args.mask,
        args.snr,
        args.npixels,
        args.dilate,
    )
    filter1, filter2 = np.array(filters.split(","), dtype=int)
    mask = bool(mask)
    sigma, percentile, snr, npixels, dilate = np.array(
        [sigma, percentile, snr, npixels, dilate], dtype=int
    )
    box1, box2 = np.array(boxs.split(","), dtype=int)
    path = globglob(args.path, xpapoint=args.xpapoint)
    name = parallelize(
        function=background_estimation_phot,
        parameters=[
            float(sigma),
            bckd,
            rms,
            (filter1, filter2),
            (box1, box2),
            n,
            DS9_BackUp_path,
            snr,
            npixels,
            dilate,
            percentile,
            mask,
            plot_,
        ],
        action_to_paralize=path,
        number_of_thread=args.number_processors,
    )
    if len(path) < 2:
        d.set("frame new ; tile yes ; file " + name)
    return name


def background_estimation_phot(
    filename,
    sigma,
    bckd,
    rms,
    filters,
    boxs,
    n=2,
    DS9backUp=DS9_BackUp_path,
    snr=3,
    npixels=15,
    dilate_size=3,
    exclude_percentile=5,
    mask=False,
    plot_=True,
):
    """Estimate backgound in a fits image
    """
    from astropy.io import fits
    from photutils import (
        make_source_mask,
        Background2D,
        MeanBackground,
        MedianBackground,
    )
    from photutils import (
        ModeEstimatorBackground,
        MMMBackground,
        SExtractorBackground,
        BiweightLocationBackground,
    )
    from photutils import (
        StdBackgroundRMS,
        MADStdBackgroundRMS,
        BiweightScaleBackgroundRMS,
    )
    from astropy.stats import SigmaClip  # , sigma_clipped_stats
    import numpy as np

    fitsfile = fits.open(filename)[0]
    data = fitsfile.data  # [400:1700,1200:2000]
    functions = {
        "MeanBackground": MeanBackground,
        "MedianBackground": MedianBackground,
        "ModeEstimatorBackground": ModeEstimatorBackground,
        "MMMBackground": MMMBackground,
        "SExtractorBackground": SExtractorBackground,
        "BiweightLocationBackground": BiweightLocationBackground,
    }
    functions_rms = {
        "StdBackgroundRMS": StdBackgroundRMS,
        "MADStdBackgroundRMS": MADStdBackgroundRMS,
        "BiweightScaleBackgroundRMS": BiweightScaleBackgroundRMS,
    }

    if mask:
        mask_source = make_source_mask(
            data, nsigma=snr, npixels=npixels, dilate_size=dilate_size
        )
    else:
        mask_source = np.ones(data.shape, dtype=bool)
    bkg_estimator = functions[bckd]()
    bkgrms_estimator = functions_rms[rms]()
    bkg = Background2D(
        data,
        boxs,
        filter_size=filters,
        sigma_clip=SigmaClip(sigma=sigma),
        bkg_estimator=bkg_estimator,
        exclude_percentile=exclude_percentile,
        bkgrms_estimator=bkgrms_estimator,
        mask=mask_source,
    )
    verboseprint("Mask, median = %0.2f" % (bkg.background_median))
    verboseprint("Mask, rms = %0.2f" % (bkg.background_rms_median))
    fitsfile.data = fitsfile.data - bkg.background  # .astype('uint16')

    name = os.path.join(
        os.path.dirname(filename)
        + "/bkgd_photutils_substracted/%s" % (os.path.basename(filename))
    )
    fitswrite(fitsfile, name)
    return name


def create_image_from_catalog(xpapoint=None, nb=int(1e3), argv=[]):
    """Create galaxy image form a sextractor catalog
    """
    import astropy
    from astropy.table import Table
    from tqdm import tqdm  # tqdm_gui,
    from astropy.modeling.functional_models import Gaussian2D
    from photutils.datasets import make_100gaussians_image
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        help="Path of the sextractor catalog to process",
        metavar="",
        default="",
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    path = args.path
    if os.path.isfile(path):
        verboseprint("Opening sextractor catalog")
        catfile = path
        try:
            catalog = Table.read(catfile)
        except astropy.io.registry.IORegistryError:
            catalog = Table.read(catfile, format="ascii")

        lx, ly = int(catalog["X_IMAGE"].max()), int(catalog["Y_IMAGE"].max())
        background = np.median(catalog["BACKGROUND"])
        image = np.ones((lx, ly)) * background
        for i in tqdm(range(len(catalog))):
            x = np.linspace(0, lx - 1, lx)
            y = np.linspace(0, ly - 1, ly)
            x, y = np.meshgrid(x, y)
            try:
                image += Gaussian2D.evaluate(
                    x,
                    y,
                    catalog[i]["FLUX_AUTO"],
                    catalog[i]["X_IMAGE"],
                    catalog[i]["Y_IMAGE"],
                    catalog[i]["A_IMAGE"],
                    catalog[i]["B_IMAGE"],
                    np.pi * catalog[i]["THETA_IMAGE"] / 180,
                ).T
            except KeyError:
                image += Gaussian2D.evaluate(
                    x,
                    y,
                    catalog[i]["amplitude"],
                    catalog[i]["x_mean"],
                    catalog[i]["y_mean"],
                    catalog[i]["x_stddev"],
                    catalog[i]["y_stddev"],
                    catalog[i]["theta"],
                )
        try:
            image_real = np.random.poisson(image).T
        except ValueError:
            image_real = image.T

    else:
        verboseprint("No catalog given, creating new image.")
        image_real = make_100gaussians_image()
    name = "/tmp/image_%s.fits" % (
        datetime.datetime.now().strftime("%y%m%d-%HH%MM%S")
    )
    fitswrite(image_real, name)
    d.set("frame new ; file " + name)
    return image_real


def gaussian_2dim(xy, amplitude, xo, yo, sigma_x, sigma_y, angle=0, offset=0):
    """Defines a gaussian function in 2D
    """
    from astropy.modeling.functional_models import Gaussian2D

    xo = float(xo)
    yo = float(yo)
    g = Gaussian2D(
        amplitude=amplitude,
        x_mean=xo,
        y_mean=yo,
        x_stddev=sigma_x,
        y_stddev=sigma_y,
        theta=angle,
    )
    x, y = xy
    return g(x, y).ravel() + offset


functions = ["interactive_plotter", "Function", "fit_ds9_plot"]
if bool(set(functions) & set(sys.argv)) | (len(sys.argv) <= 2):

    from dataphile.demos.auto_gui import Demo
    import numpy as np

    class GeneralFitNew(Demo):

        """Multiple Gaussian Features over a Polynomial Background."""

        def __init__(
            self,
            x,
            y,
            ax=None,
            linewidth=2,
            marker=None,
            EMCCD_=False,
            nb_sinusoid1D=0,
            nb_gaussians=0,
            nb_moffats=0,
            nb_voigt1D=0,
            background=True,
            exp=False,
            log=False,
            double_exp=False,
            linestyle="dotted",
            function=None,
            sigma=None,
        ):
            """Create synthetic dataset, plots, and AutoGUI."""
            from dataphile.statistics.regression.modeling import (
                Parameter,
                Model,
                CompositeModel,
                AutoGUI,
            )
            from dataphile.statistics.distributions import (
                linear1D,
                polynomial1D,
                gaussian1D,
                voigt1D,
                sinusoid1D,
            )
            from dataphile.statistics.distributions import uniform
            from scipy.optimize import curve_fit
            import matplotlib.pyplot as plt

            verboseprint(
                "nb_gaussians,  nb_moffats ,nb_voigt1D,nb_sinusoid = ",
                nb_gaussians,
                nb_moffats,
                nb_voigt1D,
                nb_sinusoid1D,
            )
            super().__init__(
                polynomial1D,
                [100, -0.01, -1e-5],
                (0, 2400),
                linspace=True,
                noise=0,
                samples=2400,
            )
            self.xdata = x[np.argsort(x)]
            self.ydata = y[np.argsort(x)]
            self.linestyle = linestyle
            self.marker = marker
            self.nb_gaussians = nb_gaussians
            self.nb_moffats = nb_moffats
            self.nb_voigt1D = nb_voigt1D
            self.log = log
            self.nb_sinusoid1D = nb_sinusoid1D
            self.background = background
            self.exp = exp
            self.double_exp = double_exp
            xdata_i = self.xdata
            ydata_i = self.ydata

            if ax is None:
                figure = plt.figure("Interactive fitting", figsize=(11, 6))
                self.figure = figure

                # create main plot
                self.ax = figure.add_axes([0.10, 0.30, 0.84, 0.66])
            else:
                self.ax = ax
            # if linestyle is not None:
            self.ax.scatter(
                xdata_i,
                ydata_i,
                color="black",
                marker=marker,
                label="data",
                lw=linewidth,
                alpha=0.8,
            )
            if linewidth > 0:
                self.ax.plot(
                    xdata_i, ydata_i, linestyle=linestyle, linewidth=linewidth,
                )
            x_inf, x_sup = self.ax.get_xlim()
            y_inf, y_sup = self.ax.get_ylim()
            self.ax.set_ylabel("y", labelpad=15)
            z = np.polyfit(x, y, deg=1)
            popt = np.poly1d(z)
            a = popt.coef[::-1][0]
            b = popt.coef[::-1][1]
            if self.background == 2:
                c, b, a = np.poly1d(np.polyfit(x, y, deg=2))
                fb, fa, fc = 10, 10, 5
                if b > 0:
                    boundsb = (b / fb - 1, fb * b + 1)
                else:
                    boundsb = (fb * b - 1, b / fb + 1)
                if a > 0:
                    boundsa = (a / fa, fa * a)
                else:
                    boundsa = (a * fa, a / fa)
                if c > 0:
                    boundsc = (-1 * fc * c, fc * c)
                else:
                    boundsc = (fc * c, -1 * fc * c)
            else:
                c = 0
                fb, fa, fc = 10, 10, 2
                if b > 0:
                    boundsb = (b / fb, fb * b)
                else:
                    boundsb = (fb * b, b / fb)
            boundsa = (a - (y.max() - y.min()), a + (y.max() - y.min()))
            boundsb = ((y.min() - a) / x.max(), (y.max() - a) / x.min())
            if boundsb[1] < boundsb[0]:
                boundsb = boundsb[::-1]
            Models = []
            background = np.nanmean(
                ydata_i[
                    (ydata_i < np.nanmean(ydata_i) + 1 * np.nanstd(ydata_i))
                    & (ydata_i > np.nanmean(ydata_i) - 1 * np.nanstd(ydata_i))
                ]
            )
            amp = np.nanmax(ydata_i) - background
            amp2 = np.nanmin(ydata_i) - background
            ampm = np.nanmax([abs(amp), abs(amp2)])
            nb_features = np.max(
                [self.nb_moffats, self.nb_gaussians, self.nb_voigt1D]
            )
            amps = (10 * [amp, amp2, (amp2 + amp) / 2])[:nb_features]
            x1, x2 = (
                xdata_i[np.argmax(ydata_i)],
                xdata_i[np.argmin(ydata_i)],
            )
            centers = (10 * [x1, x2, (x1 + x2) / 2])[:nb_features]
            xs, ys = find_maxima(x, y, conv=10, max_=True)
            xs, ys = (
                np.concatenate((xs, xs, xs)),
                np.concatenate((ys, ys, ys)),
            )
            amps, centers = (
                ys[:nb_features] - np.median(y),
                xs[:nb_features],
            )
            for i, amp, center in zip(range(self.nb_gaussians), amps, centers):
                Models.append(
                    Model(
                        gaussian1D,
                        Parameter(
                            value=amps[0],
                            bounds=(-1.5 * ampm, 1.5 * ampm),
                            label="Amplitude",
                        ),
                        Parameter(
                            value=center,
                            bounds=(
                                np.nanmin(xdata_i) - 2 * 2.35 * 2,
                                np.nanmax(xdata_i) + 2 * 2.35 * 2,
                            ),
                            label="Center",
                        ),
                        Parameter(
                            value=np.min([2, np.max(x) / 10]),
                            bounds=(1e-5, (np.nanmax(x) - np.nanmin(x)) / 2,),
                            label="Sigma",
                        ),
                        label="Gaussian%i" % (i),
                    )
                )
            for i, amp, center in zip(range(self.nb_moffats), amps, centers):
                Models.append(
                    Model(
                        moffat_1d,
                        Parameter(
                            value=amps[0],
                            bounds=(-1.5 * ampm, 1.5 * ampm),
                            label="Amplitude",
                        ),
                        Parameter(
                            value=center,
                            bounds=(np.nanmin(xdata_i), np.nanmax(xdata_i)),
                            label="Alpha",
                        ),
                        Parameter(
                            value=np.min([2, np.max(x) / 10]),
                            bounds=(1e-5, (np.nanmax(x) - np.nanmin(x)) / 2,),
                            label="Width",
                        ),
                        Parameter(
                            value=center,
                            bounds=(
                                np.nanmin(xdata_i) - 2 * 2.35 * 2,
                                np.nanmax(xdata_i) + 2 * 2.35 * 2,
                            ),
                            label="Center",
                        ),
                        label="Moffat%i" % (i),
                    )
                )

            for i, amp, center in zip(range(self.nb_voigt1D), amps, centers):
                Models.append(
                    Model(
                        voigt1D,
                        Parameter(
                            value=amps[0],
                            bounds=(-1.5 * ampm, 1.5 * ampm),
                            label="Amplitude",
                        ),
                        Parameter(
                            value=center,
                            bounds=(
                                np.nanmin(xdata_i) - 2 * 2.35 * 2,
                                np.nanmax(xdata_i) + 2 * 2.35 * 2,
                            ),
                            label="Center",
                        ),
                        Parameter(
                            value=np.min([2, np.max(x) / 10]),
                            bounds=(1e-5, (np.nanmax(x) - np.nanmin(x)) / 2,),
                            label="Sigma",
                        ),
                        Parameter(
                            value=np.min([2, np.max(x) / 10]),
                            bounds=(0, np.nanmax(xdata_i)),
                            label="Gamma",
                        ),
                        label="Voigt%i" % (i),
                    )
                )

            if (function != None) & (function != "None") & (function != "none"):
                from pyds9plugin.Macros.Fitting_Functions import functions
                from inspect import signature

                function_ = getattr(functions, function)
                names = list(signature(function_).parameters.keys())[1:]
                p_ = signature(function_).parameters
                params = [
                    Parameter(
                        value=np.mean(p_[p].default),
                        bounds=(p_[p].default[0], p_[p].default[-1]),
                        label=p,
                    )
                    for p in names
                ]
                Models.append(Model(function_, *params, label=function,))

                xsample = np.linspace(xdata_i.min(), xdata_i.max(), len(xdata_i))
                xsample = np.linspace(
                    xdata_i.min(), xdata_i.max(), len(xdata_i) * 100
                )
            else:
                xsample = np.linspace(
                    xdata_i.min(), xdata_i.max(), len(xdata_i) * 100
                )
            if self.exp:
                Models.append(
                    Model(
                        exponential_1d,
                        Parameter(
                            value=0,
                            bounds=(0, 1.5 * np.nanmax(ydata_i)),
                            label="amplitude",
                        ),
                        Parameter(
                            value=100,
                            bounds=(np.nanmin(xdata_i), np.nanmax(xdata_i),),
                            label="Length",
                        ),
                        label="Exponential decay",
                    )
                )
            if self.log:
                Models.append(
                    Model(
                        lambda x, b, c: b * np.log(x - c),
                        Parameter(
                            value=1,
                            bounds=(-np.nanmax(ydata_i), np.nanmax(ydata_i),),
                            label="factor",
                        ),
                        Parameter(
                            value=-1,
                            bounds=(-np.nanmax(xdata_i), np.nanmin(xdata_i),),
                            label="Length",
                        ),
                        label="Logarithmic background",
                    )
                )

            if self.double_exp:
                end = 10000
                p0 = [ydata_i.max() - ydata_i.min(), 10, 0.5, 5]
                try:
                    popt, pcov = curve_fit(
                        double_exponential, ydata_i[:end], ydata_i[:end], p0=p0,
                    )
                except (RuntimeError, TypeError) as e:
                    verboseprint(e)
                    popt = p0
                Models.append(
                    Model(
                        double_exponential,
                        Parameter(
                            value=popt[0],
                            bounds=(-abs(popt[0]), abs(popt[0])),
                            label="Amplitude tot",
                        ),
                        Parameter(
                            value=popt[1], bounds=(0, len(y)), label="Length 1",
                        ),
                        Parameter(
                            value=popt[2],
                            bounds=(-abs(popt[0]), abs(popt[0])),
                            label="Amp2/Amp1",
                        ),
                        Parameter(
                            value=popt[2], bounds=(0, len(y)), label="Length 2",
                        ),
                        label="Double Exponential",
                    )
                )
            if self.background == 0:
                ymin_ = (
                    1.1 * np.nanmin(y)
                    if np.nanmin(y) < 0
                    else 0.9 * np.nanmin(y)
                )
                ymax_ = (
                    1.1 * np.nanmax(y)
                    if np.nanmax(y) > 0
                    else 0.9 * np.nanmax(y)
                )
                Models.append(
                    Model(
                        uniform,
                        Parameter(
                            value=np.nanmedian(y),
                            bounds=(ymin_, ymax_),
                            label="Offset",
                        ),
                        label="Background",
                    )
                )
            if self.background == 1:
                Models.append(
                    Model(
                        linear1D,
                        Parameter(value=a, bounds=boundsa, label="scale"),
                        Parameter(value=b, bounds=boundsb, label="slope"),
                        label="Background",
                    )
                )
            # A GARDER
            #        if self.background==1:
            #            Models.append(Model(linear1d_centered,
            #                  Parameter(value=a, bounds=boundsa, label='scale'),
            #                  Parameter(value=b, bounds=boundsb, label='slope'),
            #                  Parameter(value=(x.min()+x.max())/2,
            # bounds=((x.min()+x.max())/2-1e-10,(x.min()+x.max())/2+1e-10), label='center'),
            # ,uncertainty=1 # bounds=((x.min()+x.max())/2-1,(x.min()+x.max())/2+1)
            #                  label='Background'))
            if self.background == 2:
                Models.append(
                    Model(
                        polynomial1D,
                        Parameter(
                            value=np.nanmean(y),
                            bounds=(np.nanmin(y), np.nanmax(y)),
                            label="scale",
                        ),
                        Parameter(value=0, bounds=(-0.5, 0.5), label="slope"),
                        Parameter(
                            value=0, bounds=(-5e-5, 5e-5), label="gradient"
                        ),
                        label="Background",
                    )
                )
            model = CompositeModel(*Models, label="General fit")
            curves = []
            for modeli in Models:
                curves.append(a)
            # print(model)
            # print(Models)
            # print(model(xsample))
            (model_curve,) = self.ax.plot(
                xsample, model(xsample), color="steelblue", label="model"
            )
            self.ax.legend(loc="upper right")
            self.ax.set_xlim((x_inf, x_sup))
            self.ax.set_ylim((y_inf, y_sup))
            verboseprint("autogui")
            verboseprint([model_curve] + curves)
            gui = AutoGUI(
                model,
                [model_curve],
                bbox=[0.20, 0.07, 0.75, 0.17],
                slider_options={"color": "steelblue"},
                data=(xdata_i, ydata_i),
            )
            verboseprint("model")
            self.model = Models
            verboseprint("gui")
            self.gui = gui

    class GeneralFitFunction(Demo):

        """Multiple Gaussian Features over a Polynomial Background."""

        def __init__(
            self,
            x,
            y,
            function,
            ranges,
            zdata_i=None,
            ax=None,
            plot_="Linear",
            linewidth=2,
            marker=None,
            linestyle="dotted",
            n=100,
            names=None,
        ):
            """Create synthetic dataset, plots, and AutoGUI."""
            from dataphile.statistics.regression.modeling import (
                Parameter,
                Model,
                AutoGUI,
            )
            from dataphile.statistics.distributions import polynomial1D
            import matplotlib.pyplot as plt

            super().__init__(
                polynomial1D,
                [100, -0.01, -1e-5],
                (0, 2400),
                linspace=True,
                noise=0,
                samples=2400,
            )
            self.xdata = x
            self.ydata = y
            self.linestyle = linestyle
            self.marker = marker
            self.ranges = ranges
            self.function = function

            xdata_i = self.xdata
            ydata_i = self.ydata
            if ax is None:
                if (plot_ == "Polar") | (plot_ == "PolarLog"):
                    figure = plt.figure(figsize=(8, 9))
                    self.figure = figure
                    self.ax = figure.add_axes(
                        [0.10, 0.30, 0.84, 0.66], projection="polar"
                    )
                elif plot_ == "Plot3D":
                    figure = plt.figure(figsize=(8, 9))
                    self.figure = figure
                    self.ax = figure.add_axes(
                        [0.10, 0.30, 0.84, 0.66], projection="3d"
                    )
                else:
                    figure = plt.figure(figsize=(11, 6))
                    self.figure = figure
                    self.ax = figure.add_axes([0.10, 0.30, 0.84, 0.66])
                    self.ax.set_ylabel("y", labelpad=15)
            else:
                self.ax = ax
            if plot_ == "Plot3D":
                self.ax.scatter3D(
                    xdata_i,
                    ydata_i,
                    zdata_i,
                    c=zdata_i,
                    color="black",
                    marker=marker,
                    label="data",
                    lw=linewidth,
                    alpha=0.8,
                    cmap="Greens",
                )
            elif plot_ == "LinLog":
                self.ax.set_yscale("log")
            elif plot_ == "LogLog":
                self.ax.set_xscale("log")
                self.ax.set_yscale("log")
            elif plot_ == "PolarLog":
                self.ax.set_ylim((1, np.log10(y.max())))
                self.ax.set_yscale("log")

            else:
                self.ax.scatter(
                    xdata_i,
                    ydata_i,
                    color="black",
                    marker=marker,
                    label="data",
                    lw=linewidth,
                    alpha=0.8,
                )
            if plot_ == "Polar":
                self.ax.set_rticks(self.ax.get_yticks()[1::2])

            if linewidth > 0:
                self.ax.plot(
                    xdata_i, ydata_i, linestyle=linestyle, linewidth=linewidth,
                )
            x_inf, x_sup = self.ax.get_xlim()
            y_inf, y_sup = self.ax.get_ylim()
            xsample = np.linspace(xdata_i.min(), xdata_i.max(), len(xdata_i) * n)
            parameters = []
            if names is None:
                names = ["a%i" % (i) for i in range(len(self.ranges))]
            for i, rangei in enumerate(self.ranges):
                xmin, xmax = np.array(rangei.split(","), dtype=float)
                parameters.append(
                    Parameter(
                        value=(xmax + xmin) / 2,
                        bounds=(min(xmin, xmax), max(xmin, xmax)),
                        label=names[i],
                    )
                )
            model = Model(self.function, *parameters, label="Function")
            (model_curve,) = self.ax.plot(
                xsample, model(xsample), color="steelblue", label="model"
            )
            self.ax.legend(loc="upper right")
            self.ax.set_xlim((x_inf, x_sup))
            self.ax.set_ylim((y_inf, y_sup))
            try:
                gui = AutoGUI(
                    model,
                    [model_curve],
                    bbox=[0.20, 0.07, 0.75, 0.17],
                    slider_options={"color": "steelblue"},
                    data=(xdata_i, ydata_i),
                )
                self.model = model
                self.gui = gui
            except ZeroDivisionError:
                pass


def find_maxima(x, y, conv=10, max_=True):
    """Find maxima in an array after convolution
    """
    import numpy as np

    a = np.convolve(y, np.ones(conv) / conv, mode="same")
    if max_:
        maxim = np.r_[True, a[1:] - a[:-1] > 0] & np.r_[a[:-1] - a[1:] > 0, True]
    else:
        maxim = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    xs = x[maxim][::-1]
    ys = y[np.arange(len(y))[maxim][::-1]]
    return xs[np.argsort(ys)][::-1], ys[np.argsort(ys)][::-1]


def exponential_1d(x, amplitude, stdev):
    """A one dimeional gaussian distribution.
       = amplitude * exp(-0.5 (x - center)**2 / stdev**2)
    """
    import numpy as np

    return amplitude * np.exp(-x / stdev)


def schechter_(x, phi=1e-3, sfr_=6, alpha=-1.6):
    """ Schecter function for luminosity type function
    """
    import numpy as np

    return np.log10(
        phi
        * 2.35
        * np.power(10, (alpha + 1) * (x - np.log10(sfr_)))
        * np.exp(-np.power(10, x - np.log10(sfr_)))
    )


def schechter(x, phi=3.6e-3, m=19.8, alpha=-1.6):
    """ Schecter function for luminosity type function
    """
    import numpy as np

    y = np.log10(
        0.4
        * np.log(10)
        * phi
        * 10 ** (0.4 * (m - x) * (alpha + 1))
        * (np.e ** (-pow(10, 0.4 * (m - x))))
    )
    return y[::-1]


def double_exponential(x, amplitude, stdev, amp2, stdev2):
    """A one dimensional gaussian distribution.
       = amplitude * exp(-0.5 (x - center)**2 / stdev**2)
    """
    return amplitude * (np.exp(-x / stdev) + amp2 * np.exp(-x / stdev2))


def polynom2deg(x, intercept, slope, gradient):
    """A one dimensional line."""
    return intercept + slope * x + gradient * x * x


def linear1d_centered(x, intercept, slope, x0=0):
    """A one dimensional line."""
    return slope * (x - x0) + intercept  # origine


def fit_ds9_plot(xpapoint=None, argv=[]):

    """Fit interactively any DS9 plot or catalog by different functions
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    import numpy as np
    from astropy.table import Table

    exp = (False,)
    double_exp = (False,)
    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-b",
        "--background",
        default="1",
        help="Background to fit",
        type=str,
        choices=[
            "Constant",
            "None",
            "slope",
            "Exponential",
            "DoubleExponential",
            "polynom",
            "Logarithmic",
        ],
    )
    parser.add_argument(
        "-g",
        "--gaussians",
        default=1,
        help="Number of gaussian features to fit",
        metavar="",
        type=str,
        choices=["0", "1", "2", "3", "4", "5"],
    )
    parser.add_argument(
        "-m",
        "--moffats",
        default=1,
        help="Number of moffats features to fit",
        metavar="",
        type=str,
        choices=["0", "1", "2", "3", "4", "5"],
    )
    parser.add_argument(
        "-v",
        "--voights",
        default=1,
        help="Number of voights features to fit",
        metavar="",
        type=str,
        choices=["0", "1", "2", "3", "4", "5"],
    )
    parser.add_argument(
        "-o",
        "--other_features",
        default=1,
        help="Other features to fit",
        metavar="",
        type=str,
        choices=[
            "None",
            "Import-from-Macros",
            "User-defined-interactively",
            "Schechter",
            "Double-Schechter",
            "User-defined",
        ],
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)
    axis, function, nb_sinusoid = "x", "none", 0
    exp, double_exp, log = False, False, False

    if args.background.lower() == "slope":
        bckgd = 1
    elif args.background.lower() == "none":
        bckgd = -1
    elif args.background.lower() == "polynom":
        bckgd = 2
    else:
        bckgd = 0
    try:
        d.get("plot")
    except TypeError:
        raise_create_plot(d)
        sys.exit()
    if os.path.exists(args.path):
        cat = read_v(args.path)
        x, y = cat[cat.colnames[0]], cat[cat.colnames[1]]
    else:
        if d.get("plot") != "":
            name = ""
            d.set("plot %s save /tmp/test.dat" % (name))
            x_scale = d.get("plot %s axis x log" % (name))
            y_scale = d.get("plot %s axis y log" % (name))
            tab = Table.read("/tmp/test.dat", format="ascii")
            x, y = tab["col1"], tab["col2"]
            xmin = d.get("plot %s axis x min" % (name))
            xmax = d.get("plot %s axis x max" % (name))
            ymin = d.get("plot %s axis y min" % (name))
            ymax = d.get("plot %s axis y max" % (name))
            xmin = float(xmin) if xmin != "" else -np.inf
            xmax = float(xmax) if xmax != "" else np.inf
            ymin = float(ymin) if ymin != "" else -np.inf
            ymax = float(ymax) if ymax != "" else np.inf  # TODO delete duplicate
            mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
            x, y = x[mask], y[mask]
            if x_scale == "yes":
                x = np.log10(x)
            if y_scale == "yes":
                y = np.log10(y)
            index = (np.isfinite(y)) & (np.isfinite(x))
            x, y = x[index], y[index]
        else:
            raise_create_plot(d)
            sys.exit()

    if np.nanmean(y[-10:]) > np.nanmean(y[:10]):
        y = y[::-1]

    if args.background.lower() == "exponential":
        exp = True
    if args.background.lower() == "doubleexponential":
        double_exp = True
    if args.background.lower() == "logarithmic":
        log = True
    if args.other_features == "Import-from-Macros":
        from inspect import getmembers, isfunction
        from pyds9plugin.Macros.Fitting_Functions import functions

        possible_functions = [f[0] for f in getmembers(functions, isfunction)]
        while function not in possible_functions:
            function = get(
                d,
                """What function do you want to import from pyds9plugin.Macros.
                   Fitting_Functions? Your choices are: %s. Do not hesitate
                   to write your own!"""
                % (possible_functions),
            )
            # bckgd = -1
    if args.other_features == "User-defined-interactively":
        gui = interactiv_manual_fitting(
            x,
            y,
            initial="a*median(ydata)+b*ptp(ydata)*exp(-(x-c*x[argmax(ydata)])**2/len(ydata)/d)",
        )

    else:
        gui = GeneralFitNew(
            x,
            y,
            nb_gaussians=int(args.gaussians),
            nb_moffats=int(args.moffats),
            background=int(bckgd),
            nb_voigt1D=int(args.voights),
            exp=exp,
            log=log,
            double_exp=double_exp,
            marker=".",
            linestyle="dotted",
            linewidth=1,
            function=function,
        )
    rax = plt.axes([0.01, 0.8, 0.1, 0.15], facecolor="None")
    for edge in "left", "right", "top", "bottom":
        rax.spines[edge].set_visible(False)
    scale = CheckButtons(rax, ["log"])

    def scalefunc(label):
        if gui.ax.get_yscale() == "linear":
            gui.ax.set_yscale("log")
        elif gui.ax.get_yscale() == "log":
            gui.ax.set_yscale("linear")
        gui.figure.canvas.draw_idle()

    scale.on_clicked(scalefunc)
    # gui.ax.set_title(os.path.basename(get_filename(d)))
    plt.show()
    return


def interactiv_manual_fitting(
    xdata,
    ydata,
    initial="a+b*max(ydata)*exp(-(x-c*x[argmax(ydata)])**2/len(ydata)/d)",
    dict_={},
):
    """ Creates an interactive plot to fit a model on the data
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, TextBox
    from dataphile.graphics.widgets import Slider
    from matplotlib.widgets import CheckButtons

    n = len(xdata)
    lims = np.array([0, 2])

    np_function = {a: getattr(np, a) for a in dir(np)}
    np_function.update(dict_)
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.25)
    x = np.linspace(np.nanmin(xdata), np.nanmax(xdata), n)
    dict_values = {
        "function": initial,
        "a": 1,
        "b": 1,
        "c": 1,
        "d": 1,
        "x": x,
        "xdata": xdata,
        "ydata": ydata,
    }

    (l,) = plt.plot(x, eval(initial, np_function, dict_values), lw=2)

    (datal,) = plt.plot(xdata, ydata, ".", c="black")
    ax.margins(x=0)
    rax = plt.axes([0.04, 0.85, 0.15, 0.15], facecolor="None")
    raxx = plt.axes([0.93, 0.17, 0.15, 0.15], facecolor="None")
    data_box = plt.axes([0.8, 0.75, 0.15, 0.15], facecolor="None")
    bounds_box = plt.axes([0.87, -0.029, 0.15, 0.15], facecolor="None")
    axbox = plt.axes([0.1, 0.025, 0.65, 0.04])

    button = Button(
        plt.axes([0.77, 0.025, 0.1, 0.04]),
        "Fit",
        color="white",
        hovercolor="0.975",
    )
    delete_button = Button(
        plt.axes([0.72, 0.025, 0.04, 0.04]),
        "x",
        color="white",
        hovercolor="0.975",
    )

    for edge in "left", "right", "top", "bottom":
        rax.spines[edge].set_visible(False)
        raxx.spines[edge].set_visible(False)
        data_box.spines[edge].set_visible(False)
        bounds_box.spines[edge].set_visible(False)
    scale = CheckButtons(rax, ["log"])
    scalex = CheckButtons(raxx, ["log"])
    data_button = CheckButtons(data_box, ["Data"], [True])
    bounds_button = CheckButtons(bounds_box, ["Bounds"], [False])

    def scalefunc(label):
        print(scale)
        if ax.get_yscale() == "linear":
            try:
                ax.set_yscale("log")
            except Exception:
                ax.set_ylim((np.nanmin()))
                ax.set_ylim(ymin=np.nanmin(ydata[ydata > 0]))
                ax.set_yscale("log")
        elif ax.get_yscale() == "log":
            ax.set_yscale("linear")
        fig.canvas.draw_idle()

    def scalefuncx(label):
        print(scale)
        if (ax.get_xscale() == "linear") & (dict_values["x"] > 1).any():
            ax.set_xscale("log")
            try:
                ax.set_xscale("log")
            except Exception:
                ax.set_xlim((np.nanmin()))
                ax.set_xlim(xmin=np.nanmin(xdata[xdata > 0]))
                ax.set_xscale("log")
        fig.canvas.draw_idle()

    def load_data(label):
        if data_button.get_status()[0]:
            datal.set_marker(".")
        else:
            datal.set_marker(None)
        fig.canvas.draw_idle()

    def submit(text):
        x = dict_values["x"]
        a = dict_values["a"]
        b = dict_values["b"]
        c = dict_values["c"]
        d = dict_values["d"]
        dict_values["function"] = text
        ax.set_title("", color="red")
        try:
            ydata = eval(text, np_function, dict_values)
        except Exception as e:
            ax.set_title(e, color="red")
        l.set_ydata(ydata)
        ax.set_ylim(np.min(ydata), np.max(ydata))
        plt.draw()
        return text

    def update(val):
        try:
            a = b_a.value
            b = b_b.value
            c = b_c.value
            d = b_d.value
        except AttributeError:
            a = b_a.val
            b = b_b.val
            c = b_c.val
            d = b_d.val
        text = dict_values["function"]
        ax.set_title("", color="red")
        try:
            l.set_ydata(eval(text, np_function, dict_values))
        except Exception as e:
            print(e)
            l.set_ydata(eval(initial, np_function, dict_values))
            ax.set_title(e, color="red")

        fig.canvas.draw_idle()
        dict_values["a"] = a
        dict_values["b"] = b
        dict_values["c"] = c
        dict_values["d"] = d
        return

    scale.on_clicked(scalefunc)
    scalex.on_clicked(scalefuncx)
    data_button.on_clicked(load_data)

    text_box = TextBox(
        axbox, "f(x) = ", initial=initial, color="white", hovercolor="0.975"
    )
    text_box.on_submit(submit)

    b_a = Slider(
        figure=fig,
        location=[0.1, 0.17, 0.8, 0.03],
        label="a",
        bounds=lims,
        init_value=np.mean(lims),
    )
    b_b = Slider(
        figure=fig,
        location=[0.1, 0.14, 0.8, 0.03],
        label="b",
        bounds=lims,
        init_value=np.mean(lims),
    )
    b_c = Slider(
        figure=fig,
        location=[0.1, 0.11, 0.8, 0.03],
        label="c",
        bounds=lims,
        init_value=np.mean(lims),
    )
    b_d = Slider(
        figure=fig,
        location=[0.1, 0.08, 0.8, 0.03],
        label="d",
        bounds=lims,
        init_value=np.mean(lims),
    )

    b_a.on_changed(update)
    b_b.on_changed(update)
    b_c.on_changed(update)
    b_d.on_changed(update)

    def returnf(text):
        dict_values["function"] = text

        x = dict_values["x"]
        a = dict_values["a"]
        b = dict_values["b"]
        c = dict_values["c"]
        d = dict_values["d"]
        y = eval(text, np_function, dict_values)
        l.set_ydata(y)
        ax.set_ylim(np.min(y), np.max(y))
        plt.draw()
        print("y ==========", y)
        return text

    def reset(event):
        b_a.reset()
        b_b.reset()
        b_c.reset()
        b_d.reset()

    def fit(event):
        from scipy.optimize import curve_fit

        def f(x, a, b, c, d):
            y_fit = 0
            dict_tot = {}
            for dd in [
                {
                    "a": a,
                    "b": b,
                    "c": c,
                    "d": d,
                    "x": x,
                    "y_fit": y_fit,
                    "xdata": xdata,
                    "ydata": ydata,
                },
                np_function,
            ]:
                dict_tot.update(dd)
            exec("y_fit = " + dict_values["function"], globals(), dict_tot)
            y_fit = dict_tot["y_fit"]
            return y_fit

        x = dict_values["x"]
        a = dict_values["a"]
        b = dict_values["b"]
        c = dict_values["c"]
        d = dict_values["d"]
        verboseprint("p0 = ", [a, b, c, d])
        xmin, xmax = ax.get_xlim()
        ax.set_title("", color="red")

        if bounds_button.get_status()[0]:
            bounds = (
                (lims.min(), lims.min(), lims.min(), lims.min()),
                (lims.max(), lims.max(), lims.max(), lims.max()),
            )
        else:
            bounds = (
                (-np.inf, -np.inf, -np.inf, -np.inf),
                (np.inf, np.inf, np.inf, np.inf),
            )
        verboseprint("bounds = ", bounds)
        try:
            popt, pcov = curve_fit(
                f, xdata, ydata, p0=[a, b, c, d], bounds=bounds
            )
        except Exception as e:
            ax.set_title(e, color="red")

        verboseprint("Fitting, f(x) = ", dict_values["function"])
        verboseprint("p0 = ", [a, b, c, d])
        verboseprint("Fit : ", popt)
        plt.figtext(
            0.55,
            0.93,
            "Fit: %s, std=%0.3f" % (np.around(popt, 2), np.sum(np.diag(pcov))),
            bbox={
                "facecolor": "black",
                "alpha": 1,
                "color": "white",
                "pad": 10,
            },
        )
        l.set_ydata(f(x, *popt))
        plt.draw()
        b_a.widget.set_val(popt[0])
        b_b.widget.set_val(popt[1])
        b_c.widget.set_val(popt[2])
        b_d.widget.set_val(popt[3])

    def delete(event):
        text_box.set_val("")
        return

    button.on_clicked(fit)
    delete_button.on_clicked(delete)

    def onclick(event):
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, n)
        a = dict_values["a"]
        b = dict_values["b"]
        c = dict_values["c"]
        d = dict_values["d"]
        dict_values["x"] = x
        text = dict_values["function"]
        y = eval(text, np_function, dict_values)
        dict_values["y"] = y
        l.set_xdata(x)
        l.set_ydata(y)
        ymax = 1.1 * np.nanmax(y) if np.nanmax(y) > 0 else 0.9 * np.nanmax(y)
        ymin = 0.9 * np.nanmin(y) if np.nanmin(y) > 0 else 1.1 * np.nanmin(y)
        ymax2 = (
            1.1 * np.nanmax(datal.get_ydata())
            if np.nanmax(datal.get_ydata()) > 0
            else 0.9 * np.nanmax(datal.get_ydata())
        )
        ymin2 = (
            0.9 * np.nanmin(datal.get_ydata())
            if np.nanmin(datal.get_ydata()) > 0
            else 1.1 * np.nanmin(datal.get_ydata())
        )
        if datal.get_marker() is not None:
            try:
                ax.set_ylim((np.nanmin([ymin, ymin2]), np.nanmax([ymax, ymax2])))
            except Exception:
                pass
        else:
            ax.set_ylim((ymin, ymax))
        plt.draw()
        return

    fig.canvas.mpl_connect("draw_event", onclick)
    plt.show()


def manual_fitting(
    xpapoint=None,
    initial="a*median(ydata)+b*ptp(ydata)*exp(-(x-c*x[argmax(ydata)])**2/len(ydata)/d)",
    argv=[],
):
    """Manual fitting on ds9 plot or saved table
    """
    from astropy.table import Table
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="Path of the 2 columns table to fit",
        metavar="",
    )
    args = parser.parse_args_modif(argv)
    d = DS9n(args.xpapoint)
    if isinstance(d, FakeDS9) is False:
        try:
            d.get("plot")
        except TypeError:
            raise_create_plot(d)
            sys.exit()

        if d.get("plot") != "":
            name = ""
            verboseprint(name)
            d.set("plot %s save /tmp/test.dat" % (name))
            tab = Table.read("/tmp/test.dat", format="ascii")
            x, y = tab["col1"], tab["col2"]
            xmin = d.get("plot %s axis x min" % (name))
            xmax = d.get("plot %s axis x max" % (name))
            ymin = d.get("plot %s axis y min" % (name))
            ymax = d.get("plot %s axis y max" % (name))
            xmin = float(xmin) if xmin != "" else -np.inf
            xmax = float(xmax) if xmax != "" else np.inf
            ymin = float(ymin) if ymin != "" else -np.inf
            ymax = float(ymax) if ymax != "" else np.inf
            mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax)
            x, y = x[mask], y[mask]
            index = (np.isfinite(y)) & (np.isfinite(x))
            x, y = x[index], y[index]

        else:
            x = np.linspace(0, 10, 1000)
            y = np.nan * x
            initial = "a+b*exp(-(x-c)**2/d)"
    elif os.path.exists(args.path):
        cat = read_v(args.path)
        x, y = cat[cat.colnames[0]], cat[cat.colnames[1]]
        index = (np.isfinite(y)) & (np.isfinite(x))
        x, y = x[index], y[index]
    else:
        x = np.linspace(0, 10, 1000)
        y = np.nan * x
        initial = "a+b*exp(-(x-c)**2/d)"

    # if np.nanmean(y[-10:]) > np.nanmean(y[:10]):
    #     y = y[::-1]
    interactiv_manual_fitting(x, y, initial=initial)
    return


def open_table(path):
    """ Open a table depending on its extension
    """
    from astropy.table import Table

    if os.path.isfile(path) is False:
        raise SyntaxError("Please verify your path")

    if ".csv" in path:
        tab = Table.read(path, format="csv")
    if ".fits" in path:
        tab = Table.read(path)
    else:
        try:
            tab = Table.read(path, format="ascii")
        except Exception:
            tab = Table.read(path, format="csv")
    try:
        return tab
    except NameError:
        raise SyntaxError(
            "Could not open the table, verify it is a csv/fits or ascii table."
        )


def interactive_plotter(
    xpapoint=None,
    plot_="Linear",
    path=None,
    function=lambda x, A=1, s=2, B=3: A * x ** 2 + s * x + B,
    ranges=None,
    names=None,
    argv=[],
):
    """Fit background 1d with different features
    """
    import re
    from matplotlib.widgets import RadioButtons
    import numpy as np

    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv, required=False)
    xrange = [-10, 10]

    y = 0
    if len(sys.argv) > 3:
        function = sys.argv[-4]
        xmin, xmax = np.array(sys.argv[-3].split(","), dtype=float)
        plot_ = sys.argv[-2]
        path = sys.argv[-1]
    else:
        xmin, xmax = xrange
    if type(function) == str:
        ranges = re.findall("\[(.*?)\]", function)
        number = np.sum(["a%i" % (i) in function for i in range(10)])
        print(ranges)
        if len(ranges) == 0:
            ranges = []
            for i in range(number):
                answer = ds9entry(
                    args.xpapoint,
                    "Range of a%i parameter in %s. eg: -3,5"
                    % (i, "f(x) = " + function),
                    quit_=False,
                )
                ranges.append(answer)

        def real_function(
            x, *args, function_new="y = " + re.sub("[\[].*?[\]]", "", function)
        ):  #
            y = 0
            a0, a1, a2, a3, a4, a5, a6, a7 = np.zeros(8)
            np_function = {a: getattr(np, a) for a in dir(np)}
            ldict = {"x": x, "y": y}
            dict2 = {"a%i" % (i): val for i, val in enumerate(args)}
            dict_tot = {}
            for d in [ldict, dict2, np_function]:
                dict_tot.update(d)
            exec(function_new, globals(), dict_tot)
            y = dict_tot["y"]
            if "bins" in function_new:
                return np.pad(y[0], (0, 1), "constant")
            else:
                return y

    else:
        real_function = function
        if ranges is None:
            ranges = [
                "%s,%s" % (default - 1, default + 1)
                for default in real_function.__defaults__
            ]
        names = function.__code__.co_varnames[1:]

    args = [
        np.mean(np.array(rangei.split(","), dtype=float)) for rangei in ranges
    ]
    if (path != "") & (path is not None):
        cat = open_table(path)
        x, y = cat[cat.colnames[0]], cat[cat.colnames[1]]
        n = 100
    else:
        if type(function) == str:
            if "bins" in function:
                x = np.linspace(xmin, xmax, 1000)
                n = 1
        else:
            x = np.linspace(xmin, xmax, 100)
            n = 100
        y = real_function(x, *args)

    gui = GeneralFitFunction(
        x,
        y,
        function=real_function,
        ranges=ranges,
        marker=".",
        plot_=plot_,
        linestyle="dotted",
        linewidth=0,
        n=n,
        names=names,
    )
    import matplotlib.pyplot as plt

    rax = plt.axes([0.01, 0.8, 0.1, 0.15], facecolor="None")
    for edge in "left", "right", "top", "bottom":
        rax.spines[edge].set_visible(False)
    scale = RadioButtons(rax, ("linear", "log"), active=0)

    def scalefunc(label):
        gui.ax.set_yscale(label)
        gui.ax.set_ylim(ymin=0)
        gui.figure.canvas.draw_idle()

    scale.on_clicked(scalefunc)
    gui.ax.set_title(function)
    plt.show()
    return


def main_coupon(
    file_in_name,
    file_out_name,
    ext,
    ext_seg,
    mag_zp,
    sub,
    aper_size,
    plot,
    file_seg_name,
    type,
    nomemmap,
    n_aper,
):
    """ Computes depth of an image based on aperture photometry
    """

    sigma = [5.0, 10.0]
    image_tmp, pix_scale, mag_zp = get_data_coupon(
        file_in_name, ext, mag_zp, sub, nomemmap
    )
    if type == "image" or type == "var":
        image = image_tmp
    elif type == "weight":
        image = 1.0 / image_tmp
    elif type == "rms":
        image = image_tmp * image_tmp
    else:
        raise ValueError('image type "{0}" not recognized'.format(type))

    seg = None
    if file_seg_name is not None:
        seg, seg_pix_scale, seg_mag_zp = get_data_coupon(
            file_seg_name, ext_seg, mag_zp, sub, nomemmap
        )
    flux, n_aper_used, result = throw_apers(
        image, pix_scale, aper_size, n_aper, seg, type, sub
    )

    d, d_err, flux_std = depth(flux, mag_zp, sigma, type)

    print_d = "{0:3.2f}".format(d[0])
    print_sigma = "{0:3.2f}".format(sigma[0])
    for i in range(1, len(sigma)):
        print_d += " {0:3.2f}".format(d[i])
        print_sigma += " {0:3.2f}".format(sigma[i])
    title = """{0}:\n Depth in {1:3.2f}" diam. apertures: {2:s} ({3:s} sigmas)
               +/- {4:3.2f}. flux_std = {5:3.2f}""".format(
        os.path.basename(file_in_name),
        aper_size,
        print_d,
        print_sigma,
        d_err[0],
        flux_std,
    )

    if plot:
        import matplotlib.pyplot as plt

        plot_histo(flux, flux_std, aper_size, title)
        plt.savefig(file_in_name[:-5] + "_depth.png")
        plt.show()
    return {
        "depth": d,
        "depth_eroor": d_err,
        "flux_std": flux_std,
        "n_aper_used": n_aper_used,
    }


def get_data_coupon(file_in_name, ext, mag_zp, sub, nomemmap):
    """
    Open a fits files and return attributes
    """
    import warnings
    from astropy.io import fits
    import astropy.wcs as wcs
    import numpy as np

    if nomemmap:
        fileIn = fits.open(file_in_name, memmap=False)
    else:
        fileIn = fits.open(file_in_name, memmap=True)

    image = fileIn[ext].data

    try:
        w = wcs.WCS(fileIn[ext].header)
        pix_scale = wcs.utils.proj_plane_pixel_scales(w)[0] * 3600.0
    except Exception:
        warnings.warn(
            "Using CD1_1: {0}".format(abs(fileIn[ext].header["CD1_1"])),
            RuntimeWarning,
        )
        pix_scale = abs(fileIn[ext].header["CD1_1"]) * 3600.0

    if False:
        exit(-1)

    if mag_zp is None:
        mag_zp = get_mag_zp(fileIn[ext].header)

    if sub is not None:
        if (sub[0] - 1 < 0) | (sub[1] - 1 > np.shape(image)[0] - 1):
            raise ValueError(
                "sub x-coordinates ({0}, {1}) exceed image ({2}, {3})".format(
                    sub[0], sub[1], 1, np.shape(image)[0]
                )
            )
        if (sub[2] - 1 < 0) | (sub[3] - 1 > np.shape(image)[1] - 1):
            raise ValueError(
                "sub y-coordinates ({0}, {1}) exceed image ({2}, {3})".format(
                    sub[2], sub[3], 1, np.shape(image)[1]
                )
            )
        ylim = [sub[0] - 1, sub[1] - 1]
        xlim = [sub[2] - 1, sub[3] - 1]
    else:
        xlim = [0, np.shape(image)[0] - 1]
        ylim = [0, np.shape(image)[1] - 1]
    if False:
        file_out = fits.PrimaryHDU(
            image[xlim[0] : xlim[1], ylim[0] : ylim[1]],
            header=fileIn[ext].header,
        )
        file_out.writeto(file_in_name + ".sub", clobber=True)
    return image[xlim[0] : xlim[1], ylim[0] : ylim[1]], pix_scale, mag_zp


def throw_apers(image, pix_scale, aper_size, n_aper, seg, type, sub_bkg):
    """
    Takes an image + attributes and throw n_aper
    apertures with random positions
    """

    """
    best way to proceed is to feed with segmantation map from
    sextractor and set all contiguous pixels above background as NaN
    """
    from photutils import CircularAperture
    from photutils import aperture_photometry
    import numpy as np

    if sub_bkg:
        image -= np.mean(image)

    y_ran = (np.shape(image)[0] - 1.0 - 0.0) * np.random.random(int(n_aper))
    x_ran = (np.shape(image)[1] - 1.0 - 0.0) * np.random.random(int(n_aper))

    aperture = CircularAperture(np.array([x_ran, y_ran]), r=aper_size / 2)
    var = [
        np.nanvar(a.to_mask().multiply(image)[a.to_mask().multiply(image) > 0])
        for a in aperture
    ]
    median = [
        np.nanmedian(
            ap.to_mask().multiply(image)[ap.to_mask().multiply(image) > 0]
        )
        for ap in aperture
    ]
    result = aperture_photometry(image, aperture)
    flux = result["aperture_sum"]
    result["Var"] = var
    result["Median"] = median

    if seg is not None:
        eps = 1.0e-3
        result = aperture_photometry(seg, aperture)
        flux_seg = result["aperture_sum"]

        select = (flux_seg < eps) & (np.isfinite(flux))
    else:
        select = np.isfinite(flux)

        aper_volume = aper_size ** 2 * np.pi / (60.0 * 60.0) ** 2
        n_aper_used = len(flux[select])
        # verboseprint(
        #     """{0:d} / {1:d} apertures used. {2:3.4f} /
        #                 {3:3.4f} (deg2) surface used""".format(
        #         int(n_aper_used),
        #         int(n_aper),
        #         aper_volume * float(n_aper_used),
        #         np.shape(image)[0]
        #         * np.shape(image)[1]
        #         * pix_scale ** 2
        #         / (60.0 * 60.0) ** 2,
        #     )
        # )
    if False:
        np.savetxt(
            "apertures.txt",
            np.column_stack((x_ran[select], y_ran[select], flux[select])),
            header="x y flux",
        )
    return flux[select], n_aper_used, result


def plot_histo(flux, flux_std, aperture, title):
    """ plot_ histogram to check apetures
    """
    import matplotlib.pyplot as plt
    import numpy as np

    flux_min = -10.0 * flux_std
    flux_max = 10.0 * flux_std

    # compute histogram
    hist, hist_bins = np.histogram(
        flux, range=(flux_min, flux_max), bins=50, density=True
    )

    # plot histogram
    plt.fill_between(
        0.5 * (hist_bins[1:] + hist_bins[:-1]),
        0.0,
        hist,
        color="red",
        alpha=0.5,
    )
    plt.plot(
        0.5 * (hist_bins[1:] + hist_bins[:-1]),
        hist,
        color="red",
        lw=2,
        label='Histogram (${0:3.2f}$" diam. aper)'.format(aperture),
    )
    ylim = plt.ylim(0.0,)

    # overplot gaussian centered at flux = 0.0 and scaled to flux distribution
    x = np.linspace(flux_min, flux_max, 120)
    x0 = np.median(flux)
    y = (
        1.0
        / (flux_std * np.sqrt(2.0 * np.pi))
        * np.exp(-(x - x0) * (x - x0) / (2.0 * flux_std * flux_std))
    )  # / (hist_bins[1] - hist_bins[0])#*max(hist)
    y = y * hist.max() / y.max()
    plt.plot(x, y, "-", linewidth=2, label="Gauss. (mad estimate)")
    plt.plot([0.0, 0.0], [0.0, ylim[1]], ":", color="blue")

    # Axis labels and title
    plt.ylabel("n (flux)")
    plt.xlabel("Flux (arbitrary unit)")
    plt.legend(frameon=None, fancybox=None)
    plt.title(title)


def get_mag_zp(header):
    """
    Method to get zero point out of a fits header
    """
    import numpy as np

    # add flux or zero point key here
    keys_mag_zp = ["SEXMGZPT"]

    found_key = False

    # first try mag zero point keys
    for key in keys_mag_zp:
        try:
            mag_zp = header[key]
        except (KeyError):
            pass
        else:
            found_key = True

    # then try flux zero point keys
    if not found_key:
        for key in keys_mag_zp:
            try:
                mag_zp = 2.5 * np.log10(header[key])
            except (KeyError):
                pass
            else:
                found_key = True

    # no keys found, add it to keys_mag_zp[] or set it in the options
    if not found_key:
        raise KeyError("No flux or MZP. Set --mag_zp MAG_ZEROPOINT")

    return mag_zp


def depth(flux, mag_zp, sigma, type):
    """ Compute depth of the image
    """
    import astropy.stats as astats
    import numpy as np

    flux_std = 1.4826 * astats.median_absolute_deviation(flux)
    # flux_std = astats.biweight_midvariance(flux)
    N = len(flux)
    n_sigma = len(sigma)
    n_boot = 100

    # resample flux
    flux_r = np.zeros(N)
    d_r = np.zeros((n_boot, n_sigma))

    for l in range(n_boot):
        for i in range(N):
            flux_r[i] = flux[np.random.randint(0, N - 1)]

        flux_std_r = 1.4826 * astats.median_absolute_deviation(flux_r)

        for j in range(n_sigma):
            d_r[l, j] = -2.5 * np.log10(sigma[j] * flux_std_r) + mag_zp

    d = np.zeros(n_sigma)
    d_err = np.zeros(n_sigma)

    if type == "weight" or type == "var" or type == "rms":
        flux_std = np.sqrt(np.median(flux))

    for j in range(n_sigma):
        d[j] = -2.5 * np.log10(sigma[j] * flux_std) + mag_zp
        d_err[j] = np.std(d_r[:, j])

    return d, d_err, flux_std


def get_depth_image(xpapoint=None, argv=[]):
    """Get the depth of astronomical image(s)
    """
    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-a",
        "--aperture",
        default="2",
        help="Aperture radius in pixels",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-z",
        "--zero_point_magnitude",
        default="0",
        help="Zero point magnitude of the image",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-n",
        "--number_apertures",
        default="1000",
        help="Number of apertures to throw in the image",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)

    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    mag_zp, aper_size, n_aper = (
        args.zero_point_magnitude,
        args.aperture,
        args.number_apertures,
    )
    if mag_zp.lower() == "-":
        mag_zp = None
    else:
        mag_zp = float(mag_zp)
    aper_size = float(aper_size)
    n_aper = int(n_aper)
    for filename in [filename]:
        if mag_zp is None:
            if "HSC" in filename:
                mag_zp = 27
            if ("VIRCAM" in filename) | ("MegaCam" in filename):
                mag_zp = 30
        verboseprint(filename)
        verboseprint("Zero point magnitude =", mag_zp)
        from astropy.io import fits

        ext = fits_ext(fits.open(filename))
        main_coupon(
            file_in_name=filename,
            file_out_name=None,
            ext=ext,
            ext_seg=ext,
            mag_zp=mag_zp,
            sub=None,
            aper_size=aper_size,
            plot=True,
            file_seg_name=None,
            type="image",
            nomemmap=False,
            n_aper=n_aper,
        )
    return


def run_sextractor(xpapoint=None, detector=None, path=None, argv=[]):
    """Run SExtraxtor astromatic software
    """
    import astropy
    from astropy.wcs import WCS
    from shutil import which
    import numpy as np
    from astropy.table import Table

    parser = create_parser(get_name_doc(), path=True)

    parser.add_argument(
        "-d",
        "--DETECTION_IMAGE",
        default="",
        help="Image to use for detection of the sources. If - using DS9 image",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-CN",
        "--CATALOG_NAME",
        default="",
        help="eg /tmp/catalog.fits, if none saved as $filename_ds9_cat.fits",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-CT",
        "--CATALOG_TYPE",
        help="Type of the output catalog",
        type=str,
        choices=[
            "FITS_1.0",
            "NONE",
            "ASCII",
            "ASCII_HEAD",
            "ASCII_SKYCAT",
            "ASCII_VOTABLE",
            "FITS_LDAC",
        ],
    )
    parser.add_argument(
        "-PN",
        "--PARAMETERS_NAME",
        default="sex.param",
        help="File name in ds9_package/Calibration/Sextractor/",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-DT",
        "--DETECT_TYPE",
        default="CCD",
        help="CCD (linear) or PHOTO (with gamma correction)",
        type=str,
        choices=["CCD", "PHOTO"],
    )
    parser.add_argument(
        "-DM",
        "--DETECT_MINAREA",
        default="10",
        help="min number of pixels above threshold",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-Dm",
        "--DETECT_MAXAREA",
        default="0",
        help="max number of pixels above threshold (0=unlimited)}",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-TT",
        "--THRESH_TYPE",
        default="RELATIVE",
        type=str,
        choices=["RELATIVE", "ABSOLUTE"],
    )
    parser.add_argument(
        "-dT",
        "--DETECT_THRESH",
        default="0.8",
        help="Detection threshold. 1 argument: ADUs or relative to Background RMS, see THRESH TYPE",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-AT", "--ANALYSIS_THRESH", default="2.0", type=str, metavar="",
    )
    parser.add_argument(
        "-F", "--FILTER", help="apply filter for detection", choices=["1", "0"],
    )
    parser.add_argument(
        "-FN",
        "--FILTER_NAME",
        default="NONE",
        type=str,
        metavar="",
        help="filter in ds9_pack/Calibration/Sextractor/, NONE for no filter",
    )
    parser.add_argument(
        "-DN",
        "--DEBLEND_NTHRESH",
        default="64",
        help="Number of deblending sub-thresholds",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-D",
        "--DEBLEND_MINCONT",
        default="0.0003",
        help="Minimum contrast parameter for deblending",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-C",
        "--CLEAN",
        help="Clean spurious detections?",
        type=str,
        choices=["1", "0"],
        metavar="",
    )
    parser.add_argument(
        "-CP",
        "--CLEAN_PARAM",
        default="1.0",
        help="Cleaning efficiency",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-M",
        "--MASK_TYPE",
        help="type of detection MASKing",
        type=str,
        choices=["CORRECT", "NONE", "BLANK"],
        metavar="",
    )
    parser.add_argument(
        "-WT",
        "--WEIGHT_TYPE",
        type=str,
        default="NONE",
        choices=["NONE", "NONE,MAP_VAR", "MAP_VAR", "MAP_VAR,MAP_VAR"],
        help="First one for detection image, second one for photometric image",
        metavar="",
    )
    parser.add_argument(
        "-RW",
        "--RESCALE_WEIGHTS",
        help="Rescale input weights/variances ",
        type=str,
        choices=["1", "0"],
        metavar="",
    )
    parser.add_argument(
        "-WI",
        "--WEIGHT_IMAGE",
        default="NONE",
        help="weight-map filename",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-WG",
        "--WEIGHT_GAIN",
        help="modulate gain (E/ADU) with weights?",
        type=str,
        choices=["1", "0"],
        metavar="",
    )
    parser.add_argument(
        "-FI",
        "--FLAG_IMAGE",
        default="NONE",
        help="filename for an input FLAG-image",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-FT",
        "--FLAG_TYPE",
        type=str,
        default="OR",
        choices=["OR", "AND", "MAX", "MIN", "MOST"],
        help="flag pixel combination",
        metavar="",
    )
    parser.add_argument(
        "-PA",
        "--PHOT_APERTURES",
        default="6,12,18",
        help="MAG_APER aperture diameter(s) in pixels",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-P",
        "--PHOT_AUTOPARAMS",
        default="2.5,4.0",
        help="MAG_AUTO parameters: <Kron_fact>,<min_radius>",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PP",
        "--PHOT_PETROPARAMS",
        default="2.0,4.0",
        help="MAG_PETRO parameters: <Petrosian_fact>",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PF",
        "--PHOT_FLUXFRAC",
        default="0.3,0.5,0.9",
        help="flux fraction[s] used for FLUX_RADIUS",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SL",
        "--SATUR_LEVEL",
        default="50000.0",
        help="level (in ADUs) at which arises saturation",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SK",
        "--SATUR_KEY",
        default="SATURATE",
        help="keyword for saturation level (in ADUs)",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-MZ", "--MAG_ZEROPOINT", default="0.0", type=str, metavar="",
    )
    parser.add_argument(
        "-MG",
        "--MAG_GAMMA",
        default="4.0",
        type=str,
        metavar="",
        help="gamma of emulsion (for photographic scans)",
    )
    parser.add_argument(
        "-G",
        "--GAIN",
        default="GAIN",
        type=str,
        metavar="",
        help="detector gain in e-/ADU",
    )
    parser.add_argument(
        "-PS",
        "--PIXEL_SCALE",
        default="0.0",
        help="size of pixel in arcsec (0=use FITS WCS info)",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SF",
        "--SEEING_FWHM",
        default="0.8",
        help="stellar FWHM in arcsec",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SN",
        "--STARNNW_NAME",
        default="default.nnw",
        help="File name in ds9_package/Calibration/Sextractor/",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-ct",
        "--CHECKIMAGE_TYPE",
        help="type of detection MASKing",
        metavar="",
        type=str,
        choices=[
            "NONE",
            "BACKGROUND",
            "BACKGROUND_RMSMINIBACKGROUND",
            "MINIBACK_RMS",
            "-BACKGROUND",
            "FILTERED",
            "OBJECTS",
            "-OBJECTS",
            "SEGMENTATION",
            "APERTURES",
        ],
    )
    parser.add_argument(
        "-BT",
        "--BACK_TYPE",
        default="AUTO",
        type=str,
        choices=["AUTO", "MANUAL"],
        metavar="",
    )
    parser.add_argument(
        "-BV",
        "--BACK_VALUE",
        default="0.0",
        help="Default background value in MANUAL mode",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BS",
        "--BACK_SIZE",
        default="64",
        help="Size in pixels of a background mesh.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BFS",
        "--BACK_FILTERSIZE",
        default="3",
        help="Size in background meshes of the background-filtering mask.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BFT",
        "--BACK_FILTTHRESH",
        default="0.0",
        help="Threshold above which the background map filter operates",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BPT",
        "--BACKPHOTO_TYPE",
        default="LOCAL",
        type=str,
        choices=["LOCAL", "GLOBAL"],
        metavar="",
    )
    parser.add_argument(
        "-bpt",
        "--BACKPHOTO_THICK",
        default="24",
        help="thickness of the background LOCAL annulus",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-MO",
        "--MEMORY_OBJSTACK",
        default="3000",
        help="number of objects in stack",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-MP",
        "--MEMORY_PIXSTACK",
        default="300000",
        help="number of pixels in stack",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-MB",
        "--MEMORY_BUFSIZE",
        default="1024",
        help="number of lines in buffer",
        type=str,
        metavar="",
    )  # ,metavar='',
    parser.add_argument(
        "-N", "--NTHREADS", default=os.cpu_count() - 2, type=str, metavar=""
    )
    args = parser.parse_args_modif(argv, required=True)

    d = DS9n(args.xpapoint)
    filename = globglob(args.path, xpapoint=args.xpapoint)
    if which("sex") is None:
        message(
            d,
            """Sextractor do not seem to be installed on your machine.
                      If you know it is, please add the sextractor executable
                      path to your $PATH variable in .bash_profile. Depending
                      on your image, the analysis might take a few minutes""",
        )
        verboseprint(
            "On mac run in terminal: >brew install brewsci/science/sextractor",
            verbose="1",
        )
        verboseprint(
            "or visit: https://github.com/astromatic/sextractor", verbose="1",
        )
    param_names = [
        "CATALOG_NAME",
        "CATALOG_TYPE",
        "PARAMETERS_NAME",
        "DETECT_TYPE",
        "DETECT_MINAREA",
        "DETECT_MAXAREA",
        "THRESH_TYPE",
        "DETECT_THRESH",
        "ANALYSIS_THRESH",
        "FILTER",
        "FILTER_NAME",
        "DEBLEND_NTHRESH",
        "DEBLEND_MINCONT",
        "CLEAN",
        "CLEAN_PARAM",
        "MASK_TYPE",
        "WEIGHT_TYPE",
        "RESCALE_WEIGHTS",
        "WEIGHT_IMAGE",
        "WEIGHT_GAIN",
        "FLAG_IMAGE",
        "FLAG_TYPE",
        "PHOT_APERTURES",
        "PHOT_AUTOPARAMS",
        "PHOT_PETROPARAMS",
        "PHOT_FLUXFRAC",
        "SATUR_LEVEL",
        "SATUR_KEY",
        "MAG_ZEROPOINT",
        "MAG_GAMMA",
        "GAIN",
        "PIXEL_SCALE",
        "SEEING_FWHM",
        "STARNNW_NAME",
        "CHECKIMAGE_TYPE",
        "BACK_TYPE",
        "BACK_VALUE",
        "BACK_SIZE",
        "BACK_FILTERSIZE",
        "BACKPHOTO_TYPE",
        "BACKPHOTO_THICK",
        "BACK_FILTTHRESH",
        "MEMORY_OBJSTACK",
        "MEMORY_PIXSTACK",
        "MEMORY_BUFSIZE",
        "NTHREADS",
    ]

    DETECTION_IMAGE = args.DETECTION_IMAGE  # sys.argv[4]
    ID = "NUMBER"  # None#sys.argv[3]

    param_dict = {}
    for key in zip(param_names):
        param_dict[key[0]] = getattr(args, key[0])
        verboseprint(key[0], getattr(args, key[0]))
    param_dir = resource_filename("pyds9plugin", "Sextractor")

    param_dict["FILTER"] = "Y" if param_dict["FILTER"] == "1" else "N"
    param_dict["CLEAN"] = "Y" if param_dict["CLEAN"] == "1" else "N"
    param_dict["RESCALE_WEIGHTS"] = (
        "Y" if param_dict["RESCALE_WEIGHTS"] == "1" else "N"
    )
    param_dict["WEIGHT_GAIN"] = "Y" if param_dict["WEIGHT_GAIN"] == "1" else "N"
    verboseprint("DETECTION_IMAGE =", DETECTION_IMAGE)

    if (len(filename) == 1) & (param_dict["CATALOG_NAME"] == ""):
        param_dict["CATALOG_NAME"] = filename[0][:-5] + "_cat.fits"
        cat_path = os.path.join(
            os.path.dirname(filename[0]),
            os.path.basename(filename[0]).split(".")[0],
        )

    param_dict["PARAMETERS_NAME"] = os.path.join(
        param_dir, param_dict["PARAMETERS_NAME"]
    )
    param_dict["FILTER_NAME"] = os.path.join(
        param_dir, param_dict["FILTER_NAME"]
    )
    param_dict["STARNNW_NAME"] = os.path.join(
        param_dir, param_dict["STARNNW_NAME"]
    )

    verboseprint("Image used for detection  = " + str(DETECTION_IMAGE))
    verboseprint("Image used for photometry  = " + str(filename))
    verboseprint("Parameters sextractor:")
    answer = parallelize(
        function=run_sex,
        parameters=[DETECTION_IMAGE, param_dict],
        action_to_paralize=filename,
        number_of_thread=args.NTHREADS,
    )
    if answer != 0:
        verboseprint(
            """It seems that SExtractor encountered an error.\n
                        Please verify your image(s)/parameters.
                        \nTo know more about the error run the following
                        command in a terminal:\n""",
            verbose="1",
        )
        sys.exit()
    if len(filename) > 1:
        verboseprint("Analysis ended on all images.")
        sys.exit()
    else:
        filename = filename[0]

    colors = ["Orange"]
    if os.path.isfile(param_dict["CATALOG_NAME"]):
        verboseprint(param_dict["CATALOG_NAME"])
        try:
            cat_sex = Table.read(param_dict["CATALOG_NAME"])
        except astropy.io.registry.IORegistryError:
            verboseprint("Reading ascii")
            cat_sex = Table.read(param_dict["CATALOG_NAME"], format="ascii")
        if len(cat_sex) == 1:
            verboseprint("Reading LDAC_OBJECT")
            cat_sex = Table.read(
                param_dict["CATALOG_NAME"], format="fits", hdu="LDAC_OBJECTS"
            )
        verboseprint(cat_sex)
        w = WCS(filename)
        d.set("regions showtext no")
        if len(cat_sex) == 0:
            message(d, "No source detected, verify you parameters...")
        else:
            verboseprint("Creating regions")
            if yesno(
                d,
                """%i sources detected! Do you want to load them as a catalog
                 (<10Ksources), if not, it will be loaded as regions."""
                % (len(cat_sex)),
            ):
                try:
                    if (w.is_celestial) & (
                        np.isfinite(
                            (
                                cat_sex["ALPHA_J2000"]
                                + cat_sex["DELTA_J2000"]
                                + cat_sex["B_WORLD"]
                                + cat_sex["A_WORLD"]
                            ).data
                        ).all()
                    ):
                        x, y = "ALPHA_J2000", "DELTA_J2000"
                    else:
                        x, y = "X_IMAGE", "Y_IMAGE"
                    command = """catalog import FITS %s ; catalog x %s ;
                                 catalog y %s ; catalog symbol shape
                                 ellipse  ; catalog symbol Size
                                 "$A_IMAGE * $KRON_RADIUS/2" ; catalog symbol
                                 Size2 "$B_IMAGE * $KRON_RADIUS/2"; catalog
                                 symbol angle "$THETA_IMAGE" ; mode catalog """
                    d.set(f_string(command % (param_dict["CATALOG_NAME"], x, y)))
                except ValueError as e:
                    verboseprint(e)
            else:
                reg_file = cat_path + ".reg"
                if (w.is_celestial) & (
                    np.isfinite(
                        (
                            cat_sex["ALPHA_J2000"]
                            + cat_sex["DELTA_J2000"]
                            + cat_sex["THETA_WORLD"]
                            + cat_sex["B_WORLD"]
                            + cat_sex["A_WORLD"]
                        ).data
                    ).all()
                ):
                    verboseprint("Using WCS header for regions :", reg_file)
                    create_ds9_regions(
                        [cat_sex["ALPHA_J2000"]],
                        [cat_sex["DELTA_J2000"]],
                        more=[
                            cat_sex["A_WORLD"] * cat_sex["KRON_RADIUS"] / 2,
                            cat_sex["B_WORLD"] * cat_sex["KRON_RADIUS"] / 2,
                            -cat_sex["THETA_WORLD"],
                        ],
                        form=["ellipse"] * len(cat_sex),
                        save=True,
                        ID=[np.around(cat_sex[ID], 1).astype(str)],
                        color=["Yellow"] * len(cat_sex),
                        savename=reg_file,
                        system="fk5",
                        DS9_offset=[0, 0],
                        font=10,
                    )
                else:
                    create_ds9_regions(
                        [cat_sex["X_IMAGE"]],
                        [cat_sex["Y_IMAGE"]],
                        more=[
                            cat_sex["A_IMAGE"] * cat_sex["KRON_RADIUS"] / 2,
                            cat_sex["B_IMAGE"] * cat_sex["KRON_RADIUS"] / 2,
                            cat_sex["THETA_IMAGE"],
                        ],
                        form=["ellipse"] * len(cat_sex),
                        save=True,
                        ID=[np.around(cat_sex[ID], 1).astype(str)],
                        color=[np.random.choice(colors)] * len(cat_sex),
                        savename=reg_file,
                        font=10,
                    )
                d.set("regions " + reg_file)
    else:
        verboseprint("Can not find the output sextractor catalog...")
    return


def run_sex(path, DETECTION_IMAGE, param_dict):
    if DETECTION_IMAGE == "":
        DETECTION_IMAGE = path

    if param_dict["CATALOG_NAME"] == "":
        param_dict["CATALOG_NAME"] = path[:-5] + "_cat.fits"
    for key in list(param_dict.keys()):
        if param_dict[key] == "":
            del param_dict[key]
    cat_path = os.path.join(
        os.path.dirname(path), os.path.basename(path).split(".")[0]
    )  # .fits'
    param_dict["CHECKIMAGE_NAME"] = cat_path + "_check_%s.fits" % (
        param_dict["CHECKIMAGE_TYPE"]
    )
    command = (
        "sex "
        + "%s,%s  " % (DETECTION_IMAGE, path)
        + " -WRITE_XML Y  -"
        + " -".join(
            [
                key + " " + str(param_dict[key])
                for key in list(param_dict.keys())[:]
            ]
        )
    )
    verboseprint(command)
    return os.system(command)


def sextractor_pp(xpapoint=None, detector=None, path=None, argv=[]):
    """Run sextraxtor ++ software (Beta version)
    """
    import numpy as np

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-d", "--detection_image", type=str, metavar="", required=True
    )
    parser.add_argument("-f", "--output-catalog-filename", metavar="")
    parser.add_argument(
        "-F", "--output-catalog-format", default="FITS", metavar=""
    )
    args = parser.parse_args_modif(argv, required=False)
    d = DS9n(args.xpapoint)
    filename = get_filename(d)
    param_names = [
        "output-catalog-filename",
        "output-catalog-format",
        "detect-minarea",
        "detection-threshold",
        "partition-threshold-count",
        "partition-min-contrast",
        "use-cleaning",
        "cleaning-minarea",
        "weight-image",
        "magnitude-zeropoint",
        "background-cell-size",
        "smoothing-box-size",
        "thread-count",
    ]

    DETECTION_IMAGE = args.detection_image
    param_dict = {}
    for key in zip(param_names[:1]):
        param_dict[key[0]] = getattr(args, key[0].replace("-", "_"))
    param_dict["output-catalog-filename"] = (
        filename[:-5] + "_cat.fits"
        if param_dict["output-catalog-filename"] == ""
        else param_dict["output-catalog-filename"]
    )
    if DETECTION_IMAGE == "":
        DETECTION_IMAGE = None
    else:
        DETECTION_IMAGE = DETECTION_IMAGE + ","
    verboseprint("Image used for detection  = " + str(DETECTION_IMAGE))
    verboseprint("Image used for photometry  = " + str(filename))

    for key in list(param_dict.keys()):
        if param_dict[key] == "":
            del param_dict[key]

    command = "SourceXtractor++ --detection-image %s  --" % (
        filename
    ) + " --".join(
        [key + " " + str(param_dict[key]) for key in list(param_dict.keys())[:]]
    )
    verboseprint(command)
    os.system(command)
    verboseprint(param_dict["output-catalog-filename"])
    try:
        import_table_as_region(
            args.xpapoint,
            argv="-p %s -xy pixel_centroid_x,pixel_centroid_y"
            % (param_dict["output-catalog-filename"]),
        )
    except KeyError:
        import_table_as_region(
            args.xpapoint,
            argv="-p %s -xy col1,col2" % (param_dict["output-catalog-filename"]),
        )
    if yesno(d, "Do you want to load the sextractor catalog with PRISM?",):
        d.set("prism " + param_dict["output-catalog-filename"])

    return


def ds9_swarp(xpapoint=None, argv=[]):
    """Run swarp software from DS9
    """
    from shutil import which

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-p",
        "--path",
        default="",
        help="Path of the images you want to stack, use regexp",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-i",
        "--IMAGEOUT_NAME",
        default="/tmp/test.fits",
        help="Name of the output image",
        metavar="",
    )
    parser.add_argument(
        "-w",
        "--WEIGHT_TYPE",
        type=str,
        metavar="",
        choices=["NONE", "BACKGROUND", "MAP_RMS", "MAP_VARIANCE", "MAP_WEIGHT"],
    )
    parser.add_argument(
        "-rw",
        "--RESCALE_WEIGHTS",
        help="Rescale input weights/variances",
        type=str,
        choices=["1", "0"],
    )
    parser.add_argument(
        "-W",
        "--WEIGHT_IMAGE",
        default="None",
        help="Name of the input weight image file",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-c",
        "--COMBINE",
        default="1",
        help="Combine resampled images ",
        type=str,
        choices=["1", "0"],
    )
    parser.add_argument(
        "-ct",
        "--COMBINE_TYPE",
        help="Tells SWarp how to combine resampled images",
        type=str,
        default="MEDIAN",
        choices=[
            "MEDIAN",
            "AVERAGE",
            "MIN",
            "MAX",
            "WEIGHTED",
            "CHI2",
            "CHI-MEAN",
            "SUM",
        ],
    )

    parser.add_argument(
        "-CA",
        "--CLIP_AMPFRAC",
        default="0.3",
        help="Fraction of flux variation allowed with clipping",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-CS",
        "--CLIP_SIGMA",
        default="4.0",
        help="error multiple variation allowed with clipping",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-CT",
        "--CELESTIAL_TYPE",
        default="NATIVE",
        help="Celestial coordinate system in output",
        type=str,
        choices=["NATIVE", "PIXEL", "EQUATORIAL", "GALACTIC", "ECLIPTIC"],
    )
    parser.add_argument(
        "-PT",
        "--PROJECTION_TYPE",
        default="TAN",
        help="Projection system used in output, in standard WCS notation",
        type=str,
        choices=[
            "TAN",
            "AZP",
            "STG",
            "SIN",
            "ARC",
            "ZPN",
            "ZEA",
            "AIR",
            "CYP",
            "CEA",
            "CAR",
            "MER",
            "COP",
            "COE",
            "COD",
            "COO",
            "BON",
            "PCO",
            "GLS",
            "PAR",
            "MOL",
            "AIT",
            "TSC",
            "CSC",
            "QSC",
        ],
    )
    parser.add_argument(
        "-PE",
        "--PROJECTION_ERR",
        default="0.001",
        help="Maximum position error (in pixels) allowed for bicubic-spline interpolation of the astrometric reprojection. 0 turns off interpolation.",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-C",
        "--CENTER",
        default="0",
        help="Position of the center in CENTER TYPE MANUAL mode.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-C_T",
        "--CENTER_TYPE",
        default="ALL",
        help="Tells SWarp how to center the output frame",
        type=str,
        choices=["ALL", "MOST", "MANUAL"],
    )
    parser.add_argument(
        "-PST",
        "--PIXELSCALE_TYPE",
        default="MEDIAN",
        help="Tells SWarp how to adjust the output pixel size",
        type=str,
        choices=["MEDIAN", "MIN", "MAX", "MANUAL", "FIT"],
    )
    parser.add_argument(
        "-PS",
        "--PIXEL_SCALE",
        default="0.0",
        help="Step between pixels in each dimension in PIXELSCALE_TYPE MANUAL mode. Must be expressed in arcseconds for angular coordinates.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-IS",
        "--IMAGE_SIZE",
        default="0.0",
        help="Dimensions of the output image in PIXELSCALE TYPE MANUAL or FIT mode. 0 means automatic",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-R", "--RESAMPLE", help="Resample input images ?", choices=["1", "0"],
    )
    parser.add_argument(
        "-r",
        "--RESAMPLING_TYPE",
        default="LANCZOS3",
        help="Image resampling method",
        type=str,
        choices=["LANCZOS3", "NEAREST", "BILINEAR", "LANCZOS2", "LANCZOS4"],
        metavar="",
    )
    parser.add_argument(
        "-o",
        "--OVERSAMPLING",
        default="0",
        help="Amount of oversampling in each dimension. 0 means automatic.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-I",
        "--INTERPOLATE",
        help="Interpolate bad input pixels ?",
        choices=["1", "0"],
    )

    parser.add_argument(
        "-SB",
        "--SUBTRACT_BACK",
        help="Subtraction sky background ?",
        type=str,
        choices=["1", "0"],
    )
    parser.add_argument(
        "-BS",
        "--BACK_SIZE",
        default="128",
        help="Size in pixels of a background mesh.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BFS",
        "--BACK_FILTERSIZE",
        default="3",
        help="Size in background meshes of the background-filtering mask.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BFT",
        "--BACK_FILTTHRESH",
        default="0.0",
        help="Difference threshold (in ADUs) for the background-filtering",
        type=str,
        metavar="",
    )

    args = parser.parse_args_modif(argv)

    param_names = [
        "IMAGEOUT_NAME",
        "WEIGHT_TYPE",
        "RESCALE_WEIGHTS",
        "WEIGHT_IMAGE",
        "COMBINE",
        "COMBINE_TYPE",
        "CLIP_AMPFRAC",
        "CLIP_SIGMA",
        "CELESTIAL_TYPE",
        "PROJECTION_TYPE",
        "PROJECTION_ERR",
        "CENTER_TYPE",
        "CENTER",
        "PIXELSCALE_TYPE",
        "PIXEL_SCALE",
        "IMAGE_SIZE",
        "RESAMPLE",
        "RESAMPLING_TYPE",
        "OVERSAMPLING",
        "INTERPOLATE",
        "SUBTRACT_BACK",
        "BACK_SIZE",
        "BACK_FILTERSIZE",
        "BACK_FILTTHRESH",
    ]

    param_dict = {}
    for key in zip(param_names):
        param_dict[key[0]] = getattr(args, key[0])
        verboseprint("%s : %s" % (key[0], getattr(args, key[0])))

    param_dict["SUBTRACT_BACK"] = (
        "Y" if param_dict["SUBTRACT_BACK"] == "1" else "N"
    )
    param_dict["RESAMPLE"] = "Y" if param_dict["RESAMPLE"] == "1" else "N"
    param_dict["RESCALE_WEIGHTS"] = (
        "Y" if param_dict["RESCALE_WEIGHTS"] == "1" else "N"
    )
    param_dict["COMBINE"] = "Y" if param_dict["COMBINE"] == "1" else "N"
    param_dict["INTERPOLATE"] = "Y" if param_dict["INTERPOLATE"] == "1" else "N"

    d = DS9n(args.xpapoint)
    paths = globglob(args.path, xpapoint=args.xpapoint)

    verboseprint("Images to coadd: %s" % (paths))
    param_dict["IMAGEOUT_NAME"] = os.path.join(
        os.path.dirname(paths[0]), param_dict["IMAGEOUT_NAME"]
    )
    if which("swarp") is None:
        d = DS9n(args.xpapoint)
        message(
            d,
            """SWARP do not seem to be installed in you machine.
                      If you know it is, please add the sextractor executable
                      path to your $PATH variable in .bash_profile. Depending
                      on your image, the analysis might take a few minutes""",
        )

    else:
        os.chdir(os.path.dirname(paths[0]))
        os.system("sleep 0.1")
        print(
            "swarp %s  -c default.swarp -" % (" ".join(paths))
            + " -".join(
                [
                    key + " " + str(param_dict[key])
                    for key in list(param_dict.keys())[:]
                ]
            )
        )
        answer = os.system(
            "swarp %s  -c default.swarp -" % (" ".join(paths))
            + " -".join(
                [
                    key + " " + str(param_dict[key])
                    for key in list(param_dict.keys())[:]
                ]
            )
        )
        if answer != 0:
            d = DS9n(args.xpapoint)
            message(
                d,
                """PSFex encountered an error. Please verify your
                          image(s)/parameters and enter verbose mode (shift+V)
                          for more precision about the error.""",
            )
            sys.exit()
    d.set("frame new ; tile no ; file %s" % (param_dict["IMAGEOUT_NAME"]))
    return


def resample(xpapoint=None, argv=[]):
    """Run SWARP astromatic software
    """
    from shutil import which

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-P",
        "--PIXEL_SCALE",
        default="0.0",
        help="Step between pixels in each dimension in PIXELSCALE_TYPE MANUAL mode. Must be expressed in arcseconds for angular coordinates.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-i",
        "--IMAGE_SIZE",
        default="0.0",
        help="Dimensions of the output image in PIXELSCALE TYPE MANUAL or FIT mode. 0 means automatic",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-r",
        "--RESAMPLING_TYPE",
        default="2",
        help="Image resampling method",
        type=str,
        choices=["LANCZOS3", "NEAREST", "BILINEAR", "LANCZOS2", "LANCZOS4"],
        metavar="",
    )
    parser.add_argument(
        "-s",
        "--RESAMPLE_SUFFIX",
        default=".resamp.fits",
        help="filename extension for resampled images",
        type=str,
        metavar="",
    )

    args = parser.parse_args_modif(argv, required=True)

    param_names = [
        "PIXEL_SCALE",
        "IMAGE_SIZE",
        "RESAMPLING_TYPE",
        "RESAMPLE_SUFFIX",
    ]
    params = (
        args.PIXEL_SCALE,
        args.IMAGE_SIZE,
        args.RESAMPLING_TYPE,
        args.RESAMPLE_SUFFIX,
    )

    param_dict = {}
    for key, val in zip(param_names, params):
        param_dict[key] = val
    if param_dict["PIXEL_SCALE"] != "0,0":
        param_dict["PIXELSCALE_TYPE"] = "MANUAL"
    if param_dict["IMAGE_SIZE"] != "0,0":
        param_dict["PIXELSCALE_TYPE"] = "FIT"

    for key, val in zip(param_names, params):
        if val == "":
            param_dict.pop(key, None)
    for key, val in zip(param_names, params):
        verboseprint("%s : %s" % (key, param_dict[key]))

    d = DS9n(args.xpapoint)
    paths = globglob(args.path, xpapoint=args.xpapoint)

    verboseprint("Images to coadd: %s" % (paths))

    if which("swarp") is None:
        d = DS9n(args.xpapoint)
        message(
            d,
            """SWARP do not seem to be installed in you machine. If you
                       know it is, please add the sextractor executable path to
                       your $PATH variable in .bash_profile. Depending on your
                       image, the analysis might take a few minutes""",
        )

    else:
        os.chdir(os.path.dirname(paths[0]))
        os.system("sleep 0.1")
        c = """swarp %s  -c default.swarp -COMBINE N -RESCALE_WEIGHTS N
               -IMAGEOUT_NAME /tmp/coadd.fits -""" % (
            " ".join(paths)
        )
        command = c + " -".join(
            [
                key + " " + str(param_dict[key])
                for key in list(param_dict.keys())[:]
            ]
        )
        verboseprint(command)
        answer = os.system(command)
        if answer != 0:
            d = DS9n(args.xpapoint)
            message(
                d,
                """PSFex encountered an error. Please verify your
                          image(s)/parameters and enter verbose mode (shift+V)
                          for more precision about the error.""",
            )
            sys.exit()

    d.set(
        "frame new ; tile no ; file %s"
        % (paths[0][:-5] + param_dict["RESAMPLE_SUFFIX"])
    )
    return


def ds9_psfex(xpapoint=None, argv=[]):
    """Run PSFex astromatic software
    """
    # print(1)
    from astropy.table import Table
    from shutil import which
    from astropy.io import fits

    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--ENTRY_PATH",
        help="Linux path of the FITS_LDAC (!) entry catalogs. Can contain */?",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-BT",
        "--BASIS_TYPE",
        default="PIXEL_AUTO",
        type=str,
        choices=["PIXEL_AUTO", "NONE", "PIXEL", "GAUSS-LAGUERRE", "FILE"],
        metavar="",
    )

    parser.add_argument(
        "-BN",
        "--BASIS_NUMBER",
        default="20",
        help="Basis number or parameter",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PS",
        "--PSF_SAMPLING",
        default="1.0",
        help="Sampling step in pixel units (0.0 = auto)",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PA",
        "--PSF_ACCURACY",
        default="0.01",
        help='Accuracy to expect from PSF "pixel" values',
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-ps",
        "--PSF_SIZE",
        default="45,45",
        help="Image size of the PSF model",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-CK",
        "--CENTER_KEYS",
        default="X_IMAGE,Y_IMAGE",
        help="Catalogue parameters for source pre-centering",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PK",
        "--PHOTFLUX_KEY",
        default="'FLUX_APER(2)'",
        help="Catalogue parameter for photometric norm.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PEK",
        "--PHOTFLUXERR_KEY",
        default="'FLUXERR_APER(2)'",
        help="Catalogue parameter for photometric error.",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-PSFK",
        "--PSFVAR_KEYS",
        default="X_IMAGE,Y_IMAGE",
        help="Catalogue or FITS (preceded by :) params",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PSFG",
        "--PSFVAR_GROUPS",
        default="1.1",
        help="Group tag for each context key",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PSFD",
        "--PSFVAR_DEGREES",
        default="0",
        type=str,
        choices=["0", "1", "2", "3", "4"],
        metavar="",
        help="Polynom degree for each group",
    )

    parser.add_argument(
        "-SA",
        "--SAMPLE_AUTOSELECT",
        default="0",
        type=str,
        choices=["0", "1", "Y", "N"],
        metavar="",
        help="Automatically select the FWHM?",
    )
    parser.add_argument(
        "-ST",
        "--SAMPLEVAR_TYPE",
        default="NONE",
        type=str,
        choices=["SEEING", "NONE"],
        metavar="",
        help="File-to-file PSF variability}",
    )

    parser.add_argument(
        "-SF",
        "--SAMPLE_FWHMRANGE",
        default="2.0,10.0",
        help="Allowed FWHM range",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SV",
        "--SAMPLE_VARIABILITY",
        default="0.2",
        help="Allowed FWHM variability (1.0 = 100)",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SM",
        "--SAMPLE_MINSN",
        default="20",
        help="Minimum S/N for a source to be used",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-SMAX",
        "--SAMPLE_MAXELLIP",
        default="0.3",
        help="Maximum (A-B)/(A+B) for a source to be used",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-CD",
        "--CHECKPLOT_DEV",
        default="PNG",
        type=str,
        choices=[
            "PNG",
            "XWIN",
            "TK",
            "PS",
            "PSC",
            "XFIG",
            "JPEG",
            "AQT",
            "PDF",
            "SVG",
        ],
        metavar="",
    )
    parser.add_argument(
        "-CT",
        "--CHECKPLOT_TYPE",
        default="RESIDUALS",
        type=str,
        choices=[
            "NONE",
            "FWHM",
            "ELLIPTICITY",
            "COUNTS",
            "COUNT_FRACTION",
            "CHI2",
            "RESIDUALS",
        ],
        metavar="",
    )
    parser.add_argument(
        "-CN", "--CHECKPLOT_NAME", default="check-plot", metavar=""
    )

    parser.add_argument(
        "-CIT",
        "--CHECKIMAGE_TYPE",
        default="CHI",
        type=str,
        choices=[
            "CHI",
            "PROTOTYPES",
            "SAMPLES",
            "RESIDUALS",
            "SNAPSHOTS",
            "MOFFAT",
            "-MOFFAT",
            "-SYMMETRICAL",
        ],
        metavar="",
    )
    parser.add_argument(
        "-CIN",
        "--CHECKIMAGE_NAME",
        default="check.fits",
        help="Group tag for each context key",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-PD",
        "--PSF_DIR",
        default=".",
        help="Where to write PSFs (empty=same as input)",
        type=str,
        metavar="",
    )

    parser.add_argument(
        "-HT",
        "--HOMOBASIS_TYPE",
        default="GAUSS-LAGUERRE",
        type=str,
        choices=["GAUSS-LAGUERRE", "NONE"],
        metavar="",
        help="GAUSS_LAGUERRE or no homogeneisation is computed",
    )
    parser.add_argument(
        "-HP",
        "--HOMOPSF_PARAMS",
        default="5.3,2.5",
        help="Moffat FWHM and B parameter of the idealized target PSF",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-W",
        "--WRITE_XML",
        default="Y",
        type=str,
        choices=["Y", "N", "1", "0"],
        metavar="",
        help="GAUSS_LAGUERRE or no homogeneisation is computed",
    )
    parser.add_argument(
        "-X",
        "--XML_NAME",
        default="psfex.xml",
        help="Filename for XML output",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-N",
        "--NTHREADS",
        default="1",
        help="Number of simultaneous threads, 0 = automatic",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-q",
        "--query",
        default="",
        help="Selection of objetcs in catalog:Use | for OR and \& for AND",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv)
    args.WRITE_XML = "Y" if args.WRITE_XML == "1" else "N"
    args.SAMPLE_AUTOSELECT = "Y" if args.SAMPLE_AUTOSELECT == "1" else "N"

    param_names = [
        "ENTRY_PATH",
        "BASIS_TYPE",
        "BASIS_NUMBER",
        "PSF_SAMPLING",
        "PSF_ACCURACY",
        "PSF_SIZE",
        "CENTER_KEYS",
        "PHOTFLUX_KEY",
        "PHOTFLUXERR_KEY",
        "PSFVAR_KEYS",
        "PSFVAR_GROUPS",
        "PSFVAR_DEGREES",
        "SAMPLE_AUTOSELECT",
        "SAMPLEVAR_TYPE",
        "SAMPLE_FWHMRANGE",
        "SAMPLE_VARIABILITY",
        "SAMPLE_MINSN",
        "SAMPLE_MAXELLIP",
        "CHECKPLOT_DEV",
        "CHECKPLOT_TYPE",
        "CHECKPLOT_NAME",
        "CHECKIMAGE_TYPE",
        "CHECKIMAGE_NAME",
        "PSF_DIR",
        "HOMOBASIS_TYPE",
        "HOMOPSF_PARAMS",
        # "VERBOSE_TYPE",
        "WRITE_XML",
        "XML_NAME",
        "NTHREADS",
    ]
    param_dict = {}
    for key in zip(param_names):
        param_dict[key[0]] = getattr(args, key[0])
        verboseprint("%s : %s" % (key, getattr(args, key[0])))
    # print(param_dict)
    query = args.query
    d = DS9n(args.xpapoint)
    param_dict["ENTRY_PATH"] = globglob(param_dict["ENTRY_PATH"])
    new_list = []
    if type(param_dict["ENTRY_PATH"]) is str:
        param_dict["ENTRY_PATH"] = [param_dict["ENTRY_PATH"]]

    if query != "":
        for path in param_dict["ENTRY_PATH"]:
            a = Table.read(
                path.rstrip()[::-1].rstrip()[::-1],
                format="fits",
                hdu="LDAC_OBJECTS",
            )
            a_simple = delete_multidim_columns(a)

            hdu1 = fits.open(path)

            df = a_simple.to_pandas()
            mask = df.eval(query)
            b = a[np.array(mask)]
            verboseprint("%0.1f %% of objets kept" % (len(b) / len(a)))
            verboseprint("Number of objects : %i => %i" % (len(a), len(b)))
            name = path[:-5] + "_.fits"
            hdu1[2].data = hdu1[2].data[mask]
            hdu1.writeto(name, overwrite=True)
            hdu1.close()
            new_list.append(name)
    else:
        for path in param_dict["ENTRY_PATH"]:
            new_list.append(path)
    paths = new_list

    if type(param_dict["ENTRY_PATH"]) is str:
        param_dict["XML_NAME"] = os.path.dirname(paths[0]) + "/psfex.xml"

    if which("psfex") is None:
        d = DS9n(args.xpapoint)
        message(
            d,
            """PSFex do not seem to be installed in you machine.
                      If you know it is, please add the sextractor executable
                      path to your $PATH variable in .bash_profile. Depending
                      on your image, the analysis might take a few minutes""",
        )
        sys.exit()
    else:
        verboseprint(os.path.dirname(paths[0]))
        os.chdir(os.path.dirname(paths[0]))
        os.system("sleep 0.1")
        verboseprint(
            "psfex %s -c default.psfex -%s -HOMOKERNEL_SUFFIX %s_homo.fits "
            % (
                " ".join(paths),
                " -".join(
                    [
                        key + " " + str(param_dict[key])
                        for key in list(param_dict.keys())[1:]
                    ]
                ),
                param_dict["HOMOPSF_PARAMS"].replace(",", "-"),
            )
        )
        answer = os.system(
            "psfex %s -c default.psfex -%s -HOMOKERNEL_SUFFIX %s_homo.fits "
            % (
                " ".join(paths),
                " -".join(
                    [
                        key + " " + str(param_dict[key])
                        for key in list(param_dict.keys())[1:]
                    ]
                ),
                param_dict["HOMOPSF_PARAMS"].replace(",", "-"),
            )
        )
        if answer != 0:
            d = DS9n(args.xpapoint)
            message(
                d,
                """PSFex encountered an error. Please verify your
                          input catalog (LDAC) and enter verbose mode (shift+V)
                          for more precision about the error""",
            )
            sys.exit()
    # if (args.xpapoint is not None) & (len(paths) == 1):
    d.set(
        "frame new ; file "
        + paths[0][:-5]
        + "%s_homo.fits" % (param_dict["HOMOPSF_PARAMS"].replace(",", "-"))
    )
    return


def moffat_1d(x, amp, std, alpha, x0=0):
    """1D moffat function
    """
    import numpy as np

    return amp * np.power((1 + np.square((x - x0) / std)), -alpha)


def gaus(x, a, b, sigma, lam, alpha):
    """1D gaussian centered on zero
    """
    import numpy as np

    gaus = a ** 2 * np.exp(-np.square(x / sigma) / 2)
    return gaus


def exp(x, a, b, sigma, lam, alpha):
    """1D exponential centered on zero
    """
    import numpy as np

    exp = b ** 2 * np.exp(-((x / lam) ** (1 ** 2)))
    return exp


def ds9_stiff(xpapoint=None, filename=None, argv=[]):
    """Run STIFF astromatic software
    """
    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p1",
        "--path1",
        help="Path of the red image eg. ~i (770nm) band",
        type=str,
        metavar="",
        required=True,
    )
    parser.add_argument(
        "-p2",
        "--path2",
        default="",
        help="Path of the red image eg. ~r (620nm) band",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-p3",
        "--path3",
        default="",
        help="Path of the red image eg. ~g (480nm) band",
        metavar="",
    )
    parser.add_argument(
        "-o",
        "--OUTFILE_NAME",
        default="stiff.tiff",
        help="Output image file name",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-t",
        "--IMAGE_TYPE",
        default="AUTO",
        help="Output image format.",
        type=str,
        choices=["AUTO", "TIFF", "TIFF-pyramid"],
    )
    parser.add_argument(
        "-b",
        "--BITS_PER_CHANNEL",
        default="8",
        help="BITS_PER_CHANNEL",
        type=str,
        choices=["8", "16", "-32"],
    )
    parser.add_argument(
        "-B",
        "--BINNING",
        default="1",
        help="Pixel binning factor for both axes, or along each axis.",
        type=str,
        choices=["1", "2"],
    )
    parser.add_argument(
        "-s",
        "--SKY_TYPE",
        default="AUTO",
        help="Sky level determination in each input image.",
        type=str,
        choices=["AUTO", "MANUAL"],
    )
    parser.add_argument(
        "-S",
        "--SKY_LEVEL",
        default="0.0",
        help="User-specified sky level in SKY TYPE MANUAL mode.(1<n<m_ima)",
        type=str,
    )
    parser.add_argument(
        "-mint",
        "--MIN_TYPE",
        default="QUANTILE",
        help="",
        type=str,
        choices=["GREYLEVEL", "QUANTILE", "MANUAL"],
    )
    parser.add_argument("-minl", "--MIN_LEVEL", default="0.001", metavar="")
    parser.add_argument(
        "-maxt",
        "--MAX_TYPE",
        default="QUANTILE",
        help="",
        type=str,
        choices=["GREYLEVEL", "QUANTILE", "MANUAL"],
    )
    parser.add_argument("-maxl", "--MAX_LEVEL", default="0.995", metavar="")
    parser.add_argument(
        "-g",
        "--GAMMA_TYPE",
        default="POWER-LAW",
        help="Gamma correction type.",
        type=str,
        choices=["POWER-LAW", "SRGB", "REC.709"],
    )
    parser.add_argument(
        "-G",
        "--GAMMA",
        default="2.2",
        help="exponent of the display intensity transfer curve for POWER-LAW GAMMA TYPEs.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-F",
        "--GAMMA_FAC",
        default="1.0",
        help="gamma correction factor for the luminance image component.",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-c",
        "--COLOUR_SAT",
        default="1.0",
        help="Colour saturation factor.",
        type=str,
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=False)

    d = DS9n(args.xpapoint)
    from shutil import which

    param_names = [
        "BINNING",
        "BITS_PER_CHANNEL",
        "COLOUR_SAT",
        "GAMMA",
        "GAMMA_FAC",
        "GAMMA_TYPE",
        "IMAGE_TYPE",
        "MAX_LEVEL",
        "MAX_TYPE",
        "MIN_LEVEL",
        "MIN_TYPE",
        "OUTFILE_NAME",
        "SKY_LEVEL",
        "SKY_TYPE",
    ]
    path1, path2, path3 = args.path1, args.path2, args.path3

    param_dict = {}
    for key in zip(param_names):
        param_dict[key[0]] = getattr(args, key[0])
        verboseprint("%s : %s" % (key[0], param_dict[key[0]]))

    d.set("frame last")
    if path1 == "":
        files = get_filename(d, All=True)
        if (
            (path1 == "")
            & (path2 == "")
            & (path3 == "")
            & ((len(files) == 3) | (len(files) == 1))
        ):
            try:
                path1, path2, path3 = files
            except ValueError:
                path1 = files[0]
        else:
            path1 = files[0]
    if which("stiff") is None:
        message(
            d,
            """Stiff do not seem to be installed in you machine.
                 If you know it is, please add the sextractor executable path
                 to your $PATH variable in .bash_profile. Depending on your
                 image, the analysis might take a few minutes""",
        )

    else:
        verboseprint("Going : ", path1)
        os.chdir(os.path.dirname(path1))
        if (path2 == "") & (path3 == ""):
            command = "stiff %s -" % (path1) + " -".join(
                [
                    key + " " + str(param_dict[key])
                    for key in list(param_dict.keys())[:]
                ]
            )
            verboseprint(command)
            answer = os.system(command)
            if answer != 0:
                message(
                    d,
                    """STIFF encountered an error. Please verify your
                              images/parameters and enter verbose mode
                              (shift+V) for more precision about the error.""",
                )
                sys.exit()
            d.set("frame new")
            verboseprint(param_dict["OUTFILE_NAME"])
            d.set(
                "tiff %s"
                % (
                    os.path.join(
                        os.path.dirname(path1), param_dict["OUTFILE_NAME"]
                    )
                )
            )

        else:
            command = "stiff %s %s %s -" % (path1, path2, path3) + " -".join(
                [
                    key + " " + str(param_dict[key])
                    for key in list(param_dict.keys())[:]
                ]
            )
            verboseprint(command)
            answer = os.system(command)
            if answer != 0:
                message(
                    d,
                    """STIFF encountered an error. Please verify your
                              images/parameters and enter verbose mode
                              (shift+V) for more precision about the error.""",
                )
                sys.exit()
            d.set("frame delete all ; rgb")
            d.set(
                "tiff %s"
                % (
                    os.path.join(
                        os.path.dirname(path1), param_dict["OUTFILE_NAME"]
                    )
                )
            )

            for color in ["red", "green", "blue"]:
                d.set("rgb %s ; scale minmax ; scale linear" % (color))

    return


def cosmology_calculator(xpapoint=None, argv=[]):
    """Plot the different information for a given cosmology
    """
    from dataphile.graphics.widgets import Slider
    import numpy as np

    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv, required=False)

    cosmology, redshift, H0, Omega_m, Ode0, uncertainty = (
        "LambdaCDM",
        "0.7",
        70,
        0.30,
        0.7,
        "H0:1",
    )
    redshift = np.array(redshift.split("-"), dtype=float)
    if cosmology == "w0waCDM":
        from astropy.cosmology import w0waCDM as cosmol
    elif cosmology == "w0wzCDM":
        from astropy.cosmology import w0wzCDM as cosmol
    elif cosmology == "wpwaCDM":
        from astropy.cosmology import wpwaCDM as cosmol
    elif cosmology == "LambdaCDM":
        from astropy.cosmology import LambdaCDM as cosmol
    elif cosmology == "wCDM":
        from astropy.cosmology import wCDM as cosmol

    if cosmology == "WMAP9":
        from astropy.cosmology import WMAP9 as cosmo
    elif cosmology == "WMAP7":
        from astropy.cosmology import WMAP7 as cosmo
    elif cosmology == "WMAP5":
        from astropy.cosmology import WMAP5 as cosmo
    elif cosmology == "Planck13":
        from astropy.cosmology import Planck13 as cosmo
    elif cosmology == "Planck15":
        from astropy.cosmology import Planck15 as cosmo

    elif (cosmology == "wCDM") or (cosmology == "LambdaCDM"):
        H0, Omega_m, Ode0 = np.array([H0, Omega_m, Ode0], dtype=float)
        param, uncertainty = uncertainty.split(":")
        uncertainty = float(uncertainty)
        cosmo = cosmol(H0=H0, Om0=Omega_m, Ode0=Ode0)
        verboseprint("param, uncertainty = ", param, uncertainty)
        if param.lower() == "h0":
            cosmo1 = cosmol(
                H0=H0 * (1 - 0.01 * uncertainty), Om0=Omega_m, Ode0=Ode0
            )
            cosmo2 = cosmol(
                H0=H0 * (1 + 0.01 * uncertainty), Om0=Omega_m, Ode0=Ode0
            )
        elif param.lower() == "om0":
            cosmo1 = cosmol(
                H0=H0, Om0=Omega_m * (1 - 0.01 * uncertainty), Ode0=Ode0
            )
            cosmo2 = cosmol(
                H0=H0, Om0=Omega_m * (1 + 0.01 * uncertainty), Ode0=Ode0
            )
        else:
            cosmo1 = cosmol(
                H0=H0, Om0=Omega_m, Ode0=Ode0 * (1 - 0.01 * uncertainty)
            )
            cosmo2 = cosmol(
                H0=H0, Om0=Omega_m, Ode0=Ode0 * (1 + 0.01 * uncertainty)
            )
    elif cosmology == "default_cosmology":
        from astropy.cosmology import default_cosmology

        cosmo = default_cosmology.get()
        cosmo1 = cosmo2 = cosmo
        verboseprint(cosmo1)
        verboseprint(cosmo2)
    # TODO make it quicker
    info = {}
    info["luminosity_distance"] = cosmo.luminosity_distance(redshift)
    info["age"] = cosmo.age(redshift)
    info["kpc_proper_per_arcsec"] = 1 / cosmo.arcsec_per_kpc_proper(redshift)
    info["kpc_comoving_per_arcsec"] = 1 / cosmo.arcsec_per_kpc_comoving(redshift)
    info["arcsec_per_proper_kpc"] = cosmo.arcsec_per_kpc_proper(redshift)
    info["arcsec_per_comoving_kpc"] = cosmo.arcsec_per_kpc_comoving(redshift)
    info["angular_diameter_distance"] = cosmo.angular_diameter_distance(redshift)
    info["comoving_distance"] = cosmo.comoving_distance(redshift)
    info["comoving_volume"] = cosmo.comoving_volume(redshift)
    info["lookback_distance"] = cosmo.lookback_distance(redshift)
    info["lookback_time"] = cosmo.lookback_time(redshift)
    info["scale_factor"] = cosmo.scale_factor(redshift)
    info["efunc"] = cosmo.efunc(redshift)

    zs = np.linspace(0, 5, 50)

    if type(redshift) is float:
        redshifts = np.array([redshift], dtype=float)
    else:
        redshifts = np.array(redshift, dtype=float)
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 3, figsize=(18, 9.5), sharex=True)
    a = 0.08
    redshift_ = Slider(
        figure=fig,
        location=[0.1, 0.14 - a, 0.8, 0.03],
        label="$z$",
        bounds=(0, 5),
        init_value=redshift,
    )
    H0_ = Slider(
        figure=fig,
        location=[0.1, 0.12 - a, 0.8, 0.03],
        label="$H_0$",
        bounds=(0, 100),
        init_value=H0,
    )
    Omega_m_ = Slider(
        figure=fig,
        location=[0.1, 0.10 - a, 0.8, 0.03],
        label=r"$\Omega_m$",
        bounds=(0, 1),
        init_value=Omega_m,
    )
    Ode0_ = Slider(
        figure=fig,
        location=[0.1, 0.08 - a, 0.8, 0.03],
        label="$Ode_0$",
        bounds=(0, 1),
        init_value=Ode0,
    )

    t = "U4"
    l = " - "
    p = ax1[0].plot(
        zs,
        cosmo.angular_diameter_distance(zs) / 1000,
        label="Angular diameter distance = %s"
        % (
            l.join(
                np.array(
                    cosmo.angular_diameter_distance(redshifts).value / 1000,
                    dtype=t,
                )
            )
        ),
    )
    ax1[0].set_ylabel("Gpc")
    ax1[0].legend(loc="upper left")
    for redshift in redshifts:
        a10 = ax1[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.angular_diameter_distance(redshift) / 1000).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a10v = ax1[0].plot(
            [0, redshift],
            np.ones(2)
            * (cosmo.angular_diameter_distance(redshift) / 1000).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
    p10 = ax1[0].plot(
        zs,
        cosmo.comoving_distance(zs) / 1000,
        label="Comoving distance = %s"
        % (
            l.join(
                np.array(
                    cosmo.comoving_distance(redshifts).value / 1000, dtype=t
                )
            )
        ),
    )
    for redshift in redshifts:
        a10_ = ax1[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.comoving_distance(redshift) / 1000).value],
            linestyle="dotted",
            color=p10[0].get_color(),
            label="_nolegend_",
        )
        a10v_ = ax1[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.comoving_distance(redshift) / 1000).value,
            linestyle="dotted",
            color=p10[0].get_color(),
            label="_nolegend_",
        )

    p11 = ax1[1].plot(
        zs,
        cosmo.luminosity_distance(zs) / 1000,
        label="Luminosity distance = %s"
        % (
            l.join(
                np.array(
                    cosmo.luminosity_distance(redshifts).value / 1000, dtype=t
                )
            )
        ),
    )
    ax1[1].set_ylabel("Gpc")
    ax1[1].legend(loc="upper left")
    for redshift in redshifts:
        a11 = ax1[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.luminosity_distance(redshift) / 1000).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a11v = ax1[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.luminosity_distance(redshift) / 1000).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p12 = ax1[2].plot(
        zs,
        cosmo.critical_density(zs) / 1e-29,
        label="Critical density = %s"
        % (
            l.join(
                np.array(
                    cosmo.critical_density(redshifts).value / 1e-29, dtype=t
                )
            )
        ),
    )
    ax1[2].set_ylabel("10e-29 g/cm^3")
    ax1[2].legend(loc="upper left")
    for redshift in redshifts:
        a12 = ax1[2].plot(
            redshift * np.ones(2),
            [0, (cosmo.critical_density(redshift) / 1e-29).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a12v = ax1[2].plot(
            [0, redshift],
            np.ones(2) * (cosmo.critical_density(redshift) / 1e-29).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p20 = ax2[0].plot(
        zs,
        cosmo.comoving_volume(zs) / 1e9,
        label="Comoving volume = %s"
        % (
            l.join(
                np.array(cosmo.comoving_volume(redshifts).value / 1e9, dtype=t)
            )
        ),
    )
    ax2[0].set_ylabel("Gpc^3")
    ax2[0].legend(loc="upper left")
    for redshift in redshifts:
        a20 = ax2[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.comoving_volume(redshift) / 1e9).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a20v = ax2[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.comoving_volume(redshift) / 1e9).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p21 = ax2[1].plot(
        zs,
        cosmo.lookback_time(zs),
        label="Lookback time = %s"
        % (l.join(np.array(cosmo.lookback_time(redshifts).value, dtype=t))),
    )
    for redshift in redshifts:
        a21 = ax2[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.lookback_time(redshift)).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a21v = ax2[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.lookback_time(redshift)).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p21_ = ax2[1].plot(
        zs,
        cosmo.age(zs),
        label="age = %s"
        % (l.join(np.array(cosmo.age(redshifts).value, dtype=t))),
    )
    ax2[1].legend(loc="upper left")
    ax2[1].set_ylabel("Gyr")
    for redshift in redshifts:
        a21_ = ax2[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.age(redshift)).value],
            linestyle="dotted",
            color=p21_[0].get_color(),
            label="_nolegend_",
        )
        a21v_ = ax2[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.age(redshift)).value,
            linestyle="dotted",
            color=p21_[0].get_color(),
            label="_nolegend_",
        )

    p22 = ax2[2].plot(
        zs,
        cosmo.distmod(zs),
        label="Dist mod (mu) = %s"
        % (l.join(np.array(cosmo.distmod(redshifts).value, dtype=t))),
    )
    ax2[2].legend()
    ax2[2].set_ylabel("mag")
    for redshift in redshifts:
        a22 = ax2[2].plot(
            redshift * np.ones(2),
            [0, (cosmo.distmod(redshift)).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a22v = ax2[2].plot(
            [0, redshift],
            np.ones(2) * (cosmo.distmod(redshift)).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p30 = ax3[0].plot(
        zs,
        cosmo.efunc(zs),
        label="efunc = %s" % (l.join(np.array(cosmo.efunc(redshifts), dtype=t))),
    )
    ax3[0].set_ylabel("E(z)")
    ax3[0].legend(loc="upper left")
    for redshift in redshifts:
        a30 = ax3[0].plot(
            redshift * np.ones(2),
            [0, (cosmo.efunc(redshift))],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a30v = ax3[0].plot(
            [0, redshift],
            np.ones(2) * (cosmo.efunc(redshift)),
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p31 = ax3[1].plot(
        zs,
        cosmo.scale_factor(zs),
        label="Scale factor = %s"
        % (l.join(np.array(cosmo.scale_factor(redshifts), dtype=t))),
    )
    ax3[1].legend(loc="upper left")
    ax3[1].set_xlabel("Redshift")
    ax3[1].set_ylabel("a")
    for redshift in redshifts:
        a31 = ax3[1].plot(
            redshift * np.ones(2),
            [0, (cosmo.scale_factor(redshift))],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a31v = ax3[1].plot(
            [0, redshift],
            np.ones(2) * (cosmo.scale_factor(redshift)),
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )

    p32 = ax3[2].plot(
        zs,
        1 / cosmo.arcsec_per_kpc_proper(zs),
        label="Proper = %s"
        % (
            l.join(
                np.array(
                    1 / cosmo.arcsec_per_kpc_proper(redshifts).value, dtype=t
                )
            )
        ),
    )
    for redshift in redshifts:
        a32 = ax3[2].plot(
            redshift * np.ones(2),
            [0, 1 / (cosmo.arcsec_per_kpc_proper(redshift)).value],
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
        a32v = ax3[2].plot(
            [0, redshift],
            np.ones(2) * 1 / (cosmo.arcsec_per_kpc_proper(redshift)).value,
            linestyle="dotted",
            color=p[0].get_color(),
            label="_nolegend_",
        )
    p32_ = ax3[2].plot(
        zs,
        1 / cosmo.arcsec_per_kpc_comoving(zs),
        label="Comoving = %s"
        % (
            l.join(
                np.array(
                    1 / cosmo.arcsec_per_kpc_comoving(redshifts).value, dtype=t
                )
            )
        ),
    )
    ax3[2].legend(loc="upper left")
    ax3[2].set_ylabel("'/kpc")
    for redshift in redshifts:
        a32_ = ax3[2].plot(
            redshift * np.ones(2),
            [0, 1 / (cosmo.arcsec_per_kpc_comoving(redshift)).value],
            linestyle="dotted",
            color=p32_[0].get_color(),
            label="_nolegend_",
        )
        a32v_ = ax3[2].plot(
            [0, redshift],
            np.ones(2) * 1 / (cosmo.arcsec_per_kpc_comoving(redshift)).value,
            linestyle="dotted",
            color=p32_[0].get_color(),
            label="_nolegend_",
        )

    dict_ = {"redshift": redshift, "H0": H0, "Omega_m": Omega_m, "Ode0": Ode0}

    def update(val):
        dict_["redshift"] = redshift_.value
        dict_["H0"] = H0_.value
        dict_["Omega_m"] = Omega_m_.value
        dict_["Ode0"] = Ode0_.value
        redshift = redshift_.value
        from astropy.cosmology import LambdaCDM as cosmol

        cosmo = cosmol(H0=dict_["H0"], Om0=dict_["Omega_m"], Ode0=dict_["Ode0"])
        p[0].set_ydata(cosmo.angular_diameter_distance(zs) / 1000)
        p10[0].set_ydata(cosmo.comoving_distance(zs) / 1000)
        p11[0].set_ydata(cosmo.luminosity_distance(zs) / 1000)
        p12[0].set_ydata(cosmo.critical_density(zs) / 1e-29)
        p20[0].set_ydata(cosmo.comoving_volume(zs) / 1e9)
        p21[0].set_ydata(cosmo.lookback_time(zs))
        p21_[0].set_ydata(cosmo.age(zs))
        p22[0].set_ydata(cosmo.distmod(zs))
        p30[0].set_ydata(cosmo.efunc(zs))
        p31[0].set_ydata(cosmo.scale_factor(zs))
        p32[0].set_ydata(1 / cosmo.arcsec_per_kpc_proper(zs))
        p32_[0].set_ydata(1 / cosmo.arcsec_per_kpc_comoving(zs))
        ps = [p, p10, p11, p12, p20, p21, p21_, p22, p30, p31, p32, p32_]
        a_ = [a10, a10_, a11, a12, a20, a21, a21_, a22, a30, a31, a32, a32_]
        av_ = [
            a10v,
            a10v_,
            a11v,
            a12v,
            a20v,
            a21v,
            a21v_,
            a22v,
            a30v,
            a31v,
            a32v,
            a32v_,
        ]
        im_ = [
            (cosmo.angular_diameter_distance(redshift) / 1000).value,
            (cosmo.comoving_distance(redshift) / 1000).value,
            (cosmo.luminosity_distance(redshift) / 1000).value,
            (cosmo.critical_density(redshift) / 1e-29).value,
            (cosmo.comoving_volume(redshift) / 1e9).value,
            (cosmo.lookback_time(redshift)).value,
            (cosmo.age(redshift)).value,
            (cosmo.distmod(redshift)).value,
            cosmo.efunc(redshift),
            cosmo.scale_factor(redshift),
            1 / (cosmo.arcsec_per_kpc_proper(redshift)).value,
            1 / (cosmo.arcsec_per_kpc_comoving(redshift)).value,
        ]
        legends = [
            "Angular diameter distance = %s",
            "Comoving distance = %s",
            "Luminosity distance = %s",
            "Critical density = %s",
            "Comoving volume = %s",
            "Lookback time = %s",
            "age = %s",
            "Dist mod (mu) = %s",
            "efunc = %s",
            "Scale factor = %s",
            "Proper = %s",
            "Comoving = %s",
        ]
        for i, (a, b, c, legendi, psi) in enumerate(
            zip(a_, av_, im_, legends, ps)
        ):
            a[0].set_xdata(redshift * np.ones(2))
            a[0].set_ydata([0, c])
            b[0].set_xdata([0, redshift])
            b[0].set_ydata(np.ones(2) * (c))
            psi[0].set_label(legendi % (np.around(c, 2)))
        for ax in np.array([ax1, ax2, ax3]).ravel():
            ax.legend(loc="upper left")
        fig.canvas.draw_idle()

    redshift_.on_changed(update)
    H0_.on_changed(update)
    Omega_m_.on_changed(update)
    Ode0_.on_changed(update)

    verboseprint(
        "%s : H0=%s, Om0=%s, Ode0=%s, Tcmb0=%s, Neff=%s, Ob0=%s"
        % (
            cosmology,
            cosmo.H0,
            cosmo.Om0,
            cosmo.Ode0,
            cosmo.Tcmb0,
            cosmo.Neff,
            cosmo.Ob0,
        )
    )
    plt.suptitle(
        "%s : H0=%s, Om0=%0.3f, Ode0=%0.3f, Tcmb0=%s, Neff=%0.2f, Ob0=%s"
        % (
            cosmology,
            cosmo.H0,
            cosmo.Om0,
            cosmo.Ode0,
            cosmo.Tcmb0,
            cosmo.Neff,
            cosmo.Ob0,
        ),
        y=1,
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    plt.show()

    for key in info.keys():
        verboseprint("%s : %s" % (key, info[key]))
    return


def convertissor(xpapoint=None, argv=[]):
    """Converts an astropy unit in another one
    """
    import astropy.units as u

    from decimal import Decimal

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-v",
        "--value",
        default="1",
        help="Value to convert",
        type=str,
        required=True,
        metavar="",
    )
    parser.add_argument("-u", "--unit1", default="angstrom", metavar="")
    parser.add_argument("-U", "--unit2", default="meter", metavar="")
    parser.add_argument("-z", "--redshift", default="0", metavar="")
    args = parser.parse_args_modif(argv, required=False)

    unit_dict = u.__dict__
    val, unit1_, unit2_, redshift = (
        args.value,
        args.unit1,
        args.unit2,
        args.redshift,
    )
    try:
        unit1 = unit_dict[unit1_]
    except KeyError:
        unit1 = u.imperial.__dict__[unit1_]
    try:
        unit2 = unit_dict[unit2_]
    except KeyError:
        unit2 = u.imperial.__dict__[unit2_]
    try:
        val2 = (val * unit1).to(unit2)
    except Exception as e:
        verboseprint(e)
        val2 = (val * unit1).to(unit2, equivalencies=u.spectral())
        # TODO add u.parallax(),u.dimensionless_angles(),u.temperature_energy()
        # u.temperature(),u.mass_energy()
    verboseprint(unit1_, unit1, unit2_, unit2)
    d = DS9n(args.xpapoint)
    message(
        d, "%0.2E %s = %0.2E %s" % (Decimal(val), unit1, Decimal(val2), unit2),
    )
    return


def wait_for_n(xpapoint=None):
    """Wait for N in the test suite to go to next function
    """
    while True:
        try:
            d = DS9n(xpapoint, stop=True)
            while d.get("nan") != "grey":
                time.sleep(0.1)
            d.set("nan black")
            return
        except TypeError:
            continue
    return


def download(url, file=tmp_image):
    """Download a file
    """
    from tqdm import tqdm  # , tqdm_gui
    import requests

    try:
        response = requests.get(url, stream=True)
    except requests.exceptions.RequestException as e:
        verboseprint(e)
        return False
    else:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=0.95 * total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(file, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        # progress_bar.close()
        # tqdm_gui.close(progress_bar)
        # progress_bar.display()
        # progress_bar.plt.close(progress_bar.fig)
        # plt.show(block=False)
        # plt.close('all')
        # plt.close(progress_bar.fig)
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            verboseprint("ERROR, something went wrong")
            return False
        else:
            return True


def next_step(xpapoint=None, argv=[]):
    """Goes to next function in the test suite
    """
    parser = create_parser(get_name_doc())
    args = parser.parse_args_modif(argv, required=False)

    d = DS9n(args.xpapoint)
    d.set("nan grey")
    sys.exit()
    return


def fitsconverter(fname, savefile=tmp_image):
    """Convert tiff images to fits files
    """
    from skimage import io
    import numpy as np

    img = io.imread(fname)
    if len(img.shape) > 3:
        if 3 in img.shape:
            new_image = np.nanmedian(
                img, axis=np.where(np.array(img.shape) == 3)[0][0]
            )
        else:
            raise ("Format of the image not possible")
    fitswrite(new_image, savefile)
    return savefile


def open_file(xpapoint=None, filename=None, argv=[]):
    """Open file(s) in DS9 in an easier way [DS9 required]
    """
    parser = create_parser(get_name_doc(), path=False)
    parser.add_argument(
        "-p",
        "--path",
        help="Path of the images to load, regexp accepted",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="Slice",
        help="Way to open the images",
        type=str,
        metavar="",
    )
    parser.add_argument(
        "-c",
        "--clip",
        default="0",
        help="Open screenshot image from mac",
        metavar="",
    )
    args = parser.parse_args_modif(argv)

    d = DS9n(args.xpapoint)

    if filename is None:
        filename, type_, clip = args.path, args.type, args.clip  # sys.argv[3:]
    else:
        clip, type_ = 0, "Slice"
    if clip == "1":
        from PIL import ImageGrab

        img = ImageGrab.grabclipboard()
        if img is None:
            message(
                d,
                """No image in the clipboard. Take a screenshot
                          and rerun the task.""",
            )
            sys.exit()

        else:
            img.save("/tmp/clipboard_image.jpg", "JPEG")
            filename = "/tmp/clipboard_image.jpg"
    filenames = globglob(filename, ds9_im=False)
    d.set("tile yes")
    if d.get("file") == "":
        d.set("frame delete")
    path = 1
    while (filenames == []) & (path != ""):
        path = get(
            d,
            "File not found, please verify your path. You can use a regexp",
            exit_=True,
        )
        filenames = globglob(path, ds9_im=False)
    if path == "":
        sys.exit()
    for filename in filenames:
        if os.path.isfile(filename):
            filename = filename.replace(" ", "\ ")
            verboseprint("Opening = ", filename)
            if ".reg" in os.path.basename(filename):
                d.set("regions load {}".format(filename))

            elif type_ == "Slice":
                try:
                    if (filename[-4:].lower() == ".jpg") or (
                        filename[-4:].lower() == "jpeg"
                    ):
                        d.set("jpeg {}".format(filename))
                        print("here")
                    elif filename[-4:].lower() == ".png":
                        d.set("png {}".format(filename))  #
                    elif (filename[-5:].lower() == ".tiff") | (
                        filename[-4:].lower() == ".tif"
                    ):
                        d.set("tiff {}".format(filename))  #
                    elif filename[-4:].lower() == ".gif":
                        d.set("gif {}".format(filename))  #
                    else:
                        d.set("fits new {}".format(filename))
                except ValueError:
                    message(
                        d,
                        "Could not open %s as slice. Please verify your file."
                        % (filename),
                    )
                    sys.exit()
                    d.set("frame delete")
            elif type_ == "Multi-Frames-As-Cube":
                d.set("mecube new {}".format(filename))
            elif type_ == "Multi-Frames":
                d.set("multiframe {}".format(filename))
            elif type_ == "CUBE":
                d.set("cube new")
                d.set("file {}".format(filename))

            elif type_ == "RGB":
                d.set("rgb")
                if (filename[-4:].lower() == ".jpg") or (
                    filename[-4:].lower() == "jpeg"
                ):
                    d.set("jpeg {}".format(filename))
                elif filename[-4:].lower() == ".png":
                    d.set("png {}".format(filename))  #
                elif (filename[-5:].lower() == ".tiff") | (
                    filename[-4:].lower() == ".tif"
                ):
                    d.set("tiff {}".format(filename))  #
                elif filename[-4:].lower() == ".gif":
                    d.set("gif {}".format(filename))  #

                else:
                    d.set("slice new {}".format(filename))

            elif type_ == "PRISM":
                if (filename[-5:].lower() == ".fits") or (
                    filename[-4:].lower() == ".fit"
                ):
                    d.set("prism " + filename)
                elif filename[-4:].lower() == ".csv":
                    d.set("prism import csv " + filename)
                elif filename[-4:].lower() == ".tsv":
                    d.set("prism import tsv " + filename)
                elif filename[-4:].lower() == ".xml":
                    d.set("prism import xml " + filename)
                elif filename[-4:].lower() == ".vot":
                    d.set("prism import vot " + filename)

            elif type_ == "IMPORT-3D":
                filename = fitsconverter(filename)
                d.set("multiframe {}".format(filename))

        else:
            verboseprint(
                bcolors.BLACK_RED
                + "File not found, please verify your path"
                + bcolors.END
            )

            message(d, "File not found, verify your path: %s" % (filename))
            sys.exit()
    return


def photometric_analysis_tutorial(xpapoint=None, i=0, n=1):
    """ Launch the photometric_analysis_tutorial on DS9
    """
    from shutil import which

    d = DS9n(xpapoint)
    verboseprint(
        """
              *******************************************
              *      Photometric Analysis Tutorial      *
              *******************************************
* This tutorial will show you a few functions that will help you perform
* photometric analysis in your images. It will go through the following functions:
Centering - Aperture Photometry - SExtractor (if available)
                                [Next]""",
        verbose="1",
    )

    wait_for_n(xpapoint)
    message(d, "Now let us do some centering and aperture photometry!")
    d.set("pan to 175 300 ; zoom to 2 ; regions delete all")
    verboseprint(
        """********************************************************************************
                               Centering (C)
                Instrumentation / AIT -> Centering [or Shift+C]
********************************************************************************
* This function allows you to center a region on sources using different
* algorithms, you might need to do [Shift+S] to make the sources appear.
 %i/%i - Encircle small galaxies/stars with DS9 regions, and select them all by
        hitting Cmd+a. They must be small enough to not include other objetcs.
 %i/%i - Hit [Next] when it is done! """
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2

    wait_for_n(xpapoint)

    verboseprint(
        """ %i/%i - Select the algorithm you want to use and click OK."""
        % (i, n),
        verbose="1",
    )
    i += 1

    while getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region
                       before hitting n. Please make sure to click on the
                       region after creating it and hit n""",
        )
        wait_for_n(xpapoint)
    d.set('analysis task "Centering (C)"')

    verboseprint(
        """* Well done!
********************************************************************************
                         Aperture Photometry
         Astronomical Softwares -> Photutils -> Aperture Photometry
********************************************************************************
 %i/%i - Re-select all the regions (Cmd+a) to perform aperture photometry
        on these centered regions.
        Hit [Next] when it is done!
        The zero point magnitude is 30 for this image."""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)
    while getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region
                       before hitting n. Please make sure to click on the
                       region after creating it and hit n""",
        )
        wait_for_n(xpapoint)
    d.set('analysis task "Aperture Photometry"')

    verboseprint(
        """* Perfect!
%i/%i - Hit [Next] when it is done!"""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)

    if which("sex") is not None:
        message(d, "Now let us use sextractor as it seems to be installed!")
        verboseprint(
            """Well done!\n
********************************************************************************
                               SExtractor
         Astronomical Softwares -> SExtractor -> SExtractor
********************************************************************************
* SExtractor is a very famous source extraction code that will analyze the
* entire image to try to detect sources and estimate its photometry.
* Let's use this master piece software with DS9 and compare the result.
%i/%i - You can put 30 in the field MAG_ZEROPOINT.
    You can play with the other parameters or let them as they are."""
            % (i + 1, n),
            verbose="1",
        )
        i += 1
        d.set('analysis task "SExtractor "')

        verboseprint(
            """* Well done!
%i/%i - The SExtractor catalog has been saved to $path_cat.fits:
         %s
         You can open it with TOPCAT.
%i/%i - The magnitudes of the objects are also saved in the ds9 regions.
         Select them (Cmd+a) and go to Region->Font->10 to display them!
         You can compare them to the aperture photometry you computed before!
%i/%i - Hit [Next] when you want to go to next function."""
            % (
                i,
                n,
                os.path.join(
                    resource_filename("pyds9plugin", "Images"), "stack_cat.fits",
                ),
                i + 1,
                n,
                i + 2,
                n,
            ),
            verbose="1",
        )
        i += 3
        wait_for_n(xpapoint)
    else:
        verboseprint(
            """* Perfect!
* I wanted to run SExtractor but does not seem to be  installed on your machine!
* If you want to install it run brew install brewsci/science/sextractor on a mac,
* or visit https://www.astromatic.net/software/sextractor for linux!""",
            verbose="1",
        )
    message(d, "Let us do some 2D gaussian fitting!")
    d.set("regions delete all ; zoom to 2")

    verboseprint(
        """Well done!\n
********************************************************************************
                             Fit Gaussian 2D
         Generic functions -> Image Processing -> Fit Gaussian 2D
********************************************************************************
%i/%i - Create a region over a star or galaxy and select it.
         It must be small enough to encircle only one source.
%i/%i - Then hit [Next] to use the 2D gaussian fit function!
         Don't forget that you can smooth the data before if needed."""
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2
    wait_for_n(xpapoint)
    while getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region
                       before hitting n. Please make sure to click on the
                       region after creating it and hit n""",
        )
        wait_for_n(xpapoint)
    d.set('analysis task  "Interactive 2D Gaussian Fitting"')

    verboseprint(
        """* Perfect!
* As you can see it also created an ellipse around the region!
* The major/minor axis are the gaussian's FWHMs, the angle is also saved.""",
        verbose="1",
    )
    return


def imag_quality_assesment(xpapoint=None, i=0, n=1):
    """Launches the imag_quality_assesment on DS9
    """
    d = DS9n(xpapoint)
    verboseprint(
        """
              *******************************************
              *    Image Quality Assesment Tutorial     *
              *******************************************
* This tutorial will show you a few functions that will help you asses
* the image quality in your images. It will go through the following functions:
Fit Gaussian 2D - Radial Profile -  Lock / Unlock Frames - Throughfocus
                                [Next]""",
        verbose="1",
    )
    wait_for_n(xpapoint)
    message(d, "Let us do some Radial profile!")
    d.set("regions delete all ; zoom to 2")

    verboseprint(
        """********************************************************************************
                             Radial Profile
         Instrumentation / AIT -> Focus -> Radial Profile [or r]
********************************************************************************

* This function is run on the selected region (same than before then).
* You can change the centering algortihm if you want. The source diameter needs
* only to be changed when you want to fit the image of an unresolved source by
* your instrument, such as a fiber or a laser. Then the profile is fitted
* by the convolution of a gaussian by a disk of this diameter!
* You can check the log box on the plot window to have a logarithmic y scale.
                    [Next] """,
        verbose="1",
    )

    wait_for_n(xpapoint)
    verboseprint(
        """* As you can see it plots both the radial profile and the encircled energy
* of the encircled object up to the region's radius.
* It shows several image quality estimators:
*    Moffat fit, gaussian fit, FWHM, 50%-80% Encircled energy diameters...
""",
        verbose="1",
    )
    while getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region
                      before hitting [Next]. Please make sure to click on
                      the region after creating it and hit n""",
        )

        wait_for_n(xpapoint)

    d.set('analysis task "Radial Profile (r)" ')

    verboseprint(
        """* To compute the image quality of your data, you must select one
* of the most compact (small) source in your image and run the radial profile.
* On the figure you can check the log box to have a logarithmic y scale.

%i/%i - If you want to do it by yourself create & select a region on a very
         compact spot and run the function (r). If you want to go to the next
         function close the window and hit [Next]!"""
        % (i, n),
        verbose="1",
    )
    i += 1
    wait_for_n(xpapoint)
    message(d, "Lets open several images to do some through-focus analysis!")

    verboseprint(
        """********************************************************************************
                                Open Image (O)
              Generic functions -> Setups -> Open Image [or O]
********************************************************************************
%i/%i - You can use regular expression to open several files bit hitting Shift+O
 Copy this path: %s
%i/%i - Hit [Next] to run the function"""
        % (
            i,
            n,
            resource_filename("pyds9plugin", "Images/stack????????.fits"),
            i + 1,
            n,
        ),
        verbose="1",
    )
    i += 2

    wait_for_n(xpapoint)
    d.set("frame delete all ; frame new")
    verboseprint(
        """%i/%i - Then paste it in the RegExp path field and let Type=Slice
* If you do not see any image, the cuts might not be right,
* hit shift+s to change it automatically.
%i/%i - Hit [Next] to go to next function."""
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2
    d.set('analysis task "Open Image (O)"')

    wait_for_n(xpapoint)

    verboseprint(
        """********************************************************************************
                             Lock / Unlock Frames
       Generic functions -> Setups -> Lock / Unlock Frames [or Shift+L]
********************************************************************************
* It is quite slow to match all the tiles of DS9 display at once.
* This tool allows you to do it simply and at once
* As there is not any WCS header in the images,
* there is no need to change the default parameters.
%i/%i - Click on OK and then [Next] when you want to go to the next function."""
        % (i, n),
        verbose="1",
    )
    i += 1

    d.set('analysis task "Lock/Unlock Frames (L)"')
    wait_for_n(xpapoint)
    message(d, """Let us do a throughfocus analysis!""")

    verboseprint(
        """********************************************************************************
                                  Throughfocus
              Instrumentation / AIT -> Focus -> Throughfocus
********************************************************************************
* The following throughfocus function will compute several estimators
* of image quality on every image.
%i/%i - Please define & select a circle region around the best focused PSF.
           [Next]"""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)

    verboseprint(
        """%i/%i - On the RegExp path field you can either paste
         the same pathname pattern as before or let it blank so that the
         function only uses images displayed in DS9
%i/%i - If you put the path then select alpha-numerical, else select DS9 -> OK
* next time, if throughfocus is not in alpha-numerical or time order
* you can change their display on DS9 menu: Frame -> Move frame
* Do not check the WCS box, must be used when available WCS information
* and that the spots are on the same sky region not pixel.
* Then [Next]!"""
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2

    while getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region
                      before hitting n. Please create a circle on a close to
                      focus spot, select it and press n.""",
        )
        wait_for_n(xpapoint)
    d.set('analysis task "Throughfocus  "')
    wait_for_n(xpapoint)
    return


def image_processing_tutorial(xpapoint=None, i=0, n=1):
    """Launches the image_processing_tutorial on DS9
    """
    d = DS9n(xpapoint)
    verboseprint(
        """
              *******************************************
              *        Image Processing Tutorial        *
              *******************************************

* This tutorial will show you several functions that will show you how to take
* advantage of very useful image processing tools. It will go through the following
* functions: Interactive 1D Fitting On Plot - Execute Python Command
                                [Next]""",
        verbose="1",
    )
    wait_for_n(xpapoint)
    message(d, """Now, let us try some interactive 1D fitting!""")

    verboseprint(
        """********************************************************************************
                             Interactive 1D Fitting On Plot
    Generic functions -> Image Processing -> Interactive 1D Fitting On Plot
********************************************************************************

* As a >5 pameters fit is unlikely to converge without a close initial,
* this package allows you to perform interactive 1D fitting
* with user adjustment of the initial fitting parameters!
* You might need to hit [Shift+S] to make appear some features in the image.
* You can check the log box on the plot window to have a logarithmic y scale.
* This function works on any DS9 plot (histograms or even plots generated by
  by this plugin like a radial profile or la light curve

%i/%i - Create a projection (Region->Shape->Projection) on a shape you want
         to fit. Move it until you have a fittable feature (several gaussians,
         slope, exponential or sum of them...). You can stack the projection
         with the small square on the center of your line!
%i/%i - When you are happy with the shape you have hit Next to run the function!
"""
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2

    wait_for_n(xpapoint)
    while d.get("plot") == "":
        message(
            d,
            """Please create a plot by creating a
                          Region->Shape->Projection to run this function.
                          Hit n when it is done.""",
        )
        wait_for_n(xpapoint)
    verboseprint(
        """%i/%i - Choose the background and the number of gaussians you want
         to fit the data with.
%i/%i - Now, you adjust each parameter of each feature! Change by hand the
         different parameters to fit the data and then click on Fit to run the
         fitting least square function. Then you can read the parameters of the
         features that can help you asses image quality or other information.
* When you are done, close the fitting figure and click on [Next]."""
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2
    d.set('analysis task "Interactive 1D Fitting On Plot"')
    wait_for_n(xpapoint)
    message(d, """Now let us use a basic python interpretor!""")
    verboseprint(
        """********************************************************************************
                             Execute Python Command
        Generic functions -> Image Processing -> Execute Python Command
********************************************************************************

* This command allows you to modify your image by interpreting a python command
* Enter a basic command in the Expression field such as:
*    ds9=ds9[:100,:100]                           -> Trim the image
*    ds9+=np.random.normal(0,0.5*ds9.std(),size=ds9.shape)  -> Add noise to the image
*    ds9[ds9>2]=np.nan                            -> Mask a part of the image
*    ds9=1/ds9, ds9+=1, ds9-=1                    -> Different basic expressions
*    ds9=convolve(ds9,np.ones((1,9)))[1:-1,9:-9]  -> Convolve unsymetrically
*    ds9+=np.linspace(0,1,ds9.size).reshape(ds9.shape) -> Add background
*    ds9+=30*(ds9-gaussian_filter(ds9, 1))        -> Sharpening
*    ds9=np.hypot(sobel(ds9,axis=0,mode='constant'),sobel(ds9,axis=1,mode='constant')) -> Edge Detection
*    ds9=np.abs(fftshift(fft2(ds9)))**2           -> FFT
*    ds9=correlate2d(ds9.astype('uint64'),ds9,boundary='symm',mode='same') -> Autocorr
*    ds9=interpolate_replace_nans(ds9, Gaussian2DKernel(x_stddev=5, y_stddev=5)) -> Interpolate NaNs



%i/%i - Copy the expression you want to test. [Next]"""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)
    verboseprint(
        """%i/%i - Paste it in the expression field (or invent your own)
           Do not put anything in the other fields."""
        % (i, n),
        verbose="1",
    )
    i += 1

    d.set('analysis task "Python Command/Macro"')
    verboseprint(
        """Well Done!
* You also have the possibility to run a python macro. Several example Macros
* are available here (copy it):
%s
* But you can also write your own
* The image is avilable via 'ds9' variable, you can open other images in the
* python file if needed and import any package your want.

%i/%i - Test one of the Macros avilable in the path above.
                     [Next]"""
        % (resource_filename("pyds9plugin", "Macros"), i, n),
        verbose="1",
    )

    wait_for_n(xpapoint)

    d.set('analysis task "Python Command/Macro"')

    verboseprint(
        """* Great!
%i/%i - The last field is to apply your python expression to several images!
        To do so, you can write their a regular expression matching the files on
        which you want to run the command! Try it or [Next]."""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)
    return


def generic_tools_tutorial(xpapoint=None, i=0, n=1):
    """Launches the generic_tools_tutorial on DS9
    """
    d = DS9n(xpapoint)
    d.set("nan black")

    verboseprint(
        """              *******************************************
              *          Generic Tools Tutorial         *
              *******************************************

* This tutorial will show you a few functions that will help you interacting
* with your fits images. It will go through the following functions:
* Change Display Parameters - Plot Region In 3D
* Create Header Catalog  - Filtering & organizing images

Please take some time to adjust the scale/threshold/color to improve the image
rendering in order to the objects in the image. When it is done,
                                [Next]""",
        verbose="1",
    )
    wait_for_n(xpapoint)
    message(
        d,
        """Now let me show you a much easier and quicker way to change
the display settings at once! This function will make
you gain a lot of time.""",
    )

    verboseprint(
        """********************************************************************************
                               Change settings at once
  (Analysis->)Generic functions->Setup->Change display parameters [or Shift+S]
********************************************************************************\n
 %i/%i - First, click on OK to run the function with the default parameters."""
        % (i, n),
        verbose="1",
    )
    i += 1

    d.set('analysis task "Change Display Parameters (S) "')

    verboseprint(
        """ %i/%i - Now create a region in a dark area, SELECT IT!
        and re-run the function (Shift+S). As you will see, the scale's
        thresholds for the image will be computed based on the encircled data!


         If you want to continue changing the parameters, re-run the function.
         (If you want to fo back to previous thresholds, unselect the region
         before runing it).
         Else [Next]"""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)
    d.set("pan to 630 230 ; zoom to 1")

    verboseprint(
        """********************************************************************************
                               Plot Region In 3D
                   Generic functions -> Plot Region In 3D
********************************************************************************
* Colormaps can sometimes be misleading, this function will plot in 3D
* the data you encircle with a region to help you see the variations.
* You can check the log box on the plot window to have a logarithmic z scale.
 %i/%i - Please create & select a relatively small region (d<500p), then [Next]"""
        % (i, n),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)
    while getregion(d, selected=True) is None:
        message(
            d,
            """It seems that you did not create or select the region
before hitting n. Please make sure to click on the
region after creating it and hit n""",
        )
        wait_for_n(xpapoint)
    a = d.set('analysis task "Plot Region In 3D"')  # ;time.sleep(3)
    wait_for_n(xpapoint)
    # plot_3d(d)
    verboseprint(
        """* Well done!
* If you smooth the image in DS9 is it will plot it as it is displayed!
* You can use this function on circle or box regions.
*If you run it on a 3D image, it will create a 3D plot color coded with the flux.
 %i/%i - Please create a reagion """
        % (i, n),
        verbose="1",
    )
    i += 1
    reg = resource_filename("pyds9plugin", "Images")
    if os.path.exists(os.path.join(reg, "m33_hi.fits")) is False:
        a = download(
            url="https://people.lam.fr/picouet.vincent/pyds9plugin/m33_hi.fits",
            file=os.path.join(reg, "m33_hi.fits"),
        )
    if a:
        d.set("frame delete ; file " + os.path.join(reg, "m33_hi.fits"))
        d.set("scale 99.5")
        wait_for_n(xpapoint)
        while getregion(d, selected=True) is None:
            message(
                d,
                """It seems that you did not create or select the region
before hitting n. Please make sure to click on the
region after creating it and hit n""",
            )
            wait_for_n(xpapoint)
        d.set('analysis task "Plot Region In 3D"')
        time.sleep(2)
    else:
        message(
            d,
            "An error occured while dowloading the 3D image. Next time, make sure you have a good internet connection. Going to next function.",
        )

    verboseprint(
        """Well done!

********************************************************************************
                             Create Header Catalog
         Generic functions -> Header & Data Base ->  Create Header Catalog
********************************************************************************
* The following function will take all the images matching the given pattern
* and will concatenate all the header information in a csv catalog.
* It will also add other info: pathm, name, directory, creation date, size.
* You can either give it a folder where you put (a subset of) your images
* or even search your whole documents folder: %s
* If you do not have any fit[s] file here you can use the one of this plugin:
    %s
* Do not worry if you have too many files, we will only take the first 200.
%i/%i - Copy the path you want to use and hit n when you are ready."""
        % (
            os.environ["HOME"] + "/Documents/**/*.fit*",
            os.path.join(resource_filename("pyds9plugin", "**/*.fits")),
            i,
            n,
        ),
        verbose="1",
    )
    i += 1

    wait_for_n(xpapoint)
    verboseprint(
        """%i/%i - Paste the path in the Regexp Path entry.""" % (i, n),
        verbose="1",
    )
    i += 1
    d.set('analysis task "Create Header Data Base"')
    verboseprint(
        """* Great!
%i/%i - You can open the above catalog & analyse it with TOPCAT.
         Please copy this path, you will need it!
%i/%i - Hit n when you are ready to use it!"""
        % (i, n, i + 1, n),
        verbose="1",
    )
    i += 2  # The master header catalog has been saved here: %s
    wait_for_n(xpapoint)
    verboseprint(
        """********************************************************************************
                             Filtering & organizing images
       Generic functions -> Header & Data Base ->  Filtering & organizing images
********************************************************************************
* We are now gonna use the very last function that will help you arrange all
* your fits/fit/FIT images very simply. I will basically use the header catalog
* to arrange your files verifying some conditions and orders!
* Do not worry, we will not displace your files, just create some symbolic links
* to help you see all the images you have!
%i/%i - Paste your header catalog path in the first field.
%i/%i - In the second one enter the way you want to order your images based on
         the fields (floats) that are in the header, eg:
* NAXIS,Directory      -> Differentiate images/cubes and order them by folder
* CTYPE1,CreationDate  -> Differentiate WCS types and order them by dates

%i/%i - In the last one, write the selection that you want to apply to the
        catalog, for instance, eg:
* NAXIS==2  [NAXIS==3/0]        -> Take only images [or cube/table respectively]
* BITPIX==-64 \| NAXIS2>1000    -> Certain bibpix and image heights>10000
* FileSize_Mo>10  \& WCSAXES>1  -> Big images containing WCS header
* EMGAIN>9000 \& EXPTIME<10     -> Likely photon starving images

%i/%i - Check ok when you are ready
* Note that you must use & for AND and \| for OR! """
        % (i, n, i + 1, n, i + 2, n, i + 3, n),
        verbose="1",
    )
    i += 4
    d.set('analysis task "Filtering & organizing images"')

    verboseprint(
        """%i/%i - Go to %s and enjoy your well ordered files!
%i/%i - You can copy paste these instructions somewhere if you want to keep it!
    """
        % (i, n, os.environ["HOME"] + "/DS9QuickLookPlugIn/Subsets", i + 1, n),
        verbose="1",
    )
    i += 2

    return


def test_suite(xpapoint=None, argv=[]):
    """Test suite: run several fucntions in DS9 [DS9 required]
    """
    from subprocess import Popen

    parser = create_parser(get_name_doc())
    parser.add_argument(
        "-t",
        "--tutorial",
        help="Name of the tutorial to launch",
        type=str,
        metavar="",
        choices=[
            "1-Generic-Tools",
            "2-Photometric-Analysis",
            "3-Image-Quality-Assesment",
            "4-Image-Processing",
            "5-All-In-One",
        ],
        required=True,
    )
    args = parser.parse_args_modif(argv, required=False)

    Popen(
        [" DS9Utils button %s" % (xpapoint)],
        shell=True,
        stdin=None,
        stdout=None,
        stderr=None,
        close_fds=True,
    )
    xpapoint = args.xpapoint
    d = DS9n(xpapoint)
    # verbose(xpapoint, verbose="0")
    tutorial = args.tutorial
    tutorial_number = "0"
    if tutorial == "1-Generic-Tools":
        tutorial_number += "1"
    elif tutorial == "2-Photometric-Analysis":
        tutorial_number += "2"
    elif tutorial == "3-Image-Quality-Assesment":
        tutorial_number += "3"
    elif tutorial == "4-Image-Processing":
        tutorial_number += "4"
    if tutorial == "5-All-In-One":
        tutorial_number = "1234"
    d.set("nan black")
    message(
        d,
        """Test suite for beginners: This help will go through most of
the must-know functions of this plug-in. Between each function
some message will appear to explain you the purpose of these
functions and give you some instructions.""",
    )

    verboseprint(
        """********************************************************************************
                               General Instructions
********************************************************************************\n
I will follow you during this tutorial so move me next to the DS9 window but
please do not close me. You can increase the fontsize of this window.

After a function has run you can run it again with different parameters
by launching it from the Analysis menu [always explicited under function's name]

[Next]: When you followed the instruction and I write [Next] please click on the
Next Button (or hit N on the  DS9 window) so that you go to the next function.

                 Please [Next]""",
        verbose="1",
    )
    wait_for_n(xpapoint)
    im = os.path.join(resource_filename("pyds9plugin", "Images"), "stack.fits")
    d.set("frame new ; tile no ; file %s ; zoom to fit" % (im))
    i = 1
    if "1" in tutorial_number:
        generic_tools_tutorial(xpapoint, i=i, n=13)
    if "2" in tutorial_number:
        if "1" in tutorial_number:
            d.set("frame new ; tile no ; file %s ; zoom to fit" % (im))
        photometric_analysis_tutorial(xpapoint, i=i, n=13)
    if "3" in tutorial_number:
        imag_quality_assesment(xpapoint, i=i, n=12)
    if "4" in tutorial_number:
        image_processing_tutorial(xpapoint, i=i, n=15)
    if tutorial == "5-All-In-One":
        verboseprint(
            """
********************************************************************************
         You are now ready to use the DS9 Quick Look plugin by yourself.
       You can access/change the default parameters of each function here:
       %s
********************************************************************************\n"""
            % (resource_filename("pyds9plugin", "QuickLookPlugIn.ds9.ans")),
            verbose="1",
        )

    else:
        verboseprint(
            """
********************************************************************************
             Well done, You completed the %s tutorial!
               Let's try the next one when you have some time!
       You can access/change the default parameters of each function here:
       %s
********************************************************************************\n"""
            % (
                tutorial,
                resource_filename("pyds9plugin", "QuickLookPlugIn.ds9.ans"),
            ),
            verbose="1",
        )
    time.sleep(2)
    kill_long_process(function="DS9Utils.*button")
    sys.exit()
    return


def python_command(xpapoint=None, argv=[]):
    """Interpret a python command and applies it to given image(s)
    """

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-e",
        "--exp",
        help="Expression to process. eg ds9+=1",
        type=str,
        metavar="",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--other_image",
        default=tmp_image,
        help="Path of a second image to use with",
        metavar="",
        type=str,
    )
    parser.add_argument(
        "-o", "--overwrite", default="0", help="Overwrite image", metavar=""
    )
    parser.add_argument(
        "-N",
        "--number_processors",
        default=os.cpu_count() - 2,
        help="Number of processors to use for multiprocessing analysis. Default use your total number of processors - 2.",
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)

    d = DS9n(args.xpapoint)

    path2remove, exp, eval_ = args.other_image, args.exp, 0
    verboseprint("Expression to be evaluated: %s" % (exp))
    path = globglob(args.path, xpapoint)
    verboseprint("path: %s" % (path))
    overwrite = bool(int(args.overwrite))

    if ((int(d.get("block")) > 1) | (d.get("smooth") == "yes")) & (
        len(path) == 1
    ):
        answer = ds9entry(
            args.xpapoint,
            """It seems that your loaded image is modified (smoothed or
               blocked). Do you want to run the analysis on this modified
               image? [y/n]""",
            quit_=False,
        )
        if answer == "y":
            try:
                fitsimage = d.get_pyfits()[0]
                path = [tmp_image]
                fitswrite(fitsimage, tmp_image)
            except TypeError:
                pass

    if (".fit" not in path[0]) & (len(path) == 1):
        try:
            fitsimage = d.get_pyfits()[0]
            path = [tmp_image]
            fitswrite(fitsimage, tmp_image)
        except TypeError:
            pass
    result, name = parallelize(
        function=execute_command,
        parameters=[path2remove, exp, xpapoint, bool(int(eval_)), overwrite, d,],
        action_to_paralize=path,
        number_of_thread=args.number_processors,
    )
    if (len(path) < 2) & (result is not None):
        d.set("frame new ; tile yes ; file " + name)
    return


def button(xpapoint=None):
    """ Creates a pyQt5 button to continue the different DS9 tutorial or quit
    """
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QVBoxLayout,
        QPushButton,
        QLabel,
    )
    from PyQt5.QtGui import QIcon, QPixmap
    from PyQt5.QtCore import pyqtSlot  # <--- add this line

    class Window(QWidget):
        def __init__(self):
            super(Window, self).__init__()
            self.setWindowTitle("DS9")
            self.buttonNext = QPushButton("Next", self)
            self.buttonQuit = QPushButton("Quit tutorial", self)
            self.label = QLabel(self)
            self.d = DS9n(xpapoint)
            self.label.setText("When you are ready click on Next")
            self.buttonNext.clicked.connect(self.handleButtonNext)
            self.buttonQuit.clicked.connect(self.handleButtonQuit)
            layout = QVBoxLayout(self)
            layout.addWidget(self.label)
            layout.addWidget(self.buttonNext)
            layout.addWidget(self.buttonQuit)

        @pyqtSlot()
        def handleButtonNext(self):
            self.d.set("nan grey")

        @pyqtSlot()
        def handleButtonQuit(self):
            from subprocess import Popen

            kill_long_process(function="DS9Utils.*test")
            kill_long_process(function="DS9Utils.*button")

    app = QApplication(sys.argv)
    app.setWindowIcon(
        QIcon(
            QPixmap(
                resource_filename(
                    "pyds9plugin", "doc/ref/features_files/sun.gif"
                )
            )
        )
    )
    window = Window()
    window.show()
    sys.exit(app.exec_())


def maxi_mask(xpapoint=None, argv=[]):
    """Run MaxiMask processing tool on given image(s)
    """
    from astropy.io import fits
    import numpy as np

    parser = create_parser(get_name_doc(), path=True)
    parser.add_argument(
        "-t",
        "--proba_threshold",
        default="0",
        help="Apply a threshold to the probability map",
        type=str,
        choices=["0", "1"],
    )
    parser.add_argument(
        "-b", "--batch_size", default="8", help="Size of the batch", metavar=""
    )
    parser.add_argument(
        "-m",
        "--mask",
        default="0",
        help="Single mask with power of two",
        metavar="",
    )
    parser.add_argument("-n", "--net_path", default="0", metavar="")
    parser.add_argument(
        "-F",
        "--flags",
        default="1-1-1-1-1-1-1-1-1-1-1-1-1-1",
        help="Compute flags for Cosmic Rays, hot CL, dead CL, persistence, Trails, Fringe, Nebulosities. Saturation, Spikes, Overscanned, Bright backgorund, background",
        metavar="",
    )
    parser.add_argument(
        "-P",
        "--priors",
        default="0.0007-0.0008-0.0080-0.000001-0.000001-0.00001-0.006-0.01-0.01-0.0016-0.013-0.005-0.007-0.9",
        help="Priors to use",
        metavar="",
    )
    parser.add_argument(
        "-T",
        "--thresholds",
        default="0.51-0.52-0.5-0.23-0.99-0.66-0.55-0.62-0.45-0.78-0.41-0.37-0.49-0.33",
        help="Thresholds to apply",
        metavar="",
    )
    args = parser.parse_args_modif(argv, required=True)
    print(args.flags)
    path = globglob(args.path, xpapoint=args.xpapoint)
    d = DS9n(args.xpapoint)

    prob, size, single, net = (
        args.proba_threshold,
        int(args.batch_size),
        args.mask,
        args.net_path,
    )
    os.chdir(os.path.dirname(path[0]))
    flags = """CR  %i
HCL %i
DCL %i
HP  %i
DP  %i
P   %i
STL %i
FR  %i
NEB %i
SAT %i
SP  %i
OV  %i
BBG %i
BG  %i""" % (
        *np.array(args.flags.split("-"), dtype=float),
    )

    priors = """CR  %0.7f
HCL %0.7f
DCL %0.7f
HP  %0.7f
DP  %0.7f
P   %0.7f
STL %0.7f
FR  %0.7f
NEB %0.7f
SAT %0.7f
SP  %0.7f
OV  %0.7f
BBG %0.7f
BG  %0.7f""" % (
        *np.array(args.priors.split("-"), dtype=float),
    )

    thresholds = """CR  %0.7f
HCL %0.7f
DCL %0.7f
HP  %0.7f
DP  %0.7f
P   %0.7f
STL %0.7f
FR  %0.7f
NEB %0.7f
SAT %0.7f
SP  %0.7f
OV  %0.7f
BBG %0.7f
BG  %0.7f""" % (
        *np.array(args.thresholds.split("-"), dtype=float),
    )

    verboseprint(flags)
    verboseprint(priors)
    verboseprint(thresholds)
    os.system('echo "%s" > classes.flags' % (flags))
    os.system('echo "%s" > classes.thresh' % (thresholds))
    os.system('echo "%s" > classes.priors' % (priors))

    if len(path) == 1:
        command = (
            """%s %s  -v --single_mask %s --batch_size %i --proba_thresh %s --prior_modif  True  %s"""
            % (
                "/Users/Vincent/opt/anaconda3/bin/python",
                resource_filename("pyds9plugin", "MaxiMask-1.1/maximask.py"),
                bool(int(single)),
                size,
                bool(int(prob)),
                path[0],
            )
        )
    else:
        outfile = open("/tmp/files_maxi_mask.list", "w")
        print >> outfile, "\n".join(str(i) for i in path)
        outfile.close()
        command = (
            """%s %s  -v --single_mask %s --batch_size %i --proba_thresh %s --prior_modif  True  %s"""
            % (
                "/Users/Vincent/opt/anaconda3/bin/python",
                resource_filename("pyds9plugin", "MaxiMask-1.1/maximask.py"),
                bool(int(single)),
                size,
                bool(int(prob)),
                "/tmp/files_maxi_mask.list",
            )
        )

    print(f_string(command))
    import subprocess

    try:
        a = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    for file in path:
        print(a)
        try:
            try:
                a = fits.open(file.replace(".fits", ".masks.fits"))
                b = fits.open(file)
                a[0].header = b[0].header
                a.writeto(file.replace(".fits", ".masks.fits"), overwrite=True)
            except Exception as e:
                verboseprint(e)
            d.set("frame new")
            d.set("file %s" % (file.replace(".fits", ".masks.fits")))
        #        d.set('multiframe %s'%(path.replace('.fits','.mask.fits')))
        except Exception as e:
            print(e)
            print("did not work...")
    return


def maxi_mask_cc(path=None, xpapoint=None):
    """ Runs MaxiMask processing tool on image
    """
    from astropy.io import fits
    from shutil import which

    d = DS9n(xpapoint)
    if path is None:
        path = get_filename(d)
    command = """%s ./MaxiMask-1.1/maximask.py  -v --single_mask %s --batch_size %i
           --proba_thresh %s --prior_modif  True  %s""" % (
        which("python3"),
        bool(int(True)),
        8,
        bool(int(False)),
        path,
    )
    print(command)
    import subprocess

    try:
        a = subprocess.check_output(
            command, shell=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "command '{}' return with error (code {}): {}".format(
                e.cmd, e.returncode, e.output
            )
        )
    print(a)
    try:
        try:
            a = fits.open(path.replace(".fits", ".masks.fits"))
            b = fits.open(path)
            a[0].header = b[0].header
            a.writeto(path.replace(".fits", ".masks.fits"), overwrite=True)
        except Exception as e:
            verboseprint(e)
        d.set("frame new")
        d.set("file %s" % (path.replace(".fits", ".masks.fits")))
    except Exception as e:
        print(e)
        print("did not work...")
    return


def kill_long_process(function="DS9Utils.*"):
    """ Kill pyds9plugin processes during the tutorial
    """
    import subprocess

    c = "ps -eaf|grep '%s' |awk '{ print $2}'|xargs -IAA sh -c 'kill -kill AA'"
    subprocess.Popen(
        [c % (function)],
        shell=True,
        stdin=None,
        stdout=None,
        stderr=None,
        close_fds=True,
    )
    return


def divide_catalog(path, tmp_folder="/tmp", id_="ID", n=10):
    """ Divide an ascii catalog to read it more rapidly
    """

    def file_len(fname):
        with open(fname) as f:
            for i, l in enumerate(f):
                pass
        return i + 1

    size_file = file_len(path)
    lines_per_file = int(size_file / n) + 1
    smallfile = None
    with open(path) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = os.path.join(
                    tmp_folder,
                    os.path.basename(path).split(".")[0]
                    + "_%010d.%s"
                    % (lineno + lines_per_file, path.split(".")[-1]),
                )
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()
    return  # len(cat),np.sum(lens)


def read_big_ascii_table(path, tmp_folder="/tmp", n=10):
    """ Read a big ascii catalog by dividing it to earn time
    """
    import glob
    from astropy.table import vstack
    from astropy.io import ascii
    import os

    divide_catalog(path, tmp_folder=tmp_folder, id_="ID", n=n)
    a = []
    files = glob.glob(
        os.path.join(
            tmp_folder,
            os.path.basename(path).split(".")[0]
            + "_??????????."
            + os.path.basename(path).split(".")[-1],
        )
    )
    files.sort()
    with open(files[0]) as f:
        first_line = f.readline()
    f.close()
    cols = list(filter(("").__ne__, first_line.split(" ")))
    if ["\n"] in cols:
        cols.remove("\n")
    fast_reader = {"parallel": True, "use_fast_converter": True}
    tab = ascii.read(files[0], fast_reader=fast_reader)
    print(tab.colnames, cols)
    print(len(tab.colnames), len(cols))
    print("The new %i tables have %i rows" % (n, len(tab)))
    for pathi in files:
        verboseprint(pathi)
        tab = ascii.read(pathi, fast_reader=fast_reader)
        verboseprint(tab.colnames)
        verboseprint(len(tab.colnames))
        if len(tab.colnames) == len(cols):
            tab.rename_columns(tab.colnames, cols)
        a.append(tab)
        os.remove(pathi)
    tables = vstack(a)
    return tables


def create_reg_contour(path, n=50):
    """ Create contour on some ds9 image
    """
    from astropy.io import fits
    from scipy.ndimage import grey_dilation
    from astropy.wcs import WCS
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import numpy as np

    a = fits.open(path)
    w = WCS(a[0].header)
    ds9 = a[0].data
    plt.figure()
    plt.imshow(ds9)
    if n > 1:
        ds9 = grey_dilation(ds9, size=n).astype(int)
    CS = plt.contour(ds9, levels=1)
    sizex = np.array([cs[:, 0].max() - cs[:, 0].min() for cs in CS.allsegs[0]])
    sizey = np.array([cs[:, 1].max() - cs[:, 1].min() for cs in CS.allsegs[0]])
    size_tot = np.sqrt(np.square(sizex) + np.square(sizey))
    regions = np.array(CS.allsegs[0])[size_tot > 50]  # 500
    name = path.split(".")[0] + "%s.reg" % (n)
    if os.path.isfile(name):
        os.remove(name)
    with open(name, "a") as file:
        file.write(
            """# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
"""
        )
    for i, region in enumerate(tqdm(regions)):
        if region.shape[0] > 99:
            region = region[:: int(region.shape[0] / 50)]
        if region.shape[0] > 99:
            region = region[:: int(region.shape[0] / 50)]
        new_line = (
            "polygon("
            + ",".join(
                [
                    str(np.round(a, 5)) + "," + str(np.round(b, 5))
                    for a, b in zip(
                        w.pixel_to_world(region[:, 0], region[:, 1]).ra.value,
                        w.pixel_to_world(region[:, 0], region[:, 1]).dec.value,
                    )
                ]
            )
            + ")\n"
        )
        with open(name, "a") as file:
            file.write(new_line)
    d = DS9n()
    d.set("regions delete all")
    d.set("regions " + name)
    return


# @fn_timer
# @profile
def main():
    """Main function where the arguments are defined and the other
    functions called
    """
    # "PlotSpectraDataCube": PlotSpectraDataCube,#
    # "StackDataDubeSpectrally": StackDataDubeSpectrally,#
    # "stack_images": stack_images,#stack
    # ,'NextButton':NextButton
    # "build_wcs_header": build_wcs_header,#build_wcs_header
    # "check_file": check_file,#
    # "create_contour_regions": create_contour_regions,
    # "interpolate_nans": interpolate_nans,
    # "ComputeEmGain": compute_gain,#compute_gain
    # "FFT": FFT,#FFT
    # "autocorrelation": autocorrelation,#autocorrelation
    # "analyze_fwhm": analyze_fwhm,#analyze_fwhm
    # "PlotSpectraFilters": PlotSpectraFilters,

    dict_function_generic = {
        "setup": setup,
        "guidance": guidance,
        "fit_gaussian_2d": fit_gaussian_2d,
        "organize_files": organize_files,
        "import_table_as_region": import_table_as_region,
        "add_field_to_header": add_field_to_header,
        "fit_ds9_plot": fit_ds9_plot,
        "test_suite": test_suite,
        "lock": lock,
        "create_header_catalog": create_header_catalog,
        "python_command": python_command,
        "save_region_as_catalog": save_region_as_catalog,
        "mask_regions": mask_regions,
        "create_image_from_catalog": create_image_from_catalog,
        "plot_3d": plot_3d,
        "original_settings": original_settings,
        "next_step": next_step,  #
        "background_estimation": background_estimation,
        "verbose": verbose,
        "open_file": open_file,
        "button": button,  #
        "throw_apertures": throw_apertures,
        "convert_image": convert_image,
        "manual_fitting": manual_fitting,
        # "compute_gain": compute_gain,
        "LoadDS9QuickLookPlugin": LoadDS9QuickLookPlugin,
    }
    dict_function_ait = {
        "center_region": center_region,
        "radial_profile": radial_profile,
        "throughfocus": throughfocus,
        "compute_fluctuation": compute_fluctuation,
        "light_curve": light_curve,
        "explore_throughfocus": explore_throughfocus,
        "fill_regions": fill_regions,
        "trim": trim,
        "column_line_correlation": column_line_correlation,
        "get_depth_image": get_depth_image,
        # "emccd_model": emccd_model,
    }

    dict_function_soft = {
        "ds9_swarp": ds9_swarp,
        "ds9_psfex": ds9_psfex,
        "run_sextractor": run_sextractor,
        "ds9_stiff": ds9_stiff,
        "aperture_photometry": aperture_photometry,
        "extract_sources": extract_sources,
        "cosmology_calculator": cosmology_calculator,
        "Convertissor": convertissor,
        "astrometry_net": astrometry_net,
        "resample": resample,
        "interactive_plotter": interactive_plotter,
        "sextractor_pp": sextractor_pp,
        "maxi_mask": maxi_mask,
    }

    dict_function = {}
    for d in (
        dict_function_generic,
        dict_function_ait,
        dict_function_soft,
    ):
        dict_function.update(d)
    if (len(sys.argv) == 1) | (len(sys.argv) == 2) & (
        sys.argv[-1] in ["help", "h", "-h", "--help"]
    ):
        create_folders()
        present_plugIn()
        print(
            "{0}{1:30}\033[0;0m{2} {3}".format(
                bcolors.BOLD,
                "LoadDS9QuickLookPlugin",
                bcolors.ENDC,
                dict_function["LoadDS9QuickLookPlugin"].__doc__.split("\n")[0],
            )
        )
        for dict_, color in zip(
            [dict_function_generic, dict_function_ait, dict_function_soft],
            [bcolors.OKBLUE, bcolors.FAIL, bcolors.OKGREEN],
        ):
            dict_ = dict(sorted(dict_.items(), key=lambda x: x[0].lower()))
            for function in dict_:
                if function not in [
                    "interactive_plotter",
                    "guidance",
                    "LoadDS9QuickLookPlugin",
                    "verbose",
                    "next_step",
                    "Quit",
                    "check_file",
                    "button",
                    "create_contour_regions",
                    "mask_regions",
                    "compute_fluctuation",
                    "compute_gain",
                    "emccd_model",
                    "throw_apertures",
                    "PlotSpectraFilters",
                    "add_field_to_header",
                ]:
                    print(
                        "{0}{1}{2:30}\033[0;0m{3} {4}".format(
                            color,
                            bcolors.BOLD,
                            function,
                            bcolors.ENDC,
                            dict_function[function].__doc__.split("\n")[0],
                        )
                    )
    # print("which('DS9Utils') =", which('DS9Utils'))
    # print("__file__ =", __file__)
    # print("__package__ =", __package__)
    # print("Python version = ", sys.version)
    else:
        function = sys.argv[1]
        if sys.stdin is None:
            try:
                dict_function[function]()
            except Exception as e:
                verboseprint(e)
                import traceback

                verboseprint(traceback.format_exc(), verbose="1")
                verboseprint(
                    """Turn verbose mode (shift+V) to have more information
                       about the error""",
                    verbose="1",
                )
                verboseprint("'" + "' '".join(sys.argv) + "'", verbose="1")
        else:
            dict_function[function]()
    return


if __name__ == "__main__":
    a = main()
