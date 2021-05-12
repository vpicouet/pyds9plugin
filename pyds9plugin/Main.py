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
import numpy as np
import datetime
from pkg_resources import resource_filename
from astropy.table import Table
from pyds9 import DS9, ds9_targets

try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()
from functools import wraps

DS9_BackUp_path = os.environ["HOME"] + "/DS9QuickLookPlugIn/"


def readV(path):
    """Read a table and try ascii or CSV if an error is raised"""
    if os.path.isfile(path):
        try:
            Table.read(path)
        except Exception:
            try:
                Table.read(path, format="ascii")
            except Exception:
                Table.read(path, format="csv")
    else:
        raise ValueError(path + " is not a file.")


Table.readV = staticmethod(readV)


def DevFunction(xpapoint):
    print("Important function")
    return


def verbose(xpapoint=None, verbose=None):
    """Change the configuration
    """
    if verbose is None:
        v = sys.argv[-1]
    else:
        v = verbose
    try:
        conf_dir = resource_filename("pyds9plugin", "config")
    except:
        conf_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config")
    if v is None:
        sys.exit()
    return conf_dir


def DS9n(xpapoint=None, stop=False):
    """Open a DS9 communication with DS9 software, if no session opens a new one
    else link to the last created session. Possibility to give the ssession
    you want to link"""
    targets = ds9_targets()
    if targets:
        xpapoints = [target.split(" ")[-1] for target in targets]
    else:
        xpapoints = []
    if (xpapoint is None) & (len(xpapoints) == 0):
        verboseprint("No DS9 target found")
        return
    elif len(xpapoints) != 0:
        verboseprint("%i targets found" % (len(xpapoints)))
        if xpapoint in xpapoints:
            verboseprint("xpapoint %s in targets" % (xpapoint))
        else:
            if stop:
                sys.exit()
            else:
                verboseprint("xpapoint %s NOT in targets" % (xpapoint))
                xpapoint = xpapoints[0]

    try:
        verboseprint("DS9(%s)" % (xpapoint))
        d = DS9(xpapoint)
    except (FileNotFoundError, ValueError) as e:
        verboseprint(e)
        d = DS9()
    return d


def CreateFolders(DS9_BackUp_path=os.environ["HOME"] + "/DS9QuickLookPlugIn/"):
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
    if not os.path.exists(DS9_BackUp_path + ".verbose.txt"):
        os.system("echo 1 > %s" % (DS9_BackUp_path + ".verbose.txt"))
    if not os.path.exists(DS9_BackUp_path + ".message.txt"):
        os.system("echo 1 > %s" % (DS9_BackUp_path + ".message.txt"))
    return DS9_BackUp_path


if len(sys.argv) == 1:
    CreateFolders()

message_ = bool(int(os.popen("cat %s.message.txt" % (DS9_BackUp_path)).read()))
verbose_ = bool(int(os.popen("cat %s.verbose.txt" % (DS9_BackUp_path)).read()))
if sys.stdin is not None:
    verbose(xpapoint=None, verbose=1)
else:
    verbose(xpapoint=None, verbose=0)

def Log(v=None):
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
    file_handler = RotatingFileHandler(DS9_BackUp_path + "pyds9plugin_activity.log", "a", 1000000, 1)
    logger.addHandler(file_handler)
    logging.getLogger("matplotlib.font_manager").disabled = True
    return logger


logger = Log()


def yesno(d, question="", verbose=message_):
    """Opens a native DS9 yes/no dialog box."""
    if verbose:
        verboseprint(question)
        return bool(int(d.get("""analysis message yesno {%s}""" % (question))))
    else:
        return True


def message(d, question="", verbose=message_):  #
    """Opens a native DS9 message dialog box with a message."""
    if verbose:
        return bool(int(d.set("analysis message {%s}" % (question))))
    else:
        return True


def verboseprint(*args, logger=logger, verbose=verbose_):  # False
    """Prints a message only if verbose is set to True (mostly if stdout is defined)"""
    st = " ".join([str(arg) for arg in args])
    logger.critical(st)
    if bool(int(verbose)):
        from tqdm import tqdm

        print(*args)
        with tqdm(total=1, bar_format="{postfix[0]} {postfix[1][value]:>s}", postfix=["", dict(value="")], file=sys.stdout) as t:
            for i in range(0):
                t.update()
    else:
        pass
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


def License(limit_date=20210425, l0="whoareyou"):
    """Ask for license to DS9 plugin user"""
    today = int(datetime.date.today().strftime("%Y%m%d"))
    path = DS9_BackUp_path + ".npy"
    if os.path.exists(path) is False:
        np.save(path, "00")
    license_ = np.load(path)
    if ((today > limit_date) & (str(license_) != l0)) or (today > limit_date + 10000):
        d = DS9n(sys.argv[1])
        l = get(d, "This is a trial version. We hope you enjoyed pyds9plugin. Please buy a license at https://people.lam.fr/picouet.vincent/pyds9plugin/ or enter your license key here:", exit_=False)
        if l == l0:
            np.save(path, l)
            d.set("analysis message {Thanks for activating pyds9 plugin! Enjoy your time using it!}")
        else:
            d.set("analysis message {Your license key is not valid. Verify it or contact the support if you did buy a license.}")
            sys.exit()
        return
    else:
        return


if len(sys.argv) > 2:
    License()


# @fn_timer
def main():
    """Main function where the arguments are defined and the other functions called
    """
    if len(sys.argv) == 1:
        CreateFolders(DS9_BackUp_path=os.environ["HOME"] + "/DS9QuickLookPlugIn/")
        PresentPlugIn()
        # LoadDS9QuickLookPlugin()
    elif (len(sys.argv) == 2) & ((sys.argv[-1] == "help") | (sys.argv[-1] == "h")):
        CreateFolders(DS9_BackUp_path=os.environ["HOME"] + "/DS9QuickLookPlugIn/")
        PresentPlugIn()
        # print("which('DS9Utils') =", which('DS9Utils'))
        print("__file__ =", __file__)
        print("__package__ =", __package__)
        print("Python version = ", sys.version)
        print("DS9 analysis file = ", resource_filename("pyds9plugin", "QuickLookPlugIn.ds9.ans"))
        print("Python main file = ", resource_filename("pyds9plugin", "DS9Utils.py"))
        LoadDS9QuickLookPlugin()
        sys.exit()

    else:

        DictFunction_Generic = {
            "setup": DS9setup2,
            "guidance": DS9Update,
            "fitsgaussian2D": fitsgaussian2D,
            "DS9createSubset": DS9createSubset,
            "DS9Catalog2Region": DS9Catalog2Region,
            "AddHeaderField": AddHeaderField,
            "BackgroundFit1D": BackgroundFit1D,
            "test": DS9tsuite,
            "Convolve2d": Convolve2d,
            "PlotSpectraDataCube": PlotSpectraDataCube,
            "StackDataDubeSpectrally": StackDataDubeSpectrally,
            "stack": DS9stack_new,
            "lock": DS9lock,
            "CreateHeaderCatalog": DS9CreateHeaderCatalog,
            "SubstractImage": DS9PythonInterp,
            "DS9Region2Catalog": DS9Region2Catalog,
            "DS9MaskRegions": DS9MaskRegions,
            "CreateImageFromCatalogObject": CreateImageFromCatalogObject,
            "PlotArea3D": PlotArea3D,
            "OriginalSettings": DS9originalSettings,
            "next_step": next_step,
            "BackgroundEstimationPhot": DS9BackgroundEstimationPhot,
            "verbose": verbose,
            "CreateWCS": BuildingWCS,
            "open": DS9open,
            "checkFile": checkFile,
            "ManualFitting": ManualFitting,
            "Quit": Quit,
            "Button": Button,
            "ThrowApertures": ThrowApertures,
            "CreateContourRegions": CreateContourRegions,
        }  # ,'NextButton':NextButton

        DictFunction_AIT = {
            "centering": DS9center,
            "radial_profile": DS9rp,
            "throughfocus": DS9throughfocus,
            "ComputeFluctuation": ComputeFluctuation,
            "throughfocus_visualisation": DS9visualisation_throughfocus,
            "throughslit": DS9throughslit,
            "ExploreThroughfocus": ExploreThroughfocus,
            "ReplaceWithNans": ReplaceWithNans,
            "InterpolateNaNs": DS9InterpolateNaNs,
            "Trimming": DS9Trimming,
            "ColumnLineCorrelation": DS9CLcorrelation,
            "ComputeEmGain": DS9ComputeEmGain,
            "DS9_2D_FFT": DS9_2D_FFT,
            "2D_autocorrelation": DS9_2D_autocorrelation,
            "get_depth_image": get_depth_image,
            "DS9PlotEMCCD": DS9PlotEMCCD,
            "AnalyzeFWHMThroughField": AnalyzeFWHMThroughField,
        }

        DictFunction_SOFT = {
            "DS9SWARP": DS9SWARP,
            "DS9PSFEX": DS9PSFEX,
            "RunSextractor": RunSextractor,
            "DS9saveColor": DS9saveColor,
            "AperturePhotometry": AperturePhotometry,
            "ExtractSources": DS9ExtractSources,
            "CosmologyCalculator": CosmologyCalculator,
            "Convertissor": Convertissor,
            "WCS": DS9guider,
            "DS9Resample": DS9Resample,
            "Function": Function,
            "PlotSpectraFilters": PlotSpectraFilters,
            "RunSextractorPP": RunSextractorPP,
            "MaxiMask": MaxiMask,
        }  #'Function_parametric':Function_parametric

        DictFunction = {}
        for d in (DictFunction_Generic, DictFunction_AIT, DictFunction_SOFT):  # , DictFunction_Calc, DictFunction_SOFT, DictFunction_FB, DictFunction_Delete): #DictFunction_CLAUDS
            DictFunction.update(d)
        xpapoint = sys.argv[1]
        function = sys.argv[2]
        from .DS9Utils import DS9rp#DictFunction[function]
        if function not in ["verbose", "next_step"]:  # ,'setup'
            verboseprint(
                "\n****************************************************\nDS9Utils " + " ".join(sys.argv[1:]) + "\n****************************************************"
            )  # %s %s '%(xpapoint, function) + ' '.join())
            verboseprint(sys.argv)
        if sys.stdin is None:
            try:
                DictFunction[function](xpapoint=xpapoint)
            except Exception as e:  # Exception #ValueError #SyntaxError
                verboseprint(e)
                import traceback

                pprint(traceback.format_exc())
                pprint("To have more information about the error run this in the terminal:")
                pprint(" ".join(sys.argv))
        else:
            DictFunction[function](xpapoint=xpapoint)

        if function not in ["verbose", "setup", "next_step"]:
            verboseprint("\n****************************************************")
    return


if __name__ == "__main__":
    a = main()
