import resource
import time
import glob
import os
import sys

import datetime
import argparse
import numpy as np

try:
    import IPython.core.ultratb
except ImportError:
    pass
else:
    sys.excepthook = IPython.core.ultratb.ColorTB()
from functools import wraps
from pyds9plugin.DS9Utils import *
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, vstack, hstack

from scipy.optimize import curve_fit
