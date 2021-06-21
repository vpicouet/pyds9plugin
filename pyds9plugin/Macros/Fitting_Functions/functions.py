# Function used to fit some DS9 plots
# Please be sure to define default parameters for each functino arguments
# as they will be used for the plot. The lower and upper bounds of each widget
# will be defined as (0 -> 10 x default)
# You should then keep positive default parameters and put the minus sign in
# the function definition if needed.

import numpy as np


def gaussian(x, amplitude=10, xo=10, sigma2=2, offset=10):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = offset + amplitude * np.exp(-0.5 * (np.square(x - xo) / sigma2))
    return g.ravel()


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


def double_schechter(x, phi, alpha, M, phi2, alpha2):
    return np.log10(
        10 ** schechter(x, phi, M, alpha)
        + +(10 ** schechter(x, phi2, M, alpha2))
    )


def sincos(x, a=1, b=2, theta=np.pi):
    return a * np.sin(x) + b * np.cos(x + theta)
