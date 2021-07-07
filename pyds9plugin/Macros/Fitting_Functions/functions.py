# Function used to fit some DS9 plots
# Please be sure to define a list as default parameters for each arguments
# as they will be used to define the lower and upper bounds of each widget.

import numpy as np


def gaussian(x, a=[0, 100], xo=[0, 100], sigma=[0, 10]):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = a * np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
    return g.ravel()


def defocused_gaussian(
    x,
    amp=[0, 1000],
    amp_2=[0, 1000],
    xo=[0, 100],
    sigma=[0, 10],
    sigma_2=[0, 10],
):
    """Defines a gaussian function with offset
    """
    import numpy as np

    xo = float(xo)
    g = amp * np.exp(-0.5 * (np.square(x - xo) / sigma ** 2))
    g2 = amp_2 * np.exp(-0.5 * (np.square(x - xo) / sigma_2 ** 2))
    return (g - g2).ravel()


def schechter(x, phi=[1e-3, 1e-2], m=[16, 22], alpha=[-2, -1]):
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


def double_schechter(
    x,
    phi=[1e-3, 1e-2],
    alpha=[-2, -1],
    M=[16, 22],
    phi2=[1e-3, 1e-2],
    alpha2=[-2, -1],
):
    return np.log10(
        10 ** schechter(x, phi, M, alpha)
        + +(10 ** schechter(x, phi2, M, alpha2))
    )


def sincos(x, a=[0, 10], w1=[0, 2], w2=[0, 2], theta=[0, np.pi]):
    return a * np.sin(w1 * x) + np.cos(w2 * x + theta)
