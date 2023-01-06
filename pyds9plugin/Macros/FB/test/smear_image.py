from scipy import special
import numpy as np
from scipy.sparse import dia_matrix

def variable_smearing_kernels(
    image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np

    n = 14
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    smearing_kernels = np.exp(
        -np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length
    )
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels

ds9 = ds9.astype('float64')
Smearing = 5.1
smearing_kernels = variable_smearing_kernels(
    ds9, Smearing, SmearExpDecrement=50000
)
n = smearing_kernels.shape[0]

A = dia_matrix(
    (smearing_kernels.reshape((n, -1)), np.arange(n)),
    shape=(ds9.size, ds9.size),
)
ds9 = A.dot(ds9.ravel()).reshape(ds9.shape)
