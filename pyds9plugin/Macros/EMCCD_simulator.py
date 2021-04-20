from scipy.sparse import dia_matrix

def VariableSmearingKernels(image, Smearing, SmearExpDecrement):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    kernel_shape = np.arange(6)[:, np.newaxis, np.newaxis]
    smearing_kernels = np.exp(-kernel_shape / smearing_length)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels

#-------- Variable declaration --------#
n_registers = 604
flux = 0.1
ConversionGain = 1.5
EmGain = 1500
size = (1000, 1000)
Bias = 3000
read_noise = 80
sCIC = 0.1
Smearing = 0.7
SmearExpDecrement = 50000

#--- Amplification ---#
ds9 = np.random.gamma(flux * ConversionGain, EmGain, size=size)
prob_sCIC = np.random.rand(size[0], size[1])
# sCIC positions
id_scic = prob_sCIC < sCIC
#--- Partial amplification of sCIC ---#
register = np.random.randint(1, n_registers, size=id_scic.sum())
#--- Draw at which stage of the EM register the e- is created ---#
partial_gain = np.power(EmGain, register / n_registers)
ds9[id_scic] += np.random.exponential(partial_gain)
if Smearing > 0:
    smearing_kernels = VariableSmearingKernels(ds9, Smearing,
                                               SmearExpDecrement)
    offsets = np.arange(6)
    A = dia_matrix((smearing_kernels.reshape((6, -1)), offsets),
                   shape=(ds9.size, ds9.size))
    ds9 = A.dot(ds9.ravel()).reshape(ds9.shape)
#--- Addition of read noise ---#
ds9 += np.random.normal(Bias, read_noise * ConversionGain,
                        size=ds9.shape)
