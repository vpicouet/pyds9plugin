from scipy.sparse import dia_matrix


def VariableSmearingKernels(image, Smearing, SmearExpDecrement):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np

    n = 15
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    kernel_shape = np.arange(n)[:, np.newaxis, np.newaxis]
    smearing_kernels = np.exp(-kernel_shape / smearing_length)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels


def variable_smearing_kernels(
    image, Smearing=0.7, SmearExpDecrement=50000, type_="exp"
):
    """Creates variable smearing kernels for inversion
    """
    import numpy as np

    n = 15
    smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
    if type_ == "exp":
        smearing_kernels = np.exp(
            -np.arange(n)[:, np.newaxis, np.newaxis] / smearing_length
        )
    else:
        assert 0 <= Smearing <= 1
        smearing_kernels = np.power(Smearing, np.arange(n))[
            :, np.newaxis, np.newaxis
        ] / np.ones(smearing_length.shape)
    smearing_kernels /= smearing_kernels.sum(axis=0)
    return smearing_kernels


# -------- Variable declaration --------#
n_registers = 604
flux = 0.1
ConversionGain = 1 / 4.5
EmGain = 1500
size = (2000, 1000)
Bias = 1000
sCIC = 0.1
Smearing = 1.3  # 7
SmearExpDecrement = 50000
read_noise = 50  # e-/pix * ConversionGain
n = 15
# --- Amplification ---#
flux *= np.hstack([np.ones(size), np.zeros(size)])
nsize = (2000, 2000)
imaADU = np.random.gamma(np.random.poisson(flux), abs(EmGain))
# prob_sCIC = np.random.rand(nsize[0], nsize[1])
# sCIC positions
# id_scic = prob_sCIC < sCIC
id_scic = np.random.rand(nsize[0], nsize[1]) < sCIC
register = np.random.randint(1, n_registers, size=id_scic.shape)
imaADU += np.random.gamma(
    np.random.poisson(1 * id_scic), np.power(EmGain, register / n_registers)
)
imaADU *= ConversionGain
if Smearing > 0:
    smearing_kernels = variable_smearing_kernels(imaADU, Smearing, SmearExpDecrement)
    offsets = np.arange(n)
    A = dia_matrix(
        (smearing_kernels.reshape((n, -1)), offsets), shape=(imaADU.size, imaADU.size),
    )
    imaADU = A.dot(imaADU.ravel()).reshape(imaADU.shape)
read_noise = np.random.normal(0, abs(read_noise * ConversionGain), size=id_scic.shape)
imaADU += Bias
imaADU += read_noise

ds9 = imaADU[:, ::-1]


# ds9 = np.random.gamma(np.random.poisson(flux), abs(EmGain))
# # --- Partial amplification of sCIC ---#
# id_scic = np.random.rand(nsize[0], nsize[1]) < sCIC
# register = np.random.randint(1, n_registers, size=id_scic.shape)
# # --- Draw at which stage of the EM register the e- is created ---#
# # partial_gain = np.power(EmGain, register / n_registers)
# # ds9[id_scic] += np.random.exponential(partial_gain)
# ds9 += np.random.gamma(
#     np.random.poisson(1 * id_scic), np.power(EmGain, register / n_registers)
# )

# if Smearing > 0:
#     smearing_kernels = variable_smearing_kernels(ds9, Smearing, SmearExpDecrement)
#     # smearing_kernels = VariableSmearingKernels(ds9, Smearing, SmearExpDecrement)
#     offsets = np.arange(n)
#     A = dia_matrix(
#         (smearing_kernels.reshape((n, -1)), offsets), shape=(ds9.size, ds9.size)
#     )
#     ds9 = A.dot(ds9.ravel()).reshape(ds9.shape)
# # --- Addition of read noise ---#
# ds9 *= ConversionGain
# ds9 += np.random.normal(0, 50 * ConversionGain, size=id_scic.shape)
# ds9 += Bias
# ds9 = ds9[:, ::-1]

