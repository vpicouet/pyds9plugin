from pyds9plugin.Macros.FIREBall.FB_functions import emccd_model

if ("FIREBall.py" in __file__) or (function == "execute_command"):
    from astropy.table import Table

    d = DS9n()
    fitsfile = d.get_pyfits()
    filename = get_filename(d)
    table = create_table_from_header(
        filename, exts=[0], info=""
    )  # Table(data=[[1]],names=['test'])
    filename = get_filename(d)
else:
    pass


emccd_model(xpapoint=None, path=None, smearing=1, argv=[])

# from pyds9fb.DS9FB import calc_emccdParameters
# from pyds9plugin.DS9Utils import variable_smearing_kernels  # , EMCCD


# def variable_smearing_kernels(image, Smearing=1.5, SmearExpDecrement=50000):
#     """Creates variable smearing kernels for inversion
#     """
#     import numpy as np

#     smearing_length = Smearing * np.exp(-image / SmearExpDecrement)
#     smearing_kernels = np.exp(
#         -np.arange(6)[:, np.newaxis, np.newaxis] / smearing_length
#     )
#     smearing_kernels /= smearing_kernels.sum(axis=0)
#     return smearing_kernels


# if __name__ == "__main__":
