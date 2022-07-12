from pyds9plugin.Macros.FIREBall.FB_functions import emccd_model

if "FIREBall.py" in __file__:  # or (function == "execute_command")
    from astropy.table import Table

    d = DS9n()
    fitsfile = d.get_pyfits()
    filename = get_filename(d)
    table = create_table_from_header(filename, exts=[0], info="")
    filename = get_filename(d)
else:
    pass

emccd_model(xpapoint=None, path=filename, smearing=0, argv=[])
