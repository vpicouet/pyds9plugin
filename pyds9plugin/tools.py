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
 


def AddFieldAftermatching(FinalCat=None, ColumnCat=None, path1=None, path2=None, radec1=["RA", "DEC"], radec2=["RA", "DEC"], distance=0.5, field="Z_ML", new_field=None, query=None):
    """
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    if path1 is not None:
        try:
            FinalCat = Table.read(path1)
        except:
            FinalCat = Table.read(path1, format="ascii")

    if path2 is not None:
        try:
            ColumnCat = Table.read(path2)
        except:
            ColumnCat = Table.read(path2, format="ascii")
    verboseprint("cat 1 : %i lines" % (len(FinalCat)))
    verboseprint("cat 2 : %i lines" % (len(ColumnCat)))
    # print(ColumnCat['ZFLAG'])
    print(ColumnCat)
    if query is not None:
        ColumnCat = apply_query(cat=ColumnCat, query=query, path=None, new_path=None, delete=True)
        print(ColumnCat)
        mask = np.isfinite(ColumnCat[radec2[0]]) & np.isfinite(ColumnCat[radec2[1]])
        ColumnCat = ColumnCat[mask]
    # print(ColumnCat['ZFLAG'])
    if len(radec1) == 2:
        try:
            c = SkyCoord(ra=ColumnCat[radec2[0]] * u.deg, dec=ColumnCat[radec2[1]] * u.deg)
        except Exception as e:
            print(e)
            c = SkyCoord(ra=ColumnCat[radec2[0]], dec=ColumnCat[radec2[1]])
        try:
            catalog = SkyCoord(ra=FinalCat[radec1[0]] * u.deg, dec=FinalCat[radec1[1]] * u.deg)
        except Exception:
            catalog = SkyCoord(ra=FinalCat[radec1[0]], dec=FinalCat[radec1[1]])
        #        idx, d2d, d3d = catalog.match_to_catalog_sky(c[mask])
        print(catalog)
        print(c)
        idx, d2d, d3d = catalog.match_to_catalog_sky(c)
        mask = 3600 * np.array(d2d) < distance
        print("Number of matches < %0.2f arcsec :  %i " % (distance, mask.sum()))

    elif len(radec1) == 1:
        import pandas as pd
        from pyds9plugin.DS9Utils import DeleteMultiDimCol

        ColumnCat = ColumnCat[radec2 + field]
        if new_field is not None:
            ColumnCat.rename_columns(field, new_field)
        ColumnCat.rename_column(radec2[0], "id_test")
        FinalCat = DeleteMultiDimCol(FinalCat)
        ColumnCat = DeleteMultiDimCol(ColumnCat)
        FinalCatp = FinalCat.to_pandas()
        ColumnCatp = ColumnCat.to_pandas()
        a = pd.merge(FinalCatp, ColumnCatp, left_on=radec1[0], right_on="id_test", how="left").drop("id_test", axis=1)
        return Table.from_pandas(a)  # .to_table()

    if new_field is None:
        new_field = field
    idx_ = idx[mask]
    for fieldi, new_field in zip(field, new_field):
        print("Adding field " + fieldi + " " + new_field)
        if new_field not in FinalCat.colnames:
            if type(ColumnCat[fieldi][0])==np.ndarray:
                FinalCat[new_field] = np.ones((len(FinalCat),len(ColumnCat[fieldi][0])))*-99.00
            else:
                FinalCat[new_field] = -99.00
        print(FinalCat[new_field])
        FinalCat[new_field][mask] = ColumnCat[fieldi][idx_]
        # verboseprint(FinalCat[new_field])
    return FinalCat

