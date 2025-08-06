import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from pyongc.ongc import listObjects


def convert_eq_gal(ra_list=None, dec_list=None, l_list=None, b_list=None):
    """
    Convert RA and Dec lists to Galactic coordinates (l, b), accepting either
    float values in degrees or sexagesimal strings (e.g. '10h12m30s', '41d16m30s').

    Parameters:
    -----------
    ra_list : list of str or float
        List of Right Ascension values (sexagesimal string or float degrees).
    dec_list : list of str or float
        List of Declination values (sexagesimal string or float degrees).

    Returns:
    --------
    list of tuples
        List of (l, b) Galactic coordinates in degrees.

    """
    if ra_list is not None and dec_list is not None:
        if len(ra_list) != len(dec_list):
            raise ValueError("RA and Dec lists must be of the same length.")

        # Use SkyCoord to parse mixed input types
        coords = SkyCoord(ra=ra_list, dec=dec_list, unit=(u.hourangle, u.deg),
                          frame='icrs')

        galactic = coords.galactic
        return np.round(galactic.l.deg, 4), np.round(galactic.b.deg, 4)

    elif l_list is not None and b_list is not None:
        if len(l_list) != len(b_list):
            raise ValueError("RA and Dec lists must be of the same length.")

        # Use SkyCoord to parse mixed input types
        coords = SkyCoord(l=l_list, b=b_list, unit=(u.deg, u.deg),
                          frame='galactic')

        equatorial = coords.icrs
        return np.round(equatorial.ra.hourangle, 4), np.round(equatorial.dec.deg, 4)

    else:
        raise ValueError("Either RA and Dec or l and b must be provided.")


def ongc_gal_coords(catalog="M", filename=None):
    """

    Parameters
    ----------
    catalog : str or None
        [NGC|IC|M]
    filename : str

    Returns
    -------
    tbl: astropy.table.Table

    """
    if catalog.lower() in ["ngc", "ic", "m"]:
        objects = listObjects(catalog=catalog.upper())
    else:
        objects = listObjects()

    objs = [[o.name, o.coords.flatten(), str(o.identifiers).replace("None, ", "").replace("(", "").replace(")", "")]
            for o in objects if o.coords is not None and o.coords.size == 6]
    names = [o[0] for o in objs]
    coords = [o[1] for o in objs]
    other_names = [o[2] for o in objs]
    ras  = [ f"{c[0]:02.0f}h{c[1]:02.0f}m{c[2]:05.2f}s" for c in coords]
    decs = [f"{c[3]:03.0f}d{c[4]:02.0f}m{c[5]:04.2f}s" for c in coords]

    ls, bs = convert_eq_gal(ra_list=ras, dec_list=decs)
    ras2, decs2 = convert_eq_gal(l_list=ls, b_list=bs)

    mask = abs(bs) < 10

    tbl = Table(names=('name', 'ra', 'dec', 'l', 'b', 'other_names'),
                 data=[names, ras2, decs2, ls, bs, other_names]) # [mask]
    if filename is not None:
        tbl.write(filename, format="ascii.tab", overwrite=True)

    return tbl


ongc_gal_coords("", "test.txt")

