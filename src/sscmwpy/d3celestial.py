from pathlib import Path
import json

import numpy as np
from astropy import table
import matplotlib.pyplot as plt

from coords_converter import convert_eq_gal

DATA_DIR = Path(__file__).parent.parent.parent / "data"/ "d3_celestial"


def get_stars(lim_mag=6):
    """
    List stars in the d3celestial catalogue down to a given magnitude

    Parameters
    ----------
    lim_mag : float
        Faintest magnitude in catalogue is V=14

    Returns
    -------
    tbl: astropy.table.Table
        Table with stars. Columns: ('id', 'mag', 'bv', 'ra', 'dec', 'l', 'b')

    """
    if lim_mag <= 6:
        fname = "stars.6.json"
    elif lim_mag <= 8:
        fname = "stars.8.json"
    elif lim_mag <= 14:
        fname = "stars.14.json"
    else:
        fname = "stars.14.json"
        print("Warning: limiting magnitude is too high. Returning stars with V<14")

    with open(DATA_DIR / fname) as f:
        json_dict = json.load(f)

    tbl_list = [[f["id"],
                 f["properties"]["mag"],
                 f["properties"]["bv"],
                 f["geometry"]["coordinates"][0],
                 f["geometry"]["coordinates"][1]
                ] for f in json_dict["features"]]
    tbl_array = np.array(tbl_list)
    tbl_array[tbl_array == ""] = "0."
    tbl = table.Table(tbl_array,
                      names=('id', 'mag', 'bv', 'ra', 'dec'),
                      dtype=(int, float, float, float, float))
    ls, bs = convert_eq_gal(ra_list=tbl["ra"] / 15. % 24, dec_list=tbl["dec"])
    tbl.add_columns([table.Column(data=ls, name="l"),
                     table.Column(data=bs, name="b")])

    mask = tbl["mag"] < lim_mag
    tbl = tbl[mask]

    names = get_starnames(tbl["id"])
    tbl.add_column(table.Column(data=names, name="name"))

    return tbl


def get_starnames(hipparcos_ids):
    with open(DATA_DIR / "starnames.json", "r", encoding="utf-8") as f:
        json_dict = json.load(f)

    names = [json_dict.get(str(hip), {"name": "---"})["name"]
             for hip in hipparcos_ids]
    return names


def get_milky_way_contours(surf_bri_lvl=0):
    """
    Return the contours of the Milky Way for a set surface brightness level

    By default, it returns the contours of the lowest surface brightness,
    essientially the visible plane of the Milky Way on the sky.

    Parameters
    ----------
    surf_bri_lvl : int
        [0..4] The level of surface brightness to return

    Returns
    -------
    ras_decs: list of np.arrays
        A list of all the contours for this surface brightness level

    """
    with open(DATA_DIR / "milkyway.json") as f:
        json_dict = json.load(f)

    feature = json_dict["features"][surf_bri_lvl]
    ras_decs = [table.Table(data=np.array(coords),
                            names=('ra', 'dec'),
                            dtype=(float, float))
                for coords in feature["geometry"]["coordinates"]]
    for tbl in ras_decs:
        ls, bs = convert_eq_gal(ra_list=tbl["ra"] / 15. % 24, dec_list=tbl["dec"])
        tbl.add_columns([table.Column(data=ls, name="l"),
                         table.Column(data=bs, name="b")])

    return ras_decs


def get_messier_objects(lim_mag=20):
    """
    Lists the position and properties of all Messier objects down to a given magnitude

    Parameters
    ----------
    lim_mag : float

    Returns
    -------
    tbl: astropy.table.Table
        Table of properties. Columns: ('name', 'type', 'mag', 'dim', 'ra', 'dec', 'l', 'b')

    """
    fname = "messier.json"
    with open(DATA_DIR / fname) as f:
        json_dict = json.load(f)

    tbl_list = [[f["properties"]["name"],
                 f["properties"]["type"],
                 f["properties"]["mag"],
                 f["properties"]["dim"],
                 f["geometry"]["coordinates"][0],
                 f["geometry"]["coordinates"][1]
                ] for f in json_dict["features"]]
    tbl_array = np.array(tbl_list)
    tbl_array[tbl_array == ""] = "0."
    tbl = table.Table(tbl_array,
                      names=('name', 'type', 'mag', 'dim', 'ra', 'dec'),
                      dtype=(str, str, float, str, float, float))
    ls, bs = convert_eq_gal(ra_list=tbl["ra"] / 15. % 24, dec_list=tbl["dec"])
    tbl.add_columns([table.Column(data=ls, name="l"),
                     table.Column(data=bs, name="b")])
    mask = tbl["mag"] < lim_mag

    return tbl[mask]


def get_dsos(lim_mag=6):
    """
    Lists the position and properties of all deep sky objects down to a given magnitude

    Parameters
    ----------
    lim_mag : float
        Faintest magnitude in catalogue is V=20

    Returns
    -------
    tbl: astropy.table.Table
        Table of properties. Columns: ('name', 'type', 'mag', 'dim', 'ra', 'dec', 'l', 'b')

    """
    if lim_mag <= 6:
        fname = "dsos.6.json"
    elif lim_mag <= 14:
        fname = "dsos.14.json"
    elif lim_mag <= 20:
        fname = "dsos.20.json"
    else:
        fname = "dsos.20.json"
        print("Warning: limiting magnitude is too high. Returning DSOs with V<20")

    with open(DATA_DIR / fname) as f:
        json_dict = json.load(f)

    tbl_list = [[f["properties"]["desig"],
                 f["properties"]["type"],
                 f["properties"]["mag"],
                 f["properties"]["dim"],
                 f["geometry"]["coordinates"][0],
                 f["geometry"]["coordinates"][1]
                 ] for f in json_dict["features"]]
    tbl_array = np.array(tbl_list)
    tbl_array[tbl_array == ""] = "0."
    tbl = table.Table(tbl_array,
                      names=('name', 'type', 'mag', 'dim', 'ra', 'dec'),
                      dtype=(str, str, float, str, float, float))
    tbl = tbl[np.abs(tbl["dec"]) <= 90]
    ls, bs = convert_eq_gal(ra_list=tbl["ra"] / 15. % 24, dec_list=tbl["dec"])
    tbl.add_columns([table.Column(data=ls, name="l"),
                     table.Column(data=bs, name="b")])
    areas = [int(np.prod([float(x) for x in y.split("x")]))
             if "x" in y else int(float(y)**2) for y in tbl["dim"]]
    tbl.add_column(table.Column(data=areas, name="area"))
    mask = tbl["mag"] < lim_mag

    return tbl[mask]


# print(get_stars(1))
# print(get_milky_way_contours())
# print(get_messier_objects())
# print(get_dsos())

# tbl = get_dsos(14)
# tbl = tbl[np.abs(tbl["b"]) < 10]
# plt.scatter(tbl["l"], tbl["b"], 1, c="k")
# plt.show()

def plot_mw_stars(mode="eq"):
    tbl_mw = get_milky_way_contours(0)
    tbl_stars = get_stars(3)
    tbl_messier = get_messier_objects()

    if mode == "eq":
        plt.plot(tbl_stars["ra"] / 15 % 24, tbl_stars["dec"], "o")

        for i in range(len(tbl_stars)):
            plt.annotate(" " + tbl_stars["name"][i], (tbl_stars["ra"][i] / 15 % 24, tbl_stars["dec"][i]))

        for contours in tbl_mw:
            plt.scatter(contours["ra"] / 15 % 24, contours["dec"], 1, c="k")

        plt.plot(tbl_messier["ra"] / 15 % 24, tbl_messier["dec"], "x")
        for i in range(len(tbl_messier)):
            plt.annotate(tbl_messier["name"][i], (tbl_messier["ra"][i] / 15 % 24, tbl_messier["dec"][i]))

        plt.ylabel("Dec [deg]")
        plt.xlabel("RA [dec hour-angle]")
        plt.xlim(0, 24)

    elif mode == "gal":
        plt.plot(tbl_stars["l"], tbl_stars["b"], "o")
        for i in range(len(tbl_stars)):
            plt.annotate(tbl_stars["name"][i],
                         (tbl_stars["l"][i], tbl_stars["b"][i]))
        for contours in tbl_mw:
            plt.scatter(contours["l"], contours["b"], 1, c="k")

        plt.plot(tbl_messier["l"], tbl_messier["b"], "x")
        for i in range(len(tbl_messier)):
            plt.annotate(tbl_messier["name"][i], (tbl_messier["l"][i], tbl_messier["b"][i]))

        plt.ylabel("Gal-Lat (b) [deg]")
        plt.xlabel("Gal-Long (l) [deg]")
        plt.xlim(0, 360)

    plt.grid(True)
    plt.title("D3 Celestial Catalogue")
    plt.ylim(-20, 20)

    plt.show()


# plot_mw_stars("gal")
