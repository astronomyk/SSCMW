import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, Column
from scipy.ndimage import convolve

from matplotlib import pyplot as plt
import d3celestial as d3

L_MIN=0.0
L_MAX=360.0
B_MIN=-10.0
B_MAX=10.0
SPACING=0.5


def galactic_icrs_grid():
    """
    Create a meshgrid in Galactic coordinates (l, b) and convert to ICRS (RA, Dec).

    Returns
    -------
    table : astropy.table.Table
        Table with columns [l_deg, b_deg, ra_deg, dec_deg].
    """
    # 1) Build the Galactic grid
    l = np.arange(L_MIN, L_MAX, SPACING)
    b = np.arange(B_MIN, B_MAX + SPACING / 2, SPACING)
    L, B = np.meshgrid(l, b, indexing='ij')

    # 2) Convert to RA/Dec (ICRS)
    gal = SkyCoord(l=L.ravel()*u.deg, b=B.ravel()*u.deg, frame='galactic')
    icrs = gal.icrs

    RA  = np.round(icrs.ra.hourangle,3)
    Dec = np.round(icrs.dec.deg,2)

    # 3) Create Astropy Table
    tab = Table()
    tab['patch_id'] = np.arange(len(RA)) + 1
    tab['patch_name'] = [f"{r:05.2f}_{d:+05.1f}" for r, d in zip(RA, Dec)]
    tab['l_deg']  = L.ravel()
    tab['b_deg']  = B.ravel()
    tab['ra_deg'] = RA
    tab['dec_deg'] = Dec
    tab.meta['grid_shape'] = (B.shape[0], L.shape[1])  # (n_b, n_l)
    tab.meta['spacing_deg'] = SPACING
    tab.meta['bounds'] = dict(l_min=L_MIN, l_max=L_MAX, b_min=B_MIN,
                              b_max=B_MAX)

    weights, _ = get_weights_column()
    print(len(tab), len(weights))
    tab.add_column(weights)

    return tab


def get_weights_column():
    dsos_tbl = d3.get_dsos(14)
    mask = (np.abs(dsos_tbl["b"]) <= 10)
    dsos_tbl = dsos_tbl[mask]

    n_dl = len(np.arange(L_MIN, L_MAX, SPACING))
    n_db = len(np.arange(B_MIN, B_MAX + SPACING / 2, SPACING))
    canvas = np.zeros((n_db, n_dl))

    xs = ((dsos_tbl["l"] - L_MIN) / SPACING).astype(int)
    ys = ((dsos_tbl["b"] - B_MIN) / SPACING).astype(int)
    for x, y in zip(xs, ys):
        canvas[y, x] += 1
    kernel = np.ones((3, 3))
    kernel[1, 1] += 1
    canvas = convolve(canvas, kernel, mode="constant")
    col = Column(data=canvas.T.ravel(), name="weight")  #.T because of ij meshgrid indexing in galactic_icrs_grid

    return col, canvas


# plt.imshow(get_weights_column(), origin="lower")
# plt.show()

def plot_weights_column(mode="eq"):
    patch_tbl = galactic_icrs_grid()

    if mode == "eq":
        plt.scatter(patch_tbl['ra_deg'], patch_tbl['dec_deg'],
                    c=patch_tbl['weight'], s=4)
        plt.xlabel("RA [Hour-Angle]")
        plt.ylabel("Dec [Deg]")
    elif mode == "gal":
        plt.scatter(patch_tbl['l_deg'], patch_tbl['b_deg'],
                    c=patch_tbl['weight'], s=3 * patch_tbl['weight'])
        plt.xlabel("Gal-Long (l) [deg]")
        plt.ylabel("Gal-Lat (b) [deg]")

    plt.colorbar()
    plt.title("MW Patch 'Fun-To-Observe' Index")
    plt.show()

plot_weights_column("eq")

# Example usage:
# grid_table = galactic_icrs_grid(0, 360, -10, 10, 0.5)
#plt.scatter(grid_table['ra_deg'], grid_table['dec_deg'], s=1)
#plt.show()
# print(grid_table[:25])
# grid_table.write("patch_list.tsv", format='ascii.tab', overwrite=True)
