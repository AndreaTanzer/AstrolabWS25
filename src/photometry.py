# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 18:37:36 2026

@author: chris
"""

import numpy as np
from astropy import wcs, table, time
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from photutils import detection, psf, aperture
# from photutils.aperture import CircularAperture
from matplotlib.pyplot import close
from timeit import default_timer

import helper
import plot
import io_data

def query_Simbad(star_coord_vals, radius_arcsec=10):
    upload_table = table.Table(star_coord_vals, names=["ra", "dec"])
    radius_deg = radius_arcsec/60**2  # cone search radius
    query = f"""
    SELECT
        up.ra AS input_ra, up.dec AS input_dec,
        b.main_id, b.ra, b.dec, b.otype,
        f.filter, f.flux,
        DISTANCE(POINT('ICRS', b.ra, b.dec),
                 POINT('ICRS', up.ra, up.dec)) AS dist
    FROM TAP_UPLOAD.targets AS up
    JOIN basic AS b ON 1 = CONTAINS(POINT('ICRS', b.ra, b.dec), 
                                    CIRCLE('ICRS', up.ra, up.dec, {radius_deg}))
    LEFT JOIN flux AS f ON b.oid = f.oidref
    """
    result = Simbad.query_tap(query, targets=upload_table)
    df = result.to_pandas()
    print()
    return

def aperture_photometry(frame):
    # load data
    im = frame.load() 
    stars = frame.load_stars()  
    if stars is None:
        return None
    # construct coordinate system and plot
    w = wcs.WCS(frame.header)
    xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    # 0 because python starts counting at 0
    star_coord_vals = w.all_pix2world(xypos, 0)
    star_coords = SkyCoord(ra=star_coord_vals[:, 0]*u.deg,
                          dec=star_coord_vals[:, 1]*u.deg, frame="icrs")
    title = frame.path + ", plate solved"
    plot.imshow_coords(im, w, stars, title=title)
    
    # -------------------------------------------------------------------------
    # TODO: query simbad for coordinates of stars, extract magnitudes 
    # (maybe check if star is variable), use stars (median?) to calibrate flux
    
    # from another code where upload_table contains the search_name
    # here we would construct an astropy table containing the coordinates of 
    # our detected stars. 
    # # 2. Query SIMBAD
    # query = """
    # SELECT 
    #     up.search_name, b.main_id, b.ra, b.dec, b.otype,
    #     f.filter, f.flux, b.galdim_majaxis, b.galdim_minaxis,
    #     b.rvz_radvel, b.plx_value
    # FROM TAP_UPLOAD.targets AS up
    # JOIN ident AS i ON i.id = up.search_name
    # JOIN basic AS b ON i.oidref = b.oid
    # LEFT JOIN flux AS f ON b.oid = f.oidref
    # """
    # result = Simbad.query_tap(query, targets=upload_table)
    
    # to see which columns are available
    # Simbad.list_columns("basic").pprint_all()
    # Simbad.list_columns("flux").pprint_all()
    
    simbad_data = query_Simbad(star_coord_vals)

    # -------------------------------------------------------------------------
    # aperture photometry performed here
    # fit_shape: diameter of area to look for
    xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    fwhms = psf.fit_fwhm(im, xypos=xypos, fit_shape=39)
    fwhm = int(np.median(fwhms))
    
    aptr = aperture.CircularAperture(xypos, r=fwhm)
    annulus = aperture.CircularAnnulus(xypos, r_in=2*fwhm, r_out=3*fwhm)
    aperstats = aperture.ApertureStats(im, annulus)
    bkg_mean = aperstats.mean
    phot_table = aperture.aperture_photometry(im, aptr)
    total_bkg = bkg_mean*aptr.area
    phot_table["aperture_sum"] -= total_bkg
    for col in phot_table.itercols():
        col.format = "{:.4g}"
    phot_table["id"] = stars["gaia_id"]
    phot_table["ra"] = stars["ra"]
    phot_table["dec"] = stars["dec"]
    phot_table["ra"].format = "{:.3f}"
    phot_table["dec"].format = "{:.3f}"
    time_str = frame.get("DATE-OBS")
    phot_table["t"] = time.Time(time_str, scale="utc", format="isot")
    return phot_table

def get_common_stars(table):
    # get stars visible across (almost) all frames
    grouped_stars = table.group_by("id")
    group_lengths = np.diff(grouped_stars.groups.indices)
    common_ids = grouped_stars.groups.keys[group_lengths == group_lengths.max()]
    small_table = table[np.isin(table["id"],  common_ids["id"])]
    return small_table

def extract_value(table, value, index="t", column="id"):
    df = table.to_pandas()
    matrix = df.pivot(index=index, columns=column, values=value)
    return matrix

if __name__ == "__main__":
    starttime = default_timer()
    close("all")
    repo_root = io_data.get_repo_root()
    labs = {"RR_Lyrae": "20251104_lab", "Transit": "20260114_lab"}
    directory = repo_root / "data" / labs["RR_Lyrae"]
    
    # read data
    data = io_data.read_folder(directory / "Solved")
    phot_tables = []
    magsRR = []
    data_r = data.filter(filter="r")
    for i, frame in enumerate(data_r):
        print(f"{i+1}/{len(data_r)}")
        try:
            phot_table = aperture_photometry(frame)
            if phot_table is not None:
                phot_tables.append(phot_table)
                magsRR.append(phot_table["aperture_sum"][0])
        except Exception as e:
            print(f"{i}: {e}")
            continue
    phot_table = table.vstack(phot_tables, metadata_conflicts="silent")
    small_table = get_common_stars(phot_table)
    flux = extract_value(small_table, "aperture_sum")

    print('Execution Time: %.2f s' % (default_timer()-starttime))
    
    

    
    
    