# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 18:37:36 2026

@author: chris
"""

import numpy as np
import pandas as pd
from astropy import wcs, table, time
import astropy.units as u
from astropy.coordinates import SkyCoord 
from astropy.stats import sigma_clipped_stats
from astroquery.simbad import Simbad
from photutils import detection, psf, aperture
# from photutils.aperture import CircularAperture
from matplotlib.pyplot import close
from timeit import default_timer
import warnings

import helper
import plot
import io_data
from helper import get_repo_root

def calc_zero_point(mag, flux):
    """
    Calculates magnitude where flux drops to 1

    Parameters
    ----------
    mag : TYPE
        DESCRIPTION.
    flux : TYPE
        DESCRIPTION.

    Returns
    -------
    zp_mean : TYPE
        DESCRIPTION.

    """
    zps = mag + 2.5*np.log10(flux)
    zp_mean, zp_median, zp_std = sigma_clipped_stats(zps, sigma=2)
    return zp_mean

def calc_magnitude(flux, zp):
    with warnings.catch_warnings():
        # Would warn if flux is negative. Gets casted to nan
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mag = zp - 2.5*np.log10(flux)
    return mag
    
def query_Simbad(vals_with_idx, band, radius_arcsec=5):
    upload_table = table.Table(vals_with_idx, names=["idx", "ra", "dec"])
    radius_deg = radius_arcsec/60**2  # cone search radius
    # cone search around position with given radius
    # obtain main_id, ra, dec, object_type, magnitude in given filter
    query = f"""
    SELECT
        up.idx, up.ra AS input_ra, up.dec AS input_dec, -- input idx, ra, dec
        b.main_id, b.ra, b.dec, b.otype, -- obtain main_id, ra, dec, object_type
        f.filter, f.flux,  -- obtain magnitude for each filter
        DISTANCE(POINT('ICRS', b.ra, b.dec), -- get distance from input ra, dec
                 POINT('ICRS', up.ra, up.dec)) AS dist
    FROM TAP_UPLOAD.targets AS up
    -- choose objects within radius
    JOIN basic AS b ON 1 = CONTAINS(POINT('ICRS', b.ra, b.dec), 
                                    CIRCLE('ICRS', up.ra, up.dec, {radius_deg}))
    JOIN flux AS f ON b.oid = f.oidref
    WHERE f.filter = '{band}'  -- choose objects with given band
    """
    result = Simbad.query_tap(query, targets=upload_table)
    if result is None:
        return None
    df = result.to_pandas()
    df["dist_arcsec"] = df["dist"]*60**2
    df.rename(columns={"flux": "mag"}, inplace=True)
    # Choose brightest object within search radius (usually there is only 1)
    df_brightest = df.sort_values("mag").groupby("idx").first().reset_index()
    df_brightest['idx'] = df['idx'].astype(int)
    return df_brightest

def query_Simbad_id(vals_with_idx, band="G", radius_arcsec=10):
    simbad_data = query_Simbad(vals_with_idx, band, radius_arcsec)
    df = simbad_data[["idx", "main_id"]]
    return df

def aperture_photometry(frame, plotting=False):
    # load data
    im = frame.load() 
    stars = frame.load_stars()  
    if stars is None:
        return None
    
    # construct coordinate system and plot
    w = wcs.WCS(frame.header)
    xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    star_coord_vals = w.all_pix2world(xypos, 0)
    if plotting is True:
        title = frame.path + ", plate solved"
        plot.imshow_coords(im, w, stars, title=title)
    
    # perform aperture photometry on all stars
    # fit_shape: diameter of area to look for star
    fwhms = psf.fit_fwhm(im, xypos=xypos, fit_shape=39)
    fwhm = int(np.median(fwhms))
    
    aptr = aperture.CircularAperture(xypos, r=fwhm)
    annulus = aperture.CircularAnnulus(xypos, r_in=2*fwhm, r_out=3*fwhm)
    aperstats = aperture.ApertureStats(im, annulus)
    bkg_mean = aperstats.mean
    phot_table = aperture.aperture_photometry(im, aptr)
    # rename and add some columns to table
    phot_table.rename_column("id", "idx")
    total_bkg = bkg_mean*aptr.area
    phot_table["flux"] = phot_table["aperture_sum"] - total_bkg
    for col in phot_table.itercols():
        col.format = "{:.4g}"
    time_str = frame.get("DATE-OBS")
    phot_table["t"] = time.Time(time_str, scale="utc", format="isot")
    phot_table["ra"] = star_coord_vals[:, 0]
    phot_table["dec"] = star_coord_vals[:, 1]
    # star_coords = SkyCoord(ra=star_coord_vals[:, 0]*u.deg,
    #                       dec=star_coord_vals[:, 1]*u.deg, frame="icrs")
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
    
    # calibrate image
    # 0 because python starts counting at 0
    vals_with_idx = phot_table["idx", "ra", "dec"].to_pandas().values
    simbad_data = query_Simbad(vals_with_idx, frame.get("filter"))
    flux_series = pd.Series(phot_table["flux"])
    flux_series = flux_series.loc[flux_series>0]
    simbad_data["flux"] = simbad_data["idx"].map(flux_series)
    # restrict to sources that have positive flux after background subtraction
    simbad_existent = simbad_data.loc[simbad_data["flux"]>0]
    # Calibrate image using sources
    zp = calc_zero_point(simbad_existent["mag"], simbad_existent["flux"])
    # Calculate magnitudes from flux using zero point
    phot_table["mag"] = calc_magnitude(phot_table["flux"], zp)
    # -------------------------------------------------------------------------
    # Get main_id of stars
    ids = query_Simbad_id(vals_with_idx)
    phot_df = phot_table.to_pandas()
    # drop sources without Gaia G magnitude
    merged_df = phot_df.merge(ids, on="idx", how="right")
    # In case there are multiple detections for the same star, only keep the
    # detection with the brightest flux
    # probably mainly used in overexposed images
    merged_df.sort_values(by="mag").drop_duplicates(subset="main_id", keep="first", 
                                                    inplace=True)
    phot_table_merged = table.Table.from_pandas(merged_df)
    return phot_table_merged

def get_common_stars(table):
    # get stars visible across (almost) all frames
    grouped_stars = table.group_by("idx")
    group_lengths = np.diff(grouped_stars.groups.indices)
    common_ids = grouped_stars.groups.keys[group_lengths == group_lengths.max()]
    small_table = table[np.isin(table["idx"],  common_ids["idx"])]
    return small_table

def extract_value(table, value, index="t", column="idx"):
    df = table.to_pandas()
    matrix = df.pivot(index=index, columns=column, values=value)
    return matrix

if __name__ == "__main__":
    starttime = default_timer()
    close("all")
    repo_root = get_repo_root()
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
    phot_tables = table.vstack(phot_tables, metadata_conflicts="silent")
    phot_tables.sort(["main_id", "t"])
    phot_tables["t", "main_id", "mag"].pprint_all()
    small_table = get_common_stars(phot_table)
    flux = extract_value(small_table, "aperture_sum")

    print('Execution Time: %.2f s' % (default_timer()-starttime))
    
    

    
    
    