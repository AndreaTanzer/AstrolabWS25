# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 18:37:36 2026

@author: chris
"""

import numpy as np
import pandas as pd
from astropy import wcs, table, time
from astropy.stats import sigma_clipped_stats
from astroquery.simbad import Simbad
from photutils import psf, aperture
from matplotlib.pyplot import close
from timeit import default_timer
import warnings

import helper
import plot
import io_data
from helper import get_repo_root

def calc_zero_point(mag, flux, zp_default=np.nan):
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
    zp_mean, _, _ = sigma_clipped_stats(zps, sigma=2)
    if np.isnan(zp_mean):
        # none of the detected stars have brightness data in the requested band
        zp_mean = zp_default 
        default_flag = True
    else:
        default_flag = False
    return zp_mean, default_flag

def calc_magnitude(flux, zp):
    with warnings.catch_warnings():
        # Would warn if flux is negative. Gets casted to nan
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mag = zp - 2.5*np.log10(flux)
    return mag

# Not that many stars with magnitude data across bands, calibration fails at times
def query_Simbad(upload_table, band, radius_arcsec=5):
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

def query_Simbad_id(upload_table, band="G", radius_arcsec=10):
    simbad_data = query_Simbad(upload_table, band, radius_arcsec)
    df = simbad_data[["idx", "main_id"]]
    return df

def get_star_pos(stars, header):
    if stars is None:
        return None
    xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    w = wcs.WCS(header)
    # 0 because python starts counting at 0
    star_coords = w.all_pix2world(xypos, 0)
    ra, dec = star_coords[:, 0], star_coords[:, 1]
    pos = dict(stars=stars, w=w, xypos=xypos, ra=ra, dec=dec)
    return pos

def create_upload_table(pos):
    idx = np.arange(pos["ra"].size) + 1
    coords = np.vstack([idx, pos["ra"], pos["dec"]]).T
    upload_table = table.Table(coords, names=["idx", "ra", "dec"])
    return upload_table

def aperture_photometry(frame, zp_default, fwhm=10, r_in=2, r_out=3, 
                        plotting=False, calc_fwhm=False):
    '''
    Perform aperture photometry on all detected stars in frame.
    Calibrate 

    Parameters
    ----------
    frame : TYPE
        DESCRIPTION.
    zp_default : TYPE
        DESCRIPTION.
    fwhm : int, optional
        Photometric radius is 2*fwhm. 10 seems to be pretty constant across frames.
        Probably best to keep it constant. The default is 10.
    plotting : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    phot_table_merged : TYPE
        DESCRIPTION.
    zp : TYPE
        DESCRIPTION.

    '''
    # load data
    im = frame.load() 
    stars = frame.load_stars()
    band = frame.get("filter")
    # construct coordinate system and plot
    star_pos = get_star_pos(stars, frame.header)
    if star_pos is None:
        raise AttributeError("frame has no stars extension")
    if plotting is True:
        title = frame.path + ", plate solved"
        plot.imshow_coords(im, star_pos["w"], star_pos["stars"], title=title)
    
    if calc_fwhm is True:
        # fit_shape: diameter of area to look for star
        fwhms = psf.fit_fwhm(im, xypos=star_pos["xypos"], fit_shape=39)
        fwhm = int(np.median(fwhms))
        print(f"{fwhm=:.1f}")
    
    # perform aperture photometry on all stars
    aptr = aperture.CircularAperture(star_pos["xypos"], r=fwhm)
    annulus = aperture.CircularAnnulus(star_pos["xypos"], r_in=r_in*fwhm, r_out=r_out*fwhm)
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
    phot_table["ra"] = star_pos["ra"]
    phot_table["dec"] = star_pos["dec"]

    # calibrate image
    upload_table = create_upload_table(star_pos)
    # simbad_data = query_Simbad(upload_table, frame.get("filter"))
    # flux_series = pd.Series(phot_table["flux"])
    # flux_series = flux_series.loc[flux_series>0]
    # simbad_data["flux"] = simbad_data["idx"].map(flux_series)
    # # restrict to sources that have positive flux after background subtraction
    # simbad_existent = simbad_data.loc[simbad_data["flux"]>0]
    # # Calibrate image using sources
    # zp, default_flag = calc_zero_point(simbad_existent["mag"], simbad_existent["flux"], zp_default)
    zp, default_flag = calc_zero_point(stars[f"{band}mag"], phot_table["flux"])
    phot_table["default_zp"] = default_flag
    # Calculate magnitudes from flux using zero point
    phot_table["mag"] = calc_magnitude(phot_table["flux"], zp)
    phot_table["mag"].format = "{:.2f}"
    # -------------------------------------------------------------------------
    # Get main_id of stars
    ids = query_Simbad_id(upload_table)
    phot_df = phot_table.to_pandas()
    # drop sources without Gaia G magnitude
    merged_df = phot_df.merge(ids, on="idx", how="right")
    # In case there are multiple detections for the same star, only keep the
    # detection with the brightest flux
    # probably mainly used in overexposed images
    merged_df.sort_values(by="mag").drop_duplicates(subset="main_id", keep="first", 
                                                    inplace=True)
    phot_table_merged = table.Table.from_pandas(merged_df)
    return phot_table_merged, zp

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

@helper.functimer
def band_photometry(data, band, main_id="V* RR Lyr", verbose=False, **phot_kwargs):
    phot_tables = []
    data_band = data.filter(filter=band)
    zp_prev = np.nan
    for i, frame in enumerate(data_band):
        helper.print_statusline(f"{i+1}/{len(data_band)}")
        try:
            # use previously calculated zp if calibration failed because no detected
            # star has magnitude data in the requested band
            phot_table, zp = aperture_photometry(frame, zp_prev, **phot_kwargs)
            if phot_table is not None:
                phot_tables.append(phot_table)
            if not np.isnan(zp):
                zp_prev = zp
        except Exception as e:
            print(f"{i}: {e}\n")
            continue
    phot_tables = table.vstack(phot_tables, metadata_conflicts="silent")
    phot_tables.sort(["main_id", "t"])
    phot_target = phot_tables[phot_tables["main_id"]==main_id]
    if verbose:
        phot_target["t", "main_id", "mag"].pprint_all()
        plot.plot([phot_target["t"].datetime, phot_target["mag"]], title=band, ylabel="mag", marker=".")
    return phot_target

if __name__ == "__main__":
    # TODO: Move this part into helper.get_dataset
    starttime = default_timer()
    close("all")
    repo_root = get_repo_root()
    labs = {"V* RR Lyr": "20251104_lab", "HAT-P-32": "20260114_lab"}
    name = "HAT-P-32"
    band = "V"
    directory = repo_root / "data" / labs[name]
    
    # TODO: move this part into a function, loop over all filters
    # read data
    data = io_data.read_folder(directory / "Solved")

    phot_table = band_photometry(data, band, main_id=name, r_in=1, r_out=2)
    df = phot_table.to_pandas()
    # create 5min rolling average of magnitude, save it to column mag_5m
    df = df.merge(df.set_index("t").rolling("5min")["mag"].mean().rename("mag_5m"), 
                  left_on=["t"], right_index=True)
    # convert name to valid filename (cant use *)
    fname = helper.slugify(f"{name}_{band}_photometry")
    plot.plot([df["t"], df["mag"], df["mag_5m"]], ylabel="mag", linestyle=["None", "-"], 
              marker=[".", "None"], legend=["single measurements", "5 min running average"],
              fname=repo_root / "figs" / fname)

    print('Execution Time: %.2f s' % (default_timer()-starttime))
    
    

    
    
    