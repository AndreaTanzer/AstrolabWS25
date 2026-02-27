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

def calc_zero_point(mag: table.Table, flux: table.Table, sigma: float=1.):
    """
    Calculates magnitude where flux drops to 1

    Parameters
    ----------
    mag : table.Table
        actual magnitude of stars.
    flux : table.Table
        measured flux of stars.
    sigma: float

    Returns
    -------
    zp_mean : float
        magnitude where flux is drops to 1.

    """
    zps = mag + 2.5*np.log10(flux)
    zp_mean, _, _ = sigma_clipped_stats(zps, sigma=sigma)
    # plot.plot(data=[mag, mag+2.5*np.log10(flux)], marker=["."], linestyle=["None"])
    return zp_mean

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

@helper.functimer
def calc_fwhm(data):
    fwhms = []
    for frame in data:
        im = frame.load() 
        stars = frame.load_stars()
        star_pos = get_star_pos(stars, frame.header)
        fwhm_stars = psf.fit_fwhm(im, xypos=star_pos["xypos"], fit_shape=39)
        fwhms.append(np.median(fwhm_stars))
    return np.array(fwhms)

def aperture_photometry(frame, fwhm=7, r_in=4, r_out=5, 
                        plotting=False):
    '''
    Perform aperture photometry on all detected stars in frame.
    Calibrate using 

    Parameters
    ----------
    frame : TYPE
        DESCRIPTION.
    fwhm : int, optional
        Photometric radius is 2*fwhm. 7 seems to be pretty constant across frames.
        Probably best to keep it constant. The default is 7.

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
    # plot.plot_stars(im, stars)
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
    zp = calc_zero_point(stars[f"{band}mag"], phot_table["flux"])
    # Calculate magnitudes from flux using zero point
    phot_table["mag"] = calc_magnitude(phot_table["flux"], zp)
    phot_table["mag"].format = "{:.2f}"
    phot_table["main_id"] = stars["UCAC4"]
    phot_table["zp"] = zp
    # -------------------------------------------------------------------------
    # Get main_id of stars
    # upload_table = create_upload_table(star_pos)
    # ids = query_Simbad_id(upload_table)
    # phot_df = phot_table.to_pandas()
    # # drop sources without Gaia G magnitude
    # merged_df = phot_df.merge(ids, on="idx", how="right")
    # # In case there are multiple detections for the same star, only keep the
    # # detection with the brightest flux
    # # probably mainly used in overexposed images
    # merged_df.sort_values(by="mag").drop_duplicates(subset="main_id", keep="first", 
    #                                                 inplace=True)
    # phot_table_merged = table.Table.from_pandas(merged_df)
    return phot_table

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

def extract_target(phot_table, main_id, plotting=False):
    phot_target = phot_table[phot_table["main_id"]==main_id]
    if plotting is True:
        plot.plot(data=[phot_target["t"].datetime, phot_target["mag"]],
                  title=band, ylabel="mag", marker=".")
    return phot_target

@helper.functimer
# @helper.profiler(n=50)
def band_photometry(data, band, verbose=False, **phot_kwargs):
    phot_tables = []
    data_band = data.filter(filter=band)
    for i, frame in enumerate(data_band):
        helper.print_statusline(f"{i+1}/{len(data_band)}")
        try:
            # use previously calculated zp if calibration failed because no detected
            # star has magnitude data in the requested band
            phot_table = aperture_photometry(frame, **phot_kwargs)
            if phot_table is not None:
                phot_tables.append(phot_table)
        except Exception as e:
            print(f"{i}: {e}\n")
            continue
    phot_tables = table.vstack(phot_tables, metadata_conflicts="silent")
    phot_tables.sort(["main_id", "t"])
    return phot_tables

def plot_phot(phot, name, band, t_roll="30min", title=None):
    df = phot.to_pandas()
    # create 5min rolling average of magnitude, save it to column mag_5m
    df = df.merge(df.set_index("t").rolling(t_roll)["mag"].mean().rename("mag_rolling"), 
                  left_on=["t"], right_index=True)
    # convert name to valid filename (cant use *)
    fname = helper.slugify(f"{name}_{band}_photometry")
    # plot_kwargs = dict(data=[df["t"], df["mag"], df["mag_rolling"], df["zp"]/3], 
    #                   ylabel="mag", linestyle=["None", "-", "-"], marker=[".", "None", "None"], 
    #                   legend=["single measurements", f"{t_roll} running average", "zp-16mag"])
    plot_kwargs = [dict(data=[df["t"], df["aperture_sum"], df["flux"]], title=title,
                        ylabel="flux", legend=["raw flux", "background subtracted"]),
                   dict(data=[df["t"], df["mag"], df["mag_rolling"]],
                        ylabel="measurements/mag", linestyle=["None", "-"], marker=[".", "None"],
                        legend=["single measurements", f"{t_roll} running average"]),
                   dict(data=[df["t"], df["zp"]], ylabel="zeropoints/mag")
                   ]
    plot.subplots(1, 3, [plot.plot_on_ax,]*3, plot_kwargs, title=None, 
                  add_colorbar=False, figsize=(8, 4.5), fname=repo_root / "figs" / fname)

if __name__ == "__main__":
    starttime = default_timer()
    # close("all")
    repo_root = get_repo_root()
    labs = ["20251104_lab", "20260114_lab"]
    UCAC4 = helper.get_dataset(labs[0], "UCAC4")
    name = helper.get_dataset(labs[0], "name")
    band = "V"
    directory = repo_root / "data" / labs[0]
    
    # TODO: move this part into a function, loop over all filters
    # read data
    data = io_data.read_folder(directory / "Solved")
    # fwhm = calc_fwhm(data.filter(filter=band))
    phot_table = band_photometry(data, band, r_in=2, r_out=3)
    phot_target = extract_target(phot_table, UCAC4)
    
    stars = data[0].load_stars()
    UCAC4_ref = stars["UCAC4"][4]
    mag_ref = stars["Vmag"][4]
    phot_ref = extract_target(phot_table, UCAC4_ref)
    
    plot_phot(phot_target, name, band, title="Target Star")
    plot_phot(phot_ref, f"UCAC4 {UCAC4_ref}", band, 
              title=f"UCAC4 {UCAC4_ref}, V={mag_ref:.3f}mag")
    # df = phot_target.to_pandas()
    # # create 5min rolling average of magnitude, save it to column mag_5m
    # t_roll = "15min"
    # df = df.merge(df.set_index("t").rolling(t_roll)["mag"].mean().rename("mag_rolling"), 
    #               left_on=["t"], right_index=True)
    # # convert name to valid filename (cant use *)
    # fname = helper.slugify(f"{name}_{band}_photometry")
    # # plot_kwargs = dict(data=[df["t"], df["mag"], df["mag_rolling"], df["zp"]/3], 
    # #                   ylabel="mag", linestyle=["None", "-", "-"], marker=[".", "None", "None"], 
    # #                   legend=["single measurements", f"{t_roll} running average", "zp-16mag"])
    # plot_kwargs = [dict(data=[df["t"], df["mag"], df["mag_rolling"]], 
    #                     ylabel="measurements/mag", linestyle=["None", "-"], marker=[".", "None"],
    #                     legend=["single measurements", f"{t_roll} running average"]),
    #                dict(data=[df["t"], df["zp"]], ylabel="zeropoints/mag")
    #                ]
    # plot.subplots(1, 2, [plot.plot_on_ax,]*2, plot_kwargs, title=None, 
    #              add_colorbar=False, figsize=(8, 4.5), fname=repo_root / "figs" / fname)


    print('Execution Time: %.2f s' % (default_timer()-starttime))
    
    

    
    
    