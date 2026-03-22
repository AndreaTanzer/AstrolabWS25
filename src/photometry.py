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
from timeit import default_timer
import warnings
import os
import helper
import plot
import io_data
from helper import get_repo_root


def compute_flux(im, xypos, fwhm, r_in, r_out):
    aptr = aperture.CircularAperture(xypos, r=r_in*fwhm)
    annulus = aperture.CircularAnnulus(xypos, r_in=r_in*fwhm, r_out=r_out*fwhm)
    aperstats = aperture.ApertureStats(im, annulus)
    bkg_mean = aperstats.mean
    phot_table = aperture.aperture_photometry(im, aptr)
    # rename and add some columns to table
    phot_table.rename_column("id", "idx")
    total_bkg = bkg_mean * aptr.area
    phot_table["flux"] = phot_table["aperture_sum"] - total_bkg
    
    return phot_table

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
    if not hasattr(calc_zero_point, "_call_count"):
        calc_zero_point._call_count = 0

    calc_zero_point._call_count += 1

    zps = mag + 2.5*np.log10(flux)
    zp_mean, _, _ = sigma_clipped_stats(zps, sigma=sigma)
    # my kernel died after ~200 plots :/
    # if calc_zero_point._call_count % 10 == 0:
    #     fname = repo_root / "figs" / f"zp_debug_{calc_zero_point._call_count:03d}.png"
    #     plot.plot(data=[mag, mag + 2.5*np.log10(flux)], marker=["."], linestyle=["None"], 
    #         title="Photometric Zero Point Determination", xlabel="Catalog Magnitude", 
    #         ylabel="Magnitude of flux=1", fname = fname, showPlot=False)
        
    return zp_mean

def calc_magnitude(flux, zp):
    with warnings.catch_warnings():
        # Would warn if flux is negative. Gets casted to nan
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mag = zp - 2.5*np.log10(flux)
    return mag


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

def aperture_photometry(frame, fwhm=7, r_in=4, r_out=5, star_ids: pd.Series=None):
    '''
    Perform aperture photometry on all detected stars in a single frame.
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
    phot_table : TYPE
        DESCRIPTION.

    '''
    # load data
    im = frame.load() 
    stars = frame.load_stars()
    if star_ids is not None:
        mask = np.isin(stars["UCAC4"], star_ids)
    else:
        mask = np.ones_like(stars["UCAC4"], dtype=bool)
    band = frame.get("filter")
    # construct coordinate system and plot
    star_pos = get_star_pos(stars, frame.header)
    if star_pos is None:
        raise AttributeError("frame has no stars extension")
    # plot.plot_stars(im, stars)
    # perform aperture photometry on all stars
    phot_table = compute_flux(im, star_pos["xypos"], fwhm, r_in, r_out)
    phot_table["norm_aperture_sum"] = phot_table["aperture_sum"]/frame.get("EXPOSURE")
    phot_table["norm_flux"] = phot_table["flux"]/frame.get("EXPOSURE")
    
    for col in phot_table.itercols():
        col.format = "{:.4g}"
    time_str = frame.get("DATE-OBS")
    phot_table["t"] = time.Time(time_str, scale="utc", format="isot")
    phot_table["ra"] = star_pos["ra"]
    phot_table["dec"] = star_pos["dec"]

    # calibrate image
    zp = calc_zero_point(stars[f"{band}mag"][mask], phot_table["flux"][mask])
    zp_norm = calc_zero_point(stars[f"{band}mag"][mask], phot_table["norm_flux"][mask])
    # Calculate magnitudes from flux using zero point
    phot_table["mag"] = calc_magnitude(phot_table["flux"], zp)
    phot_table["norm_mag"] = calc_magnitude(phot_table["norm_flux"], zp_norm)
    phot_table["mag"].format = "{:.2f}"
    phot_table["main_id"] = stars["UCAC4"]
    phot_table["zp"] = zp
    phot_table["norm_zp"] = zp_norm
    return phot_table

@helper.functimer
# @helper.profiler(n=50)
def band_photometry(data_band, band, verbose=False, **phot_kwargs):
    phot_tables = []
    n_total = len(data_band)

    for i, frame in enumerate(data_band):
        helper.print_statusline(f"{i+1}/{len(data_band)}")

        try:
            # use previously calculated zp if calibration failed because no detected
            # star has magnitude data in the requested band
            phot_table = aperture_photometry(frame, **phot_kwargs)
            status = "NONE" if phot_table is None else f"OK rows={len(phot_table)}"
            print(f"({i+1}/{n_total}) {status}")

            if phot_table is not None:
                phot_tables.append(phot_table)

        except Exception as e:
            print(f"({i+1}/{n_total}) ERR {type(e).__name__}: {e}")
            continue

    if len(data_band) == 0:
        raise ValueError(f"No frames found for band={band!r}. Available filters: {data.unique('filter')}")

    if len(phot_tables) == 0:
        raise RuntimeError(f"All frames failed for band={band!r}. " "Scroll up for the per-frame error messages." )

    phot_tables = table.vstack(phot_tables, metadata_conflicts="silent")
    phot_tables.sort(["main_id", "t"])
    return phot_tables

def gen_light_curves(data, labname):
    stars = data[0].load_stars()
    UCAC4 = helper.get_dataset(labname, "UCAC4")
    name = helper.get_dataset(labname, "name")
    bands = data.unique("filter")
    calib_modes = ["single", "common", "all"]
    light_curves = {}
    light_curves_ref = {}
    for band in bands:
        light_curves[band] = {}
        light_curves_ref[band] = {}
        data_band = data.filter(filter=band)
        star_ids = helper.get_common_star_ids(data_band, tol=1)
        UCAC4_ref = star_ids[2]
        mask = np.isin(stars["UCAC4"], star_ids[2])
        mag_ref = stars["Vmag"][mask][0]
        for mode in calib_modes:
            match mode:
                case "single":
                    phot_table = band_photometry(data_band, band, r_in=2, r_out=5, 
                                                 star_ids=star_ids[1])
                case "common":
                    phot_table = band_photometry(data_band, band, r_in=2, r_out=5, 
                                                 star_ids=star_ids)
                case "all":
                    phot_table = band_photometry(data_band, band, r_in=2, r_out=5)
            phot_target = extract_target(phot_table, UCAC4)
            phot_ref = extract_target(phot_table, UCAC4_ref)
            plot.phot_norm(phot_target, name, band+f"_{mode}", title=name)
            plot.phot_norm(phot_ref, f"UCAC4 {UCAC4_ref}", band+f"_{mode}", 
                      title=f"UCAC4 {UCAC4_ref}, V={mag_ref:.3f}mag")
            light_curves[band][mode] = phot_target
            light_curves_ref[band][mode] = phot_ref
    return light_curves, light_curves_ref


if __name__ == "__main__":
    starttime = default_timer()
    labs = ["20251104_lab", "20260114_lab"]
    repo_root = get_repo_root()
    light_curves = []
    for lab in labs:
        directory = repo_root / "data" / lab
        data = io_data.read_folder(directory / "Solved")
        light_curve = gen_light_curves(data, lab)
        light_curves.append(light_curve)
    
    
    # UCAC4 = helper.get_dataset(labs[0], "UCAC4")
    # name = helper.get_dataset(labs[0], "name")
    # band = "V"
    # directory = repo_root / "data" / labs[0]
    
    # # TODO: move this part into a function, loop over all filters
    # # read data
    # if (directory / "Solved").exists():
    #     print("Solved content (first 20):")
    #     print(sorted(os.listdir(directory / "Solved"))[:20])
    # print("=======================\n")
    # data = io_data.read_folder(directory / "Solved")
    # data_band = data.filter(filter=band)
    # # fwhm = calc_fwhm(data_band)
    # star_ids = helper.get_common_star_ids(data_band)
    # phot_table = band_photometry(data_band, band, r_in=2, r_out=5, star_ids=star_ids[1])
    # phot_target = extract_target(phot_table, UCAC4)
    
    # stars = data[0].load_stars()
    # UCAC4_ref = stars["UCAC4"][1]
    # mag_ref = stars["Vmag"][1]
    # phot_ref = extract_target(phot_table, UCAC4_ref)
    
    # # plot_phot(phot_target, name, band+"_single", title="Target Star")
    # # plot_phot(phot_ref, f"UCAC4 {UCAC4_ref}", band+"_single", 
    # #           title=f"UCAC4 {UCAC4_ref}, V={mag_ref:.3f}mag")
    # plot.phot_norm(phot_target, name, band+"_single", title="RR Lyrae")
    # plot.phot_norm(phot_ref, f"UCAC4 {UCAC4_ref}", band+"_single", 
    #           title=f"UCAC4 {UCAC4_ref}, V={mag_ref:.3f}mag")

    print('Execution Time: %.2f s' % (default_timer()-starttime))
    
    

    
    
    
