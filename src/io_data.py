# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:14:45 2026

@author: chris
"""

import os
import glob
from pathlib import Path
import warnings
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

import helper

def read_folder(directory, sci_frame=True):
    '''
    Outputs a list of all hdus in folder given by path.

    Parameters
    ----------
    directory : str
        Path to folder with data relative to file.
    sci_frame : bool, optional
        If true, only read the hdr files, not the data. Stored in ScienceFrame.
        This will be used to load and calibrate one science frame later on.
        The default is False.

    Returns
    -------
    hdus : list
        list of hdu or list of ScienceFrame (if sci_frame=True).
    '''
    directory = os.path.abspath(os.path.expanduser(directory))
    if sci_frame is True:
        frames = helper.ScienceFrameList()
    else:
        frames = helper.CalibFrameList()
    files = glob.glob(os.path.join(directory, "*.[Ff][Ii][Tt]*"))
    for fname in files:
        with fits.open(fname) as hdul:
            header = hdul[0].header.copy()
        if sci_frame:
            frames.append(helper.ScienceFrame(fname, header))
        else:
            frames.append(helper.CalibFrame(fname, header))
    return frames

def read(directory):
    '''
    Read header of Science, Bias, Dark and Flats folders.
    Store them in ScienceFrameList or CalibFrameList

    Parameters
    ----------
    directory : str, optional
        path to directory with data. The default is "../data/20251104_lab/".

    Returns
    -------
    sci : ScienceFrameList
        ScienceFrames for each file in Science
    calibration : dict of HDU
        contains bias, dark and flat hdus.

    '''
    bias_dir = directory / "Bias"
    dark_dir = directory / "Dark"
    flat_dir = directory / "Flats"
    science_dir = directory / "Science"

    print("Bias directory:", bias_dir)
    print("Dark directory:", dark_dir)
    print("Flat directory:", flat_dir)
    print("Science directory:", science_dir)

    bias = read_folder(bias_dir, sci_frame=False)
    dark = read_folder(dark_dir, sci_frame=False)
    flat = read_folder(flat_dir, sci_frame=False)
    sci = read_folder(science_dir)
  
    calibration = {"bias": bias, "dark": dark, "flat": flat}
    nx = sci.unique("NAXIS1")
    ny = sci.unique("NAXIS2")

    if len(nx) != 1 or len(ny) != 1:
        raise ValueError(f"Inconsistent science frame dimensions: nx={nx}, ny={ny}")

    nx_sci = nx[0]
    ny_sci = ny[0]
    for key in calibration:
        for frame in calibration[key]:  # iterate over frames
            nx = frame.get("NAXIS1")
            ny = frame.get("NAXIS2")
            # Would need to interpolate calibration frame, not implemented
            if nx < nx_sci or ny < ny_sci:
                raise ValueError(f"Calibration frame smaller than science frame: nx={nx}, ny={ny}, expected >= ({nx_sci}, {ny_sci})")
            if nx > nx_sci or ny > ny_sci:  # need to bin calibration frame
                x_ratio = nx/nx_sci
                y_ratio = ny/ny_sci
                if x_ratio != y_ratio or x_ratio != int(x_ratio):
                    raise ValueError(f"Incompatible binning ratio: x_ratio={x_ratio}, y_ratio={y_ratio}, sci=({nx_sci}, {ny_sci}), calib=({nx}, {ny})")
                frame.bin_size = int(x_ratio)

    return sci, calibration

def write_reduced_frame(reduced, header, path, new_object_name):
    '''
    Output reduced data and header as HDUL to given path
    Copy and modify header to add info about processing

    Parameters
    ----------
    reduced : np.darray
        Reduced Science Frame.
    header : astropy.io.fits.header.Header
        header of original science frame.
    path : str
        where to save hdul.
    new_object_name : str
        object field is set to this name if not None. 

    Returns
    -------
    None.

    '''
    hdr = header.copy()
    if new_object_name is not None:
        hdr["OBJECT"] = new_object_name
    hdr["IMAGETYP"] = ("Reduced Light Frame", "Pipeline-processed")
    hdr["HISTORY"] = "Bias subtracted"
    hdr["HISTORY"] = "Dark current subtracted"
    hdr["HISTORY"] = "Flat-field corrected"
    hdu = fits.PrimaryHDU(data=reduced.astype(np.float32), header=hdr, do_not_scale_image_data=True)
    hdu.writeto(path, overwrite=True)
    return

def write_solved_frame(outpath: str|Path, stars, wcs_solution):
    '''
    Updates header with coordinates of frame and matched stars

    Parameters
    ----------
    outpath : str | Path
        path to save file
    stars : astropy.table.Table
        Contains gaia_id, coordinates and position in image of stars
    wcs_solution : wcs.wcs.WCS
        Contains coordinates, orientation and transformation of field.

    Returns
    -------
    None.

    '''
    path = Path(outpath)
    output_dir = path.parent.parent / "Solved"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / path.name
    with fits.open(outpath, memmap=True) as hdul:
        # update history
        hdr = hdul[0].header
        hist = hdr.get("HISTORY", [])
        if isinstance(hist, str):
            hist = [hist]
        
        filtered = [h for h in hist 
                    if "PLATE SOLVED" not in h.upper()
                    or "FAILED PLATE SOLVING" not in h.upper()]
        del hdr["HISTORY"]
        for line in filtered:
            hdr.add_history(line)
        hdr.add_history("Plate Solved")
        hdr.update(wcs_solution.to_header()) 
        
        # clearing meta prevents
        # WARNING: Attribute `date` of type <class 'str'> cannot be added to FITS Header - skipping
        # WARNING: Attribute `version` of type <class 'dict'> cannot be added to FITS Header - skipping
        stars.meta = {}
        # create STARS extension
        stars_hdu = fits.BinTableHDU(stars, name="STARS")
        new_hdul = fits.HDUList([fits.PrimaryHDU(data=hdul[0].data, header=hdr), stars_hdu])
        new_hdul.writeto(output_path, overwrite=True)
    return

def write_header_failed(outpath):
    ''' 
    Update history with "Failed Plate Solving"
    '''
    with fits.open(outpath, mode="update") as hdul:
        hdr = hdul[0].header
        hist = hdr.get("HISTORY", [])
        if isinstance(hist, str):
            hist = [hist]
        # Avoid duplicates
        if not any("FAILED PLATE SOLVING" in h.upper() for h in hist):
            hdr.add_history("Failed Plate Solving")
    hdul.flush()
    return 


if __name__ == "__main__":
    directory = helper.get_repo_root() / "data/20260114_lab/"
    sci, hdus = read(directory)
    pass