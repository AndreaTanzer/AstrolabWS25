# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:14:45 2026

@author: chris
"""

import os
import glob
import numpy as np
from astropy.io import fits
from helper import get_repo_root

# from helper import ScienceFrame, ScienceFrameList, CalibFrame, CalibFrameList
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

def write_solved_frame(frame, new_wcs, path):
    header = frame.header.copy()
    header.update(new_wcs.to_header())
    fits.writeto(path, frame.load(), header, overwrite=True)



if __name__ == "__main__":
    directory = get_repo_root() / "data/20260114_lab/"
    sci, hdus = read(directory)
    pass