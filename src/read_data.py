# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:14:45 2026

@author: chris
"""

import os
import glob
from astropy.io import fits
from timeit import default_timer

from helper import ScienceFrame, ScienceFrameList, CalibFrameList


def read_folder(path, sci_frame=False):
    '''
    Outputs a list of all hdus in folder given by path.

    Parameters
    ----------
    path : str
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
    print(f"Reading now: {path}")
    starttime = default_timer()
    
    if sci_frame is True:
        hdus = ScienceFrameList()
    else:
        hdus = CalibFrameList()
    files = glob.glob(os.path.join(path, "*.[Ff][Ii][Tt]*"))
    for fname in files:
        with fits.open(fname) as hdul:
            if sci_frame is True:  # Store in ScienceFrame, dont load data
                hdus.append(ScienceFrame(fname, hdul[0].header.copy()))
            else:  # Read data too
                hdus.append(hdul[0].copy())
    
    dt = default_timer() - starttime
    path_end = "\\".join(path.split(sep="\\")[-2:])
    print(f"Finished reading {len(files)} files from {path_end} in {dt:.2f} s")
    return hdus

def read(directory="../data/20251104_lab/", science_data=False):
    '''
    Read header and data of Bias, Dark and Flats folders.
    If science_data=False, Only read hdrs of science data and store in ScienceFrameList
    If True, also read all science data. Unnessecary and dangerous for RAM.

    Parameters
    ----------
    directory : str, optional
        path to directory with data. The default is "../data/20251104_lab/".
    science_data : bool, optional
        wheter to read science data. The default is False.

    Returns
    -------
    sci : ScienceFrameList
        ScienceFrames for each file in Science
    calibration : dict of HDU
        contains bias, dark and flat hdus.

    '''
    directory = os.path.abspath(os.path.expanduser(directory))
    bias = read_folder(os.path.join(directory, "Bias"))
    dark = read_folder(os.path.join(directory, "Dark"))
    flat = read_folder(os.path.join(directory, "Flats"))
    if science_data is True:
        sci = read_folder(os.path.join(directory, "Science"))
    else:
        sci = read_folder(os.path.join(directory, "Science"), sci_frame=True)

    calibration = {"bias": bias, "dark": dark, "flat": flat}
    return sci, calibration




if __name__ == "__main__":
    directory = "../data/20260114_lab/"
    # hdrs_bias, bias = read_folder(directory+"Bias/")
    # hdrs_flats, flats = read_folder(directory+"Flats/")
    # hdrs, datas = read_folder(directory+"Science/")
    sci, hdus = read(directory)
    pass