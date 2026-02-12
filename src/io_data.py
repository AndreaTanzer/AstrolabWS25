# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:14:45 2026

@author: chris
"""

import os
import glob
import numpy as np
from astropy.io import fits
from timeit import default_timer

# from helper import ScienceFrame, ScienceFrameList, CalibFrame, CalibFrameList
import helper


def read_folder(path, sci_frame=True):
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
    starttime = default_timer()
    
    if sci_frame is True:
        frames = helper.ScienceFrameList()
    else:
        frames = helper.CalibFrameList()
    files = glob.glob(os.path.join(path, "*.[Ff][Ii][Tt]*"))
    for fname in files:
        with fits.open(fname) as hdul:
            header = hdul[0].header.copy()
        if sci_frame:
            frames.append(helper.ScienceFrame(fname, header))
        else:
            frames.append(helper.CalibFrame(fname, header))
    
    dt = default_timer() - starttime
    path_end = "\\".join(path.split(sep="\\")[-2:])
    print(f"Finished reading {len(files)} files from {path_end} in {dt:.2f} s")
    return frames

def read(directory="../data/20251104_lab/"):
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
    directory = os.path.abspath(os.path.expanduser(directory))
    bias = read_folder(os.path.join(directory, "Bias"), sci_frame=False)
    dark = read_folder(os.path.join(directory, "Dark"), sci_frame=False)
    flat = read_folder(os.path.join(directory, "Flats"), sci_frame=False)
    sci = read_folder(os.path.join(directory, "Science"))
    
    calibration = {"bias": bias, "dark": dark, "flat": flat}
    nx = sci.unique("NAXIS1")
    ny = sci.unique("NAXIS2")
    assert len(nx) == 1 and len(ny) == 1
    nx_sci = nx[0]
    ny_sci = ny[0]
    for key in calibration:
        for frame in calibration[key]:  # iterate over frames
            nx = frame.get("NAXIS1")
            ny = frame.get("NAXIS2")
            # Would need to interpolate calibration frame, not implemented
            assert nx >= nx_sci and ny >= ny_sci 
            if nx > nx_sci or ny > ny_sci:  # need to bin calibration frame
                x_ratio = nx/nx_sci
                y_ratio = ny/ny_sci
                assert x_ratio == y_ratio and x_ratio == int(x_ratio)
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
    hdu = fits.PrimaryHDU(data=reduced.astype(np.float32), header=hdr,
                          do_not_scale_image_data=True)
    hdu.writeto(path, overwrite=True)
    return


if __name__ == "__main__":
    directory = "../data/20260114_lab/"
    # hdrs_bias, bias = read_folder(directory+"Bias/", sci_frame=False)
    # hdrs_flats, flats = read_folder(directory+"Flats/", sci_frame=False)
    # hdrs, datas = read_folder(directory+"Science/")
    sci, hdus = read(directory)
    pass