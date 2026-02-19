# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 22:15:03 2026

@author: chris
"""

import os
import numpy as np

import helper
from io_data import read, write_reduced_frame
import plot

def create_master_frames(calib):
    '''
    Creates master bias, dark and flat fields according to moodle.
    Master dark is normalized to exposure time.
    Assumes that all dark fields have same exposure time and that
    flat fields are independent of filter used.

    Parameters
    ----------
    calib : dict
        dictionary with keys ['bias', 'dark', 'flat'].
        values are of type CalibFrameList containing the PrimaryHDU objects

    Returns
    -------
    mbias : numpy.ndarray
        master bias.
    mdark_rate : numpy.ndarray
        master dark normalized to exposure time in seconds.
    mflat : numpy.ndarray
        master flat.

    '''
    print("Creating Master Frame")
    # Step 1, create master bias
    mbias = np.median(calib["bias"].stack_data(), axis=0)
    
    # Step 2, substract master bias from darks and flats
    dark = calib["dark"].stack_data() - mbias
    flat = calib["flat"].stack_data() - mbias
    
    # Step 3, create master dark
    exposure_dark = calib["dark"].exposures[0]  # always the same exposure time
    assert np.allclose(calib["dark"].exposures, exposure_dark)
    mdark = np.median(dark, axis=0)
    # so we dont have to keep track of exposure dark, slightly changes Step 4
    mdark_rate = mdark/exposure_dark
    
    # Step 4, correct flats for dark current, weighted by exposure time
    # ratio broadcasted to (N, ny, nx)
    flat -= calib["flat"].exposures[:, None, None]*mdark_rate
    
    # Step 5, median combine flats
    mflat = np.median(flat, axis=0)
    
    # Step 6, normalize to obtain master flat
    mflat /= np.median(mflat)
    mflat = np.where(mflat<0.2, 1, mflat)
    return mbias, mdark_rate, mflat

def reduce(sci, mbias, mdark_rate, mflat):
    '''
    Reduce science frame given master bias, dark_rate and flat frames.
    This is 1.5x faster than doing the whole calculation at once because no 
    additional space needs to be allocated,
    ie. return (data - mbias - sci.exposure*mdark_rate)/mflat - median()

    Parameters
    ----------
    sci : ScienceFrame
        Image to load and reduce.
    mbias : numpy.ndarray
        master bias.
    mdark_rate : numpy.ndarray
        master dark normalized to exposure time in seconds.
    mflat : numpy.ndarray
        master flat.

    Returns
    -------
    data : numpy.ndarray
        reduced science image.

    '''
    data = sci.load()
    data -= mbias  # Step 2
    data -= sci.get("EXPOSURE")*mdark_rate  # Step 4
    data /= mflat  # Step 7
    # For photometry. We should actually use sigma_clipped_stats for this but
    # the difference is ~1e-2 per pix with max values ~1e5 -> negligible
    # median is 10 times faster (1s per pic vs 10s)
    data -= np.median(data)
    return data


    
@helper.functimer
def reduce_all(scis, mbias, mdark_rate, mflat, directory, new_object_name=None, 
               force_reduction=False, plotting=False, cutout_height=200):
    '''
    Reduce all science frames and save them in folder "Reduced". 
    Does nothing if reduced file already exists unless force_reduction=True.
    Naming convention: {color}_{i+1:03}.{ID}.FIT, eg V_001.GCVS_RR____Lyr_B.FIT

    Parameters
    ----------
    scis : ScienceFrameList
        List of all science frames, only contains headers and filenames.
    directory : str
        path to data, eg "../data/20251104_lab".
    force_reduction : bool, optional
        If true, reduces data even if a reduced file exists. The default is False.
    plotting : bool, optional
        If True, plots first image, reduced image and histogram for each filter. 
        The default is False.
    cutout_height : int, optional
        height of displayed image in px. The default is 100 
    Returns
    -------
    None.

    '''
    # positions = {"B": (1360, 1760), "V": (1300, 1680), "i": (1425, 1860), 
    #              "r": (1390, 1810), "u": (1370, 1775)}
    outdir = os.path.join(os.path.abspath(os.path.expanduser(directory)), "Reduced")
    os.makedirs(outdir, exist_ok=True)
    for color in scis.unique("filter"):  # iterate over filters
        frames = scis.filter(filter=color)
        print(f"Reducing filter {color}: n={len(frames)}")
        if plotting is True:
            reduced = reduce(frames[0], mbias, mdark_rate, mflat)
            name = os.path.basename(directory)
            positions = helper.get_dataset(name, "centers")
            plot.reduction(frames[0], reduced, color, positions, cutout_height, 
                           figdir="../figs")
            
        for i, frame in enumerate(frames):  # iterate over images
            # Make Filepath
            hdr = frame.header.copy()
            ID = hdr.get("OBJECT", "UNKNOWN")
            filename = f"{color}_{i+1:03}.{ID}.FIT"
            path = os.path.join(outdir, filename)
            if os.path.exists(path) and not force_reduction:
                # File already exists, no need to reduce
                continue
            # reduce image
            reduced = reduce(frame, mbias, mdark_rate, mflat)
            write_reduced_frame(reduced, hdr, path, new_object_name)
    return

def data_reduction(indir, rename_HAT=False, **reduce_all_kwargs):
    print("Starting Data reduction")
    scis, calibration = read(indir)
    mbias, mdark_rate, mflat = create_master_frames(calibration)
    new_object_name = "HAT-P-32" if rename_HAT else None
    reduce_all(scis, mbias, mdark_rate, mflat, indir, new_object_name=new_object_name, **reduce_all_kwargs)

