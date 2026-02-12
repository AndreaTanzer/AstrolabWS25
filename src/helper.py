# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 21:36:16 2026

@author: chris
"""

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from PIL import Image
from functools import wraps
from timeit import default_timer

class ScienceFrame:
    '''
    Store filename and header in a class. Gives quick access to exposure and 
    filter. Load data using load
    '''
    def __init__(self, path, header, focal_length: float=4500):
        self.path = path
        self.header = header
        self.focal_length = focal_length
    
    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return self.header.get(key.upper(), default)
    
    @property
    def coord(self):
        ra = self.get("OBJCTRA")
        dec = self.get("OBJCTDEC")
        return SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    
    @property 
    def scale(self):
        ''' 
        arcsec/px
        '''
        rad_to_arcsec = 180/np.pi*60**2
        return 1e-3*rad_to_arcsec*self.get("XPIXSZ")/self.focal_length
            
    def load(self, dtype=float):
        # Load science image data
        with fits.open(self.path, memmap=False) as hdul:
            return hdul[0].data.astype(dtype)
        
        
class ScienceFrameList(list):
    '''
    Container for ScienceFrames
    '''
    def filter(self, **criteria):
        '''
        Filter list of ScienceFrame by criteria. Case insensitive
        Keys can be either:
        - ScienceFrame properties (filter, exposure)
        - FITS header keywords (OBJECT, CCD-TEMP, AIRMASS, ...)
        '''
        def match(frame):
            return all(frame.get(key) == val
                       for key, val in criteria.items())
        return ScienceFrameList(f for f in self if match(f))
    
    def unique(self, key):
        '''
        Return unique values of a property or FITS header keyword.
        '''
        return sorted({f.get(key) for f in self})
    
class CalibFrame:
    """
    Represents a single calibration frame (bias, dark, flat).
    Stores path and header; data is loaded on demand.
    """
    def __init__(self, path, header, bin_size=None):
        self.path = path
        self.header = header
        self.bin_size = bin_size  # needed to bin flats
        
    def _bin_data(self, data):
        return np.asarray(Image.fromarray(data).reduce(self.bin_size))
    
    def load(self, dtype=float):
        with fits.open(self.path, memmap=False) as hdul:
            data = hdul[0].data.astype(dtype)
        if self.bin_size is not None:
            data = self._bin_data(data)
        return data

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        return self.header.get(key.upper(), default)
    
    @property
    def exposure(self):
        return self.header.get("EXPOSURE")

class CalibFrameList(list):
    """
    Container for calibration frames (bias, dark, flat).
    """

    def stack_data(self, dtype=float):
        return np.stack([f.load(dtype) for f in self])

    def stack_hdr(self):
        return [f.header for f in self]

    @property
    def exposures(self):
        return np.array([f.exposure for f in self])

    def filter(self, **criteria):
        def match(frame):
            return all(frame.get(k) == v for k, v in criteria.items())
        return CalibFrameList(f for f in self if match(f))

    def unique(self, key):
        return sorted({f.get(key) for f in self})
    
def functimer(f):
    '''
    prints runtime of function and function name
    Usage:
        @functimer
        def function(x):
            do_something
            return

    Parameters
    ----------
    f : function
        function that is timed.

    Returns
    -------
    function
        wrapped function.

    '''
    @wraps(f)
    def wrap(*args, **kwargs):
        starttime = default_timer()
        result = f(*args, **kwargs)
        endtime = default_timer()
        delta = endtime-starttime
        # print('%r took %.3f s' %(f.__name__, endtime-starttime))
        print(f"{f.__name__} took {delta:.3f} s")
        return result
    return wrap

def bin_data(data, bin_size):
    return np.asarray(Image.fromarray(data).reduce(bin_size))

def get_dataset(name):
    if name in DATASETS:
        return DATASETS[name]
    else:
        raise ValueError(f"invalid {name=}. Has to be in {DATASETS.keys()}")
        return

def calc_px_scale(px_size: float, focal_length: float=4500):
    '''
    Converts px size on chip in micrometer to px scale in arcsec/px using
    focal length of telescope in mm

    Parameters
    ----------
    px_size : float
        XPIXSZ.
    focal_length : float, optional
        Of Lustbühel cassegrain. The default is 4500.

    Returns
    -------
    px_scale
        arcsec/px.

    '''
    rad_to_arcsec = 180/np.pi*60**2
    return 1e-3*rad_to_arcsec*px_size/focal_length
    
DATASETS = {"20251104_lab": {"B": (1360, 1760), "V": (1300, 1680), 
                             "i": (1425, 1860), "r": (1390, 1810), 
                             "u": (1370, 1775)},
            "20260114_lab": {"B": (610, 780), "V": (490, 705), "r": (625, 800)}
            }