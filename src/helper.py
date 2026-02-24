# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 21:36:16 2026

@author: chris
"""

import sys
import os
import re
import unicodedata
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from PIL import Image
from functools import wraps
from timeit import default_timer
import cProfile
import pstats
from pathlib import Path


def get_repo_root(base_dir: Path | None = None) -> Path:
    """Return the repository root directory (parent of src/)."""
    return base_dir or Path(__file__).resolve().parents[1]


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
    
    @property
    def radius(self):
        ''' 
        half diagonal in arcsec
        '''
        return 0.5*np.sqrt(self.get("NAXIS1")**2 + self.get("NAXIS2")**2
                           )*self.scale*u.arcsec
            
    def load(self, dtype=float):
        ''' 
        Load science image data, eg. im=frame.load()
        '''
        with fits.open(self.path, memmap=False) as hdul:
            return hdul[0].data.astype(dtype)
    
    def load_stars(self):
        ''' 
        loads matched gaia catalogue if it exists, eg. stars=frame.load_stars()
        '''
        with fits.open(self.path) as hdul:
            if len(hdul) > 1 and "STARS" in hdul:
                return Table.read(hdul["STARS"])
            else:
                print(f"No matched stars in {self.path}")
                return None
        
        
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
    
class HiddenPrints:
    # https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
    # Usage:
    # with HiddenPrints():
    #   print("This will not be printed")
    # print("This will be printed as before")
    def __enter__(self):
        self._original_stdout = sys.stdout
        # Redirect to devnull, but don't "close" the system handle
        self._devnull = open(os.devnull, 'w')
        sys.stdout = self._devnull

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Swap the original back BEFORE closing devnull
        sys.stdout = self._original_stdout
        self._devnull.close()
    
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

def profiler(f_py=None, n=10, sortBy='cumtime'):
    '''
    Usage: @profiler
            def function():

    Parameters
    ----------
    n : int, optional
        Number of subfunctions printed. The default is 10.
    sortBy : str, optional
        one of ('ncalls','tottime','cumtime'). The default is 'tottime'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    assert callable(f_py) or f_py is None

    def _decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            result = f(*args, **kwargs)
            profiler.disable()
            stats = pstats.Stats(
                profiler).sort_stats(sortBy)
            stats.print_stats(n)
            return result
        return wrapper
    return _decorator(f_py) if callable(f_py) else _decorator

def bin_data(data, bin_size):
    return np.asarray(Image.fromarray(data).reduce(bin_size))

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

def print_statusline(msg: str):
    last_msg_length = len(
        getattr(print_statusline, 'last_msg', ''))
    print(' ' * last_msg_length, end='\r')
    print(msg, end='\r')
    # Some say they needed this, I didn't.
    sys.stdout.flush()
    setattr(print_statusline, 'last_msg', msg)

def print_on_line(msg: str, ncols: int=int(1e3)):
    # go to the (start of the) previous line: \033[F
    # move along ncols: \033[{ncols}G
    print(f"\033[F\033[{ncols}G{msg}")
    
def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')
    
def get_dataset(labname, kind):
    dataset = DATASETS.get(labname)
    if dataset is not None: 
        dataset = dataset.get(kind)
    if dataset is None:
        if labname not in DATASETS:
            raise ValueError(f"invalid {labname=}. Has to be in {DATASETS.keys()}")
        if kind not in DATASETS[labname]:
            raise ValueError(f"invalid {kind=}. Has to be in {DATASETS[labname].keys()}")
    return dataset

DATASETS = {"20251104_lab": {"centers": {"B": (1360, 1760), "V": (1300, 1680), 
                                         "i": (1425, 1860), "r": (1390, 1810), 
                                         "u": (1370, 1775)},
                             "find_stars": {
                                "B": dict(thresh=200, roundhi=0.5, 
                                          sharplo=0.5, sharphi=0.9, peaklo=100), 
                                "V": dict(thresh=200, peaklo=200), 
                                "i": dict(thresh=200, peaklo=300), 
                                "r": dict(thresh=150, peaklo=150), 
                                "u":  dict(thresh=100, peaklo=50)},
                             },
            "20260114_lab": {"centers": {"B": (610, 780), "V": (490, 705), 
                                         "r": (625, 800)},
                             "find_stars": {
                                 "B": dict(thresh=1000, peaklo=1000), # (1000, 800)
                                 "V": dict(thresh=3000, peaklo=3000),
                                 "r": dict(thresh=1000, peaklo=1000)}
                             }
            }