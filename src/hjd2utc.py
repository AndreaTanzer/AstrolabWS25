# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 07:58:43 2026

@author: ChatGPT (almost unchanged)
https://rr-lyr.irap.omp.eu/dbrr/rrdb-V2.0_08.3.php?RR+Lyr& can be used to get the hjd, the ra_deg and the dec_deg
"""

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

def hjd2utc(hjd, ra_deg, dec_deg, location=None, print_result=False):
    """
    Convert HJD to UTC by removing heliocentric light-time correction.

    Parameters
    ----------
    hjd : float or array
        Heliocentric Julian Date
    ra_deg, dec_deg : float
        Target coordinates in degrees
    location : EarthLocation (optional)
        Observer location (default: geocenter)

    Returns
    -------
    utc_times : astropy.time.Time
        Times in UTC scale
    """
    
    if location is None:
        location = EarthLocation.of_site("greenwich")  # fallback
    
    # Target coordinates
    target = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    
    # Initial guess: treat HJD as JD in UTC
    t = Time(hjd, format='jd', scale='utc', location=location)
    
    # Compute heliocentric light travel time
    ltt_helio = t.light_travel_time(target, kind='heliocentric')
    
    # Subtract correction to recover emission time at Earth
    t_utc = t - ltt_helio
    
    if print_result:
        print(t_utc.iso)
    
    return t_utc


#example usage
if __name__ == '__main__':
    t_utc = hjd_to_utc(2460999.2879, ra_deg=291.36629, dec_deg=42.78436)
    print(t_utc.iso)