# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 18:37:36 2026

@author: chris
"""

import numpy as np
from astropy.stats import sigma_clipped_stats
from astroquery.astrometry_net import AstrometryNet
from photutils import detection, psf, aperture
# from photutils.aperture import CircularAperture
from matplotlib.pyplot import close
from timeit import default_timer

import helper
import plot
from io_data import read_folder

def find_stars(im: np.ndarray, thresh: float=200, fwhm: int=13, 
               roundhi: float=0.1, sharphi=0.9, minflux=500, printing: bool=False):
    '''
    Find stars with given constraints. returns table of stars sorted by brightness

    Parameters
    ----------
    im : np.ndarray
        Image containing stars.
    thresh : float, optional
        minimal peak brightness of stars. The default is 300.
    fwhm : int, optional
        maximum allowed width of star. The default is 9.
    roundness : float, optional
        maximum allowed deviation from round. The default is 0.1.
    printing : bool, optional
        wheter to print result. The default is True.

    Returns
    -------
    stars: astropy QTable

    '''
    # Find stars
    finder = detection.DAOStarFinder(thresh, fwhm, roundhi=roundhi, sharphi=sharphi)
    stars = finder.find_stars(im)  # None if no stars found
    # format cols in output (just for convenience)
    for col in stars.itercols():
        col.format = "{:.4g}"
    stars = stars[stars["flux"] > minflux]  # exclude hotpix
    stars.sort("xcentroid")
    if printing is True:
        stars.pprint()
    stars.sort("mag") # astroquery needs brightest stars first
    return stars

def plate_solve(frame, stars):
    ast = AstrometryNet()
    ast.api_key = "ogvqskenvnkfuqkj"
    scale = helper.calc_px_scale(frame.get("XPIXSZ"))
    wcs_header = ast.solve_from_source_list(stars["xcentroid"], stars["ycentroid"],
                    frame.get("NAXIS1"), frame.get("NAXIS2"), # parity=0,
                    center_dec=frame.coord.dec.degree, center_ra=frame.coord.ra.degree,
                    radius=0.1, scale_lower=0.9*scale, scale_upper=1.1*scale, 
                    scale_units="arcsecperpix", solve_timeout=120)
    return wcs_header

def aperture_photometry(frame):
    im = frame.load()
    im_stats = sigma_clipped_stats(im)  # mean, median, sigma
    im -= im_stats[1]  # subtract median
    
    stars = find_stars(im, minflux=1e4)
    xypos = list(zip(stars["xcentroid"], stars["ycentroid"]))
    # fit_shape: diameter of area to look for
    fwhms = psf.fit_fwhm(im, xypos=xypos, fit_shape=39)
    fwhm = int(np.median(fwhms))
    
    aptr = aperture.CircularAperture(xypos, r=fwhm)
    annulus = aperture.CircularAnnulus(xypos, r_in=2*fwhm, r_out=3*fwhm)
    aperstats = aperture.ApertureStats(im, annulus)
    bkg_mean = aperstats.mean
    phot_table = aperture.aperture_photometry(im, aptr)
    total_bkg = bkg_mean*aptr.area
    phot_table["aperture_sum_bkgsub"] = phot_table["aperture_sum"] - total_bkg
    for col in phot_table.itercols():
        col.format = "{:.4g}"
    return phot_table

if __name__ == "__main__":
    starttime = default_timer()
    #close("all")
    directory="../data/20251104_lab"
    # directory="../data/20260114_lab"
    
    # read data
    data = read_folder(directory+"/Reduced")
    phot_tables = []
    magsRR = []
    for i, frame in enumerate(data):
        print(f"{i}/{len(data)}")
        try:
            phot_table = aperture_photometry(frame)
            phot_tables.append(phot_table)
            if i == 47:
                print()
            magsRR.append(phot_table["aperture_sum_bkgsub"][0])
        except Exception as e:
            print(f"{i}: {e}")
            continue
    # frame = data[0]
    # im = frame.load()
    # im_stats = sigma_clipped_stats(im)  # mean, median, sigma
    # im -= im_stats[1]  # subtract median
    # #plot.imshow(im)
    
    # stars = find_stars(im, minflux=1e4)
    # xypos = list(zip(stars["xcentroid"], stars["ycentroid"]))
    # fwhms = psf.fit_fwhm(im, xypos=xypos, fit_shape=39)
    # fwhm = int(np.median(fwhms))
    
    # aptr = aperture.CircularAperture(xypos, r=fwhm)
    # annulus = aperture.CircularAnnulus(xypos, r_in=2*fwhm, r_out=3*fwhm)
    # aperstats = aperture.ApertureStats(im, annulus)
    # bkg_mean = aperstats.mean
    # phot_table = aperture.aperture_photometry(im, aptr)
    # total_bkg = bkg_mean*aptr.area
    # phot_table["aperture_sum_bkgsub"] = phot_table["aperture_sum"] - total_bkg
    # for col in phot_table.itercols():
    #     col.format = "{:.4g}"
    
    
    
    #  wcs_header = plate_solve(frame, stars)
    print('Execution Time: %.2f s' % (default_timer()-starttime))
    
    

    
    
    