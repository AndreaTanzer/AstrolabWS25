# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:32:20 2026

@author: chris
"""

import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import wcs
import astropy.units as u
from astroquery.gaia import Gaia
from photutils import detection
from photutils.centroids import centroid_com
from matplotlib.pyplot import close


import helper
import plot
from io_data import read_folder


def find_centroid(im, stars, fwhm):
    ny, nx = im.shape
    x_new = []
    y_new = []
    xy = []
    box = int(3*fwhm)
    for x0, y0 in zip(stars["xcentroid"], stars["ycentroid"]):
        x0 = int(x0)
        y0 = int(y0)
        x_min = max(0, x0 - box)
        x_max = min(nx, x0 + box)
        y_min = max(0, y0 - box)
        y_max = min(ny, y0 + box)
        cutout = im[y_min:y_max, x_min:x_max]
        cy, cx = centroid_com(cutout)
        x_new.append(x_min + cx)
        y_new.append(y_min + cy)
        xy.append((x_min + cx, y_min + cy))
    return x_new, y_new
        
    
def find_stars(im: np.ndarray, thresh: float=200, fwhm: int=13, 
               roundhi: float=0.1, sharplo=0.5, sharphi=0.9, printing: bool=True):
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
    finder = detection.DAOStarFinder(thresh, fwhm, roundhi=roundhi, 
                                     sharplo=sharplo, sharphi=sharphi)
    stars = finder.find_stars(im)  # None if no stars found
    # format cols in output (just for convenience)
    stars = stars[stars["flux"] > 1000]  # exclude hotpix
    stars["xcentroid"], stars["ycentroid"] = find_centroid(im, stars, fwhm)
    stars.sort("xcentroid")
    for col in stars.itercols():
        col.format = "{:.4g}"
    if printing is True:
        stars.pprint()
    stars.sort("mag") # astroquery needs brightest stars first
    return stars




if __name__ == "__main__":
    # close("all")
    directory="../data/20251104_lab"
    # directory="../data/20260114_lab"
    
    # read data
    data = read_folder(directory+"/Reduced")
    frame = data[0]
    im = frame.load()
    im_stats = sigma_clipped_stats(im)  # mean, median, sigma
    im -= im_stats[1]  # subtract median
    # plot.imshow(im)
    
    stars = find_stars(im, thresh=300, printing=False)
    nx = frame.get("NAXIS1")
    ny = frame.get("NAXIS2")
    radius = frame.scale*nx*u.arcsec
    
    job = Gaia.cone_search_async(frame.coord, radius=radius)
    result = job.get_results()
    gaia = result[result["phot_g_mean_mag"] < 16]
    gaia_coords = SkyCoord(gaia["ra"], gaia["dec"], unit=(u.deg, u.deg))
    x = stars["xcentroid"]
    y = stars["ycentroid"]
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nx/2, ny/2]
    w.wcs.cdelt = np.array([-frame.scale/3600, frame.scale/3600])
    w.wcs.crval = [frame.coord.ra.deg, frame.coord.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    # match detected stars to gaia stars
    sky_est = w.pixel_to_world(x, y)
    idx1, sep1, _ = sky_est.match_to_catalog_sky(gaia_coords)
    idx2, sep2, _ = gaia_coords.match_to_catalog_sky(sky_est)
    mutual = np.arange(len(x)) == idx2[idx1]
    good = mutual & (sep1 < 0.15 * u.deg)
    
    # # Contaminated by outliers
    # idx, sep2d, _ = sky_est.match_to_catalog_sky(gaia_coords)
    # # fit wcs
    # w_fitted = wcs.utils.fit_wcs_from_points((x, y), gaia_coords[idx], 
    #                                          projection="TAN")
    # sky_new = w_fitted.pixel_to_world(x, y)
    # idx2, sep2d2, _ = sky_new.match_to_catalog_sky(gaia_coords)
    # print(f"median residual={np.median(sep2d2.to(u.arcsec))}")
    # good = sep2d2 < 10*u.arcsec
    # w_refit = wcs.utils.fit_wcs_from_points((x[good], y[good]), 
    #                                         gaia_coords[idx2[good]],
    #                                         projection="TAN")
    # sky_refit = w_refit.pixel_to_world(x, y)
    # idx2, sep2d2, _ = sky_refit.match_to_catalog_sky(gaia_coords)





