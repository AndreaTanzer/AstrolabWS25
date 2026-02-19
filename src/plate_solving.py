# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:32:20 2026

@author: chris
"""

import os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import cv2
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from astropy import wcs
import astropy.units as u
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from astroquery.gaia import Gaia
from astropy.table import Table
import astroalign as aa
from photutils import detection
import matplotlib.pyplot as plt
import warnings

import helper
import plot
import io_data
        
class PlateSolver:
    def __init__(self, frame):
        self.frame = frame
        self.path = frame.path
        self._im = None
        self.stars = None
    
    @property
    def im(self):
        ''' 
        To avoid loading image when just checking if it was solved before
        '''
        if self._im is None:
            self._im = self.frame.load()
        return self._im
    
    def in_history(self, string):
        ''' 
        Check if string is contained in any line of history
        '''
        history = self.frame.header.get("HISTORY", [])
        return any(string.upper() in str(record).upper() for record in history)
        
    @property
    def is_plate_solved(self):
        return self.in_history("Plate Solved")
    
    @property
    def failed_plate_solving(self):
        return self.in_history("Failed Plate Solving")
    
    def add_history(self, new_entry):
        '''
        Adds new_entry to history unless it is already part of it

        Parameters
        ----------
        new_entry : str
            New entry to history, eg Plate Solved or Failed Plate Solving.

        Returns
        -------
        header : fits.header.Header
            new header with added history.

        '''
        header = self.frame.header.copy()
        # otherwise multiple entries after multiple calls
        old_hist = [str(h) for h in header["HISTORY"]
                    if new_entry.upper() not in str(h).upper()]
        del header["HISTORY"]
        for line in old_hist:
            header.add_history(line)
        header.add_history(new_entry)
        return header
    
    def write_solved_frame(self, stars: Table, wcs_solution: wcs.wcs.WCS):
        '''
        Updates header with coordinates of frame and matched stars

        Parameters
        ----------
        stars : astropy.table.Table
            Contains gaia_id, coordinates and position in image of stars
        wcs_solution : wcs.wcs.WCS
            Contains coordinates, orientation and transformation of field.

        Returns
        -------
        None.

        '''
        path = Path(self.path)
        output_dir = path.parent.parent / "Solved"
        print(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / path.name
        with fits.open(self.path, memmap=True) as hdul:
            # update history
            hdr = hdul[0].header
            hist = hdr.get("HISTORY", [])
            if isinstance(hist, str):
                hist = [hist]
            filtered = [h for h in hist 
                        if "PLATE SOLVED" or "FAILED PLATE SOLVING" 
                        not in h.upper()]
            while "HISTORY" in hdr:
                del hdr["HISTORY"]
            for line in filtered:
                hdr.add_history(line)
            hdr.add_history("Plate Solved")
            hdr.update(wcs_solution.to_header())
            #primary_hdu = fits.PrimaryHDU(data=self.im, header=header)
            
            # create STARS extension
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=AstropyWarning)
                # would print
                # WARNING: Attribute `date` of type <class 'str'> cannot be added to FITS Header - skipping
                # WARNING: Attribute `version` of type <class 'dict'> cannot be added to FITS Header - skipping
                stars_hdu = fits.BinTableHDU(stars, name="STARS")
            hdul.append(stars_hdu)
            hdul.writeto(output_path, overwrite=True)
        return
    
    def write_header_failed(self):
        ''' 
        Update history with "Failed Plate Solving"
        '''
        with fits.open(self.path, mode="update") as hdul:
            hdr = hdul[0].header
            hist = hdr.get("HISTORY", [])
            if isinstance(hist, str):
                hist = [hist]
            # Avoid duplicates
            if not any("FAILED PLATE SOLVING" in h.upper() for h in hist):
                hdr.add_history("Failed Plate Solving")
        hdul.flush()
        return 
       
    def find_stars(self, thresh: float=200, fwhm: int=13, roundhi: float=1, 
                   sharplo: float=0, sharphi: float=1, peaklo: float=100, 
                   printing: bool=False):
        '''
        Find stars with given constraints. Writes table of stars sorted by brightness

        Parameters
        ----------
        thresh : float, optional
            minimal peak brightness of stars. The default is 200.
        fwhm : int, optional
            Gaussian kernel width ~ width of stars. The default is 13.
        roundhi, sharplo, sharphi : float, optional
            parameters of DAOStarFinder.
        peaklo : float, optional
            minimum central peak of found star. Vastly different between obs nights
        printing : bool, optional
            wheter to print result. The default is True.

        Returns
        -------
        None

        '''
        # only place where opencv is needed. Takes ~0.3s, alternatives need ~10s
        im_filtered = cv2.medianBlur(self.im.astype('float32'), 3)
        # Find stars
        finder = detection.DAOStarFinder(thresh, fwhm, roundhi=roundhi, 
                                         sharplo=sharplo, sharphi=sharphi)
        stars = finder.find_stars(im_filtered)  # None if no stars found
        assert stars is not None
        stars = stars[stars["peak"] > peaklo]  # exclude very low contrast stars
        stars.sort("xcentroid")
        # format cols in output (just for convenience)
        for col in stars.itercols():
            col.format = "{:.4g}"
        n_stars = len(stars)
        if printing is True:
            stars.pprint_all()
            print(f"found {n_stars} stars")
        stars.sort("mag") # astroquery needs brightest stars first
        self.stars = stars
        assert n_stars > 5
    
    def query_gaia(self, radius: u.quantity.Quantity, mag_limit: float=13,
                   n_max: int=200):
        '''
        Search for gaia stars within radius of frame coordinates
        
        Possible Errors:
            requests.exceptions.HTTPError: 500
            HTTPError: Error 404

        Parameters
        ----------
        radius : u.quantity.Quantity
            search radius around target coords.
        mag_limit : float, optional
            faintest star to include. The default is 13.
        n_max : int, optional
            restrict number of returned stars. Otherwise aa.find_transform in
            solve_wcs might time out

        Returns
        -------
        gaia_table : astropy.table.table.Table
            table with coordinates and magnitudes of stars.

        '''
        ra = self.frame.coord.ra.deg
        dec = self.frame.coord.dec.deg
        sr = (2 * radius).to(u.deg).value # Search radius in degrees
        query = f"""
        SELECT source_id, ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {sr}))
        AND phot_g_mean_mag < {mag_limit}
        ORDER BY phot_g_mean_mag ASC
        """
        # get gaia stars
        job = Gaia.launch_job_async(query)
        gaia_table = job.get_results()
        return gaia_table[:n_max]
    
    def solve_wcs(self, gaia_table, max_stars=30, plotting=False) -> wcs.wcs.WCS:
        '''
        Plate solve using star positions in image and coordinates of Gaia stars

        Parameters
        ----------
        gaia_table : astropy.table.table.Table
            table with coordinates and magnitudes of stars.
        max_stars : int
            maximum number of stars in im used to match field.
            needed because otherwise magnitude limit of gaia is exceeded
            and aa.find_transform would run into a timeout
        plotting : bool
            wheter to plot or not. Mostly for debugging. Would take ages, might
            fail with parallel processing

        Returns
        -------
        w_final : wcs.wcs.WCS
            Contains coordinates, orientation and transformation of field.

        '''
        nx = self.frame.get("NAXIS1")
        ny = self.frame.get("NAXIS2")
        # setup WCS
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [nx/2, ny/2]
        w.wcs.cdelt = np.array([-self.frame.scale/3600, self.frame.scale/3600])
        w.wcs.crval = [self.frame.coord.ra.deg, self.frame.coord.dec.deg]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        # project GAIA to image coords
        gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], 
                               unit=(u.deg, u.deg))
        gaia_px_x, gaia_px_y = w.world_to_pixel(gaia_coords)
        # restrict to area around visible image
        # cant use radius alone because star is off-center
        mask = (gaia_px_x > -nx*0.5) & (gaia_px_x < nx*1.5) & \
           (gaia_px_y > -ny*0.5) & (gaia_px_y < ny*1.5)
        source_pts = np.transpose((self.stars['xcentroid'], self.stars['ycentroid']))
        source_pts = source_pts[:max_stars]
        target_pts = np.transpose((gaia_px_x[mask], gaia_px_y[mask]))
        nsource = len(source_pts)
        ntarget = len(target_pts)
        # optional: plot star fields
        if plotting is True:
            plt.figure(figsize=(10, 8))
            plt.scatter(source_pts[:,0], source_pts[:,1], color='red', 
                        label='Detected', alpha=0.5)
            plt.scatter(target_pts[:,0], target_pts[:,1], color='blue',
                        label='Gaia (Projected)', marker='+')
            plt.title("Visual Overlap Check")
            plt.legend()
            plt.show()
        # check if enough detected stars and Gaia stars are available
        if nsource < 5 or ntarget < 5:
            plot_stars(self.im, self.stars, title=f"{self.path}")
            # print(f"Too few source points: {nsource} or target points: {ntarget}")
            self.write_header_failed()
            return None
        # match sources with targets using similar triangles
        try:
            transf, (source_list, target_list) = aa.find_transform(source_pts, target_pts)
        except aa.MaxIterError:
            plot_stars(self.im, self.stars, title=f"{self.path}")
            # print(f"Astroalign failed for {self.path}, too few stars, cloudy?")
            self.write_header_failed()
            return None
        except TypeError as e:
            print(e)
            return None
        # build WCS from matched pairs
        tree = cKDTree(target_pts)
        distances, indices = tree.query(target_list)
        matched_coords = gaia_coords[mask][indices]
        w_final = wcs.utils.fit_wcs_from_points(xy=(source_list[:,0], source_list[:,1]), 
                                                world_coords=matched_coords,
                                                projection=w)
        # check diffs between fit and sources
        reprojected_x, reprojected_y = w_final.world_to_pixel(matched_coords)
        rms = np.sqrt(np.mean((reprojected_x - source_list[:,0])**2 + (
                                reprojected_y - source_list[:,1])**2))
        if rms > 2:  # px
            print(f"{self.path}")
            print(f"RMS Fit Error: {rms:.3f} pixels")
            rotation = np.degrees(np.arctan2(w_final.wcs.cd[0, 1], w_final.wcs.cd[1, 1]))
            scale_fit = wcs.utils.proj_plane_pixel_scales(w_final)[0] * 3600
            print("--- Fit Results ---")
            print(f"Matched Stars: {len(source_list)}")
            print(f"Field Rotation: {rotation:.2f}°")
            print(f"Measured Scale: {scale_fit:.4f} arcsec/px")
            print(f"Focal Length Check: {4500 * (self.frame.scale / scale_fit):.1f} mm")
        self.write_solved_frame(self.stars["xcentroid", "ycentroid"], w_final)
        return w_final



def plot_stars(im, stars, title=""):
    ny, nx = im.shape
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(stars['xcentroid'], stars['ycentroid'], s=50, edgecolors='r', facecolors='none', label='Detected')
    plot.imshow_on_ax(ax, im)
    fig.legend()
    ax.set_title(title)
    plt.show()

# @helper.functimer
def plate_solve_filter(data: helper.ScienceFrameList, force_solve=False, 
                       plotting=False, **find_kwargs):
    '''
    Plate solve all frames in data

    Parameters
    ----------
    data : helper.ScienceFrameList
        DESCRIPTION.
    force_solve : bool, optional
        If true resolve images that were solved before. The default is False.
    **find_kwargs : dict
        kwargs of PlateSolver.find_stars.

    Raises
    ------
    e
        query_gaia exception if it failed 10 times.

    Returns
    -------
    None.

    '''
    n_frames = len(data)
    gaia_table = None
    for i, d in enumerate(data): 
        solver = PlateSolver(d)
        already_done = solver.is_plate_solved or solver.failed_plate_solving
        if already_done and not force_solve:
            # already plate solved, no need to calculate anything
            # or solving failed before
            continue
        # find gaia stars once, than match each image to those stars
        if gaia_table is None:
            for j in range(10):  # randomly throws errors
                try:  # if larger than 2r, star wouldnt be in FOV
                    gaia_table = solver.query_gaia(1.5*d.radius)
                    break
                except Exception as e:
                    if j == 9:
                        raise e
                    continue
                
        print(f"plate solving {i+1}/{n_frames}", end="")
        try:
            solver.find_stars(**find_kwargs)
        except AssertionError:
            hdr = solver.add_history("Failed Plate Solving")
            hdu = fits.PrimaryHDU(data=solver.frame.load(), header=hdr)
            hdu.writeto(solver.path, overwrite=True)
            print(f"not enough stars: {len(solver.stars) if solver.stars else 0}")
            print(f"{solver.path}")
            
            continue
        print(f", found {len(solver.stars)} stars")
        # helper.print_on_line(f", found {len(solver.stars)} stars")
        solver.solve_wcs(gaia_table, plotting=plotting)

@helper.functimer
#@helper.profiler(n=20)
def plate_solve_all(data: helper.ScienceFrameList, force_solve: dict | bool=False, 
                    printing: bool=True, filter_kwargs: dict=None):
    '''
    Add WCS information to header. Includes coordinates, orientation, scale

    Parameters
    ----------
    data : helper.ScienceFrameList
        science frames containing data.
    force_solve : dict, bool, optional
        specify if all filters or a specific filter must be solved. The default is False
    printing : bool, optional
        specify if plate_solve_filter should print. The default is True.
    filter_kwargs : dict, optional
        

    Returns
    -------
    None.

    '''
    if filter_kwargs is None:
        name = data[0].path.split("\\")[-3]
        filter_kwargs = helper.get_dataset(name, "find_stars")

    filters = data.unique("filter")
    force_dict = {}
    if isinstance(force_solve, bool):
        for filt in filters:
            force_dict[filt] = force_solve
    else:
        for filt in filters:
            if filt in force_solve:
                force_dict[filt] = force_solve[filt]
            else:
                force_dict[filt] = False
    for filt in filters:
        current_kwargs = filter_kwargs.get(filt, {})
        if printing is True:
            plate_solve_filter(data.filter(filter=filt), 
                               force_solve=force_dict[filt], 
                               **current_kwargs)
        else:
            with helper.HiddenPrints():
                plate_solve_filter(data.filter(filter=filt), plotting=True,
                                   force_solve=force_dict[filt], 
                                   **current_kwargs)

# def plate_solve(frame, stars):
      # this is what its supposed to be but doesnt work
      # (would have taken 5-10min per image)
#     ast = AstrometryNet()
#     ast.api_key = "ogvqskenvnkfuqkj"
#     scale = helper.calc_px_scale(frame.get("XPIXSZ"))
#     wcs_header = ast.solve_from_source_list(stars["xcentroid"], stars["ycentroid"],
#                     frame.get("NAXIS1"), frame.get("NAXIS2"), # parity=0,
#                     center_dec=frame.coord.dec.degree, center_ra=frame.coord.ra.degree,
#                     radius=0.1, scale_lower=0.9*scale, scale_upper=1.1*scale, 
#                     scale_units="arcsecperpix", solve_timeout=120)
#     return wcs_header

if __name__ == "__main__":
    plt.close("all")
    directory="../data/20251104_lab"
    directory="../data/20260114_lab"
    # read data
    data = io_data.read_folder(directory+"/Reduced")
    plate_solve_all(data, force_solve=True, printing=False)
    
    # example usage
    frame = data.filter(filter="B")[5]
    stars = frame.load_stars()  
    xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    w = wcs.WCS(frame.header)
    


