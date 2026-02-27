# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 18:32:20 2026

@author: chris
"""

import numpy as np
from scipy.spatial import cKDTree
import cv2
from astropy.coordinates import SkyCoord
from astropy import wcs, table
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier
import astroalign as aa
from photutils import detection, centroids
import matplotlib.pyplot as plt
import os
import time

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
    
    def center_stars(self, im, x_init, y_init, box_size):
        """ Recenter stars using centroid_sources
            Changes position by upt to 0.5 px"""
        x_new, y_new = centroids.centroid_sources(im, x_init, y_init, 
            box_size=box_size, centroid_func=centroids.centroid_2dg)
        return x_new, y_new
    
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
        if stars is None:
            return None
        stars = stars[stars["peak"] > peaklo]  # exclude very low contrast stars
        # recentering, possible but takes longer and barely changes output
        # stars["xcentroid"], stars["ycentroid"] = self.center_stars(
        #     self.im, stars["xcentroid"], stars["ycentroid"], fwhm*3)
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
        sr = radius.to(u.deg).value # Search radius in degrees
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
    
    def query_vizier(self, radius: u.quantity.Quantity, mag_limit: float=13,
                     n_max: int=300):
        viz = Vizier(columns=["UCAC4", "_RAJ2000", "_DEJ2000", "Bmag", "Vmag", "rmag", "imag"],
                     column_filters={"Vmag": f"<{mag_limit}"}, catalog="I/322A",
                     row_limit=-1)
        result = viz.query_region(self.frame.coord, radius=2*radius)[0]
        result.rename_columns(["_RAJ2000", "_DEJ2000"], ["ra", "dec"])
        # otherwise sorted by id~coords, throws out bright
        result.sort("Vmag")
        return result[:n_max]
    
    def query_around_vizier(self, w, radius_arcsec=20, mag_limit: float=13):
        coords_with_idx = self.get_star_coords(w)
        _, ra, dec = coords_with_idx.T
        # _q is the original index in result
        upload_table = table.QTable([ra*u.deg, dec*u.deg], 
                                    names=["_RAJ2000", "_DEJ2000"])
        # catalog="I/322A" corresponds to UCAC4
        viz = Vizier(columns=["UCAC4", "_RAJ2000", "_DEJ2000", "Bmag", "Vmag", "rmag", "imag"],
                     column_filters={"Vmag": f"<{mag_limit}"}, catalog="I/322A")
        result = viz.query_region(upload_table, radius=f"{radius_arcsec}s")[0]
        result.rename_columns(["_RAJ2000", "_DEJ2000"], ["ra", "dec"])
        return result
    
    def solve_wcs(self, star_table, max_stars=30, plotting=False) -> wcs.wcs.WCS:
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
        # project star coords to pos in image
        star_coords = SkyCoord(ra=star_table['ra'], dec=star_table['dec'], 
                               unit=(u.deg, u.deg))
        star_px_x, star_px_y = w.world_to_pixel(star_coords)
        # restrict to area around visible image
        # cant use radius alone because star is off-center
        mask = (star_px_x > -nx*0.5) & (star_px_x < nx*1.5) & \
           (star_px_y > -ny*0.5) & (star_px_y < ny*1.5)
        source_pts = np.transpose((self.stars['xcentroid'], self.stars['ycentroid']))
        source_pts = source_pts[:max_stars]
        target_pts = np.transpose((star_px_x[mask], star_px_y[mask]))  # [:max_stars]
        nsource = len(source_pts)
        ntarget = len(target_pts)
        # check if enough detected stars and Gaia stars are available
        if nsource < 5 or ntarget < 5:
            if plotting:
                plot_stars(self.im, self.stars, title=f"{self.path.name}")
            # print(f"Too few source points: {nsource} or target points: {ntarget}")
            io_data.write_header_failed(self.path)
            return None
        # match sources with targets using similar triangles
        print(f"solve_wcs: {self.path}", flush=True)
        print(f"solve_wcs: nsource={nsource} ntarget={ntarget} max_stars={max_stars}", flush=True)
        print(f"solve_wcs: source_pts shape={source_pts.shape} target_pts shape={target_pts.shape}", flush=True)

        t0 = time.perf_counter()
        print("solve_wcs: aa.find_transform START", flush=True)
        try:
            _, (source_list, target_list) = aa.find_transform(source_pts, target_pts)
        except aa.MaxIterError:
            print("solve_wcs: aa.find_transform MaxIterError", flush=True)
            if plotting:
                plot_stars(self.im, self.stars, title=f"{self.path.name}")
            # print(f"Astroalign failed for {self.path}, too few stars, cloudy?")
            io_data.write_header_failed(self.path)
            return None
        except TypeError as e:
            print(f"solve_wcs: aa.find_transform TypeError: {e}", flush=True)
            return None
        dt = time.perf_counter() - t0
        print(f"solve_wcs: aa.find_transform DONE dt_s={dt:.2f} matched={len(source_list)}", flush=True)
        # build WCS from matched pairs
        tree = cKDTree(target_pts)
        _, indices = tree.query(target_list)
        matched_coords = star_coords[mask][indices]
        w_final = wcs.utils.fit_wcs_from_points(xy=(source_list[:,0], source_list[:,1]), 
                                                world_coords=matched_coords,
                                                projection=w)
        # check diffs between fit and sources
        reprojected_x, reprojected_y = w_final.world_to_pixel(matched_coords)
        rms = np.sqrt(np.mean((reprojected_x - source_list[:,0])**2 + (
                                reprojected_y - source_list[:,1])**2))
        if rms > 2:  # px
            print(f"{self.path.name}")
            print(f"RMS Fit Error: {rms:.3f} pixels")
            rotation = np.degrees(np.arctan2(w_final.wcs.cd[0, 1], w_final.wcs.cd[1, 1]))
            scale_fit = wcs.utils.proj_plane_pixel_scales(w_final)[0] * 3600
            print("--- Fit Results ---")
            print(f"Matched Stars: {len(source_list)}")
            print(f"Field Rotation: {rotation:.2f}°")
            print(f"Measured Scale: {scale_fit:.4f} arcsec/px")
            print(f"Focal Length Check: {4500 * (self.frame.scale / scale_fit):.1f} mm")
        return w_final

    def get_star_coords(self, w: wcs.wcs.WCS) -> np.array:
        xypos = np.transpose((self.stars['xcentroid'], self.stars['ycentroid']))
        # 0 because python starts counting at 0
        star_coords = w.all_pix2world(xypos, 0)
        ra, dec = star_coords[:, 0], star_coords[:, 1]
        idx = np.arange(ra.size) + 1
        coords_with_idx = np.vstack([idx, ra, dec]).T
        return coords_with_idx
    
    def extract_star_data(self, w: wcs.wcs.WCS):
        result = self.query_around_vizier(w)
        # Add centroids
        result["xcentroid"] = self.stars[result["_q"]-1]["xcentroid"]
        result["ycentroid"] = self.stars[result["_q"]-1]["ycentroid"]
        # Drop duplicates, keeping star closest to input coords
        df = result.to_pandas()
        # calc dist between search coords and found coords
        coords_with_idx = self.get_star_coords(w)
        search_coords = coords_with_idx[result["_q"]-1][:, 1:]
        found_coords = df[["ra", "dec"]].values
        df["coord_err"] = np.linalg.norm(search_coords-found_coords, axis=1)
        # keep stars with lowest distance
        df = df.loc[df.groupby("_q")["coord_err"].idxmin()]
        filtered = table.Table.from_pandas(df)
        return filtered


def plot_stars(im, stars, title=""):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(stars['xcentroid'], stars['ycentroid'], s=50, edgecolors='r', facecolors='none', label='Detected')
    plot.imshow_on_ax(ax, im)
    fig.legend()
    ax.set_title(title)
    plt.show()

# @helper.functimer
# @helper.profiler(n=50)
def plate_solve_filter(all_data: helper.ScienceFrameList, filt: str, 
                       force_solve=False, plotting=False, **find_kwargs):
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
    data = all_data.filter(filter=filt)
    n_frames = len(data)
    star_table = None
    init_solver = PlateSolver(data[0])
    # find gaia stars once, than match each image to those stars
    if star_table is None:
        for i in range(10):  # randomly throws errors
            try:  # if larger than 2r, star wouldnt be in FOV
                # star_table = solver.query_gaia(2*d.radius)
                star_table = init_solver.query_gaia(2*data[0].radius)
                star_table.sort("phot_g_mean_mag")
                break
            except Exception as e:
                if i == 9:
                    raise e
                continue
    for i, d in enumerate(data): 
        solver = PlateSolver(d)
        already_done = solver.is_plate_solved or solver.failed_plate_solving
        if already_done and not force_solve:
            # already plate solved, no need to calculate anything
            # or solving failed before
            continue                
        print(f"plate solving {i+1}/{n_frames}", end="")
        try:
            solver.find_stars(**find_kwargs)
        except AssertionError:
            print("  step: find_stars FAILED (AssertionError)", flush=True)
            io_data.write_header_failed(solver.path)
            continue
        print(f"  step: find_stars DONE  found {len(solver.stars)} stars", flush=True)
        print("  step: solve_wcs START", flush=True)

        # helper.print_on_line(f", found {len(solver.stars)} stars")
        w = solver.solve_wcs(star_table, plotting=plotting)
        print("  step: solve_wcs RETURNED", flush=True)

        if w is None:
            print("  step: solve_wcs returned None", flush=True)
            continue
        # needed for photometric calibration
        print("  step: extract_star_data START", flush=True)
        star_data = solver.extract_star_data(w)
        print("  step: extract_star_data DONE", flush=True)
        print("  step: write_solved_frame START", flush=True)
        io_data.write_solved_frame(solver.path, star_data, w)
        print("  step: write_solved_frame DONE", flush=True)
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
    filter_kwargs = helper.get_filter_kwargs(data, filter_kwargs)
    filters = data.unique("filter")
    force_dict = helper.build_force_dict(filters, force_solve)

    filters = data.unique("filter")

    for filt in filters:
        current_kwargs = filter_kwargs.get(filt, {})
        if printing is True:
            plate_solve_filter(data, filt, plotting=False,
                               force_solve=force_dict[filt], 
                               **current_kwargs)
        else:
            with helper.HiddenPrints():
                plate_solve_filter(data.filter, filt, plotting=False,
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
    # directory="../data/20260114_lab"
    # read data
    data = io_data.read_folder(directory+"/Reduced")
    # plate_solve_all(data, force_solve=True, printing=False)
    plate_solve_filter(data, "B", force_solve=True)
    
    # example usage
    # data = io_data.read_folder(directory+"/Solved")
    # frame = data.filter(filter="B")[5]
    # stars = frame.load_stars()  
    # xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    # w = wcs.WCS(frame.header)
    


