# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 20:18:20 2026

@author: chris
"""

import os 
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import logging

import helper
import io_data
from plate_solving import PlateSolver


# used locally by each worker process
_worker_star_table = None

def solve_single_frame_task(frame, find_kwargs, force_solve):
    """
    The actual task run for every image.
    """
    global _worker_star_table
    name = str(frame.path).split(os.sep)[-1]
    solver = PlateSolver(frame)
    already_done = solver.is_plate_solved or solver.failed_plate_solving
    if already_done and not force_solve:
        # already plate solved, no need to calculate anything
        # or solving failed before
        return None  # f"SKIP: {name}"
    
    try:
        # 1. Detection
        solver.find_stars(**find_kwargs)
        if solver.stars is None:
            return f"FAIL: {name} - No stars found"
    except AssertionError as e:
        io_data.write_header_failed(solver.path)
        # print(f"not enough stars: {len(solver.stars) if solver.stars else 0}")
        # print(f"{solver.path}")
        return f"FAIL: {name} - {str(e)}"
    # 2. Solving (Using the table already in worker memory)
    w = solver.solve_wcs(_worker_star_table)
    if w is None:
        return f"FAIL: {name} - Couldn't solve"
    # 3. Use coordinates to search at all star positions with weaker pos constraints,
    #    extract magnitudes photometric calibration
    star_data = solver.extract_star_data(w)
    io_data.write_solved_frame(solver.path, star_data, w)
    return None  # f"DONE: {name} ({len(solver.stars)} stars)"

def init_worker(shared_table):
    """
    This runs ONCE per CPU core when the process starts.
    It sets the Gaia table in the worker's global memory.
    """
    global _worker_star_table
    _worker_star_table = shared_table
    # Suppress No sources found warnings
    # warnings.filterwarnings('ignore', category=UserWarning, append=True)
    # Suppress Gaia password warnings
    logging.getLogger('astroquery').setLevel(logging.ERROR)

@helper.functimer
def plate_solve_all(data: helper.ScienceFrameList, force_solve: bool=False, 
                    filter_kwargs: dict=None):
    '''
    Add WCS information to header. Includes coordinates, orientation, scale

    Parameters
    ----------
    data : helper.ScienceFrameList
        science frames containing data.
    force_solve : dict, bool, optional
        specify if all filters or a specific filter must be solved. The default is False
    filter_kwargs : dict, optional
        

    Returns
    -------
    None.

    '''
    # get find_stars kwargs for each filter
    if filter_kwargs is None:
        path_str = str(data[0].path)
        name = path_str.split(os.sep)[-3]
        filter_kwargs = helper.get_dataset(name, "find_stars")
    
    # 2. Global Star Query
    print("Fetching Global UCAC4 reference catalog...")
    temp_solver = PlateSolver(data[0])
    for j in range(10):  # randomly throws errors
        try:  # if larger than 2r, star wouldnt be in FOV
            star_table = temp_solver.query_vizier(2*data[0].radius)
            # gaia_table = temp_solver.query_gaia(1.5*data[0].radius)
            break
        except Exception as e:
            if j == 9:
                raise e
    
    # force_solve for each filter, defaulting to False if not specified
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
                
    # Create task list for all filters
    all_tasks = []
    for frame in data:
        filt = frame.get("filter")
        # Match the specific kwargs for this frame's filter
        current_kwargs = filter_kwargs.get(filt, {})
        # Check force_solve logic for this specific filter
        should_force = force_dict.get(filt, False)
        all_tasks.append((frame, current_kwargs, should_force))
    
    # Launch 1 pool for entire dataset
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Launching pool for {len(all_tasks)} frames across {num_workers} cores...")
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, 
                             initargs=(star_table,)) as executor:
        # We pass the pre-matched kwargs directly into the submit call
        futures = [executor.submit(solve_single_frame_task, t[0], t[1], t[2]) for t in all_tasks]
        
        for future in futures:
            output = future.result()
            if output is not None:
                print(output)


if __name__ == "__main__":
    plt.close("all")
    directory="../data/20251104_lab"
    directory="../data/20260114_lab"
    # read data
    data = io_data.read_folder(directory+"/Reduced")
    plate_solve_all(data, force_solve=True)
    
    # example usage
    # data = io_data.read_folder(directory+"/Solved")
    # frame = data.filter(filter="B")[5]
    # stars = frame.load_stars()  
    # xypos = np.transpose((stars['xcentroid'], stars['ycentroid']))
    # w = wcs.WCS(frame.header)
    


