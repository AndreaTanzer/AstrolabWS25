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
    global _worker_star_table

    name = str(frame.path).split(os.sep)[-1]
    print(f"[WORKER] START {name}", flush=True)

    solver = PlateSolver(frame)

    already_done = solver.is_plate_solved or solver.failed_plate_solving
    if already_done and not force_solve:
        print(f"[WORKER] SKIP {name}", flush=True)
        return None

    try:
        print(f"[WORKER] {name} find_stars START", flush=True)
        solver.find_stars(**find_kwargs)
        print(f"[WORKER] {name} find_stars DONE ({len(solver.stars)})", flush=True)
    except AssertionError as e:
        print(f"[WORKER] {name} find_stars FAIL", flush=True)
        io_data.write_header_failed(solver.path)
        return f"FAIL: {name} - {str(e)}"

    print(f"[WORKER] {name} solve_wcs START", flush=True)
    w = solver.solve_wcs(_worker_star_table)
    print(f"[WORKER] {name} solve_wcs RETURNED", flush=True)

    if w is None:
        print(f"[WORKER] {name} solve_wcs NONE", flush=True)
        return f"FAIL: {name} - Couldn't solve"

    print(f"[WORKER] {name} extract_star_data START", flush=True)
    star_data = solver.extract_star_data(w)
    print(f"[WORKER] {name} extract_star_data DONE", flush=True)

    io_data.write_solved_frame(solver.path, star_data, w)
    print(f"[WORKER] DONE {name}", flush=True)

    return None
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
    filter_kwargs = _get_filter_kwargs(data, filter_kwargs)
    
    # 2. Global Star Query
    star_table = _query_global_ucac4_table(data)

    # force_solve for each filter, defaulting to False if not specified
    filters = data.unique("filter")
    force_dict = _build_force_dict(filters, force_solve)
                
    # Create task list for all filters
    tasks = _build_tasks(data, filter_kwargs, force_dict)

    _run_pool(tasks, star_table)
        

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
    

def _get_filter_kwargs(data, filter_kwargs):
    # get find_stars kwargs for each filter
    if filter_kwargs is not None:
        return filter_kwargs
    path_str = str(data[0].path)
    name = path_str.split(os.sep)[-3]
    return helper.get_dataset(name, "find_stars")


def _query_global_ucac4_table(data):
    # 2. Global Star Query
    print("Fetching Global UCAC4 reference catalog.")
    temp_solver = PlateSolver(data[0])
    for j in range(10):  # randomly throws errors
        try:  # if larger than 2r, star wouldnt be in FOV
            return temp_solver.query_vizier(2 * data[0].radius)
        except Exception as e:
            if j == 9:
                raise e
    # unreachable, but explicit
    raise RuntimeError("UCAC4 query failed unexpectedly")


def _build_force_dict(filters, force_solve):
    # force_solve for each filter, defaulting to False if not specified
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
    return force_dict


def _build_tasks(data, filter_kwargs, force_dict):
    # Create task list for all filters
    all_tasks = []
    for frame in data:
        filt = frame.get("filter")
        # Match the specific kwargs for this frame's filter
        current_kwargs = filter_kwargs.get(filt, {})
        # Check force_solve logic for this specific filter
        should_force = force_dict.get(filt, False)
        all_tasks.append((frame, current_kwargs, should_force))
    return all_tasks


def _run_pool(tasks, star_table):
    # Launch 1 pool for entire dataset
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Launching pool for {len(tasks)} frames across {num_workers} cores.")

    futures = []
    names = []

    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker, initargs=(star_table,)) as executor:
        # We pass the pre-matched kwargs directly into the submit call

        for (frame, current_kwargs, should_force) in tasks:
            futures.append(executor.submit(solve_single_frame_task, frame, current_kwargs, should_force))
            names.append(str(frame.path).split(os.sep)[-1])

        total = len(futures)
        for i, (future, name) in enumerate(zip(futures, names), start=1):
            print(f"[{i}/{total}] waiting: {name}")
            output = future.result()  # explicit blocking: preserves current behavior
            if output is not None:
                print(output)
