# -*- coding: utf-8 -*-
import os
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import matplotlib.pyplot as plt
from astropy.utils.exceptions import AstropyWarning

import helper
import io_data
from plate_solving import PlateSolver

# Global variables for worker memory
_worker_star_table = None
_worker_alt_star_table = None

def init_worker(main_table, alt_table):
    """
    Sets the reference tables once per worker process to avoid pickling overhead.
    """
    global _worker_star_table, _worker_alt_star_table
    _worker_star_table = main_table
    _worker_alt_star_table = alt_table
    
    # Silence overhead chatter
    logging.getLogger('astroquery').setLevel(logging.ERROR)
    warnings.simplefilter("ignore", category=AstropyWarning)

def solve_single_frame_task(frame, find_kwargs, force_solve):
    """
    Unified task: Uses recursive thresholding and dual-catalog fallbacks.
    """
    global _worker_star_table, _worker_alt_star_table
    name = frame.path.name
    solver = PlateSolver(frame)

    # 1. Check if already done
    if (solver.is_plate_solved or solver.failed_plate_solving) and not force_solve:
        return None 

    try:
        # 2. Recursive Star Detection (Your Logic)
        solver.find_stars(**find_kwargs)
        n_stars = len(solver.stars) if solver.stars else 0
        
        # Create a copy to modify
        current_params = dict(thresh=float(find_kwargs.get("thresh", 200)), 
                              peaklo=float(find_kwargs.get("peaklo", 5)))
        
        # Adjust if too many ( > 30)
        for _ in range(3):
            if n_stars <= 30: break
            current_params["thresh"] *= 1.5
            current_params["peaklo"] *= 1.5
            solver.find_stars(**current_params)
            n_stars = len(solver.stars) if solver.stars else 0
            
        # Adjust if too few ( < 11)
        for _ in range(3):
            if n_stars >= 11: break
            current_params["thresh"] *= 0.7
            current_params["peaklo"] *= 0.7
            solver.find_stars(**current_params)
            n_stars = len(solver.stars) if solver.stars else 0

        if n_stars == 0:
            return f"FAIL: {name} (0 stars detected)"

        # 3. Solving with Fallback (Your Logic)
        w = solver.solve_wcs(_worker_star_table)
        status_suffix = ""
        
        if w is None and _worker_alt_star_table is not None:
            w = solver.solve_wcs(_worker_alt_star_table)
            status_suffix = " (using alt table)"

        if w is None:
            io_data.write_header_failed(solver.path)
            return f"FAIL: {name} ({n_stars} stars) - Solve failed"

        # 4. Final Extraction and IO
        star_data = solver.extract_star_data(w)
        io_data.write_solved_frame(solver.path, star_data, w)
        return f"DONE: {name} ({n_stars} stars){status_suffix}"

    except Exception as e:
        return f"ERR: {name} - {str(e)}"

@helper.functimer
def plate_solve_all(data: helper.ScienceFrameList, force_solve=False, filter_kwargs=None):
    """
    Orchestrator using Chris's structural breakdown but your dual-query logic.
    """
    # 1. Setup Settings
    filter_kwargs = _get_filter_kwargs(data, filter_kwargs)
    
    # 2. Global Star Query (Restored your Vizier + Gaia logic)
    main_table, alt_table = _query_reference_catalogs(data)
    
    # 3. Build tasks
    filters = data.unique("filter")
    force_dict = _build_force_dict(filters, force_solve)
    tasks = _build_tasks(data, filter_kwargs, force_dict)
    
    # 4. Run Pool
    _run_pool(tasks, main_table, alt_table)

# --- Chris's Helper Structure (Modified to support your fallback) ---

def _query_reference_catalogs(data):
    print("Fetching Global Reference catalogs (Gaia & VizieR)...")
    temp_solver = PlateSolver(data[0])
    for j in range(10):
        try:
            # We fetch both as you had in your HEAD
            viz = temp_solver.query_vizier(2.0 * data[0].radius)
            gaia = temp_solver.query_gaia(1.5 * data[0].radius)
            return gaia, viz
        except Exception as e:
            if j == 9: raise e
    return None, None

def _get_filter_kwargs(data, filter_kwargs):
    if filter_kwargs is not None: return filter_kwargs
    name = str(data[0].path).split(os.sep)[-3]
    return helper.get_dataset(name, "find_stars")

def _build_force_dict(filters, force_solve):
    if isinstance(force_solve, bool):
        return {filt: force_solve for filt in filters}
    return {filt: force_solve.get(filt, False) for filt in filters}

def _build_tasks(data, filter_kwargs, force_dict):
    all_tasks = []
    for frame in data:
        filt = frame.get("filter")
        all_tasks.append((frame, filter_kwargs.get(filt, {}), force_dict.get(filt, False)))
    return all_tasks

def _run_pool(tasks, main_table, alt_table):
    num_workers = max(1, os.cpu_count() - 1)
    print(f"Launching pool for {len(tasks)} frames across {num_workers} cores.")
    
    with ProcessPoolExecutor(max_workers=num_workers, 
                             initializer=init_worker, 
                             initargs=(main_table, alt_table)) as executor:
        
        futures = [executor.submit(solve_single_frame_task, *t) for t in tasks]
        
        for i, future in enumerate(futures, start=1):
            res = future.result()
            if res: print(f"[{i}/{len(tasks)}] {res}")

if __name__ == "__main__":
    plt.close("all")
    # Update this path to your current test directory
    directory = "../data/20251104_lab"
    data = io_data.read_folder(directory + "/Reduced")
    plate_solve_all(data, force_solve=True)