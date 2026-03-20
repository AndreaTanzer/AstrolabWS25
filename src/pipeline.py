from reduce_data import data_reduction
from plate_solving_parallel import plate_solve_all
from photometry import gen_light_curves
from io_data import read_folder
from helper import functimer
# from plate_solving import plate_solve_all

@functimer
def run_pipeline(repo_root, labname, force=False, verbose=False):
    directory = repo_root / "data" / labname
    print("Data directory:", directory)
    
    # perform bias and dark current subtraction, flat fielding
    # saved in directory / "Reduced"
    # takes ~1.5min
    data_reduction(directory, plotting=False, force_reduction=force)
    reduced = read_folder(directory / "Reduced")
    
    # takes ~40min if concurrent version is used vs ~10 min for parallel
    # find stars in image and get coordinates/orientation of field of view
    # Here we loose some images because they cant be plate solved. Usually because
    # there are too few stars (cloudy) or too many (overexposed)
    plate_solve_all(reduced, force_solve=force, verbose=verbose)
    
    # takes ~2min
    solved = read_folder(directory / "Solved")
    light_curve = gen_light_curves(solved, labname)
    return light_curve