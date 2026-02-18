import reduce_data
from plate_solving_parallel import plate_solve_all
from io_data import read_folder


def run_pipeline(repo_root, labname, force=False):
    directory = repo_root / "data" / labname
    print("Using data directory:", directory)
    # perform bias and dark current subtraction, flat fielding
    # saved in directory / "Reduced"
    reduce_data.data_reduction(directory, plotting=False, force_reduction=force)
    reduced = read_folder(directory / "Reduced")
    
    # takes ~40min if concurrent version is used vs ~10 min for parallel
    # find stars in image and get coordinates/orientation of field of view
    # Here we loose some images because they cant be plate solved. Usually because
    # there are too few stars (cloudy) or too many (overexposed)
    plate_solve_all(reduced, force_solve=force)