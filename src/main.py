# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:11 2026

@author: chris
"""

from matplotlib.pyplot import close
import io_data
import reduce_data
from plate_solving_parallel import plate_solve_all


#example usage
# data = io_data.read_folder(directory / "Solved")  # list of all dataframes
# data.unique('exposure')  # list of different exposure times
# data.unique('filter')  # list of different filters
# r = data.filter(filter='r')  # list of all dataframes with given filter
# 
# img = r[0].load()  # load an image into memory
# plot.imshow(img)  # plot the image
# stars = r[0].load_stars()  # load positions (xpix, ypix) of


def run_pipeline(repo_root, labname, force=False):
    directory = repo_root / "data" / labname
    print("Using data directory:", directory)
    # perform bias and dark current subtraction, flat fielding
    # saved in directory / "Reduced"
    reduce_data.data_reduction(directory, plotting=False, force_reduction=force)
    reduced = io_data.read_folder(directory / "Reduced")
    
    # takes ~40min if concurrent version is used vs ~10 min for parallel
    # find stars in image and get coordinates/orientation of field of view
    # Here we loose some images because they cant be plate solved. Usually because
    # there are too few stars (cloudy) or too many (overexposed)
    plate_solve_all(reduced, force_solve=force)
    

if __name__ == "__main__":
    close("all")
    labs = {"RR_Lyrae": "20251104_lab", "Transit": "20260114_lab"}
    repo_root = io_data.get_repo_root()
    run_pipeline(repo_root, labs["RR_Lyrae"])
    run_pipeline(repo_root, labs["Transit"])
    
    


