# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:11 2026

@author: chris
"""

from matplotlib.pyplot import close
from pipeline import run_pipeline
from helper import get_repo_root



#example usage
# data = io_data.read_folder(directory / "Solved")  # list of all dataframes
# data.unique('exposure')  # list of different exposure times
# data.unique('filter')  # list of different filters
# r = data.filter(filter='r')  # list of all dataframes with given filter
# 
# img = r[0].load()  # load an image into memory
# plot.imshow(img)  # plot the image
# stars = r[0].load_stars()  # load positions (xpix, ypix) of

if __name__ == "__main__":
    close("all")
    labs = {"RR_Lyrae": "20251104_lab", "Transit": "20260114_lab"}
    repo_root = get_repo_root()
    run_pipeline(repo_root, labs["RR_Lyrae"])
    run_pipeline(repo_root, labs["Transit"])
    
    


