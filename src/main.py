# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:11 2026

@author: chris
"""

from matplotlib.pyplot import close
import io_data
import reduce_data


#example usage
# sci_reduced = io_data.read_folder(directory+"/Reduced")
# sci_reduced.unique('exposure')
# sci_reduced.unique('filter')
# sci_reduced.filter(filter='r')
# img = sci_reduced[0].load()
# plot.imshow(img)

def data_reduction(indir, rename_HAT=False, **reduce_all_kwargs):
    scis, calibration = io_data.read(indir)
    mbias, mdark_rate, mflat = reduce_data.create_master_frames(calibration)
    if rename_HAT is True:
        name_HAT = "HAT-P-32"
    else: 
        name_HAT = None
    reduce_data.reduce_all(scis, mbias, mdark_rate, mflat, directory, 
                           new_object_name=name_HAT, **reduce_all_kwargs)
    return


close("all")
repo_root = io_data.get_repo_root()
directory = repo_root / "data" / "20251104_lab"
print("Using data directory:", directory)
data_reduction(directory, plotting=True)

sci_reduced = io_data.read_folder(directory+"/Reduced")

