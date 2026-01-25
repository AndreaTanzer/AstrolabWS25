# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:11 2026

@author: chris
"""

import os
import numpy as np
from matplotlib.pyplot import close

import io_data
import reduce_data
import plot
from helper import functimer




close("all")
directory="../data/20251104_lab"
directory="../data/20260114_lab"
scis, calibration = io_data.read(directory)
mbias, mdark_rate, mflat = reduce_data.create_master_frames(calibration)
reduce_data.reduce_all(scis, mbias, mdark_rate, mflat, directory, plotting=True)
# Do this for 2026 lab to change object name. Remove force_reduction once done
# reduce_data.reduce_all(scis, mbias, mdark_rate, mflat, directory, 
#                        new_object_name="HAT-P-32", force_reduction=True)

sci_reduced = io_data.read_folder(directory+"/Reduced")