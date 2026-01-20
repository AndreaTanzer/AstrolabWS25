# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:14:45 2026

@author: chris
"""

import glob
from astropy.io import fits
import os


def read_folder(path):
    print(f"Reading now: {path}")

    hdrs = []
    datas = []
    files = glob.glob(os.path.join(path, "*.[Ff][Ii][Tt]*"))

    for fname in files:
        with fits.open(fname) as hdul:
            hdrs.append(hdul[0].header)
            datas.append(hdul[0].data)
    print(f"Finished reading {len(files)} files from {path}")
    return hdrs, datas

def read(directory="data/20251104_lab/"):
    directory = os.path.abspath(os.path.expanduser(directory))
    hdrs_bias, bias = read_folder(os.path.join(directory, "Bias"))
    hdrs_flat, flats = read_folder(os.path.join(directory, "Flats"))
    hdrs_sci, scis = read_folder(os.path.join(directory, "Science"))

    hdr = {"sci":hdrs_sci, "bias": hdrs_bias, "flat": hdrs_flat}
    data = {"sci":scis, "bias": bias, "flat": flats}
    return hdr, data



if __name__ == "__main__":
    # directory = "../20251104_lab/"
    # hdrs_bias, bias = read_folder(directory+"Bias/")
    # hdrs_flats, flats = read_folder(directory+"Flats/")
    # hdrs, datas = read_folder(directory+"Science/")
    hdrs, datas = read()
