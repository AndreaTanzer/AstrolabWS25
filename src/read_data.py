# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 17:14:45 2026

@author: chris
"""

import glob
from astropy.io import fits


def read_folder(path):
    hdrs = []
    datas = []
    for fname in glob.glob(path+"*.fit"):
        with fits.open(fname) as hdul:
            hdrs.append(hdul[0].header)
            datas.append(hdul[0].data)
    return hdrs, datas

def read(directory="../20251104_lab/"):
    print("Hi")
    hdrs_bias, bias = read_folder(directory+"Bias/")
    hdrs_flat, flats = read_folder(directory+"Flats/")
    hdrs_sci, scis = read_folder(directory+"Science/")
    hdr = {"sci":hdrs_sci, "bias": hdrs_bias, "flat": hdrs_flat}
    data = {"sci":scis, "bias": hdrs_bias, "flat": hdrs_flat}
    return hdr, data



if __name__ == "__main__":
    # directory = "../20251104_lab/"
    # hdrs_bias, bias = read_folder(directory+"Bias/")
    # hdrs_flats, flats = read_folder(directory+"Flats/")
    # hdrs, datas = read_folder(directory+"Science/")
    hdrs, datas = read()
