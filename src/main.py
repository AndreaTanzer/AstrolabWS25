# -*- coding: utf-8 -*-
"""
Created on Sat Jan 24 15:46:11 2026

@author: chris
"""

from matplotlib.pyplot import close
from pipeline import run_pipeline
from helper import get_repo_root

if __name__ == "__main__":
    
    close("all")
    labs = {"RR_Lyrae": "20251104_lab", "Transit": "20260114_lab"}
    repo_root = get_repo_root()
    run_pipeline(repo_root, labs["RR_Lyrae"])
    run_pipeline(repo_root, labs["Transit"])
    
    


