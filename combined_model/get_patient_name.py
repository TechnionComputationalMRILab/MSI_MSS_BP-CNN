#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:42:03 2021

@author: hadar.hezi@bm.technion.ac.il
"""
import re
import numpy as np


def get_patient_name(file):
    # Each patch file has a path of the following: {path to data folder}blk-{random string}-TCGA-{patient_code}-01Z-00-DX1.png
    dash_num =3
    # find blk in the path
    start_ind = file.find("blk-")
    if start_ind > -1:      
        file = file[start_ind:]
    start_ind = file.find("-TCGA-")
    # the name is TCGA-{patient_code}
    name = file[start_ind+1:]
    name = name[:12]
    return name