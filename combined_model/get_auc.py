#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 12:54:57 2022

@author: hadar.hezi@bm.technion.ac.il
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
import os
import sys

from prepare_data import Prepare
from log_file import *
from compute_roc import *



def get_auc(summary,p,hp,mode):   
    labels,probs = roc_per_patient(summary,p,hp,mode)
    fpr, tpr, MSI_auc = compute_roc(labels, probs)
    plot_roc(fpr,tpr,MSI_auc,hp,mode)
    # precision, recall, thresholds = precision_recall_curve(labels, probs)

    return MSI_auc