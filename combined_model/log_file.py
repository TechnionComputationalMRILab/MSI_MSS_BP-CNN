#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:18:46 2020

@author: hadar.hezi@bm.technion.ac.il
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
import time
import os
import sys
from typing import NamedTuple
import matplotlib.pyplot as plt


# from prepare_data import Prepare
from compute_roc import * 
        


def save_results(path,summary):
    preds = summary.preds
    labels = summary.labels
    sub_labels = summary.sub_labels
    pos_probs = summary.pos_label_probs
    paths = summary.paths
    data_dict = {'preds':preds,'labels':labels,'sub_labels':sub_labels,'pos_probs':pos_probs, 'paths':paths}
    df = pd.DataFrame(data_dict)
    df.to_csv(path)
    
    
def read_results(path):
   if os.path.isfile(path):
       df = pd.read_csv(path) 
       preds = np.array(df['preds'])
       labels = np.array(df['labels'])
       sub_labels =  np.array(df['sub_labels'])
       pos_probs =  np.array(df['pos_probs'])
       paths = np.asarray(list(df['paths']))
   return summary(preds,labels,sub_labels,pos_probs,paths)   

def get_probs_histogram(summary,valid=False):
    labels_num = max(summary.sub_labels)
    for j in range(labels_num+1):       
        # all inds of sub-label 0 
        ind = torch.nonzero(summary.sub_labels==j)
        # choose the first patch
        path = summary.paths[ind[0]]
        if valid==True:
             path_valid = summary.paths[ind[-1]]
             # extract patient name
             patient_name = get_patient_name(path_valid)
             #indidces to all patches of this patient
             indices = [i for i, x in enumerate(summary.paths) if patient_name in x]
             pos_curr_patient = summary.pos_label_probs[indices]
             hist, bin_edges = np.histogram(pos_curr_patient,bins=10)
             plt.figure()
             plt.title(f'validation patient:{patient_name} label: {j}')
             plt.hist(bin_edges[:-1], bin_edges, weights=hist)
        # extract patient name
        patient_name = get_patient_name(path)
        #indidces to all patches of this patient
        indices = [i for i, x in enumerate(summary.paths) if patient_name in x]
        pos_curr_patient = summary.pos_label_probs[indices]
        hist, bin_edges = np.histogram(pos_curr_patient,bins=10)
        plt.figure()
        plt.title(f'patient:{patient_name} label: {j}')
        plt.hist(bin_edges[:-1], bin_edges, weights=hist)
    
def patient_histogram(train_summary, test_summary):
    # will print hostograms of the probabilities per patient
    # will choose 9 patients total from each set train,valid,test
    # one from each class MSI, MSS, MSI low
    # train_summary = read_results(train_res_file)
    # test_summary(test_res_file)
    get_probs_histogram(train_summary,True)
    get_probs_histogram(test_summary)
    
    
    
class summary(NamedTuple):
#     """
    # Saves all loaded labels,paths and calculated probs
    # """

    preds: list
    labels: list
    sub_labels: list
    pos_label_probs: list
    paths: list
        