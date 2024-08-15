#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:18:46 2020

@author: hadar.hezi@bm.technion.ac.il
"""
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
    
def save_prediction(patient_names,predictions,labels):
    # patient_names=patient_names.cpu().detach()
    # predictions=predictions.cpu().detach()
    
    data_dict = {'name':patient_names,'MSI_probability':predictions,'labels':labels}
    df = pd.DataFrame(data_dict)
    df.to_csv('MSI_probs_per_patient_test.csv')
    
def read_results(path):
   if os.path.isfile(path):
       df = pd.read_csv(path) 
       preds =np.array(df['preds'])
       labels = np.array(df['labels'])
       #sub_labels =  np.array(df['sub_labels'])
       pos_probs = np.array(df['pos_probs'])
       paths = np.asarray(list(df['paths']))
   return summary(preds,labels,pos_probs,paths)   

    
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
    #sub_labels: list
    pos_label_probs: list
    paths: list
        