#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:51:18 2022

@author: hadar.hezi@bm.technion.ac.il
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score,recall_score
from sklearn.metrics import precision_recall_fscore_support
import torch.nn.functional as F
import random

from compute_roc import *
import log_file
from log_file import summary
from get_auc import *


def test_fusion(model1,model2,mlp,prepare,hp): 
    model1.eval()
    model2.eval()
    mlp.eval()
    num_correct = 0
    epoch_preds = []
    epoch_paths = []
    epoch_labels = [] 
    epoch_sub_labels=[]
    epoch_pos_probs=[]
    test_data_path =  f"{hp['root_dir']}/{hp['test_res_file']}_{hp['curr_fold']}.csv"
      
    with torch.no_grad():  
        # load data.
        for  inputs, labels,sub_labels,paths in prepare.dataloaders['test']:       
            paths = np.array(paths)
            inputs = inputs.to(prepare.device)
            labels = labels.to(prepare.device)
            sub_labels = sub_labels.to(prepare.device)
            # outputs from the trained BP-CNNs
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            # insert to the combined model
            mlp_in = torch.hstack((outputs1,outputs2))
            outputs = mlp(mlp_in)
            # convert outputs to probabilities
            prob_y= F.softmax(outputs,1)
            # predicted labels
            _, preds = torch.max(prob_y, 1)
            # The MSI probability
            prob_y_msi = prob_y[:,0]
            
            epoch_preds.append(preds)
            epoch_labels.append(labels)
            epoch_sub_labels.append(sub_labels)
            epoch_paths.append(paths)
            epoch_pos_probs.append(prob_y_msi)
            # calculate accuracy using labels or sub_labels
            num_correct += torch.sum(torch.eq(labels,preds)).item()
        
       
        epoch_preds = np.array(torch.cat(epoch_preds).cpu())
        epoch_labels= np.array(torch.cat(epoch_labels).cpu())
        epoch_sub_labels= np.array(torch.cat(epoch_sub_labels).cpu())
        epoch_pos_probs = np.array(torch.cat(epoch_pos_probs).cpu())
        epoch_paths = np.hstack(epoch_paths)
        
        accuracy = (num_correct / len(epoch_preds)) *100
        print("test acc: ", accuracy)
        #recall take msi labels
        y_true_tmp = np.array(epoch_labels)
        y_true = np.array(epoch_labels)
        y_true[y_true_tmp==1]=0
        y_true[y_true_tmp==0]=1

        y_pred_tmp = np.array(epoch_preds)
        y_pred = np.array(epoch_preds)
        y_pred[y_pred_tmp==1]=0
        y_pred[y_pred_tmp==0]=1
               
        rec = recall_score(y_true, y_pred, average=None)
        print("mss rec: ", rec[0],"msi rec: ",rec[1])
        all_summary = summary(epoch_preds,epoch_labels,epoch_sub_labels,epoch_pos_probs,epoch_paths)
        log_file.save_results(test_data_path, all_summary)
        # area under the curve patient-level
        test_auc = get_auc(all_summary,prepare,mode='test',hp=hp)

    return test_auc
