#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 08:38:28 2020

@author: hadar.hezi@bm.technion.ac.il
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score




def compute_roc(labels, probs):
    # cast labels and probs to tensor
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
        
    if torch.is_tensor(probs):
       probs = probs.cpu().numpy()
   
    # compute AUC
    MSI_tp_auc = roc_auc_score(labels, probs)
    print('auc:',MSI_tp_auc)
    # calculate roc curves
    lr_fpr, lr_tpr, thresh = roc_curve(labels, probs)
      
    return lr_fpr, lr_tpr, MSI_tp_auc 
 
    
def plot_roc_with_ci(fpr,tpr,auc_,hp,mode):
    fig, ax = plt.subplots() 
    ax.plot(fpr[1], tpr[1], color='green',
                lw=2, label='mean ROC curve (area = %0.2f)' % auc_[1])
    ax.fill_between(fpr[2], tpr[2],tpr[1], color='g', alpha=.1)
    ax.fill_between(fpr[0], tpr[1],tpr[0], color='g', alpha=.1)
    ax.plot([0, 1], [0, 1], color='g', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize']='large'
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    ax.set_title("{}".format(hp['experiment']))
    ax.legend(loc="lower right")
    ax.grid(True)
    # plt.show()
   
    fig.savefig("{}roc_{}.png".format(hp['root_dir'],mode))
    # fig.close()
    
        
    
def roc_per_patient(summary,p,hp,mode):

    if mode == 'valid':
        patient_names = p.valid_names
    elif mode =='test':
        patient_names = p.test_patients

    true_MSI_per_patient = []
    pred_MSI_per_patient = []


    for j,name in enumerate(patient_names):

       # files of this patient
        indices = [i for i, x in enumerate(summary.paths) if name in x]
        if len(indices) == 0:
            continue
    
        # MSI labels for each patient
        true_MSI = (summary.labels[indices[0]] == 0)
        # equal to 1 for MSI
        true_MSI_score = int(true_MSI)
        pred_patient = summary.preds[indices]
        #pred_patient = 1-pred_patient
        pred_MSI_score = np.mean(pred_patient)
    
        pred_MSI_per_patient.append(pred_MSI_score.item())
        true_MSI_per_patient.append(true_MSI_score)
    return true_MSI_per_patient,pred_MSI_per_patient
    
