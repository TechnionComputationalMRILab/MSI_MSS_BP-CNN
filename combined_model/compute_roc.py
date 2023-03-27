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
 
def plot_roc(lr_fpr,lr_tpr, MSI_tp_auc,hp,mode):
    fig, ax = plt.subplots() 
    lw = 2   
    ax.plot(lr_fpr, lr_tpr, color='darkorange',
               lw=lw, label='ROC curve (area = %0.2f)' % MSI_tp_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize']='large'
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title("{}".format(hp['experiment']))
    ax.legend(loc="lower right")
    ax.grid(True)  
    fig.savefig("{}roc_{}.png".format(hp['root_dir'],mode))
    
def plot_roc_with_matlab(lr_fpr,lr_tpr,lr_fpr_matlab,lr_tpr_matlab,MSI_tp_auc,
                         MSI_tp_auc_matlab):
    plt.figure()
    lw = 2   
    plt.plot(lr_fpr, lr_tpr, color='darkorange',
               lw=lw, label='ROC curve (area = %0.2f)' % MSI_tp_auc)
    plt.plot(lr_fpr_matlab, lr_tpr_matlab, color='yellow',
               lw=lw, label='ROC curve matlab (area = %0.2f)' % MSI_tp_auc_matlab)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    # plt.savefig('roc_auc.png')
    
def plot_roc_with_ci(fpr,tpr,auc_,hp,mode):
    fig, ax = plt.subplots() 
    # plt.rcParams.update({'font.size':12})
    # ax.plot(fpr[2], tpr[2], color='green',
                # lw=1, label='ROC curve high CI(area = %0.2f)' % auc_[2])
    # ax.plot(fpr[0], tpr[0], color='green',
                # lw=1, label='ROC curve low CI (area = %0.2f)' % auc_[0])
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
    

        
def count_patches_per_patient(patient_names,labels,paths):
    # will save for each ptaient the true labels it has 
    true_labels_per_patient = {}
    # adding 1 for the for loop
    labels_range = max(labels) + 1
    for name in patient_names:
        #take all relevant indices for this patient
        true_labels_per_patient[name] = {}
        indices = [i for i, x in enumerate(paths) if name in x]
        patient_paths = paths[indices]
        true_labels_per_patient[name]['paths']= patient_paths
        patient_labels = labels[indices]     
        for l in range(labels_range):
            # zeros vector is populated with ones in indices of the current label
            zeros = torch.zeros_like(patient_labels)
            zeros = torch.where(patient_labels==l,torch.tensor(1,device=labels.device),torch.tensor(0,device=labels.device))
            count_l = torch.sum(zeros)
            true_labels_per_patient[name][l] = count_l
    return true_labels_per_patient
        
    
def roc_per_patient(summary,p,hp,mode):

    if mode == 'valid':
        patient_names = p.valid_names
    elif mode =='test':
        patient_names = p.test_patients

    true_MSI_per_patient = []
    pred_MSI_per_patient = []

    # if mode=='train':
        # p.knn = KNeighborsClassifier(n_neighbors=170)
    # k - plot histograms of k patients
    # k = 10
    # plot_patients_hist(patient_names,k,paths,true_labels_per_patient,mode)
    #calaulate prediction rate and true label rate for each patient

    for j,name in enumerate(patient_names):

       # files of this patient
        indices = [i for i, x in enumerate(summary.paths) if name in x]
        # the test list might contain patient names who's label is -1
        if len(indices) == 0:
            continue
    
        # MSI labels for each patient
        true_MSI = (summary.labels[indices[0]] == 0)
        # true_MSS = (summary.labels[indices[0]] == 1)
        # equal to 1 for MSI
        true_MSI_score = int(true_MSI)
        # save the results per patient
        # tiles_num.append(true_MSI+true_MSS)
        #take MSI probabilities in same indices of the files
        # patient_patches_num = len(indices)
        prob_patient = summary.pos_label_probs[indices]
        # prob_patient = prob_patient.sort()
        # max_probs = prob_patient[-5:]
        pred_MSI_score = np.mean(prob_patient)
    
        pred_MSI_per_patient.append(pred_MSI_score.item())
        # will be 1 for MSI
        # if torch.is_floating_point(true_MSI_score):
        #     print(f"{true_MSI_score} is continous!")
        true_MSI_per_patient.append(true_MSI_score)
    #     # hist_list = hist_list[:,:2]
    #     labels_list = np.asarray(labels_list)
    #     # labels_list = labels_list.reshape(-1,1)
    #     print("hist_list",hist_list.shape)
    #     print("labels_list",labels_list.shape)
    #     p.knn.fit(hist_list, labels_list)
    #     return
    # else:
    #log_file.save_prediction(patient_names,pred_MSI_per_patient,true_MSI_per_patient)
    return true_MSI_per_patient,pred_MSI_per_patient
    
