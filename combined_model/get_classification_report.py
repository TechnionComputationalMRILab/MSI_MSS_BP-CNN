#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:30:10 2021

@author: hadar.hezi@bm.technion.ac.il
"""
import pandas as pd
from hyperparams import  hyperparams
import numpy as np
import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.metrics import classification_report,precision_recall_curve,cohen_kappa_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import log_file
import compute_roc
import prepare_data

import pickle

# per patch precision-recall
def get_precision_recall(hp):
    folds_num = hp['n_folds']

    for i in range(folds_num):
        path = f"{hp['root_dir']}{hp['valid_res_file']}_{i}.csv"
        df = pd.read_csv(path) 
        preds = np.array(df['preds'])
        labels = np.array(df['sub_labels'])
        # union GS sub classes
        labels[labels==2]=1
        preds[preds==2]=1
        print(classification_report(labels, preds))
        
def get_patient_precision_recall(hp,feature):
    # folds_num = hp['n_folds']
    folds_num=5
    mode = 'test'
    p = prepare_data.Prepare(hp)
    f_score_list = []
    root_dir = hp['root_dir']
    res_file = hp['test_res_file']
    final_cm = np.zeros([2, 2])
    stack_cm=[]
    for i in range(folds_num):

        with (open(f"{root_dir}roc_out_{i}.p", "rb")) as openfile:
            data = pickle.load(openfile)
        labels = data['labels']
        probs = data['probs']

        precision, recall, thresholds = precision_recall_curve(labels, probs)
        # convert to f score         
        fscore = (2 * precision * recall) / (precision + recall)
        fscore= fscore[~np.isnan(fscore)]
        # locate the index of the largest f score
        ix = argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f,precision:=%.3f,recall=%.3f' % (thresholds[ix], fscore[ix],precision[ix],recall[ix]))
        f_score_list.append(fscore[ix])
        # threshold for classifying to the positive label - GS
        probs = np.array(probs)
        labels = np.array(labels)
        preds = np.zeros(probs.shape)
        gs_ind = np.nonzero(probs>thresholds[ix])[0]
        gs_ind=gs_ind.astype('int')
        cin_ind = np.nonzero(probs<=thresholds[ix])[0]
        preds[gs_ind]=1
        preds[cin_ind]=0

        plt.rcParams.update({'font.size':14})

        kappa = cohen_kappa_score(preds, labels, weights=None, sample_weight=None)
        print(f"fold {i} kappa ={kappa} ")
        cm = confusion_matrix(labels,preds)
        stack_cm.append(cm) 
        final_cm+=cm
    stack_cm = np.stack(stack_cm)
    std_cm = np.std(stack_cm,0)
    classes = ['MSS','MSI']
    average_cm = np.divide(final_cm,folds_num)
    plt.figure(figsize=(6, 4))
    print(plt.rcParams.get('figure.figsize'))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = average_cm.max() / 2.

    for i in range(average_cm.shape[0]):
        for j in range(average_cm.shape[1]):
            plt.text(j, i, '{0:.2f}'.format(average_cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(std_cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center", fontsize=10,
                     color="white" if average_cm[i, j] > thresh else "black")

    #plt.tight_layout()
    # disp = ConfusionMatrixDisplay(confusion_matrix=average_cm,display_labels=['MSS','MSI'])
    # disp.plot(cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.rcParams.update({'font.size':14})
    # plt.rcParams.update({'xtick.labelsize':14}) 
    # plt.rcParams.update({'ytick.labelsize':14})
    plt.title(f"confusion matrix {feature} model")
 
    plt.savefig(f"{hp['root_dir']}cm_{feature}.eps",dpi=500)
    plt.savefig(f"{hp['root_dir']}cm_{feature}.png",dpi=500,bbox_inches = "tight")
    #plt.savefig(f"{hp['root_dir']}cm_{feature}.png",dpi=500)
    mean_fscore = np.mean(f_score_list)
    print("mean f score: ", mean_fscore,"std: ", np.std(f_score_list))

    
    
hp = hyperparams()
get_patient_precision_recall(hp,"Combined")