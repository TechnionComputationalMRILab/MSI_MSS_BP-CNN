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
import make_dist_table
import pickle


def get_patient_precision_recall(hp,feature):
    folds_num=5
    mode = 'test'
    f_score_list = []
    kappa_list=[]
    root_dir = hp['root_dir']
    res_file = hp['test_res_file']
    final_cm = np.zeros([2, 2])
    stack_cm=[]
    print(final_cm)
    for i in range(folds_num):
        # load patients MSI probabilities and labels
        with (open(f"roc_out_{i}.p", "rb")) as openfile:
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
        # using the threshold for classifying to the positive label
        probs = np.array(probs)
        labels = np.array(labels)
        preds = np.zeros(probs.shape)
        pos_ind = np.nonzero(probs>thresholds[ix])[0]
        neg_ind = np.nonzero(probs<=thresholds[ix])[0]
        preds[pos_ind]=1
        preds[neg_ind]=0 
        kappa = cohen_kappa_score(labels, preds, weights=None, sample_weight=None)  
        kappa_list.append(kappa)        
        cm = confusion_matrix(labels, preds)
        stack_cm.append(cm) 
        final_cm+=cm
    # construct the average confusion matrix
    stack_cm = np.stack(stack_cm)
    # standard deviation of patients classification
    std_cm = np.std(stack_cm,0)
    classes = ['MSS','MSI']
    # average result of patients classification
    average_cm = np.divide(final_cm,folds_num)
    plt.figure()
    print(plt.rcParams.get('figure.figsize'))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    # plotting matrix with displayed std's
    thresh = average_cm.max() / 2.
    for i in range(average_cm.shape[0]):
        for j in range(average_cm.shape[1]):
            plt.text(j, i, '{0:.2f}'.format(average_cm[i, j]) + '\n$\pm$' + '{0:.2f}'.format(std_cm[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center", fontsize=10,
                     color="white" if average_cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"confusion matrix {feature} model")
    plt.savefig(f"cm_{feature}.png",dpi=500)

    mean_fscore = np.mean(f_score_list)
    mean_kappa = np.mean(kappa_list)
    print("mean f score: ", mean_fscore, "average kappa:" , mean_kappa)
    print("std of f score: " , np.std(f_score_list),"std of kappa score: " , np.std(kappa_list))

  
hp = hyperparams()
get_patient_precision_recall(hp,"baseline")