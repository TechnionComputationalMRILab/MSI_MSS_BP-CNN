#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:30:10 2021

@author: hadar.hezi@bm.technion.ac.il
"""
import pandas as pd
from hyperparams import  hyperparams
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
from numpy import argmax
from sklearn.metrics import auc,classification_report,precision_recall_curve,cohen_kappa_score, confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay,average_precision_score
from scipy import stats
import matplotlib.patches as mpatches
import seaborn
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
    #mode = 'test'
    f_score_list = []
    root_dir = hp['root_dir']
    #res_file = hp['test_res_file']
    final_cm = np.zeros([2, 2])
    stack_cm=[]
    aps = []
    for i in range(folds_num):

        with (open(f"{root_dir}roc_out_{i}.p", "rb")) as openfile:
            data = pickle.load(openfile)
        labels = data['labels']
        probs = data['probs']

        precision, recall, thresholds = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        auc_pr = auc(recall,precision)
        print("auc pr: ", auc_pr)
        aps.append(ap)
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
    plt.figure()
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
                     verticalalignment="center", fontsize=14,
                     color="white" if average_cm[i, j] > thresh else "black")

    #plt.tight_layout()
    #disp = ConfusionMatrixDisplay(confusion_matrix=average_cm,display_labels=['MSS','MSI'])
    #disp.plot(cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.rcParams.update({'font.size':14})
    plt.rcParams.update({'xtick.labelsize':14}) 
    plt.rcParams.update({'ytick.labelsize':14})
    plt.title(f"{feature} model")
 
    plt.savefig(f"{hp['root_dir']}cm_{feature}.eps",dpi=500)
    plt.savefig(f"{hp['root_dir']}cm_{feature}.png",dpi=500)
    plt.close()
    mean_fscore = np.mean(f_score_list)
    print("mean f score: ", mean_fscore)

    print("std of f score: " , np.std(f_score_list))
    with (open(f"{root_dir}f_scores.p", "rb")) as openfile:
        data = pickle.load(openfile)
        f_base = data['f']
        data['f_BP'] = f_score_list
    pickle.dump( data, open( f"{root_dir}f_scores.p", "wb" ) )
    # paried t-test f-scores
    print(stats.ttest_rel(f_base, f_score_list))
    print("mean base f score: ", np.mean(f_base), "base f std: ", np.std(f_base))
    # ap p-value
    base_dir = "/project/MSI_MSS_project/base_cimp/"
    with (open(f"{base_dir}ap.p", "rb")) as openfile:
        data = pickle.load(openfile)
        base_aps = data['ap']
    # paired t-test of the AUC results
#def paired_t_test(samples_a,samples_b):
    print(stats.wilcoxon(aps, base_aps,alternative='greater'))
    # CI for AP
    sorted_ap = np.array(aps)
    sorted_ap.sort()
    confidence_lower = sorted_ap[int(0.05 * len(sorted_ap))]
    confidence_upper = sorted_ap[int(0.95 * len(sorted_ap))]
    mean_ap = np.mean(aps)
    auc_ = [confidence_lower,mean_ap,confidence_upper]
    print("AP CI: " , auc_)
    print("mean precision score: ", mean_ap, "base f std: ", np.std(aps))

    #print(np.std(samples_a))
    #print(np.std(samples_b))
    #return "{:.2f}".format(np.std(samples_a)),  "{:.2f}".format(np.std(samples_b))
def get_mean_pr(root):    
    total_ap=[]
    total_labels = []
    total_probs = []
    
    #plt.figure()
    #ax = plt.axes()
    for j in range(5):
        with (open(f"{root}roc_out_{j}.p", "rb")) as openfile:
            data = pickle.load(openfile)
        labels = data['labels']
        probs = data['probs']
        total_labels.append(labels)
        total_probs.append(probs)
        ap = average_precision_score(labels, probs)
        total_ap.append(ap)
    # mean_ap = np.mean(total_ap)
    sorted_ap = np.array(total_ap)
    sorted_ap.sort()
    confidence_lower = sorted_ap[int(0.05 * len(sorted_ap))]
    # # nonzero returns tuple
    index = np.nonzero(total_ap==confidence_lower)
    index_lo = index[0]
    total_probs = np.array(total_probs)
    probs_lo = total_probs[index_lo][0]
    print(index_lo)
    print("len probs lo: ", probs_lo)
    #precision_lo, recall, _ = precision_recall_curve(labels, probs_lo) 
    #ax.step(recall, precision_lo, color='b',alpha=0.3,lw=1)
    confidence_upper = sorted_ap[int(0.95 * len(sorted_ap))]
    index= np.nonzero(total_ap==confidence_upper)
    index_hi = index[0]
    probs_hi = total_probs[index_hi][0]
    #precision_hi, recall, _ = precision_recall_curve(labels, probs_hi) 
    #ax.step(recall, precision_hi, color='b',alpha=0.3,lw=1)
    total_labels = np.concatenate(total_labels)
    total_probs = np.concatenate(total_probs)   
    #precision, recall, _ = precision_recall_curve(total_labels, total_probs)                                     


    #plt.figure()
    #ax = plt.axes()
    #ax.step(recall, precision, color='b',lw=2,alpha=0.8)
    #ax.grid(True)
    #ax.fill_between(recall,precision_hi, precision, alpha=0.2, color='b')

    return total_labels,total_probs, probs_hi,probs_lo, labels,total_ap

    
# plots the ROC of baseline vs BP-CNN results    
def plot_base_sub(hp,feature):

    root1 = 'C:/Users/hadar/Downloads/biomedical_eng/winter_2022/research/base_cimp/'
    root2 = f"{hp['root_dir']}"
    total_labels1,total_probs1, probs_hi1,probs_lo1, labels1,aps1 = get_mean_pr(root1)
    precision1, recall1, _ = precision_recall_curve(total_labels1, total_probs1) 
    precision_hi1, recall_hi1, _ = precision_recall_curve(labels1, probs_hi1)
    precision_lo1, recall_lo1, _ = precision_recall_curve(labels1, probs_lo1)   
    total_labels2,total_probs2, probs_hi2,probs_lo2, labels2,aps2 = get_mean_pr(root2)
    precision2, recall2, _ = precision_recall_curve(total_labels2, total_probs2)  
    precision_hi2, recall_hi2, _ = precision_recall_curve(labels2, probs_hi2)
    precision_lo2, recall_lo2, _ = precision_recall_curve(labels2, probs_lo2)  

    std1 = "{:.2f}".format(np.std(aps1))
    std2 = "{:.2f}".format(np.std(aps2))
    plt.figure()
    plt.rcParams.update({'font.size':14})
    ax = plt.axes()
    ax.step(recall_hi1, precision_hi1, color='g',alpha=0.3,lw=1)
    ax.step(recall_lo1, precision_lo1, color='g',alpha=0.3,lw=1)
    ax.step(recall1, precision1, color='g',lw=2,alpha=0.8)
    ax.step(recall_hi2, precision_hi2, color='b',alpha=0.3,lw=1)
    ax.step(recall_lo2, precision_lo2, color='b',alpha=0.3,lw=1)
    ax.step(recall2, precision2, color='b',lw=2,alpha=0.8)
    #data1 = {"fpr": recall1, "tpr_lo":tpr1[0], "tpr_mean": tpr1[1], "tpr_hi": tpr1[2]}
    #df1 = pd.DataFrame(data1)
    #data2 = {"fpr": fpr2[0], "tpr_lo":tpr2[0], "tpr_mean": tpr2[1], "tpr_hi": tpr2[2]}
    #df2 = pd.DataFrame(data2)
    #ax1 = seaborn.lineplot(x=df1.loc[:,'fpr'].values,y=df1.loc[:,'tpr_mean'].values,color='green')
    #ax1.fill_between(fpr1[0], tpr1[0],tpr1[1],color='green', alpha=.1)
    #ax1.fill_between(fpr1[0], tpr1[1],tpr1[2],color='green', alpha=.1)
    #ax2 = seaborn.lineplot(x=df2.loc[:,'fpr'].values,y=df2.loc[:,'tpr_mean'].values,color='blue')
    #ax2.fill_between(fpr2[0], tpr2[0],tpr2[1],color='blue', alpha=.1)
    #ax2.fill_between(fpr2[0], tpr2[1],tpr2[2],color='blue', alpha=.1)
    #ax2 = seaborn.lineplot(data=[fpr2[1],tpr2[1]],color='blue')
    ap1 = np.mean(aps1)
    ap2 = np.mean(aps2)
    ap1 = "{:.2f}".format(ap1)
    ap2 = "{:.2f}".format(ap2)
    # add legend
    top_bar = mpatches.Patch(color='green', label=f'Baseline (area='+ap1+r'$\pm$'+std1+')')
    bottom_bar = mpatches.Patch(color='blue', label='$BP-CNN_{CIMP}$ (area='+ap2+r'$\pm$'+std2+')')
    plt.legend(handles=[top_bar, bottom_bar],fontsize=14)

    #ax.set_xlabel('Precision')
    #ax.set_ylabel('Recall')
    # plt.rcParams.update({'font.size':14})
    # ax.legend(loc="lower right")
    # plt.rcParams.update({'font.size':18})
    ax.set_title(f"PR Baseline vs {feature}",fontsize=20)

    ax.grid(True)
    #roc_boxplot(aucs1,aucs2,feature)  

    data_aucs = {"aucs_base": aps1, "aucs_sub": aps2}
    pickle.dump( data_aucs, open( f"{hp['root_dir']}aucs.p", "wb" ) )
    plt.xlabel('Recall',fontsize=14)
    plt.ylabel('Precision',fontsize=14)
    #plt.title('PR curve CIMP')
    plt.savefig("{}ap_{}_vs.png".format(hp['root_dir'],'test'),dpi=500,bbox_inches="tight") 
    plt.close()


    
    
hp = hyperparams()
#get_patient_precision_recall(hp,"CIMP")
#get_mean_pr(f"{hp['root_dir']}")
plot_base_sub(hp,'CIMP')