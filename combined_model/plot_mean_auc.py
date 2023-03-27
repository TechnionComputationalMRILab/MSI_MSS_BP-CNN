from log_file import *
from hyperparams import hyperparams
from prepare_data import *
from compute_roc import *
import pickle
import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

def plot_mean_auc(p,mode,hp):
    total_tpr =[]
    total_fpr=[]
    total_auc=[]

    for j in range(hp['n_folds']):
    # loading patients labels and probabilities
        try:
            with (open(f"{hp['root_dir']}roc_out_{j}.p", "rb")) as openfile:
            # with (open(f"{root}roc_out_pca_{j}.p", "rb")) as openfile:
                data = pickle.load(openfile)
                labels = np.array(data['labels'])
                probs = np.array(data['probs'])
        except:
            summary = read_results(f"{hp['root_dir']}/{hp['test_res_file']}_{j}.csv")
            labels,probs = roc_per_patient(summary,p,hp,mode)
            
            #pickle data
            data = {"labels": labels, "probs": probs}
            pickle.dump( data, open( f"{hp['root_dir']}roc_out_{j}.p", "wb" ) )
        # ROC
        lr_fpr, lr_tpr, MSI_tp_auc = compute_roc(labels, probs)                                                
        print("iter ", j, "AUC: ", MSI_tp_auc)
        # interpoating the fpr axis
        mean_fpr = np.linspace(0, 1, 200)
        interp_tpr = np.interp(mean_fpr, lr_fpr, lr_tpr)
        interp_tpr[0] = 0.0
        total_tpr.append(interp_tpr)
        total_fpr.append(lr_fpr)
        total_auc.append(MSI_tp_auc)
    
    mean_tpr = np.mean(total_tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(total_auc)
    std_tpr = np.std(total_tpr, axis=0)

    # 95 CI of AUC results
    sorted_auc = np.array(total_auc)
    sorted_auc.sort()
    confidence_lower = sorted_auc[int(0.05 * len(sorted_auc))]
    # nonzero returns tuple
    index = np.nonzero(total_auc==confidence_lower)
    index_lo = index[0]
    
    # take the first occurenrce from indices list
    tpr_lo_ci = total_tpr[index_lo[0]]
    fpr_lo_ci = total_fpr[index_lo[0]]
    
    confidence_upper = sorted_auc[int(0.95 * len(sorted_auc))]
    index= np.nonzero(total_auc==confidence_upper)
    index_hi = index[0]
    tpr_hi_ci = total_tpr[index_hi[0]]
    fpr_hi_ci = total_fpr[index_hi[0]]

    fpr = [mean_fpr,mean_fpr,mean_fpr]
    tpr = [tpr_lo_ci,mean_tpr,tpr_hi_ci]
    auc_ = [confidence_lower,mean_auc,confidence_upper]
    print(auc_)
    plot_roc_with_ci(fpr,tpr,auc_,hp,mode)
    return mean_auc

    
def get_mean_roc(root):    
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    for j in range(5):
        with (open(f"{root}roc_out_{j}.p", "rb")) as openfile:
            data = pickle.load(openfile)
        labels = data['labels']
        probs = data['probs']
        lr_fpr, lr_tpr, MSI_tp_auc = compute_roc(labels, probs)
                                                
        print("iter ", j, "AUC: ", MSI_tp_auc)
    
        mean_fpr = np.linspace(0, 1, 200)
        interp_tpr = np.interp(mean_fpr, lr_fpr, lr_tpr)
        interp_tpr[0] = 0.0
        total_tpr.append(interp_tpr)
        total_fpr.append(lr_fpr)
        total_auc.append(MSI_tp_auc)    
    mean_tpr = np.mean(total_tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    return mean_fpr,mean_tpr,mean_auc,total_auc

# plots a boxplot of baseline vs BP-CNN AUC results    
def roc_boxplot(aucs_base,aucs_sub,feature):
    data = [aucs_base,aucs_sub]

    print("medians: ", np.median(aucs_base),np.median(aucs_sub))
    labels = ["baseline",f"{feature}"]
    labelsize = 22
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = labelsize
    fig= plt.figure(figsize=(6, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    # rectangular box plot
    bplot = ax.boxplot(data,
                         vert=True, 
                         showmeans=True,# vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax.set_ylim([0.67, 0.9])
    ax.set_title(f"AUC results of baseline and {feature}",fontsize=24)
    ax.grid(True)
    # fill with colors
    colors = ['pink', 'lightblue']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.savefig(f"{hp['root_dir']}boxplot_roc.png",dpi=500, bbox_inches = "tight")
# paired t-test of the AUC results
def paired_t_test(samples_a,samples_b):
    print(stats.ttest_rel(samples_a, samples_b))
    print(np.std(samples_a))
    print(np.std(samples_b))

# plots the ROC of baseline vs BP-CNN results    
def plot_base_sub(hp,feature):
    mode = 'test'
    root1 = "/project/MSI_MSS_project/base_cimp/"
    root2 =  hp['root_dir']
    fpr1,tpr1,auc1,aucs1 = get_mean_roc(root1)
    fpr2,tpr2,auc2,aucs2 = get_mean_roc(root2)
    fig, ax = plt.subplots(figsize=(6, 4)) 
    plt.rcParams.update({'font.size':14})
    ax.plot(fpr1, tpr1, color='green',
                lw=1, label='base model(area = %0.2f)' % auc1)
    ax.plot(fpr2, tpr2, color='blue',
                lw=1, label=f'{feature} model (area = %0.2f)' % auc2)
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    plt.rcParams.update({'font.size':18})
    ax.set_title(f"ROC Base vs {feature}")
    plt.rcParams.update({'font.size':14})
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.savefig("{}roc_{}_vs.png".format(hp['root_dir'],mode),bbox_inches="tight") 
    roc_boxplot(aucs1,aucs2,feature)  
    paired_t_test(aucs1,aucs2)
    
hp = hyperparams() 
p = Prepare(hp)  
#plot_mean_auc(p,'test',hp)  
plot_base_sub(hp,"Combined") 