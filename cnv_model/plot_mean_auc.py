from log_file import *
from hyperparams import hyperparams
# from visualize_model import visualize_model
#from prepare_data import *
from compute_roc import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from scipy import stats
from scipy import io
import pandas as pd
import seaborn


def plot_mean_auc(p,mode,hp):
    total_tpr =[]
    total_fpr=[]
    total_auc=[]
    for j in range(hp['n_folds']):
    # loading patients labels and probabilities
        try:
            with (open(f"{hp['root_dir']}roc_out_{j}.p", "rb")) as openfile:
                data = pickle.load(openfile)
                labels = np.array(data['labels'])
                probs = np.array(data['probs'])
        except:
            print(f"{hp['root_dir']}/{hp['test_res_file']}_{j}.csv")
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
    return fpr,tpr,mean_auc,total_auc
# plots a boxplot of baseline vs BP-CNN AUC results    
def roc_boxplot(aucs_base,aucs_sub,feature):
    data = [aucs_base,aucs_sub]

    print("medians: ", np.median(aucs_base),np.median(aucs_sub))
    labels = ["baseline",f"{feature}"]
    labelsize = 22
    mpl.rcParams['xtick.labelsize'] = labelsize
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = labelsize
    fig= plt.figure()
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
    return "{:.2f}".format(np.std(samples_a)),  "{:.2f}".format(np.std(samples_b))

# plots the ROC of baseline vs BP-CNN results    
def plot_base_sub(hp,feature):

    mode = 'test'
    root1 = 'C:/Users/hadar/Downloads/biomedical_eng/winter_2022/research/base_cnv/'
    root2 = f"{hp['root_dir']}"
    fpr1,tpr1,auc1,aucs1 = get_mean_roc(root1)
    fpr2,tpr2,auc2,aucs2 = get_mean_roc(root2)
    
    plt.rcParams.update({'font.size':14})
    std1, std2=  paired_t_test(aucs1,aucs2)
    data1 = {"fpr": fpr1[0], "tpr_lo":tpr1[0], "tpr_mean": tpr1[1], "tpr_hi": tpr1[2]}
    df1 = pd.DataFrame(data1)
    data2 = {"fpr": fpr2[0], "tpr_lo":tpr2[0], "tpr_mean": tpr2[1], "tpr_hi": tpr2[2]}
    df2 = pd.DataFrame(data2)
    ax1 = seaborn.lineplot(x=df1.loc[:,'fpr'].values,y=df1.loc[:,'tpr_mean'].values,color='green')
    ax1.fill_between(fpr1[0], tpr1[0],tpr1[1],color='green', alpha=.1)
    ax1.fill_between(fpr1[0], tpr1[1],tpr1[2],color='green', alpha=.1)
    ax2 = seaborn.lineplot(x=df2.loc[:,'fpr'].values,y=df2.loc[:,'tpr_mean'].values,color='blue')
    ax2.fill_between(fpr2[0], tpr2[0],tpr2[1],color='blue', alpha=.1)
    ax2.fill_between(fpr2[0], tpr2[1],tpr2[2],color='blue', alpha=.1)

    auc1 = "{:.2f}".format(auc1)
    auc2 = "{:.2f}".format(auc2)
    # add legend
    top_bar = mpatches.Patch(color='green', label=f'Baseline (area='+auc1+r'$\pm$'+std1+')')
    bottom_bar = mpatches.Patch(color='blue', label='$BP-CNN_{CIMP}$ (area='+auc2+r'$\pm$'+std2+')')
    plt.legend(handles=[top_bar, bottom_bar],fontsize=14)

    ax1.set_xlabel('False Positive Rate',fontsize=14)
    ax1.set_ylabel('True Positive Rate',fontsize=14)
    ax1.set_title(f"ROC Baseline vs {feature}",fontsize=20)

    ax1.grid(True)
    plt.savefig("{}roc_{}_vs.png".format(hp['root_dir'],mode),dpi=500,bbox_inches="tight") 
    plt.close()
    roc_boxplot(aucs1,aucs2,feature)  

    data_aucs = {"aucs_base": aucs1, "aucs_sub": aucs2}
    pickle.dump( data_aucs, open( f"{hp['root_dir']}aucs.p", "wb" ) )


def delong(hp):
    root1 = '/project/MSI_MSS_project/base_cimp/'
    root2 = f"{hp['root_dir']}"
    pvals = []
    for j in range(5):
        with (open(f"{root1}roc_out_{j}.p", "rb")) as openfile:
                data1 = pickle.load(openfile)
        with (open(f"{root2}roc_out_{j}.p", "rb")) as openfile:
                data2 = pickle.load(openfile)
        labels = np.array(data1['labels'])
        probs1 = np.array(data1['probs'])
        probs2 = np.array(data2['probs'])
        pval = delong_roc_test(labels, probs1, probs2)
        print(pval)
        pvals.append(pval)
    data_delong = {"pval": pvals}
    pickle.dump( data_delong, open( f"{hp['root_dir']}delong.p", "wb" ) )

hp = hyperparams() 

plot_mean_auc(p,'test',hp)  
#plot_base_sub(hp,"CNV") 
#delong(hp)


