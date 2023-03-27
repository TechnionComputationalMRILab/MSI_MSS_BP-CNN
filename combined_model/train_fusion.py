#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:38:24 2022

@author: hadar.hezi@bm.technion.ac.il
"""


import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score,recall_score
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import torch.nn.functional as F
import time
import os
import copy

import log_file
from log_file import summary
from compute_roc import *
from plot import plot_loss_acc
from get_auc import *
    
def train_fusion(model1,model2,mlp ,criterion, optimizer,prepare, hp,saved_state=None, 
                early_stopping=7):
    since = time.time()
    best_auc = 0.0
    best_f1 = 0
    best_rec=0
    lr = hp['lr']
    # trained BP-CNNs are fixed
    model1.eval()
    model2.eval()
    Result = namedtuple("Result", "f1 loss")
    train_res = Result(f1=[], loss=[])
    valid_res = Result(f1=[], loss=[])
    epochs_without_improvement = 0
    train_data_path = f"{hp['root_dir']}{hp['train_res_file']}_{hp['curr_fold']}.csv"
    model_checkpoint =  f"{hp['root_dir']}{hp['checkpoint_save']}_{hp['curr_fold']}.pt"
    num_epochs = hp['num_epochs']
    for epoch in range(num_epochs):  
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            epoch_preds,epoch_paths,epoch_labels,epoch_sub_labels, epoch_pos_probs = [],[],[],[],[]
            if phase == 'train':
                mlp.train()  # Set model to training mode
                    
            else:
                mlp.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels,sub_labels, paths in prepare.dataloaders[phase]:
                paths = np.array(paths)
                inputs = inputs.to(prepare.device)
                labels=labels.to(prepare.device)
                sub_labels=sub_labels.to(prepare.device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # outputs from trained BP-CNNs
                    outputs1 = model1(inputs)
                    outputs2 = model2(inputs)
                    # insert to mlp
                    mlp_in = torch.hstack((outputs1,outputs2))
                    outputs = mlp(mlp_in)
                    loss = criterion(outputs, labels)    
                    # transfer MSI/MSS outputs to probabilities
                    prob_y = F.softmax(outputs, dim=1)
                    # predicted labels
                    _, preds = torch.max(prob_y, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                             
                    # MSI probability
                    prob_y_msi = prob_y[:,0]
                    epoch_preds.append(preds)
                    epoch_labels.append(labels)
                    epoch_sub_labels.append(sub_labels)
                    epoch_pos_probs.append(prob_y_msi)
                    epoch_paths.append(paths)
                   
                # statistics
                running_loss += loss.item() * inputs.size(0)

                # calculate accuracy with labels OR sub_labels
                running_corrects += torch.sum(preds ==labels.data)
            # save epoch results            
            epoch_preds_tensor = np.array(torch.cat(epoch_preds).cpu().detach())
            epoch_labels_tensor = np.array(torch.cat(epoch_labels).cpu().detach())
            epoch_sub_labels_tensor = np.array(torch.cat(epoch_sub_labels).cpu().detach())
            epoch_pos_probs_tensor = np.array(torch.cat(epoch_pos_probs).cpu().detach())
            epoch_paths_tensor = np.hstack(epoch_paths)
        
            epoch_loss = running_loss / prepare.dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / len(epoch_preds_tensor)) *100
            f1 = f1_score(epoch_labels_tensor,epoch_preds_tensor,average='weighted')
            if phase == 'train':
                train_res.f1.append(f1)
                train_res.loss.append(epoch_loss)
                train_summary = summary(epoch_preds_tensor,epoch_labels_tensor,epoch_sub_labels_tensor,epoch_pos_probs_tensor,epoch_paths_tensor)
            else:
                valid_res.f1.append(f1)
                valid_res.loss.append(epoch_loss)
                valid_summary = summary(epoch_preds_tensor,epoch_labels_tensor,epoch_sub_labels_tensor,epoch_pos_probs_tensor,epoch_paths_tensor)
                #acc per class
                acc_per_class = cm.diagonal()/cm.sum(axis=1)
                print("accuracy per class:" , acc_per_class)
                #recall take msi labels
                y_true_tmp = np.array(epoch_labels_tensor)
                y_true = np.array(epoch_labels_tensor)
                y_true[y_true_tmp==1]=0
                y_true[y_true_tmp==0]=1

                y_pred_tmp = np.array(epoch_preds_tensor)
                y_pred = np.array(epoch_preds_tensor)
                y_pred[y_pred_tmp==1]=0
                y_pred[y_pred_tmp==0]=1
       
                rec = recall_score(y_true, y_pred, average=None)
                print("mss rec: ", rec[0],"msi rec: ",rec[1])
                # calculate AUC patient level
                valid_auc = get_auc(valid_summary,prepare,mode='valid',hp=hp)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print(f"{phase} f1: {f1}")
            # deep copy the model
            if phase == 'valid':
                if valid_auc>=best_auc or rec[1] >= best_rec:
                # if valid_auc>=best_auc:
                    epochs_without_improvement=0
                    if valid_auc>=best_auc:
                         best_auc = valid_auc
                    if rec[1] >= best_rec:                       
                         best_rec = rec[1]
                   
                    print("**saving model")
                    mlp.train()
                    saved_state = dict(
                            lr = lr,
                            best_auc=best_auc,
                            ewi=epochs_without_improvement,
                            model_state=mlp.state_dict(),
                        )
                    torch.save(saved_state, model_checkpoint)
                    # reduce step size
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr*0.97
                    lr = lr*0.97 
                else: 
                    epochs_without_improvement+=1  
        if epochs_without_improvement == early_stopping:
            actual_num_epochs = epoch
            break  

    log_file.save_results(train_data_path, train_summary)
    log_file.save_results(f"{hp['root_dir']}{hp['valid_res_file']}_{hp['curr_fold']}.csv", valid_summary)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val AUC: {:4f}'.format(best_auc))
    plot_loss_acc(train_res,valid_res,hp)


    del loss
    del prepare.dataloaders['train']
    return mlp,saved_state



