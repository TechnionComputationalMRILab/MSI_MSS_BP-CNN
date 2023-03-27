#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 09:08:06 2021

@author: hadar.hezi@bm.technion.ac.il
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from collections import namedtuple
import sklearn.metrics as sk
import pandas as pd
import os
import statistics
import random


from prepare_data import Prepare
import log_file
from log_file import summary
from test_fusion import test_fusion
from plot import *
from train_fusion import train_fusion
from hyperparams import hyperparams
from mlp import MLPmodel_10

# boolean whether to run fold
def get_fold(k,curr_i):
    if curr_i not in k:
        return False
    else: return True
    
    
# which transfer learning model to use
def set_model(hp):
    if hp['model_type'] == 'resnext':
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=hp['pretrained'])
    elif hp['model_type'] == 'inception':
        model_ft = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=hp['pretrained'])
    elif hp['model_type'] == 'efficient':
        model_ft = EfficientNet.from_pretrained('efficientnet-b7',num_classes=hp['num_classes'])
        
    if hp['unfixed'] == False:
        model_ft = p.train_params(hp,model_ft)   
        
    num_ftrs = model_ft.fc.in_features
     # define classifier layers 
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 100),
        nn.Linear(100, hp['num_classes'])
        )
    model_ft = model_ft.to(p.device)
    if  hp['optimizer_type'] == 'SGD':
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=hp['lr'],momentum=0.99)
    elif  hp['optimizer_type'] == 'Adam':
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=hp['lr'])
    # print(model_ft)
    return model_ft, optimizer_ft

def seed_torch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

torch.cuda.empty_cache()
# hyper-parameters dictionary
hp = hyperparams()
# set seeds for reproducability
seed_torch(hp['newseed'])
# define criterion function
criterion = nn.CrossEntropyLoss()
cross_valid =  hp['n_folds']
print(hp)
# object for data and model definitions
p = Prepare(hp)
for i in range(cross_valid):
    saved_state = None
    hp['curr_fold'] = i
    train_bool = get_fold(hp['folds'],i)
    # will get the next fold
    p.prepare_data(hp) 
    print(p.device)
    # create the data loaders
    p.create_train_validation_loaders()
    # root to the trained BP-CNN-SNP models
    root2 = f"/project/MSI_MSS_project/snp_model/"
    # create two inception model instances
    model1,_ = set_model(hp)
    model2,_ = set_model(hp)
    # load the saved BP-CNNs
    if os.path.isfile(f"{hp['root_dir']}{hp['checkpoint_load']}_{i}.pt"):  
        # loads BP-CNN-CIMP
        model1,_  = p.load_model(f"{hp['root_dir']}{hp['checkpoint_load']}_{i}.pt",model1)
    if os.path.isfile(f"{root2}{hp['checkpoint_load']}_{i}.pt"):  
        # loads BP-CNN-SNP
        model2,_  = p.load_model(f"{root2}{hp['checkpoint_load']}_{i}.pt",model2)
    # The combined model
    mlp = MLPmodel_10(hp)
    mlp = mlp.to(p.device)
    optimizer_ft = optim.SGD(mlp.parameters(), lr=hp['lr'],momentum=0.99)
    if(train_bool):
        # training loop
        mlp,_ = train_fusion(model1,model2,mlp, criterion, optimizer_ft,p,hp,saved_state,early_stopping=8)
        torch.cuda.empty_cache()
        #create the test loadrer
        p.create_test_loader()
        # the test loop
        test_auc = test_fusion(model1,model2,mlp,p,hp)
    #load best model
    torch.cuda.empty_cache()
    p.create_test_loader()
    mlp = MLPmodel_10(hp)
    mlp = mlp.to(p.device)
    # # load previously trained model
    if os.path.isfile(f"{hp['root_dir']}{hp['checkpoint_save']}_{i}.pt"):   
        mlp,_  = p.load_model(f"{hp['root_dir']}{hp['checkpoint_save']}_{i}.pt",mlp)
    test_auc =  test_fusion(model1,model2,mlp,p,hp)


            
    
