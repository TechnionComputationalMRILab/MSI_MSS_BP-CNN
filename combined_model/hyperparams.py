#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:28:58 2021

@author: hadar.hezi@bm.technion.ac.il
"""

import numpy as np

def hyperparams():
    hp = dict(
        batch_size=64, lr=5e-4, eps=1e-8, num_workers=2,num_classes=3
    )

    hp['newseed'] = 42
    hp['num_epochs'] = 15
    hp['n_folds'] = 5
    hp['curr_fold'] = 0
    hp['folds'] = [4]
    hp['train_res_file'] = 'train_res_data_snp'
    hp['valid_res_file'] = 'valid_res_data_snp'
    hp['test_res_file'] = 'test_res_data_snp_fusion'
    hp['checkpoint_load'] = 'model_snp'
    hp['checkpoint_save'] = 'model_mlp'
    hp['optimizer_type'] = 'Adam'
    hp['model_type'] = 'inception'
    hp['pretrained'] = True
    hp['unfixed'] = False
    hp['experiment'] = 'MSI MSS combined'
    hp['root_dir']='/project/MSI_MSS_project/snp_fusion_2/'
    hp['data_dir'] = '/project/data/'
    # hp['root_dir']='/tcmldrive/hadar/from_dgx/snp_fusion/'
    # hp['data_dir'] ='/tcmldrive/databases/Public/TCGA/data'
    # ========================
    return hp
