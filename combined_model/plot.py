#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 13:06:43 2021

@author: hadar.hezi@bm.technion.ac.il
"""

from collections import namedtuple
import matplotlib.pyplot as plt

def plot_loss_acc(train_res,valid_res,hp):
     # Plot loss and F1
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    for i, loss_acc in enumerate(('loss', 'f1')):
        axes[i].plot(getattr(train_res, loss_acc))
        axes[i].plot(getattr(valid_res, loss_acc))
        axes[i].set_title(loss_acc.capitalize(), fontweight='bold')
        axes[i].set_xlabel('Epoch')
        axes[i].legend(('train', 'valid'))
        axes[i].grid(which='both', axis='y')
        plt.savefig(f'train_valid_{loss_acc}.png')
        plt.close()