#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 12:15:16 2022

@author: hadar.hezi@bm.technion.ac.il
"""

import torch.nn as nn
import torch

class MLPmodel_10(nn.Module):
  """A single linear layer"""
  def __init__(self,hp):
    super(MLPmodel_10,self).__init__()
    # output classes MSI/MSS
    self.out_class = 2
    # input dimension is concatenating two BP-CNNs with 3-classes output each
    self.dim_in = hp['num_classes']*2

    self.mlp = nn.Sequential(   
        #nn.LeakyReLU(negative_slope=self.slope,inplace=True),
        nn.Linear(self.dim_in,self.out_class),
        )

        
  def forward(self, ev):

    ev = self.mlp(ev)
    return ev