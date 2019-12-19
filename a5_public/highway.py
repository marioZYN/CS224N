#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn

### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, int_dim, out_dim):
        super(Highway, self).__init__()
        self.proj = nn.Linear(int_dim, out_dim)
        self.gate = nn.Linear(int_dim, out_dim)
    
    def forward(self, x):
        y_proj = torch.relu(self.proj(x))
        y_gate = torch.sigmoid(self.gate(x))
        y = y_proj.mul(y_gate) + (1 - y_gate).mul(x)

        return y

### END YOUR CODE 

