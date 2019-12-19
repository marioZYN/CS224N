#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, in_dim, out_dim, k=5):
        super(CNN, self).__init__()
        self.layer = nn.Conv1d(in_dim, out_dim, k)
    
    def forward(self, x):
        y = self.layer(x)
        y = torch.relu(y)
        m_word = y.shape[-1]
        y = torch.nn.functional.max_pool1d(y, kernel_size=m_word)

        return y

### END YOUR CODE

