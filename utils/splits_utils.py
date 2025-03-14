#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = True
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)
    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb] for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]
        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data, [train_index, rest_index[:val_lb], rest_index[val_lb:]]


def random_heterophilic_splits(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=0):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    
    data.train_mask = index_to_mask(train_idx, size=n)
    data.val_mask = index_to_mask(val_idx, size=n)
    data.test_mask = index_to_mask(test_idx, size=n)
    
    return data, (train_idx, val_idx, test_idx)

def random_amazon_splits(data, train_ratio=0.025, val_ratio=0.025, test_ratio=0.95, seed=0):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = data.num_nodes
    perm = torch.randperm(n)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]
    
    data.train_mask = index_to_mask(train_idx, size=n)
    data.val_mask = index_to_mask(val_idx, size=n)
    data.test_mask = index_to_mask(test_idx, size=n)
    
    return data, (train_idx, val_idx, test_idx)

def idx_split(idx, ratio, seed=0):
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)
    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    return idx1, idx2

def graph_split(idx_train, idx_val, idx_test, rate, seed=0):
    idx_test_ind, idx_test_tran = idx_split(idx_test, rate, seed)
    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    M1, M2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:M1]
    obs_idx_val = obs_idx_all[M1 : M1 + M2]
    obs_idx_test = obs_idx_all[M1 + M2 :]
    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind









