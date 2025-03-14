#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch_geometric.utils import degree
from scipy.linalg import expm
from parser import parse_args
from utils.splits_utils import random_planetoid_splits
import model  
from baselines import *
def fix_random(seed_val):
    torch.manual_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def check_optimizer_for_node_t(optimizer, model):

    for param_group in optimizer.param_groups:
        for param in param_group['params']:

            if param is model.diffusion.node_t:
                print("Found node_t in optimizer!")
            else:
                print(f"Parameter shape: {param.shape}")
def run_training(net, optimizer, data_obj):
    net.train()
    optimizer.zero_grad()

    output_logits = net(data_obj.x, data_obj.edge_index)
    loss_val = F.nll_loss(output_logits[data_obj.train_mask], data_obj.y[data_obj.train_mask])
    loss_val.backward()
    optimizer.step()
    return loss_val.item()

def evaluate_model(net, data_obj):
    net.eval()
    with torch.no_grad():
        logits = net(data_obj.x, data_obj.edge_index)
        train_pred = logits[data_obj.train_mask].max(dim=1)[1]
        train_accuracy = train_pred.eq(data_obj.y[data_obj.train_mask]).sum().item() / data_obj.train_mask.sum().item()

        val_pred = logits[data_obj.val_mask].max(dim=1)[1]
        val_accuracy = val_pred.eq(data_obj.y[data_obj.val_mask]).sum().item() / data_obj.val_mask.sum().item()

        test_pred = logits[data_obj.test_mask].max(dim=1)[1]
        test_accuracy = test_pred.eq(data_obj.y[data_obj.test_mask]).sum().item() / data_obj.test_mask.sum().item()
    return train_accuracy, val_accuracy, test_accuracy

def main():
    args, _ = parse_args()
    fix_random(args.seed)

    dataset_name = args.dataset.lower()
    #dataset_name = args.dataset.lower()
    data_file = os.path.join("data", dataset_name, "processed", "data.pt")
    if not os.path.exists(data_file):
        print("Data file not found at:", data_file)
        sys.exit(1)

    data_obj = torch.load(data_file)
    eg=data_obj.edge_index
    data_obj.edge_index, _ = remove_self_loops(data_obj.edge_index)
    
   
    if not hasattr(data_obj, "train_mask"):
        num_classes = int(data_obj.y.max().item() + 1)
        data_obj, _ = random_planetoid_splits(data_obj, num_classes=num_classes,
                                               percls_trn=22, val_lb=37, Flag=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_obj = data_obj.to(device)

    N = data_obj.x.size(0)

    in_features = data_obj.x.size(1)
    n_classes = int(data_obj.y.max().item() + 1)
    degree_values = degree(data_obj.edge_index[0], num_nodes=N)
    degree_values = degree_values.cpu().numpy()
    final_t = []
    patience = args.early_stopping

    net_model = model.LHKGNN(in_dim=in_features,
                             hidden=args.hidden,
                             out_dim=n_classes,
                             K=args.K,
                             node_t_init=args.node_t_init,
                             dp=args.dropout,
                             num=N)
    net_model = net_model.to(device)

    optimizer = torch.optim.Adam(net_model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    #check_optimizer_for_node_t(optimizer, net_model)

    best_val_acc = 0.0

    epochs_without_improvement = 0
    last_val_acc = -1  
    for epoch in range(1, args.epochs + 1):
        loss_epoch = run_training(net_model, optimizer, data_obj)
        _, val_acc, test_acc = evaluate_model(net_model, data_obj)

       
        if last_val_acc != -1 and val_acc < last_val_acc * (1 - args.decrease_threshold):  
            print(f"stopping early.")
            break

        last_val_acc = val_acc  

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            final_test_acc = test_acc
            epochs_without_improvement = 0  
            final_t = net_model.diffusion.node_t.clone()
            final_t = F.relu(final_t)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch:04d} due to no improvement in validation accuracy.")
            break

        if epoch % 20 == 0:
            print(f"Epoch {epoch:04d}: Loss = {loss_epoch:.4f}, Val Acc = {val_acc:.4f}, Test Acc = {test_acc:.4f}")

    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Test Accuracy (corresponding to best val): {final_test_acc:.4f}")



if __name__ == "__main__":
    main()


# In[ ]:




