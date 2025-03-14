#!/usr/bin/env python
# coding: utf-8

# In[3]:

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="NGHD Node-Level Heat Diffusion Model Training Experiment"
    )
    parser.add_argument("--dataset", type=str, default="citeseer",
                        help="Name of the dataset (e.g., cora)")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.77,
                        help="Dropout rate")
    parser.add_argument("--K", type=int, default=10,
                        help="Highest order for heat diffusion")
    parser.add_argument("--node_t_init", type=float, default=3.12,
                        help="Initial value for node-level diffusion parameter")
    parser.add_argument("--seed", type=int, default=88,
                        help="Random seed")
    parser.add_argument('--decrease_threshold', type=float, default=0.5, help='Threshold for sudden decrease in val_acc')
    parser.add_argument('--early_stopping', type=int, default=200)
    return parser.parse_known_args()


# In[ ]:




