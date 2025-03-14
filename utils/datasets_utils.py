#!/usr/bin/env python
# coding: utf-8

# In[4]:


#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import shutil
import pickle
import torch
import numpy as np
import json
import networkx as nx
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data, download_url
from torch_geometric.utils import to_undirected, from_networkx, coalesce,remove_self_loops
from torch_geometric.datasets import Planetoid, Amazon, WikipediaNetwork, Actor, WebKB

from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import WikipediaNetwork


class HeterophilyDataset(InMemoryDataset):
    def __init__(self, root='data/', name=None, 
                 raw_data_dir=None, 
                 train_percent=0.01, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset should be saved.
            name (str): Name of the dataset (options: 'chameleon', 'actor', 'squirrel').
            raw_data_dir (str): Path to the raw dataset directory.
            train_percent (float): Percentage of training data.
            transform (callable, optional): A function/transform that takes in a `Data` object and returns a transformed version.
            pre_transform (callable, optional): A function/transform that takes in a `Data` object and returns a transformed version before saving.
        """
        existing_dataset = ['chameleon', 'actor', 'squirrel']
        if name not in existing_dataset:
            raise ValueError(f"Dataset name must be one of: {existing_dataset}")
        else:
            self.name = name

        self._train_percent = train_percent

        if raw_data_dir is not None and os.path.isdir(raw_data_dir):
            self.raw_data_dir = raw_data_dir
        elif raw_data_dir is None:
            self.raw_data_dir = None
        elif not os.path.isdir(raw_data_dir):
            raise ValueError(f"Raw dataset directory '{raw_data_dir}' does not exist!")

        if not os.path.isdir(root):
            os.makedirs(root)

        self.root = root

        super(HeterophilyDataset, self).__init__(root, transform, pre_transform)

   
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [self.name]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        if self.name in ['chameleon', 'squirrel']:
      
            unprocessed_data = WikipediaNetwork(
                root=self.root, name=self.name, geom_gcn_preprocess=False, transform=None
            )
         
            unprocessed_data = unprocessed_data[0]  

           
            processed_data = WikipediaNetwork(
                root=self.root, name=self.name, geom_gcn_preprocess=True, transform=None
            )
        
            processed_data = processed_data[0]  

        
            unprocessed_edge_index = unprocessed_data.edge_index
            processed_edge_index = processed_data.edge_index
            features = processed_data.x
            labels = processed_data.y

          
            unprocessed_edge_index, _ = remove_self_loops(unprocessed_edge_index)
            unprocessed_edge_index = to_undirected(unprocessed_edge_index)

            processed_data.edge_index = unprocessed_edge_index

      
            data = Data(x=features, edge_index=processed_data.edge_index, y=labels)

        else:
      
            with open(self.raw_data_dir, 'rb') as f:
                data = pickle.load(f)

            edge_index = data['edge_index']
            features = data['features']
            labels = data['labels']

       
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)

 
            data = Data(x=features, edge_index=edge_index, y=labels)

     
        if self.pre_transform is not None:
            data = self.pre_transform(data)

  
        data = T.NormalizeFeatures()(data)

    
        torch.save(data, self.processed_paths[0])

  
        return data


    def __repr__(self):
        return '{}()'.format(self.name)



class WebKB(InMemoryDataset):
    """
    The WebKB datasets used in the
    "Geom-GCN: Geometric Graph Convolutional Networks"
    paper.

    """

    url = ('https://raw.githubusercontent.com/graphdml-uiuc-jlu/geom-gcn/'
           'master/new_data')

    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        assert self.name in ['cornell', 'texas', 'washington', 'wisconsin']

        super(WebKB, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ['out1_node_feature_label.txt', 'out1_graph_edges.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        """Downloads the raw data files from the provided URL."""
        for name in self.raw_file_names:
            download_url(f'{self.url}/{self.name}/{name}', self.raw_dir)

    def process(self):
  
        with open(self.raw_paths[0], 'r') as f:
            data = f.read().split('\n')[1:-1]
            x = [[float(v) for v in r.split('\t')[1].split(',')] for r in data]
            x = torch.tensor(x, dtype=torch.float)

            y = [int(r.split('\t')[2]) for r in data]
            y = torch.tensor(y, dtype=torch.long)

        # Load edge indices
        with open(self.raw_paths[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
            data = [[int(v) for v in r.split('\t')] for r in data]
            edge_index = torch.tensor(data, dtype=torch.long).t().contiguous()
            edge_index, _ = remove_self_loops(edge_index)
            edge_index = to_undirected(edge_index)
            edge_index, _ = coalesce(edge_index, None, x.size(0), x.size(0))

        # Create Data object
        data = Data(x=x, edge_index=edge_index, y=y)

        # Apply pre-transforms if available
        data = data if self.pre_transform is None else self.pre_transform(data)

        # Save the processed data
        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return f'{self.name.capitalize()}()'

def get_dataset(name):
    allowed_datasets = ['cora', 'citeseer', 'pubmed', 'computers', 'photo',
                        'chameleon', 'actor', 'squirrel', 'texas', 'cornell']
    name = name.lower()

    if name not in allowed_datasets:
        raise ValueError(f"Dataset {name} is not supported. Please select from {allowed_datasets}.")


    root_path = '../data/'
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_path, name=name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(root=root_path, name=name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(root=root_path, name=name, transform=T.NormalizeFeatures())
    elif name in ['actor']:
        dataset = Actor(root=root_path, transform=T.NormalizeFeatures())
  
    elif name in ['texas', 'cornell']:
        dataset = WebKB(root=os.path.join(root_path, name), name=name, transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Dataset {name} is not supported in this loader.")
    
    return dataset, dataset[0]

def process_dataset(name):
    allowed_datasets = [
        'cora', 'citeseer', 'pubmed',  # Planetoid datasets
        'computers', 'photo',          # Amazon datasets
        'chameleon', 'squirrel', 'actor',  # Heterophily datasets
        'actor',                       # Actor dataset
        'texas', 'cornell',            # WebKB datasets
    ]
    name = name.lower()

    if name not in allowed_datasets:
        raise ValueError(f"Dataset {name} is not supported. Please select from {allowed_datasets}.")

    root_path = '../data/'
    
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=root_path, name=name,transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(root=root_path, name=name,transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        # Load unprocessed version
        unprocessed_data = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=False, transform=T.NormalizeFeatures()
        )
        unprocessed_data = unprocessed_data[0]  # Get the first Data object

        # Load processed version
        processed_data = WikipediaNetwork(
            root=root_path, name=name, geom_gcn_preprocess=True, transform=T.NormalizeFeatures()
        )
        processed_data = processed_data[0]  # Get the first Data object

        # Merge the unprocessed and processed edge indices
        unprocessed_edge_index = unprocessed_data.edge_index
        processed_edge_index = processed_data.edge_index
        features = processed_data.x
        labels = processed_data.y

        # Remove self-loops and convert to undirected graph
        unprocessed_edge_index, _ = remove_self_loops(unprocessed_edge_index)
        unprocessed_edge_index = to_undirected(unprocessed_edge_index)

        # Assign cleaned edge index to the processed version
        processed_data.edge_index = unprocessed_edge_index

        # Create a new Data object
        data = Data(x=features, edge_index=processed_data.edge_index, y=labels)

        dataset = None  # We don't need the whole dataset object for this case, only the Data object
    elif name == 'actor':
        # Specify actor dataset's raw directory
        actor_dir = os.path.join(root_path, "actor", "raw")
        if not os.path.exists(actor_dir):
            raise FileNotFoundError(f"Directory '{actor_dir}' does not exist. Please check your dataset location.")

        # Node features and labels file
        node_file = os.path.join(actor_dir, "out1_node_feature_label.txt")
        if not os.path.exists(node_file):
            raise FileNotFoundError(f"File '{node_file}' not found. Please check your dataset location.")

        with open(node_file, 'r') as f:
            lines = f.readlines()

        # Set feature dimension to 932 (0-931)
        num_features = 932

        features_list = []
        labels_list = []

        for line in lines[1:]:  # Skip header line
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue  # Ignore malformed lines

            try:
                # Extract feature index list
                indices = list(map(int, parts[1].split(",")))
            except:
                indices = []

            # Initialize a zero vector of length 932
            vector = [0.0] * num_features
            for idx in indices:
                # Ensure the index is within the range of 0-931
                if 0 <= idx < num_features:
                    vector[idx] = 1.0

            features_list.append(vector)
            labels_list.append(int(parts[2]))  # Read the class label

        # Convert to PyTorch Tensor
        x = torch.tensor(features_list, dtype=torch.float)
        y = torch.tensor(labels_list, dtype=torch.long)

        # Standardize features (Normalize)
        x = T.NormalizeFeatures()(Data(x=x, edge_index=None, y=y)).x

        # Read edge information
        edge_file = os.path.join(actor_dir, "out1_graph_edges.txt")
        if not os.path.exists(edge_file):
            raise FileNotFoundError(f"File '{edge_file}' not found. Please check your dataset location.")

        edges = []
        with open(edge_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip header line

        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            edges.append([int(parts[0]), int(parts[1])])

        # Convert to PyTorch format
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Preprocess: remove self-loops and convert to undirected graph
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = to_undirected(edge_index)

        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset = None

    elif name in ['texas', 'cornell']:
        dataset = WebKB(root=os.path.join(root_path, name), name=name,transform=T.NormalizeFeatures())
    else:
        raise ValueError(f"Dataset {name} is not supported in this loader.")

    # Return only the Data object (not the whole dataset)
    return dataset,dataset[0]




def save_processed_data(data, dataset_name, dataset_dir):
    """
    Save the processed data directly into the dataset's folder.

    Args:
        data (Data): PyTorch Geometric data object.
        dataset_name (str): Name of the dataset.
        dataset_dir (str): Dataset directory to save processed data.
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    save_path = os.path.join(dataset_dir, 'processed', 'data.pt')
    torch.save(data, save_path)
    print(f"Processed data saved to: {save_path}")









