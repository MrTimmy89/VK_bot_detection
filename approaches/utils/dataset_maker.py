import copy
import math
import numpy as np
import dgl
from dgl.nn import GraphConv
from dgl.dataloading import GraphDataLoader
from dgl.data import DGLDataset
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.utils.data import Subset
        
        
def connected_subset(dataset):
    Gs = []
    for gr in dataset:
        G = nx.Graph(gr[0].to_networkx())
        if nx.algorithms.components.number_connected_components(G) == 1:
            Gs.append(gr)
    return torch.utils.data.dataset.Subset(Gs, range(len(Gs)))
    
    
def train_test_val_split(c_dataset, train_idx, test_idx, val_idx):
    X_train = []
    X_test = []
    X_val = []
    y_train = []
    y_test = []
    y_val = []
    for id_ in train_idx:
        X_train.append(c_dataset[id_][0])
        y_train.append(c_dataset[id_][-1])
    for id_ in test_idx:
        X_test.append(c_dataset[id_][0])
        y_test.append(c_dataset[id_][-1])
    for id_ in val_idx:
        X_val.append(c_dataset[id_][0])
        y_val.append(c_dataset[id_][-1])
    X_train = dgl.batch(X_train)
    X_test = dgl.batch(X_test)
    X_val = dgl.batch(X_val)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)
    y_val = torch.Tensor(y_val)
    return X_train, X_test, X_val, y_train, y_test, y_val


class ContrastiveDataset(DGLDataset):
    def __init__(self, filename, augmentations):
        self.filename = filename
        self.graphs = None
        self.labels = None
        self.augmentations = augmentations
        super().__init__(name=filename)

    def process(self):
        graphs, graph_data = dgl.load_graphs(self.filename)
        self.graphs = graphs
        self.labels = graph_data['labels']

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        aug1, aug2 = None, None
        if len(self.augmentations) >= 1:
            a_f = self.augmentations[0]
            aug1 = a_f.transform(g)
        if len(self.augmentations) == 2:
            a_s = self.augmentations[1]
            aug2 = a_s.transform(g)
        l = self.labels[idx]
        return g, aug1, aug2, l
        
        
class GraphAugmentation():
    def __init__(self, type, ratio=0.2, node_feat='attr'):
        self.type = type
        self.ratio = ratio
        self.node_feat = node_feat
    
    def transform(self, g):
        if self.type == 'drop_nodes':
            return self.drop_nodes(g)
        elif self.type == 'pert_edges':
            return self.pert_edges(g)
        elif self.type == 'attr_mask':
            return self.attr_mask(g)
        elif self.type == 'rw_subgraph':
            return self.rw_subgraph(g)
        elif self.type == 'identical':
            return g
    
    def drop_nodes(self, g):
        nodes_num = g.nodes().shape[0]
        nodes = np.random.choice(g.nodes()[1:], 
                               size=int(math.ceil(self.ratio*nodes_num)),
                               replace=False)
        aug = copy.deepcopy(g)
        aug.remove_nodes(nodes)
        aug = aug.remove_self_loop()
        aug = aug.add_self_loop()
        return aug

    def pert_edges(self, g):
        aug = copy.deepcopy(g)
        n_edges = g.edges()[0].shape[0]

        for i in range (0, int(math.ceil(self.ratio * n_edges))):
            v_idx = random.randint(0, n_edges - 1)

            new_u = np.random.choice(g.nodes(), 
                               size = 1,
                               replace = True)[0]
            aug.edges()[1][v_idx] = new_u
        return aug

    def attr_mask(self, g):
        aug = copy.deepcopy(g)
        nodes_num = aug.ndata[self.node_feat].shape[0]
        attrs_num = aug.ndata[self.node_feat].shape[1]
        total = nodes_num * attrs_num
        mask = np.full(total, False)
        inds = np.random.choice(total, 
                              size=int(math.ceil(self.ratio * total)),
                              replace=False)
        mask[inds] = True
        mask = mask.reshape(nodes_num, -1)
        aug.ndata[self.node_feat].masked_fill_(torch.tensor(mask), 1)
        return aug

    def rw_subgraph(self, g):
        aug_1 = copy.deepcopy(g)
        nodes_num = g.ndata[self.node_feat].shape[0]
        curr_node = np.random.choice(nodes_num, 
                              size=1,
                              replace=False)
        
        N_prev = 0

        graph_size = 0
        new_edges_v = []
        new_edges_u = []
        aug_1 = aug_1.remove_self_loop()
        new_num_dict = {}
        prev_step_v = []
        while N_prev <= int(math.ceil(self.ratio*nodes_num)):
            from_v, to_v = aug_1.out_edges(curr_node)
            if len(to_v) == 0:
                if len(prev_step_v) == 0:
                    curr_node = np.random.choice(nodes_num, 
                              size=1,
                              replace=True)
                else:
                    curr_node = np.random.choice(prev_step_v, 
                              size=1,
                              replace=True)
                continue
            for i in range(N_prev, N_prev + len(to_v)):
                if new_num_dict.get(from_v[0].item()) is None:
                    new_num_dict[from_v[0].item()] = N_prev
                    new_edges_v.append(N_prev)
                    N_prev += 1
                else:
                    new_edges_v.append(new_num_dict.get(from_v[0].item()))

            for i in to_v:
                if new_num_dict.get(i.item()) is None:
                    new_num_dict[i.item()] = N_prev
                    new_edges_u.append(N_prev)
                    N_prev += 1
                else:
                    new_edges_u.append(new_num_dict.get(i.item()))
            
            prev_step_v = to_v
            curr_node = np.random.choice(to_v, 
                              size=1,
                              replace=True)
            graph_size += 1 + len(to_v)
        
        aug = dgl.graph((new_edges_v, new_edges_u))
        aug.ndata['attr'] = torch.ones(aug.num_nodes(), 10)

        return aug
        
