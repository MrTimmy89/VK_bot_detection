import copy
import math
import random
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
        
        
class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, emb_dim, output_dim):
        
        super().__init__()
        self.input_dim = input_dim
        self.conv = dgl.nn.GraphConv(input_dim, output_dim, allow_zero_in_degree=True)
        self.layer1 = nn.Sequential(
            nn.Linear(output_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.LeakyReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(emb_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU()
        )
        self.avgpool = dgl.nn.AvgPooling()

    def forward(self, g, node_feat):
        x = self.conv(g, g.ndata[node_feat])
        x = F.relu(x)
        x = self.avgpool(g, x)
        x = F.relu(x)
        x = self.layer1(x)
        output = self.layer2(x)
        return output
    
    def save(self, mode):
        with open(f"encoder_{mode}.pkl", "wb") as handler:
            pickle.dump(self, handler)
            

def train(encoder, aug_batch1, aug_batch2, opt):
    
    aug_1 = encoder(aug_batch1, "attr")
    aug_2 = encoder(aug_batch2, "attr")
    
    opt.zero_grad()
    loss = ntxent(aug_1, aug_2)
   
    loss.backward()
    opt.step()
    return loss.item()
    
    
def ntxent(x1, x2, tau=0.1):
    def norm(x):
        return torch.sqrt(torch.square(x).sum())
    
    def sim(x, y):
        return x.T @ y / (norm(x) * norm(y))
    
    N = x1.shape[0]
    L = 0
    for i in range(N):
        delim = 0
        for j in range(N):
            if i != j:
                delim += torch.exp(sim(x1[i], x2[j]) / tau)
        L += -torch.log((torch.exp(sim(x1[i], x2[i]) / tau)) / delim)
    L /= N
    return L
