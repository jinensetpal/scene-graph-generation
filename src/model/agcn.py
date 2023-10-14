#!/usr/bin/env python3

from torch_geometric.nn.conv import GCNConv
from src import const
import torch.nn as nn
import torch


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj = nn.Linear(const.N_CLASSES, const.agcn.PROJECTION)
        self.softmax = nn.Softmax(dim=1)


class aGCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.gcn = GCNConv()
