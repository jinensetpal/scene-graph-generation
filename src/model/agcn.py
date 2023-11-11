#!/usr/bin/env python3

from torch_geometric.nn.conv import HeteroConv, GATConv
from .. import const
import torch.nn as nn


class aGCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = HeteroConv({
            ('object', 'skip', 'object'): GATConv(const.repn.PROJECTION, const.N_RELATIONS, add_self_loops=False),
            ('object', 'to', 'relation'): GATConv((const.repn.PROJECTION,)*2, const.N_RELATIONS, add_self_loops=False),
            ('relation', 'to', 'object'): GATConv((const.repn.PROJECTION,)*2, const.N_RELATIONS, add_self_loops=False)}, aggr='sum')
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        predictions = []
        for graph in x:
            features = {k: v['features'] for k, v in graph.node_items()}
            edge_indices = {k: v['edge_index'] for k, v in graph.edge_items()}
            x = self.conv(features, edge_indices)
            x = self.softmax(x['relation'])
            graph['relation'].logits = x

            predictions.append(graph)

        return predictions
