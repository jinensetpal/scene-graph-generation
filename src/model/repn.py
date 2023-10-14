#!/usr/bin/env python3

from torch_geometric.data import Data
from itertools import combinations
from src.utils.nms import nms
from src import const
import torch.nn as nn
import torch


class RePN(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj_subj = nn.Sequential(nn.Linear(const.N_CLASSES, const.repn.HIDDEN),
                                       nn.ReLU(),
                                       nn.Linear(const.repn.HIDDEN, const.repn.PROJECTION))
        self.proj_obj = nn.Sequential(nn.Linear(const.N_CLASSES, const.repn.HIDDEN),
                                      nn.ReLU(),
                                      nn.Linear(const.repn.HIDDEN, const.repn.PROJECTION))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        graphs = []
        for prediction in x:
            n_pairs = prediction['boxes'].shape[0]
            score_matrix = torch.zeros([n_pairs,] * 2)

            for subj_idx, obj_idx in combinations(range(n_pairs), 2):
                score_matrix[subj_idx][obj_idx] = torch.matmul(self.proj_subj(prediction['scores'][subj_idx][:-1]) * prediction['features'][subj_idx],
                                                               self.proj_obj(prediction['scores'][obj_idx][:-1]) * prediction['features'][obj_idx])

            score_matrix = self.sigmoid(score_matrix).fill_diagonal_(torch.nan)
            pairs = score_matrix.view(-1).sort(descending=True)
            pairs = nms(pairs.indices[pairs.values.isnan() == False][:const.repn.TOP_K],  # sorted in descending order of confidence
                        prediction['boxes'], n_pairs)

            boxes = torch.empty(1, 2, 4)
            features = torch.empty(1, 3, 1024)
            for pair in pairs:
                idx = (pair // n_pairs, pair % n_pairs)
                box_set = (torch.vstack([prediction['boxes'][idx[0]],
                                         prediction['boxes'][idx[1]]])).unsqueeze(0)
                features_set = (torch.vstack([prediction['features'][idx[0]],
                                              prediction['features'][idx[1]]])).unsqueeze(0)

                features = torch.vstack([features, torch.vstack([features_set[0], features_set.mean(dim=1)]).unsqueeze(0)])
                boxes = torch.vstack([boxes, box_set])

            boxes = boxes[1:]
            features = features[1:]

            graphs.append(self._generate_graph(boxes, features))
        return graphs

    @staticmethod
    def _generate_graph(box_pairs, features):
        node_boxes = torch.unique(box_pairs.view(-1, 4), dim=0)
        node_features = torch.unique(features[:, :2].reshape(-1, features.shape[-1]), dim=0)
        node_features = torch.vstack([node_features, features[:, 2]])
        ln = node_boxes.tolist()

        edge_indices = torch.vstack([torch.tensor(list(combinations(range(node_boxes.shape[0]), 2))),
                                     torch.tensor([[ln.index(box.tolist()), node_boxes.shape[0] + idx] for idx, boxes in enumerate(box_pairs) for box in boxes], dtype=torch.int32)])
        return Data(x=node_features, edge_index=edge_indices, boxes=node_boxes)
