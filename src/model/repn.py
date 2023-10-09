#!/usr/bin/env python3

from IPython import embed

from torchvision.ops import box_iou
from itertools import combinations
from src.utils import Graph
from src import const
import torch.nn as nn
import torch

class RePN(nn.Module):
    def __init__(self):
        super().__init__()

        self.proj_subj = nn.Linear(const.N_CLASSES, const.repn.PROJECTION)
        self.proj_obj = nn.Linear(const.N_CLASSES, const.repn.PROJECTION)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        graphs = []
        for prediction in x:
            n_pairs = prediction['boxes'].shape[0]
            score_matrix = torch.zeros([n_pairs,] * 2)

            for subj_idx, obj_idx in combinations(range(n_pairs), 2):
                score_matrix[subj_idx][obj_idx] = torch.matmul(self.proj_subj(prediction['scores'][subj_idx][:-1]) * prediction['features'][subj_idx],
                                                               self.proj_obj(prediction['scores'][obj_idx][:-1]) * prediction['features'][obj_idx])

            pairs = self.sigmoid(score_matrix).fill_diagonal_(torch.nan).view(-1).sort(descending=True)
            pairs = pairs.indices[pairs.values.isnan() == False][:const.repn.TOP_K]

            boxes = torch.empty(1, 2, 4)
            scores = torch.empty(1, 91)
            features = torch.empty(1, 1024)
            for pair in pairs: 
                idx = (pair // n_pairs, pair % n_pairs)
                box_set = (torch.vstack([prediction['boxes'][idx[0]],
                                         prediction['boxes'][idx[1]]])).unsqueeze(0)

                larger_box_idx = (torch.abs(box_set[0, :, 0] - box_set[0, :, 1]) * torch.abs(box_set[0, :, 0] - box_set[0, :, 1])).argmax()
                features = torch.vstack([features, prediction['features'][idx[larger_box_idx]].unsqueeze(0)])
                scores = torch.vstack([scores, prediction['scores'][idx[larger_box_idx]].unsqueeze(0)])
                boxes = torch.vstack([boxes, box_set])

            idx = [box_iou(boxes[1:, 0], boxes[1:, 1]).diag() < const.repn.IOU_THRESH]
            boxes = boxes[1:][idx]
            scores = scores[1:][idx]
            features = features[1:][idx]

            graphs.append(Graph(boxes, features, scores))
        return graphs
