#!/usr/bin/env python3

from src.utils import get_proposals
from src import const
import torch


class GraphLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, y_true, y_pred):
        losses = {'relation': [],
                  'predicate': [],
                  'object': []}
        for true_graph, pred_graph in zip(y_true, y_pred):
            true_prop = get_proposals(true_graph)
            pred_prop = get_proposals(pred_graph)

            rel_idx = torch.tensor([pred_prop.index(pred) if pred in pred_prop else -1 for pred in true_prop])

            hit_idx = []
            obj_idx = []
            max_thresh = true_graph['object'].boxes + const.BBOX_THRESH
            min_thresh = true_graph['object'].boxes - const.BBOX_THRESH
            for idx, pred_obj in enumerate(pred_graph['object'].boxes.to(torch.int)):
                try:
                    obj_idx.append(torch.all(torch.cat([pred_obj < max_thresh, min_thresh < pred_obj], dim=1), dim=1).tolist().index(True))
                    hit_idx.append(idx)
                except: pass

            pred_obj = torch.zeros(true_graph['object'].id.shape[0], const.N_OBJECTS)
            pred_rel = torch.zeros(true_graph['relation'].id.shape[0], const.N_RELATIONS)

            for idx, val in zip(torch.where(rel_idx != -1), torch.index_select(pred_graph['relation'].logits, 0, rel_idx[rel_idx != -1])): pred_rel[idx] = val
            for idx, val in zip(hit_idx, torch.index_select(pred_graph['object'].logits, 0, torch.tensor(obj_idx, dtype=torch.int))): pred_obj[idx] = val

            losses['object'].append(self.ce(true_graph['object'].id, pred_obj))
            losses['predicate'].append(self.ce(true_graph['relation'].id, pred_rel))
            losses['relation'].append(self.bce(torch.hstack([torch.ones(rel_idx.shape[0]), torch.zeros(len(true_prop) - rel_idx.shape[0])]),
                                               torch.ones(len(true_prop), dtype=torch.float)).item())

        return [torch.tensor(sum(loss) / len(loss), requires_grad=True) for loss in losses.values()]
