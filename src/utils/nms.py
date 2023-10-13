#!/usr/bin/env python3

from torchvision.ops.boxes import _box_inter_union
from src import const
import torch


def nms(pairs, boxes, n_pairs):
    trimmed_pairs = []
    ignored_ind = []
    for gold_ind in range(len(pairs)):
        if gold_ind in ignored_ind: continue
        trimmed_pairs.append(pairs[gold_ind])

        for pair, ind in enumerate(pairs[gold_ind+1:]):
            if gold_ind + ind in ignored_ind: continue

            idx = ((pair // n_pairs, pair % n_pairs),
                   (pairs[-1] // n_pairs, trimmed_pairs[-1] % n_pairs))

            box_batch = torch.vstack([boxes[idx[i][j]] for i in range(2) for j in range(2)])
            inter, union = _box_inter_union(box_batch[:2], box_batch[2:])
            iou = inter.sum() / union.sum()

            if iou > const.repn.IOU_THRESH: ignored_ind.append(gold_ind + ind)
    return trimmed_pairs
