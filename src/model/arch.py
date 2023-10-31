#!/usr/bin/env python3

from ..data.visualgenome import Dataset
from .backbone import get_backbone
from IPython import embed
from .repn import RePN
from .agcn import aGCN
from .. import const
import torch


class SceneGraphGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = get_backbone()
        self.repn = RePN()
        self.agcn = aGCN()

    def forward(self, x):
        x = self.backbone(x)
        x = self.repn(x)
        return self.agcn(x)


if __name__ == '__main__':
    dataset = Dataset()
    model = SceneGraphGenerator().to(const.DEVICE)
    model.eval()

    with torch.no_grad():
        y_pred = model(dataset[0][0].unsqueeze(0).to(const.DEVICE))
    embed()
