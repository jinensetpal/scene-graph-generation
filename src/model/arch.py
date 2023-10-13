#!/usr/bin/env python3

from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from ..data.visualgenome import Dataset
from IPython import embed
from .repn import RePN
from .gnn import GNN
from .. import const
import torch


class SceneGraphGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.repn = RePN()
        # self.gnn = GNN()

    def forward(self, x):
        x = self.backbone(x)
        x = self.repn(x)
        # x = self.gnn(x)
        # return self.postproc(x),
        return x


if __name__ == '__main__':
    dataset = Dataset()
    model = SceneGraphGenerator().to(const.DEVICE)
    model.eval()

    with torch.no_grad():
        y_pred = model(dataset[0].unsqueeze(0).to(const.DEVICE))
    embed()
