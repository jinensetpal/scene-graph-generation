#!/usr/bin/env python3

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
from .gnn import GNN
import torch

class SceneGraphGenerator:
    def __init__(self):
        self.backbone = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        self.gnn = GNN()

    def predict(self, x):
        x = self.backbone(x)
        x = self.preproc(x)
        x = self.gnn(x)
        return self.postproc(x)


if __name__ == '__main__':
    model = SceneGraphGenerator()
    mdoel.eval()

    inp = [torch.rand(3, 300, 400), 
           torch.rand(3, 500, 400)]

    with torch.no_grad():
        print(model.predict(inp))
