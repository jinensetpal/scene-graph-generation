#!/usr/bin/env python3

import torch

class Node:
    def __init__(self, node_id, features, scores, bbox):
        self.bbox = bbox
        self.scores = scores
        self.features = features

        self.node_id = node_id
        self.edges = {}

    def associate(self, node):
        pass

class Graph:
    def __init__(self, features, scores, boxes):

    def 

