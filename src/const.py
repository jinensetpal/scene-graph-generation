#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

REPO_NAME = 'ML-Purdue/scene-graph-generation'
DATASOURCE_NAME = 'visualgenome'
DATASET_NAME = 'visualgenome'
BUCKET_NAME = 's3://visualgenome'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_RELATIONS = 1014  # len(set(json.load(open('relationship_synsets.json')).values()))
N_OBJECTS = 5996  # len(set(json.load(open('object_synsets.json')).values()))
SPLITS = {'train': 70,
          'valid': 10,
          'test': 20}
LOSS_ALPHAS = {'relation': 0.01,
               'predicate': 0.01,
               'object': 0.01}
EPOCHS = 10
LEARNING_RATE = 1E-2
BBOX_THRESH = 15
MOMENTUM = 0.9


@dataclass
class backbone:
    N_CLASSES = 90


@dataclass
class repn:
    TOP_K = 128
    IOU_THRESH = .7
    HIDDEN = 512
    PROJECTION = 1024  # same as RoI pooling dimensions


@dataclass
class agcn:
    PROJECTION = 256


@dataclass
class training:
    BATCH_SIZE = 16
    LEARNING_RATE = 1E-2
