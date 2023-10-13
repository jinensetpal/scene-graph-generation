#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import torch

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 90


@dataclass
class repn:
    TOP_K = 128
    IOU_THRESH = .01
    PROJECTION = 1024  # same as RoI pooling dimensions


@dataclass
class training:
    LEARNING_RATE = 1E-2
