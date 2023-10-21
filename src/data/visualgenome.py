#!/usr/bin/env python3

from glob import glob
from .. import const
import torchvision
import random
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.paths = random.choice(glob(str(const.DATA_DIR / 'images' / '*')))

    def len(self):
        return 1

    def __getitem__(self, idx):
        return torchvision.io.read_image(self.paths[idx]) / 255
