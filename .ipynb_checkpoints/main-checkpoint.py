#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:48:23 2022

@author: cmt
"""

import torch, torchvision, os, PIL, pdb
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import glob
from PIL import Image, ExifTags
import colorsys
import random
import pylab
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.optim as optim
from tqdm.notebook import tqdm
import os
import random
from IPython import display
import ignite
from ignite.engine import Engine, Events
import ignite.distributed as idist
from ignite.metrics import FID, InceptionScore
from ignite.contrib.handlers import ProgressBar
import warnings
import sys
from utils import *
from dataloaders import image_loader
from WGAN import *
from DCGAN import *


NET = sys.argv[1]
TAG = sys.argv[2]
DATA_PATH = sys.argv[3]

if __name__ == "__main__":
    
    image_size = 128 if (NET=="WGAN") else 64
    z_dim = 200 if (NET=="WGAN") else 100
    batch_size = 64
    TAG = NET + '_' + TAG    

    trainset = image_loader(train=True, image_size=image_size,
                            data_path=DATA_PATH)
    train_dataloader = DataLoader(trainset, shuffle=True, batch_size=batch_size,
                                  num_workers=12, drop_last=True)
    
    testset = image_loader(train=False, image_size=image_size,
                           data_path=DATA_PATH)
    test_dataloader = DataLoader(testset, batch_size=batch_size,
                                 num_workers=12, drop_last=True)
    
    if NET == "WGAN":
        wgan = train_wgan(train_dataloader, test_dataloader, z_dim, TAG)
        wgan.run()
    elif NET == "DCGAN":
        dcgan = train_dcgan(train_dataloader, test_dataloader, z_dim, TAG)
        dcgan.run()
    else:
        raise NotImplementedError("Unavailable GAN")
    
    
    
    
