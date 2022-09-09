#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:00:16 2022

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
#%%

def show(tensor, num=25, path='', name=''):
    '''
    Plots grid of input images given Pytorch images batch.
    '''
    data = tensor.detach().cpu()
    grid = make_grid(data[:num], nrow=5).permute(1,2,0)

    plt.imshow(grid.clip(0,1))
    plt.axis('off')
    plt.savefig(os.path.join(path,name))
    plt.axis('off')
    
    
def get_gp(real, fake, crit, alpha, gamma=10):
    ''' 
    Calculate Gradient Penalty in WGAN.
    '''
    mix_images = real * alpha + fake * (1-alpha) # 128 x 3 x 128 x 128
    mix_scores = crit(mix_images) # 128 x 1

    gradient = torch.autograd.grad(
      inputs = mix_images,
      outputs = mix_scores,
      grad_outputs=torch.ones_like(mix_scores),
      retain_graph=True,
      create_graph=True,
    )[0] # 128 x 3 x 128 x 128

    gradient = gradient.view(len(gradient), -1)   # 128 x 49152
    gradient_norm = gradient.norm(2, dim=1) 
    gp = gamma * ((gradient_norm-1)**2).mean()

    return gp


def gen_noise(num, z_dim, device='cuda'):
    '''
    Generate noise Pytorch tensor.
    '''
    return torch.randn(num, z_dim, device=device) # 128 x 200


def interpolate(batch):
    '''
    Interpolate dataset images to 299x299x3 to be able to work with 
    IS and FID prebuilt functions.
    '''
    arr = []
    for img in batch:
        pil_img = transforms.ToPILImage()(img)
        resized_img = pil_img.resize((299,299), Image.BILINEAR)
        arr.append(transforms.ToTensor()(resized_img))
    return torch.stack(arr)
