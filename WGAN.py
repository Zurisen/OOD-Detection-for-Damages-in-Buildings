#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:03:34 2022

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
from utils import *
#%%

class Generator(nn.Module):
    def __init__(self, z_dim=64, d_dim=16):
        super(Generator, self).__init__()
        self.z_dim=z_dim

        self.gen = nn.Sequential(
                ## ConvTranspose2d: in_channels, out_channels, kernel_size, stride=1, padding=0
                ## Calculating new width and height: (n-1)*stride -2*padding +ks
                ## n = width or height
                ## ks = kernel size
                ## we begin with a 1x1 image with z_dim number of channels (200)
                nn.ConvTranspose2d(z_dim, d_dim * 32, 4, 1, 0), ## 4x4 (ch: 200, 512)
                nn.BatchNorm2d(d_dim*32),
                nn.ReLU(True),

                nn.ConvTranspose2d(d_dim*32, d_dim*16, 4, 2, 1), ## 8x8 (ch: 512, 256)
                nn.BatchNorm2d(d_dim*16),
                nn.ReLU(True),

                nn.ConvTranspose2d(d_dim*16, d_dim*8, 4, 2, 1), ## 16x16 (ch: 256, 128)
                #(n-1)*stride -2*padding +ks = (8-1)*2-2*1+4=16
                nn.BatchNorm2d(d_dim*8),
                nn.ReLU(True),

                nn.ConvTranspose2d(d_dim*8, d_dim*4, 4, 2, 1), ## 32x32 (ch: 128, 64)
                nn.BatchNorm2d(d_dim*4),
                nn.ReLU(True),            

                nn.ConvTranspose2d(d_dim*4, d_dim*2, 4, 2, 1), ## 64x64 (ch: 64, 32)
                nn.BatchNorm2d(d_dim*2),
                nn.ReLU(True),            

                nn.ConvTranspose2d(d_dim*2, 3, 4, 2, 1), ## 128x128 (ch: 32, 3)
                nn.Tanh() ### produce result in the range from -1 to 1
        )


    def forward(self, noise):
        x=noise.view(len(noise), self.z_dim, 1, 1)  # 128 x 200 x 1 x 1
        return self.gen(x)


class Critic(nn.Module):
    def __init__(self, d_dim=16):
        super(Critic, self).__init__()

        self.crit = nn.Sequential(
          # Conv2d: in_channels, out_channels, kernel_size, stride=1, padding=0
          ## New width and height: # (n+2*pad-ks)//stride +1
          nn.Conv2d(3, d_dim, 4, 2, 1), #(n+2*pad-ks)//stride +1 = (128+2*1-4)//2+1=64x64 (ch: 3,16)
          nn.InstanceNorm2d(d_dim), 
          nn.LeakyReLU(0.2),

          nn.Conv2d(d_dim, d_dim*2, 4, 2, 1), ## 32x32 (ch: 16, 32)
          nn.InstanceNorm2d(d_dim*2), 
          nn.LeakyReLU(0.2),

          nn.Conv2d(d_dim*2, d_dim*4, 4, 2, 1), ## 16x16 (ch: 32, 64)
          nn.InstanceNorm2d(d_dim*4), 
          nn.LeakyReLU(0.2),

          nn.Conv2d(d_dim*4, d_dim*8, 4, 2, 1), ## 8x8 (ch: 64, 128)
          nn.InstanceNorm2d(d_dim*8), 
          nn.LeakyReLU(0.2),

          nn.Conv2d(d_dim*8, d_dim*16, 4, 2, 1), ## 4x4 (ch: 128, 256)
          nn.InstanceNorm2d(d_dim*16), 
          nn.LeakyReLU(0.2),

          nn.Conv2d(d_dim*16, 1, 4, 1, 0), #(n+2*pad-ks)//stride +1=(4+2*0-4)//1+1= 1X1 (ch: 256,1)

        )


    def forward(self, image):
        # image: 128 x 3 x 128 x 128
        crit_pred = self.crit(image) # 128 x 1 x 1 x 1
        return crit_pred.view(len(crit_pred),-1) ## 128 x 1  
  

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias,0)

    if isinstance(m,nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias,0)

class train_wgan():
    def __init__(self, train_dataloader, test_dataloader, z_dim, tag):
        
        ## Class variables
        self.gen_losses = []
        self.crit_losses = []
        self.cur_step = 0
        self.epoch = 0
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.z_dim = z_dim
        self.crit_cycles = 5
        self.image_size = 128
        self.n_epochs = 45
        self.show_step = 100
        self.tag = tag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ## Initialize Generator and Discriminator (Critic) classes 
        self.gen = Generator(z_dim=z_dim).to(self.device)
        self.crit = Critic().to(self.device)
        self.gen=self.gen.apply(init_weights)
        self.crit=self.crit.apply(init_weights)
        
        ## Different optimizer options for generator and discriminator
        self.crit_opt = optim.Adam(self.crit.parameters(),
                                   lr=1e-4, betas=(0.5, 0.9))
        self.gen_opt = optim.Adam(self.gen.parameters(),
                                  lr=1e-4, betas=(0.5, 0.9))
        
        ## Evaluator for metrics
        self.fid_metric = FID(device=self.device)
        self.is_metric = InceptionScore(device=self.device,
                                        output_transform=lambda x: x[0])
        self.evaluator = Engine(self.evaluation_step)
        self.is_metric.attach(self.evaluator, "is")
        self.fid_metric.attach(self.evaluator, "fid")
        
        ## Saving and loading path creation
        if not os.path.exists('./info'):
            os.mkdir('./info')
        if not os.path.exists(f'./info/{self.tag}/'):
            os.mkdir(f'./info/{self.tag}/')
        self.root_path=f'./info/{self.tag}/'
            
    def evaluation_step(self, Engine, batch):
        '''
        Apply interpolation to fake and real batch images.
        '''
        with torch.no_grad():
            noise = gen_noise(len(batch[0]), self.z_dim)
            self.gen.eval()
            fake_batch = self.gen(noise)
            fake = interpolate(fake_batch)
            real = interpolate(batch[0])
            return fake, real
        
    def training_losses(self):
        gen_mean=sum(self.gen_losses[-self.show_step:]) / self.show_step
        crit_mean = sum(self.crit_losses[-self.show_step:]) / self.show_step
        print(f"Epoch: {self.epoch}: Step {self.cur_step}: Generator loss: {self.gen_mean}, critic loss: {self.crit_mean}")
    
    def evaluate_fid_is(self):
        self.evaluator.run(self.test_dataloader,max_epochs=1)
        metrics = self.evaluator.state.metrics
        fid_score = metrics['fid']
        is_score = metrics['is']
        print(f"*   FID : {fid_score:4f}")
        print(f"*    IS : {is_score:4f}")
        
    def save_checkpoint(self):
        torch.save({
          'epoch': self.epoch,
          'gen_losses': self.gen_losses,
          'crit_losses': self.crit_losses,
          'fids': self.evaluator.state.metrics['fid'],
          'iss': self.evaluator.state.metrics['is'],        
          'model_state_dict': self.gen.state_dict(),
          'optimizer_state_dict': self.gen_opt.state_dict()      
        }, f"{self.root_path}Generator.pkl")
    
        torch.save({
          'epoch': self.epoch,
          'crit_losses': self.crit_losses,
          'fids': self.evaluator.state.metrics['fid'],
          'iss': self.evaluator.state.metrics['is'],
          'model_state_dict': self.crit.state_dict(),
          'optimizer_state_dict': self.crit_opt.state_dict()      
        }, f"{self.root_path}Critic.pkl")
      
        print("Saved checkpoint")
    
    def load_checkpoint(self):
        checkpoint = torch.load(f"{self.root_path}Generator.pkl")
        self.gen.load_state_dict(checkpoint['model_state_dict'])
        self.gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
        checkpoint = torch.load(f"{self.root_path}Critic.pkl")
        self.crit.load_state_dict(checkpoint['model_state_dict'])
        self.crit_opt.load_state_dict(checkpoint['optimizer_state_dict'])
    
        print("Loaded checkpoint")
        
    def run(self):
        for self.epoch in range(self.n_epochs):
            for real, _ in tqdm(self.train_dataloader):
                cur_bs= len(real) #128
                real=real.to(self.device)
        
                ### CRITIC
                mean_crit_loss = 0
                for _ in range(self.crit_cycles):
                    self.crit_opt.zero_grad()
        
                    noise=gen_noise(cur_bs, self.z_dim)
                    fake = self.gen(noise)
                    crit_fake_pred = self.crit(fake.detach())
                    crit_real_pred = self.crit(real)
        
                    alpha=torch.rand(len(real),1,1,1,device=self.device,
                                     requires_grad=True) # 128 x 1 x 1 x 1
                    gp = get_gp(real, fake.detach(), crit, alpha)
        
                    crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp
        
                    mean_crit_loss+=crit_loss.item() / self.crit_cycles
        
                    crit_loss.backward(retain_graph=True)
                    self.crit_opt.step()
        
                self.crit_losses+=[mean_crit_loss]
        
                ### GENERATOR
                self.gen_opt.zero_grad()
                noise = gen_noise(cur_bs, self.z_dim)
                fake = self.gen(noise)
                crit_fake_pred = self.crit(fake)
        
                gen_loss = -crit_fake_pred.mean()
                gen_loss.backward()
                self.gen_opt.step()
        
                self.gen_losses+=[gen_loss.item()]
        
                ### STATS  
                if (self.cur_step % self.show_step == 0 and self.cur_step > 0):
                    self.evaluate_fid_is()
                    show(fake, wandbactive=1, name='fake')
                    show(real, wandbactive=1, name='real')
                    self.training_losses()
                    
                    print("Saving checkpoint: ", self.cur_step, self.save_step)
                    self.save_checkpoint()
        
                self.cur_step+=1