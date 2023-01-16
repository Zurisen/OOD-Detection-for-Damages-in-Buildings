#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:38:06 2022

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
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # final state size. 3 x 64 x 64
        )

    def forward(self, noise):
        x = self.model(noise.view(len(noise), self.z_dim, 1, 1))
        return x
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
    
    
def gen_noise(num, z_dim, device='cuda'):
    return torch.randn(num, z_dim, device=device) # 128 x 200

class train_dcgan():
    def __init__(self, train_dataloader, test_dataloader, z_dim, tag):
        
        ## Class variables
        self.gen_losses = []
        self.crit_losses = []
        self.fids = []
        self.iss = []
        self.cur_step = 0
        self.epoch = 0
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.z_dim = z_dim
        self.crit_cycles = 5
        self.image_size = 128
        self.n_epochs = 30
        self.show_step = 100
        self.tag = tag
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ## Initialize Generator and Discriminator (Critic) classes 
        self.gen = Generator(z_dim=z_dim).to(self.device)
        self.crit = Critic().to(self.device)
        #self.gen=self.gen.apply(init_weights)
        #self.crit=self.crit.apply(init_weights)
        
        ## Different optimizer options for generator and discriminator
        self.crit_opt = optim.Adam(self.crit.parameters(),
                                   lr=2e-4, betas=(0.5, 0.999))
        self.gen_opt = optim.Adam(self.gen.parameters(),
                                  lr=2e-4, betas=(0.5, 0.999))
        self.criterion = nn.BCELoss()
        
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
        print(f"Epoch: {self.epoch}: Step {self.cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
    
    def evaluate_fid_is(self):
        self.evaluator.run(self.test_dataloader,max_epochs=1)
        metrics = self.evaluator.state.metrics
        fid_score = metrics['fid']
        self.fids += [fid_score]
        is_score = metrics['is']
        self.iss += [is_score]
        print(f"*   FID : {fid_score:4f}")
        print(f"*    IS : {is_score:4f}")
        
    def save_checkpoint(self):
        torch.save({
          'epoch': self.epoch,     
          'model_state_dict': self.gen.state_dict(),
          'optimizer_state_dict': self.gen_opt.state_dict()      
        }, f"{self.root_path}Generator.pkl")
    
        torch.save({
          'epoch': self.epoch,
          'model_state_dict': self.crit.state_dict(),
          'optimizer_state_dict': self.crit_opt.state_dict()      
        }, f"{self.root_path}Critic.pkl")
        
        torch.save({
          'epoch': self.epoch,
          'crit_losses': self.crit_losses,
          'gen_losses':self.gen_losses,
          'fids': self.fids,
          'iss': self.iss
        }, f"{self.root_path}Metrics.pkl")      
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
        real_label = 1
        fake_label = 0
        for self.epoch in range(self.n_epochs):
            for real, _ in tqdm(self.train_dataloader):
                cur_bs= len(real) #128
                real=real.to(self.device)
                self.crit_opt.zero_grad()
                # Format batch
                b_size = real.size(0)
                label = torch.full((b_size,), real_label,
                                   dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output1 = self.crit(real).view(-1)
                # Calculate loss on all-real batch
                crit_loss_real = self.criterion(output1, label)
                # Calculate gradients for D in backward pass
                crit_loss_real.backward()
        
                ## Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.z_dim, 1, 1, 
                                    device=self.device)
                # Generate fake image batch with G
                fake = self.gen(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output2 = self.crit(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                crit_loss_fake = self.criterion(output2, label)
                # Calculate the gradients for this batch, accumulated (summed) with previous gradients
                crit_loss_fake.backward()
                # Compute error of D as sum over the fake and the real batches
                crit_loss = crit_loss_real + crit_loss_fake
                # Update D
                self.crit_opt.step()
                
                self.crit_losses += [crit_loss.item()]
        
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.gen_opt.zero_grad()
                label.fill_(real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output3 = self.crit(fake).view(-1)
                # Calculate G's loss based on this output
                gen_loss = self.criterion(output3, label)
                # Calculate gradients for G
                gen_loss.backward()
                # Update G
                self.gen_opt.step()
                
                self.gen_losses+=[gen_loss.item()]
        
                ### STATS  
                if (self.cur_step % self.show_step == 0 and self.cur_step > 0):
                    self.evaluate_fid_is()
                    show(fake, path=self.root_path, name=str(self.epoch))
                    #show(real, path=self.root_path, name=str(self.epoch))

                    self.training_losses()
                    
                    print("Saving checkpoint...")
                    self.save_checkpoint()
        
                self.cur_step+=1
