import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
import numpy.random as nr
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class DinDataset(torch.utils.data.Dataset):
    def __init__(self, image_size, data_path='../data_in'):
        'Initialization'
        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                                    )
        self.image_paths = glob.glob(data_path+'/*/*.jpg')
        
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        X = self.transform(image)
        y = torch.tensor(class_onehot[image_path.split('/')[-2]])
        return X, y

def getDin1(batch_size, img_size=32, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building Din data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(DinDataset(img_size, data_root),
                                                   batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_loader.dataset,
                                                                          torch.arange(int(0.1*len(train_loader.dataset))) ),
                                                  batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getTargetDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'Din1':
        train_loader, test_loader = getDin1(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)

    return train_loader, test_loader
