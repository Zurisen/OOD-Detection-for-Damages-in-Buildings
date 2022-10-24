import torch
from PIL import Image, ExifTags
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import glob
import numpy.random as nr
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class DinDataset(torch.utils.data.Dataset):
    def __init__(self, train, image_size, class_onehot, 
            data_path='../data/Din1', trans=None, synth_data=''):
        'Initialization'
        if train:
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        self.class_onehot = class_onehot
        self.transform = trans

        if trans == None:
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), 
                                        transforms.ToTensor()]
                                        )
        self.image_paths = np.array(glob.glob(data_path+'/*/*.jpg'))
        
        if train and (synth_data != ''):
            for damage in class_onehot:
                self.image_paths = np.append(self.image_paths,
                        glob.glob(f'../data/WGAN_Din{synth_data}/'+damage+'/*.jpg'))


    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        
        X = self.transform(image)
        y = torch.tensor(self.class_onehot[image_path.split('/')[-2]])
        return X, y


def getDin1(batch_size, img_size=32, data_root='../data/Din1', train=True, val=True, synth_data='', **kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building Din data loader with {} workers".format(num_workers))
    ds = []
    class_onehot = {"nodmg":0, "cracks":1}
    if train:
        #trainset1 = DinDataset(True, img_size, class_onehot, 
        #        data_root, trans=transforms.Compose([
        #            transforms.Resize((img_size, img_size)),
        #            transforms.RandomRotation(degrees=(90)),
        #            transforms.ToTensor()
        #            ]))
        trainset2 = DinDataset(True, img_size, class_onehot, data_root,
                synth_data=synth_data)
        trainset3 = DinDataset(True, img_size, class_onehot, 
                data_root, trans=transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.GaussianBlur(3),
                    transforms.ToTensor()
                    ]),
                synth_data=synth_data)
 
        train_loader = torch.utils.data.DataLoader(trainset2+trainset3,
                batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(DinDataset(False, img_size, class_onehot, data_root),
                batch_size=batch_size, shuffle=False, **kwargs)
        
        ds.append(test_loader)
    ds = ds[0] if len(ds)==1 else ds
    return ds


def getDin2(batch_size, img_size=32, data_root='../data/Din2', train=True, val=True, synth_data='',**kwargs):
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building Din data loader with {} workers".format(num_workers))
    ds = []
    class_onehot = {"nodmg":0, "cracks":1, "spalling":2}
    if train:
        #trainset1 = DinDataset(True, img_size, class_onehot, 
        #        data_root, trans=transforms.Compose([
        #            transforms.Resize((img_size, img_size)),
        #            transforms.RandomRotation(degrees=(90)),
        #            transforms.ToTensor()
        #            ]))
        trainset2 = DinDataset(True, img_size, class_onehot, data_root,
                synth_data=synth_data)
        trainset3 = DinDataset(True, img_size, class_onehot, 
                data_root, trans=transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.GaussianBlur(3),
                    transforms.ToTensor()
                    ]),
                synth_data=synth_data)
 
        train_loader = torch.utils.data.DataLoader(trainset2+trainset3,
                batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(DinDataset(False, img_size, class_onehot, data_root),
                batch_size=batch_size, shuffle=False, **kwargs)
        
        ds.append(test_loader)
    ds = ds[0] if len(ds)==1 else ds
    return ds


def getCIFAR10(batch_size, img_size=32, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getTargetDataSet(data_type, batch_size, imageSize, dataroot, synth_data=''):
    if data_type == 'Din1':
        train_loader, test_loader = getDin1(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1, synth_data=synth_data)
    if data_type == 'Din2':
        train_loader, test_loader = getDin2(batch_size=batch_size, img_size=imageSize,data_root=dataroot, num_workers=1, synth_data=synth_data)
    return train_loader, test_loader


def getNonTargetDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'Din1':
        _, test_loader = getDin1(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)
    
    if data_type == 'Din2':
        _, test_loader = getDin2(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)

    elif data_type == 'Dout1':
        testsetout = datasets.ImageFolder(dataroot, transform= transforms.Compose(
            [transforms.Resize((imageSize, imageSize)),
             transforms.ToTensor()]
            ))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    
    elif data_type == 'Dout2':
        testsetout = datasets.ImageFolder(dataroot, transform= transforms.Compose(
            [transforms.Resize((imageSize, imageSize)),
             transforms.ToTensor()]
            ))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)

    return test_loader

