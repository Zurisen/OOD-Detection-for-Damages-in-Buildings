from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import numpy as np
import torchvision.utils as vutils
import models
import pandas as pd
import os

from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='VGG13 Training Code')
parser.add_argument('--batch-size', type=int, default=128, help='Batch Size for training')
parser.add_argument('--epochs', type=int, default=60, help='Number of training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='log info after x batches')
parser.add_argument('--dataset', required=True, default='Din1', help='Training datasets "Din1" or "Din2"')
parser.add_argument('--dataroot', required=True, help='Specify path to training dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the training images')
parser.add_argument('--outf', default='./outf/basicmodel', help='Checkpoint saving folder')
parser.add_argument('--wd', type=float, default=0.0, help='Weight Decay')
parser.add_argument('--droprate', type=float, default=0.0, help='Learning rate decay')
parser.add_argument('--decreasing_lr', default='2', help='decreasing learning rate strategy')
parser.add_argument('--synth_data', type=str, default='None', help='synthetic WGAN data: "10", "20", "70"')


args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.synth_data == "None":
    args.synth_data = ''

if args.dataset == 'Din1':
    num_classes = 2
elif args.dataset == 'Din2':
    num_classes = 3

synth_data_string = "_synth"+str(args.synth_data) if args.synth_data != '' else ''
#args.outf = '/work3/s202464/master-thesis/src/outf/ce'+'_'+args.dataset+'_'+str(num_classes) + synth_data_string

if not os.path.exists(args.outf):
    os.mkdir(args.outf)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size,
        args.imageSize, os.path.join(args.dataroot,args.dataset), synth_data = args.synth_data)

print('Load model')
model = models.vgg13(num_classes=num_classes)
#print(model)

if args.cuda:
    model.cuda()

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr)#, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    '''Main training loop'''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test(epoch):
    '''Test Model'''
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for data, target in test_loader:
        total += data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = F.log_softmax(model(data))
        test_loss += F.nll_loss(output, target).data.item()
        pred = torch.argmax(output.data, dim=1) # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    
    return correct/total, test_loss

#################
### Main loop ###
#################
acc = 0.0
acc_arr = np.array([])
loss_arr = np.array([])
for epoch in range(1, args.epochs + 1):
    train(epoch)
    metrics_df = pd.DataFrame()
    acc_, loss_ = test(epoch)
    acc_arr = np.append(acc_arr, acc_)
    loss_arr = np.append(loss_arr, loss_)
    metrics_df['acc'] = acc_arr
    metrics_df['loss'] = loss_arr
    metrics_df.to_csv(args.outf+'/metrics.csv', index=False)
    if epoch in decreasing_lr:
        optimizer.param_groups[0]['lr'] *= args.droprate
    if acc_ > acc:
        # do checkpointing
        print('...saving checkpoint...')
        torch.save(model.state_dict(), '%s/model.pth' % args.outf)
        acc = acc_
