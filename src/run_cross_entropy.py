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
import os

from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='Training code - cross entropy')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--dataset', required=True, default='Din1', help='Din1')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='./outf/basicmodel', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
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
args.outf = '/work3/s202464/master-thesis/src/outf/ce'+'_'+args.dataset+'_'+str(num_classes) + synth_data_string

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
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
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

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))


for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch in decreasing_lr:
        optimizer.param_groups[0]['lr'] *= args.droprate
    test(epoch)
    if epoch % 20 == 0:
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))
