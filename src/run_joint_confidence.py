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
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser(description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=128, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=100, help='how many batches to wait before logging training status')
parser.add_argument('--dataset', required=True, default='Din1', help='In-distribution dataset')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='./outf/ganmodel', help='folder to output images and model checkpoints')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='60', help='decreasing strategy')
parser.add_argument('--beta', type=float, default=1, help='penalty parameter for KL term')
parser.add_argument('--synth_data', type=str, default='None', help='synthetic WGAN data: "10", "20", "70"')

args = parser.parse_args()
if args.synth_data == "None":
    args.synth_data = ''

#if args.dataset == 'Din1':
#    args.beta = 0.1
#    args.batch_size = 128

if args.dataset == 'Din1':
    num_classes = 2
elif args.dataset == 'Din2':
    num_classes = 3

synth_data_string = "_synth"+str(args.synth_data) if args.synth_data != '' else ''
args.outf = '/work3/s202464/master-thesis/src/outf/joint'+'_'+args.dataset+'_'+str(num_classes) + synth_data_string

if not os.path.exists(args.outf):
    os.mkdir(args.outf)

print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('load data: ',args.dataset)
train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, 
        args.imageSize, os.path.join(args.dataroot, args.dataset), synth_data = args.synth_data)

print('Load model')
model = models.vgg13(num_classes=num_classes)
#print(model)

print('load GAN')
nz = 100
netG = models.Generator(1, nz, 64, 3) # ngpu, nz, ngf, nc
netD = models.Discriminator(1, 3, 64) # ngpu, nc, ndf
# Initial setup for GAN
real_label = 1
fake_label = 0
criterion = nn.BCELoss()
fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)

if args.cuda:
    model.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    fixed_noise = fixed_noise.cuda()
fixed_noise = Variable(fixed_noise)

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
#optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        gan_target = torch.FloatTensor(target.size()).fill_(0)
        uniform_dist = torch.Tensor(data.size(0), num_classes).fill_((1./num_classes))

        if args.cuda:
            data, target = data.cuda(), target.cuda()
            gan_target, uniform_dist = gan_target.cuda(), uniform_dist.cuda()

        data, target, uniform_dist = Variable(data), Variable(target), Variable(uniform_dist)

        #### Discriminator update ####
        # Pass real images and backward propagate
        gan_target.fill_(real_label)
        targetv = Variable(gan_target)
        optimizerD.zero_grad()
        output = netD(data)
        errD_real = criterion(output, targetv.reshape(-1,1))
        errD_real.backward()
        D_x = output.data.mean()

        # Generate fake image from noise and update Discriminator weights
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        targetv = Variable(gan_target.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, targetv.reshape(-1,1))
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        
        #### Generator Update ####
        optimizerG.zero_grad()
        # Compute the original Generator loss
        targetv = Variable(gan_target.fill_(real_label))  
        output = netD(fake)
        errG = criterion(output, targetv.reshape(-1,1))
        D_G_z2 = output.data.mean()

        # Compute the added KL divergence loss and update Generator weights
        KL_fake_output = F.log_softmax(model(fake))
        errG_KL = F.kl_div(KL_fake_output, uniform_dist)*num_classes
        generator_loss = errG + args.beta*errG_KL
        generator_loss.backward()
        optimizerG.step()

        #### Classifier update ####
        # original cross entropy loss
        optimizer.zero_grad()
        output = F.log_softmax(model(data))
        loss = F.nll_loss(output, target)

        # Compute KL divergence loss term for the classifier and update classifier
        noise = torch.FloatTensor(data.size(0), nz, 1, 1).normal_(0, 1).cuda()
        if args.cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        fake = netG(noise)
        KL_fake_output = F.log_softmax(model(fake))
        KL_loss_fake = F.kl_div(KL_fake_output, uniform_dist)*num_classes
        total_loss = loss + args.beta*KL_loss_fake
        total_loss.backward()
        optimizer.step()

        ## Print stats
        if batch_idx % args.log_interval == 0:
            print('Classification Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, KL fake Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item(), KL_loss_fake.data.item()))
            fake = netG(fixed_noise)
            vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(args.outf, epoch), normalize=True)

def test(epoch):
    '''
    Function for evaluating test set performance per epoch.
    '''
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
    return (correct/total)

#################
### Main loop ###
#################
acc = 0.0
for epoch in range(1, args.epochs + 1):
    train(epoch)
    acc_ = test(epoch)
    if epoch in decreasing_lr:
        optimizerG.param_groups[0]['lr'] *= args.droprate
        optimizerD.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    if acc_ > acc:
        # do checkpointing
        print('...saving checkpoint...')
        torch.save(netG.state_dict(), '%s/netG.pth' % args.outf)
        torch.save(netD.state_dict(), '%s/netD.pth' % args.outf)
        torch.save(model.state_dict(), '%s/model.pth' % args.outf)
        acc = acc_
