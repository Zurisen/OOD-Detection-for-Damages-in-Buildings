from __future__ import print_function
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import os
import models
import data_loader
import calculate_log as callog
from scipy import misc
import argparse

parser = argparse.ArgumentParser(description='Test code - measure OOD detection performance using ODIN')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataset', required=True, help='target dataset: Din1')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height/width of the input image to network')
parser.add_argument('--outf', default='./outf_res/unnamedODIN', help='folder to output the softmax results')
parser.add_argument('--out_dataset', required=True, help='out-of-dist dataset: Dout1')
parser.add_argument('--num_classes', required=True, type=int, default=3, help='number of classes')
parser.add_argument('--pre_trained_net', required=True, default='', help='path to pre trained net')
parser.add_argument('--noise', type=float, default=0.0014, help='ODIN noise')
parser.add_argument('--temperature', type=int, default=1000, help='ODIN temperature')

args = parser.parse_args()
print(args)
args.cuda = not args.no_cuda and torch.cuda.is_available()
print("Random Seed: ", args.seed)
torch.manual_seed(args.seed)

if not os.path.exists(args.outf):
    os.mkdir(args.outf)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers':1, 'pin_memory': True} if args.cuda else {}

print('Load model')
model = models.vgg13(num_classes=args.num_classes)
model.load_state_dict(torch.load(args.pre_trained_net))
#print(model)

print('load target data: ', args.dataset)
_, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size,
        args.imageSize, os.path.join(args.dataroot, args.dataset))

print('load non target data: ', args.out_dataset)
nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size,
        args.imageSize, os.path.join(args.dataroot, args.out_dataset))

if args.cuda:
    model.cuda()


def generate_ODIN(net1, path, testloader10, testloader, noiseMagnitude1=0.0014, temper=1000):
    net1.eval()
    criterion = nn.CrossEntropyLoss()
    CUDA_DEVICE = 'cuda'
    t0 = time.time()
    g1 = open(os.path.join(path, "confidence_ODIN_In.txt"), 'w')
    g2 = open(os.path.join(path, "confidence_ODIN_Out.txt"), 'w')
    N = 10000
    print("Processing in-distribution images")
########################################In-distribution###########################################
    for j, data in enumerate(testloader10):
        #if j<1000: continue
        images, _ = data
        
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        
        # Using temperature scaling
        outputs = outputs / temper
	
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        #g1.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        g1.write("{}\n".format(np.max(nnOutputs)))
        #if j % 100 == 99:
        #    print("{:4} images processed, {:.1f} seconds used.".format(j+1, time.time()-t0))
        #    t0 = time.time()
        
        if j == N - 1: break


    t0 = time.time()
    print("Processing out-of-distribution images")
###################################Out-of-Distributions#####################################
    for j, data in enumerate(testloader):
        #if j<1000: continue
        images, _ = data
    
        inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad = True)
        outputs = net1(inputs)
        
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))

        # Using temperature scaling
        outputs = outputs / temper
  
        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Normalizing the gradient to binary in {0, 1}
        gradient =  (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0] )/(63.0/255.0)
        gradient[0][1] = (gradient[0][1] )/(62.1/255.0)
        gradient[0][2] = (gradient[0][2])/(66.7/255.0)
        # Adding small perturbations to images
        tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
        outputs = net1(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs)/np.sum(np.exp(nnOutputs))
        g2.write("{}\n".format(np.max(nnOutputs)))
        #g2.write("{}, {}, {}\n".format(temper, noiseMagnitude1, np.max(nnOutputs)))
        #if j % 100 == 99:
        #    print("{:4} images processed, {:.1f} seconds used.".format(j+1, time.time()-t0))
        #    t0 = time.time()

        if j== N-1: break

generate_ODIN(model, args.outf, test_loader, nt_test_loader, 
        noiseMagnitude1=args.noise, temper=args.temperature)
callog.metric(args.outf,method='odin')
