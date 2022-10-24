"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
import os
import lib_generation

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='Test code: Mahalanobis detector')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='in-order dataset')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--imageSize', type=int, default=32, help='the height/width of the input image to network')
parser.add_argument('--outf', required=True, default='./outf_res/', help='folder to output results')
parser.add_argument('--pre_trained_net', required=True, default='', help='path to pre-trained net')
parser.add_argument('--out_dataset', required=True, default='', help='ood dataset')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

if args.dataset == 'Din1':
    num_classes = 2
elif args.dataset == 'Din2':
    num_classes = 3


def main():
    # set the path to pre-trained model and output
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
        
    # load networks
    print('Load model')
    model = models.vgg13(num_classes=num_classes).cuda()
    model.load_state_dict(torch.load(args.pre_trained_net))

    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size,
            args.imageSize, os.path.join(args.dataroot, args.dataset))
    
    print('load non target data', args.out_dataset)
    nt_test_loader = data_loader.getNonTargetDataSet(args.out_dataset, args.batch_size,
            args.imageSize, os.path.join(args.dataroot, args.out_dataset))


    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, num_classes, feature_list, train_loader)
    
    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('Noise: ' + str(magnitude))
        for i in range(num_output):
            M_in = lib_generation.get_Mahalanobis_score(model, test_loader, num_classes, args.outf,
                    True, sample_mean, precision, i, magnitude)
            #M_in = np.asarray(M_in, dtype=np.float32)
            #if i == 0:
            #    Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            #else:
            #    Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)
            
        print('Out-distribution: ' + args.out_dataset) 
        for i in range(num_output):
            M_out = lib_generation.get_Mahalanobis_score(model, nt_test_loader, num_classes, args.outf,
                    False, sample_mean, precision, i, magnitude)
                #M_out = np.asarray(M_out, dtype=np.float32)
                #if i == 0:
                #    Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
                #else:
                #    Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)

            #Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
            #Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
            #Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_in)
            #file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset , out_dist))
            #Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
            #np.save(file_name, Mahalanobis_data)
    
if __name__ == '__main__':
    main()
    callog.metric(args.outf, method='mahalanobis')
