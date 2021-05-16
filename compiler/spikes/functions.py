import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys 

import Config as C
import Resnet as res
import Resnet2 as res2
import VGG as vgg
import VGG2 as vgg2
import Mobilenet as mobile
import Lenet as le
import RMSE as R
 
from torch.autograd import Variable
from torchvision.datasets.mnist import MNIST
from torch.utils.data import DataLoader
from torch.distributions import Categorical
from torch.distributions import Normal
from collections import namedtuple

   
#flag for network and dataset types
network_type = C.network_type()
dataset_type = C.dataset_type()
device_id = C.device_id()

#load data set
data_train = MNIST('./data', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
data_test = MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
# cifar_10_trainset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
# cifar_10_testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
# cifar_100_trainset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=True, download=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
# cifar_100_testset = torchvision.datasets.CIFAR100(root='./data/cifar100', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))




#set the dataloader based on the flag value
if (dataset_type == 0):
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(data_test, batch_size=2048, num_workers=8)
elif (dataset_type == 1):
    data_train_loader = DataLoader(cifar_10_trainset, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(cifar_10_testset, batch_size=1024, shuffle=False, num_workers=8)
elif (dataset_type == 2):
    data_train_loader = DataLoader(cifar_100_trainset, batch_size=128, shuffle=True, num_workers=8)
    data_test_loader = DataLoader(cifar_100_testset, batch_size=1024, shuffle=False, num_workers=8)
 



def train(epoch):
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):      
        optimizer.zero_grad()
        images, labels=images.cuda(device_id), labels.cuda(device_id)
        images, labels=Variable(images), Variable(labels)
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    print('Train - Epoch %d, Loss: %f' % (epoch, loss.detach().cpu().item()))

def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels=images.cuda(device_id), labels.cuda(device_id)
            images, labels=Variable(images), Variable(labels)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
    return float(total_correct) / len(data_test)



def adjust_learning_rate(optimizer, epoch):   
    if (network_type == 0):  base_lr = 0.01
    if (network_type == 1):  base_lr = 0.01
    if (network_type == 2):  base_lr = 0.05  
    if (network_type == 3):  base_lr = 0.05
    if (network_type == 5):  base_lr = 0.05
    if (C.if_pretrained() == 0):  base_lr = base_lr  *0.5 *0.5 *0.5 *0.5
    lr = base_lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#setting the target network
if (network_type == 0): net = le.LeNet()
elif (network_type == 1): net = mobile.Mobile()
elif (network_type == 2): net = vgg.VGGs(16)
elif (network_type == 3): net = res.Resnet(18)
elif (network_type == 5): net = res2.Resnet2(44)
elif (network_type == 6): net = vgg2.VGGs(9)
#define net property
net.cuda(device_id)
criterion = nn.CrossEntropyLoss()
if (network_type == 0):  optimizer = optim.Adam(net.parameters(), 0.01)
if (network_type == 1):  optimizer = optim.Adam(net.parameters(), 0.01)
if (network_type == 2):  optimizer = optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)
if (network_type == 3):  optimizer = optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)
if (network_type == 5):  optimizer = optim.SGD(net.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
if (network_type == 6):  optimizer = optim.SGD(net.parameters(), 0.05, momentum=0.9, weight_decay=5e-4)

num_epoch = 50
if __name__ == '__main__':
    if C.if_pretrained():
        net.load_state_dict(torch.load('models/' + str(network_type) + '_' + str(dataset_type) + '.pt'))

    for epoch in range(num_epoch):
        adjust_learning_rate(optimizer, epoch)
        test()
        train(epoch)

    if not C.if_pretrained():
        torch.save(net.state_dict(), 'models/' + str(network_type) + '_' + str(dataset_type) + '.pt')
    else:
        torch.save(net.state_dict(), 'models/' + str(network_type) + '_' + str(dataset_type) + '_' +
                                     str(C.resolution()[0]) + ',' + str(C.resolution()[1]) + '_qat.pt')
