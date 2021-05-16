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


def network_type():
    network_id = 0  # 0 for LeNet5, 1 for Mobilenet, 2 for VGG16, 3 for ResNet
    return network_id 
  

def dataset_type():
    dataset_id = 0  # 0 for MINST, 1 for CIFAR10, 2 for CIFAR100
    return dataset_id

def num_class():
    if (dataset_type() == 0): num_of_class = 10
    if (dataset_type() == 1): num_of_class = 10
    if (dataset_type() == 2): num_of_class = 100
    return num_of_class

def device_id():
    device_to_run = 0
    return device_to_run
   
def if_pretrained(): 
    ifPretrained = 0  # 0 for regular, 1 quantization aware training
    return ifPretrained 
     
     
def resolution():   
 
    resActivation = 3
    resWeight = 3
    
    return resActivation, resWeight 


   
