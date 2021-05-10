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

from spikes import Config as C
from spikes import RMSE as R
 

class QuantizeThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, quantize_bit, mode):
        return quantize_tensor(input, quantize_bit, mode)

    @staticmethod
    def backward(ctx, grad_output): 
        return grad_output, None, None


def quantize_tensor(tensor_to_quantize, quantize_bit, mode):

    #find the spacing
    tensor_max, tensor_min = torch.max(tensor_to_quantize), torch.min(tensor_to_quantize)
    if (mode == 0 ): scale = 5.12
    else: scale = max(abs(tensor_max.item()), abs(tensor_min.item()))
 
    
    spacing = scale/(2**quantize_bit-1)
    
    #spacing = scale/quantize_bit
    
    #quantize the pruned weight
    tensor_quantized = torch.round(tensor_to_quantize/spacing)*spacing
     
     
     
     
#    if(mode == 0):
#        mean_value = torch.mean(((tensor_quantized- tensor_to_quantize)**2)).item()
#        max_value = torch.max(torch.abs(tensor_quantized- tensor_to_quantize)).item()
#        min_value = torch.min(torch.abs(tensor_quantized- tensor_to_quantize)).item()
#        num_elements = torch.numel(tensor_to_quantize)
#        #print (mean_value.item(), max_value.item(), min_value.item())
#        R.mean_value = (R.mean_value* R.num_elements +mean_value* num_elements)/(R.num_elements + num_elements)
#        R.num_elements += num_elements
#        if (R.min_value > min_value): R.min_value= min_value
#        if (R.max_value < max_value): R.max_value= max_value

       
    return tensor_quantized
    

class Conv2dPrune(nn.Conv2d):
    #overwrite the forward function for pruning
    def forward(self, input):

        res_activation, res_weight = C.resolution()
        self.input_quantized =   QuantizeThrough.apply(input, res_activation, 0) 
        self.weight_quantized =  QuantizeThrough.apply(self.weight, res_weight-1, 1) 
        if (C.if_pretrained()):
            return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            #return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)       
            return F.conv2d(self.input_quantized, self.weight_quantized, self.bias, self.stride, self.padding, self.dilation, self.groups)  
 ########################modify weight back to weight_quantized
 
class LinearPrune(nn.Linear):
    #overwrite the forward function for pruning
    def forward(self, input):
        res_activation, res_weight = C.resolution()
        self.input_quantized =   QuantizeThrough.apply(input, res_activation, 0) 
        self.weight_quantized =  QuantizeThrough.apply(self.weight, res_weight-1, 1) 
        if (C.if_pretrained()):
            return F.linear(input, self.weight, self.bias)
        else:
            #return F.linear(input, self.weight, self.bias)
            return F.linear(self.input_quantized, self.weight_quantized, self.bias)


