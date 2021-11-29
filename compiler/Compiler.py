"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 22/04/2021

  Description: Main program to generate configuration and memory intialization
               from PyTorch network

"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import mnist, CIFAR10, CIFAR100
import os

from Initialization import Initialization
from Processing import Processing
from Memory import Memory
from Instructions import Instructions
from Simulation import Simulation
from training.models import *

class Compiler:
    def __init__(self, layers, inputs, config):
        self.config              = config
        config["input_size"]     = inputs[0][0].shape[1]
        config["input_channels"] = inputs[0][0].shape[0]

        # Initialize compiler modules
        self.processing     = Processing(layers, config)
        self.initialization = Initialization(layers, inputs, config)
        self.memory         = Memory(layers, config)
        self.instructions   = Instructions(layers, config)

        # Link compiler modules
        self.processing.link(self.initialization, self.memory, self.instructions)
        self.initialization.link(self.processing, self.memory, self.instructions)
        self.memory.link(self.processing, self.initialization, self.instructions)
        self.instructions.link(self.processing, self.initialization, self.memory)

    def run(self):
        # Delete previously generated files
        folder = './generated'
        for filename in os.listdir(folder):
            os.remove(os.path.join(folder, filename))

        # Run sub-processes
        self.processing.generate()
        self.processing.duplicate_conv(self.config["cu_duplication"])
        self.processing.write_to_file()

        self.initialization.layer_scaling_factors()
        self.initialization.write_weight_files()
        self.initialization.write_input_file(self.config["input_index"])

        self.memory.generate()

        self.instructions.generate()
        self.instructions.write_to_file()

        self.memory.write_to_file()


if __name__ == "__main__":
    config = {
        "network":        ("lenet", "alexnet", "vgg", "fang")[0],
        "dataset":        ("mnist-32", "mnist-28", "cifar10", "cifar100")[0],
        "model_path":     None,
        "res_weight":     None,
        "sigma_weight":   3,
        "res_activation": None,
        "bits_margin":    4,
        "dram_data_bits": 512,
        "dram_addr_bits": 29,
        "memory_limit":   10e6,
        "cu_duplication": 2,
        "input_index":    0
    }

    if config["dataset"] == "mnist-32":
        trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        data_train = mnist.MNIST('./data', download=True, transform=trans)
        data_test  = mnist.MNIST('./data', train=False, download=True, transform=trans)
    elif config["dataset"] == "mnist-28":
        trans = transforms.Compose([transforms.ToTensor()])
        data_train = mnist.MNIST('./data', download=True, transform=trans)
        data_test  = mnist.MNIST('./data', train=False, download=True, transform=trans)
    elif config["dataset"] == "cifar10":
        trans = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor()])
        data_train = CIFAR10(root='./data', train=True, download=True, transform=trans)
        data_test  = CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    elif config["dataset"] == "cifar100":
        trans = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor()])
        data_train = CIFAR100(root='./data', train=True, download=True, transform=trans)
        data_test  = CIFAR100(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    if config["network"] == "lenet":
        network = LeNet()
        if not config["model_path"]:     config["model_path"] = "training/checkpoint/lenet.pth"
        if not config["res_weight"]:     config["res_weight"] = 3
        if not config["res_activation"]: config["res_activation"] = 4
    elif config["network"] == "alexnet":
        network = AlexNet(batchnorm=True, num_classes=10)
        if not config["model_path"]:     config["model_path"]     = "training/checkpoint/alexnet_bn.pth"
        if not config["res_weight"]:     config["res_weight"]     = 6
        if not config["res_activation"]: config["res_activation"] = 6
    elif config["network"] == "vgg":
        network = VGG("VGG11", batchnorm=True, num_classes=100)
        if not config["model_path"]:     config["model_path"]     = "training/checkpoint/vgg_bn.pth"
        if not config["res_weight"]:     config["res_weight"]     = 6
        if not config["res_activation"]: config["res_activation"] = 6
    elif config["network"] == "fang":
        network = Fang()
        if not config["model_path"]:     config["model_path"]     = "training/checkpoint/fang.pth"
        if not config["res_weight"]:     config["res_weight"]     = 4
        if not config["res_activation"]: config["res_activation"] = 4

    # Compile
    compiler = Compiler(network.layer_list, data_train, config)
    compiler.run()

    # Simulate compiled network
    sim = Simulation(network.layer_list)
    sim.quantize_weights()
    sim.eval()

    output = sim(torch.stack([image for image, label in data_test]))
    pred = output.detach().max(1)[1]
    total_correct = pred.eq(torch.tensor([label for image, label in data_test]).view_as(pred)).sum()
    print("Correct: {} / {}".format(total_correct, len(data_test)))
