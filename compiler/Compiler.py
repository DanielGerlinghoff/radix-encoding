"""
  Company:     A*STAR IHPC
  Engineer:    Gerlinghoff Daniel
  Create Date: 22/04/2021

  Description: Main program to generate configuration and memory intialization
               from PyTorch network

"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST

from Initialization import Initialization
from Processing import Processing
from Memory import Memory
from Instructions import Instructions
from Simulation import Simulation
from models.Lenet import LeNet5

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
    network    = LeNet5()
    layers     = network.layer_list
    config = {
        "model_path":     "models/Lenet.pt",
        "res_weight":     3,
        "sigma_weight":   3,
        "res_activation": 3,
        "bits_margin":    4,
        "dram_data_bits": 512,
        "dram_addr_bits": 29,
        "memory_limit":   10e6,
        "cu_duplication": 2,
        "input_index":    0
    }

    data_train = MNIST('./data', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    data_test  = MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    compiler = Compiler(layers, data_train, config)
    compiler.run()

    sim = Simulation(layers)
    sim.load_state_dict(torch.load(config["model_path"]))
    sim.quantize_weights()
    sim.eval()

    output = sim(torch.stack([image for image, label in data_test]))
    pred = output.detach().max(1)[1]
    total_correct = pred.eq(torch.tensor([label for image, label in data_test]).view_as(pred)).sum()
    print("Correct: {} / {}".format(total_correct, len(data_test)))
