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
from spikes.Lenet import LeNet
from spikes import Config

class Compiler:
    def __init__(self, layers, weights, inputs):
        input_size     = inputs[0][0].shape[1]
        input_channels = inputs[0][0].shape[0]

        self.initialization = Initialization(layers, inputs, weights)
        self.processing     = Processing(layers, input_size, input_channels)
        self.memory         = Memory(layers, (input_size, input_channels))
        self.instructions   = Instructions(layers, (input_size, input_channels), self.processing, self.memory)

    def generate_config(self):
        self.processing.generate()
        self.processing.duplicate()
        self.processing.write_to_file()

        self.initialization.layer_scaling_factors()
        self.initialization.write_weight_files()
        self.initialization.write_input_file(0)

        self.memory.generate()

        self.instructions.generate(self.memory.kernel_fit)
        self.instructions.write_to_file()

        self.memory.write_to_file(len(self.instructions.instr))


if __name__ == "__main__":
    network    = LeNet()
    layers     = network.layer_list
    model_path = "spikes/models/0_0" + ("_3,3_qat.pt" if Config.if_pretrained() else ".pt")

    data_train = MNIST('spikes/data', download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))
    data_test  = MNIST('spikes/data', train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]))

    compiler = Compiler(layers, model_path, data_train)
    compiler.generate_config()

    sim = Simulation(layers)
    sim.load_state_dict(torch.load(model_path))
    sim.quantize_weights()
    sim.eval()

    output = sim(torch.stack([image for image, label in data_test]))
    pred = output.detach().max(1)[1]
    total_correct = pred.eq(torch.tensor([label for image, label in data_test]).view_as(pred)).sum()
    print("Correct: {} / {}".format(total_correct, len(data_test)))
