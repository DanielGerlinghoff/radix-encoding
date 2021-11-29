# <ins>E</ins>nd-to-<ins>E</ins>nd Framework for Accelerating Spiking Neural Networks with <ins>E</ins>merging <ins>N</ins>eural <ins>E</ins>ncoding (E<sup>3</sup>NE)


## Usage
### Compiler Framework
The compiler framework involves Python scripts for both the training of neural network models and the compilation for FPGA deployment. In the first step, floating-point PyTorch models of the desired network structure are trained. We adopted the training procedure of [kuangliu's repository](https://github.com/kuangliu/pytorch-cifar) for our purpose. In a second step, the model and input data are converted to a spiking neural network (SNN) using radix encoding. Furthermore, memory initialization files and a SystemVerilog package for hardware configuration is created. The following models have been tested with the respective datasets:
* LeNet-5 | MNIST (32x32)
* AlexNet | CIFAR-10
* VGG-11 | CIFAR-100
* CNN after [1] | MNIST (28x28)

After making desired changes in the script file, model training is initiated by executing the following:
```
python3 compiler/training/main.py
```

The compiler allows the adjustment of hardware parameters. It also contains the test settings of the four abovementioned models. Hardware parameter descriptions and script call are shown below. The simulation script is bit-equivalent representation of the execution in the silicon

| Parameter        | Description                                                          |
| ---------------- | -------------------------------------------------------------------- |
| res\_weight      | Number of bits used to represent weights in the network              |
| sigma\_weight    | Parameter *r* for weight quantization (see Eq. 2)                    |
| res\_activation  | Bit resolution of activations, i.e. number of time steps             |
| bits\_margin     | Number of additional bits for storing partial sums to avoid overflow |
| dram\_data\_bits | Width of DRAM data lines                                             |
| dram\_addr\_bits | Width of DRAM address lines                                          |
| memory\_limit    | Available memory for weight storage                                  |
| cu\_duplication  | Duplication of convolution units to increase throughput              |

```
python3 Compiler.py
```

> [1] H. Fang, Z. Mei, A. Shrestha, Z. Zhao, Y. Li and Q. Qiu, "Encoding, Model, and Architecture: Systematic Optimization for Spiking Neural Network in FPGAs," 2020 IEEE/ACM International Conference On Computer Aided Design (ICCAD), 2020, pp. 1-9.


### Hardware Generation and Deployment
TODO


### Execution on Hardware
TODO


## Citation
*D. Gerlinghoff, Z. Wang, X. Gu, R. S. M. Goh and T. Luo, "E3NE: An End-to-End Framework for Accelerating Spiking Neural Networks with Emerging Neural Encoding on FPGAs," in IEEE Transactions on Parallel and Distributed Systems, doi: 10.1109/TPDS.2021.3128945.*

 [Access on *IEEE Xplore*](https://ieeexplore.ieee.org/abstract/document/9619972)
