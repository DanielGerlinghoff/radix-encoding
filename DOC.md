# Documentation Notes for End-to-End Framework for Accelerating Spiking Neural Networks with Emerging Neural Encoding


## Conventions
* *Kernels* refer to parameters used in convolution layers, while *Weights* are for linear layers


## Processor
### Configuration Parameters
| Parameter      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| `CPAR`         | Parallel output channels for convolution unit                               |
| `CSTR`         | Stride for convolution unit                                                 |
| `CPAD`         | Padding for convolution unit                                                |
| `PPAR`         | Parallel output channels for pooling unit                                   |
| `LCHN`         | Linear layer input channels                                                 |
| `LRELU`        | Linear layer apply ReLU or not                                              |
| `OUTM`         | Output logic mode for storing partial sums: direct, delete, add             |
| `SCL`          | Radix point position for activations                                        |
| `KSEL`, `WSEL` | Selection of BRAM to fetch kernels and weights                              |
| `WADR`         | BRAM address to start reading weights of a linear layer                     |
| `DADR`         | Start address for reading weights or kernels from DRAM                      |
| `ASELR`        | Selection of ping-pong buffer to read from                                  |
| `ASELW`        | Selection of ping-pong buffer to write to                                   |
| `ASTPF`        | Address step forward to separate time steps while writing ping-pong buffer  |
| `ASTPB`        | Address step backward to reset address for next channel                     |
| `ASRC`         | Source address of activations for linear layer                              |
| `ADST`         | Destination address of partial sums for linear layer                        |


### Convolution Unit Execution Sequence
TODO


### Pooling Unit Execution Sequence
1. Configure `PPAR`, `ASELR`, `ASELW`, `ASTPF`, `ASTPB`
2. Enable pooling unit
3. For each group of parallel-processed channels:
    1. For every row in the channel:
        1. Load the activation row of all parallel channels and all time steps
        2. Reset to zero partial sums
        3. Set ping-pong write base address using the `ACTS` instruction
        4. Start processing one activation row
        5. Configure `OUTM` to discard every other row
        6. Write non-discarded output rows to the ping-pong buffer after processing finished
        7. Wait for write to finish


### Linear Unit Execution Sequence
1. Configure `LCHN`, `SCL`, `LRELU`, `ASELR`, `ASELW`, `WSEL`, `ASTPF` and `ASTPB`
2. Enable linear processing unit
3. For each group of parallel-processed output features:
    1. Reset to zero partial sums
    2. For every time step:
        1. Configure `WADR`, `ASRC` and `ADST`
        2. Iterate through all input features starting from `ASRC`, simultaneously loading weights starting from `WADR`
        3. Apply activation function (if `LRELU`) and write result to `ADST` in ping-pong buffer
    3. Wait for write to finish
4. Disable linear processing unit
