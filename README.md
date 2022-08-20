# bte-pytorch
Fully Differentiable Boltzmann Transport Equation solver in PyTorch

## Supported Algorithms

Only support D1V1,D1V2,D1V3 now.

- DVM
    - space: 1,2,WENO5
    - time: RK1/2/3,IMEX-RK1/2/3,...
    - collision: BGK, Binary
- Grad
    - ...
- 13 Moment Equation

[Documents](https://bte-pytorch.readthedocs.io/en/latest/index.html#)

## Installation

```bash
pip install bte-pytorch
```
## Usage

run the examples in `./examples/`
