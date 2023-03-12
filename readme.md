# SoftTreeMax Policy Gradient 

This repository contains an implementation of the SoftTreeMax Policy Gradient algorithm, as described in the paper:

[SoftTreeMax: Exponential Variance Reduction in Policy Gradient via Tree Expansion (arXiv:2209.13966)](https://arxiv.org/pdf/2301.13236.pdf)

SoftTreeMax is a Reinforcement Learning algorithm that generalizes PPO to tree-expansion (model-based). It builds on NVIDIA [CuLE](https://github.com/NVlabs/cule) [Dalton et al., 2019] for an efficient GPU-based tree expansion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker 19 or newer.
- Access to NVIDIA Docker Catalog. Visit the [NGC website](https://ngc.nvidia.com/signup) and follow the instructions. This will grant you access to the base docker image (from the Dockerfile) and ability to run on NVIDIA GPU using the nvidia runtime flag.


### Installing

Clone the project to get the Dockerfile and build by running 
```
docker build -t stm .
```

### Usage

1. Start the docker: 
   ```
   docker run --runtime=nvidia -it stm /bin/bash
   ```
3. Run the code: 
   ```
   cd softtreemax; python main.py
   ``` 
   See main.py for optional parameters. For example, for tree depth 2 run: 
   ```
   python main.py --tree-depth=2
   ```

## License

This project is licensed under the NVIDIA License.

## Acknowledgments

If you use this project please cite:
```
@article{dalal2023softtreemax,
  title={SoftTreeMax: Exponential Variance Reduction in Policy Gradient via Tree Expansion},
  author={Dalal, Gal and Hallak, Assaf and Thoppe, Gugan and Mannor, Shie and Chechik, Gal},
  journal={arXiv preprint arXiv:2301.13236},
  year={2023}
}
```




