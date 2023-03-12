# SoftTreeMax Policy Gradient 

This repository contains an implementation of the SoftTreeMax Policy Gradient algorithm, as described in the paper:

[SoftTreeMax: Exponential Variance Reduction in Policy Gradient via Tree Expansion (arXiv:2209.13966)](https://arxiv.org/pdf/2301.13236.pdf)

SoftTreeMax is a Reinforcement Learning algorithm that generalizes PPO to tree-expansion (model-based). It builds on NVIDIA [CuLE](https://github.com/NVlabs/cule) [Dalton et al., 2019] for an efficient GPU-based tree expansion.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Docker 19 or newer.
- Access to NVIDIA Docker Catalog. Visit the [NGC website](https://ngc.nvidia.com/signup) and follow the instructions. This will grant you access to the base docker image (from the Dockerfile) and ability to run on NVIDIA GPU using the nvidia runtime flag.


### Installation

Clone the project to get the Dockerfile and build by running 
```
docker build -t stm .
```

### Usage

1. Start the docker: 
   ```
   docker run --runtime=nvidia -it stm /bin/bash
   ```
2. CD to project directory:
    ```
   cd softtreemax
    ```
    
3. Train example:
   ```
   python main.py --env_name=BreakoutNoFrameskip-v4 --tree_depth=2 --run_type=train
   ``` 
   See main.py for additional parameters. 
   
   At the end of the train run, the file name of the saved agent will be printed. 
   By default, the model will be saved into `saved_agents` directory and its name will match the [wandb](https://wandb.com) run id. 
   For example, `saved_agents/qfmve636.zip`.
4. Evaluate example:
    ```
   python main.py --env_name=BreakoutNoFrameskip-v4 --tree_depth=2 --run_type=evaluate --model_filename=saved_agents/qfmve636.zip --n_eval_episodes=20
   ``` 
   At the end of the evaluation run, both episode rewards and lengths will be printed. These include the per-episode 
   vectors of length n_eval_episodes, as well as their averages.
   
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




