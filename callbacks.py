import numpy as np
import torch

import wandb
import os
import sys

from stable_baselines3.common.callbacks import BaseCallback


class WandbTrainingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(WandbTrainingCallback, self).__init__(verbose)
        self.total_rewards = 0
        self.episode_length = 0
        self.prev_life = 0
        self.warm_start = 0

    def _on_step(self) -> bool:
        # self.total_rewards += np.mean(self.locals["rewards"])
        infos = self.locals["infos"]
        if "orig_reward" in infos:
            self.total_rewards += np.mean(infos["orig_reward"])
        elif type(infos) == list:
            # self.total_rewards += np.mean([info["orig_reward"] for info in infos])
            self.total_rewards += np.mean(infos[0]["orig_reward"])
        self.episode_length += 1

        info = infos[0]
        if "ale.lives" in info:
            ale_lives = int(info["ale.lives"][0]) if type(info["ale.lives"]) == list else int(info["ale.lives"])
            done = ale_lives > self.prev_life
            if done and self.prev_life > 1:
                print("Error: Lives went up but did not reach 0 before!")
            # assert self.prev_life <= 1, "Error: Lives went up but did not reach 0 before!"
            self.prev_life = ale_lives
        else:
            done = int(self.locals["dones"][0])
        if type(done) == np.ndarray:
            done = any(done)
        # if False:
        if done and sys.gettrace() is None:
            cur_step = self.model.num_timesteps + self.warm_start
            # TODO: fix here - depth 0 has no alpha
            if hasattr(self.locals["self"].policy, "alpha"):
                wandb.log({"train\\alpha": self.locals["self"].policy.alpha.item()}, step=cur_step)
            if hasattr(self.locals["self"].policy, "beta"):
                wandb.log({"train\\beta": self.locals["self"].policy.beta.item()}, step=cur_step)
            wandb.log({"train\episodic_reward": self.total_rewards}, step=cur_step)
            wandb.log({"train\episodic_length": self.episode_length}, step=cur_step)
            wandb.log({"num_steps": self.model.num_timesteps}, step=cur_step)
            grad_var = 0
            for param_name in self.locals["self"].policy.gradients_history:
                param_value = self.locals["self"].policy.gradients_history[param_name]
                grad_var += torch.mean(torch.var(torch.stack(param_value), dim=0)).item()
                self.locals["self"].policy.gradients_history[param_name] = param_value[-100:]
            wandb.log({"train\\policy_weights_grad_var": grad_var}, step=cur_step)
            if hasattr(self.locals["self"].policy, "cule_bfs"):
                wandb.log({"effective depth": np.mean(self.locals["self"].policy.cule_bfs.effective_depth)}, step=cur_step)
                self.locals["self"].policy.cule_bfs.effective_depth = []
            for key, val in self.locals["self"].logger.name_to_value.items():
                wandb.log({key: val}, step=cur_step)
            self.total_rewards = 0
            self.episode_length = 0

