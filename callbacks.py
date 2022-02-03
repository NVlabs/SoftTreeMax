import numpy as np
import wandb

from stable_baselines3.common.callbacks import BaseCallback


class WandbTrainingCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super(WandbTrainingCallback, self).__init__(verbose)
        self.total_rewards = 0
        self.episode_length = 0
        self.prev_life = 0

    def _on_step(self) -> bool:
        self.total_rewards += np.mean(self.locals["rewards"])
        self.episode_length += 1
        info = self.locals["infos"][0]
        if info.get("ale.lives"):
            done = int(info["ale.lives"][0]) > self.prev_life
            self.prev_life = int(info["ale.lives"][0])
        else:
            done = int(self.locals["dones"][0])
        if done:
            wandb.log({"train\episodic_reward": self.total_rewards}, step=self.model.num_timesteps)
            wandb.log({"train\episodic_length": self.episode_length}, step=self.model.num_timesteps)
            wandb.log({"num_steps": self.model.num_timesteps}, step=self.model.num_timesteps)
            self.total_rewards = 0
            self.episode_length = 0