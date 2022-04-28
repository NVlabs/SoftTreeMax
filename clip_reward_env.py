import gym
import numpy as np


class ClipRewardEnv2(gym.Wrapper):

    def __init__(self, env: gym.Env):
        gym.Wrapper.__init__(self, env)

    def step(self, action: int):
        obs, reward, done, info = self.env.step(action)
        info["orig_reward"] = reward
        reward = np.sign(reward)
        return obs, reward, done, info
