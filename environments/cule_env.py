from collections import deque
import torch
import cv2  # Note that importing cv2 before torch may cause segfaults?
from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom
import gym
import numpy as np
from gym import spaces
import math


class CuleEnv():
    def __init__(self, device, env_kwargs, n_frame_stack=4):
        self.device = device
        self.env_kwargs = env_kwargs
        cart = AtariRom(env_kwargs['env_name'])
        actions = cart.minimal_actions()
        self.env = AtariEnv(num_envs=1, device=torch.device('cpu'), **env_kwargs)
        super(AtariEnv, self.env).reset(0)
        self.env.reset(initial_steps=1, verbose=1)
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.n_frame_stack = n_frame_stack  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=n_frame_stack)
        self.training = True  # Consistent with model training mode

        # Stable baselines requirements
        self.reward_range = (-math.inf, math.inf)
        self.metadata = {"render.modes": ["human", "rgb_array"]}
        orig_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (orig_shape[0] * n_frame_stack, orig_shape[1], orig_shape[2]),
                                                np.uint8)
        self.action_space = spaces.Discrete(len(actions))  # self.gpu_env.action_space

    def _reset_buffer(self):
        for _ in range(self.n_frame_stack):
            self.state_buffer.append(torch.zeros(84, 84, device=self.device))

    def reset(self):
        obs = torch.zeros(84, 84, device=self.device)
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.env.step([0])  # Use a no-op after loss of life
        else:
            # Reset internals
            self._reset_buffer()
            # Perform up to 30 random no-ops before starting
            obs = self.env.reset(initial_steps=1, verbose=1)
            obs = obs[0, :, :, 0].to(self.device)

        obs = obs / 255
        self.last_frame = obs
        self.state_buffer.append(obs)
        # self.env.step(torch.tensor([1]))
        self.lives = self.env.lives  # TODO: update for other games
        # self.lives = None # self.ale.lives() #TODO: update for other games
        return torch.stack(list(self.state_buffer), 0)

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        obs, reward, done, info = self.env.step(torch.tensor([action]))
        if self.lives is None:
            self.lives = self.env.lives.item()
        obs = obs[0, :, :, 0].to(self.device) / 255
        self.state_buffer.append(obs)
        self.last_frame = obs
        # Detect loss of life as terminal in training mode
        lives = info['ale.lives'][0]
        if self.training:
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
        self.lives = lives
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 0), reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self):
        # TODO: adapt to cule
        cv2.imshow('screen', self.last_frame)
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
