from collections import deque
import torch
import cv2  # Note that importing cv2 before torch may cause segfaults?
from torchcule.atari import Env as AtariEnv
from torchcule.atari import Rom as AtariRom
import gym
import numpy as np
from gym import spaces
import math
from stable_baselines3.common.vec_env import VecEnv


class CuleEnvMultiple(VecEnv):
    def __init__(self, device, env_kwargs, n_frame_stack=4, noop_max=30, clip_reward=True, fire_reset=True,
                 n_envs=1):
        self.device = device
        self.env_kwargs = env_kwargs
        cart = AtariRom(env_kwargs['env_name'])
        self.num_envs = n_envs
        actions = cart.minimal_actions()
        self.env = AtariEnv(num_envs=n_envs, device=device, **env_kwargs)
        super(AtariEnv, self.env).reset(0)
        self.env.reset(initial_steps=1, verbose=1)
        self.lives = 0  # Life counter (used in DeepMind training)
        self.n_frame_stack = n_frame_stack  # Number of frames to concatenate
        self.state_buffer = deque([], maxlen=n_frame_stack)
        self.training = True  # Consistent with model training mode
        self.noop_max = noop_max
        self.clip_reward = clip_reward
        self.fire_reset = fire_reset

        # Stable baselines requirements
        self.reward_range = (-math.inf, math.inf)
        self.metadata = {"render.modes": ["human", "rgb_array"]}
        orig_shape = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (orig_shape[0] * n_frame_stack, orig_shape[1], orig_shape[2]),
                                                np.uint8)
        self.action_space = spaces.Discrete(len(actions))  # self.gpu_env.action_space

    def _reset_buffer(self):
        for _ in range(self.n_frame_stack):
            self.state_buffer.append(torch.zeros(self.num_envs, 84, 84, device=self.device))

    def reset(self):
        # Reset internals
        self._reset_buffer()
        # Perform up to 30 random no-ops before starting
        noops = np.random.randint(self.noop_max + 1)
        obs = self.env.reset(initial_steps=noops, verbose=1)
        self.env.lives[0] = 5  # Assaf: reset doesn't handle this?
        obs = obs.reshape((self.num_envs, 84, 84)).to(self.device)

        # obs = obs / 255
        self.last_frame = obs
        self.state_buffer.append(obs)
        # self.env.step(torch.tensor([1]))
        self.lives = self.env.lives.cpu().numpy()  # TODO: update for other games
        # self.lives = None # self.ale.lives() #TODO: update for other games
        return torch.stack(list(self.state_buffer), 1).cpu().numpy()

    def step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        obs, reward, done, info = self.env.step(torch.tensor(action, device=self.env.device))
        infos = [{'ale.lives': info['ale.lives'][i]} for i in range(len(info['ale.lives']))]
        for i in range(len(infos)):
            infos[i]["orig_reward"] = reward[i].cpu().numpy()
            infos[i]["done"] = done[i].cpu().numpy()
        if self.clip_reward:
            reward = torch.sign(reward)
        if self.lives is None:
            self.lives = self.env.lives.item()
        obs = obs[:, :, :, 0].to(self.device)  # / 255
        self.state_buffer.append(obs)
        self.last_frame = obs
        # Detect loss of life as terminal in training mode
        self.lives = info['ale.lives'].cpu().numpy()
        # Return state, reward, done
        return torch.stack(list(self.state_buffer), 1).cpu().numpy(), reward.cpu().numpy(), \
               done.type(torch.float).cpu().numpy(), infos

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

    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    def step_wait(self):
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        raise NotImplementedError()

    def get_attr(self, attr_name: str, indices=None):
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError()

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        """
        Check if environments are wrapped with a given wrapper.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: True if the env is wrapped, False otherwise, for each env queried.
        """
        raise NotImplementedError()

    def seed(self, seed=None):
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        pass