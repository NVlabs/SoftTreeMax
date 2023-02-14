import sys

from clip_reward_env import ClipRewardEnv2
import torch as th

sys.path.append('../stable-baselines3/')

from policies.actor_critic_depth0 import ActorCriticCnnPolicyDepth0

from callbacks import WandbTrainingCallback
from policies.actor_critic_ts import ActorCriticCnnTSPolicy
from utils import str2bool
import gym
import wandb
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import get_device, get_linear_fn
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from environments.cule_env import CuleEnv
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
from environments.cule_env_multiple import CuleEnvMultiple

if sys.gettrace() is not None:
    os.environ["WANDB_MODE"] = "dryrun"

parser = argparse.ArgumentParser()
parser.add_argument('--total_timesteps', type=int, default=200000000)
parser.add_argument('--train_freq', type=int, default=50000)
parser.add_argument('--exploration_initial_eps', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2.5e-04)
parser.add_argument('--target_update_interval', type=int, default=10000)
parser.add_argument('--exploration_final_eps', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--env_name', type=str, default='AlienNoFrameskip-v4')
parser.add_argument('--use_cule', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--evaluate_freq', type=int, default=100)
parser.add_argument('--tree_depth', type=int, default=0)
parser.add_argument('--learning_starts', type=int, default=1000)
parser.add_argument('--normalize_images', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--exploration_fraction', type=float, default=0.02)
parser.add_argument('--eval_saved_agent', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--saved_model', type=str, default=None)
parser.add_argument('--n_frame_stack', type=int, default=4)
parser.add_argument('--n_eval_ep', type=int, default=10)
parser.add_argument('--clip_reward', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--noop_max', type=int, default=30)
parser.add_argument('--learn_alpha', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--learn_beta', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--max_width', type=int, default=-1)
parser.add_argument('--experiment_type', type=str, default="")  # Runtime_optimization, Debug, Paper_main, Ablation, Hyperparameter_sweep
parser.add_argument('--experiment_description', type=str, default="")
parser.add_argument('--hash_buffer_size', type=int, default=-1)
parser.add_argument('--n_envs', type=int, default=128)
parser.add_argument('--n_envs_timesteps', type=int, default=5000000)
parser.add_argument('--use_leaves_v', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--is_cumulative_mode', type=str2bool, nargs='?', const=True, default=False)



wandb.init(config=parser.parse_args(), project="pg-tree")
config = wandb.config
print('tree_depth: {}'.format(config.tree_depth))

env_kwargs = dict(env_name=config.env_name, color_mode='gray', repeat_prob=0.0, rescale=True, episodic_life=True,
                  frameskip=4)

fire_reset = config.env_name not in ['AsterixNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4',
                                     'FreewayNoFrameskip-v4', 'MsPacmanNoFrameskip-v4',
                                     'SkiingNoFrameskip-v4', 'TutankhamNoFrameskip-v4']
if config.use_cule:
    env = CuleEnvMultiple(env_kwargs=env_kwargs, device='cuda:0', n_frame_stack=config.n_frame_stack,
                          clip_reward=config.clip_reward, noop_max=config.noop_max, fire_reset=fire_reset,
                          n_envs=config.n_envs)

print("Environment: ", config.env_name, "Num actions: ", env.action_space.n)

ppo_def_lr = get_linear_fn(config.learning_rate, 0, 1)
ppo_def_clip = get_linear_fn(0.1, 0, 1)
PPO_params = {'learning_rate': ppo_def_lr, 'n_epochs': 3, 'gamma': 0.99, 'n_steps': 128, 'batch_size': 32,
              'ent_coef': 0.01, 'vf_coef': 1.0, 'gae_lambda': 0.95, 'clip_range': ppo_def_clip}
model_nenvs = PPO(policy=ActorCriticCnnPolicyDepth0, env=env, verbose=2, **PPO_params)
callbacks = [WandbTrainingCallback()]
model_nenvs.learn(total_timesteps=config.n_envs_timesteps, log_interval=None, callback=callbacks)

print("Finished first stage!")

env = CuleEnv(env_kwargs=env_kwargs, device=get_device(), n_frame_stack=config.n_frame_stack,
                  clip_reward=config.clip_reward, noop_max=config.noop_max, fire_reset=fire_reset)

ppo_def_lr = get_linear_fn(config.learning_rate, 0, 1)
ppo_def_clip = get_linear_fn(0.1, 0, 1)
# * (1+config.tree_depth)
PPO_params = {'learning_rate': ppo_def_lr, 'n_epochs': 3, 'gamma': 0.99, 'n_steps': 128, 'batch_size': 32,
              'ent_coef': 0.01, 'vf_coef': 1.0, 'gae_lambda': 0.95, 'clip_range': ppo_def_clip}
if config.tree_depth == 0:
    model = PPO(policy=ActorCriticCnnPolicyDepth0, env=env, verbose=2, **PPO_params)
else:
    hash_buffer_size = max(config.hash_buffer_size, PPO_params['n_steps'])
    max_width = int(config.max_width / env.action_space.n) if config.max_width != -1 else -1
    policy_kwargs = {'step_env': env, 'gamma': config.gamma, 'tree_depth': config.tree_depth,
                     'buffer_size': hash_buffer_size, 'learn_alpha': config.learn_alpha,
                     'learn_beta': config.learn_beta, 'max_width': max_width, 'use_leaves_v': config.use_leaves_v}
    model = PPO(policy=ActorCriticCnnTSPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs, **PPO_params)

# data, params, pytorch_variables = load_from_zip_file("./gxa0fpr9_mspacman_35M.zip", device="auto", custom_objects=None, print_system_info=False)
# model.policy.load_state_dict(params['policy'], strict=False)
warm_state = model_nenvs.policy.state_dict()
own_state = model.policy.state_dict()
for name, param in warm_state.items():
    if not name.startswith('features_extractor'):
         continue
    if isinstance(param, th.nn.Parameter):
        # backwards compatibility for serialized parameters
        param = param.data
    own_state[name].copy_(param)

wandb_callback = WandbTrainingCallback()
wandb_callback.warm_start = model_nenvs.num_timesteps
callbacks = [wandb_callback]
model.learn(total_timesteps=config.total_timesteps, log_interval=None, callback=callbacks)
