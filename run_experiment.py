import sys
sys.path.append('../stable-baselines3/')
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
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from environments.cule_env import CuleEnv
from stable_baselines3.common.env_util import make_vec_env

# from wandb.integration.sb3 import WandbCallback

# os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_BASE_URL"] = "http://api.wandb.ai"

parser = argparse.ArgumentParser()
parser.add_argument('--total_timesteps', type=int, default=40000000)
parser.add_argument('--train_freq', type=int, default=50000)
parser.add_argument('--exploration_initial_eps', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2.5e-05)
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
parser.add_argument('--learn_alpha', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--max_width', type=int, default=-1)

wandb.init(config=parser.parse_args(), project="pg-tree")
config = wandb.config
normalize_images = config.normalize_images
print('tree_depth: {}'.format(config.tree_depth))

# episodic_life is false since we take care of that ourselves
env_kwargs = dict(env_name=config.env_name, color_mode='gray', repeat_prob=0.0, rescale=True, episodic_life=False,
                  frameskip=4)

fire_reset = config.env_name not in ['AsterixNoFrameskip-v4', 'CrazyClimberNoFrameskip-v4',
                                     'FreewayNoFrameskip-v4', 'MsPacmanNoFrameskip-v4',
                                     'SkiingNoFrameskip-v4', 'TutankhamNoFrameskip-v4']
if config.use_cule:
    env = CuleEnv(env_kwargs=env_kwargs, device=get_device(), n_frame_stack=config.n_frame_stack,
                  clip_reward=config.clip_reward, noop_max=config.noop_max, fire_reset=fire_reset)
    # env = make_vec_env(CuleEnv, n_envs=8, env_kwargs={'env_kwargs': env_kwargs, 'device': get_device(),
    #                                             'n_frame_stack':config.n_frame_stack})
else:
    orig_env = gym.make(config.env_name)
    # rew_normalization_factor, obs_normalization_factor = env_normalization_table.get(config.env_name, (1.0, 1.0))
    # env = ClipRewardEnv(orig_env, rew_normalization_factor) #contains functions needed for DFS
    # env = WarpFrame(orig_env)
    env = MaxAndSkipEnv(orig_env, skip=4)
    # env = FireResetEnv(env)
    if config.warp_frame:
        env = WarpFrame(env)
    if config.n_frame_stack > 1:
        env = FrameStack(env, config.n_frame_stack)

print("Environment: ", config.env_name, "Num actions: ", env.action_space.n)

ppo_def_lr = get_linear_fn(2.5e-4, 0, 1)
ppo_def_clip = get_linear_fn(0.1, 0, 1)
# * (1+config.tree_depth)
PPO_params = {'learning_rate': ppo_def_lr, 'n_epochs': 3 , 'gamma': 0.99, 'n_steps': 128, 'batch_size': 32,
              'ent_coef': 0.01, 'vf_coef': 1.0, 'gae_lambda': 0.95, 'clip_range': ppo_def_clip}
if config.tree_depth == 0:
    model = PPO(policy=ActorCriticCnnPolicy, env=env, verbose=2, **PPO_params)
else:
    hash_buffer_size = 2048
    policy_kwargs = {'step_env': env, 'gamma': config.gamma, 'tree_depth': config.tree_depth,
                     'buffer_size': hash_buffer_size, 'learn_alpha': config.learn_alpha,
                     'max_width': config.max_width}
    model = PPO(policy=ActorCriticCnnTSPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs, **PPO_params)
# learning_rate=config.learning_rate,



if not config.eval_saved_agent:
    # save agent folder and name
    saved_agents_dir = 'saved_agents'
    if not os.path.isdir(saved_agents_dir):
        os.makedirs(saved_agents_dir)

    # for logging std of random policies: (i) pass eval_callback to model.learn(); (ii) set exploration_final_eps=1;
    # (iii) set learning_starts=10000000
    # eval_callback = EvalCallback(eval_env=env, n_eval_episodes=5, eval_freq=50000, render=False, verbose=0)
    # save agent
    saved_agent_file = '{}/{}'.format(saved_agents_dir, wandb.run.id)
    # checkpoint_callback = CheckpointCallback(save_freq=int(2e6), save_path=saved_agents_dir, name_prefix=wandb.run.id)
    callbacks = [WandbTrainingCallback()]
    model.learn(total_timesteps=config.total_timesteps, log_interval=None, callback=callbacks)

    print("Saving agent in " + saved_agent_file)
    model.save(saved_agent_file)
else:
    # eval_freq of callback needs to be atleast train_freq to get desirable behavior
    eval_callback = EvalCallback(eval_env=env, n_eval_episodes=250, eval_freq=dqn_train_freq, render=False, verbose=1)
    # model = model.load('saved_agents/10h6okjk_10000000_steps.zip', env=env) # expects native_vizoom_resolution=True, extra_action=False
    model = model.load(config.saved_model, env=env, policy_kwargs=model.policy_kwargs)  # expects native_vizoom_resolution=True, extra_action=False
    # model.q_net.tree_depth = config.tree_depth
    # model.q_net.forward_model = env
    model.learn(total_timesteps=1, log_interval=None, callback=eval_callback)


