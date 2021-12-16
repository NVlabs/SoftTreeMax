from utils import str2bool
import gym
import wandb
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from environments.cule_env import CuleEnv

# os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_BASE_URL"] = "http://api.wandb.ai"

parser = argparse.ArgumentParser()
parser.add_argument('--total_timesteps', type=int, default=50000000)
parser.add_argument('--train_freq', type=int, default=50000)
parser.add_argument('--exploration_initial_eps', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=2.5e-05)
parser.add_argument('--target_update_interval', type=int, default=10000)
parser.add_argument('--exploration_final_eps', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=4)
parser.add_argument('--env_name', type=str, default='cule_BreakoutNoFrameskip-v4')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--evaluate_freq', type=int, default=100)
parser.add_argument('--tree_depth', type=int, default=0)
parser.add_argument('--learning_starts', type=int, default=1000)
parser.add_argument('--normalize_images', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--exploration_fraction', type=float, default=0.02)
parser.add_argument('--eval_saved_agent', type=str2bool, nargs='?', const=True, default=False)
parser.add_argument('--saved_model', type=str, default=None)
parser.add_argument('--warp_frame', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--frame_stack', type=int, default=4)
parser.add_argument('--n_eval_ep', type=int, default=10)

args = vars(parser.parse_args())
wandb.init(config=args, project="pg-tree")
config = wandb.config
normalize_images = config.normalize_images
print('tree_depth: {}'.format(config.tree_depth))

if config.env_name.startswith('cule'):
    env_name = config.env_name[5:]
    color = 'rgb'
    rescale = False
    if config.warp_frame:
        color = 'gray'
        rescale = True
    env = CuleEnv(env_name=env_name, tree_depth=config.tree_depth, gamma=config.gamma,
                          color_mode=color, rescale_frame=rescale, episodic_life=True)
else:
    orig_env = gym.make(config.env_name)
    # rew_normalization_factor, obs_normalization_factor = env_normalization_table.get(config.env_name, (1.0, 1.0))
    # env = ClipRewardEnv(orig_env, rew_normalization_factor) #contains functions needed for DFS
    # env = WarpFrame(orig_env)
    env = MaxAndSkipEnv(orig_env, skip=4)
    # env = FireResetEnv(env)
    if config.warp_frame:
        env = WarpFrame(env)
    if config.frame_stack > 1:
        env = FrameStack(env, config.frame_stack)

policy_kwargs = {}
model = PPO(env, verbose=1, learning_rate=config.learning_rate, gamma=config.gamma, policy_kwargs=policy_kwargs)

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
    checkpoint_callback = CheckpointCallback(save_freq=int(2e6), save_path=saved_agents_dir, name_prefix=wandb.run.id)
    model.learn(total_timesteps=config.total_timesteps, log_interval=None, callback=checkpoint_callback,
                saved_agent_file=saved_agent_file)

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


