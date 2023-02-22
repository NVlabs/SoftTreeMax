# Built-ins
import sys
import os

from environments.cule_env_multiple import CuleEnvMultiple

# For NGC runs, TODO: Remove this in final version
sys.path.append("../stable-baselines3/")

# Externals
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_device, get_linear_fn

# Internals
from environments.cule_env import CuleEnv
from policies.actor_critic_ts import ActorCriticCnnTSPolicy
from policies.actor_critic_depth0 import ActorCriticCnnPolicyDepth0
from callbacks import WandbTrainingCallback
from utils import create_parser


# from wandb.integration.sb3 import WandbCallback
if sys.gettrace() is not None:
    os.environ["WANDB_MODE"] = "dryrun"
# os.environ["WANDB_BASE_URL"] = "http://api.wandb.ai"


def main():
    # Input arguments
    parser = create_parser()
    wandb.init(config=parser.parse_args(), project="pg-tree")
    config = wandb.config

    # TODO: decide " or " for all files
    # Setting environment
    env_kwargs = dict(env_name=config.env_name, color_mode="gray", repeat_prob=0.0, rescale=True, episodic_life=True,
                      frameskip=4)
    fire_reset = config.env_name not in ["AsterixNoFrameskip-v4", "CrazyClimberNoFrameskip-v4",
                                         "FreewayNoFrameskip-v4", "MsPacmanNoFrameskip-v4",
                                         "SkiingNoFrameskip-v4", "TutankhamNoFrameskip-v4"]
    if config.tree_depth == 0:
        env = CuleEnvMultiple(env_kwargs=env_kwargs, device="cuda:0", n_frame_stack=config.n_frame_stack,
                              clip_reward=config.clip_reward, noop_max=config.noop_max, fire_reset=fire_reset,
                              n_envs=config.n_envs)
    else:
        env = CuleEnv(env_kwargs=env_kwargs, device=get_device(), n_frame_stack=config.n_frame_stack,
                      clip_reward=config.clip_reward, noop_max=config.noop_max, fire_reset=fire_reset)
    print("Environment:", config.env_name, "Num actions:", env.action_space.n, "Tree depth:", config.tree_depth)

    # Setting PPO parameters
    ppo_def_lr = get_linear_fn(config.learning_rate, 0, 1)
    ppo_def_clip = get_linear_fn(0.1, 0, 1)
    PPO_params = {"learning_rate": ppo_def_lr, "n_epochs": 3, "gamma": 0.99, "n_steps": 128, "batch_size": 32,
                  "ent_coef": 0.01, "vf_coef": 1.0, "gae_lambda": 0.95, "clip_range": ppo_def_clip}

    # Setting PPO models
    if config.tree_depth == 0:
        model = PPO(policy=ActorCriticCnnPolicyDepth0, env=env, verbose=2, **PPO_params)
    else:
        # Hash buffer saves previous states and their trees for reuse in evaluate_actions
        hash_buffer_size = max(config.hash_buffer_size, PPO_params["n_steps"])
        # Input max width sets the maximum number of environments, since the leaves are opened we divide it here to match
        max_width = int(config.max_width / env.action_space.n) if config.max_width != -1 else -1
        policy_kwargs = {"step_env": env, "gamma": config.gamma, "tree_depth": config.tree_depth,
                         "buffer_size": hash_buffer_size, "learn_alpha": config.learn_alpha,
                         "learn_beta": config.learn_beta, "max_width": max_width, "use_leaves_v": config.use_leaves_v,
                         "is_cumulative_mode": config.is_cumulative_mode, "regularization": config.regularization}
        model = PPO(policy=ActorCriticCnnTSPolicy, env=env, verbose=1, policy_kwargs=policy_kwargs, **PPO_params)

    # save agent folder and name
    saved_agents_dir = "saved_agents"
    if not os.path.isdir(saved_agents_dir):
        os.makedirs(saved_agents_dir)
    # save agent
    saved_agent_file = "{}/{}".format(saved_agents_dir, wandb.run.id)
    callbacks = [WandbTrainingCallback()]
    model.learn(total_timesteps=config.total_timesteps, log_interval=None, callback=callbacks)
    # print("Saving agent in " + saved_agent_file)
    # TODO: fix save agent
    # model.save(saved_agent_file)

if __name__ == "__main__":
    main()
