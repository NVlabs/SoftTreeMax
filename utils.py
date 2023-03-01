import argparse
import torch
import torch.backends.cudnn
import numpy as np
import random


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def groupby_sum(value, labels) -> (torch.Tensor, torch.LongTensor):
    uniques = labels.unique().tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    val_key = {val: key for key, val in zip(uniques, range(len(uniques)))}

    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels, labels_count = labels.unique(dim=0, return_counts=True)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(0, labels, value)
    new_labels = torch.LongTensor(list(map(val_key.get, unique_labels[:, 0].tolist())))
    return result, new_labels

def add_regularization_logits(logits, epsilon):
    A = logits.shape[1]
    probs = torch.exp(logits - torch.logsumexp(logits, dim=1, keepdim=True))
    new_probs = (1-epsilon) * probs + epsilon/A
    new_logits = torch.log(new_probs)
    return  new_logits


def create_parser():
    # TODO: Add help
    # TODO: remove irrelevant parameters?
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=200000000,
                        help="Number of environment steps for training")
    parser.add_argument("--learning_rate", type=float, default=2.5e-04, help="Optimizer learning rate")
    parser.add_argument("--seed", type=int, default=4, help="Seed for all pseudo-random generators")
    parser.add_argument("--env_name", type=str, default="AlienNoFrameskip-v4", help="Environment name")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tree_depth", type=int, default=0, help="SoftTreeMax depth (0 corresponds to standard PPO)")
    parser.add_argument("--clip_reward", type=str2bool, nargs="?", const=True, default=True,
                        help="Reward clipping wrapper")
    parser.add_argument("--learn_alpha", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to treat alpha (weight of root value) as a learnable parameter or constant")
    parser.add_argument("--learn_beta", type=str2bool, nargs="?", const=True, default=True,
                        help="Whether to treat beta (temperature parameter) as a learnable parameter or constant")
    parser.add_argument("--max_width", type=int, default=-1,
                        help="Maximal SoftTreeMax width, beyond which the tree will be truncated. "
                             "Use -1 to not limit width.")
    parser.add_argument("--experiment_type", type=str, default="", help="Free text to describe experiment goal")
    # experiment_type examples: Runtime_optimization, Debug, Paper_main, Ablation, Hyperparameter_sweep
    parser.add_argument("--experiment_description", type=str, default="",
                        help="Free text to describe experiment sub-goal")
    parser.add_argument("--hash_buffer_size", type=int, default=-1, help="Size of buffer which stores leaf values")
    parser.add_argument("--use_leaves_v", type=str2bool, nargs="?", const=True, default=False,
                        help="Whether to use the value at the leaves or reward only")
    parser.add_argument("--is_cumulative_mode", type=str2bool, nargs="?", const=True, default=False,
                        help="True for Cumulative SoftTreeMax. False for Exponentiated SoftTreeMax")
    parser.add_argument("--regularization", type=float, default=0.001, help="Minimal probability for all actions")
    parser.add_argument("--n_envs", type=int, default=256, help="Number of parallel PPO environments on GPU")
    # Evaluation fields
    parser.add_argument("--run_type", type=str, default="train", help="Train or evaluate")  # train or evaluate
    parser.add_argument("--model_filename", type=str, default=None, help="Filename to store or load model")
    parser.add_argument("--n_eval_episodes", type=int, default=200, help="Number of evaluation episodes")
    return parser
