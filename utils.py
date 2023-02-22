import argparse
import torch

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
    parser.add_argument("--total_timesteps", type=int, default=200000000)
    parser.add_argument("--train_freq", type=int, default=50000)
    parser.add_argument("--exploration_initial_eps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2.5e-04)
    parser.add_argument("--target_update_interval", type=int, default=10000)
    parser.add_argument("--exploration_final_eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--env_name", type=str, default="AlienNoFrameskip-v4")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--evaluate_freq", type=int, default=100)
    parser.add_argument("--tree_depth", type=int, default=0)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--exploration_fraction", type=float, default=0.02)
    parser.add_argument("--eval_saved_agent", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--saved_model", type=str, default=None)
    parser.add_argument("--n_frame_stack", type=int, default=4)
    parser.add_argument("--n_eval_ep", type=int, default=10)
    parser.add_argument("--clip_reward", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--noop_max", type=int, default=30)
    parser.add_argument("--learn_alpha", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--learn_beta", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--max_width", type=int, default=-1)
    parser.add_argument("--experiment_type", type=str, default="")  # Runtime_optimization, Debug, Paper_main, Ablation, Hyperparameter_sweep
    parser.add_argument("--experiment_description", type=str, default="")
    parser.add_argument("--hash_buffer_size", type=int, default=-1)
    parser.add_argument("--use_leaves_v", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--is_cumulative_mode", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--regularization", type=float, default=0.001)
    parser.add_argument("--n_envs", type=int, default=256)
    return parser