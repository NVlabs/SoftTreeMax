import argparse
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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