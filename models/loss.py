import torch


def log_l2_loss(x, target):
    return torch.norm(x.log1p_() - target.log1p_(), dim=(2, 3))