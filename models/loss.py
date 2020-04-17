import torch


def log_l2_loss(x, target):
    # there is an issue with torch.norm (https://github.com/pytorch/pytorch/issues/30704) that's why it's done this way
    return (x.log1p_() - target.log1p_()).pow(2).sum(dim=(2, 3))
