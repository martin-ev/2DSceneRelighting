import torch


def log_l2_loss(x, target):
    # there is an issue with torch.norm (https://github.com/pytorch/pytorch/issues/30704) that's why it's done this way
    b, c, h, w = x.size()
    n = float(b*c*h*w)  # total number of elements
    return (torch.log1p(x) - torch.log1p(target)).pow(2).sum() / n
