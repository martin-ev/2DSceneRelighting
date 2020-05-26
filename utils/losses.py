import torch
import torch.nn as nn
from math import pi
from kornia.color import hsv_to_rgb


def log_l2_loss(x, target):
    # there is an issue with torch.norm (https://github.com/pytorch/pytorch/issues/30704) that's why it's done this way
    b, c, h, w = x.size()
    n = float(b * c * h * w)  # total number of elements
    return (torch.log1p(x) - torch.log1p(target)).pow(2).sum() / n


def _expand_hsv(tensor, v_channels):
    tensor_hs, tensor_v = tensor.split([2, v_channels], dim=1)
    return torch.cat((tensor_hs.repeat_interleave(v_channels, dim=1), tensor_v), dim=1)


def hsv_envmap_loss(x, target):
    v_channels = x.size()[1] - 2
    x_hsv = _expand_hsv(x, v_channels)
    target_hsv = _expand_hsv(target, v_channels)
    channels = x_hsv.size()[1]
    # view only to match requirements for tensor shape in colorspace conversion
    return log_l2_loss(hsv_to_rgb(x_hsv.view(-1, 3, channels // 3, 1)),
                       hsv_to_rgb(target_hsv.view(-1, 3, channels // 3, 1)))


def cos_loss(a1, a2):
    return torch.mean(1. - torch.cos((a1 - a2)/180.*pi))


_DISTANCES = {
    "cos": cos_loss,
    "logL2": log_l2_loss,
    1: nn.L1Loss(),
    2: nn.MSELoss()
}

_FACTORS = {
    "cos": lambda f: 1.,
    "logL2": lambda f: 1.,
    1: lambda f: f,
    2: lambda f: f**2
}


class _Loss(nn.Module):    
    def __init__(self):
        super(_Loss, self).__init__()


class _SimpleLoss(_Loss):    
    def __init__(self, p=2, factor=1.):
        super().__init__()
        self.distance = _DISTANCES[p]
        self.factor = factor

    def forward(self, prediction, groundtruth):
        return self.factor * self.distance(prediction, groundtruth)


class _SimpleLossWithRef(_SimpleLoss):    
    def __init__(self, p=2):
        super().__init__(p=p, factor=1.)

    def forward(self, prediction, groundtruth, ref1, ref2):
        return super().forward(prediction, groundtruth)/super().forward(ref1, ref2)


class ReconstructionLoss(_SimpleLoss):    
    def __init__(self, p=2):
        super().__init__(p=p, factor=1.)


class ColorPredictionLoss(_SimpleLoss):    
    def __init__(self, p=2):
        factor = _FACTORS[p](1/2000)
        super().__init__(p=p, factor=factor)


class DirectionPredictionLoss(_SimpleLoss):    
    def __init__(self, p="cos"):
        factor = _FACTORS[p](1.)
        super().__init__(p=p, factor=factor)


class SceneLatentLoss(_SimpleLoss):    
    def __init__(self, p=2):
        super().__init__(p=p)


class LightLatentLoss(_SimpleLoss):    
    def __init__(self, p=2):
        super().__init__(p=p)


class SceneLatentLossWithRef(_SimpleLossWithRef):    
    def __init__(self, p=2):
        super().__init__(p=p)


class LightLatentLossWithRef(_SimpleLossWithRef):    
    def __init__(self, p=2):
        super().__init__(p=p)
        
        
class GANLoss(_Loss):    
    def __init__(self):
        super().__init__()
        self.distance = nn.MSELoss()

    def forward(self, disc_out_fake, disc_out_real):
        return self.distance(disc_out_fake, torch.zeros(disc_out_fake.size()).to(disc_out_fake.device)) + \
               self.distance(disc_out_real, torch.ones(disc_out_real.size()).to(disc_out_real.device))


class FoolGANLoss(_Loss):    
    def __init__(self):
        super().__init__()
        self.distance = nn.MSELoss()

    def forward(self, disc_out_fake):
        return self.distance(disc_out_fake, torch.ones(disc_out_fake.size()).to(disc_out_fake.device))
