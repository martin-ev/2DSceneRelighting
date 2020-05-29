from kornia.losses import psnr_loss as kornia_psnr
from kornia.losses import ssim as kornia_ssim


def psnr(image_batch, groundtruth_batch):
    return kornia_psnr(image_batch, groundtruth_batch, 1.)


def ssim(image_batch, groundtruth_batch):
    # window size and reduction based on https://github.com/Po-Hsun-Su/pytorch-ssim
    return kornia_ssim(image_batch, groundtruth_batch, window_size=11, reduction='mean')
