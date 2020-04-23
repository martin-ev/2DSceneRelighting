# From: https://github.com/bonlime/pytorch-tools/blob/master/pytorch_tools/metrics/psnr.py

import torch
import numpy as np
import cv2


def psnr(img1, img2):
    """
    Peak Signal to Noise Ratio
    Inputs are expected to be in range [0, 255]
    """
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


class SSIM:
    """
    Structure similarity
    Inputs are expected to be in range [0, 255]
    """
    def __init__(self, levels=255):
        self.name = "SSIM"
        self.levels = levels

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self.ssim(img1, img2, self.levels)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self.ssim(img1, img2, self.levels))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self.ssim(np.squeeze(img1), np.squeeze(img2), self.levels)
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def ssim(img1, img2, levels):
        c1 = (0.01 * levels) ** 2
        c2 = (0.03 * levels) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
                (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return ssim_map.mean()


class DSSIM:
    """
    Structural dissimilarity
    Inputs are expected to be in range [0, 255]
    """
    def __init__(self, levels=255):
        self.name = "DSSIM"
        self.levels = levels

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self.dssim(img1, img2, self.levels)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                dssims = []
                for i in range(3):
                    dssims.append(self.dssim(img1, img2, self.levels))
                return np.array(dssims).mean()
            elif img1.shape[2] == 1:
                return self.dssim(np.squeeze(img1), np.squeeze(img2), self.levels)
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def dssim(img1, img2, levels):
        ssim = SSIM.ssim(img1, img2, levels)
        return (1 - ssim) / 2
