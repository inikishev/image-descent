import torch
import numpy as np
from scipy.ndimage import gaussian_filter

def smooth_gaussian(image:torch.Tensor, amount):
    return torch.from_numpy(gaussian_filter(image.numpy(), amount))