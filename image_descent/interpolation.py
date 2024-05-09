from collections.abc import Sequence, Callable, Iterable
from scipy.ndimage import map_coordinates
import numpy as np
import torch

def get_interpolated_value_scipy(img:np.ndarray, coord:Sequence[float] | torch.Tensor, mode='nearest', order=1):
    coord_array = np.array(coord)
    for i, (s, c) in enumerate(zip(img.shape, coord)):
        coord_array[i] = s*c
    return map_coordinates(img, np.expand_dims(coord_array, 1), mode=mode, order=order)[0]

def get_interpolated_value_torch(img:torch.Tensor, coord: torch.Tensor, mode:str = 'bicubic'):
    # add batch and channel dimensions
    img = img.unsqueeze(0).unsqueeze(0)
    while coord.ndim < 4: coord = coord.unsqueeze(0)
    return torch.nn.functional.grid_sample(img, coord, mode=mode, padding_mode="border", align_corners=False)[0,0,0,0]

def get_interpolated_value_neighbours(img:torch.Tensor, coord:torch.Tensor):
    x0, y0 = coord.int()
    # get values
    v00 = img[x0, y0]
    v01 = img[x0, y0+1]
    v10 = img[x0+1, y0]
    v11 = img[x0+1, y0+1]
    ...