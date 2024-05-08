from collections.abc import Sequence, Callable, Iterable
from scipy.ndimage import map_coordinates
import numpy as np
import torch

def imread(__path_or_array):
    if isinstance(__path_or_array, np.ndarray): return __path_or_array
    if isinstance(__path_or_array, torch.Tensor): return __path_or_array.numpy()
    if isinstance(__path_or_array, (list, tuple)): return np.asanyarray(__path_or_array)
    elif not isinstance(__path_or_array, str): raise ValueError(f"Invalid type {type(__path_or_array)} for `imread`")
    exceptions = []
    try:
        import matplotlib.pyplot as plt
        return plt.imread(__path_or_array)
    except Exception as e: exceptions.append(e)

    try:
        import cv2
        return cv2.imread(__path_or_array) # pylint:disable=E1101
    except Exception as e: exceptions.append(e)

    try:
        import skimage.io
        return skimage.io.imread(__path_or_array)
    except Exception as e: exceptions.append(e)

    try:
        import PIL.Image
        return PIL.Image.open(__path_or_array)
    except Exception as e: exceptions.append(e)

    try:
        import torchvision
        return torchvision.io.read_image(__path_or_array).numpy()
    except Exception as e: exceptions.append(e)

    try:
        return np.load(__path_or_array)
    except Exception as e: exceptions.append(e)

    try:
        return torch.load(__path_or_array)
    except Exception as e: exceptions.append(e)

    exceptions_str = '\n\n'.join(exceptions)
    raise ValueError(f"imread: Could not read image from {__path_or_array}:\n{exceptions_str}")

def ensure_channel_last(image:np.ndarray):
    if isinstance(image, torch.Tensor): image = image.numpy()
    if image.ndim == 3:
        if image.shape[0] < image.shape[2]:
            return image.transpose((1,2,0))
        else: return image
    elif image.ndim == 2: return np.expand_dims(image, 2)
    else: raise ValueError(f"ensure_channel_last: Image has invalid shape {image.shape}, it must be 2 or 3 dimensional.")

def to_black_and_white(image:np.ndarray | torch.Tensor):
    """black and whites a CHANNEL LAST IMAGE!!!"""
    if isinstance(image, torch.Tensor): image = image.numpy()
    image = image.astype(np.float64) # type:ignore
    if image.ndim == 3:
        return image.mean(2)
    elif image.ndim == 2: return image
    else: raise ValueError(f"to_black_and_white: Image has invalid shape {image.shape}, it must be 2 or 3 dimensional.")

def normalize_to_range(x:torch.Tensor | np.ndarray, min=0, max=1): #pylint:disable=W0622
    if isinstance(x, torch.Tensor): x = x.numpy()
    x = x.astype(np.float64) # type:ignore
    x -= x.min()
    if x.max()!=0:
        x /= x.max()
    else: return x
    x *= max - min
    x += min
    return x

def z_normalize(x:torch.Tensor | np.ndarray):
    if isinstance(x, torch.Tensor): x = x.numpy()
    x = x.astype(np.float64) # type:ignore
    if x.std() != 0: return (x - x.mean()) / x.std()
    return x - x.mean()


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __add__(self, other: "Compose | Callable | Iterable"):
        if isinstance(other, Compose):
            return Compose(*self.transforms, *other.transforms)
        elif callable(other):
            return Compose(*self.transforms, other)
        else:
            return Compose(*self.transforms, *other)

    def __str__(self):
        return f"Compose({', '.join(str(t) for t in self.transforms)})"

def get_interpolated_value(img:np.ndarray, coord:Sequence[float] | torch.Tensor, order=3, img_max=None):
    if img_max is None: img_max = img.max()
    coord_array = np.array(coord)
    for i, (s, c) in enumerate(zip(img.shape, coord)):
        coord_array[i] = s*c
    return map_coordinates(img, np.expand_dims(coord_array, 1), mode='constant', cval=img_max, order=order)[0]

def get_interpolated_value_torch(img:torch.Tensor, coord: torch.Tensor, mode:str):
    # add batch and channel dimensions
    img = img.unsqueeze(0).unsqueeze(0)
    while coord.ndim < 4: coord = coord.unsqueeze(0)
    return torch.nn.functional.grid_sample(img, coord, mode=mode, padding_mode="border", align_corners=False)[0,0,0,0]

