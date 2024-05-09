from collections.abc import Sequence, Callable, Iterable
import logging
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
        image = cv2.imread(__path_or_array) # pylint:disable=E1101
        if isinstance(image, np.ndarray) and image.size > 0: return image
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

def prepare_image(image:np.ndarray | torch.Tensor) -> torch.Tensor:
    # convert to tensor
    if isinstance(image, np.ndarray): image = torch.from_numpy(image.copy()).to(torch.float64)

    if image.ndim > 3: raise ValueError(f"prepare_image: Image has invalid shape {image.shape}, it must be 2 or 3 dimensional.")

    # convert to black and white
    if image.ndim == 3:
        if image.shape[0] < image.shape[2]: image = image.mean(0)
        else: image = image.mean(2)

    # normalize to 0-1 range
    image = image.type(torch.float64)
    image = image - image.min()
    if image.max() != 0: image = image / image.max()
    else: logging.warning("prepare_image: the image seems to be completely flat (e.g. fully black)")

    return image