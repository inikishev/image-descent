import torch
import numpy as np

def get_gradients_numpy(image:torch.Tensor):
    image = image.detach().cpu().numpy()
    return list(reversed([torch.from_numpy(i) for i in np.gradient(image, edge_order=1)]))

def get_gradients_by_shifting(image:torch.Tensor):
    return (image[1:] - image[:-1], image[:,1:] - image[:,:-1])