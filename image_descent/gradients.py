import torch
import numpy as np

def get_gradients_numpy(image:torch.Tensor):
    image = image.detach().cpu().numpy()
    return list(reversed([torch.from_numpy(i) for i in np.gradient(image, edge_order=1)]))

def get_gradients_by_shifting(image:torch.Tensor):
    return (image[1:] - image[:-1], image[:,1:] - image[:,:-1])


def out_of_bounds_soft(coords:torch.Tensor, grad:torch.Tensor):
    # if optimizer decided to go out of bounds, gradients will point inwards
    if coords[0] < -1: grad[0] = - (coords[0].abs() - 1) ** 3
    if coords[0] > 1: grad[0] = (coords[0] - 1) ** 3
    if coords[1] < -1: grad[1] = - (coords[1].abs() - 1) ** 3
    if coords[1] > 1: grad[1] = (coords[1] - 1) ** 3

    return grad

def out_of_bounds_hard(coords:torch.Tensor, grad:torch.Tensor, strength = 0.003):
    # if optimizer decided to go out of bounds, gradients will point inwards
    if coords[0] < -1: grad[0] = - strength
    if coords[0] > 1: grad[0] = strength
    if coords[1] < -1: grad[1] = - strength
    if coords[1] > 1: grad[1] = strength

    return grad
