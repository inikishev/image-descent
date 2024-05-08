from typing import Any, Optional
from collections.abc import Sequence, Callable
from functools import partial
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

from .imagetools import (
    imread,
    ensure_channel_last,
    to_black_and_white,
    normalize_to_range,
    get_interpolated_value,
    get_interpolated_value_torch,
    Compose,
)

def _auto_compose(x: Optional[Callable | Sequence[Callable]]):
    if isinstance(x, Sequence): return Compose(*x)
    return x

class ImageDescent(torch.nn.Module):
    def __init__(
        self,
        image: str | np.ndarray | torch.Tensor,
        mode:str = 'bicubic',
        coord_scale: int | float = 100,
        dtype:torch.dtype = torch.float64,
        init:Callable | torch.Tensor | np.ndarray | Sequence = torch.rand,
        loader: Callable | Sequence[Callable] = (imread, ensure_channel_last, to_black_and_white, normalize_to_range),
        grad_tfms: Optional[Callable | Sequence[Callable]] = lambda x: gaussian_filter(x, sigma=2),
    ):
        """Image descent.

        Args:
            image (str | np.ndarray | torch.Tensor): Path to an image, or a numpy array/torch.tensor/list of an image. Your image will be turned into black and white.
            mode (str, optional): Coordinate interpolation mode. Defaults to 'bilinear'.
            coord_scale (int | float, optional): The gradient will be multiplied by this, which is equivalent to making the coordinates smaller by this factor. This basically acts as a multiplier to learning rate, to make it closer to real learning rates. Defaults to 100.
            dtype (torch.dtype, optional): Data type that all calcualtions will be performed in. You can see how precision affects the optimization path, if you want. Defaults to torch.float64.
            init (Callable | torch.Tensor | np.ndarray | Sequence, optional): Initial coordinates, either a sequence with two values or a callable. Defaults to torch.rand.
            loader (Callable | Sequence[Callable], optional): This is what `image` is passed to. Must return torch tensor or numpy array. Defaults to (imread, ensure_channel_last, to_black_and_white, normalize_to_range).
            grad_tfms (_type_, optional): transforms to the gradient, by default smoothes the gradient using gaussian filter to avoid flat areas due to imprecision. Defaults to lambdax:gaussian_filter(x, sigma=2).
        """
        super().__init__()

        self.dtype = dtype

        # load the image
        self.image:torch.Tensor = _auto_compose(loader)(image) # type:ignore
        if not isinstance(self.image, torch.Tensor): self.image = torch.as_tensor(self.image, dtype=self.dtype)
        else: self.image = self.image.to(self.dtype)


        self.mode = mode
        self.coord_scale = coord_scale

        # gradients are evaluated using np.gradient, which returns a tuple for each dimension,
        # e.g. for a 2D black and white image we get x-axis gradient and y-axis gradient, both same shape as image
        self.gradient = list(reversed([torch.as_tensor(i, dtype=self.dtype) for i in np.gradient(self.image, edge_order=1)]))
        self.ndim = len(self.gradient)

        # we smooth the gradient by default using gaussian filter, because otherwise it tends to get stuck on quantized flat areas.
        if grad_tfms is not None:
            grad_tfms = _auto_compose(grad_tfms)
            self.gradient = [grad_tfms(i) for i in self.gradient] # type:ignore
            self.gradient = [torch.as_tensor(i, dtype=self.dtype) if not isinstance(i, torch.Tensor) else i.to(self.dtype) for i in self.gradient]

        # set initial coordinates, init can be callable or a predefined coord
        if callable(init): self.coords = torch.nn.Parameter(init(self.image.ndim))
        else:
            if isinstance(init[0], int): init = [i/s for i, s in zip(init, self.image.shape)]
            if isinstance(init, torch.Tensor): self.coords = torch.nn.Parameter(init.to(self.dtype))
            else: self.coords = torch.nn.Parameter(torch.from_numpy(np.asanyarray(init)).to(self.dtype))

        # history of coords
        self.coord_history = []
        self.loss_history = []

    @torch.no_grad()
    def step(self):
        """Calculates gradient at current coordinates and puts it into `grad` attribute so that you can call optimizer.step()."""
        detached_coords = self.coords.detach().clone()
        # save current coords to the history
        self.coord_history.append(detached_coords)

        # we just get the gradient for each axis at current coordinates, and since coords will be floats, the values will be interpolated
        grad = torch.zeros(self.ndim, dtype=self.dtype)
        for i, grad_dim in enumerate(self.gradient): grad[i] = get_interpolated_value_torch(img = grad_dim, coord=detached_coords, mode=self.mode)

        # then we set that gradient into the grad attribute that all optimizers use
        # gradients are accumulated as usual
        if self.coords.grad is None: self.coords.grad = grad * self.coord_scale # type:ignore
        else: self.coords.grad += self.coord_scale

        # return loss, which is value of the image at current coords
        loss = get_interpolated_value_torch(img = self.image, coord = detached_coords, mode=self.mode) # pylint:disable=E1102
        self.loss_history.append(loss)
        return loss

    # so that its a proper nn.Module
    def forward(self): return self.step()

    def plot_image(self, figsize=None):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax.set_title("Image")
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.imshow(self.image, cmap='gray')
        return fig, ax

    def plot_gradients(self, figsize=None):
        fig, ax = plt.subplots(1, self.ndim, figsize=figsize)
        for i, grad in enumerate(self.gradient): 
            ax[i].set_title(f"gradient {i}")
            ax[i].set_axis_off()
            ax[i].set_frame_on(False)
            ax[i].imshow(grad, cmap='gray')
        return fig, ax

    def plot_losses(self, figsize=None, show=False):
        from .plot import qlinechart
        return qlinechart(self.loss_history, title='Losses', xlabel='step', ylabel='loss', figsize=figsize, show=show)

    def plot_path(self, cmap='Blues',linecolor='blue',figsize=None, show=False):
        from image_descent.plot import Figure
        fig = Figure()
        coords = [[((x+1)/2)*self.image.shape[1], ((y+1)/2)*self.image.shape[0]] for x, y in self.coord_history]
        fig.add().imshow(self.image).path2d(coords, c=self.loss_history, cmap=cmap,s=12,marker_alpha=0.3,linewidth=0.5,line_alpha=1,linecolor=linecolor).style_img().style_chart("Optimization path",legend=False)
        if show: fig.show(figsize)
        else: fig.create(figsize)
        return fig