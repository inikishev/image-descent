from typing import Optional,Any
from collections.abc import Sequence, Callable
import matplotlib.pyplot as plt
import numpy as np
import torch
from .image_tools import imread, prepare_image
from .interpolation import get_interpolated_value_torch
from .gradients import get_gradients_by_shifting
from .smoothing import smooth_gaussian
from .python_tools import Compose
from .plotting import ax_plot

def load_image(__path_or_array) -> torch.Tensor:
    image = imread(__path_or_array)
    return prepare_image(image) # type:ignore

class ImageDescent(torch.nn.Module):
    def __init__(
        self,
        __path_or_array: str | np.ndarray | torch.Tensor | Sequence,
        coords: Sequence[int | float] | Callable,
        scale = 100,
        dtype: torch.dtype = torch.float32,
        smooth: int | None = 2,
        loader: Callable | Sequence[Callable] = load_image,
        grad_fn: Callable | Sequence[Callable] = get_gradients_by_shifting,
        smooth_fn: Optional[Callable | Sequence[Callable]] = smooth_gaussian,
        interp_fn = get_interpolated_value_torch,

        img_init: Optional[Callable | Sequence[Callable]] = None,
        img_step: Optional[Callable | Sequence[Callable]] = None,
        grad_init: Optional[Callable | Sequence[Callable]] = None,
        grad_step: Optional[Callable | Sequence[Callable]] = None,
    ):
        """Perform descent on an image as if it was loss landscape.

        Args:
            __path_or_array (str | np.ndarray | torch.Tensor | Sequence): Path to your image or array of already loaded image. It will be converted into black and white and normalized to 0-1 range.
            coords (Sequence[int  |  float] | Callable): Initial coordinates, either integer pixel coordinates, or floating point coords in (-1,-1) to (1,1) range.
            scale (int, optional): Scales the coordinates, equivalent to multiplying learning rate by this value. This is just to make learning rates smaller and closer to real ones. Defaults to 100.
            dtype (torch.dtype, optional): Data type in which calculations will be performed. Defaults to torch.float32.
            smooth (int | None, optional): Image will be smoothed by this amount of reduce flat areas due to compression. By default uses gaussian blur with sigma=smooth. Defaults to 2.
            loader (Callable | Sequence[Callable], optional): Function to load the image and make it black and white. Defaults to load_image.
            grad_fn (Callable | Sequence[Callable], optional): Function to calculate the gradients. Defaults to get_gradients_by_shifting.
            smooth_fn (Optional[Callable  |  Sequence[Callable]], optional): Function to smooth the image with `smooth`. Defaults to smooth_gaussian.
            interp_fn (_type_, optional): Function to interpolate float coordinates. Defaults to get_interpolated_value_torch.
            img_init (Optional[Callable  |  Sequence[Callable]], optional): Optional function or sequence of functions that will be applied to the image after loading it and before calculating gradients, e.g. any additional transforms you need like resizing or whatever. Defaults to None.
            img_step (Optional[Callable  |  Sequence[Callable]], optional): Optional function or sequence of functions that will be applied to the image before each step, for example random transformations like randomly adding noise, etc. If this is specified, gradient will be recalculated each step from the transformed image. Defaults to None.
            grad_init (Optional[Callable  |  Sequence[Callable]], optional): Optional function or sequence of functions that will be applied to the gradients after calculating them. Defaults to None.
            grad_step (Optional[Callable  |  Sequence[Callable]], optional): Optional function or sequence of functions that will be applied to the gradients before each step, for example random transformations like randomly adding noise, etc. Defaults to None.
        """
        super().__init__()
        self.scale = scale
        self.dtype = dtype
        self.smooth = smooth
        self.grad_fn = grad_fn
        self.interp_fn = interp_fn

        # load the image
        self.image: torch.Tensor = Compose(loader)(__path_or_array).to(self.dtype)
        self.ndim = self.image.ndim
        self.shape = self.image.shape

        # smooth the image to avoid flat areas due to imprecision
        if (smooth_fn is not None) and (smooth is not None): self.image = Compose(smooth_fn)(self.image, smooth)

        # get coords
        if callable(coords): coords = coords()
        # integer coordinates are considered to be pixel coordinates, we change them into (-1, 1) range.
        if isinstance(coords[0], int): coords = [((i / s) * 2) - 1 for i, s in zip(coords, self.image.shape)] # type:ignore
        if isinstance(coords, np.ndarray): self.coords = torch.nn.Parameter(torch.from_numpy(coords).to(self.dtype))
        elif isinstance(coords, torch.Tensor): self.coords = torch.nn.Parameter(coords.to(self.dtype))
        else: self.coords = torch.nn.Parameter(torch.from_numpy(np.asanyarray(coords)).to(self.dtype))

        # initial transforms
        self.image = Compose(img_init)(self.image)

        # get gradients
        self.gradients: tuple[torch.Tensor, torch.Tensor] = tuple(reversed([i.to(self.dtype) for i in Compose(self.grad_fn)(self.image)]))

        self.gradients = Compose(grad_init)(self.gradients)

        # those will run each step
        if img_step is not None: self.img_step = Compose(img_step)
        else: self.img_step = None
        self.grad_step = Compose(grad_step)

        # history
        self.coords_history = []
        self.loss_history = []

    def _image_gradient_fn_step(self):
        # if there are random transforms, we apply them
        # if image transforms are applied, gradient needs to be recalculated
        if self.img_step is not None:
            image = self.img_step(self.image)
            gradients: tuple[torch.Tensor, torch.Tensor] = tuple(reversed([i.to(self.dtype) for i in Compose(self.grad_fn)(image)]))
        else: image, gradients = self.image, self.gradients
        # random gradient transforms
        gradients = self.grad_step(gradients)

        return image, gradients

    @torch.no_grad
    def forward(self):
        image, gradients = self._image_gradient_fn_step()

        # save coords to history
        coords_detached = self.coords.detach().cpu().clone() # pylint:disable=E1102
        self.coords_history.append(coords_detached)

        # we get the gradient for each axis at current coordinates, and since coords will be floats, the values will be interpolated
        grad = torch.zeros(self.ndim, dtype=self.dtype)
        for i, param_grad in enumerate(gradients): grad[i] = self.interp_fn(param_grad, coords_detached)

        # then we set that gradient into the grad attribute that all optimizers use
        # gradients are accumulated as usual
        if self.coords.grad is None: self.coords.grad = grad * self.scale # type:ignore
        else: self.coords.grad += grad * self.scale

        # return loss, which is value of the image at current coords
        loss = self.interp_fn(image, coords_detached)
        self.loss_history.append(loss)
        return loss

    def step(self): return self.forward()

    # Plotting
    def rel2abs(self, coord):
        return [((c + 1) / 2) * s for c,s in zip(coord, self.shape)]

    def get_coord_history_pixels(self):
        return [self.rel2abs(i) for i in self.coords_history]

    def plot_image(self, figsize=None, show=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax.set_title("Loss landscape")
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.imshow(self.image, cmap='gray')
        if show: plt.show()
        return fig, ax

    def plot_gradients(self, figsize=None, show=False):
        fig, ax = plt.subplots(1, self.ndim, figsize=figsize, layout='tight')
        for i, grad in enumerate(self.gradients):
            ax[i].set_title(f"Gradient for {i+1} coordinate")
            ax[i].set_axis_off()
            ax[i].set_frame_on(False)
            ax[i].imshow(grad, cmap='gray')
        if show: plt.show()
        return fig, ax

    def plot_image_and_grad(self, figsize=None, show=False):
        fig, ax = plt.subplots(1, self.ndim+1, figsize=figsize, layout='tight')
        for i in range(len(ax)):
            ax[i].set_frame_on(False)
            ax[i].set_axis_off()
            if i == 0:
                ax[i].set_title("Loss landscape")
                ax[i].imshow(self.image, cmap='gray')
                current_coord = self.rel2abs(self.coords.detach().cpu()) # pylint:disable=E1102
                ax[i].scatter([current_coord[0]], [current_coord[1]], s=4)
            else:
                ax[i].set_title(f"Gradient for {i} coordinate")
                ax[i].imshow(self.gradients[i-1], cmap='gray')
        if show: plt.show()
        return fig, ax

    def plot_transforms(self, n=3, figsize=None, show=False):
        fig, ax = plt.subplots(n, self.ndim+1, figsize=figsize, layout='tight')
        for i in range(n):
            image, gradients = self._image_gradient_fn_step()
            for j in range(len(ax[i])):
                ax[i][j].set_frame_on(False)
                ax[i][j].set_axis_off()
                if j == 0:
                    ax[i][j].set_title(f"Loss landscape {j}")
                    ax[i][j].imshow(image, cmap='gray')
                    current_coord = self.rel2abs(self.coords.detach().cpu()) # pylint:disable=E1102
                    ax[i][j].scatter([current_coord[0]], [current_coord[1]], s=4)
                else:
                    ax[i][j].set_title(f"Gradient for {j} coordinate")
                    ax[i][j].imshow(gradients[j-1], cmap='gray')
        if show: plt.show()
        return fig, ax

    def plot_losses(self, figsize=None, show=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax_plot(ax, self.loss_history)
        if show: plt.show()
        return fig, ax

    def plot_path(self, figsize=None, show=False):
        """Plots the optimization path on top of the loss landscape image. Color of the dots represents loss at that step (blue=lowest loss)"""
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax.set_title("Optimization path")
        ax.set_axis_off()
        ax.set_frame_on(False)
        ax.imshow(self.image, cmap='gray')
        ax.plot(*list(zip(*self.get_coord_history_pixels())), linewidth=0.5, color='red', zorder=0)
        ax.scatter(*list(zip(*self.get_coord_history_pixels())), c=self.loss_history, s=4, cmap='turbo', zorder=1, alpha=0.75)
        if show: plt.show()
        return fig, ax