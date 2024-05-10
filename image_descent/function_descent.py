from typing import Optional,Any
from collections.abc import Sequence, Callable
import matplotlib.pyplot as plt
import numpy as np
import torch
from .python_tools import Compose
from .plotting import ax_plot

class FunctionDescent2D(torch.nn.Module):
    def __init__(
        self,
        fn: Callable | Sequence[Callable],
        coords: Optional[Sequence[int | float] | Callable] = None,
        dtype: torch.dtype = torch.float32,
        before_step: Optional[Callable | Sequence[Callable]] = None,
        after_step: Optional[Callable | Sequence[Callable]] = None,
        normalize: Optional[int] = 100,
        mode: str = 'unpacked',
        xlim: Optional[Sequence[int|float]] = None,
        ylim: Optional[Sequence[int|float]] = None,
        lims_from_surface: bool = True,
        minimum: Optional[Sequence[int|float]]  = None
    ):
        """Perform gradient descent on a 2D function

        Args:
            fn (Callable | Sequence[Callable]):
            The function to optimize, must accept two pytorch scalars and return a pytorch tensor, e.g;
            ```py
            def rosenbrock(x:torch.Tensor, y:torch.Tensor):
                return (1-x)**2 + 100*(y-x**2)**2
            ```

            coords (Optional[Sequence[int  |  float]  |  Callable], optional):
            Initial coordinates, e.g. `(x, y)`. If `fn` has `start` method and this is None, it will be used to get the coordinates.
            Otherwise this must be specified. Defaults to None.

            dtype (torch.dtype, optional):
            Data type in which calculations will be performed. Defaults to torch.float32.

            before_step (Optional[Callable  |  Sequence[Callable]], optional):
            Optional function or sequence of functions that gets applied to coordinates before each step, and is part of the backpropagation.
            Can be used to add noise, etc. Defaults to None. Example:
            ```py
            def add_noise(coords:torch.Tensor):
                return coords + torch.randn_like(coords) * 0.01
            ```

            after_step (Optional[Callable  |  Sequence[Callable]], optional):
            Optional function or sequence of functions that gets applied to loss after each step, and is part of the backpropagation.
            Defaults to None. Example:
            ```py
            def pow_loss(loss:torch.Tensor):
                return loss**2
            ```

            normalize (int, optional):
            If not None, adds normalization to 0-1 range to `fn` by calculating `normalize`*`normalize` grid of values and using minimum and maximum. Defaults to 100.

            mode (str, optional):
            `unpacked` means `fn` gets passed `coords[0], coords[1]`, `packed` means `fn` gets passed `coords` and currently doesn't work.
            Defaults to 'unpacked'.

            xlim (tuple[float, float], optional):
            Optionally specify x-axis limits for plotting as `(left, right)` tuple.
            Does not prevent optimizer going outside of the limit.
             If `fn` has `domain` method and this is None, it will be used to get the x and y limis.
            Defaults to None.

            ylim (tuple[float, float], optional):
            Optionally specify y-axis limits for plotting as `(top, bottom)` tuple.
            Does not prevent optimizer going outside of the limit.
            If `fn` has `domain` method and this is None, it will be used to get the x and y limis.
            Defaults to None.

            lims_from_surface (bool):
            Whether to get `xlim` and `ylim` from `fn` if it has `domain` method and `xlim` and `ylim` are `None`. Defaults to True.

            minimum (tuple[float, float], optional):
            Optional `(x,y)` coordinates of the global minimum.
            If specified, distance to minimum will be logged each step into `distance_to_minimum_history`.
            If `fn` has `minimum` method and this is None, it will be used to get the minimum.
            Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        self.fn = Compose(fn)
        self.dtype = dtype
        self.mode = mode
        self.xlim = xlim
        self.ylim = ylim

        self.before_step = Compose(before_step)
        self.after_step = Compose(after_step)

        # get coords from surface if it has them
        if coords is None:
            if hasattr(fn, "start"): coords = fn.start() # type:ignore
            else: raise ValueError("No coords provided and surface has no coords method")

        # get coords
        if callable(coords): coords = coords()
        if isinstance(coords, np.ndarray): self.coords = torch.nn.Parameter(torch.from_numpy(coords).to(self.dtype))
        elif isinstance(coords, torch.Tensor): self.coords = torch.nn.Parameter(coords.to(self.dtype))
        else: self.coords = torch.nn.Parameter(torch.from_numpy(np.asanyarray(coords)).to(self.dtype))

        # get limits from surface if it has them
        if xlim is None and ylim is None and lims_from_surface:
            if hasattr(fn, "domain"): self.xlim, self.ylim = fn.domain() # type:ignore

        # get minimum from surface if it has it
        self.minimum = minimum
        if self.minimum is None:
            if hasattr(fn, "minimum"): self.minimum = fn.minimum() # type:ignore

        # history
        self.coords_history = []
        self.loss_history = []
        self.distance_to_minimum_history = []

        if normalize and (self.xlim is not None) and (self.ylim is not None):
            _, _, z = self.compute_image(steps=normalize)
            vmin, vmax = np.min(z), np.max(z)-np.min(z)
            self.fn = Compose(fn, lambda x, y: (x-vmin) / vmax)

    def forward(self):
        # have coords to history
        self.coords_history.append(self.coords.detach().cpu().clone()) # pylint:disable=E1102

        # save distance to minimum
        if self.minimum is not None: self.distance_to_minimum_history.append(torch.norm(self.coords - torch.tensor(self.minimum, dtype=self.dtype)).detach().cpu().clone())

        # get loss
        coords = self.before_step(self.coords)
        if self.mode == 'unpacked': loss:torch.Tensor = self.fn(coords[0], coords[1])
        elif self.mode == 'packed': loss:torch.Tensor = self.fn(coords)
        else: raise ValueError(f"Unknown mode {self.mode}")
        loss = self.after_step(loss)

        # save loss to history
        self.loss_history.append(loss.detach().cpu().clone())

        return loss

    def step(self): return self.forward()

    def compute_image(self, xlim=None, ylim=None, steps=1000, auto_expand = True):
        if xlim is None: xlim = self.xlim
        if ylim is None: ylim = self.ylim

        if (xlim is None or ylim is None) or auto_expand:
            if len(self.coords_history) == 0: xvals, yvals = [-1,1],[-1,1]
            else: xvals, yvals = list(zip(*self.coords_history))
            if xlim is None: xlim = min(xvals), max(xvals)
            if ylim is None: ylim = min(yvals), max(yvals)

        if auto_expand and len(self.coords_history) != 0:
            xlim = min(*xvals, xlim[0]), max(*xvals, xlim[1]) # type:ignore
            ylim = min(*yvals, ylim[0]), max(*yvals, ylim[1]) # type:ignore

        xstep = (xlim[1] - xlim[0]) / steps
        ystep = (ylim[1] - ylim[0]) / steps

        y, x = torch.meshgrid(torch.arange(ylim[0], ylim[1], xstep), torch.arange(xlim[0], xlim[1], ystep), indexing='xy')
        z = [self.fn(xv, yv).numpy() for xv, yv in zip(x, y)]
        self.computed_image = (x.numpy(), y.numpy(), z)
        return self.computed_image

    def plot_image(self, xlim=None, ylim=None, cmap='gray', levels=20, figsize=None, show=False,return_fig=False):
        image = self.compute_image(xlim, ylim)
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax.set_title("Loss landscape")
        ax.set_frame_on(False)
        cmesh = ax.pcolormesh(*image, cmap=cmap, zorder=0)
        if levels: ax.contour(*image, linewidths=0.5, alpha=0.5, cmap='Spectral', levels=levels)
        current_coord = self.coords.detach().cpu() # pylint:disable=E1102
        minimum = self.minimum
        ax.scatter([current_coord[0]], [current_coord[1]], s=4)
        if minimum is not None: ax.scatter([minimum[0]], [minimum[1]], s=64, c='lime', marker='+', zorder=4, alpha=0.5)
        fig.colorbar(cmesh, ax=ax)
        if show: plt.show()
        if return_fig: return fig, ax

    def plot_losses(self, figsize=None, show=False,return_fig=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax_plot(ax, self.loss_history)
        if show: plt.show()
        if return_fig: return fig, ax

    def plot_distance_to_minimum(self, figsize=None, show=False,return_fig=False):
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='tight')
        ax_plot(ax, self.distance_to_minimum_history)
        if show: plt.show()
        if return_fig: return fig, ax

    def plot_path(self, xlim=None, ylim=None, surface_cmap='gray', levels=20, figsize=None, show=False, return_fig=False):
        """Plots the optimization path on top of the loss landscape image. Color of the dots represents loss at that step (blue=lowest loss)"""
        fig, ax = self.plot_image(xlim=xlim, ylim=ylim, cmap=surface_cmap, levels = levels, figsize=figsize, return_fig=True)#type:ignore
        ax.set_title("Optimization path")
        ax.set_frame_on(False)

        ax.plot(*list(zip(*self.coords_history)), linewidth=0.5, color='blue', zorder=2)
        ax.scatter(*list(zip(*self.coords_history)), c=self.loss_history, s=16, cmap='Spectral', zorder=1, alpha=0.75, marker='x')
        if show: plt.show()
        if return_fig: return fig, ax